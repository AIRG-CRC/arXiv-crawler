"""Reconcile the local manifest against the shared bucket.

Two devices crawling the same corpus each keep their own `data/manifest.db`, so neither
knows what the other has finished. The bucket does: an object at
``<prefix>/md/<shard>/<id>.md`` is proof that a paper has been converted, whoever converted
it. Reading that back into the manifest before a run is what stops the second device
re-downloading and re-converting work that is already done.

The scan is per shard, and only over shards this manifest still has unfinished rows in. A
shard whose rows are all `done` cannot learn anything from being listed, so skipping it is
free and sound; on a fresh device every shard is open and it degrades to a full scan, which
is exactly the case that wants one. Per shard rather than per prefix also bounds memory at
one shard's worth of ids instead of the whole corpus.

A paper is marked done only when **both** its md and its meta object exist. `write_outputs`
uploads md, then tables, then meta, unlinking each as it goes, so a device killed between
the first and the last leaves an md object with no meta -- and its own row is not `done`
either, because that is written after the conversion returns. Accepting the md alone would
mean nobody ever produces that meta.

There is deliberately no resume cursor. See `MinioStore.iter_objects` for the three reasons
a high-water mark over these key names silently skips papers forever.

That full scan is slow -- hours, on a bucket of millions of objects -- so it is no longer what
a run does by default. Each device publishes a **done-journal** of the papers it finished
(`_state/done/<device>/<seq>.txt.gz`), and the start-of-run sync reads only the segments
written since it last looked: work proportional to what happened in between, not to the size
of the corpus. A per-paper existence check before download (`sync.check_before_download`)
catches anything the journal has not caught up with. `sync --full` still runs the scan.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import re
import socket
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .logging_setup import console
from .objectstore import MinioStore, ObjectStoreError, id_candidates_from_object_name
from .partition import (
    BUCKETS, LiveSlice, Slice, plan_partition_of, plan_ranges, ranges_from_weights,
)
from .state import Manifest

log = logging.getLogger(__name__)

DEVICE_ENV = "ARXIV_CRAWLER_DEVICE"
MARKER_PREFIX = "_state/sync"
# The authoritative split, written once and read by every machine. Without it each device
# decides its own device count from its own config, so changing the count means editing every
# machine and any one you forget silently crawls the wrong slice -- or the same slice as
# somebody else. One object removes that whole class of mistake: change it here, and every
# run picks the new allocation up at its next start.
PLAN_OBJECT = "_state/partition.json"
PLAN_VERSION = 2
# Each device's done-journal: `<device>/<seq>.txt.gz`, one gzip'd list of finished ids per
# segment. Reading these is what replaced listing the whole bucket at the start of a run.
JOURNAL_PREFIX = "_state/done"
# Ids per segment for the bulk publishes (the first snapshot, a dump); the per-heartbeat
# segments are far smaller.
JOURNAL_SEGMENT_IDS = 200_000
# Rows handed to the manifest per transaction while reading the journal.
JOURNAL_APPLY_BATCH = 50_000
# A marker older than this says nothing about what another device is doing right now, so it
# is not evidence of a partition clash.
MARKER_FRESH_HOURS = 24
# What `None` -- "no partition" -- means, as a range.
WHOLE = Slice(0, BUCKETS)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def device_name(cfg: Any = None) -> str:
    """What this machine calls itself in the bucket.

    The environment comes first because `config.yaml` is tracked in git, so a per-device
    value there conflicts on every pull -- the same reasoning that keeps the MinIO
    credentials out of it. The hostname is only a fallback: it is unstable (DHCP names, a
    `.local` that comes and goes) and putting a machine name in a shared bucket is a small
    leak, so it is worth being able to override.
    """
    name = (os.environ.get(DEVICE_ENV)
            or getattr(cfg, "device", None)
            or socket.gethostname().split(".")[0]
            or "unknown")
    return re.sub(r"[^A-Za-z0-9._-]", "-", str(name))[:64] or "unknown"


@dataclass
class SyncReport:
    """What one sync saw and did. Every number is exact; none of them are estimates."""

    device: str = ""
    dry_run: bool = False
    shards_scanned: int = 0
    shards_skipped: int = 0
    objects_seen: int = 0
    unrecognised: int = 0            # object names that are not artefacts of ours
    absent_from_manifest: int = 0    # real papers, but not in this manifest's scope
    already_done: int = 0
    newly_marked: int = 0
    no_pdf_recovered: int = 0        # recorded 404 here, but converted elsewhere
    in_flight_skipped: int = 0       # a local run holds these; left alone on purpose
    md_without_meta: int = 0
    duration_s: float = 0.0
    mode: str = "full"               # "full" (bucket listing) or "journal"
    segments_read: int = 0           # journal segments read from other devices
    devices_read: int = 0            # devices that had new segments
    published: int = 0               # ids this device added to its own journal

    def lines(self) -> list[str]:
        verb = "would mark" if self.dry_run else "marked"
        if self.mode == "journal":
            out = [
                f"journal: {self.segments_read:,} new segment(s) from "
                f"{self.devices_read:,} device(s), {self.objects_seen:,} id(s), "
                f"in {self.duration_s:.1f}s",
                f"{verb} {self.newly_marked:,} paper(s) done from the journal",
            ]
            if self.published:
                out.append(f"published {self.published:,} id(s) to this device's journal")
        else:
            out = [
                f"scanned {self.shards_scanned:,} shard(s), skipped "
                f"{self.shards_skipped:,} already complete",
                f"{self.objects_seen:,} object(s) listed in {self.duration_s:.1f}s",
                f"{verb} {self.newly_marked:,} paper(s) done from the bucket",
            ]
        if self.already_done:
            out.append(f"{self.already_done:,} were already done here")
        if self.no_pdf_recovered:
            out.append(f"{self.no_pdf_recovered:,} had been recorded as no_pdf "
                       f"(the other device got a PDF)")
        if self.absent_from_manifest:
            out.append(f"{self.absent_from_manifest:,} object(s) are papers this manifest "
                       f"does not have — a wider scope on the other device")
        if self.md_without_meta:
            out.append(f"{self.md_without_meta:,} paper(s) have md but no meta and were "
                       f"left alone — run `dump` on the device that produced them")
        if self.unrecognised:
            out.append(f"{self.unrecognised:,} object name(s) were not recognised")
        if self.in_flight_skipped:
            out.append(f"{self.in_flight_skipped:,} are in flight in a local run, untouched")
        return out


def _list_shard(
    store: MinioStore, shard: str, require_meta: bool
) -> tuple[str, dict[str, tuple[tuple[str, ...], int, str | None]], set[str], int, int]:
    """One shard's md listing, and its meta listing when there is anything to check.

    Runs on a pool thread: it touches only the bucket, never the manifest, whose connection
    belongs to the thread that opened it.
    """
    found: dict[str, tuple[tuple[str, ...], int, str | None]] = {}
    seen = unrecognised = 0
    for name, size, modified in store.iter_objects(f"md/{shard}"):
        seen += 1
        candidates = id_candidates_from_object_name(name, kind="md")
        if not candidates:
            unrecognised += 1
            continue
        stamp = modified.isoformat(timespec="seconds") if modified else None
        found[name] = (candidates, size, stamp)
    have_meta: set[str] = set()
    if found and require_meta:
        for name, _size, _modified in store.iter_objects(f"meta/{shard}"):
            seen += 1
            have_meta.update(id_candidates_from_object_name(name, kind="meta"))
    return shard, found, have_meta, seen, unrecognised


def sync_from_bucket(
    manifest: Manifest,
    store: MinioStore,
    *,
    require_meta: bool = True,
    dry_run: bool = False,
    device: str = "",
    progress: Any = None,
    announce: Any = console,
    threads: int = 1,
    skip_shards: Iterable[str] = (),
    on_shard: Callable[[str], None] | None = None,
) -> SyncReport:
    """Read the bucket into the manifest. Safe to run repeatedly; it only ever adds.

    Shards are listed `threads` at a time -- the listing is network-bound and the shards are
    independent -- while every manifest write stays on this thread. `skip_shards` and
    `on_shard` are what let an interrupted scan resume: the caller records each shard as it
    is applied and passes the set back next time. That is a set of finished shards, not a
    key cursor, so none of `iter_objects`' objections to cursors apply.
    """
    started = time.monotonic()
    report = SyncReport(device=device, dry_run=dry_run, mode="full")

    every_shard = manifest.shards()
    skip = set(skip_shards)
    open_shards = sorted(set(manifest.shards(open_only=True)) - skip)
    report.shards_skipped = max(0, len(every_shard) - len(open_shards))

    def _apply(result: tuple) -> None:
        shard, found, have_meta, seen, unrecognised = result
        report.objects_seen += seen
        report.unrecognised += unrecognised
        report.shards_scanned += 1
        if progress is not None:
            progress.update(1)
        rows: list[tuple[str, int, str | None]] = []
        considered = 0
        for candidates, size, stamp in found.values():
            if require_meta and not have_meta.intersection(candidates):
                report.md_without_meta += 1
                continue
            considered += 1
            rows.extend((candidate, size, stamp) for candidate in candidates)
        if rows:
            counts = manifest.mark_done_from_objects(rows, dry_run=dry_run)
            report.already_done += counts["already_done"]
            report.newly_marked += counts["newly_marked"]
            report.no_pdf_recovered += counts["no_pdf_recovered"]
            report.in_flight_skipped += counts["in_flight_skipped"]
            report.absent_from_manifest += max(0, considered - counts["matched"])
        if on_shard is not None and not dry_run:
            on_shard(shard)

    threads = max(1, int(threads or 1))
    if threads == 1:
        for shard in open_shards:
            _apply(_list_shard(store, shard, require_meta))
    else:
        # A bounded window rather than one future per shard up front, so an error stops
        # the scan promptly instead of after every remaining listing has been queued.
        queue = list(reversed(open_shards))
        with ThreadPoolExecutor(threads, thread_name_prefix="sync-scan") as pool:
            running = set()
            try:
                while queue or running:
                    while queue and len(running) < threads * 2:
                        running.add(pool.submit(_list_shard, store, queue.pop(),
                                                require_meta))
                    finished, running = wait(running, return_when=FIRST_COMPLETED)
                    for fut in finished:
                        _apply(fut.result())
            finally:
                for fut in running:
                    fut.cancel()

    report.duration_s = time.monotonic() - started
    return report


# --- the done-journal --------------------------------------------------------------------
def _segment_seq(name: str) -> int | None:
    leaf = name.rsplit("/", 1)[-1]
    if not leaf.endswith(".txt.gz"):
        return None
    try:
        return int(leaf[: -len(".txt.gz")])
    except ValueError:
        return None


class DoneJournal:
    """This device's list of finished papers in the bucket, and its place in everyone else's.

    Writing: `record()` collects ids as papers finish; `flush()` publishes them as one
    segment. Segments are numbered by a counter kept in `state_path`, so within one device's
    folder key order is write order -- which is the property that makes the reader's
    `start_after` cursor safe here when it is not safe anywhere else in the bucket.

    Reading: `pull_journal` keeps a cursor per other device in the same file.

    Losing `state_path` costs nothing permanent: the counter is recovered by listing this
    device's own folder, and the cursors restart from the beginning (every id is applied
    again, which is idempotent).
    """

    def __init__(self, store: MinioStore, device: str, state_path: Path | None = None):
        self.store = store
        self.device = device
        self.state_path = state_path
        self._lock = threading.Lock()
        self._pending: list[str] = []
        self.state: dict[str, Any] = {}
        if state_path is not None and state_path.exists():
            try:
                loaded = json.loads(state_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.state = loaded
            except (OSError, ValueError) as exc:
                log.warning("could not read %s (%s); starting the journal state afresh",
                            state_path, exc)

    # -- state ----------------------------------------------------------------------
    def save(self) -> None:
        if self.state_path is None:
            return
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
            tmp.replace(self.state_path)
        except OSError as exc:
            log.warning("could not write %s: %s", self.state_path, exc)

    def folder(self, device: str | None = None) -> str:
        return f"{JOURNAL_PREFIX}/{device or self.device}"

    def _name(self, seq: int) -> str:
        parts = [p for p in (self.store.settings.prefix.strip("/"), self.folder(),
                             f"{seq:010d}.txt.gz") if p]
        return "/".join(parts)

    def _next_seq(self) -> int:
        if "next_seq" in self.state:
            return int(self.state["next_seq"])
        # No local record: carry on after whatever this device already wrote, so a lost
        # state file can never make a new segment sort before an old one.
        highest = -1
        for name, _s, _m in self.store.iter_objects(self.folder()):
            seq = _segment_seq(name)
            if seq is not None:
                highest = max(highest, seq)
        return highest + 1

    # -- writing --------------------------------------------------------------------
    def record(self, arxiv_id: str) -> None:
        with self._lock:
            self._pending.append(arxiv_id)

    def publish(self, ids: list[str]) -> int:
        """Write `ids` as one or more new segments. Returns how many ids were written."""
        written = 0
        for start in range(0, len(ids), JOURNAL_SEGMENT_IDS):
            chunk = ids[start:start + JOURNAL_SEGMENT_IDS]
            seq = self._next_seq()
            body = gzip.compress(("\n".join(chunk) + "\n").encode("utf-8"))
            self.store.put_bytes(self._name(seq), body, content_type="application/gzip")
            self.state["next_seq"] = seq + 1
            self.save()
            written += len(chunk)
        return written

    def flush(self) -> int:
        """Publish whatever has been recorded since the last flush. Never raises.

        The ids go back into the batch if the write fails, and `flushed_through` only moves
        on a success -- so a flush that fails, and a crash before one, are both picked up by
        the next run's `catch_up`.
        """
        mark = _utcnow()
        with self._lock:
            ids, self._pending = self._pending, []
        if not ids:
            if self.state.get("snapshot_done"):
                self.state["flushed_through"] = mark
            return 0
        try:
            written = self.publish(ids)
        except Exception as exc:            # noqa: BLE001 - retried on the next tick
            log.warning("could not publish %d journal id(s): %s", len(ids), exc)
            with self._lock:
                self._pending[:0] = ids
            return 0
        if self.state.get("snapshot_done"):
            self.state["flushed_through"] = mark
            self.save()
        return written

    def catch_up(self, manifest: Manifest) -> int:
        """Publish what the journal is missing. Run once at the start of a run.

        The first time on a device that is everything it knows is done -- the snapshot that
        lets every other device skip a full bucket scan. After that, only its own
        conversions newer than the last successful flush: what a crash lost.
        """
        since = self.state.get("flushed_through") if self.state.get("snapshot_done") else None
        mark = _utcnow()
        ids = manifest.journal_ids(since)
        written = self.publish(ids) if ids else 0
        self.state["snapshot_done"] = True
        self.state["flushed_through"] = mark
        self.save()
        return written


def pull_journal(
    manifest: Manifest,
    store: MinioStore,
    journal: DoneJournal,
    *,
    dry_run: bool = False,
    progress: Any = None,
) -> SyncReport:
    """Apply every journal segment the other devices wrote since this one last looked.

    Each device's folder is listed from its cursor onward, so the cost is the number of new
    segments. A cursor only moves once the ids before it are committed: a pull interrupted
    half-way re-reads from the last committed segment, which is harmless because marking a
    paper done twice is a no-op.
    """
    started = time.monotonic()
    report = SyncReport(device=journal.device, dry_run=dry_run, mode="journal")
    cursors: dict[str, str] = journal.state.setdefault("cursors", {})

    batch: list[tuple[str, int | None, str | None]] = []
    advance: dict[str, str] = {}

    def _apply() -> None:
        if batch:
            distinct = len({row[0] for row in batch})
            counts = manifest.mark_done_from_objects(batch, dry_run=dry_run)
            report.already_done += counts["already_done"]
            report.newly_marked += counts["newly_marked"]
            report.no_pdf_recovered += counts["no_pdf_recovered"]
            report.in_flight_skipped += counts["in_flight_skipped"]
            report.absent_from_manifest += max(0, distinct - counts["matched"])
        batch.clear()
        if not dry_run and advance:
            cursors.update(advance)
            journal.save()
        advance.clear()

    for other in sorted(set(store.list_prefixes(JOURNAL_PREFIX)) - {journal.device}):
        segments = [
            (name, modified)
            for name, _size, modified in store.iter_objects(
                journal.folder(other), start_after=cursors.get(other))
            if _segment_seq(name) is not None
        ]
        if segments:
            report.devices_read += 1
        for name, modified in segments:
            try:
                ids = gzip.decompress(store.get_bytes(name)).decode("utf-8").split()
            except (OSError, EOFError, UnicodeDecodeError) as exc:
                # A torn segment is skipped rather than retried forever: the existence
                # check before download covers whatever it would have said.
                log.warning("unreadable journal segment %s: %s", name, exc)
                ids = []
            stamp = modified.isoformat(timespec="seconds") if modified else None
            batch.extend((i, None, stamp) for i in ids)
            report.objects_seen += len(ids)
            report.segments_read += 1
            advance[other] = name
            if progress is not None:
                progress.update(1)
            if len(batch) >= JOURNAL_APPLY_BATCH:
                _apply()
    _apply()

    report.duration_s = time.monotonic() - started
    return report


# --- the shared allocation ---------------------------------------------------------------
def plan_name(prefix: str = "") -> str:
    parts = [p for p in (prefix.strip("/"), PLAN_OBJECT) if p]
    return "/".join(parts)


def read_plan(store: MinioStore) -> dict[str, Any] | None:
    """The shared allocation, or None if there is not one. Never fatal.

    A bucket that cannot be read must not stop a crawl, so a failure here means "no plan"
    and the machine falls back to its own configuration. Both formats are accepted: a
    version-1 plan (`devices` + `assignments`, equal slices only) is read as the equal split
    it describes -- see `partition.plan_ranges`.
    """
    try:
        raw = store.get_bytes(plan_name(store.settings.prefix))
    except Exception as exc:                # noqa: BLE001 - absent is the common case
        log.debug("no shared partition plan: %s", exc)
        return None
    try:
        plan = json.loads(raw)
    except ValueError as exc:
        log.warning("the shared partition plan is not valid JSON: %s", exc)
        return None
    if not isinstance(plan, dict) or not (plan.get("ranges") or plan.get("devices")):
        log.warning("the shared partition plan has no allocation in it; ignoring it")
        return None
    return plan


def _validate_ranges(ranges: dict[str, Slice]) -> None:
    claimed = sorted(((r, n) for n, r in ranges.items() if not r.empty),
                     key=lambda item: (item[0].lo, item[1]))
    for (a, name_a), (b, name_b) in zip(claimed, claimed[1:]):
        if a.overlaps(b):
            raise ValueError(f"'{name_a}' ({a.describe()}) and '{name_b}' ({b.describe()}) "
                             f"overlap -- they would crawl the same papers")


def write_plan(
    store: MinioStore,
    devices: int | None = None,
    assignments: dict[str, int] | None = None,
    *,
    ranges: dict[str, Slice] | None = None,
    weights: dict[str, float] | None = None,
    order: Iterable[str] = (),
    by: str = "",
) -> dict[str, Any]:
    """Publish the allocation. Raises on failure -- this one is deliberate, so it must not
    fail quietly the way telemetry may.

    Three ways to describe it, most specific first: explicit `ranges`; relative `weights`
    (laid out in `order`, see `partition.ranges_from_weights`); or the equal split
    `devices` + `assignments{name: index}` that the plan held before shares could differ.
    Whichever is given, what is stored is the ranges, so every reader sees one format.
    """
    if ranges is None and weights is not None:
        ranges = ranges_from_weights(order, weights)
    if ranges is None:
        if devices is None or devices < 1:
            raise ValueError(f"device count must be at least 1, got {devices}")
        assignments = assignments or {}
        for name, index in sorted(assignments.items()):
            if not 0 <= index < devices:
                raise ValueError(
                    f"'{name}' is assigned slice {index}, which does not exist in a "
                    f"{devices}-device split (valid: 0-{devices - 1})")
        taken: dict[int, str] = {}
        for name, index in sorted(assignments.items()):
            if index in taken:
                raise ValueError(
                    f"'{name}' and '{taken[index]}' are both assigned slice {index}")
            taken[index] = name
        ranges = {name: Slice.equal(devices, index) for name, index in assignments.items()}
    _validate_ranges(ranges)

    plan: dict[str, Any] = {
        "version": PLAN_VERSION,
        "ranges": {k: v.to_json() for k, v in sorted(ranges.items(),
                                                     key=lambda kv: (kv[1].lo, kv[0]))},
    }
    if weights is not None:
        plan["weights"] = {k: float(v) for k, v in sorted(weights.items())}
    plan["updated_at"] = _utcnow()
    plan["updated_by"] = by or device_name(None)
    body = json.dumps(plan, indent=2, sort_keys=False).encode("utf-8") + b"\n"
    store.put_bytes(plan_name(store.settings.prefix), body,
                    content_type="application/json")
    return plan


def plan_partition(plan: dict[str, Any] | None, device: str) -> Slice | None:
    """The `Slice` this plan gives `device`, or None if it does not name it.

    Returning None rather than guessing is the point: an unassigned device that silently
    defaulted to slice 0 would collide with whichever machine really owns slice 0, which is
    the exact failure the plan exists to prevent.
    """
    return plan_partition_of(plan, device)


def plan_splits(plan: dict[str, Any] | None) -> bool:
    """Does this plan divide the corpus, as opposed to giving it all to one machine?"""
    ranges = plan_ranges(plan)
    return bool(ranges) and not any((r.lo, r.hi) == (0, BUCKETS) for r in ranges.values())


def unassigned_buckets(plan: dict[str, Any] | None) -> int:
    """How many of the 256 buckets no device in the plan owns."""
    covered: set[int] = set()
    for r in plan_ranges(plan).values():
        covered.update(range(r.lo, r.hi))
    return BUCKETS - len(covered)


def describe_plan(plan: dict[str, Any] | None) -> list[str]:
    if not plan:
        return ["no shared allocation — each machine uses its own configuration"]
    ranges = plan_ranges(plan)
    weights = plan.get("weights") or {}
    lines = [f"{len(ranges)} device(s), set by "
             f"{plan.get('updated_by', '?')} at {plan.get('updated_at', '?')}"]
    for name, r in sorted(ranges.items(), key=lambda kv: (kv[1].empty, kv[1].lo, kv[0])):
        weight = f"  (weight {weights[name]:g})" if name in weights else ""
        where = "parked" if r.empty else f"buckets {r.lo}-{r.hi - 1}"
        lines.append(f"  {r.share * 100:5.1f}%  {name:<20} {where}{weight}")
    gap = unassigned_buckets(plan)
    if gap:
        lines.append(f"  {gap / BUCKETS * 100:5.1f}%  (unassigned: {gap} bucket(s) that no "
                     f"device will crawl)")
    return lines


def plan_weights(plan: dict[str, Any] | None) -> dict[str, float]:
    """The relative weights behind a plan.

    A plan written with weights keeps them; one written as ranges or as a version-1 equal
    split reads as each device's width in buckets, which reproduces the same layout.
    """
    if not plan:
        return {}
    stored = plan.get("weights")
    ranges = plan_ranges(plan)
    if isinstance(stored, dict) and set(stored) >= set(ranges):
        try:
            return {str(k): float(v) for k, v in stored.items()}
        except (TypeError, ValueError):
            pass
    return {name: float(r.hi - r.lo) for name, r in ranges.items()}


def merge_shares(
    existing: dict[str, Any] | None,
    shares: dict[str, float] | None = None,
    *,
    drop: Iterable[str] = (),
    equal: Iterable[str] | None = None,
) -> tuple[dict[str, float], list[str]]:
    """The weights and layout order for a new plan: `existing`, changed as asked.

    `equal` replaces every weight with 1 for those names (`--auto`); `shares` then sets
    individual weights (`--share NAME=W`, 0 parks a machine); `drop` removes names
    altogether. The order keeps the devices already in the plan where they were, by their
    range, and appends newcomers -- so the boundaries only slide, and the papers that change
    hands are the difference between the old shares and the new.
    """
    weights = plan_weights(existing)
    order = [name for name, _r in sorted(plan_ranges(existing).items(),
                                         key=lambda kv: (kv[1].lo, kv[1].empty, kv[0]))]
    if equal is not None:
        names = list(equal)
        weights = {name: 1.0 for name in names}
        order += sorted(n for n in names if n not in order)
    for name, weight in (shares or {}).items():
        if weight < 0:
            raise ValueError(f"'{name}': a share cannot be negative, got {weight:g}")
        weights[name] = float(weight)
        if name not in order:
            order.append(name)
    for name in drop:
        weights.pop(name, None)
    order = [n for n in order if n in weights]
    return weights, order


# --- device markers ---------------------------------------------------------------------
def marker_name(device: str, *, prefix: str = "") -> str:
    """The marker's full object name.

    `put_bytes` takes a complete key, so the configured prefix has to be applied here --
    `iter_objects` applies it on the way back, and a marker written without it is a marker
    no other device can ever find.
    """
    parts = [p for p in (prefix.strip("/"), MARKER_PREFIX, f"{device}.json") if p]
    return "/".join(parts)


def publish_marker(
    store: MinioStore,
    device: str,
    report: SyncReport | None = None,
    *,
    partition: Slice | None = None,
    run: dict[str, Any] | None = None,
    state: str = "",
    local_path: Path | None = None,
) -> None:
    """Record this device's sync and progress in the bucket and on disk. Never fatal.

    The bucket copy is what lets each device see what the others are doing -- it is read by
    the `devices` command and by the partition cross-check. A read-only credential, or a
    policy that forbids writing under `_state/`, must not abort a twelve-hour crawl over
    telemetry nothing depends on, so every failure here is a log line.
    """
    payload: dict[str, Any] = {
        "device": device,
        "updated_at": _utcnow(),
        "range": (partition or WHOLE).to_json(),
    }
    if report is not None:
        payload["synced_at"] = _utcnow()
        payload["sync"] = asdict(report)
    if run is not None:
        payload["run"] = {**run, "state": state or run.get("state", "running")}
    elif state:
        payload["state"] = state
    body = json.dumps(payload, indent=2, sort_keys=False).encode("utf-8") + b"\n"
    if local_path is not None:
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(body)
        except OSError as exc:
            log.warning("could not write %s: %s", local_path, exc)
    try:
        store.put_bytes(marker_name(device, prefix=store.settings.prefix), body,
                        content_type="application/json")
    except Exception as exc:                # noqa: BLE001 - telemetry, never fatal
        log.warning("could not publish the sync marker: %s", exc)


def forget_device(store: MinioStore, device: str) -> bool:
    """Delete a device's marker, so it stops appearing anywhere.

    `--unassign` takes a machine out of the allocation but leaves its marker, which is right
    for one that is only away for a while. A machine that is gone for good needs the marker
    gone too: it is what `devices` lists, and what `--auto` rebuilds the fleet from -- so
    while it exists, every `--auto` resurrects the machine you just removed.
    """
    return store.remove_object(marker_name(device, prefix=store.settings.prefix))


def marker_range(marker: dict[str, Any]) -> Slice:
    """The slice a marker says its device was on.

    Markers from before uneven shares carry `devices` + `device_index` instead; those are
    read as the equal slice they describe, so a mixed fleet mid-upgrade still compares.
    """
    try:
        if "range" in marker:
            return Slice.from_json(marker["range"])
        devices = int(marker.get("devices", 1) or 1)
        return Slice.equal(devices, int(marker.get("device_index", 0) or 0))
    except (TypeError, ValueError):
        return WHOLE


def read_markers(store: MinioStore) -> list[dict[str, Any]]:
    """Every device's marker. Returns what it can; a failure here is never fatal."""
    markers: list[dict[str, Any]] = []
    try:
        names = [n for n, _s, _m in store.iter_objects(MARKER_PREFIX) if n.endswith(".json")]
    except Exception as exc:                # noqa: BLE001
        log.warning("could not list the sync markers: %s", exc)
        return markers
    for name in names:
        try:
            markers.append(json.loads(store.get_bytes(name)))
        except Exception as exc:            # noqa: BLE001 - a torn marker costs nothing
            log.warning("could not read %s: %s", name, exc)
    return markers


class Heartbeat(threading.Thread):
    """Republishes this device's marker while a run is in progress.

    Without it a marker only says what a device was doing at the moment it last synced,
    which on a twelve-hour crawl is not much. With it, `devices` on any machine shows a view
    that is at most `interval` seconds stale -- enough to answer "is the other one still
    going, and how fast".

    Each beat also does the two other things a running crawl owes the fleet: it publishes
    this device's newly finished papers to its done-journal, and -- given `live` -- it
    re-reads the shared allocation, so a share changed with `devices --share` reaches a
    running crawl within one interval instead of at its next restart.

    A thread rather than a call in the dispatch loop: the loop turns over at 1 Hz and must
    not be held up by a network write whose timeouts are measured in seconds. Daemon, so it
    can never keep the process alive, and every publish is already non-fatal.
    """

    def __init__(
        self,
        store: MinioStore,
        device: str,
        *,
        snapshot: Any,
        interval: float = 60.0,
        partition: Slice | None = None,
        local_path: Path | None = None,
        journal: "DoneJournal | None" = None,
        live: LiveSlice | None = None,
    ):
        super().__init__(name="sync-heartbeat", daemon=True)
        self.store = store
        self.device = device
        # Honoured as given. The floor belongs where the config is read, not here -- a
        # class that quietly ignores its argument cannot be tested at speed.
        self.interval = max(float(interval), 0.01)
        self.partition = partition
        self.local_path = local_path
        self.journal = journal
        self.live = live
        self._snapshot = snapshot
        self._quit = threading.Event()

    def _follow_plan(self) -> None:
        """Adopt the allocation's current slice for this device, if it names one.

        A plan that has gone missing, or that no longer names this device, changes nothing:
        carrying on with the slice it had is safer than guessing a new one mid-run.
        """
        if self.live is None:
            return
        plan = read_plan(self.store)
        if not plan:
            return
        assigned = plan_partition_of(plan, self.device)
        if assigned is None:
            return
        new = None if (assigned.lo, assigned.hi) == (0, BUCKETS) else assigned
        if self.live.update(new):
            log.info("allocation changed under the running crawl: now %s",
                     new.describe() if new else "the whole corpus")

    def _publish(self, state: str) -> None:
        if self.journal is not None:
            self.journal.flush()
        try:
            self._follow_plan()
        except Exception:                   # noqa: BLE001 - keep the last slice
            log.exception("could not re-read the shared allocation")
        try:
            run = self._snapshot()
        except Exception:                   # noqa: BLE001 - a snapshot must never matter
            log.exception("could not read the run snapshot")
            return
        partition = self.live.current if self.live is not None else self.partition
        publish_marker(self.store, self.device, partition=partition, run=run,
                       state=state, local_path=self.local_path)

    def run(self) -> None:
        while not self._quit.wait(self.interval):
            self._publish("running")

    def finish(self, state: str = "finished") -> None:
        """Stop, and publish one last marker saying how the run ended."""
        self._quit.set()
        self._publish(state)
        self.join(timeout=5.0)


def _hours_since(stamp: str | None) -> float:
    if not stamp:
        return float("inf")
    try:
        when = datetime.fromisoformat(stamp)
    except ValueError:
        return float("inf")
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - when).total_seconds() / 3600.0


def check_partition_agreement(
    markers: Iterable[dict[str, Any]],
    device: str,
    partition: Slice | None,
    *,
    plan: dict[str, Any] | None = None,
    announce: Any = console,
) -> list[str]:
    """Warn when the fleet does not actually cover the corpus.

    Two devices on overlapping slices crawl the same papers, and a gap is crawled by
    nobody -- silently, and for as long as nobody checks. It is the one misconfiguration
    here that costs real work, and a handful of tiny objects is a cheap way to catch it.

    With a shared allocation the comparison is against the plan, which lets a rollout be
    told apart from a fault. A marker records the slice of that machine's *last* beat, so
    right after a share changes the others disagree until their next heartbeat (or, on a
    stopped machine, its next start) -- that is reported as a notice. Only a machine whose
    marker says it is **running right now** on another slice is returned as a problem.

    Warns; never blocks. A stale marker must not stop a crawl.
    """
    problems: list[str] = []
    notices: list[str] = []

    def _fresh(marker: dict[str, Any]) -> bool:
        seen = marker.get("updated_at") or marker.get("synced_at")
        return _hours_since(seen) <= MARKER_FRESH_HOURS

    def _running(marker: dict[str, Any]) -> bool:
        return (marker.get("run") or {}).get("state") == "running" and _fresh(marker)

    if plan:
        ranges = plan_ranges(plan)
        for marker in markers:
            other = marker.get("device")
            if not other:
                continue
            expected = ranges.get(other)
            if expected is None:
                if _fresh(marker):
                    notices.append(
                        f"'{other}' is not in the allocation, so it follows its own config "
                        f"— `devices --share {other}=<weight>` to bring it in, or "
                        f"`devices --forget {other}` if it is gone for good")
                continue
            ran = marker_range(marker)
            if ran != expected:
                if _running(marker):
                    problems.append(
                        f"'{other}' is crawling {ran.describe()} right now, but the "
                        f"allocation gives it {expected.describe()} — it switches at its "
                        f"next heartbeat; if it does not, it is an older version: restart it")
                else:
                    notices.append(
                        f"'{other}' last ran on {ran.describe()}; it will pick up "
                        f"{expected.describe()} when it next starts")
        gap = unassigned_buckets(plan)
        if gap:
            problems.append(
                f"{gap} of {BUCKETS} bucket(s) ({gap / BUCKETS * 100:.1f}%) are assigned to "
                f"no device, so that share of the corpus will not be crawled by anyone")
    else:
        mine = partition or WHOLE
        for marker in markers:
            other = marker.get("device")
            if not other or other == device or not _fresh(marker):
                continue
            theirs = marker_range(marker)
            if theirs.overlaps(mine):
                problems.append(
                    f"'{other}' last ran on {theirs.describe()}, which overlaps this "
                    f"device's {mine.describe()}: both will crawl the same papers. Set one "
                    f"shared allocation instead — `devices --auto` or `devices --share`")

    for notice in notices:
        announce("  · %s", notice)
        log.info("%s", notice)
    for problem in problems:
        announce("  ⚠ %s", problem)
        log.warning("%s", problem)
    return problems


@dataclass
class SyncSettings:
    """Resolved once in the parent, then used by the sync and the marker."""

    device: str = ""
    require_meta: bool = True
    on_start: bool = True
    marker: bool = True

    @classmethod
    def from_config(cls, cfg: Any) -> "SyncSettings":
        return cls(
            device=device_name(cfg),
            require_meta=bool(getattr(cfg, "require_meta", True)),
            on_start=bool(getattr(cfg, "on_start", True)),
            marker=bool(getattr(cfg, "marker", True)),
        )


def run_sync(
    cfg: Any,
    manifest: Manifest,
    store: MinioStore,
    *,
    partition: Slice | None = None,
    dry_run: bool = False,
    progress: Any = None,
    announce: Any = console,
    full: bool = False,
) -> tuple[SyncReport, DoneJournal]:
    """The whole start-of-run reconciliation: sync, publish, cross-check.

    By default this reads the other devices' done-journals -- seconds, whatever the size of
    the bucket -- after publishing whatever this device's own journal is missing (its whole
    `done` set, the first time). `full=True` is the bucket-wide listing that used to run on
    every start; it now runs only when asked for, resumes where an interrupted one stopped,
    and lists `sync.scan_threads` shards at a time.

    Returns the journal too, so a run can keep appending to it as papers finish.
    """
    sync_cfg = getattr(cfg, "sync", None)
    settings = SyncSettings.from_config(sync_cfg)
    journal = DoneJournal(store, settings.device, getattr(cfg.paths, "journal_state", None))

    if full:
        done_shards: list[str] = journal.state.setdefault("scanned_shards", [])

        def _on_shard(shard: str) -> None:
            done_shards.append(shard)
            journal.save()

        report = sync_from_bucket(
            manifest, store,
            require_meta=settings.require_meta, dry_run=dry_run,
            device=settings.device, progress=progress, announce=announce,
            threads=int(getattr(sync_cfg, "scan_threads", 8) or 1),
            skip_shards=done_shards, on_shard=_on_shard,
        )
        if not dry_run:
            # Finished, so the next `--full` is a fresh scan rather than a no-op.
            journal.state.pop("scanned_shards", None)
            journal.save()
    else:
        published = 0
        if not dry_run:
            published = journal.catch_up(manifest)
        report = pull_journal(manifest, store, journal, dry_run=dry_run, progress=progress)
        report.published = published

    if settings.marker:
        check_partition_agreement(read_markers(store), settings.device, partition,
                                  plan=read_plan(store), announce=announce)
        if not dry_run:
            publish_marker(store, settings.device, report, partition=partition,
                           local_path=getattr(cfg.paths, "sync_state", None))
    return report, journal


__all__ = [
    "DoneJournal", "Heartbeat", "ObjectStoreError", "SyncReport", "SyncSettings",
    "check_partition_agreement", "describe_plan", "device_name", "forget_device",
    "marker_name", "marker_range", "merge_shares", "plan_weights", "pull_journal",
    "plan_name", "plan_partition", "publish_marker", "read_markers", "read_plan",
    "run_sync", "sync_from_bucket", "write_plan",
]
