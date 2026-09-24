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
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .logging_setup import console
from .objectstore import MinioStore, ObjectStoreError, id_candidates_from_object_name
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
# A marker older than this says nothing about what another device is doing right now, so it
# is not evidence of a partition clash.
MARKER_FRESH_HOURS = 24


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

    def lines(self) -> list[str]:
        verb = "would mark" if self.dry_run else "marked"
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


def sync_from_bucket(
    manifest: Manifest,
    store: MinioStore,
    *,
    require_meta: bool = True,
    dry_run: bool = False,
    device: str = "",
    progress: Any = None,
    announce: Any = console,
) -> SyncReport:
    """Read the bucket into the manifest. Safe to run repeatedly; it only ever adds."""
    started = time.monotonic()
    report = SyncReport(device=device, dry_run=dry_run)

    every_shard = manifest.shards()
    open_shards = sorted(manifest.shards(open_only=True))
    report.shards_skipped = max(0, len(every_shard) - len(open_shards))

    for shard in open_shards:
        found: dict[str, tuple[tuple[str, ...], int, str | None]] = {}
        for name, size, modified in store.iter_objects(f"md/{shard}"):
            report.objects_seen += 1
            candidates = id_candidates_from_object_name(name, kind="md")
            if not candidates:
                report.unrecognised += 1
                continue
            stamp = modified.isoformat(timespec="seconds") if modified else None
            found[name] = (candidates, size, stamp)
        report.shards_scanned += 1
        if progress is not None:
            progress.update(1)
        if not found:
            continue

        have_meta: set[str] = set()
        if require_meta:
            for name, _size, _modified in store.iter_objects(f"meta/{shard}"):
                report.objects_seen += 1
                have_meta.update(id_candidates_from_object_name(name, kind="meta"))

        rows: list[tuple[str, int, str | None]] = []
        considered = 0
        for candidates, size, stamp in found.values():
            if require_meta and not have_meta.intersection(candidates):
                report.md_without_meta += 1
                continue
            considered += 1
            rows.extend((candidate, size, stamp) for candidate in candidates)
        if not rows:
            continue

        counts = manifest.mark_done_from_objects(rows, dry_run=dry_run)
        report.already_done += counts["already_done"]
        report.newly_marked += counts["newly_marked"]
        report.no_pdf_recovered += counts["no_pdf_recovered"]
        report.in_flight_skipped += counts["in_flight_skipped"]
        report.absent_from_manifest += max(0, considered - counts["matched"])

    report.duration_s = time.monotonic() - started
    return report


# --- the shared allocation ---------------------------------------------------------------
def plan_name(prefix: str = "") -> str:
    parts = [p for p in (prefix.strip("/"), PLAN_OBJECT) if p]
    return "/".join(parts)


def read_plan(store: MinioStore) -> dict[str, Any] | None:
    """The shared allocation, or None if there is not one. Never fatal.

    A bucket that cannot be read must not stop a crawl, so a failure here means "no plan"
    and the machine falls back to its own configuration.
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
    if not isinstance(plan, dict) or not plan.get("devices"):
        log.warning("the shared partition plan has no device count; ignoring it")
        return None
    return plan


def write_plan(
    store: MinioStore,
    devices: int,
    assignments: dict[str, int],
    *,
    by: str = "",
) -> dict[str, Any]:
    """Publish the allocation. Raises on failure -- this one is deliberate, so it must not
    fail quietly the way telemetry may."""
    if devices < 1:
        raise ValueError(f"device count must be at least 1, got {devices}")
    for name, index in sorted(assignments.items()):
        if not 0 <= index < devices:
            raise ValueError(
                f"'{name}' is assigned slice {index}, which does not exist in a "
                f"{devices}-device split (valid: 0-{devices - 1})")
    taken: dict[int, str] = {}
    for name, index in sorted(assignments.items()):
        if index in taken:
            raise ValueError(f"'{name}' and '{taken[index]}' are both assigned slice {index}")
        taken[index] = name

    plan = {
        "devices": int(devices),
        "assignments": {k: int(v) for k, v in sorted(assignments.items())},
        "updated_at": _utcnow(),
        "updated_by": by or device_name(None),
    }
    body = json.dumps(plan, indent=2, sort_keys=False).encode("utf-8") + b"\n"
    store.put_bytes(plan_name(store.settings.prefix), body,
                    content_type="application/json")
    return plan


def plan_partition(plan: dict[str, Any] | None, device: str) -> tuple[int, int] | None:
    """`(devices, index)` this plan gives `device`, or None if it does not name it.

    Returning None rather than guessing is the point: an unassigned device that silently
    defaulted to slice 0 would collide with whichever machine really owns slice 0, which is
    the exact failure the plan exists to prevent.
    """
    if not plan:
        return None
    devices = int(plan.get("devices") or 1)
    assignments = plan.get("assignments") or {}
    if device not in assignments:
        return None
    return (devices, int(assignments[device]))


def describe_plan(plan: dict[str, Any] | None) -> list[str]:
    if not plan:
        return ["no shared allocation — each machine uses its own configuration"]
    lines = [f"{plan['devices']} device(s), set by "
             f"{plan.get('updated_by', '?')} at {plan.get('updated_at', '?')}"]
    assignments = plan.get("assignments") or {}
    for name, index in sorted(assignments.items(), key=lambda kv: (kv[1], kv[0])):
        lines.append(f"  slice {index + 1} of {plan['devices']}  {name}")
    unassigned = plan["devices"] - len(assignments)
    if unassigned > 0:
        lines.append(f"  {unassigned} slice(s) not assigned to any device yet")
    return lines


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
    partition: tuple[int, int] | None = None,
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
        "devices": partition[0] if partition else 1,
        "device_index": partition[1] if partition else 0,
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
        partition: tuple[int, int] | None = None,
        local_path: Path | None = None,
    ):
        super().__init__(name="sync-heartbeat", daemon=True)
        self.store = store
        self.device = device
        # Honoured as given. The floor belongs where the config is read, not here -- a
        # class that quietly ignores its argument cannot be tested at speed.
        self.interval = max(float(interval), 0.01)
        self.partition = partition
        self.local_path = local_path
        self._snapshot = snapshot
        self._quit = threading.Event()

    def _publish(self, state: str) -> None:
        try:
            run = self._snapshot()
        except Exception:                   # noqa: BLE001 - a snapshot must never matter
            log.exception("could not read the run snapshot")
            return
        publish_marker(self.store, self.device, partition=self.partition, run=run,
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
    partition: tuple[int, int] | None,
    *,
    plan: dict[str, Any] | None = None,
    announce: Any = console,
) -> list[str]:
    """Warn when the fleet does not actually cover the corpus.

    Two devices on the same slice crawl the same papers and nothing crawls the rest --
    silently, and for as long as nobody checks. It is the one misconfiguration here that
    costs real work, and a handful of tiny objects is a cheap way to catch it.

    With a shared allocation the comparison is against the plan, which lets a rollout be
    told apart from a fault. A marker records the device count of that machine's *last run*,
    so immediately after the count changes every other machine's marker disagrees -- that is
    a machine yet to restart, not a machine doing the wrong thing, and it is reported as a
    notice. A machine whose marker says it is **running right now** on the wrong slice is
    the real problem, and only that is returned as one.

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
        devices = int(plan.get("devices") or 1)
        assignments = plan.get("assignments") or {}
        for marker in markers:
            other = marker.get("device")
            if not other:
                continue
            ran = (int(marker.get("devices", 1) or 1), int(marker.get("device_index", 0) or 0))
            expected = assignments.get(other)
            if expected is None:
                if _fresh(marker):
                    notices.append(
                        f"'{other}' is not in the allocation, so it follows its own config "
                        f"— `devices --assign {other}=<n>` to bring it in, or "
                        f"`devices --forget {other}` if it is gone for good")
                continue
            if ran != (devices, int(expected)):
                where = (f"slice {ran[1] + 1} of {ran[0]}",
                         f"slice {int(expected) + 1} of {devices}")
                if _running(marker):
                    problems.append(
                        f"'{other}' is crawling {where[0]} right now, but the allocation "
                        f"puts it on {where[1]} — it is on the wrong slice until it restarts")
                else:
                    notices.append(
                        f"'{other}' last ran on {where[0]}; it will pick up {where[1]} "
                        f"when it next starts")
        unassigned = devices - len(assignments)
        if unassigned > 0:
            problems.append(
                f"{unassigned} of {devices} slice(s) are assigned to no device, so that "
                f"share of the corpus will not be crawled by anyone")
    else:
        devices = partition[0] if partition else 1
        index = partition[1] if partition else 0
        for marker in markers:
            other = marker.get("device")
            if not other or other == device or not _fresh(marker):
                continue
            if int(marker.get("devices", 1) or 1) != devices:
                problems.append(
                    f"'{other}' last ran with --devices {marker.get('devices')}, this one "
                    f"has {devices}: the slices do not cover the corpus. Set one shared "
                    f"allocation instead — `devices --set-devices N --auto`")
            elif int(marker.get("device_index", 0) or 0) == index:
                problems.append(
                    f"'{other}' is also using --device-index {index}: both devices will "
                    f"crawl the same slice and nothing will crawl the others")

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
    partition: tuple[int, int] | None = None,
    dry_run: bool = False,
    progress: Any = None,
    announce: Any = console,
) -> SyncReport:
    """The whole start-of-run reconciliation: sync, publish, cross-check."""
    settings = SyncSettings.from_config(getattr(cfg, "sync", None))
    report = sync_from_bucket(
        manifest, store,
        require_meta=settings.require_meta, dry_run=dry_run,
        device=settings.device, progress=progress, announce=announce,
    )
    if settings.marker:
        check_partition_agreement(read_markers(store), settings.device, partition,
                                  plan=read_plan(store), announce=announce)
        if not dry_run:
            publish_marker(store, settings.device, report, partition=partition,
                           local_path=getattr(cfg.paths, "sync_state", None))
    return report


__all__ = [
    "Heartbeat", "ObjectStoreError", "SyncReport", "SyncSettings",
    "check_partition_agreement", "describe_plan", "device_name", "forget_device",
    "marker_name",
    "plan_name", "plan_partition", "publish_marker", "read_markers", "read_plan",
    "run_sync", "sync_from_bucket", "write_plan",
]
