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
    report: SyncReport,
    *,
    partition: tuple[int, int] | None = None,
    local_path: Path | None = None,
) -> None:
    """Record this device's sync in the bucket and on disk. Never fatal.

    The bucket copy is what lets each device see when the others last synced, and what the
    partition cross-check reads. A read-only credential, or a policy that forbids writing
    under `_state/`, must not abort a twelve-hour crawl over telemetry nothing depends on.
    """
    payload = {
        "device": device,
        "synced_at": _utcnow(),
        "devices": partition[0] if partition else 1,
        "device_index": partition[1] if partition else 0,
        "report": asdict(report),
    }
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
    announce: Any = console,
) -> list[str]:
    """Warn when another device's recent marker disagrees about the split.

    Two devices that both take index 0 crawl the same half of the corpus and neither ever
    touches the other half -- silently, and for as long as nobody checks. It is the one
    misconfiguration of this feature that costs real work, and a handful of tiny objects is
    a cheap way to catch it. Warn only: a stale marker must never stop a crawl.
    """
    devices = partition[0] if partition else 1
    index = partition[1] if partition else 0
    problems: list[str] = []
    for marker in markers:
        other = marker.get("device")
        if not other or other == device:
            continue
        if _hours_since(marker.get("synced_at")) > MARKER_FRESH_HOURS:
            continue
        if int(marker.get("devices", 1)) != devices:
            problems.append(
                f"device '{other}' last ran with --devices {marker.get('devices')}, "
                f"this one has {devices}: the slices do not cover the corpus")
        elif int(marker.get("device_index", 0)) == index:
            problems.append(
                f"device '{other}' is also using --device-index {index}: both devices will "
                f"crawl the same slice and nothing will crawl the others")
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
                                  announce=announce)
        if not dry_run:
            publish_marker(store, settings.device, report, partition=partition,
                           local_path=getattr(cfg.paths, "sync_state", None))
    return report


__all__ = [
    "ObjectStoreError", "SyncReport", "SyncSettings", "check_partition_agreement",
    "device_name", "marker_name", "publish_marker", "read_markers", "run_sync",
    "sync_from_bucket",
]
