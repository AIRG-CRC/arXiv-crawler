"""Which papers belong to this device.

Two devices crawling the same corpus into the same bucket must not claim the same paper,
and there is no shared transaction to arrange that with: each device owns its own SQLite
manifest, and the only thing between them is the bucket, which is far too slow to consult
once per paper. So the work is partitioned by a function of the arXiv id that every device
computes identically, and each device claims only its own residue class. No coordination,
no round trips, no lease to expire.

The key has to come from the *id*, not from the row. ``rowid % devices`` is the obvious
cheap answer and it is wrong: rowid is assigned by insertion order within one database, so
two devices whose `prepare` runs differed in scope, in snapshot, or merely in order hold
different rowids for the same paper -- and the two "complementary" slices then both claim
some papers and skip others, which is worse than not partitioning at all. It is not stable
under ``VACUUM`` either.

``zlib.crc32`` rather than the builtin ``hash()``: string hashing is randomised per process,
so ``hash()`` would repartition the corpus on every run, silently. That is the single failure
mode this module exists to prevent, so it is worth being explicit about.

256 buckets means any device count up to 256 divides the corpus into near-equal slices, and
the count can change between runs without breaking anything -- a reshuffle only means some
paper is crawled by the other device next time, and the bucket sync reconciles that anyway.
"""

from __future__ import annotations

import os
import zlib
from typing import Any

BUCKETS = 256

# Per-device settings belong in the environment, not in `config.yaml`: that file is tracked
# in git, so a per-device value there conflicts on every pull. Same convention the MinIO
# credentials already use.
DEVICES_ENV = "ARXIV_CRAWLER_DEVICES"
INDEX_ENV = "ARXIV_CRAWLER_DEVICE_INDEX"


def bucket_for(arxiv_id: str) -> int:
    """The partition key for one paper. Stable across processes, platforms and versions."""
    return zlib.crc32(arxiv_id.encode("utf-8")) % BUCKETS


def _as_int(value: Any, source: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{source} must be an integer, got {value!r}") from None


def _from_env(name: str) -> int | None:
    return _as_int(os.environ.get(name), name)


def resolve_partition(
    devices: int | None = None,
    index: int | None = None,
    cfg: Any = None,
    *,
    plan: dict[str, Any] | None = None,
    device: str = "",
    follow_plan: bool = True,
) -> tuple[int, int] | None:
    """`(devices, index)` for this device, or None for "the whole corpus".

    Four sources, in descending precedence:

    1. an explicit CLI flag, so a one-off run can always differ from everything else;
    2. the **shared allocation** in the bucket, when it names this device -- this is what
       makes changing the device count a single edit rather than one per machine, and what
       stops a machine nobody remembered to update from crawling the wrong slice;
    3. the environment;
    4. `sync.devices` / `sync.device_index` in `config.yaml`.

    The plan sits above the environment and the config deliberately. If it sat below them, a
    machine with a stale `devices: 2` in its own config would ignore a corpus that had moved
    to four, which is the failure the plan exists to remove. `follow_plan=False`
    (`sync.follow_plan: false`) opts a machine out.

    A plan that exists but does not name this device returns nothing from source 2 and falls
    through, rather than guessing slice 0 -- guessing would collide with whichever machine
    genuinely owns slice 0.
    """
    plan_devices = plan_index = None
    if follow_plan and plan and device:
        assigned = plan_partition_of(plan, device)
        if assigned is not None:
            plan_devices, plan_index = assigned

    if devices is None:
        devices = plan_devices
    if devices is None:
        devices = _from_env(DEVICES_ENV)
    if devices is None:
        devices = _as_int(getattr(cfg, "devices", None), "sync.devices")
    if index is None:
        index = plan_index
    if index is None:
        index = _from_env(INDEX_ENV)
    if index is None:
        index = _as_int(getattr(cfg, "device_index", None), "sync.device_index")
    if devices is None and index is None:
        return None

    devices = 1 if devices is None else devices
    index = 0 if index is None else index
    if devices < 1:
        raise ValueError(f"device count must be at least 1, got {devices}")
    if devices > BUCKETS:
        raise ValueError(f"device count cannot exceed {BUCKETS}, got {devices}")
    if not 0 <= index < devices:
        raise ValueError(
            f"device index must be between 0 and {devices - 1} for {devices} devices, "
            f"got {index}"
        )
    return None if devices == 1 else (devices, index)


def plan_partition_of(plan: dict[str, Any] | None, device: str) -> tuple[int, int] | None:
    """What the shared allocation gives `device`, or None if it does not name it.

    Duplicated from `sync.plan_partition` on purpose: this module stays free of anything
    that imports the object store, so the partition logic can be reasoned about -- and
    tested -- without a bucket anywhere near it.
    """
    if not plan:
        return None
    try:
        devices = int(plan.get("devices") or 1)
        assignments = plan.get("assignments") or {}
        if device not in assignments:
            return None
        return devices, int(assignments[device])
    except (TypeError, ValueError):
        return None


def describe(partition: tuple[int, int] | None) -> str:
    if not partition:
        return "the whole corpus"
    devices, index = partition
    return f"slice {index + 1} of {devices}"
