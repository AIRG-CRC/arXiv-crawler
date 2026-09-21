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

BUCKETS = 256

# Per-device settings belong in the environment, not in `config.yaml`: that file is tracked
# in git, so a per-device value there conflicts on every pull. Same convention the MinIO
# credentials already use.
DEVICES_ENV = "ARXIV_CRAWLER_DEVICES"
INDEX_ENV = "ARXIV_CRAWLER_DEVICE_INDEX"


def bucket_for(arxiv_id: str) -> int:
    """The partition key for one paper. Stable across processes, platforms and versions."""
    return zlib.crc32(arxiv_id.encode("utf-8")) % BUCKETS


def _from_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None


def resolve_partition(
    devices: int | None = None, index: int | None = None
) -> tuple[int, int] | None:
    """`(devices, index)` for this device, or None for "the whole corpus".

    Explicit arguments win over the environment, so a one-off run can override a shell
    profile. `None` is returned for a single device so the claim SQL stays byte-for-byte
    what it was before partitioning existed.
    """
    if devices is None:
        devices = _from_env(DEVICES_ENV)
    if index is None:
        index = _from_env(INDEX_ENV)
    if devices is None and index is None:
        return None

    devices = 1 if devices is None else int(devices)
    index = 0 if index is None else int(index)
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


def describe(partition: tuple[int, int] | None) -> str:
    if not partition:
        return "the whole corpus"
    devices, index = partition
    return f"slice {index + 1} of {devices}"
