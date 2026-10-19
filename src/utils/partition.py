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

A device owns a contiguous *range* of the 256 buckets -- a `Slice` -- rather than a residue
class. Ranges are what make uneven shares expressible ("this machine 60%, that one 40%"), and
they are what keep a reallocation cheap: a new set of weights only slides the boundaries,
so the buckets that change hands are the difference between the old and new shares (50/50 to
60/40 hands over 10% of the corpus). Under `bucket % n` changing `n` moved almost every paper
to a different device.

The granularity is 1/256 of the corpus, about 0.4%, which is finer than anyone sets a share.
"""

from __future__ import annotations

import os
import threading
import zlib
from dataclasses import dataclass
from typing import Any, Iterable

BUCKETS = 256

# Per-device settings belong in the environment, not in `config.yaml`: that file is tracked
# in git, so a per-device value there conflicts on every pull. Same convention the MinIO
# credentials already use.
DEVICES_ENV = "ARXIV_CRAWLER_DEVICES"
INDEX_ENV = "ARXIV_CRAWLER_DEVICE_INDEX"


def bucket_for(arxiv_id: str) -> int:
    """The partition key for one paper. Stable across processes, platforms and versions."""
    return zlib.crc32(arxiv_id.encode("utf-8")) % BUCKETS


@dataclass(frozen=True)
class Slice:
    """The buckets `lo <= bucket < hi` -- one device's share of the corpus.

    Always truthy, even when empty: `None` is what means "the whole corpus", and an empty
    slice -- a device whose weight was set to 0 -- must filter everything *out*, not be
    mistaken for no filter at all.
    """

    lo: int
    hi: int

    def __post_init__(self) -> None:
        if not 0 <= self.lo <= self.hi <= BUCKETS:
            raise ValueError(f"a slice must lie within 0-{BUCKETS}, got {self.lo}-{self.hi}")

    @classmethod
    def equal(cls, devices: int, index: int) -> "Slice":
        """Slice `index` of `devices` near-equal ones -- what --devices/--device-index mean."""
        return cls(index * BUCKETS // devices, (index + 1) * BUCKETS // devices)

    @classmethod
    def from_json(cls, value: Any) -> "Slice":
        lo, hi = value
        return cls(int(lo), int(hi))

    def to_json(self) -> list[int]:
        return [self.lo, self.hi]

    @property
    def empty(self) -> bool:
        return self.hi <= self.lo

    @property
    def share(self) -> float:
        return (self.hi - self.lo) / BUCKETS

    def contains(self, bucket: int) -> bool:
        return self.lo <= bucket < self.hi

    def sql(self, column: str = "bucket") -> tuple[str, tuple[int, int]]:
        """A predicate for this slice, without its leading AND/WHERE."""
        return f"{column} >= ? AND {column} < ?", (self.lo, self.hi)

    def describe(self) -> str:
        if self.empty:
            return "no share (0%)"
        return f"buckets {self.lo}-{self.hi - 1} ({self.share * 100:.1f}%)"

    def overlaps(self, other: "Slice") -> bool:
        return not (self.empty or other.empty) and self.lo < other.hi and other.lo < self.hi


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


def ranges_from_weights(order: Iterable[str], weights: dict[str, float]) -> dict[str, Slice]:
    """Contiguous slices, laid out in `order`, sized in proportion to `weights`.

    Largest remainder, so the sizes add up to exactly `BUCKETS` and each is within one
    bucket of its exact share. A positive weight always gets at least one bucket -- a device
    you gave a share to and which then crawls nothing is a surprise -- and a weight of 0 gets
    an empty slice, which is how a machine is parked without dropping it from the plan.

    Laying the slices out in a stable order is what keeps reallocation cheap: the boundaries
    slide, so only the difference between the old and new shares changes hands.
    """
    names = list(dict.fromkeys(order))
    for name in weights:
        if name not in names:
            names.append(name)
    positive = [n for n in names if float(weights.get(n, 0) or 0) > 0]
    if not positive:
        raise ValueError("at least one device needs a weight above 0")
    if len(positive) > BUCKETS:
        raise ValueError(f"at most {BUCKETS} devices can hold a share")
    if any(float(weights.get(n, 0) or 0) < 0 for n in names):
        raise ValueError("weights cannot be negative")

    total = sum(float(weights[n]) for n in positive)
    exact = {n: float(weights[n]) / total * BUCKETS for n in positive}
    size = {n: int(exact[n]) for n in positive}
    spare = BUCKETS - sum(size.values())
    for n in sorted(positive, key=lambda n: (-(exact[n] - size[n]), names.index(n)))[:spare]:
        size[n] += 1
    for n in positive:
        if size[n] == 0:
            donor = max(positive, key=lambda d: size[d])
            size[donor] -= 1
            size[n] = 1

    out: dict[str, Slice] = {}
    lo = 0
    for name in names:
        width = size.get(name, 0)
        out[name] = Slice(lo, lo + width)
        lo += width
    return out


def resolve_partition(
    devices: int | None = None,
    index: int | None = None,
    cfg: Any = None,
    *,
    plan: dict[str, Any] | None = None,
    device: str = "",
    follow_plan: bool = True,
) -> Slice | None:
    """This device's `Slice`, or None for "the whole corpus".

    Four sources, in descending precedence:

    1. an explicit CLI flag, so a one-off run can always differ from everything else;
    2. the **shared allocation** in the bucket, when it names this device -- this is what
       makes changing the split a single edit rather than one per machine, and what stops a
       machine nobody remembered to update from crawling the wrong slice;
    3. the environment;
    4. `sync.devices` / `sync.device_index` in `config.yaml`.

    Sources 1, 3 and 4 describe an equal split (`devices`, `index`); the plan can describe
    any split. The plan sits above the environment and the config deliberately. If it sat
    below them, a machine with a stale `devices: 2` in its own config would ignore a corpus
    that had moved to four, which is the failure the plan exists to remove.
    `follow_plan=False` (`sync.follow_plan: false`) opts a machine out.

    A plan that exists but does not name this device returns nothing from source 2 and falls
    through, rather than guessing slice 0 -- guessing would collide with whichever machine
    genuinely owns slice 0.
    """
    if devices is None and index is None and follow_plan and plan and device:
        assigned = plan_partition_of(plan, device)
        if assigned is not None:
            return None if (assigned.lo, assigned.hi) == (0, BUCKETS) else assigned

    if devices is None:
        devices = _from_env(DEVICES_ENV)
    if devices is None:
        devices = _as_int(getattr(cfg, "devices", None), "sync.devices")
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
    return None if devices == 1 else Slice.equal(devices, index)


def plan_ranges(plan: dict[str, Any] | None) -> dict[str, Slice]:
    """Every device's slice in a plan, whichever format it was written in.

    Version 1 plans (`devices` + `assignments`) predate uneven shares; they are read as the
    equal split they describe. Anything malformed reads as an empty allocation.
    """
    if not plan:
        return {}
    try:
        if "ranges" in plan:
            return {str(k): Slice.from_json(v) for k, v in (plan.get("ranges") or {}).items()}
        devices = int(plan.get("devices") or 1)
        return {str(k): Slice.equal(devices, int(v))
                for k, v in (plan.get("assignments") or {}).items()}
    except (TypeError, ValueError):
        return {}


def plan_partition_of(plan: dict[str, Any] | None, device: str) -> Slice | None:
    """What the shared allocation gives `device`, or None if it does not name it.

    Kept here rather than next to the plan I/O in `sync`: this module stays free of
    anything that imports the object store, so the partition logic can be reasoned about --
    and tested -- without a bucket anywhere near it.
    """
    return plan_ranges(plan).get(device)


class LiveSlice:
    """This device's slice, as it stands *now* -- changed under a running crawl.

    The heartbeat thread re-reads the shared allocation and calls `update`; the dispatch loop
    reads `current` on every claim and compares `version` once per turn to notice a change.
    One lock and a counter, rather than a queue of events: only the latest allocation
    matters, and a change that is superseded before the loop sees it needs no handling.
    """

    def __init__(self, current: "Slice | None" = None):
        self._lock = threading.Lock()
        self._current = current
        self.version = 0

    @property
    def current(self) -> "Slice | None":
        with self._lock:
            return self._current

    def update(self, new: "Slice | None") -> bool:
        """Adopt `new`. True if it differs from what was there."""
        with self._lock:
            if new == self._current:
                return False
            self._current = new
            self.version += 1
            return True


def describe(partition: Slice | None) -> str:
    if not partition:
        return "the whole corpus"
    return partition.describe()
