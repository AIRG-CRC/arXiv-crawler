"""Keeping the machine usable while the crawler runs.

A conversion worker holding a model is a big, long-lived process, and docling's grows as
it works: measured on this project, ~0.3 GB per paper with no sign of levelling off, on
top of a ~2.7 GB working set for a 128-page document. Four of those will eventually walk
into the kernel's OOM killer, and the kernel does not pick the process that caused the
problem -- it picks the biggest convenient victim, which on a desktop is the editor.

So two defences, both cheap:

  * `available()` lets the orchestrator stop handing out new work while memory is short,
    which turns a crash into a slowdown;
  * `RSS_LIMIT`-style recycling (see `convert.max_tasks_per_child`) caps how long any one
    worker lives, so growth is bounded by construction rather than by hope.

Everything here degrades to "no opinion" off Linux rather than guessing.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

MEMINFO = Path("/proc/meminfo")


def _meminfo() -> dict[str, int]:
    """/proc/meminfo as bytes. Empty when unavailable, e.g. off Linux."""
    try:
        out: dict[str, int] = {}
        for line in MEMINFO.read_text().splitlines():
            key, _, rest = line.partition(":")
            parts = rest.split()
            if parts:
                out[key] = int(parts[0]) * 1024      # values are in kB
        return out
    except (OSError, ValueError):
        return {}


def available_bytes() -> int | None:
    """Memory that can be handed out without swapping, or None if unknowable.

    `MemAvailable` is the kernel's own estimate and is the right number here: `MemFree`
    ignores reclaimable page cache and would make a healthy machine look exhausted.
    """
    info = _meminfo()
    value = info.get("MemAvailable")
    return int(value) if value is not None else None


def total_bytes() -> int | None:
    info = _meminfo()
    value = info.get("MemTotal")
    return int(value) if value is not None else None


def rss_bytes(pid: int | None = None) -> int | None:
    """Resident set size of a process, straight from /proc."""
    try:
        fields = Path(f"/proc/{pid or os.getpid()}/statm").read_text().split()
        return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


def resolve_floor(setting: float | int | str | None) -> int | None:
    """Interpret `convert.memory_floor` as bytes.

    Accepts a fraction of total RAM (``0.15``), an absolute size (``"4GB"``), or None to
    disable the guard entirely.
    """
    if setting in (None, "", False):
        return None
    total = total_bytes()
    if isinstance(setting, (int, float)) and not isinstance(setting, bool):
        if 0 < float(setting) < 1:                   # a share of the machine
            return int(float(setting) * total) if total else None
        return int(setting)                          # already bytes
    text = str(setting).strip().upper().replace("I", "")
    for suffix, scale in (("TB", 2**40), ("GB", 2**30), ("MB", 2**20), ("KB", 2**10)):
        if text.endswith(suffix):
            try:
                return int(float(text[: -len(suffix)]) * scale)
            except ValueError:
                break
    try:
        return int(float(text))
    except ValueError:
        log.warning("could not interpret memory_floor=%r; guard disabled", setting)
        return None


def human(n: float | None) -> str:
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


_malloc_trim = None
_trim_looked_up = False


def release_memory() -> bool:
    """Hand freed heap back to the operating system. Returns whether it ran.

    PDF conversion allocates and frees in a pattern that leaves glibc's arenas badly
    fragmented, and glibc keeps those arenas rather than returning them. The effect is a
    worker that *looks* like it is leaking: measured here, +0.3 GB per paper, climbing
    2.70 -> 3.06 -> 3.35 -> 3.68 GB over four documents with no sign of levelling off.
    Four workers doing that will find the OOM killer, and the kernel's victim is whatever
    is biggest and handy -- on a desktop, the editor rather than the crawler.

    `malloc_trim(0)` returns the free arenas. With it, the same loop sits flat at ~2.3 GB.
    It is a glibc call, so it simply does not run elsewhere; costs a few milliseconds.
    """
    global _malloc_trim, _trim_looked_up
    if not _trim_looked_up:
        _trim_looked_up = True
        try:
            import ctypes
            import ctypes.util

            libc_name = ctypes.util.find_library("c") or "libc.so.6"
            candidate = ctypes.CDLL(libc_name)
            if hasattr(candidate, "malloc_trim"):
                candidate.malloc_trim.argtypes = [ctypes.c_size_t]
                candidate.malloc_trim.restype = ctypes.c_int
                _malloc_trim = candidate.malloc_trim
        except (OSError, AttributeError, ImportError):
            _malloc_trim = None
    if _malloc_trim is None:
        return False
    try:
        _malloc_trim(0)
        return True
    except Exception:                                # never fail a paper over this
        return False


class MemoryGuard:
    """Back-pressure: stop starting new papers when the machine is running short.

    This does not make a worker smaller. It stops the orchestrator adding to the problem,
    so the papers already in flight can finish and hand their memory back. A run that
    slows down is strictly better than a desktop session the kernel decided to kill.
    """

    def __init__(self, floor_bytes: int | None, *, on_pause=None):
        self.floor = floor_bytes
        self._on_pause = on_pause
        self.paused = False
        self.pauses = 0

    @property
    def enabled(self) -> bool:
        return bool(self.floor)

    def has_headroom(self) -> bool:
        """True when it is safe to dispatch more work."""
        if not self.floor:
            return True
        free = available_bytes()
        if free is None:                             # unknowable: do not pretend
            return True
        if free < self.floor:
            if not self.paused:
                self.paused = True
                self.pauses += 1
                log.warning("pausing dispatch: %s available, floor is %s",
                            human(free), human(self.floor))
                if self._on_pause:
                    self._on_pause(free, self.floor)
            return False
        self.paused = False
        return True
