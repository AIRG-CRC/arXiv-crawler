"""Per-worker checkpointing.

Each worker owns exactly one file, ``data/checkpoints/worker-NN.json``, and every
completed paper increments that worker's counter by one. Because no two workers ever
touch the same file there is nothing to lock, and a torn write can only ever damage one
worker's tally -- never the shared record of progress.

The manifest remains the authority on *which* papers still need doing; these files are
the authority on *how far a run got*, which is what makes a resumed run able to say
"continuing from 1,240" instead of restarting its progress bar at zero.

Layout while a run is in flight:

    data/checkpoints/run.json          run id, start time, target, settings
    data/checkpoints/worker-00.json    one per worker, counters plus last paper seen
    data/checkpoints/worker-01.json

On clean completion those collapse to a single ``summary.json`` and the per-worker files
are removed, so the directory never accumulates debris across runs.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUN_FILE = "run.json"
SUMMARY_FILE = "summary.json"
WORKER_GLOB = "worker-*.json"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".ckpt-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=False)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


@dataclass
class WorkerState:
    """One worker's tally. `processed` is the sum of the three outcome counters."""

    slot: int
    pid: int = 0
    processed: int = 0
    done: int = 0
    failed: int = 0
    no_pdf: int = 0
    last_paper: str = ""
    started_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)


@dataclass
class Totals:
    processed: int = 0
    done: int = 0
    failed: int = 0
    no_pdf: int = 0
    workers: int = 0


class CheckpointStore:
    """Reads and writes the per-worker checkpoint files.

    `bump()` is the only mutating call: one paper, one increment, one file rewritten.
    It is called from the orchestrator's single result-handling thread, and guarded by a
    lock regardless so the store stays correct if that ever becomes concurrent.
    """

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._workers: dict[int, WorkerState] = {}
        self._slot_of: dict[int, int] = {}     # pid -> slot
        self._lock = threading.Lock()
        self.run_id = ""

    # --- run lifecycle ---------------------------------------------------------------
    def begin(self, *, target: int, settings: dict[str, Any] | None = None) -> Totals:
        """Start or resume a run. Returns what previous, unfinished workers achieved."""
        carried = self.load_workers()
        self.run_id = uuid.uuid4().hex[:12]
        # A completed run leaves only summary.json; its counters are history, not
        # progress, so a fresh run starts from zero rather than inheriting them.
        (self.root / SUMMARY_FILE).unlink(missing_ok=True)
        _atomic_write_json(self.root / RUN_FILE, {
            "run_id": self.run_id,
            "started_at": _utcnow(),
            "target": target,
            "resumed_from": asdict(self.totals(carried)),
            "settings": settings or {},
        })
        self._workers = {w.slot: w for w in carried}
        self._slot_of = {w.pid: w.slot for w in carried if w.pid}
        return self.totals(carried)

    def finalize(self, tallies: dict[str, int] | None = None) -> None:
        """Collapse a finished run to one summary file and clear the worker files."""
        totals = self.totals()
        payload = {
            "run_id": self.run_id,
            "finished_at": _utcnow(),
            "totals": asdict(totals),
            "per_worker": [asdict(w) for w in sorted(self._workers.values(), key=lambda w: w.slot)],
        }
        if tallies:
            payload["run_tallies"] = tallies
        _atomic_write_json(self.root / SUMMARY_FILE, payload)
        for path in self.root.glob(WORKER_GLOB):
            path.unlink(missing_ok=True)
        (self.root / RUN_FILE).unlink(missing_ok=True)

    # --- mutation --------------------------------------------------------------------
    def bump(self, pid: int, status: str, arxiv_id: str) -> WorkerState:
        """Record one completed paper against the worker that handled it."""
        with self._lock:
            state = self._state_for(pid)
            state.processed += 1
            if status == "done":
                state.done += 1
            elif status == "no_pdf":
                state.no_pdf += 1
            else:
                state.failed += 1
            state.last_paper = arxiv_id
            state.updated_at = _utcnow()
            _atomic_write_json(self._path(state.slot), asdict(state))
            return state

    def _state_for(self, pid: int) -> WorkerState:
        """Map a process id to a stable slot, so file names stay small and ordered."""
        slot = self._slot_of.get(pid)
        if slot is None:
            slot = self._slot_of[pid] = self._next_free_slot()
            self._workers[slot] = WorkerState(slot=slot, pid=pid)
        state = self._workers.get(slot)
        if state is None:
            state = self._workers[slot] = WorkerState(slot=slot, pid=pid)
        state.pid = pid
        return state

    def _next_free_slot(self) -> int:
        used = set(self._workers)
        slot = 0
        while slot in used:
            slot += 1
        return slot

    def _path(self, slot: int) -> Path:
        return self.root / f"worker-{slot:02d}.json"

    # --- reading ---------------------------------------------------------------------
    def load_workers(self) -> list[WorkerState]:
        states: list[WorkerState] = []
        for path in sorted(self.root.glob(WORKER_GLOB)):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                states.append(WorkerState(**{
                    k: v for k, v in raw.items() if k in WorkerState.__dataclass_fields__
                }))
            except (OSError, ValueError, TypeError):
                # A checkpoint torn by a hard kill costs that worker's tally, nothing
                # more -- the manifest still knows exactly which papers are outstanding.
                continue
        return states

    def totals(self, states: list[WorkerState] | None = None) -> Totals:
        states = list(self._workers.values()) if states is None else states
        return Totals(
            processed=sum(s.processed for s in states),
            done=sum(s.done for s in states),
            failed=sum(s.failed for s in states),
            no_pdf=sum(s.no_pdf for s in states),
            workers=len(states),
        )

    def describe(self) -> str:
        """Human-readable status, for `main.py checkpoint`."""
        summary = self.root / SUMMARY_FILE
        states = self.load_workers()
        if not states and summary.exists():
            data = json.loads(summary.read_text(encoding="utf-8"))
            t = data.get("totals", {})
            return (f"last run finished {data.get('finished_at', '?')}: "
                    f"{t.get('processed', 0):,} processed across {t.get('workers', 0)} worker(s)")
        if not states:
            return "no checkpoints yet"

        lines = [f"{'worker':<9}{'processed':>11}{'done':>9}{'failed':>9}{'no_pdf':>9}  last paper"]
        lines.append("-" * 68)
        for s in sorted(states, key=lambda s: s.slot):
            lines.append(f"{s.slot:<9}{s.processed:>11,}{s.done:>9,}{s.failed:>9,}"
                         f"{s.no_pdf:>9,}  {s.last_paper}")
        t = self.totals(states)
        lines.append("-" * 68)
        lines.append(f"{'total':<9}{t.processed:>11,}{t.done:>9,}{t.failed:>9,}{t.no_pdf:>9,}")
        run = self.root / RUN_FILE
        if run.exists():
            data = json.loads(run.read_text(encoding="utf-8"))
            lines.append(f"\nrun {data.get('run_id')} started {data.get('started_at')}, "
                         f"target {data.get('target', 0):,} paper(s) — interrupted, resumable")
        return "\n".join(lines)
