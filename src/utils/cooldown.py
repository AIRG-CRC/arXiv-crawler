"""A global pause for when arXiv stops answering.

arXiv's export host answers ``406 Not Acceptable`` -- and sometimes ``403`` -- when it
decides a client is asking for too much. That is not a fact about the paper: the same id
fetches fine later, and every other request in flight gets the same answer at the same
moment. Handled as an ordinary request error, which is what happens to any status that is
neither 404 nor in ``RETRY_STATUS``, each paper burns all of ``crawl.max_attempts`` inside a
few seconds of backoff and is then recorded ``failed_download`` with a lifetime attempt
spent against ``retry.max_attempts``. So the crawler makes the most requests it ever makes
at precisely the moment arXiv is refusing them, and permanently marks good papers bad.

The answer is to stop, globally, for a long time. One gate, shared by every download
thread: the first thread to see a throttle closes it, waits behind a countdown, and reopens
it, while every other thread blocks. The paper that ran into the block is then retried in
place *without spending an attempt*, because waiting out a block is not a try.

The wait doubles each round the block is still there when the gate reopens -- 1h, 2h, 4h,
6h -- and resets the moment arXiv answers anything at all, 404 included: a 404 proves the
host is talking to us as well as a 200 does. After ``max_rounds`` fruitless rounds the run
is over: ``stop`` is set so the orchestrator winds down cleanly, and ``gave_up`` tells the
caller to exit non-zero, so an auto-restart wrapper does not walk straight back in.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Iterable

from tqdm import tqdm

from .logging_setup import console

log = logging.getLogger(__name__)

# How often the countdown wakes to check `stop`, and how often a bar-less cooldown says it
# is still waiting. The tick is short so a Ctrl-C is answered promptly; the heartbeat is
# long because its only job is to prove the process is alive.
TICK_SECONDS = 1.0
HEARTBEAT_SECONDS = 600

# Substrings that mark a 200 response body as a block page rather than a PDF. arXiv also
# answers 200 with an HTML "PDF is being generated, retry shortly" interstitial, which is
# genuinely per-paper and must keep its existing backoff-and-retry handling -- only the
# body distinguishes the two. Deliberately conservative: a false positive costs hours of
# idle time, so anything not clearly a block falls through to the old behaviour.
BLOCK_MARKERS: tuple[bytes, ...] = (
    b"has been blocked",
    b"been blocked",
    b"access denied",
    b"automated requests",
    b"too many requests",
    b"rate limit",
    b"slow down",
    b"unusual traffic",
)


def human_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    if total < 60:
        return f"{total}s"
    if total < 3600:
        minutes, rest = divmod(total, 60)
        return f"{minutes}m{rest:02d}s" if rest else f"{minutes}m"
    hours, rest = divmod(total, 3600)
    return f"{hours}h{rest // 60:02d}m"


def looks_like_block_page(head: bytes) -> bool:
    """Does the start of a non-PDF response body look like arXiv refusing us?"""
    low = head[:2048].lower()
    return any(marker in low for marker in BLOCK_MARKERS)


class Cooldown:
    """The gate. One instance per run, shared by every download thread.

    `enabled` is False when the feature is switched off (``crawl.cooldown_seconds <= 0``),
    in which case `enter` never blocks and `trip` never waits -- a throttle then falls
    through to the pre-existing per-paper error handling.
    """

    def __init__(
        self,
        *,
        stop: threading.Event,
        seconds: float = 3600.0,
        max_seconds: float = 21600.0,
        statuses: Iterable[int] = (403, 406),
        escalate: bool = True,
        max_rounds: int = 4,
        position: int | None = 1,
        on_resume: Callable[[], None] | None = None,
        announce: Callable[..., None] = console,
    ):
        self._stop = stop
        self._initial = float(seconds)
        self.max_seconds = max(float(max_seconds), self._initial)
        self.escalate = bool(escalate)
        self.max_rounds = max(int(max_rounds), 1)
        self.enabled = self._initial > 0 and bool(tuple(statuses))
        # Empty when disabled, so `status in cooldown.statuses` is simply never true and a
        # throttle falls through to the pre-existing per-paper error handling untouched.
        self.statuses = frozenset(int(s) for s in statuses) if self.enabled else frozenset()
        self.gave_up = False
        self.rounds_served = 0          # how many cooldowns this run has actually waited

        self._position = position
        self._on_resume = on_resume
        self._announce = announce

        self._open = threading.Event()
        self._open.set()                # set == open == downloads may proceed
        self._lock = threading.Lock()
        self._generation = 0
        self._next = self._initial
        self._rounds = 0                # consecutive rounds with no clean response since

    @property
    def paused(self) -> bool:
        """Is the gate shut right now? Reported in the heartbeat, so another machine can
        tell "stalled" from "waiting out a block"."""
        return not self._open.is_set()

    # --- what a download thread calls ---------------------------------------------------
    def enter(self) -> int | None:
        """Block while the gate is closed. Returns the generation seen, or None if stopping.

        The generation is what makes a stale trip harmless -- see `trip`. Callers must pass
        it back unchanged.
        """
        while not self._open.is_set():
            if self._stop.is_set():
                return None
            self._open.wait(0.5)
        return None if self._stop.is_set() else self._generation

    def trip(
        self,
        gen: int | None,
        status: int | str,
        arxiv_id: str,
        retry_after: float | None = None,
    ) -> bool:
        """Report a throttle. True if the caller should try this paper again.

        Exactly one thread per round runs the countdown. The others either find the gate
        already closed (and re-enter, which blocks them) or arrive with a `gen` from before
        the last resume -- meaning their 406 was issued during a block that has since been
        waited out, so waiting again would be waiting for nothing. With eight download
        threads that check is the difference between one hour of cooldown and eight
        consecutive hours, each one escalating the ladder.
        """
        if not self.enabled:
            return False
        with self._lock:
            if self._stop.is_set():
                return False
            if gen is None or gen != self._generation:
                return True                     # already waited out; just go again
            if not self._open.is_set():
                return True                     # another thread owns this round
            if self._rounds >= self.max_rounds:
                # Waited the whole ladder and arXiv has still not said anything civil.
                # Ending the run is the only remaining useful action.
                self.gave_up = True
                self._stop.set()
                self._announce(
                    "  ✗ arXiv still refusing after %d cooldown(s) — ending the run; "
                    "try again later", self._rounds,
                )
                log.error("gave up after %d cooldown round(s) at HTTP %s", self._rounds, status)
                return False

            self._rounds += 1
            self.rounds_served += 1
            round_no = self._rounds
            wait_for = self._next
            if retry_after:
                # Strictly better information than a guessed hour, when arXiv sends it.
                wait_for = max(wait_for, float(retry_after))
            wait_for = min(wait_for, self.max_seconds)
            self._open.clear()

        try:
            self._sleep(wait_for, status, arxiv_id, round_no)
        finally:
            self._resume()
        return not self._stop.is_set()

    def note_clean(self) -> None:
        """arXiv answered something that is not a throttle. Reset the ladder.

        Called for a 404 as well as a success: a 404 is proof the host is answering, and
        treating a run of withdrawn papers as "still blocked" would escalate the next
        unrelated 406 straight to two hours.
        """
        if not self.enabled or (self._rounds == 0 and self._next == self._initial):
            return                              # fast path: the common case takes no lock
        with self._lock:
            self._rounds = 0
            self._next = self._initial

    # --- internals ----------------------------------------------------------------------
    def _resume(self) -> None:
        """Hand control back: reset the caller's rate limiter and bar, then open the gate.

        The hook runs *before* the gate opens, so the first request after a cooldown cannot
        go out against a token bucket that refilled to `burst` while we waited. Both steps
        are guarded: a hook that raised would leave the gate shut and every download thread
        parked for the rest of the process.
        """
        if self._on_resume is not None:
            try:
                self._on_resume()
            except Exception:                   # never let telemetry wedge the run
                log.exception("cooldown resume hook failed")
        with self._lock:
            self._generation += 1
            if self.escalate:
                self._next = min(self._next * 2, self.max_seconds)
            self._open.set()

    def _sleep(self, seconds: float, status: int | str, arxiv_id: str, round_no: int) -> None:
        self._announce(
            "  ⏸ arXiv answered HTTP %s on %s — pausing every download for %s "
            "(cooldown %d of %d)",
            status, arxiv_id, human_duration(seconds), round_no, self.max_rounds,
        )
        log.warning("cooldown %d/%d: HTTP %s on %s, waiting %ds",
                    round_no, self.max_rounds, status, arxiv_id, int(round(seconds)))

        label = f"arXiv HTTP {status}"
        bar = self._make_bar(max(1, int(round(seconds))),
                             f"{label} — {human_duration(seconds)} left")

        def _repaint(left: float) -> None:
            # The remaining time is known exactly, so it is written rather than inferred:
            # tqdm's own {remaining} needs a rate estimate and shows "?" until the first
            # tick, which on an hour-long pause is the least reassuring moment to show it.
            if bar is not None:
                bar.set_description_str(f"{label} — {human_duration(left)} left",
                                        refresh=False)
        started = time.monotonic()
        deadline = started + seconds
        shown = 0                           # whole seconds already credited to the bar
        next_heartbeat = HEARTBEAT_SECONDS
        try:
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                if self._stop.wait(min(TICK_SECONDS, left)):
                    break
                elapsed = int(time.monotonic() - started)
                if bar is not None:
                    if elapsed > shown:
                        _repaint(deadline - time.monotonic())
                        bar.update(elapsed - shown)
                        shown = elapsed
                elif elapsed >= next_heartbeat:
                    # Nowhere to draw a bar, so say something occasionally rather than leave
                    # a nohup log silent for six hours.
                    next_heartbeat += HEARTBEAT_SECONDS
                    self._announce("  ⏸ still paused — %s left",
                                   human_duration(deadline - time.monotonic()))
        finally:
            if bar is not None:
                bar.close()

        if self._stop.is_set():
            self._announce("  ▶ cooldown interrupted — winding down")
        else:
            self._announce("  ▶ resuming after %s", human_duration(time.monotonic() - started))

    def _make_bar(self, total: int, desc: str) -> Any:
        """The countdown, or None when there is nowhere to draw it.

        tqdm silently draws nothing for a bar whose position is past the terminal's last
        row, so a deep position on a short terminal would leave the run looking frozen for
        hours with no explanation. Detect that and fall back to the heartbeat instead.
        """
        if self._position is None:
            return None
        try:
            bar = tqdm(
                total=total, position=self._position, leave=False, unit="s",
                # One tick a second keeps `stop` responsive; repainting only every five
                # keeps a non-tty log to dozens of lines instead of 21,600.
                mininterval=5.0, smoothing=0, desc=desc,
                bar_format="  ⏸ {desc} {bar}",
            )
        except Exception:                       # no terminal, closed stream, ...
            return None
        rows = getattr(bar, "nrows", None) or 20
        if self._position >= rows - 1:
            bar.close()
            return None
        return bar
