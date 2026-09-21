"""The global throttle gate.

Every test here runs with a sub-second `seconds` so the suite stays fast; the mechanism is
identical at 3600. What is actually being pinned down:

  * a throttle costs no retry attempt, while an ordinary 500 still does
  * a stale trip -- a 406 issued before the last resume -- does not wait a second time
  * `stop` mid-countdown returns promptly *and* reopens the gate, because a gate left shut
    hangs `dl_pool.shutdown(wait=True)` for the rest of the wait
"""

from __future__ import annotations

import threading
import time

import pytest

from src.utils import crawler as C
from src.utils.cooldown import Cooldown, human_duration, looks_like_block_page
from src.utils.state import DONE, FAILED_DOWNLOAD, NO_PDF, PENDING, PaperRow


@pytest.fixture(autouse=True)
def _no_per_paper_backoff(monkeypatch):
    """Skip `_sleep_for_retry`.

    The per-paper exponential backoff is tested by the retry suite, not here, and at
    max_attempts=3 it costs several seconds per case. The cooldown's own waits are real.
    """
    monkeypatch.setattr(C, "_sleep_for_retry", lambda response, attempt, stop: None)


def _cooldown(**kw):
    kw.setdefault("stop", threading.Event())
    kw.setdefault("seconds", 0.05)
    kw.setdefault("max_seconds", 10.0)
    kw.setdefault("position", None)          # no bar in tests
    kw.setdefault("announce", lambda *a: None)
    return Cooldown(**kw)


# --- the gate ------------------------------------------------------------------------
def test_disabled_cooldown_is_inert():
    cd = _cooldown(seconds=0)
    assert not cd.enabled
    # An empty status set is what makes a throttle fall through to the old error handling.
    assert cd.statuses == frozenset()
    assert cd.enter() == 0
    assert cd.trip(0, 406, "2301.00001") is False
    assert cd.rounds_served == 0


def test_trip_closes_then_reopens_the_gate():
    cd = _cooldown()
    gen = cd.enter()
    assert cd.trip(gen, 406, "2301.00001") is True
    assert cd.enter() == gen + 1             # generation advanced exactly once
    assert cd.rounds_served == 1


def test_enter_blocks_while_another_thread_waits():
    cd = _cooldown(seconds=0.4)
    gen = cd.enter()
    order: list[str] = []

    def sleeper():
        cd.trip(gen, 406, "2301.00001")
        order.append("resumed")

    def waiter():
        time.sleep(0.05)                     # let the sleeper close the gate first
        cd.enter()
        order.append("entered")

    threads = [threading.Thread(target=sleeper), threading.Thread(target=waiter)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(5)
    assert order == ["resumed", "entered"]


def test_stale_generation_does_not_wait_again():
    """The hole that turns one round of throttling into N consecutive cooldowns."""
    cd = _cooldown(seconds=0.05)
    stale = cd.enter()
    assert cd.trip(stale, 406, "a") is True  # round one, served
    assert cd.rounds_served == 1

    # A second thread's 406 was issued during that block and only now reports it. Waiting
    # again would be waiting for a block already waited out.
    started = time.monotonic()
    assert cd.trip(stale, 406, "b") is True
    assert time.monotonic() - started < 0.04
    assert cd.rounds_served == 1


# --- the ladder ----------------------------------------------------------------------
def test_ladder_doubles_and_a_clean_answer_resets_it():
    cd = _cooldown(seconds=0.05, max_seconds=1.0)
    waits: list[float] = []
    for _ in range(3):
        gen = cd.enter()
        start = time.monotonic()
        cd.trip(gen, 406, "x")
        waits.append(time.monotonic() - start)
    assert waits[1] > waits[0] * 1.5 and waits[2] > waits[1] * 1.5

    cd.note_clean()
    gen = cd.enter()
    start = time.monotonic()
    cd.trip(gen, 406, "x")
    assert time.monotonic() - start < waits[1]      # back to the first rung


def test_ladder_is_capped_by_max_seconds():
    cd = _cooldown(seconds=0.05, max_seconds=0.06)
    for _ in range(3):
        cd.trip(cd.enter(), 406, "x")
    assert cd._next <= 0.06


def test_retry_after_raises_the_wait_but_never_past_the_cap():
    cd = _cooldown(seconds=0.01, max_seconds=0.08)
    start = time.monotonic()
    cd.trip(cd.enter(), 429, "x", retry_after=10.0)
    elapsed = time.monotonic() - start
    assert 0.05 < elapsed < 0.5              # honoured, but clamped to max_seconds


def test_escalation_can_be_switched_off():
    cd = _cooldown(seconds=0.05, escalate=False)
    for _ in range(2):
        cd.trip(cd.enter(), 406, "x")
    assert cd._next == pytest.approx(0.05)


def test_gives_up_after_max_rounds_and_sets_stop():
    stop = threading.Event()
    cd = _cooldown(stop=stop, seconds=0.01, max_rounds=2)
    assert cd.trip(cd.enter(), 406, "x") is True
    assert cd.trip(cd.enter(), 406, "x") is True
    # The third report has nothing left to wait for.
    assert cd.trip(cd.enter(), 406, "x") is False
    assert cd.gave_up and stop.is_set()
    assert cd.rounds_served == 2


# --- interruption --------------------------------------------------------------------
def test_stop_mid_countdown_returns_promptly_and_reopens_the_gate():
    stop = threading.Event()
    cd = _cooldown(stop=stop, seconds=30.0)
    gen = cd.enter()
    result: list[bool] = []

    th = threading.Thread(target=lambda: result.append(cd.trip(gen, 406, "x")))
    th.start()
    time.sleep(0.2)
    stop.set()
    th.join(5)
    assert not th.is_alive()                 # a shut gate here hangs pool shutdown
    assert result == [False]                 # the run is over; do not retry the paper
    assert cd.enter() is None                # and every other thread is released


def test_trip_after_stop_does_not_start_a_wait():
    stop = threading.Event()
    cd = _cooldown(stop=stop, seconds=30.0)
    gen = cd.enter()
    stop.set()
    start = time.monotonic()
    assert cd.trip(gen, 406, "x") is False
    assert time.monotonic() - start < 0.1


def test_resume_hook_runs_before_the_gate_reopens():
    seen: list[bool] = []
    cd = _cooldown()
    cd._on_resume = lambda: seen.append(cd._open.is_set())
    cd.trip(cd.enter(), 406, "x")
    assert seen == [False]                   # gate still shut when the hook ran


def test_a_raising_resume_hook_still_reopens_the_gate():
    def boom() -> None:
        raise RuntimeError("no")

    cd = _cooldown(on_resume=boom)
    cd.trip(cd.enter(), 406, "x")
    assert cd._open.is_set()


# --- helpers -------------------------------------------------------------------------
@pytest.mark.parametrize("seconds,expected", [
    (9, "9s"), (60, "1m"), (90, "1m30s"), (3600, "1h00m"), (5400, "1h30m"), (21600, "6h00m"),
])
def test_human_duration(seconds, expected):
    assert human_duration(seconds) == expected


def test_block_page_sniffing_is_conservative():
    assert looks_like_block_page(b"<html><body>Your IP has been blocked</body></html>")
    assert looks_like_block_page(b"<html>Too many requests, please slow down</html>")
    # The "PDF is being generated" interstitial must keep its per-paper retry handling.
    assert not looks_like_block_page(
        b"<html><body>Your PDF is being generated, please retry shortly.</body></html>")
    assert not looks_like_block_page(b"")


# --- download_one: the attempt accounting --------------------------------------------
class _Resp:
    """The slice of requests.Response that download_one touches."""

    def __init__(self, status, body=b"", headers=None):
        self.status_code = status
        self.headers = headers or {}
        self._body = body
        self.closed = False

    def iter_content(self, chunk_size=0):
        yield self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise C.requests.HTTPError(f"{self.status_code} Client Error")

    def close(self):
        self.closed = True


class _Session:
    """Stands in for ArxivSession, handing out a scripted sequence of responses."""

    def __init__(self, cfg, responses, cooldown):
        self.cfg = cfg
        self.cooldown = cooldown
        self.requests = 0
        self._responses = list(responses)

    @property
    def session(self):
        return self

    def get(self, url, timeout=None, stream=False):
        self.requests += 1
        return self._responses.pop(0)

    def pdf_url(self, row):
        return "http://example.invalid/pdf/x"


class _Cfg:
    max_attempts = 3
    timeout = 1
    chunk_size = 4096


def _run_download(tmp_path, responses, cooldown):
    row = PaperRow(arxiv_id="2301.00001", version="v1", shard="2301")
    session = _Session(_Cfg(), responses, cooldown)
    limiter = C.RateLimiter(1000.0, 1000)
    return session, C.download_one(row, session, limiter, tmp_path, threading.Event())


def test_a_throttle_then_a_pdf_succeeds_without_spending_an_attempt(tmp_path):
    cd = _cooldown(seconds=0.01)
    session, outcome = _run_download(
        tmp_path, [_Resp(406), _Resp(200, b"%PDF-1.7 body")], cd)
    assert outcome.status == DONE
    assert session.requests == 2
    assert cd.rounds_served == 1
    # A clean answer arrived, so the next throttle starts at the first rung again.
    assert cd._rounds == 0


def test_a_paper_that_only_ever_throttles_comes_back_pending(tmp_path):
    """Pending, not failed_download: the orchestrator records it without an attempt."""
    cd = _cooldown(seconds=0.01, max_rounds=2)
    _, outcome = _run_download(tmp_path, [_Resp(406)] * 6, cd)
    assert outcome.status == PENDING


def test_a_retryable_status_still_spends_every_attempt(tmp_path):
    cd = _cooldown(seconds=0.01)
    session, outcome = _run_download(tmp_path, [_Resp(503)] * 3, cd)
    assert outcome.status == FAILED_DOWNLOAD
    assert session.requests == 3             # max_attempts, all spent
    assert cd.rounds_served == 0             # and no cooldown for a 503


def test_404_is_a_clean_answer(tmp_path):
    cd = _cooldown(seconds=0.01)
    cd._rounds, cd._next = 2, 5.0
    _, outcome = _run_download(tmp_path, [_Resp(404)], cd)
    assert outcome.status == NO_PDF
    assert cd._rounds == 0 and cd._next == cd._initial


def test_a_block_page_behind_a_200_trips_the_cooldown(tmp_path):
    cd = _cooldown(seconds=0.01)
    session, outcome = _run_download(tmp_path, [
        _Resp(200, b"<html>Access denied: automated requests</html>"),
        _Resp(200, b"%PDF-1.7 body"),
    ], cd)
    assert outcome.status == DONE
    assert cd.rounds_served == 1
    assert not (tmp_path / "tmp" / "2301.00001.pdf.part").exists()


def test_a_generated_pdf_interstitial_is_still_an_ordinary_failure(tmp_path):
    cd = _cooldown(seconds=0.01)
    _, outcome = _run_download(tmp_path, [
        _Resp(200, b"<html>PDF is being generated</html>")] * 3, cd)
    assert outcome.status == FAILED_DOWNLOAD
    assert cd.rounds_served == 0


def test_with_the_cooldown_disabled_a_406_behaves_as_before(tmp_path):
    cd = _cooldown(seconds=0)
    session, outcome = _run_download(tmp_path, [_Resp(406)] * 3, cd)
    assert outcome.status == FAILED_DOWNLOAD
    assert session.requests == 3
    assert "406" in (outcome.error or "")


def test_the_response_is_closed_before_the_wait(tmp_path):
    cd = _cooldown(seconds=0.01)
    throttle = _Resp(406)
    session, _ = _run_download(tmp_path, [throttle, _Resp(200, b"%PDF-1.7 x")], cd)
    assert throttle.closed


def test_an_interrupt_before_the_request_spends_nothing(tmp_path):
    row = PaperRow(arxiv_id="2301.00001", version="v1", shard="2301")
    session = _Session(_Cfg(), [], _cooldown())
    stop = threading.Event()
    stop.set()
    outcome = C.download_one(row, session, C.RateLimiter(1000.0, 1000), tmp_path, stop)
    assert outcome.status == PENDING
    assert session.requests == 0


# --- the pieces the cooldown leans on ------------------------------------------------
def test_rate_limiter_drain_empties_the_bucket():
    limiter = C.RateLimiter(1000.0, 4)
    limiter.drain()
    start = time.monotonic()
    limiter.acquire()
    assert time.monotonic() - start >= 0.0005       # had to wait for a fresh token


def test_arxiv_session_has_an_inert_cooldown_by_default():
    session = C.ArxivSession(_Cfg())
    assert session.cooldown.enabled is False
