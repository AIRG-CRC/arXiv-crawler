"""Download layer and the parallel pipeline orchestrator.

Two pools, because the two halves of the work have opposite shapes:

  * downloading is IO-bound and rate-capped  -> ThreadPoolExecutor
  * PDF parsing is CPU-bound in a C extension -> ProcessPoolExecutor

They are joined by a bounded queue so staged PDFs cannot pile up on disk if conversion
falls behind. All manifest writes go to a single `ManifestWriter` thread.

Rate limiting is global across every worker: raising `--download-workers` increases
concurrency, never the request rate past `--rps`. arXiv asks that automated clients stay
around bursts of 4 req/s, and use export.arxiv.org rather than arxiv.org.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import signal
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from .checkpoint import CheckpointStore
from .converter import convert_and_write
from .logging_setup import console, quiet_worker_logging
from .memory import MemoryGuard, human, resolve_floor
from .paths import staged_pdf_path
from .state import (
    DONE,
    FAILED_CONVERT,
    FAILED_DOWNLOAD,
    IN_FLIGHT,
    NO_PDF,
    PENDING,
    RETRYABLE,
    Manifest,
    ManifestWriter,
    PaperRow,
    TaskResult,
)

log = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF-"
RETRY_STATUS = {429, 500, 502, 503, 504}
__version__ = "0.1.0"

# Every key a caller may format. Returned even by the early "nothing to do" exit, so a
# summary line never has to guard against a missing counter.
EMPTY_TALLIES = {"processed": 0, "done": 0, "no_pdf": 0, "failed": 0, "retried": 0}


class RateLimiter:
    """Thread-safe token bucket shared by every download worker."""

    def __init__(self, rate_per_sec: float, burst: int):
        self.rate = max(rate_per_sec, 0.01)
        self.capacity = max(float(burst), 1.0)
        self._tokens = self.capacity
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self.capacity, self._tokens + (now - self._updated) * self.rate
                )
                self._updated = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait_for = (1.0 - self._tokens) / self.rate
            time.sleep(min(wait_for, 1.0))


class ArxivSession:
    """A `requests` session per thread, with arXiv-appropriate headers."""

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self._local = threading.local()

    @property
    def session(self) -> requests.Session:
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            s.headers.update({
                "User-Agent": f"arxiv-crawler/{__version__} (+{self.cfg.contact})",
                "Accept": "application/pdf",
            })
            self._local.session = s
        return s

    def pdf_url(self, row: PaperRow) -> str:
        return f"{self.cfg.base_url.rstrip('/')}/pdf/{row.arxiv_id}{row.version}"


class DownloadOutcome:
    __slots__ = ("path", "size", "sha256", "status", "error")

    def __init__(self, *, path=None, size=None, sha256=None, status=DONE, error=None):
        self.path, self.size, self.sha256 = path, size, sha256
        self.status, self.error = status, error


def _sleep_for_retry(response: requests.Response | None, attempt: int, stop: threading.Event) -> None:
    """Exponential backoff with jitter, but honour an explicit Retry-After."""
    delay = min(60.0, 2.0**attempt) + random.uniform(0, 1.0)
    if response is not None:
        header = response.headers.get("Retry-After")
        if header:
            try:
                delay = max(delay, float(header))
            except ValueError:
                pass
    stop.wait(delay)


def download_one(
    row: PaperRow,
    session: ArxivSession,
    limiter: RateLimiter,
    data_dir: Path,
    stop: threading.Event,
) -> DownloadOutcome:
    """Fetch one PDF into `data/tmp/`. Never raises; failures come back as a status."""
    cfg = session.cfg
    target = staged_pdf_path(data_dir, row.arxiv_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(".pdf.part")
    url = session.pdf_url(row)
    last_error = "unknown error"

    for attempt in range(cfg.max_attempts):
        if stop.is_set():
            # The user interrupted; this paper did not fail. Hand it straight back to
            # `pending` without burning an attempt, so a plain `run` picks it up again.
            return DownloadOutcome(status=PENDING)
        limiter.acquire()
        response = None
        try:
            response = session.session.get(url, timeout=cfg.timeout, stream=True)

            if response.status_code == 404:
                return DownloadOutcome(status=NO_PDF, error="404 (withdrawn or no PDF)")
            if response.status_code in RETRY_STATUS:
                last_error = f"HTTP {response.status_code}"
                _sleep_for_retry(response, attempt, stop)
                continue
            response.raise_for_status()

            digest = hashlib.sha256()
            size = 0
            with part.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=cfg.chunk_size):
                    if not chunk:
                        continue
                    if size == 0 and not chunk.startswith(PDF_MAGIC):
                        # arXiv answers 200 with an HTML "PDF is being generated"
                        # interstitial, so the status code alone proves nothing.
                        raise ValueError("response body is not a PDF")
                    fh.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)

            if size == 0:
                raise ValueError("empty response body")

            os.replace(part, target)   # atomic: never leaves a truncated-looking PDF
            return DownloadOutcome(path=target, size=size, sha256=digest.hexdigest())

        except (requests.RequestException, ValueError, OSError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            part.unlink(missing_ok=True)
            _sleep_for_retry(response, attempt, stop)
        finally:
            if response is not None:
                response.close()

    return DownloadOutcome(status=FAILED_DOWNLOAD, error=last_error[:500])


class WorkerBars:
    """One `tqdm` line per conversion worker, under the main bar.

    Deliberately *not* a slice of the work: the pool hands each worker one PDF at a time
    and that is what keeps every worker busy while downloads trickle in at the rate cap.
    Pre-assigning a fixed batch per worker would idle whichever workers drew the easy
    papers, and force enough PDFs onto disk to defeat the bounded staging queue.

    These bars have no total for the same reason -- there is no per-worker denominator to
    count towards, only what a worker has actually finished, which is the number worth
    watching. `enabled=False` skips construction entirely, so the default costs nothing.
    """

    def __init__(self, n: int, *, enabled: bool = True):
        self.enabled = enabled and n > 0
        self._slots: dict[int, int] = {}          # pid -> bar index
        self._bars: list[tqdm] = []
        if not self.enabled:
            return
        # `{desc}` rather than `{postfix}`: tqdm prefixes a postfix with ", " when it
        # renders, which reads as a stray comma in a custom format string.
        self._bars = [
            tqdm(
                position=i + 1, leave=False, unit="paper", smoothing=0.1,
                bar_format="  worker {desc} {n_fmt} papers [{elapsed}, {rate_fmt}]",
                desc=f"{i:>2} · idle",
            )
            for i in range(n)
        ]

    def bump(self, pid: int, arxiv_id: str) -> None:
        if not self.enabled:
            return
        slot = self._slots.get(pid)
        if slot is None:
            if len(self._slots) >= len(self._bars):
                return                            # pool replaced a crashed worker
            slot = self._slots[pid] = len(self._slots)
        bar = self._bars[slot]
        bar.set_description_str(f"{slot:>2} · {arxiv_id}", refresh=False)
        bar.update(1)

    def close(self) -> None:
        for bar in self._bars:
            bar.close()


def run_pipeline(
    cfg: Any,
    *,
    limit: int | None = None,
    keep_pdf: bool = False,
    worker_bars: bool = False,
) -> dict[str, int]:
    """Drive download -> convert -> manifest until the queue drains or Ctrl-C arrives.

    A run opens with the papers an earlier run failed on (`retry.on_start`), then moves
    on to fresh `pending` work. Retries come first because they are the smaller, more
    informative set: if the converter was broken last run you find out in the first few
    seconds rather than after another hour of new downloads.
    """
    cfg.paths.ensure()
    data_dir = cfg.paths.data_dir

    manifest = Manifest(cfg.paths.manifest_db)
    reclaimed = manifest.reset_stale()
    if reclaimed:
        log.info("re-queued %d row(s) left in_flight by a previous run", reclaimed)

    # The retry worklist is fixed here, before anything is dispatched -- see
    # Manifest.claimable_ids for why it is a snapshot and not a repeated query.
    retry_cfg = getattr(cfg, "retry", None)
    max_attempts = getattr(retry_cfg, "max_attempts", 4)
    in_run_enabled = bool(getattr(retry_cfg, "in_run", True))
    retry_ids: list[str] = []
    if getattr(retry_cfg, "on_start", True):
        retry_ids = manifest.claimable_ids(RETRYABLE, max_attempts=max_attempts)

    stats = manifest.stats()
    remaining = stats.get(PENDING, 0) + len(retry_ids)
    todo = min(remaining, limit) if limit else remaining
    if not todo:
        stuck = manifest.count_claimable(RETRYABLE) - len(retry_ids)
        log.info("nothing pending; run `prepare` first or use `retry`")
        console("nothing to do — the manifest is empty or complete."
                + (f" {stuck:,} paper(s) are out of retry attempts; "
                   f"`retry --max-attempts N` to force them." if stuck > 0 else ""))
        manifest.close()
        return dict(EMPTY_TALLIES)

    if retry_ids:
        console(f"retrying {len(retry_ids):,} previously failed paper(s) first")
        log.info("retry-first pass: %d paper(s) below the %d-attempt ceiling",
                 len(retry_ids), max_attempts)

    stop = threading.Event()
    previous_sigint = signal.getsignal(signal.SIGINT)

    def _on_sigint(_signum: int, _frame: Any) -> None:
        if stop.is_set():                       # second Ctrl-C: give up immediately
            signal.signal(signal.SIGINT, previous_sigint)
            raise KeyboardInterrupt
        log.warning("interrupt received - finishing in-flight work, press again to abort")
        stop.set()

    signal.signal(signal.SIGINT, _on_sigint)

    session = ArxivSession(cfg.crawl)
    limiter = RateLimiter(cfg.crawl.rate_per_sec, cfg.crawl.burst)
    writer = ManifestWriter(cfg.paths.manifest_db)
    writer.start()

    # Checkpoints: one file per worker, incremented once per completed paper. A resumed
    # run continues the progress bar from where the interrupted one stopped instead of
    # restarting at zero.
    checkpoints = CheckpointStore(cfg.paths.checkpoints_dir)
    carried = checkpoints.begin(target=todo, settings={
        "download_workers": cfg.crawl.workers,
        "convert_workers": cfg.convert.workers,
        "converter": cfg.convert.converter,
        "rate_per_sec": cfg.crawl.rate_per_sec,
    })
    if carried.processed:
        log.info("resuming: %s paper(s) already processed by %d worker(s) in the "
                 "interrupted run", f"{carried.processed:,}", carried.workers)

    tallies = dict(EMPTY_TALLIES)
    max_inflight = max(4 * cfg.convert.workers, 2 * cfg.crawl.workers)
    bar = tqdm(
        total=todo + carried.processed, initial=carried.processed,
        unit="paper", desc="crawl+convert", smoothing=0.05, position=0,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )
    bar.set_postfix(ok=carried.done, fail=carried.failed, refresh=False)
    bars = WorkerBars(cfg.convert.workers, enabled=worker_bars)

    # Papers that failed and still have tries left, waiting to go round again. They stay
    # `in_flight` in the manifest throughout, because this run has not let go of them.
    requeue: deque[PaperRow] = deque()
    in_run_used: dict[str, int] = {}
    in_run_budget = getattr(retry_cfg, "in_run_attempts", 1) if in_run_enabled else 0
    fresh_dispatched = 0        # newly claimed rows in the last _claim; see the loop
    live_workers: set[int] = set()   # pids that have finished a paper in *this* run

    def _try_again_this_run(row: PaperRow, result: TaskResult) -> bool:
        """Should this failure go back on the queue instead of being recorded as final?

        Two ceilings, and both have to hold. The per-run budget stops one pathological
        paper from monopolising a worker, and `max_attempts` -- counted against the
        attempts the row already carried when it was claimed -- is the lifetime cap, so
        a paper cannot be retried here *and* again on every future run without end.
        """
        if not in_run_budget or stop.is_set():
            return False
        if result.status == NO_PDF:        # 404 is a fact, not a failure
            return False
        used = in_run_used.get(row.arxiv_id, 0)
        if used >= in_run_budget:
            return False
        if row.attempts + used + 1 >= max_attempts:
            return False
        in_run_used[row.arxiv_id] = used + 1
        return True

    def _record(result: TaskResult, row: PaperRow | None = None) -> None:
        if result.status not in (DONE, NO_PDF) and row is not None \
                and _try_again_this_run(row, result):
            # Record the attempt and its error, but leave the row `in_flight` and hand it
            # straight back to the dispatcher. Deliberately not counted towards the bar
            # or the checkpoints: this is the same paper, not a new one.
            writer.submit(TaskResult(
                arxiv_id=result.arxiv_id, status=IN_FLIGHT,
                error=result.error, count_attempt=True,
            ))
            tallies["in_run_retries"] = tallies.get("in_run_retries", 0) + 1
            requeue.append(row)
            console("  ↻ %s failed (%s) — retrying now",
                    result.arxiv_id, (result.error or result.status)[:90])
            return

        writer.submit(result)
        tallies["processed"] += 1
        if result.status == DONE:
            tallies["done"] += 1
            if result.converter and result.converter != cfg.convert.converter:
                # Worth a line: the paper is saved, but by a plainer backend than the one
                # configured, and the difference shows up in the output.
                tallies["fell_back"] = tallies.get("fell_back", 0) + 1
                console("  ↳ %s converted by fallback '%s'",
                        result.arxiv_id, result.converter)
        elif result.status == NO_PDF:
            tallies["no_pdf"] += 1
        else:
            tallies["failed"] += 1
            # The one thing that still reaches the terminal: what broke, and on which
            # paper. Written through tqdm so it scrolls above the bar instead of
            # shredding it.
            console("  ✗ %s  %s", result.arxiv_id, result.error or result.status)
        checkpoints.bump(result.worker_id or os.getpid(), result.status, result.arxiv_id)
        if result.worker_id:
            bars.bump(result.worker_id, result.arxiv_id)
            live_workers.add(result.worker_id)
        totals = checkpoints.totals()
        # `w` is workers seen working *in this run* over the number configured. It used to
        # be checkpoints.totals().workers, which counts accumulated worker-NN.json files
        # -- those carry over from every interrupted run and are only cleared on a clean
        # finish, so it read "w=17" on a two-worker run. That was measuring history, not
        # concurrency.
        bar.set_postfix(ok=totals.done, fail=totals.failed,
                        w=f"{len(live_workers)}/{cfg.convert.workers}", refresh=False)
        bar.update(1)

    def _download(row: PaperRow) -> tuple[PaperRow, DownloadOutcome]:
        return row, download_one(row, session, limiter, data_dir, stop)

    def _claim(want: int) -> list[PaperRow]:
        """This run's own failures first, then the retry worklist, then fresh `pending`.

        Requeued rows are already claimed and already in memory, so they need no trip to
        the manifest. Ids from the cross-run worklist are consumed off the snapshot as
        they are handed out, so that pass walks the list exactly once.
        """
        nonlocal fresh_dispatched
        rows: list[PaperRow] = []
        while requeue and len(rows) < want:
            rows.append(requeue.popleft())
        # Requeues are the same papers coming round again, so they do not spend the
        # `--limit` budget; only newly claimed rows count as progress through the queue.
        want -= len(rows)
        fresh_dispatched = 0
        if want > 0 and retry_ids:
            head = retry_ids[:want]
            del retry_ids[:want]
            claimed = manifest.claim_ids(head)
            tallies["retried"] += len(claimed)
            rows += claimed
            want -= len(claimed)
            fresh_dispatched += len(claimed)
        if want > 0:
            batch = manifest.claim_batch(want)
            rows += batch
            fresh_dispatched += len(batch)
        return rows

    # Back-pressure, so that running out of memory slows the crawl down instead of
    # handing the kernel's OOM killer a choice of victims. It does not shrink a worker;
    # it stops adding papers until the ones in flight have handed their memory back.
    guard = MemoryGuard(
        resolve_floor(getattr(cfg.convert, "memory_floor", None)),
        on_pause=lambda free, floor: console(
            "  ⏸ pausing new work: %s RAM free, floor is %s "
            "(lower convert.workers if this is constant)",
            human(free), human(floor),
        ),
    )

    dl_pool = ThreadPoolExecutor(cfg.crawl.workers, thread_name_prefix="dl")
    # `max_tasks_per_child` retires a worker after N papers and starts a fresh one, so
    # whatever the per-paper heap release cannot reclaim cannot accumulate for a whole
    # run either. Only supported from Python 3.11.
    pool_kwargs: dict[str, Any] = {"initializer": quiet_worker_logging}
    recycle_after = getattr(cfg.convert, "max_tasks_per_child", None)
    if recycle_after:
        try:
            cv_pool = ProcessPoolExecutor(
                cfg.convert.workers, max_tasks_per_child=int(recycle_after), **pool_kwargs
            )
        except (TypeError, ValueError) as exc:       # older Python, or an odd mp context
            log.warning("max_tasks_per_child unavailable (%s); workers will not recycle", exc)
            cv_pool = ProcessPoolExecutor(cfg.convert.workers, **pool_kwargs)
    else:
        cv_pool = ProcessPoolExecutor(cfg.convert.workers, **pool_kwargs)
    downloads: set[Future] = set()
    conversions: dict[Future, PaperRow] = {}
    dispatched = 0

    # A conversion is supposed to be capped by convert.timeout inside the worker. This is
    # the check that the cap is actually holding: a worker that blows well past it is
    # wedged in something a Python-level signal cannot interrupt, and the only useful
    # thing the orchestrator can do is say so rather than let the bar sit still.
    stuck_deadline = max(3.0 * cfg.convert.timeout, 60.0)
    started_at: dict[Future, float] = {}
    warned: set[Future] = set()

    def _check_for_wedged_conversions() -> None:
        now = time.monotonic()
        for fut, row in conversions.items():
            if fut in warned or now - started_at.get(fut, now) < stuck_deadline:
                continue
            warned.add(fut)
            waiting = int(now - started_at[fut])
            console("  ⏳ %s still converting after %ds (timeout is %ds) — worker may be "
                    "wedged; the paper will be recorded as failed if it never returns",
                    row.arxiv_id, waiting, cfg.convert.timeout)
            log.warning("conversion of %s exceeded %ds", row.arxiv_id, waiting)

    try:
        while True:
            # Top up the download pool, bounded so tmp/ cannot fill without limit — and
            # not at all while memory is short.
            held_back = not stop.is_set() and not guard.has_headroom()
            if not stop.is_set() and not held_back:
                room = max_inflight - len(downloads) - len(conversions)
                budget = (todo - dispatched) if limit else room
                want = max(0, min(room, budget))
                if want:
                    for row in _claim(want):
                        downloads.add(dl_pool.submit(_download, row))
                    dispatched += fresh_dispatched

            if not downloads and not conversions:
                # An empty pool normally means the queue drained and the run is over. It
                # can also mean the memory guard refused to dispatch and the last paper
                # in flight just finished -- which is not "done", it is "waiting". Ending
                # the run there would silently stop short of the target. Keyed off
                # whether dispatch was actually skipped, not off a second reading of the
                # guard: memory recovering between the two would look like "drained".
                if held_back:
                    time.sleep(1.0)
                    continue
                break

            done, _ = wait(downloads | set(conversions), timeout=1.0,
                           return_when=FIRST_COMPLETED)
            _check_for_wedged_conversions()

            for fut in done:
                if fut in downloads:
                    downloads.discard(fut)
                    row, outcome = fut.result()
                    if outcome.status == PENDING:
                        # Interrupted before it started: requeue silently, and do not
                        # let it count towards progress or the checkpoint tallies.
                        writer.submit(TaskResult(arxiv_id=row.arxiv_id, status=PENDING))
                        continue
                    if outcome.status != DONE:
                        _record(TaskResult(
                            arxiv_id=row.arxiv_id, status=outcome.status,
                            error=outcome.error, count_attempt=True,
                        ), row)
                        continue
                    submitted = cv_pool.submit(
                        convert_and_write,
                        row, outcome.path, data_dir, cfg.convert,
                        base_url=cfg.crawl.base_url,
                        pdf_bytes=outcome.size, pdf_sha256=outcome.sha256,
                        keep_pdf=keep_pdf,
                    )
                    conversions[submitted] = row
                    started_at[submitted] = time.monotonic()
                else:
                    row = conversions.pop(fut)
                    started_at.pop(fut, None)
                    warned.discard(fut)
                    try:
                        _record(fut.result(), row)
                    except Exception as exc:      # worker died (segfault, OOM, ...)
                        # Record it as a failure *against the paper*, not just in the
                        # tally. Without the attempts increment a PDF that segfaults the
                        # C extension would be re-downloaded and re-crash on every run
                        # the retry pass touches it -- forever.
                        log.error("conversion worker crashed on %s: %s", row.arxiv_id, exc)
                        _record(TaskResult(
                            arxiv_id=row.arxiv_id, status=FAILED_CONVERT,
                            error=f"worker died: {type(exc).__name__}: {exc}"[:500],
                            count_attempt=True,
                        ), row)
    finally:
        bars.close()
        bar.close()
        dl_pool.shutdown(wait=True)
        cv_pool.shutdown(wait=True)
        writer.stop()
        if stop.is_set():
            # Interrupted: leave the per-worker files in place so the next run resumes
            # the count, and `main.py checkpoint` can show where it stopped.
            log.info("checkpoints kept in %s — rerun `run` to resume",
                     cfg.paths.checkpoints_dir)
        else:
            checkpoints.finalize(tallies)
        # Anything still in_flight was interrupted; hand it back for the next run.
        manifest.reset_stale()
        manifest.close()
        signal.signal(signal.SIGINT, previous_sigint)
        if not keep_pdf:
            for leftover in cfg.paths.tmp_dir.glob("*.pdf*"):
                leftover.unlink(missing_ok=True)

    return tallies
