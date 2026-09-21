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
from dataclasses import asdict
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from .checkpoint import CheckpointStore
from .converter import convert_and_write
from .cooldown import Cooldown, looks_like_block_page
from .logging_setup import console, quiet_worker_logging
from .memory import MemoryGuard, human, resolve_floor
from .partition import describe as describe_partition
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

    def drain(self) -> None:
        """Empty the bucket, so the next request waits a full 1/rate.

        Called on resuming from a cooldown. The bucket refills while nothing is being
        downloaded, so after an hour's pause it is at `capacity` -- and the first thing a
        just-lifted block would see is a burst of `burst` requests, which is how the block
        was earned in the first place.
        """
        with self._lock:
            self._tokens = 0.0
            self._updated = time.monotonic()


class ArxivSession:
    """A `requests` session per thread, with arXiv-appropriate headers."""

    def __init__(self, cfg: Any, cooldown: Cooldown | None = None):
        self.cfg = cfg
        # The cooldown rides here rather than as a parameter to `download_one`: this is
        # already the per-run "how we talk to arXiv" object, and the fetch signature stays
        # as it was. A disabled default keeps `ArxivSession(cfg)` usable on its own.
        self.cooldown = cooldown or Cooldown(stop=threading.Event(), seconds=0)
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


def _retry_after_seconds(response: requests.Response | None) -> float | None:
    """The `Retry-After` header as seconds, when the server sent a usable one."""
    if response is None:
        return None
    header = response.headers.get("Retry-After")
    if not header:
        return None
    try:
        return float(header)
    except ValueError:
        return None                         # the HTTP-date form; not worth parsing


def _sleep_for_retry(response: requests.Response | None, attempt: int, stop: threading.Event) -> None:
    """Exponential backoff with jitter, but honour an explicit Retry-After."""
    delay = min(60.0, 2.0**attempt) + random.uniform(0, 1.0)
    explicit = _retry_after_seconds(response)
    if explicit is not None:
        delay = max(delay, explicit)
    stop.wait(delay)


def download_one(
    row: PaperRow,
    session: ArxivSession,
    limiter: RateLimiter,
    data_dir: Path,
    stop: threading.Event,
) -> DownloadOutcome:
    """Fetch one PDF into `data/tmp/`. Never raises; failures come back as a status.

    `attempt` is counted explicitly rather than by a `for` loop because a throttle must not
    spend one: waiting out a block the server is applying to every request is not an attempt
    at this paper. `throttle_waits` bounds that separately, so one pathological paper cannot
    park a download thread indefinitely.
    """
    cfg = session.cfg
    cooldown = session.cooldown
    target = staged_pdf_path(data_dir, row.arxiv_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(".pdf.part")
    url = session.pdf_url(row)
    last_error = "unknown error"
    attempt = 0
    throttle_waits = 0

    while attempt < cfg.max_attempts:
        if stop.is_set():
            # The user interrupted; this paper did not fail. Hand it straight back to
            # `pending` without burning an attempt, so a plain `run` picks it up again.
            # Checked *before* the cooldown gate on purpose: on Ctrl-C every queued download
            # returns here in microseconds instead of blocking on a gate that, mid-cooldown,
            # nobody is going to reopen in time for `dl_pool.shutdown(wait=True)`.
            return DownloadOutcome(status=PENDING)
        generation = cooldown.enter()
        if generation is None:
            return DownloadOutcome(status=PENDING)
        limiter.acquire()
        response = None
        throttled: int | None = None
        retry_after: float | None = None
        try:
            response = session.session.get(url, timeout=cfg.timeout, stream=True)

            if response.status_code in cooldown.statuses:
                # First, and before raise_for_status: 403/406 are 4xx, so the generic
                # handler below would otherwise spend an attempt -- and all five of them
                # within a few seconds -- on a status that says nothing about this paper.
                # The wait itself happens after the `try`, once the response is closed.
                throttled = response.status_code
                retry_after = _retry_after_seconds(response)
            elif response.status_code == 404:
                cooldown.note_clean()
                return DownloadOutcome(status=NO_PDF, error="404 (withdrawn or no PDF)")
            elif response.status_code in RETRY_STATUS:
                last_error = f"HTTP {response.status_code}"
                _sleep_for_retry(response, attempt, stop)
                attempt += 1
                continue
            else:
                response.raise_for_status()

                digest = hashlib.sha256()
                size = 0
                blocked = False
                with part.open("wb") as fh:
                    for chunk in response.iter_content(chunk_size=cfg.chunk_size):
                        if not chunk:
                            continue
                        if size == 0 and not chunk.startswith(PDF_MAGIC):
                            # arXiv answers 200 with HTML for two unrelated reasons: a "PDF
                            # is being generated" interstitial, which is per-paper and
                            # transient, and a block page, which is neither. The status code
                            # proves nothing either way -- only the body separates them.
                            log.warning("non-PDF body for %s: %r", row.arxiv_id, chunk[:200])
                            if cooldown.enabled and looks_like_block_page(chunk):
                                throttled = response.status_code
                                blocked = True
                                break
                            raise ValueError("response body is not a PDF")
                        fh.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)

                if blocked:
                    part.unlink(missing_ok=True)
                elif size == 0:
                    raise ValueError("empty response body")
                else:
                    os.replace(part, target)   # atomic: never a truncated-looking PDF
                    cooldown.note_clean()
                    return DownloadOutcome(path=target, size=size, sha256=digest.hexdigest())

        except (requests.RequestException, ValueError, OSError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            part.unlink(missing_ok=True)
            _sleep_for_retry(response, attempt, stop)
            attempt += 1
            continue
        finally:
            if response is not None:
                response.close()

        # Only a throttle reaches here, and only with the response already closed by the
        # `finally` above: a streamed connection and its pooled socket must not be held open
        # across a wait measured in hours.
        last_error = f"HTTP {throttled} (arXiv throttle)"
        if throttle_waits >= cooldown.max_rounds or not cooldown.trip(
                generation, throttled, row.arxiv_id, retry_after):
            # Either this one paper has waited out its share of rounds, or the run is over.
            # `PENDING` hands it back untouched -- no attempt, no failure recorded.
            return DownloadOutcome(status=PENDING)
        throttle_waits += 1

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

    @property
    def next_position(self) -> int:
        """The first free line under the main bar and any worker bars.

        Read from the bars actually constructed rather than recomputed from the worker
        count, so a caller cannot drift from what `enabled` decided.
        """
        return 1 + len(self._bars)

    def close(self) -> None:
        for bar in self._bars:
            bar.close()


def run_pipeline(
    cfg: Any,
    *,
    limit: int | None = None,
    keep_pdf: bool = False,
    worker_bars: bool = False,
    retry_all: bool = False,
    to_minio: bool = False,
    partition: tuple[int, int] | None = None,
    sync_bucket: bool | None = None,
    claim_any: bool = False,
) -> dict[str, int]:
    """Drive download -> convert -> manifest until the queue drains or Ctrl-C arrives.

    A run opens with the papers an earlier run failed on (`retry.on_start`), then moves
    on to fresh `pending` work. Retries come first because they are the smaller, more
    informative set: if the converter was broken last run you find out in the first few
    seconds rather than after another hour of new downloads.

    `partition` is this device's `(devices, index)` slice, or None for the whole corpus.
    Every claim is filtered by it, so two devices running at once never hand the same paper
    to two converters -- see utils.partition for why the key is hashed from the id.
    """
    cfg.paths.ensure()
    data_dir = cfg.paths.data_dir

    # Resolved here, in the parent, so a misconfiguration fails before a single paper is
    # downloaded rather than once per worker. Passed to workers as a plain dict because
    # spawn has to pickle it, and a live client cannot cross a process boundary.
    minio_settings: dict[str, Any] | None = None
    store = None
    if to_minio:
        from .objectstore import MinioSettings, MinioStore

        settings = MinioSettings.from_config(cfg.minio)
        settings.validate()
        store = MinioStore(settings)
        store.ensure_bucket()                  # also proves the endpoint is reachable
        console("storing to %s", store.describe())
        minio_settings = asdict(settings)

    manifest = Manifest(cfg.paths.manifest_db)
    reclaimed = manifest.reset_stale()
    if reclaimed:
        log.info("re-queued %d row(s) left in_flight by a previous run", reclaimed)

    # Learn what the other device has already converted, before anything is claimed. This
    # has to come before the retry snapshot and before `stats()`: a paper that failed here
    # but converted elsewhere must not be re-downloaded, and `todo` -- including the
    # "nothing to do" exit -- has to be computed after the reconciliation, or a finished
    # second device would re-crawl the corpus. The ManifestWriter has not started yet, so
    # this write is uncontended.
    wants_sync = (getattr(getattr(cfg, "sync", None), "on_start", True)
                  if sync_bucket is None else sync_bucket)
    if wants_sync and store is not None:
        from .sync import run_sync

        try:
            report = run_sync(cfg, manifest, store, partition=partition)
            for line in report.lines():
                console("  %s", line)
        except Exception as exc:               # noqa: BLE001
            # A bucket that cannot be read means crawling from the local manifest alone --
            # some duplicated work at worst. It must never stop the run.
            log.warning("bucket sync skipped", exc_info=exc)
            console("  ⚠ bucket sync skipped (%s: %s) — using the local manifest only",
                    type(exc).__name__, exc)

    # The retry worklist is fixed here, before anything is dispatched -- see
    # Manifest.claimable_ids for why it is a snapshot and not a repeated query.
    retry_cfg = getattr(cfg, "retry", None)
    # `None` means no ceiling: every failed paper is retried on every run. That is what
    # `--retry-all` and `retry.max_attempts: null` both select. Worth having because on
    # this project every "permanent" failure so far turned out to be a fixable bug, and
    # a lifetime cap would have locked those papers out of the run that fixed them.
    max_attempts = None if retry_all else getattr(retry_cfg, "max_attempts", 4)
    in_run_enabled = bool(getattr(retry_cfg, "in_run", True))
    retry_ids: list[str] = []
    if getattr(retry_cfg, "on_start", True):
        retry_ids = manifest.claimable_ids(
            RETRYABLE, max_attempts=max_attempts, partition=partition)

    stats = manifest.stats()
    # `stats` stays whole-corpus, because that is what the bar measures. What is left *for
    # this device*, though, is the partitioned count -- using the corpus figure would set
    # the run's target (and the Pending= readout) too high by a factor of `devices`.
    pending_here = manifest.count_claimable((PENDING,), partition=partition)
    if partition:
        console("device %s — %s paper(s) pending in this slice",
                describe_partition(partition), f"{pending_here:,}")
    remaining = pending_here + len(retry_ids)
    todo = min(remaining, limit) if limit else remaining
    if not todo:
        stuck = manifest.count_claimable(RETRYABLE, partition=partition) - len(retry_ids)
        log.info("nothing pending; run `prepare` first or use `retry`")
        console("nothing to do — the manifest is empty or complete."
                + (f" {stuck:,} paper(s) are out of retry attempts; "
                   f"`retry --max-attempts N` to force them." if stuck > 0 else ""))
        manifest.close()
        return dict(EMPTY_TALLIES)

    # Always say what the retry pass decided, even when the answer is "nothing". A silent
    # start is indistinguishable from a broken one, and a paper held back by the attempt
    # ceiling was previously invisible: the explanation only appeared in the "nothing to
    # do" branch, which never fires while millions of papers are still pending.
    total_failed = manifest.count_claimable(RETRYABLE, partition=partition)
    if not getattr(retry_cfg, "on_start", True):
        if total_failed:
            console("retry: %s failed paper(s) left alone (retry.on_start is off)",
                    f"{total_failed:,}")
    elif retry_ids:
        console("retry: %s previously failed paper(s) queued first",
                f"{len(retry_ids):,}")
        log.info("retry-first pass: %d of %d failed paper(s), ceiling %s",
                 len(retry_ids), total_failed, max_attempts)
    else:
        console("retry: no failed papers to retry")

    stuck = total_failed - len(retry_ids)
    if stuck > 0:
        console("retry: %s failed paper(s) skipped — out of attempts (retry.max_attempts"
                "=%s); `run --retry-all` to try them anyway", f"{stuck:,}", max_attempts)
        log.info("%d paper(s) above the attempt ceiling", stuck)

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

    # Checkpoints record per-worker detail for the `checkpoint` command. They are NOT the
    # bar's source of numbers: their counters only cover runs that were interrupted --
    # `finalize` deletes the files on a clean finish -- so they drift from reality without
    # bound. Measured on this manifest: the bar opened at 108,492 against 221,056 papers
    # actually converted, under-reporting by more than half the corpus.
    checkpoints = CheckpointStore(cfg.paths.checkpoints_dir)
    checkpoints.begin(target=todo, settings={
        "download_workers": cfg.crawl.workers,
        "convert_workers": cfg.convert.workers,
        "converter": cfg.convert.converter,
        "rate_per_sec": cfg.crawl.rate_per_sec,
    })
    # Everything the bar reports comes from the manifest, so it can never disagree with
    # `status`. The bar measures the corpus: converted papers out of papers in scope.
    total_papers = stats.get("total", 0)
    done_at_start = stats.get(DONE, 0)
    pending_at_start = pending_here
    claimed_from_pending = 0        # drives the Pending= readout as rows are claimed

    tallies = dict(EMPTY_TALLIES)
    max_inflight = max(4 * cfg.convert.workers, 2 * cfg.crawl.workers)
    bar = tqdm(
        total=total_papers, initial=done_at_start,
        unit="paper", desc="crawl+convert", smoothing=0.05, position=0,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )

    def _refresh_postfix() -> None:
        bar.set_postfix_str(
            f"Fail={tallies['failed']:,}, Done={tallies['done']:,}, "
            f"Workers={_live_workers()}/{cfg.convert.workers}, "
            f"Pending={pending_at_start - claimed_from_pending:,}",
            refresh=False,
        )
    bars = WorkerBars(cfg.convert.workers, enabled=worker_bars)

    # A throttle is a fact about the host, not about a paper, so it pauses every download at
    # once. Built after the bars because the countdown needs the first free line under them,
    # and resumed through a hook: the token bucket has to be emptied before the gate reopens
    # (an idle bucket refills to `burst`, and a burst is how a block is earned), and the main
    # bar's clock re-based (nothing advances it during the pause, so the next paper would
    # otherwise be charged the whole wait and the ETA would read in weeks).
    def _after_cooldown() -> None:
        limiter.drain()
        bar.unpause()

    cooldown = Cooldown(
        stop=stop,
        seconds=getattr(cfg.crawl, "cooldown_seconds", 3600),
        max_seconds=getattr(cfg.crawl, "cooldown_max_seconds", 21600),
        statuses=getattr(cfg.crawl, "cooldown_statuses", None) or (),
        escalate=getattr(cfg.crawl, "cooldown_escalate", True),
        max_rounds=getattr(cfg.crawl, "cooldown_max_rounds", 4),
        position=bars.next_position,
        on_resume=_after_cooldown,
    )
    session.cooldown = cooldown

    # Papers that failed and still have tries left, waiting to go round again. They stay
    # `in_flight` in the manifest throughout, because this run has not let go of them.
    requeue: deque[PaperRow] = deque()
    in_run_used: dict[str, int] = {}
    in_run_budget = getattr(retry_cfg, "in_run_attempts", 1) if in_run_enabled else 0
    fresh_dispatched = 0        # newly claimed rows in the last _claim; see the loop

    def _live_workers() -> int:
        """How many conversion processes the pool currently has. Best-effort: the
        attribute is private, so an unexpected shape falls back to the configured count
        rather than putting a wrong number on the bar."""
        processes = getattr(cv_pool, "_processes", None)
        return len(processes) if processes else cfg.convert.workers

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
        # `max_attempts is None` means no lifetime ceiling, so only the per-run budget
        # above applies -- which still terminates, because it is finite.
        if max_attempts is not None and row.attempts + used + 1 >= max_attempts:
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
                    result.arxiv_id, (result.error or result.status)[:160])
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
        # The bar counts *converted* papers, so only a success advances it. A failed or
        # withdrawn paper still has no markdown on disk, and moving the bar for it would
        # claim otherwise. `update(0)` keeps tqdm's clock and rate current regardless.
        #
        # Fail/Done describe *this run*; they used to come from checkpoints.totals(),
        # which sums leftover files from previous interrupted runs -- so the bar could
        # report `fail=10` with no failures in the manifest and no `✗` ever printed.
        _refresh_postfix()
        bar.update(1 if result.status == DONE else 0)

    def _download(row: PaperRow) -> tuple[PaperRow, DownloadOutcome]:
        return row, download_one(row, session, limiter, data_dir, stop)

    def _claim(want: int) -> list[PaperRow]:
        """This run's own failures first, then the retry worklist, then fresh `pending`.

        Requeued rows are already claimed and already in memory, so they need no trip to
        the manifest. Ids from the cross-run worklist are consumed off the snapshot as
        they are handed out, so that pass walks the list exactly once.
        """
        nonlocal fresh_dispatched, claimed_from_pending
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
            batch = manifest.claim_batch(want, partition=partition)
            rows += batch
            fresh_dispatched += len(batch)
            # Only these leave the `pending` pool; retries come from the failed statuses.
            claimed_from_pending += len(batch)
        if claim_any and partition and not rows and want > 0 and not stop.is_set():
            stolen = _steal(want)
            rows += stolen
            fresh_dispatched += len(stolen)
        return rows

    # How many rows `_steal` will look at before giving the dispatch loop its turn back.
    # Without a bound, a tail where every remaining paper is already in the bucket would
    # sit in one `_claim` call doing a HEAD per paper for the rest of the queue.
    STEAL_INSPECT_FACTOR = 10

    def _steal(want: int) -> list[PaperRow]:
        """Claim outside this device's slice, once the slice itself has drained.

        Each candidate is checked against the bucket first: the other device may well have
        converted it already, and re-downloading a paper that is stored is precisely the
        waste the partition exists to prevent. One HEAD per paper is affordable here and
        nowhere else, because this only ever runs at the tail of a run.

        Two devices stealing at once can still race onto the same paper. The cost is a
        duplicated download and an idempotent overwrite -- never a corrupt object.
        """
        taken: list[PaperRow] = []
        inspected = 0
        while len(taken) < want and inspected < want * STEAL_INSPECT_FACTOR:
            batch = manifest.claim_batch(want - len(taken))
            if not batch:
                break                       # nothing left anywhere; the run is finishing
            for row in batch:
                inspected += 1
                if store is not None and store.exists(store.name_for("md", row.arxiv_id)):
                    # Already converted elsewhere. Record it and move on -- no download,
                    # no conversion, and not counted as this run's own work.
                    writer.submit(TaskResult(arxiv_id=row.arxiv_id, status=DONE,
                                             remote_only=True))
                    tallies["already_elsewhere"] = tallies.get("already_elsewhere", 0) + 1
                    bar.update(1)
                    continue
                taken.append(row)
        if taken or tallies.get("already_elsewhere"):
            log.info("claim-any: took %d paper(s) from outside slice %s", len(taken),
                     partition)
        return taken

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
    # The log path is passed explicitly: under spawn a worker inherits nothing, so it
    # cannot discover where the parent is logging.
    pool_kwargs: dict[str, Any] = {
        "initializer": quiet_worker_logging,
        "initargs": (str(cfg.paths.logs_dir / "crawler.log"),),
    }
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
                        keep_pdf=keep_pdf, minio=minio_settings,
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
        # Pools first, bars second. A cooldown parks a download thread inside its own
        # `tqdm`, and closing the main bar while that bar is still live moves the cursor
        # under it -- the countdown then redraws in the wrong place and blanks the wrong
        # line on close. Shutting the pools down first means every bar below position 0
        # has already closed itself by the time the main bar does.
        dl_pool.shutdown(wait=True)
        cv_pool.shutdown(wait=True)
        bars.close()
        bar.close()
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

    if cooldown.rounds_served:
        tallies["cooldowns"] = cooldown.rounds_served
    if cooldown.gave_up:
        # The caller turns this into a non-zero exit, so a restart wrapper waits instead of
        # relaunching straight back into the block.
        tallies["throttled_out"] = 1

    return tallies
