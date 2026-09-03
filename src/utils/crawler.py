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
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from .checkpoint import CheckpointStore
from .converter import convert_and_write
from .paths import staged_pdf_path
from .state import (
    DONE,
    FAILED_DOWNLOAD,
    NO_PDF,
    PENDING,
    Manifest,
    ManifestWriter,
    PaperRow,
    TaskResult,
)

log = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF-"
RETRY_STATUS = {429, 500, 502, 503, 504}
__version__ = "0.1.0"


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


def run_pipeline(cfg: Any, *, limit: int | None = None, keep_pdf: bool = False) -> dict[str, int]:
    """Drive download -> convert -> manifest until the queue drains or Ctrl-C arrives."""
    cfg.paths.ensure()
    data_dir = cfg.paths.data_dir

    manifest = Manifest(cfg.paths.manifest_db)
    reclaimed = manifest.reset_stale()
    if reclaimed:
        log.info("re-queued %d row(s) left in_flight by a previous run", reclaimed)

    stats = manifest.stats()
    remaining = stats.get("pending", 0)
    todo = min(remaining, limit) if limit else remaining
    if not todo:
        log.info("nothing pending; run `prepare` first or use `retry`")
        manifest.close()
        return {"processed": 0}

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

    tallies = {"processed": 0, "done": 0, "no_pdf": 0, "failed": 0}
    max_inflight = max(4 * cfg.convert.workers, 2 * cfg.crawl.workers)
    bar = tqdm(
        total=todo + carried.processed, initial=carried.processed,
        unit="paper", desc="crawl+convert", smoothing=0.05,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )
    bar.set_postfix(ok=carried.done, fail=carried.failed, refresh=False)

    def _record(result: TaskResult) -> None:
        writer.submit(result)
        tallies["processed"] += 1
        if result.status == DONE:
            tallies["done"] += 1
        elif result.status == NO_PDF:
            tallies["no_pdf"] += 1
        else:
            tallies["failed"] += 1
        checkpoints.bump(result.worker_id or os.getpid(), result.status, result.arxiv_id)
        totals = checkpoints.totals()
        bar.set_postfix(ok=totals.done, fail=totals.failed, w=totals.workers, refresh=False)
        bar.update(1)

    def _download(row: PaperRow) -> tuple[PaperRow, DownloadOutcome]:
        return row, download_one(row, session, limiter, data_dir, stop)

    dl_pool = ThreadPoolExecutor(cfg.crawl.workers, thread_name_prefix="dl")
    cv_pool = ProcessPoolExecutor(cfg.convert.workers)
    downloads: set[Future] = set()
    conversions: set[Future] = set()
    dispatched = 0

    try:
        while True:
            # Top up the download pool, bounded so tmp/ cannot fill without limit.
            if not stop.is_set():
                room = max_inflight - len(downloads) - len(conversions)
                budget = (todo - dispatched) if limit else room
                want = max(0, min(room, budget))
                if want:
                    for row in manifest.claim_batch(want):
                        downloads.add(dl_pool.submit(_download, row))
                        dispatched += 1

            if not downloads and not conversions:
                break

            done, _ = wait(downloads | conversions, timeout=1.0, return_when=FIRST_COMPLETED)

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
                        ))
                        continue
                    conversions.add(cv_pool.submit(
                        convert_and_write,
                        row, outcome.path, data_dir, cfg.convert,
                        base_url=cfg.crawl.base_url,
                        pdf_bytes=outcome.size, pdf_sha256=outcome.sha256,
                        keep_pdf=keep_pdf,
                    ))
                else:
                    conversions.discard(fut)
                    try:
                        _record(fut.result())
                    except Exception as exc:      # worker died (segfault, OOM, ...)
                        log.error("conversion worker crashed: %s", exc)
                        tallies["failed"] += 1
                        bar.update(1)
    finally:
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
