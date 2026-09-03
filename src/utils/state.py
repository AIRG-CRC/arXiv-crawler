from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator


PENDING = "pending"
IN_FLIGHT = "in_flight"
DONE = "done"
NO_PDF = "no_pdf"
FAILED_DOWNLOAD = "failed_download"
FAILED_CONVERT = "failed_convert"

RETRYABLE = (FAILED_DOWNLOAD, FAILED_CONVERT)

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
  arxiv_id         TEXT PRIMARY KEY,
  version          TEXT,
  shard            TEXT,
  title            TEXT,
  authors          TEXT,      -- JSON array
  categories       TEXT,      -- space-separated, as arXiv publishes it
  primary_category TEXT,
  doi              TEXT,
  journal_ref      TEXT,
  license          TEXT,
  date_released    TEXT,      -- v1 submission date, ISO
  date_updated     TEXT,      -- latest version date, ISO
  abstract         TEXT,
  status           TEXT NOT NULL DEFAULT 'pending',
  pdf_bytes        INTEGER,
  pdf_sha256       TEXT,
  md_bytes         INTEGER,
  tables_bytes     INTEGER,
  n_pages          INTEGER,
  n_tables         INTEGER,
  n_chars          INTEGER,
  low_text         INTEGER NOT NULL DEFAULT 0,
  attempts         INTEGER NOT NULL DEFAULT 0,
  error            TEXT,
  completed_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_papers_status ON papers(status);
CREATE INDEX IF NOT EXISTS idx_papers_cat    ON papers(primary_category);
"""

_ROW_COLUMNS = (
    "arxiv_id", "version", "shard", "title", "authors", "categories",
    "primary_category", "doi", "journal_ref", "license",
    "date_released", "date_updated", "abstract",
)


@dataclass
class PaperRow:
    """Everything a download/convert worker needs. Must stay picklable — it crosses
    a process boundary into the conversion pool."""

    arxiv_id: str
    version: str
    shard: str
    title: str = ""
    authors: str = "[]"
    categories: str = ""
    primary_category: str = ""
    doi: str | None = None 
    date_released: str | None = None
    date_updated: str | None = None

    @property
    def author_list(self) -> list[str]:
        try:
            return json.loads(self.authors)
        
        except (TypeError, ValueError):
            return []

    @property
    def category_list(self) -> list[str]:
        return self.categories.split()


@dataclass
class TaskResult:
    """One unit of manifest mutation, produced by a worker and applied by the writer."""

    arxiv_id: str
    status: str
    error: str | None = None
    pdf_bytes: int | None = None
    pdf_sha256: str | None = None
    md_bytes: int | None = None
    tables_bytes: int | None = None
    n_pages: int | None = None
    n_tables: int | None = None
    n_chars: int | None = None
    low_text: bool = False
    count_attempt: bool = False
    worker_id: int = 0        # pid of the process that handled it; for checkpoints


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: Path, *, readonly: bool = False) -> sqlite3.Connection:
    """Open a tuned connection"""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row

    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")

    if not readonly:
        conn.executescript(SCHEMA)
    return conn


class Manifest:
    """Read/claim side of the manifest. One instance per thread that touches SQLite."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = connect(db_path)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Manifest":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # Main injest function
    def add_papers(self, rows: Iterable[PaperRow], batch_size: int = 5000) -> int:
        sql = (
            f"INSERT OR IGNORE INTO papers ({', '.join(_ROW_COLUMNS)}) "
            f"VALUES ({', '.join('?' * len(_ROW_COLUMNS))})"
        )

        inserted, batch = 0, []

        for row in rows:
            batch.append(tuple(getattr(row, c) for c in _ROW_COLUMNS))

            if len(batch) >= batch_size:
                inserted += self._flush(sql, batch)
                batch.clear()

        if batch:
            inserted += self._flush(sql, batch)

        return inserted


    def _flush(self, sql: str, batch: list[tuple]) -> int:
        with self.conn:
            cur = self.conn.executemany(sql, batch)
        return cur.rowcount

    # Worker helper function --> No duplicate task handling between workers
    def claim_batch(self, n: int, statuses: tuple[str, ...] = (PENDING,)) -> list[PaperRow]:
        """Atomically move up to `n` rows to `in_flight` and return them.

        The UPDATE...RETURNING is a single statement, so two claimers can never hand the
        same paper to two workers even if the dispatcher is ever parallelised.
        """
        placeholders = ", ".join("?" * len(statuses))
        sql = (
            f"UPDATE papers SET status = ? WHERE arxiv_id IN ("
            f"  SELECT arxiv_id FROM papers WHERE status IN ({placeholders}) LIMIT ?"
            f") RETURNING {', '.join(_ROW_COLUMNS)}"
        )
        with self.conn:
            cur = self.conn.execute(sql, (IN_FLIGHT, *statuses, n))
            return [PaperRow(**dict(r)) for r in cur.fetchall()]

    def reset_stale(self) -> int:
        """Return rows abandoned `in_flight` by a crashed run back to `pending`."""
        with self.conn:
            cur = self.conn.execute(
                "UPDATE papers SET status = ? WHERE status = ?", (PENDING, IN_FLIGHT)
            )
        return cur.rowcount

    def reset_failed(self, stage: str | None = None, max_attempts: int = 4) -> int:
        """Re-queue retryable failures that still have attempts left."""
        statuses = {
            "download": (FAILED_DOWNLOAD,),
            "convert": (FAILED_CONVERT,),
            None: RETRYABLE,
        }[stage]
        placeholders = ", ".join("?" * len(statuses))
        with self.conn:
            cur = self.conn.execute(
                f"UPDATE papers SET status = ?, error = NULL "
                f"WHERE status IN ({placeholders}) AND attempts < ?",
                (PENDING, *statuses, max_attempts),
            )
        return cur.rowcount


    # Report progress function
    def stats(self) -> dict[str, int]:
        cur = self.conn.execute("SELECT status, COUNT(*) AS n FROM papers GROUP BY status")
        counts = {r["status"]: r["n"] for r in cur}
        counts["total"] = sum(counts.values())
        counts["low_text"] = self.conn.execute(
            "SELECT COUNT(*) FROM papers WHERE low_text = 1"
        ).fetchone()[0]
        return counts

    def iter_done(self) -> Iterator[sqlite3.Row]:
        yield from self.conn.execute(
            "SELECT * FROM papers WHERE status = ? AND md_bytes IS NOT NULL", (DONE,)
        )


class ManifestWriter(threading.Thread):
    """Owns the only write connection used during a run.

    Workers push `TaskResult`s onto `.queue`; this thread applies them in batched
    transactions. Call `.stop()` to drain and shut down cleanly.
    """

    _SENTINEL = object()

    def __init__(self, db_path: Path, *, batch_size: int = 64, flush_interval: float = 2.0):
        super().__init__(name="manifest-writer", daemon=True)
        self.db_path = db_path
        self.queue: queue.Queue = queue.Queue()
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.applied = 0

    def submit(self, result: TaskResult) -> None:
        self.queue.put(result)

    def stop(self) -> None:
        self.queue.put(self._SENTINEL)
        self.join()

    def run(self) -> None:
        conn = connect(self.db_path)
        pending: list[TaskResult] = []
        last_flush = time.monotonic()
        stopping = False
        try:
            while not (stopping and not pending):
                timeout = max(0.05, self.flush_interval - (time.monotonic() - last_flush))
                try:
                    item = self.queue.get(timeout=timeout)
                    if item is self._SENTINEL:
                        stopping = True
                    else:
                        pending.append(item)
                except queue.Empty:
                    pass

                due = (
                    len(pending) >= self.batch_size
                    or (pending and time.monotonic() - last_flush >= self.flush_interval)
                    or (stopping and pending)
                )
                if due:
                    self._apply(conn, pending)
                    pending.clear()
                    last_flush = time.monotonic()
        finally:
            if pending:
                self._apply(conn, pending)
            conn.close()

    def _apply(self, conn: sqlite3.Connection, results: list[TaskResult]) -> None:
        sql = """
            UPDATE papers SET
              status       = :status,
              error        = :error,
              pdf_bytes    = COALESCE(:pdf_bytes, pdf_bytes),
              pdf_sha256   = COALESCE(:pdf_sha256, pdf_sha256),
              md_bytes     = COALESCE(:md_bytes, md_bytes),
              tables_bytes = COALESCE(:tables_bytes, tables_bytes),
              n_pages      = COALESCE(:n_pages, n_pages),
              n_tables     = COALESCE(:n_tables, n_tables),
              n_chars      = COALESCE(:n_chars, n_chars),
              low_text     = :low_text,
              attempts     = attempts + :count_attempt,
              completed_at = :completed_at
            WHERE arxiv_id = :arxiv_id
        """
        # Built explicitly rather than from asdict(), so a new TaskResult field cannot
        # leak into the statement's parameter set.
        payload = [{
            "arxiv_id": r.arxiv_id, "status": r.status, "error": r.error,
            "pdf_bytes": r.pdf_bytes, "pdf_sha256": r.pdf_sha256,
            "md_bytes": r.md_bytes, "tables_bytes": r.tables_bytes,
            "n_pages": r.n_pages, "n_tables": r.n_tables, "n_chars": r.n_chars,
            "low_text": int(r.low_text), "count_attempt": int(r.count_attempt),
            "completed_at": _utcnow() if r.status in (DONE, NO_PDF) else None,
        } for r in results]
        with conn:
            conn.executemany(sql, payload)
        self.applied += len(payload)
