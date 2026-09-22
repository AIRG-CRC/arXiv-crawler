from __future__ import annotations

import json
import logging
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from .partition import bucket_for

log = logging.getLogger(__name__)

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
  date_released    TEXT,      -- v1 submission date, ISO
  date_updated     TEXT,      -- latest version date, ISO
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
  completed_at     TEXT,
  remote_only      INTEGER NOT NULL DEFAULT 0,  -- output is in the bucket, not on disk
  bucket           INTEGER NOT NULL DEFAULT -1  -- crc32(arxiv_id) %% 256; the device slice
);
CREATE INDEX IF NOT EXISTS idx_papers_status ON papers(status);
CREATE INDEX IF NOT EXISTS idx_papers_cat    ON papers(primary_category);
CREATE INDEX IF NOT EXISTS idx_papers_status_bucket ON papers(status, bucket);
"""

# `executescript(SCHEMA)` is all CREATE ... IF NOT EXISTS, so it does precisely nothing to a
# manifest created before a column existed. Added columns need an explicit ALTER, and the
# only place that can reliably happen is when the connection is opened.
MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("remote_only", "ALTER TABLE papers ADD COLUMN remote_only INTEGER NOT NULL DEFAULT 0"),
    ("bucket", "ALTER TABLE papers ADD COLUMN bucket INTEGER NOT NULL DEFAULT -1"),
)

_ROW_COLUMNS = (
    "arxiv_id", "version", "shard", "title", "authors", "categories",
    "primary_category", "doi", "date_released", "date_updated", "bucket"
)

# What a claim hands back. `attempts` is read but never inserted -- it is maintained by
# the writer -- so it cannot live in _ROW_COLUMNS, which drives the INSERT. The
# orchestrator needs it to decide whether a paper that just failed has tries left.
_CLAIM_COLUMNS = _ROW_COLUMNS + ("attempts",)


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
    attempts: int = 0         # tries already spent, as of the moment it was claimed
    bucket: int = -1          # device partition key; derived, never passed in by hand

    def __post_init__(self) -> None:
        # Derived here rather than at each construction site, so a row can never reach the
        # manifest without a partition key -- and a row read back from a claim keeps the
        # value already stored, because that one is never negative.
        if self.bucket < 0:
            self.bucket = bucket_for(self.arxiv_id)

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
    remote_only: bool = False  # the artefacts went to the bucket; nothing is left on disk
    count_attempt: bool = False
    worker_id: int = 0        # pid of the process that handled it; for checkpoints
    converter: str = ""       # the backend that actually produced it, fallback included


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _backfill_buckets(conn: sqlite3.Connection, batch_size: int = 20_000) -> int:
    """Fill the partition key for rows that predate the column.

    Batched and committed as it goes, so an interrupted backfill resumes rather than
    restarting: the `bucket < 0` predicate is its own progress marker.
    """
    total = 0
    while True:
        ids = [r[0] for r in conn.execute(
            "SELECT arxiv_id FROM papers WHERE bucket < 0 LIMIT ?", (batch_size,)
        )]
        if not ids:
            break
        conn.executemany(
            "UPDATE papers SET bucket = ? WHERE arxiv_id = ?",
            [(bucket_for(paper_id), paper_id) for paper_id in ids],
        )
        conn.commit()
        total += len(ids)
    if total:
        log.info("filled the partition key for %d row(s)", total)
    return total


def _migrate(conn: sqlite3.Connection) -> None:
    """Bring an existing manifest up to the current column set."""
    columns = {r[1] for r in conn.execute("PRAGMA table_info(papers)")}
    if not columns:
        return                              # brand new file; SCHEMA is about to build it
    added = []
    for name, ddl in MIGRATIONS:
        if name not in columns:
            conn.execute(ddl)
            added.append(name)
    if added:
        conn.commit()
        log.info("manifest migrated: added column(s) %s", ", ".join(added))
    if conn.execute("SELECT 1 FROM papers WHERE bucket < 0 LIMIT 1").fetchone():
        # Only ever true once per manifest, but it can take a few seconds over millions of
        # rows -- hence the log line, so a slow first start has an explanation.
        log.info("filling the partition key for existing rows; this happens once")
        _backfill_buckets(conn)


def connect(db_path: Path, *, readonly: bool = False) -> sqlite3.Connection:
    """Open a tuned connection"""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row

    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")

    if not readonly:
        # Migrate first: SCHEMA now builds an index over `bucket`, which an older manifest
        # does not have a column for yet, and CREATE INDEX on a missing column is an error.
        _migrate(conn)
        conn.executescript(SCHEMA)
    return conn


def _partition_clause(partition: tuple[int, int] | None) -> tuple[str, tuple]:
    """`AND bucket % devices = index`, or nothing at all.

    `bucket % n` is not sargable, so this rides as a residual filter on idx_papers_status:
    the scan still stops at LIMIT, it just inspects roughly `n`x as many index entries to
    fill a batch. Negligible while pending work is plentiful, and the alternative -- a
    per-device index -- would have to be rebuilt every time the device count changed.

    `None` yields an empty clause, so a single-device run issues exactly the SQL it did
    before partitioning existed.
    """
    if not partition:
        return "", ()
    devices, index = partition
    return " AND bucket % ? = ?", (devices, index)


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
    def claim_batch(
        self,
        n: int,
        statuses: tuple[str, ...] = (PENDING,),
        *,
        max_attempts: int | None = None,
        partition: tuple[int, int] | None = None,
    ) -> list[PaperRow]:
        """Atomically move up to `n` rows to `in_flight` and return them.

        The UPDATE...RETURNING is a single statement, so two claimers can never hand the
        same paper to two workers even if the dispatcher is ever parallelised.

        `max_attempts` caps how many times a paper may be handed out in total. It is what
        keeps the retry-first pass from re-downloading a permanently broken PDF on every
        single run; `pending` rows have `attempts = 0`, so it is a no-op for fresh work.
        """
        placeholders = ", ".join("?" * len(statuses))
        ceiling = "" if max_attempts is None else " AND attempts < ?"
        slice_sql, slice_params = _partition_clause(partition)
        sql = (
            f"UPDATE papers SET status = ? WHERE arxiv_id IN ("
            f"  SELECT arxiv_id FROM papers WHERE status IN ({placeholders}){ceiling}"
            f"{slice_sql} LIMIT ?"
            f") RETURNING {', '.join(_CLAIM_COLUMNS)}"
        )
        params: tuple = (IN_FLIGHT, *statuses)
        if max_attempts is not None:
            params += (max_attempts,)
        params += slice_params
        with self.conn:
            cur = self.conn.execute(sql, (*params, n))
            return [PaperRow(**dict(r)) for r in cur.fetchall()]

    def count_claimable(
        self,
        statuses: tuple[str, ...],
        *,
        max_attempts: int | None = None,
        partition: tuple[int, int] | None = None,
    ) -> int:
        """How many rows `claim_batch` would eventually hand out for these statuses."""
        placeholders = ", ".join("?" * len(statuses))
        ceiling = "" if max_attempts is None else " AND attempts < ?"
        slice_sql, slice_params = _partition_clause(partition)
        params: tuple = statuses if max_attempts is None else (*statuses, max_attempts)
        return self.conn.execute(
            f"SELECT COUNT(*) FROM papers WHERE status IN ({placeholders}){ceiling}{slice_sql}",
            params + slice_params,
        ).fetchone()[0]

    def claimable_ids(
        self,
        statuses: tuple[str, ...],
        *,
        max_attempts: int | None = None,
        partition: tuple[int, int] | None = None,
    ) -> list[str]:
        """The ids `claim_batch` would hand out, read once and up front.

        The retry pass takes this snapshot before it dispatches anything. A paper that
        fails *again* mid-run goes straight back to `failed_convert`, and without a fixed
        worklist the very next claim would pick it up and retry it inside the same run,
        round and round. Naming the ids in advance makes "one attempt per paper per run"
        a property of the dispatch rather than a race that usually works out.
        """
        placeholders = ", ".join("?" * len(statuses))
        ceiling = "" if max_attempts is None else " AND attempts < ?"
        slice_sql, slice_params = _partition_clause(partition)
        params: tuple = statuses if max_attempts is None else (*statuses, max_attempts)
        return [
            r[0] for r in self.conn.execute(
                f"SELECT arxiv_id FROM papers WHERE status IN ({placeholders}){ceiling}{slice_sql}",
                params + slice_params,
            )
        ]

    def claim_ids(self, ids: list[str], statuses: tuple[str, ...] = RETRYABLE) -> list[PaperRow]:
        """Claim named rows, skipping any whose status moved on since the snapshot."""
        if not ids:
            return []
        id_slots = ", ".join("?" * len(ids))
        status_slots = ", ".join("?" * len(statuses))
        with self.conn:
            cur = self.conn.execute(
                f"UPDATE papers SET status = ? "
                f"WHERE arxiv_id IN ({id_slots}) AND status IN ({status_slots}) "
                f"RETURNING {', '.join(_CLAIM_COLUMNS)}",
                (IN_FLIGHT, *ids, *statuses),
            )
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
    def stats(self, *, partition: tuple[int, int] | None = None) -> dict[str, int]:
        """Counts by status, for the whole manifest or for one device's slice.

        `partition` is what lets a run measure its own share: with two devices the corpus
        figure is the shared goal but it is not what this process is working towards, and a
        progress bar whose total is twice its reachable maximum reports an ETA twice as long
        as the truth.
        """
        where, params = ("", ())
        if partition:
            where, params = " WHERE bucket % ? = ?", partition
        cur = self.conn.execute(
            f"SELECT status, COUNT(*) AS n FROM papers{where} GROUP BY status", params)
        counts = {r["status"]: r["n"] for r in cur}
        counts["total"] = sum(counts.values())
        counts["low_text"] = self.conn.execute(
            f"SELECT COUNT(*) FROM papers WHERE low_text = 1"
            f"{where.replace(' WHERE ', ' AND ') if where else ''}", params
        ).fetchone()[0]
        return counts

    # --- reconciling against object storage ------------------------------------------
    def shards(self, *, open_only: bool = False) -> list[str]:
        """Every shard in the manifest, or only those with unfinished work in them.

        A shard whose rows are all `done` cannot learn anything from being listed, so
        skipping it is free and sound. `no_pdf` counts as unfinished on purpose: an md
        object for a paper this device recorded as a 404 means the other device did get a
        PDF, and recovering that is worth the listing.
        """
        sql = "SELECT DISTINCT shard FROM papers"
        params: tuple = ()
        if open_only:
            sql += " WHERE status <> ?"
            params = (DONE,)
        return [r[0] for r in self.conn.execute(sql, params) if r[0]]

    def mark_done_from_objects(
        self, found: Iterable[tuple[str, int, str | None]], *, dry_run: bool = False
    ) -> dict[str, int]:
        """Mark papers done because their artefacts are already in the bucket.

        `found` yields `(arxiv_id_candidate, md_bytes, completed_at)`. Candidates rather
        than ids because an object name cannot be inverted with certainty on its own -- see
        `objectstore.id_candidates_from_object_name` -- so both readings are offered and
        the join decides. At most one can exist, since one contains a slash and the other
        does not.

        A TEMP table rather than a chain of `IN (...)` lists: it makes every count exact
        (the difference between "already done here" and "not in this manifest at all" is
        worth knowing -- the second just means the other device had a wider scope), and it
        makes `--dry-run` simply the same work minus the UPDATE.
        """
        conn = self.conn
        conn.execute(
            "CREATE TEMP TABLE IF NOT EXISTS sync_found ("
            "  arxiv_id TEXT PRIMARY KEY, md_bytes INTEGER, completed_at TEXT)"
        )
        conn.execute("DELETE FROM sync_found")
        conn.executemany("INSERT OR REPLACE INTO sync_found VALUES (?, ?, ?)", found)
        # Drop the readings that are not papers here, so what remains is one row per object
        # that this manifest actually knows about.
        conn.execute(
            "DELETE FROM sync_found WHERE arxiv_id NOT IN (SELECT arxiv_id FROM papers)")

        def _count(status: str) -> int:
            return conn.execute(
                "SELECT COUNT(*) FROM sync_found f JOIN papers p USING (arxiv_id) "
                "WHERE p.status = ?", (status,)
            ).fetchone()[0]

        counts = {
            "matched": conn.execute("SELECT COUNT(*) FROM sync_found").fetchone()[0],
            "already_done": _count(DONE),
            "no_pdf_recovered": _count(NO_PDF),
            "in_flight_skipped": _count(IN_FLIGHT),
        }
        counts["newly_marked"] = (
            counts["matched"] - counts["already_done"] - counts["in_flight_skipped"]
        )
        if dry_run:
            conn.rollback()
            return counts

        with conn:
            cur = conn.execute(
                "UPDATE papers SET status = ?, remote_only = 1, error = NULL, "
                "  md_bytes = COALESCE(md_bytes, "
                "    (SELECT md_bytes FROM sync_found f WHERE f.arxiv_id = papers.arxiv_id)), "
                "  completed_at = COALESCE(completed_at, "
                "    (SELECT completed_at FROM sync_found f WHERE f.arxiv_id = papers.arxiv_id)) "
                "WHERE arxiv_id IN (SELECT arxiv_id FROM sync_found) "
                # `in_flight` is excluded because a standalone `sync` will routinely overlap
                # this device's own run: flipping a live row to `done` only has the
                # ManifestWriter overwrite it moments later, and if that paper then fails,
                # `_apply` writes completed_at = NULL and the timestamp is lost.
                "  AND status NOT IN (?, ?)",
                (DONE, DONE, IN_FLIGHT),
            )
            counts["newly_marked"] = cur.rowcount
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
              remote_only  = :remote_only,
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
            "low_text": int(r.low_text), "remote_only": int(r.remote_only),
            "count_attempt": int(r.count_attempt),
            "completed_at": _utcnow() if r.status in (DONE, NO_PDF) else None,
        } for r in results]
        with conn:
            conn.executemany(sql, payload)
        self.applied += len(payload)
