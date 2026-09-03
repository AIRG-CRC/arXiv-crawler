"""Bulk-load the per-paper metadata JSON into Postgres.

The JSON files under ``data/meta/`` are the portable source of truth; this is an
optional convenience for querying the catalog. Rows are streamed through
``COPY ... FROM STDIN`` into an UNLOGGED staging table and then merged with
``INSERT ... ON CONFLICT DO UPDATE``, which is orders of magnitude faster than
row-by-row inserts at corpus scale, and idempotent so it can be re-run at any time.

    pip install "psycopg[binary]"
    python -m src.scripts.ingest_postgres --dsn postgresql://user@localhost/arxiv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterator

from ..config import Config

log = logging.getLogger("ingest_postgres")

COLUMNS = [
    "id", "title", "authors", "date_released", "date_updated", "doi",
    "categories", "primary_category", "source_url",
    "n_pages", "n_tables", "n_chars", "md_path", "tables_path",
    "raw",
]


def iter_meta_files(meta_dir: Path) -> Iterator[Path]:
    yield from sorted(meta_dir.rglob("*.json"))


def to_tuple(record: dict) -> tuple:
    """Flatten one metadata record into the column order above."""
    def blank_to_none(v: object) -> object:
        return None if v in ("", []) else v

    return (
        record.get("id"),
        record.get("title") or "",
        record.get("authors") or [],
        blank_to_none(record.get("date_released")),
        blank_to_none(record.get("date_updated")),
        blank_to_none(record.get("doi")),
        record.get("categories") or [],
        blank_to_none(record.get("primary_category")),
        blank_to_none(record.get("source_url")),
        record.get("n_pages"),
        record.get("n_tables"),
        record.get("n_chars"),
        blank_to_none(record.get("md_path")),
        blank_to_none(record.get("tables_path")),
        json.dumps(record, ensure_ascii=False),
    )


def ingest(dsn: str, meta_dir: Path, table: str, batch_size: int = 5000) -> int:
    try:
        import psycopg
    except ImportError:
        raise SystemExit(
            'psycopg is not installed. Run:  pip install "psycopg[binary]"'
        ) from None

    schema_sql = (Path(__file__).parent / "schema.sql").read_text()
    cols = ", ".join(COLUMNS)
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in COLUMNS if c != "id")
    total = 0

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(schema_sql)
            cur.execute(
                f"CREATE TEMP TABLE staging (LIKE {table} INCLUDING DEFAULTS) "
                f"ON COMMIT DROP"
            )

            with cur.copy(f"COPY staging ({cols}) FROM STDIN") as copy:
                for path in iter_meta_files(meta_dir):
                    try:
                        record = json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError) as exc:
                        log.warning("skipping %s: %s", path, exc)
                        continue
                    if not record.get("id"):
                        continue
                    copy.write_row(to_tuple(record))
                    total += 1
                    if total % batch_size == 0:
                        log.info("staged %s records", f"{total:,}")

            cur.execute(
                f"INSERT INTO {table} ({cols}) SELECT {cols} FROM staging "
                f"ON CONFLICT (id) DO UPDATE SET {updates}"
            )
        conn.commit()
    return total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", type=Path)
    ap.add_argument("--dsn", help="overrides postgres.dsn from config.yaml")
    ap.add_argument("--table", help="overrides postgres.table")
    ap.add_argument("--meta-dir", type=Path, help="defaults to <data_dir>/meta")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    cfg = Config.load(args.config)
    dsn = args.dsn or cfg.postgres.dsn
    if not dsn:
        print("no DSN: pass --dsn or set postgres.dsn in config.yaml", file=sys.stderr)
        return 2

    meta_dir = args.meta_dir or cfg.paths.meta_dir
    if not meta_dir.exists():
        print(f"no metadata at {meta_dir}; run the pipeline first", file=sys.stderr)
        return 2

    n = ingest(dsn, meta_dir, args.table or cfg.postgres.table)
    log.info("ingested %s record(s) into %s", f"{n:,}", args.table or cfg.postgres.table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
