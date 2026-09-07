"""arxiv-crawler command line interface.

    prepare   load the Kaggle metadata snapshot into the manifest
    run       download + convert, in parallel, resumably
    status    progress report
    retry     re-queue retryable failures
    verify    cross-check the manifest against what is actually on disk
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import Config
from .utils import paths as P
from .utils.checkpoint import CheckpointStore
from .utils.crawler import run_pipeline
from .utils.logging_setup import configure_logging
from .utils.prepare_data import prepare
from .utils.state import DONE, FAILED_CONVERT, FAILED_DOWNLOAD, NO_PDF, PENDING, Manifest

log = logging.getLogger("arxiv_crawler")

# `run` is a progress bar, not a transcript: it logs to the file and keeps stderr clear.
# The other commands are one-shot reports, so their handful of lines belong on stderr.
QUIET_COMMANDS = {"run"}


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# --- commands ------------------------------------------------------------------------
def cmd_prepare(cfg: Config, args: argparse.Namespace) -> int:
    cfg.override(
        paths_metadata_file=args.metadata,
        scope_categories=args.categories,
        scope_primary_only=args.primary_only or None,
        scope_date_from=getattr(args, "from"),
        scope_date_to=args.to,
        scope_max_papers=args.limit,
    )
    counts = prepare(cfg)
    log.info(
        "read %(read)s records, %(matched)s matched the scope, %(inserted)s new row(s) inserted",
        {k: f"{v:,}" for k, v in counts.items()},
    )
    return 0


def cmd_run(cfg: Config, args: argparse.Namespace) -> int:
    cfg.override(
        crawl_workers=args.download_workers,
        convert_workers=args.convert_workers,
        crawl_rate_per_sec=args.rps,
        crawl_burst=args.burst,
        convert_converter=args.converter,
        retry_on_start=False if args.no_retry_failed else None,
        retry_max_attempts=args.max_attempts,
    )
    if cfg.crawl.contact == "ai@crc.calvin.ac.id":
        # Printed, not just logged, despite `run` being otherwise silent: crawling arXiv
        # without identifying yourself is a policy problem, and a warning nobody sees is
        # no warning at all.
        message = ("warning: crawl.contact is still the placeholder - arXiv asks automated "
                   "clients to identify themselves. Set a real address in config.yaml.")
        log.warning("%s", message)
        print(message, file=sys.stderr)
    tallies = run_pipeline(
        cfg, limit=args.limit, keep_pdf=args.keep_pdf, worker_bars=args.worker_bars
    )
    summary = (
        "processed {processed:,}: {done:,} converted, {no_pdf:,} without a PDF, "
        "{failed:,} failed".format(**tallies)
    )
    if tallies.get("retried"):
        summary += f" ({tallies['retried']:,} were retries of earlier failures)"
    log.info("%s", summary)
    if tallies.get("processed"):
        print(summary)
        if tallies.get("failed"):
            print(f"full detail in {cfg.paths.logs_dir / 'crawler.log'}")
    return 0


def cmd_status(cfg: Config, _args: argparse.Namespace) -> int:
    with Manifest(cfg.paths.manifest_db) as m:
        s = m.stats()
        total = s.pop("total", 0)
        low_text = s.pop("low_text", 0)
        if not total:
            print("Manifest is empty. Run `prepare` first.")
            return 0

        print(f"\nManifest: {cfg.paths.manifest_db}")
        print(f"{'status':<18}{'papers':>12}{'share':>9}")
        print("-" * 39)
        for status in (DONE, PENDING, NO_PDF, FAILED_DOWNLOAD, FAILED_CONVERT):
            n = s.pop(status, 0)
            print(f"{status:<18}{n:>12,}{n / total * 100:>8.1f}%")
        for status, n in sorted(s.items()):
            print(f"{status:<18}{n:>12,}{n / total * 100:>8.1f}%")
        print("-" * 39)
        print(f"{'total':<18}{total:>12,}")
        if low_text:
            print(f"\n{low_text:,} paper(s) flagged low_text (likely scanned; consider "
                  f"re-running them with --converter docling)")

        row = m.conn.execute(
            "SELECT COUNT(*) n, SUM(pdf_bytes) pdf, SUM(md_bytes) md, SUM(tables_bytes) tb, "
            "SUM(n_tables) tables FROM papers WHERE status = ?", (DONE,)
        ).fetchone()
        if row["n"]:
            print(f"\nconverted output   {_human((row['md'] or 0) + (row['tb'] or 0))}"
                  f"  from {_human(row['pdf'] or 0)} of PDF"
                  f"  ({row['tables'] or 0:,} tables extracted)")
    return 0


def cmd_retry(cfg: Config, args: argparse.Namespace) -> int:
    stage = None if args.stage == "all" else args.stage
    with Manifest(cfg.paths.manifest_db) as m:
        n = m.reset_failed(stage, max_attempts=args.max_attempts)
    log.info("re-queued %d paper(s); run `run` to process them", n)
    if stage in (None, "convert"):
        log.info("note: PDFs are deleted after conversion, so a convert retry re-downloads")
    return 0


def cmd_verify(cfg: Config, args: argparse.Namespace) -> int:
    data_dir = cfg.paths.data_dir
    missing: list[str] = []
    with Manifest(cfg.paths.manifest_db) as m:
        checked = 0
        for row in m.conn.execute("SELECT * FROM papers WHERE status = ?", (DONE,)):
            checked += 1
            aid = row["arxiv_id"]
            expected = [P.md_path(data_dir, aid), P.meta_path(data_dir, aid)]
            if (row["n_tables"] or 0) > 0:
                expected.append(P.tables_path(data_dir, aid))
            if any(not p.exists() for p in expected):
                missing.append(aid)

        print(f"checked {checked:,} paper(s) marked done; {len(missing):,} missing files")
        for aid in missing[:20]:
            print(f"  missing output: {aid}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20:,} more")

        if missing and args.fix:
            m.conn.executemany(
                "UPDATE papers SET status = ?, error = 'outputs missing' WHERE arxiv_id = ?",
                [(PENDING, aid) for aid in missing],
            )
            m.conn.commit()
            print(f"re-queued {len(missing):,} paper(s)")

    stragglers = list(cfg.paths.tmp_dir.glob("*.pdf*"))
    if stragglers:
        print(f"{len(stragglers)} staged PDF(s) left in {cfg.paths.tmp_dir}")
    return 1 if missing and not args.fix else 0


def cmd_checkpoint(cfg: Config, args: argparse.Namespace) -> int:
    store = CheckpointStore(cfg.paths.checkpoints_dir)
    print(f"\nCheckpoints: {cfg.paths.checkpoints_dir}\n")
    print(store.describe())
    if args.clear:
        for path in cfg.paths.checkpoints_dir.glob("*.json"):
            path.unlink(missing_ok=True)
        print("\ncleared — the manifest still knows what is outstanding")
    return 0


# --- argument parsing ----------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="arxiv-crawler", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--config", type=Path, help="path to config.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("prepare", help="load the metadata snapshot into the manifest")
    sp.add_argument("--metadata", type=Path, help="path to arxiv-metadata-oai-snapshot.json")
    sp.add_argument("--categories", type=_csv, metavar="cs.LG,cs.CL")
    sp.add_argument("--primary-only", action="store_true",
                    help="match --categories against the primary category only")
    sp.add_argument("--from", metavar="YYYY-MM", help="earliest v1 submission month")
    sp.add_argument("--to", metavar="YYYY-MM", help="latest v1 submission month")
    sp.add_argument("--limit", type=int, help="cap on papers inserted")
    sp.set_defaults(func=cmd_prepare)

    sr = sub.add_parser("run", help="download and convert, in parallel")
    sr.add_argument("--download-workers", type=int)
    sr.add_argument("--convert-workers", type=int)
    sr.add_argument("--rps", type=float, help="global request rate ceiling, shared by all workers")
    sr.add_argument("--burst", type=int)
    sr.add_argument("--converter",
                    choices=["pymupdf", "pdfplumber", "docling", "opendataloader", "markitdown"])
    sr.add_argument("--limit", type=int, help="stop after this many papers")
    sr.add_argument("--keep-pdf", action="store_true", help="keep staged PDFs (debugging)")
    sr.add_argument("--no-retry-failed", action="store_true",
                    help="skip the retry pass and go straight to pending work")
    sr.add_argument("--max-attempts", type=int,
                    help="total tries a paper gets before the retry pass gives up "
                         "(default: retry.max_attempts in config.yaml)")
    sr.add_argument("--worker-bars", action="store_true",
                    help="one progress line per conversion worker, under the main bar")
    sr.set_defaults(func=cmd_run)

    ss = sub.add_parser("status", help="progress report")
    ss.set_defaults(func=cmd_status)

    st = sub.add_parser("retry", help="re-queue retryable failures")
    st.add_argument("--stage", choices=["download", "convert", "all"], default="all")
    st.add_argument("--max-attempts", type=int, default=4)
    st.set_defaults(func=cmd_retry)

    sc = sub.add_parser("checkpoint", help="per-worker progress of the current or last run")
    sc.add_argument("--clear", action="store_true",
                    help="delete the checkpoint files (progress counters only, not work state)")
    sc.set_defaults(func=cmd_checkpoint)

    sv = sub.add_parser("verify", help="cross-check the manifest against files on disk")
    sv.add_argument("--fix", action="store_true", help="re-queue papers whose outputs vanished")
    sv.set_defaults(func=cmd_verify)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Config.load(args.config)
    configure_logging(
        cfg.paths.logs_dir,
        verbose=args.verbose,
        quiet=args.command in QUIET_COMMANDS,
    )
    try:
        return args.func(cfg, args)
    except FileNotFoundError as exc:
        # In quiet mode nothing else reaches the terminal, so an error that stops the
        # command has to be printed as well as logged.
        log.error("%s", exc)
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        log.warning("aborted")
        print("aborted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
