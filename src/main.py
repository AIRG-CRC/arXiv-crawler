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
from .utils.converter import REGISTRY
from .utils.crawler import run_pipeline
from .utils.logging_setup import configure_logging
from .utils.partition import DEVICES_ENV, INDEX_ENV, resolve_partition
from .utils.prepare_data import prepare
from .utils.state import DONE, FAILED_CONVERT, FAILED_DOWNLOAD, NO_PDF, PENDING, Manifest

log = logging.getLogger("arxiv_crawler")

# Commands that own the terminal with a progress bar: they log to the file and keep
# stderr clear. The rest are one-shot reports, so their few lines belong on stderr.
QUIET_COMMANDS = {"run", "run-minio", "dump", "test-paper", "sync"}

# Sourced from the registry rather than a hand-written list, so a new backend is
# selectable the moment it is registered.
CONVERTERS = sorted(REGISTRY)


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
        crawl_cooldown_seconds=args.cooldown_seconds,
    )
    if args.retry_all:
        # `override` skips None values, so the "no ceiling" case is set directly.
        cfg.retry.max_attempts = None
    try:
        partition = resolve_partition(args.devices, args.device_index)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if cfg.crawl.contact == "ai@crc.calvin.ac.id":
        # Printed, not just logged, despite `run` being otherwise silent: crawling arXiv
        # without identifying yourself is a policy problem, and a warning nobody sees is
        # no warning at all.
        message = ("warning: crawl.contact is still the placeholder - arXiv asks automated "
                   "clients to identify themselves. Set a real address in config.yaml.")
        log.warning("%s", message)
        print(message, file=sys.stderr)
    tallies = run_pipeline(
        cfg, limit=args.limit, keep_pdf=args.keep_pdf, worker_bars=args.worker_bars,
        retry_all=args.retry_all, to_minio=getattr(args, "to_minio", False),
        partition=partition, sync_bucket=args.sync, claim_any=args.claim_any,
    )
    summary = (
        "processed {processed:,}: {done:,} converted, {no_pdf:,} without a PDF, "
        "{failed:,} failed".format(**tallies)
    )
    if tallies.get("retried"):
        summary += f" ({tallies['retried']:,} were retries of earlier failures)"
    log.info("%s", summary)
    if tallies.get("already_elsewhere"):
        summary += (f"; {tallies['already_elsewhere']:,} were already in the bucket "
                    f"and were not re-crawled")
    if tallies.get("cooldowns"):
        summary += f"; paused {tallies['cooldowns']} time(s) for arXiv throttling"
    if tallies.get("processed"):
        print(summary)
        if tallies.get("failed"):
            print(f"full detail in {cfg.paths.logs_dir / 'crawler.log'}")
    if tallies.get("throttled_out"):
        message = ("arXiv was still refusing requests after the full cooldown ladder, so the "
                   "run stopped early. Nothing is lost -- outstanding papers are back in the "
                   "queue. Wait a few hours before running again.")
        log.warning("%s", message)
        print(message, file=sys.stderr)
        # EX_TEMPFAIL. A supervisor or `while true` wrapper that relaunches on 0 would walk
        # straight back into the block and undo the whole point of the cooldown.
        return 75
    return 0


def cmd_run_minio(cfg: Config, args: argparse.Namespace) -> int:
    """`run`, but the converted paper goes to the bucket and the local copy is removed."""
    args.to_minio = True
    return cmd_run(cfg, args)


def cmd_dump(cfg: Config, args: argparse.Namespace) -> int:
    """Upload the corpus already on disk, then drop the local copies."""
    from tqdm import tqdm

    from .utils.migrate import count_artefacts, dump_to_bucket, prune_empty_dirs
    from .utils.objectstore import MinioSettings, MinioStore, ObjectStoreError

    data_dir = cfg.paths.data_dir
    try:
        settings = MinioSettings.from_config(cfg.minio)
        store = MinioStore(settings)
        if not args.dry_run:
            # A dry run never opens a connection, so it should not demand credentials --
            # listing what *would* be sent is exactly what you want before setting them.
            settings.validate()
            store.ensure_bucket()
    except ObjectStoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    total = count_artefacts(data_dir)
    if not total:
        print(f"nothing to dump — no files under {data_dir}/{{md,tables,meta}}")
        return 0

    target = min(total, args.limit) if args.limit else total
    print(f"{'would upload' if args.dry_run else 'uploading'} {target:,} file(s) "
          f"from {data_dir} to {store.describe()}")
    if not args.keep_local and not args.dry_run:
        print("local copies are removed once each upload succeeds")

    bar = tqdm(total=target, unit="file", desc="dump", smoothing=0.05)
    try:
        report = dump_to_bucket(
            store, data_dir,
            keep_local=args.keep_local, skip_existing=args.skip_existing,
            limit=args.limit, dry_run=args.dry_run, progress=bar,
        )
    finally:
        bar.close()

    print(f"\nuploaded {report.uploaded:,}  skipped {report.skipped:,}  "
          f"failed {report.failed:,}  ({_human(report.bytes_sent)} sent)")
    for message in report.errors[:10]:
        print(f"  ✗ {message}")
    if len(report.errors) > 10:
        print(f"  ... and {len(report.errors) - 10:,} more; see the log")
    if not args.keep_local and not args.dry_run:
        pruned = prune_empty_dirs(data_dir)
        if pruned:
            print(f"removed {pruned:,} empty shard director{'y' if pruned == 1 else 'ies'}")
    if report.failed:
        print("re-run `dump` to retry the failures — the local copies are still there")
    return 1 if report.failed else 0


def cmd_sync(cfg: Config, args: argparse.Namespace) -> int:
    """Learn from the bucket what another device has already converted."""
    from tqdm import tqdm

    from .utils.objectstore import MinioSettings, MinioStore, ObjectStoreError
    from .utils.sync import run_sync

    try:
        partition = resolve_partition(args.devices, args.device_index)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    run_marker = cfg.paths.checkpoints_dir / "run.json"
    if run_marker.exists() and not args.force:
        # That file exists exactly while a run is in flight, or after one was interrupted.
        # Syncing under a live run is safe -- in_flight rows are excluded -- but it is much
        # more likely to be a mistake than an intention.
        print(f"a run appears to be in flight ({run_marker}). Papers it holds are left "
              f"alone; pass --force to sync anyway.", file=sys.stderr)
        return 2

    try:
        settings = MinioSettings.from_config(cfg.minio)
        settings.validate()
        store = MinioStore(settings)
        # `bucket_present`, not `ensure_bucket`: a reader that creates a missing bucket turns
        # a typo into an empty new bucket and then reports nothing to do.
        if not store.bucket_present():
            print(f"error: no bucket {settings.bucket!r} at {settings.endpoint}",
                  file=sys.stderr)
            return 2
    except ObjectStoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:                # noqa: BLE001 - an unreachable host, usually
        log.error("could not reach the bucket", exc_info=exc)
        print(f"error: could not reach the bucket ({type(exc).__name__}: {exc})",
              file=sys.stderr)
        return 2

    with Manifest(cfg.paths.manifest_db) as manifest:
        shards = len(manifest.shards(open_only=True))
        if not shards:
            print("nothing to sync — every paper in this manifest is already done")
            return 0
        print(f"{'checking' if args.dry_run else 'syncing'} {store.describe()}")
        bar = tqdm(total=shards, unit="shard", desc="sync", smoothing=0.05)
        try:
            report = run_sync(cfg, manifest, store, partition=partition,
                              dry_run=args.dry_run, progress=bar)
        except Exception as exc:            # noqa: BLE001 - the network, mid-scan
            log.error("sync failed", exc_info=exc)
            print(f"\nerror: sync failed ({type(exc).__name__}: {exc}). Whatever it had "
                  f"already marked is committed; run it again.", file=sys.stderr)
            return 2
        finally:
            bar.close()

    for line in report.lines():
        print(f"  {line}")
    if args.dry_run:
        print("\n(dry run — the manifest was not changed)")
    return 0


def cmd_test_paper(cfg: Config, args: argparse.Namespace) -> int:
    """Delegates to `src.test_paper`, so the two entry points cannot drift apart."""
    from .test_paper import run as run_test_paper

    return run_test_paper(cfg, args)


def cmd_status(cfg: Config, _args: argparse.Namespace) -> int:
    with Manifest(cfg.paths.manifest_db) as m:
        s = m.stats()
        total = s.pop("total", 0)
        low_text = s.pop("low_text", 0)
        if not total:
            print("Manifest is empty. Run `prepare` first.")
            return 0

        print(f"\nManifest: {cfg.paths.manifest_db}")
        marker = cfg.paths.sync_state
        if marker.exists():
            try:
                import json

                state = json.loads(marker.read_text(encoding="utf-8"))
                print(f"Last bucket sync: {state.get('synced_at', '?')} "
                      f"as device '{state.get('device', '?')}' "
                      f"({state.get('report', {}).get('newly_marked', 0):,} marked done)")
            except (OSError, ValueError):
                pass
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
    """Cross-check the manifest against the files on disk.

    Rows flagged `remote_only` are skipped rather than checked: their artefacts were
    uploaded and the local copies removed, which is the whole point of `run-minio` and of
    `sync`. Checking them against disk reported every paper of a bucket-backed corpus as
    missing -- and `--fix` then re-queued the entire corpus, discarding weeks of work.
    """
    data_dir = cfg.paths.data_dir
    missing: list[str] = []
    with Manifest(cfg.paths.manifest_db) as m:
        checked = remote = 0
        for row in m.conn.execute("SELECT * FROM papers WHERE status = ?", (DONE,)):
            if row["remote_only"]:
                remote += 1
                continue
            checked += 1
            aid = row["arxiv_id"]
            expected = [P.md_path(data_dir, aid), P.meta_path(data_dir, aid)]
            if (row["n_tables"] or 0) > 0:
                expected.append(P.tables_path(data_dir, aid))
            if any(not p.exists() for p in expected):
                missing.append(aid)

        print(f"checked {checked + remote:,} paper(s) marked done")
        if remote:
            print(f"  {remote:,} stored remotely (not checked against disk)")
        print(f"  {checked:,} checked on disk; {len(missing):,} missing files")
        for aid in missing[:20]:
            print(f"  missing output: {aid}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20:,} more")

        if missing and args.fix:
            # A `--fix` that would re-queue most of the corpus is far likelier to be a
            # misconfigured data_dir, or a `remote_only` flag that never got set, than a
            # genuine mass deletion. Re-downloading and re-converting 200,000 papers is not
            # something to do on an inference from a directory listing.
            share = len(missing) / checked if checked else 0.0
            if share > 0.5 and not args.force:
                print(f"\nrefusing to re-queue {share:.0%} of the papers checked on disk. "
                      f"If the local copies really are gone, re-run with --fix --force; if "
                      f"they are in the bucket, run `sync` instead.", file=sys.stderr)
                return 2
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

    def add_run_flags(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """`run` and `run-minio` differ only in where the output goes, so they share
        every flag rather than keeping two lists in step by hand."""
        parser.add_argument("--download-workers", type=int)
        parser.add_argument("--convert-workers", type=int)
        parser.add_argument("--rps", type=float,
                            help="global request rate ceiling, shared by all workers")
        parser.add_argument("--burst", type=int)
        parser.add_argument("--converter", choices=sorted(CONVERTERS))
        parser.add_argument("--limit", type=int, help="stop after this many papers")
        parser.add_argument("--keep-pdf", action="store_true",
                            help="keep staged PDFs (debugging)")
        parser.add_argument("--no-retry-failed", action="store_true",
                            help="skip the retry pass and go straight to pending work")
        parser.add_argument("--retry-all", action="store_true",
                            help="retry every failed paper, ignoring the attempt ceiling")
        parser.add_argument("--max-attempts", type=int,
                            help="total tries a paper gets before the retry pass gives up "
                                 "(default: retry.max_attempts in config.yaml)")
        parser.add_argument("--worker-bars", action="store_true",
                            help="one progress line per conversion worker, under the main bar")
        parser.add_argument("--devices", type=int, metavar="N",
                            help="how many devices are sharing this corpus "
                                 f"(default: ${DEVICES_ENV}, else 1)")
        parser.add_argument("--device-index", type=int, metavar="I",
                            help="which slice this device takes, 0-based (default: "
                                 f"${INDEX_ENV}, else 0). Two devices must never share an "
                                 "index: each would crawl the same half of the corpus and "
                                 "neither would touch the other")
        parser.add_argument("--claim-any", action="store_true",
                            help="once this device's slice is drained, take papers from "
                                 "outside it (each checked against the bucket first)")
        parser.add_argument("--sync", dest="sync", action="store_true", default=None,
                            help="reconcile with the bucket before crawling "
                                 "(default: sync.on_start, for run-minio only)")
        parser.add_argument("--no-sync", dest="sync", action="store_false",
                            help="skip the bucket reconciliation")
        parser.add_argument("--cooldown", type=int, dest="cooldown_seconds", metavar="SECONDS",
                            help="pause every download this long when arXiv answers 406/403, "
                                 "doubling each round it persists; 0 disables "
                                 "(default: crawl.cooldown_seconds)")
        return parser

    sr = add_run_flags(sub.add_parser("run", help="download and convert, in parallel"))
    sr.set_defaults(func=cmd_run, to_minio=False)

    sm = add_run_flags(sub.add_parser(
        "run-minio", help="like `run`, but store each paper in MinIO and drop the local copy"))
    sm.set_defaults(func=cmd_run_minio, to_minio=True)

    sd = sub.add_parser("dump", help="upload the corpus already on disk to MinIO")
    sd.add_argument("--keep-local", action="store_true",
                    help="copy instead of move: leave the local files in place")
    sd.add_argument("--skip-existing", action="store_true",
                    help="do not re-send objects already in the bucket (one HEAD per file)")
    sd.add_argument("--limit", type=int, help="stop after this many files")
    sd.add_argument("--dry-run", action="store_true",
                    help="report what would be uploaded, touching nothing")
    sd.set_defaults(func=cmd_dump)

    stp = sub.add_parser("test-paper",
                         help="download, convert and store one paper; ignores the manifest")
    stp.add_argument("paper", help="arXiv id, versioned id, or an arxiv.org URL")
    stp.add_argument("--converter", choices=sorted(CONVERTERS))
    stp.add_argument("--minio", action="store_true",
                     help="store the output in the bucket and drop the local copy")
    stp.add_argument("--keep-pdf", action="store_true", help="keep the staged PDF")
    stp.add_argument("--title", help="title for the front matter (metadata is not fetched)")
    stp.add_argument("--show", type=int, nargs="?", const=2000, metavar="N",
                     help="print the first N characters of the markdown")
    stp.set_defaults(func=cmd_test_paper)

    sy = sub.add_parser("sync",
                        help="mark papers done that another device already put in the bucket")
    sy.add_argument("--dry-run", action="store_true",
                    help="report what would be marked, changing nothing")
    sy.add_argument("--devices", type=int, metavar="N",
                    help="recorded in this device's marker, for the partition cross-check")
    sy.add_argument("--device-index", type=int, metavar="I")
    sy.add_argument("--force", action="store_true",
                    help="sync even though a run looks like it is in flight")
    sy.set_defaults(func=cmd_sync)

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
    sv.add_argument("--force", action="store_true",
                    help="with --fix, re-queue even when most papers look missing")
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
