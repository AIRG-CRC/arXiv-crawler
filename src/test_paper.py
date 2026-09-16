"""End-to-end check on a single paper: download, convert, store.

Deliberately its own module and its own entry point, so it can be run without touching
the manifest or the crawl:

    python -m src.test_paper 1706.03762
    python -m src.test_paper 1706.03762 --minio
    python -m src.test_paper 2010.11929 --converter lightonocr --minio

`python -m src.main test-paper <id>` dispatches here too, so the command exists in both
places and cannot drift between them.

This does one request to arXiv, writes into `data/` exactly as a real run would, and --
with `--minio` -- uploads and removes the local copy. Nothing is recorded in the manifest:
this is a probe, not a unit of the crawl.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import requests

from .config import Config
from .utils import paths as P
from .utils.converter import REGISTRY, convert_and_write
from .utils.logging_setup import configure_logging
from .utils.state import DONE, PaperRow

log = logging.getLogger("test_paper")

# Accepts a bare id, a versioned id, or any arxiv.org URL pointing at one.
_ID = re.compile(r"(?:abs/|pdf/)?(?P<id>[a-z-]+(?:\.[A-Z]{2})?/\d{7}|\d{4}\.\d{4,5})(?P<v>v\d+)?")


def parse_id(text: str) -> tuple[str, str]:
    """Return ``(arxiv_id, version)``; version is "" when the input did not pin one."""
    match = _ID.search(text.strip())
    if not match:
        raise SystemExit(f"could not read an arXiv id from {text!r}")
    return match.group("id"), match.group("v") or ""


def fetch_pdf(cfg: Config, arxiv_id: str, version: str, target: Path) -> int:
    """One polite request to arXiv. Returns the byte count written."""
    url = f"{cfg.crawl.base_url.rstrip('/')}/pdf/{arxiv_id}{version}"
    headers = {
        "User-Agent": f"arxiv-crawler/test-paper (+{cfg.crawl.contact})",
        "Accept": "application/pdf",
    }
    log.info("fetching %s", url)
    response = requests.get(url, headers=headers, timeout=cfg.crawl.timeout, stream=True)
    response.raise_for_status()

    target.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with target.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=cfg.crawl.chunk_size):
            if not chunk:
                continue
            if size == 0 and not chunk.startswith(b"%PDF-"):
                # arXiv answers 200 with an HTML interstitial while a PDF is generated.
                raise SystemExit("arXiv did not return a PDF (still generating?)")
            fh.write(chunk)
            size += len(chunk)
    if not size:
        raise SystemExit("arXiv returned an empty body")
    return size


def run(cfg: Config, args: argparse.Namespace) -> int:
    arxiv_id, version = parse_id(args.paper)
    cfg.override(convert_converter=args.converter)
    cfg.paths.ensure()

    minio_settings = None
    if args.minio:
        from dataclasses import asdict

        from .utils.objectstore import MinioSettings, MinioStore

        settings = MinioSettings.from_config(cfg.minio)
        settings.validate()
        store = MinioStore(settings)
        store.ensure_bucket()
        print(f"target      {store.describe()}")
        minio_settings = asdict(settings)
    else:
        print(f"target      {cfg.paths.data_dir} (local)")

    row = PaperRow(
        arxiv_id=arxiv_id, version=version or "v1", shard=P.shard_for(arxiv_id),
        title=args.title or f"arXiv:{arxiv_id}", authors="[]",
        categories="", primary_category="",
    )

    staged = P.staged_pdf_path(cfg.paths.data_dir, arxiv_id)
    started = time.monotonic()
    pdf_bytes = fetch_pdf(cfg, arxiv_id, version, staged)
    fetched_at = time.monotonic()
    print(f"downloaded  {pdf_bytes:,} bytes in {fetched_at - started:.1f}s")

    result = convert_and_write(
        row, staged, cfg.paths.data_dir, cfg.convert,
        base_url=cfg.crawl.base_url,
        pdf_bytes=pdf_bytes, pdf_sha256=None,
        keep_pdf=args.keep_pdf, minio=minio_settings,
    )
    elapsed = time.monotonic() - fetched_at

    print(f"converter   {result.converter or cfg.convert.converter}")
    print(f"status      {result.status}  ({elapsed:.1f}s)")
    if result.status != DONE:
        print(f"error       {result.error}")
        return 1

    print(f"pages       {result.n_pages}")
    print(f"tables      {result.n_tables}")
    print(f"characters  {result.n_chars:,}")
    if result.low_text:
        print("            flagged low_text — likely a scan with no text layer")

    if minio_settings:
        from .utils.objectstore import object_name

        prefix = minio_settings.get("prefix", "")
        for kind in ("md", "tables", "meta"):
            if kind == "tables" and not result.n_tables:
                continue
            print(f"uploaded    {object_name(kind, arxiv_id, prefix=prefix)}")
    else:
        print(f"markdown    {P.md_path(cfg.paths.data_dir, arxiv_id)}")
        if result.n_tables:
            print(f"tables      {P.tables_path(cfg.paths.data_dir, arxiv_id)}")
        print(f"metadata    {P.meta_path(cfg.paths.data_dir, arxiv_id)}")

    if args.show:
        target = P.md_path(cfg.paths.data_dir, arxiv_id)
        if target.exists():
            print("\n" + "-" * 70)
            print(target.read_text(encoding="utf-8")[: args.show])
        else:
            print("\n(local copy already uploaded and removed; use --keep-local to read it)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="test-paper", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("paper", help="arXiv id, versioned id, or an arxiv.org URL")
    p.add_argument("-c", "--config", type=Path, help="path to config.yaml")
    p.add_argument("--converter", choices=sorted(REGISTRY),
                   help="backend to use (default: convert.converter)")
    p.add_argument("--minio", action="store_true",
                   help="store the output in the bucket and drop the local copy")
    p.add_argument("--keep-pdf", action="store_true", help="keep the staged PDF")
    p.add_argument("--title", help="title for the front matter (metadata is not fetched)")
    p.add_argument("--show", type=int, nargs="?", const=2000, metavar="N",
                   help="print the first N characters of the markdown")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Config.load(args.config)
    configure_logging(cfg.paths.logs_dir, verbose=args.verbose, quiet=not args.verbose)
    try:
        return run(cfg, args)
    except requests.RequestException as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
