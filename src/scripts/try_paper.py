"""Convert one arXiv paper and inspect -- or compare -- the parsers.

    # one parser, write the output where you can read it
    python -m src.scripts.try_paper https://arxiv.org/abs/1706.03762 --converter pymupdf

    # compare every parser that is installed, side by side
    python -m src.scripts.try_paper 2010.11929 --compare

    # a specific set, with equation marking off
    python -m src.scripts.try_paper hep-th/9711200 \
        --compare pymupdf,pdfplumber --no-equations

Output goes to `data/samples/<id>/<converter>.{md,tables.md,json}` so two runs can be
diffed directly.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import replace
from pathlib import Path
import requests

from ..config import Config
from ..utils.converter import REGISTRY, ConversionResult, get_converter
from ..utils.paths import safe_id
from ..utils.state import PaperRow
from ..utils.writer import atomic_write_text, render_body, render_tables
from ..utils.state import Manifest
from ..utils.paths import shard_for

# https://arxiv.org/abs/2301.12345v2 | /pdf/2301.12345 | arXiv:2301.12345 | bare ids
_ID_PATTERNS = (
    re.compile(r"arxiv\.org/(?:abs|pdf)/(?P<id>[a-z\-.]+/\d{7}|\d{4}\.\d{4,5})(?:v(?P<v1>\d+))?"),
    re.compile(r"^\s*(?:arXiv:)?(?P<id>[a-z\-.]+/\d{7}|\d{4}\.\d{4,5})(?:v(?P<v2>\d+))?\s*$", re.I),
)


def parse_reference(text: str) -> tuple[str, str]:
    """Turn a link or id into ``(arxiv_id, version)``; version is '' if unpinned."""
    for pattern in _ID_PATTERNS:
        m = pattern.search(text.strip())
        if m:
            version = m.groupdict().get("v1") or m.groupdict().get("v2") or ""
            return m.group("id"), (f"v{version}" if version else "")
        
    raise SystemExit(
        f"could not read an arXiv id from {text!r}\n"
        "expected something like https://arxiv.org/abs/1706.03762, arXiv:1706.03762, "
        "or 1706.03762"
    )


def fetch_pdf(cfg: Config, arxiv_id: str, version: str, dest: Path) -> Path:
    """Download the PDF once and cache it"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"using cached PDF  {dest}  ({dest.stat().st_size / 1e6:.2f} MB)")
        return dest

    url = f"{cfg.crawl.base_url.rstrip('/')}/pdf/{arxiv_id}{version}"
    print(f"downloading       {url}")

    response = requests.get(
        url, timeout=cfg.crawl.timeout, stream=True,
        headers={"User-Agent": f"arxiv-crawler/0.1 (+{cfg.crawl.contact})"},
    )
    response.raise_for_status()

    body = response.content

    if not body.startswith(b"%PDF-"):
        raise SystemExit(f"{url} did not return a PDF (got {len(body)} bytes of something else)")
    
    dest.write_bytes(body)
    print(f"saved             {dest}  ({len(body) / 1e6:.2f} MB)")
    return dest


def fetch_metadata(cfg: Config, arxiv_id: str) -> dict:
    if cfg.paths.manifest_db.exists():
        with Manifest(cfg.paths.manifest_db) as m:
            row = m.conn.execute(
                "SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)
            ).fetchone()
            if row:
                return dict(row)
    return {}


def make_row(cfg: Config, arxiv_id: str, version: str) -> PaperRow:
    meta = fetch_metadata(cfg, arxiv_id)
    return PaperRow(
        arxiv_id=arxiv_id,
        version=version or meta.get("version") or "",
        shard=shard_for(arxiv_id),
        title=meta.get("title") or f"arXiv:{arxiv_id}",
        authors=meta.get("authors") or "[]",
        categories=meta.get("categories") or "",
        primary_category=meta.get("primary_category") or "",
        doi=meta.get("doi"),
        date_released=meta.get("date_released"),
        date_updated=meta.get("date_updated"),
    )


def run_one(name: str, cfg: Config, pdf: Path) -> tuple[ConversionResult | None, float, str]:
    """Convert with one backend. Returns ``(result, seconds, error)``."""
    start = time.perf_counter()
    try:
        result = get_converter(name, cfg.convert).convert(pdf)
        return result, time.perf_counter() - start, ""
    except ImportError as exc:
        return None, time.perf_counter() - start, f"not installed ({exc.name})"
    except Exception as exc:  # noqa: BLE001 - report, never abort the comparison
        return None, time.perf_counter() - start, f"{type(exc).__name__}: {exc}"


def write_sample(out_dir: Path, name: str, row: PaperRow, result: ConversionResult) -> None:
    atomic_write_text(out_dir / f"{name}.md", render_body(row, result, converter=name))
    if result.tables:
        atomic_write_text(out_dir / f"{name}.tables.md", render_tables(row, result))
    atomic_write_text(out_dir / f"{name}.json", json.dumps({
        "converter": name,
        "n_pages": result.n_pages,
        "n_chars": result.n_chars,
        "n_tables": result.n_tables,
        "tables": [{
            "index": t.index, "page": t.page, "kind": t.kind,
            "rows": t.n_rows, "cols": t.n_cols,
            "caption": t.caption, "columns": t.columns,
        } for t in result.tables],
    }, indent=2, ensure_ascii=False) + "\n")


def _preview(result: ConversionResult, chars: int) -> str:
    body = result.body_markdown.strip()
    return body[:chars] + ("..." if len(body) > chars else "")


def report(rows: list[tuple[str, ConversionResult | None, float, str]], detail: bool) -> None:
    print(f"\n{'converter':<16}{'time':>9}{'pages':>8}{'chars':>10}{'tables':>8}"
          f"{'algos':>7}{'caption':>9}  note")
    print("-" * 82)
    for name, result, secs, error in rows:
        if result is None:
            print(f"{name:<16}{secs:>8.1f}s{'-':>8}{'-':>10}{'-':>8}{'-':>7}{'-':>9}  {error}")
            continue
        algos = sum(1 for t in result.tables if t.kind == "pseudocode")
        captioned = sum(1 for t in result.tables if t.caption)
        print(f"{name:<16}{secs:>8.1f}s{result.n_pages:>8,}{result.n_chars:>10,}"
              f"{result.n_tables:>8,}{algos:>7,}{captioned:>9,}")

    ok = [(n, r) for n, r, _, _ in rows if r is not None]
    if len(ok) > 1:
        best_text = max(ok, key=lambda kv: kv[1].n_chars)
        best_tables = max(ok, key=lambda kv: kv[1].n_tables)
        print(f"\nmost body text     {best_text[0]} ({best_text[1].n_chars:,} chars)")
        print(f"most tables        {best_tables[0]} ({best_tables[1].n_tables} found)")
        print("\nThese are counts, not quality. A backend can 'win' on tables by "
              "misreading prose as\ntabular data — open the .tables.md files and look "
              "before trusting a number.")

    if detail:
        for name, result, _, _ in rows:
            if result is None:
                continue
            print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
            print(_preview(result, 900))
            for t in result.tables[:2]:
                print(f"\n--- {name}: table {t.index} ({t.kind}, page {t.page}) ---")
                if t.caption:
                    print(f"caption: {t.caption}")
                print("\n".join(t.markdown.splitlines()[:12]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paper", help="arXiv URL, arXiv:ID, or a bare id")
    ap.add_argument("-c", "--config", type=Path)
    ap.add_argument("--converter", choices=sorted(REGISTRY), default="pymupdf")
    ap.add_argument("--compare", nargs="?", const="ALL", metavar="a,b,c",
                    help="run several backends; bare --compare tries all of them")
    ap.add_argument("--out", type=Path, help="output directory (default data/samples/<id>)")
    ap.add_argument("--detail", action="store_true", help="print body and table previews")
    ap.add_argument("--table-strategy", choices=["lines_strict", "lines", "text"],
                    help="override convert.table_strategy for this run")
    ap.add_argument("--no-pseudocode", action="store_true",
                    help="keep algorithm floats as tables instead of fenced code")
    ap.add_argument("--no-equations", action="store_true",
                    help="do not wrap display equations in $$")
    ap.add_argument("--max-pages", type=int)
    args = ap.parse_args(argv)

    cfg = Config.load(args.config)
    cfg.convert = replace(
        cfg.convert,
        table_strategy=args.table_strategy or cfg.convert.table_strategy,
        detect_pseudocode=not args.no_pseudocode,
        preserve_equations=not args.no_equations,
        max_pages=args.max_pages or cfg.convert.max_pages,
        timeout=max(cfg.convert.timeout, 600),   # interactive use, not a corpus run
    )

    arxiv_id, version = parse_reference(args.paper)
    print(f"paper             arXiv:{arxiv_id}{version}")

    out_dir = args.out or (cfg.paths.data_dir / "samples" / safe_id(arxiv_id))
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = fetch_pdf(cfg, arxiv_id, version, out_dir / f"{safe_id(arxiv_id)}.pdf")
    row = make_row(cfg, arxiv_id, version)

    if args.compare:
        names = sorted(REGISTRY) if args.compare == "ALL" else [
            n.strip() for n in args.compare.split(",") if n.strip()
        ]
        unknown = [n for n in names if n not in REGISTRY]
        if unknown:
            raise SystemExit(f"unknown converter(s) {unknown}; choose from {sorted(REGISTRY)}")
    else:
        names = [args.converter]

    print(f"strategy          {cfg.convert.table_strategy}"
          f"  pseudocode={cfg.convert.detect_pseudocode}"
          f"  equations={cfg.convert.preserve_equations}")

    rows = []
    for name in names:
        print(f"running           {name} ...", end=" ", flush=True)
        result, secs, error = run_one(name, cfg, pdf)
        print(f"{secs:.1f}s" + (f"  [{error}]" if error else ""))
        if result is not None:
            write_sample(out_dir, name, row, result)
        rows.append((name, result, secs, error))

    report(rows, args.detail)
    print(f"\noutput            {out_dir}")
    if len(names) > 1:
        pair = [n for n, r, _, _ in rows if r is not None][:2]
        if len(pair) == 2:
            print(f"diff              diff {out_dir / (pair[0] + '.md')} "
                  f"{out_dir / (pair[1] + '.md')}")
    return 0 if any(r is not None for _, r, _, _ in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
