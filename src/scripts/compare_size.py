# Compare downloaded PDF size against the Markdown the pipeline kept.

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

from ..config import Config
from ..utils.state import DONE, Manifest

FULL_CORPUS = 2_800_000   # approximate arXiv size, for the extrapolation


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} TB"


def collect(manifest: Manifest) -> list[dict]:
    rows = []
    for r in manifest.conn.execute(
        "SELECT arxiv_id, primary_category, pdf_bytes, md_bytes, tables_bytes, "
        "n_pages, n_tables, low_text FROM papers WHERE status = ? AND md_bytes IS NOT NULL",
        (DONE,),
    ):
        pdf = r["pdf_bytes"] or 0
        text = (r["md_bytes"] or 0) + (r["tables_bytes"] or 0)
        rows.append({
            "arxiv_id": r["arxiv_id"],
            "primary_category": r["primary_category"] or "?",
            "pdf_bytes": pdf,
            "text_bytes": text,
            "ratio": pdf / text if text else 0.0,
            "n_pages": r["n_pages"] or 0,
            "n_tables": r["n_tables"] or 0,
            "low_text": bool(r["low_text"]),
        })
    return rows


def _percentiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {}
    def at(q: float) -> float:
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]
    return {"p10": at(0.10), "p50": at(0.50), "p90": at(0.90), "p99": at(0.99)}


def report(rows: list[dict], top: int) -> None:
    if not rows:
        print("No converted papers yet. Run `python -m src.main run` first.")
        return

    n = len(rows)
    pdf_total = sum(r["pdf_bytes"] for r in rows)
    text_total = sum(r["text_bytes"] for r in rows)
    ratios = [r["ratio"] for r in rows if r["ratio"]]

    print(f"\nConverted papers: {n:,}\n")
    print(f"{'':<16}{'total':>14}{'mean':>12}{'median':>12}")
    print("-" * 54)
    print(f"{'PDF (deleted)':<16}{_human(pdf_total):>14}"
          f"{_human(pdf_total / n):>12}"
          f"{_human(statistics.median(r['pdf_bytes'] for r in rows)):>12}")
    print(f"{'Markdown kept':<16}{_human(text_total):>14}"
          f"{_human(text_total / n):>12}"
          f"{_human(statistics.median(r['text_bytes'] for r in rows)):>12}")
    print("-" * 54)
    if text_total:
        print(f"\noverall compression   {pdf_total / text_total:,.1f}x smaller as Markdown")
    if ratios:
        pct = _percentiles(ratios)
        print("per-paper ratio       " + "  ".join(f"{k}={v:,.1f}x" for k, v in pct.items()))

    pages = sum(r["n_pages"] for r in rows)
    tables = sum(r["n_tables"] for r in rows)
    with_tables = sum(1 for r in rows if r["n_tables"])
    low = sum(1 for r in rows if r["low_text"])
    print(f"\npages                 {pages:,} ({pages / n:,.1f} per paper)")
    print(f"tables extracted      {tables:,} "
          f"({with_tables:,} papers have at least one -- {with_tables / n * 100:,.1f}%)")
    if low:
        print(f"low_text flagged      {low:,} ({low / n * 100:,.1f}%) -- likely scans, "
              f"consider re-running with --converter docling")

    # --- per-category ---------------------------------------------------------------
    by_cat: dict[str, dict[str, float]] = {}
    for r in rows:
        acc = by_cat.setdefault(r["primary_category"], {"n": 0, "pdf": 0, "text": 0, "tables": 0})
        acc["n"] += 1
        acc["pdf"] += r["pdf_bytes"]
        acc["text"] += r["text_bytes"]
        acc["tables"] += r["n_tables"]

    print(f"\nBy primary category (top {top} by paper count)")
    print(f"{'category':<16}{'papers':>9}{'PDF':>12}{'Markdown':>12}{'ratio':>9}{'tables':>9}")
    print("-" * 67)
    for cat, a in sorted(by_cat.items(), key=lambda kv: -kv[1]["n"])[:top]:
        ratio = a["pdf"] / a["text"] if a["text"] else 0
        print(f"{cat:<16}{int(a['n']):>9,}{_human(a['pdf']):>12}"
              f"{_human(a['text']):>12}{ratio:>8,.1f}x{int(a['tables']):>9,}")

    # --- extrapolation --------------------------------------------------------------
    per_paper = text_total / n
    print(f"\nExtrapolated to the full corpus (~{FULL_CORPUS:,} papers), on this sample:")
    print(f"  Markdown output   {_human(per_paper * FULL_CORPUS)}")
    print(f"  PDF transferred   {_human(pdf_total / n * FULL_CORPUS)} (downloaded, then discarded)")
    if n < 100:
        print(f"  note: only {n} paper(s) sampled -- treat these figures as very rough")


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows):,} row(s) to {path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", type=Path)
    ap.add_argument("--csv", type=Path, help="also write the per-paper rows to CSV")
    ap.add_argument("--top", type=int, default=20, help="categories to show")
    args = ap.parse_args(argv)

    cfg = Config.load(args.config)
    if not cfg.paths.manifest_db.exists():
        print(f"no manifest at {cfg.paths.manifest_db}; run `prepare` first", file=sys.stderr)
        return 2

    with Manifest(cfg.paths.manifest_db) as m:
        rows = collect(m)
    report(rows, args.top)
    if args.csv and rows:
        write_csv(rows, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
