from __future__ import annotations

import json
import logging
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

from tqdm import tqdm

from .paths import shard_for
from .state import Manifest, PaperRow

log = logging.getLogger(__name__)


def iter_records(path: Path, *, progress: bool = True, limit: int | None = None) -> Iterator[dict]:
    total = path.stat().st_size
    bar = tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024,
        desc="reading snapshot", disable=not progress,
    )
    n = 0
    with path.open("r", encoding="utf-8") as fh, bar:
        for line in fh:
            bar.update(len(line.encode("utf-8", "ignore")))
            line = line.strip()

            if not line:
                continue

            try:
                yield json.loads(line)

            except json.JSONDecodeError:
                log.warning("skipping malformed JSON line at byte ~%d", bar.n)
                continue

            n += 1
            if limit is not None and n >= limit:
                return


def _version_of(versions: list[dict] | None) -> tuple[str, str | None, str | None]:
    """Return version number, first version date and the latest version date"""
    if not versions:
        return "v1", None, None

    def _num(v: dict) -> int:
        try:
            return int(str(v.get("version", "v1")).lstrip("v"))
        except ValueError:
            return 0

    ordered = sorted(versions, key=_num)

    def _date(v: dict) -> str | None:
        raw = v.get("created")
        if not raw:
            return None
        
        try:
            return parsedate_to_datetime(raw).date().isoformat()
        
        except (TypeError, ValueError):
            return None

    latest_version = ordered[-1].get("version", "v1")
    v1_date = _date(ordered[0])
    latest_date = _date(ordered[-1])

    return latest_version, v1_date, latest_date


def parse_record(rec: dict) -> dict[str, Any]:
    """Normalise one snapshot record into the fields required"""
    version, released, updated = _version_of(rec.get("versions"))
    categories = (rec.get("categories") or "").split()

    authors: list[str] = [] # Format: Lastname, firstname
    for parts in rec.get("authors_parsed") or []:
        name = ", ".join(p for p in parts[:2] if p)

        if len(parts) > 2 and parts[2]:
            name = f"{name} {parts[2]}"

        if name:
            authors.append(name)


    return {
        "id": rec.get("id", ""),
        "version": version,
        "title": " ".join((rec.get("title") or "").split()),
        "authors": authors,
        "authors_raw": rec.get("authors") or "",
        "categories": categories,
        "primary_category": categories[0] if categories else "",
        "doi": rec.get("doi"),
        "date_released": released,
        "date_updated": updated or rec.get("update_date"),
        "n_versions": len(rec.get("versions") or []),
    }


def _in_window(date: str | None, lo: str | None, hi: str | None) -> bool:
    """Compare an ISO date against inclusive ``YYYY-MM`` bounds."""
    if lo is None and hi is None:
        return True
    
    if not date:
        return False
    
    ym = date[:7]
    return (lo is None or ym >= lo) and (hi is None or ym <= hi)


def matches_scope(parsed: dict, scope: Any) -> bool:
    if scope.categories:
        wanted = set(scope.categories)
        have = {parsed["primary_category"]} if scope.primary_only else set(parsed["categories"])
        if not (wanted & have):
            return False
    return _in_window(parsed["date_released"], scope.date_from, scope.date_to)


def to_row(parsed: dict) -> PaperRow:
    return PaperRow(
        arxiv_id=parsed["id"],
        version=parsed["version"],
        shard=shard_for(parsed["id"]),
        title=parsed["title"],
        authors=json.dumps(parsed["authors"], ensure_ascii=False),
        categories=" ".join(parsed["categories"]),
        primary_category=parsed["primary_category"],
        doi=parsed["doi"],
        date_released=parsed["date_released"],
        date_updated=parsed["date_updated"],
    )


def prepare(cfg: Any, *, progress: bool = True) -> dict[str, int]:
    """Load the snapshot into the manifest, honouring scope"""
    metadata = cfg.paths.metadata_file
    if not metadata.exists():
        raise FileNotFoundError(
            f"metadata snapshot not found at {metadata}\n"
            "Download it with:\n"
            "  kaggle datasets download -d Cornell-University/arxiv "
            "-p data/metadata --unzip"
        )

    cfg.paths.ensure()
    counts = {"read": 0, "matched": 0, "inserted": 0}
    cap = cfg.scope.max_papers

    def _rows() -> Iterable[PaperRow]:
        for rec in iter_records(metadata, progress=progress):
            counts["read"] += 1
            parsed = parse_record(rec)
            if not parsed["id"] or not matches_scope(parsed, cfg.scope):
                continue
            counts["matched"] += 1
            yield to_row(parsed)
            if cap is not None and counts["matched"] >= cap:
                return

    with Manifest(cfg.paths.manifest_db) as m:
        counts["inserted"] = m.add_papers(_rows())
    return counts
