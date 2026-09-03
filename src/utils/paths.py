from __future__ import annotations

import re
from pathlib import Path

OLD_ID = re.compile(
    r"^(?P<archive>[a-z][a-z\-]*(?:\.[A-Za-z]{2})?)/(?P<yy>\d{2})(?P<mm>\d{2})(?P<num>\d{3})$"
)
NEW_ID = re.compile(r"^(?P<yy>\d{2})(?P<mm>\d{2})\.(?P<num>\d{4,5})$")

MISC_SHARD = "misc"


def safe_id(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_")


def shard_for(arxiv_id: str) -> str:
    """The ``yymm`` shard for an ID.

    Returns ``"misc"`` for anything unparseable
    """
    m = NEW_ID.match(arxiv_id) or OLD_ID.match(arxiv_id)

    if m is None:
        return MISC_SHARD
    
    mm = m.group("mm")

    if not ("01" <= mm <= "12"):
        return MISC_SHARD
    
    return f"{m.group('yy')}{mm}"


def md_path(data_dir: Path, arxiv_id: str) -> Path:
    return data_dir / "md" / shard_for(arxiv_id) / f"{safe_id(arxiv_id)}.md"


def tables_path(data_dir: Path, arxiv_id: str) -> Path:
    return data_dir / "tables" / shard_for(arxiv_id) / f"{safe_id(arxiv_id)}.tables.md"


def meta_path(data_dir: Path, arxiv_id: str) -> Path:
    return data_dir / "meta" / shard_for(arxiv_id) / f"{safe_id(arxiv_id)}.json"


def staged_pdf_path(data_dir: Path, arxiv_id: str) -> Path:
    """Transient location for a downloaded PDF, before conversion deletes it."""
    return data_dir / "tmp" / f"{safe_id(arxiv_id)}.pdf"


def tables_link(arxiv_id: str) -> str:
    """Relative link from a paper's ``md/`` file to its ``tables/`` file.

    Both live at ``<root>/<kind>/<shard>/<file>``, so the hop is always ``../../``.
    """
    return f"../../tables/{shard_for(arxiv_id)}/{safe_id(arxiv_id)}.tables.md"
