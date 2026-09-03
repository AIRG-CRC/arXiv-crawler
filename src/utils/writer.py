"""Serialise a conversion into the three on-disk artefacts.

Every file is written to a temporary name in its final directory and then `os.replace`d
into place. A crash mid-write therefore leaves either the old file or the new one, never
a half-written file that a resumed run would mistake for complete.

The tables file is written for retrieval, not for reading top to bottom: each table is
one `## Table N` section carrying its own provenance -- paper, arXiv id, caption,
columns -- so splitting the file on those headings yields chunks that still make sense
on their own, which a bare pipe table does not.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from .paths import md_path, meta_path, safe_id, tables_link, tables_path

if TYPE_CHECKING:  # avoids a converter <-> writer import cycle at runtime
    from .converter import ConversionResult, TableBlock
    from .state import PaperRow

_MARKER_RE = re.compile(r"\[\[TABLE:(\d+)\]\]")

# Exactly the fields the metadata JSON carries. Anything else stays in the manifest.
METADATA_FIELDS = (
    "id", "title", "authors", "date_released", "date_updated", "doi",
    "categories", "primary_category", "source_url",
    "n_pages", "n_tables", "n_chars", "md_path", "tables_path",
)


def atomic_write_text(path: Path, text: str) -> int:
    """Write `text` to `path` atomically. Returns the byte count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return len(data)


def _front_matter(fields: dict[str, Any]) -> str:
    body = yaml.safe_dump(
        {k: v for k, v in fields.items() if v not in (None, "", [])},
        sort_keys=False, allow_unicode=True, default_flow_style=False, width=100,
    )
    return f"---\n{body}---\n\n"


def _link_markers(body: str, arxiv_id: str, has_tables: bool) -> str:
    """Point each ``[[TABLE:n]]`` marker at its heading in the tables file."""
    if not has_tables:
        return body
    link = tables_link(arxiv_id)
    return _MARKER_RE.sub(
        lambda m: f"[[TABLE:{m.group(1)}]]({link}#table-{m.group(1)})", body
    )


def render_body(row: "PaperRow", result: "ConversionResult", converter: str) -> str:
    fm = _front_matter({
        "id": row.arxiv_id,
        "version": row.version,
        "title": row.title,
        "authors": row.author_list,
        "date_released": row.date_released,
        "date_updated": row.date_updated,
        "doi": row.doi,
        "categories": row.category_list,
        "primary_category": row.primary_category,
        "n_pages": result.n_pages,
        "n_tables": result.n_tables,
        "converter": converter,
    })
    note = (
        f"> Truncated: only the first {result.n_pages} pages were converted.\n\n"
        if result.truncated else ""
    )
    return fm + note + _link_markers(result.body_markdown, row.arxiv_id, bool(result.tables)) + "\n"


def render_table_block(row: "PaperRow", table: "TableBlock") -> str:
    """One retrieval-ready section for a single table.

    The provenance lines are the point. A pipe table on its own is close to meaningless
    once it has been split away from its paper -- an embedding of `| 91.2 | 0.88 |` is
    noise. Repeating the title, id and caption in every section costs a few hundred
    bytes and makes each chunk independently answerable.
    """
    label = "Algorithm" if table.kind == "pseudocode" else "Table"
    parts = [f"## Table {table.index}", ""]

    parts.append(f"**Paper:** {row.title} (arXiv:{row.arxiv_id})")
    if row.category_list:
        parts.append(f"**Categories:** {', '.join(row.category_list)}")
    if table.caption:
        parts.append(f"**Caption:** {table.caption}")

    where = f"page {table.page}, " if table.page else ""
    if table.kind == "pseudocode":
        parts.append(f"**Content:** {label} block ({where}{table.n_rows} lines)")
    else:
        parts.append(f"**Shape:** {where}{table.n_rows} rows x {table.n_cols} columns")
        if table.columns:
            parts.append(f"**Columns:** {', '.join(table.columns)}")

    parts += ["", table.markdown, ""]
    return "\n".join(parts)


def render_tables(row: "PaperRow", result: "ConversionResult") -> str:
    header = [
        f"# Tables — {row.title}",
        "",
        f"Extracted from arXiv:{row.arxiv_id}. "
        f"Each section below is self-contained and can be chunked on its `## Table` heading.",
        "",
    ]
    return "\n".join(header + [render_table_block(row, t) for t in result.tables])


def build_metadata(
    row: "PaperRow",
    result: "ConversionResult",
    *,
    base_url: str,
) -> dict[str, Any]:
    """The per-paper metadata record. Exactly `METADATA_FIELDS`, in that order."""
    record = {
        "id": row.arxiv_id,
        "title": row.title,
        "authors": row.author_list,
        "date_released": row.date_released,
        "date_updated": row.date_updated,
        "doi": row.doi,
        "categories": row.category_list,
        "primary_category": row.primary_category,
        "source_url": f"{base_url.rstrip('/')}/abs/{row.arxiv_id}{row.version}",
        "n_pages": result.n_pages,
        "n_tables": result.n_tables,
        "n_chars": result.n_chars,
        "md_path": str(Path("md") / row.shard / f"{safe_id(row.arxiv_id)}.md"),
        "tables_path": (
            str(Path("tables") / row.shard / f"{safe_id(row.arxiv_id)}.tables.md")
            if result.tables else None
        ),
    }
    assert tuple(record) == METADATA_FIELDS, "metadata field set drifted"
    return record


def write_outputs(
    data_dir: Path,
    row: "PaperRow",
    result: "ConversionResult",
    *,
    converter: str,
    base_url: str = "https://arxiv.org",
) -> tuple[int, int]:
    """Write the body, tables and metadata files. Returns ``(md_bytes, tables_bytes)``."""
    md_written = atomic_write_text(
        md_path(data_dir, row.arxiv_id), render_body(row, result, converter)
    )

    tables_written = 0
    tpath = tables_path(data_dir, row.arxiv_id)
    if result.tables:
        tables_written = atomic_write_text(tpath, render_tables(row, result))
    else:
        # A re-conversion that now finds no tables must not leave a stale file behind.
        tpath.unlink(missing_ok=True)

    atomic_write_text(
        meta_path(data_dir, row.arxiv_id),
        json.dumps(build_metadata(row, result, base_url=base_url),
                   ensure_ascii=False, indent=2) + "\n",
    )
    return md_written, tables_written
