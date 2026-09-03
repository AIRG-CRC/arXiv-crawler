import json

import pytest
import yaml

from src.utils.converter import ConversionResult, TableBlock
from src.utils.paths import md_path, meta_path, tables_path
from src.utils.state import PaperRow
from src.utils.writer import (
    METADATA_FIELDS, atomic_write_text, build_metadata, render_body, render_tables,
    write_outputs,
)


@pytest.fixture()
def row():
    return PaperRow(
        arxiv_id="2301.12345", version="v2", shard="2301",
        title="A Study of Things",
        authors=json.dumps(["Doe, Jane", "Roe, Richard"]),
        categories="cs.LG stat.ML", primary_category="cs.LG",
        doi="10.1000/xyz", journal_ref="J. Things 1:2,2023",
        license="http://creativecommons.org/licenses/by/4.0/",
        date_released="2023-01-25", date_updated="2023-01-30",
        abstract="We study things.",
    )


@pytest.fixture()
def result():
    return ConversionResult(
        body_markdown="# Intro\n\nProse.\n\n[[TABLE:1]]\n\nMore prose.",
        tables=[TableBlock(
            index=1, page=4, n_rows=2, n_cols=3,
            markdown="| Parser | Acc | F1 |\n| --- | --- | --- |\n| Ours | 91.2 | 0.88 |",
            caption="Table 4: The Transformer generalizes well.",
            columns=["Parser", "Acc", "F1"],
        )],
        n_pages=12, n_chars=41822,
    )


# --- metadata: exactly the agreed field set ------------------------------------------
def test_metadata_contains_exactly_the_agreed_fields(row, result):
    meta = build_metadata(row, result, base_url="https://export.arxiv.org")
    assert tuple(meta) == METADATA_FIELDS


def test_metadata_values(row, result):
    meta = build_metadata(row, result, base_url="https://export.arxiv.org")
    assert meta["id"] == "2301.12345"
    assert meta["title"] == "A Study of Things"
    assert meta["authors"] == ["Doe, Jane", "Roe, Richard"]
    assert meta["date_released"] == "2023-01-25"
    assert meta["date_updated"] == "2023-01-30"
    assert meta["doi"] == "10.1000/xyz"
    assert meta["categories"] == ["cs.LG", "stat.ML"]
    assert meta["primary_category"] == "cs.LG"
    assert meta["source_url"] == "https://export.arxiv.org/abs/2301.12345v2"
    assert meta["n_pages"] == 12 and meta["n_tables"] == 1 and meta["n_chars"] == 41822
    assert meta["md_path"] == "md/2301/2301.12345.md"
    assert meta["tables_path"] == "tables/2301/2301.12345.tables.md"


@pytest.mark.parametrize("dropped", [
    "version", "journal_ref", "license", "abstract", "truncated", "low_text",
    "pdf_bytes", "pdf_sha256", "converter", "converted_at",
])
def test_dropped_fields_really_are_gone(row, result, dropped):
    assert dropped not in build_metadata(row, result, base_url="https://x")


def test_tables_path_is_null_when_there_are_no_tables(row):
    bare = ConversionResult(body_markdown="Prose only.", n_pages=3, n_chars=11)
    assert build_metadata(row, bare, base_url="https://x")["tables_path"] is None


# --- body ----------------------------------------------------------------------------
def test_front_matter_is_valid_yaml(row, result):
    _, fm, _ = render_body(row, result, converter="pymupdf").split("---\n", 2)
    meta = yaml.safe_load(fm)
    assert meta["id"] == "2301.12345"
    assert meta["authors"] == ["Doe, Jane", "Roe, Richard"]
    assert meta["categories"] == ["cs.LG", "stat.ML"]


def test_markers_link_to_the_tables_file(row, result):
    body = render_body(row, result, converter="pymupdf")
    assert "[[TABLE:1]](../../tables/2301/2301.12345.tables.md#table-1)" in body


def test_markers_stay_bare_when_there_are_no_tables(row):
    bare = ConversionResult(body_markdown="Prose only.", n_pages=3, n_chars=11)
    assert "](" not in render_body(row, bare, converter="pymupdf")


def test_truncation_is_announced(row, result):
    result.truncated = True
    assert "Truncated" in render_body(row, result, converter="pymupdf")


# --- tables file: each section must stand alone as a retrieval chunk -----------------
def test_each_table_section_carries_its_own_provenance(row, result):
    """A pipe table split away from its paper is close to meaningless -- an embedding
    of `| 91.2 | 0.88 |` is noise. Every section repeats what it needs."""
    section = render_tables(row, result).split("## Table 1", 1)[1]
    assert "A Study of Things" in section
    assert "arXiv:2301.12345" in section
    assert "cs.LG, stat.ML" in section
    assert "Table 4: The Transformer generalizes well." in section
    assert "Parser, Acc, F1" in section
    assert "| Ours | 91.2 | 0.88 |" in section


def test_table_headings_match_the_marker_anchors(row, result):
    body = render_body(row, result, converter="pymupdf")
    tables = render_tables(row, result)
    assert "## Table 1" in tables and "#table-1)" in body


def test_chunking_on_the_heading_yields_self_contained_sections(row):
    many = ConversionResult(
        body_markdown="[[TABLE:1]] [[TABLE:2]]",
        tables=[
            TableBlock(index=i, page=i, n_rows=1, n_cols=2, columns=["A", "B"],
                       caption=f"Table {i}: caption {i}",
                       markdown="| A | B |\n| --- | --- |\n| 1 | 2 |")
            for i in (1, 2)
        ],
        n_pages=5, n_chars=100,
    )
    sections = render_tables(row, many).split("\n## ")[1:]
    assert len(sections) == 2
    for section in sections:
        assert "arXiv:2301.12345" in section      # provenance survives the split
        assert "A Study of Things" in section


def test_pseudocode_section_is_labelled_as_an_algorithm(row):
    algo = ConversionResult(
        body_markdown="[[TABLE:1]]",
        tables=[TableBlock(
            index=1, page=3, n_rows=4, n_cols=1, kind="pseudocode",
            caption="Algorithm 1: Adam",
            markdown="```text\n1: Require: alpha\n2: while not converged do\n```",
        )],
        n_pages=9, n_chars=500,
    )
    section = render_tables(row, algo)
    assert "Algorithm block" in section
    assert "```text" in section
    assert "rows x" not in section        # shape is meaningless for an algorithm


# --- files ---------------------------------------------------------------------------
def test_write_outputs_produces_all_three_files(tmp_path, row, result):
    md_bytes, tables_bytes = write_outputs(
        tmp_path, row, result, converter="pymupdf", base_url="https://export.arxiv.org")
    assert md_path(tmp_path, "2301.12345").exists()
    assert tables_path(tmp_path, "2301.12345").exists()
    assert meta_path(tmp_path, "2301.12345").exists()
    assert md_bytes > 0 and tables_bytes > 0

    meta = json.loads(meta_path(tmp_path, "2301.12345").read_text())
    assert tuple(meta) == METADATA_FIELDS


def test_no_tables_file_when_there_are_none(tmp_path, row):
    bare = ConversionResult(body_markdown="Prose only.", n_pages=3, n_chars=11)
    _, tables_bytes = write_outputs(tmp_path, row, bare, converter="pymupdf")
    assert tables_bytes == 0
    assert not tables_path(tmp_path, "2301.12345").exists()


def test_reconversion_clears_a_stale_tables_file(tmp_path, row, result):
    write_outputs(tmp_path, row, result, converter="pymupdf")
    assert tables_path(tmp_path, "2301.12345").exists()
    bare = ConversionResult(body_markdown="Prose only.", n_pages=3, n_chars=11)
    write_outputs(tmp_path, row, bare, converter="pymupdf")
    assert not tables_path(tmp_path, "2301.12345").exists()


def test_atomic_write_leaves_no_partial_file(tmp_path):
    target = tmp_path / "out.md"
    atomic_write_text(target, "first")
    atomic_write_text(target, "second")
    assert target.read_text() == "second"
    assert not list(tmp_path.glob(".tmp-*"))
