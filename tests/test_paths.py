from pathlib import Path

import pytest

from src.utils.paths import (
    MISC_SHARD, md_path, meta_path, safe_id, shard_for, staged_pdf_path,
    tables_link, tables_path,
)


@pytest.mark.parametrize(
    "arxiv_id,expected",
    [
        ("2301.12345", "2301"),      # new style, 5-digit
        ("0704.0001", "0704"),       # new style, 4-digit (the first ever)
        ("hep-th/9901001", "9901"),  # old style
        ("math.GT/0309136", "0309"), # old style with a subject class
        ("cond-mat/0512345", "0512"),
    ],
)
def test_shard_for_known_shapes(arxiv_id, expected):
    assert shard_for(arxiv_id) == expected


@pytest.mark.parametrize("bad", ["garbage", "", "2301", "2399.0001", "foo/12", "2313.0001"])
def test_shard_for_degrades_instead_of_raising(bad):
    """A malformed record must not be able to kill a multi-day run."""
    assert shard_for(bad) == MISC_SHARD


def test_safe_id_escapes_the_slash():
    assert safe_id("hep-th/9901001") == "hep-th_9901001"
    assert safe_id("2301.12345") == "2301.12345"


def test_output_paths_are_sharded_and_consistent():
    root = Path("/data")
    assert md_path(root, "hep-th/9901001") == root / "md/9901/hep-th_9901001.md"
    assert tables_path(root, "2301.12345") == root / "tables/2301/2301.12345.tables.md"
    assert meta_path(root, "2301.12345") == root / "meta/2301/2301.12345.json"
    assert staged_pdf_path(root, "2301.12345") == root / "tmp/2301.12345.pdf"


def test_tables_link_resolves_from_the_md_file():
    """The link is relative to md/<shard>/, so it must climb exactly two levels."""
    root = Path("/data")
    source = md_path(root, "2301.12345")
    resolved = (source.parent / tables_link("2301.12345")).resolve()
    assert resolved == tables_path(root, "2301.12345").resolve()
