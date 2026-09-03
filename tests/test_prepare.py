import json

import pytest

from src.config import Config
from src.utils.prepare_data import (
    _version_of, iter_records, matches_scope, parse_record, prepare, to_row,
)

RECORD = {
    "id": "0704.0001",
    "submitter": "Pavel Nadolsky",
    "authors": r"C. Bal\'azs, E. L. Berger",
    "title": "Calculation of prompt diphoton\n  production cross sections",
    "comments": "37 pages, 15 figures",
    "journal-ref": "Phys.Rev.D76:013009,2007",
    "doi": "10.1103/PhysRevD.76.013009",
    "categories": "hep-ph cs.LG",
    "license": None,
    "abstract": "  A fully differential\n calculation.  ",
    "versions": [
        {"version": "v1", "created": "Mon, 2 Apr 2007 19:18:42 GMT"},
        {"version": "v2", "created": "Tue, 24 Jul 2007 20:10:27 GMT"},
    ],
    "update_date": "2008-11-26",
    "authors_parsed": [["Balázs", "C.", ""], ["Berger", "E. L.", "Jr"]],
}


def test_parse_record_normalises_the_fields_we_keep():
    p = parse_record(RECORD)
    assert p["id"] == "0704.0001"
    assert p["title"] == "Calculation of prompt diphoton production cross sections"
    assert p["abstract"] == "A fully differential calculation."
    assert p["authors"] == ["Balázs, C.", "Berger, E. L. Jr"]
    assert p["categories"] == ["hep-ph", "cs.LG"]
    assert p["primary_category"] == "hep-ph"
    assert p["doi"] == "10.1103/PhysRevD.76.013009"
    assert p["journal_ref"] == "Phys.Rev.D76:013009,2007"


def test_version_pinning_picks_the_highest_not_the_last():
    """Downloads must be reproducible, so we pin vN rather than tracking /pdf/<id>."""
    versions = [
        {"version": "v10", "created": "Tue, 24 Jul 2007 20:10:27 GMT"},
        {"version": "v2", "created": "Mon, 2 Apr 2007 19:18:42 GMT"},
    ]
    latest, first_date, last_date = _version_of(versions)
    assert latest == "v10"                 # not "v2" -- string sort would get this wrong
    assert first_date == "2007-04-02"
    assert last_date == "2007-07-24"


def test_version_of_handles_missing_versions():
    assert _version_of(None) == ("v1", None, None)
    assert _version_of([]) == ("v1", None, None)


def test_dates_come_from_the_versions_array():
    p = parse_record(RECORD)
    assert p["date_released"] == "2007-04-02"   # v1
    assert p["date_updated"] == "2007-07-24"    # v2


class _Scope:
    def __init__(self, **kw):
        self.categories = kw.get("categories", [])
        self.primary_only = kw.get("primary_only", False)
        self.date_from = kw.get("date_from")
        self.date_to = kw.get("date_to")
        self.max_papers = kw.get("max_papers")


def test_category_filter_matches_any_category_by_default():
    p = parse_record(RECORD)
    assert matches_scope(p, _Scope(categories=["cs.LG"]))       # a cross-list counts
    assert matches_scope(p, _Scope(categories=["hep-ph"]))
    assert not matches_scope(p, _Scope(categories=["math.AG"]))


def test_primary_only_ignores_cross_lists():
    p = parse_record(RECORD)
    assert not matches_scope(p, _Scope(categories=["cs.LG"], primary_only=True))
    assert matches_scope(p, _Scope(categories=["hep-ph"], primary_only=True))


@pytest.mark.parametrize(
    "lo,hi,expected",
    [(None, None, True), ("2007-01", None, True), ("2007-05", None, False),
     (None, "2007-04", True), (None, "2007-03", False), ("2007-04", "2007-04", True)],
)
def test_date_window_is_inclusive_on_both_ends(lo, hi, expected):
    assert matches_scope(parse_record(RECORD), _Scope(date_from=lo, date_to=hi)) is expected


def test_to_row_serialises_for_sqlite():
    row = to_row(parse_record(RECORD))
    assert row.arxiv_id == "0704.0001" and row.version == "v2" and row.shard == "0704"
    assert row.categories == "hep-ph cs.LG"          # stored space-separated
    assert json.loads(row.authors) == ["Balázs, C.", "Berger, E. L. Jr"]
    assert row.author_list[0] == "Balázs, C."        # and round-trips back


def _snapshot(path, n=50):
    with path.open("w", encoding="utf-8") as fh:
        for i in range(n):
            rec = dict(RECORD)
            rec["id"] = f"23{(i % 12) + 1:02d}.{i:05d}"
            rec["categories"] = "cs.LG stat.ML" if i % 2 else "math.AG"
            fh.write(json.dumps(rec) + "\n")
        fh.write("\n")                 # blank lines are skipped
        fh.write("{not json}\n")       # and so is garbage, with a warning
    return path


def test_iter_records_survives_blank_and_malformed_lines(tmp_path):
    path = _snapshot(tmp_path / "snap.json", n=5)
    assert len(list(iter_records(path, progress=False))) == 5


def _config(tmp_path, snapshot, **scope):
    cfg = Config()
    cfg.paths.data_dir = tmp_path / "data"
    cfg.paths.metadata_file = snapshot
    for k, v in scope.items():
        setattr(cfg.scope, k, v)
    return cfg


def test_prepare_loads_and_filters(tmp_path):
    snap = _snapshot(tmp_path / "snap.json", n=50)
    cfg = _config(tmp_path, snap, categories=["cs.LG"])
    counts = prepare(cfg, progress=False)
    assert counts["read"] == 50
    assert counts["matched"] == 25 and counts["inserted"] == 25


def test_prepare_honours_max_papers(tmp_path):
    snap = _snapshot(tmp_path / "snap.json", n=50)
    counts = prepare(_config(tmp_path, snap, max_papers=7), progress=False)
    assert counts["matched"] == 7 and counts["inserted"] == 7


def test_prepare_is_idempotent_and_widening_adds_work(tmp_path):
    snap = _snapshot(tmp_path / "snap.json", n=50)
    narrow = _config(tmp_path, snap, categories=["cs.LG"])
    assert prepare(narrow, progress=False)["inserted"] == 25
    assert prepare(narrow, progress=False)["inserted"] == 0     # nothing new

    wide = _config(tmp_path, snap)
    assert prepare(wide, progress=False)["inserted"] == 25      # the other half


def test_prepare_reports_a_useful_error_when_the_snapshot_is_absent(tmp_path):
    cfg = _config(tmp_path, tmp_path / "nope.json")
    with pytest.raises(FileNotFoundError, match="kaggle datasets download"):
        prepare(cfg, progress=False)
