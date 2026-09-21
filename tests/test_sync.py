"""Reading the bucket back into the manifest.

The properties worth protecting: a paper is only believed finished when both its md and
its meta are in the bucket; a row another device is holding `in_flight` is never touched;
`--dry-run` changes nothing; and the report distinguishes "already done here" from "not in
this manifest at all", because those mean completely different things.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from src.utils.objectstore import (
    MinioSettings,
    MinioStore,
    id_candidates_from_object_name,
    id_from_object_name,
    object_name,
)
from src.utils.state import DONE, IN_FLIGHT, NO_PDF, PENDING, Manifest, PaperRow
from src.utils.sync import (
    check_partition_agreement,
    device_name,
    marker_name,
    publish_marker,
    read_markers,
    sync_from_bucket,
)
from tests.test_objectstore import FakeClient

IDS = ["2301.00001", "2301.00002", "2301.00003", "hep-th/9901001"]


def _store(names=(), prefix="arxiv"):
    client = FakeClient()
    for name in names:
        client.objects[name] = b"x" * 10
    settings = MinioSettings(endpoint="host:9000", bucket="airg", prefix=prefix,
                             access_key="k", secret_key="s")
    return MinioStore(settings, client=client), client


def _manifest(tmp_path, ids=IDS):
    m = Manifest(tmp_path / "m.db")
    m.add_papers([
        PaperRow(arxiv_id=i, version="v1", shard="9901" if "/" in i else "2301")
        for i in ids
    ])
    return m


def _objects(ids, *, prefix="arxiv", kinds=("md", "meta")):
    return [object_name(kind, i, prefix=prefix) for i in ids for kind in kinds]


# --- names ----------------------------------------------------------------------------
@pytest.mark.parametrize("arxiv_id", [
    "2301.00001", "0708.1102", "1706.03762", "hep-th/9901001", "cond-mat/0703023",
    "math.GT/0309136",
])
@pytest.mark.parametrize("prefix", ["", "arxiv", "corpus/arxiv"])
def test_object_names_round_trip(arxiv_id, prefix):
    name = object_name("md", arxiv_id, prefix=prefix)
    assert id_from_object_name(name, prefix=prefix) == arxiv_id


def test_tables_objects_are_not_mistaken_for_markdown():
    name = object_name("tables", "2301.00001", prefix="arxiv")
    assert name.endswith(".md")                      # the trap
    assert id_candidates_from_object_name(name, kind="md") == ()
    assert id_from_object_name(name, prefix="arxiv") is None


@pytest.mark.parametrize("name", [
    "", "nonsense", "arxiv/md/2301", "arxiv/meta/2301/2301.00001.json",
    "arxiv/md/2301/2301.00001.md.bak", "arxiv/md/2301/.md",
])
def test_names_that_are_not_markdown_artefacts(name):
    assert id_from_object_name(name, prefix="arxiv") is None


def test_both_readings_are_offered_for_an_underscored_stem():
    assert id_candidates_from_object_name("arxiv/md/9901/hep-th_9901001.md", kind="md") == (
        "hep-th_9901001", "hep-th/9901001")


# --- listing ---------------------------------------------------------------------------
def test_iter_objects_joins_the_prefix_with_a_trailing_slash():
    store, client = _store(["arxiv/md/2301/a.md", "arxiv/mdx/2301/b.md"])
    names = [n for n, _s, _m in store.iter_objects("md/2301")]
    assert names == ["arxiv/md/2301/a.md"]           # "mdx" must not match "md"
    assert client.listings[-1] == "arxiv/md/2301/"


def test_iter_objects_reports_size_and_time():
    store, client = _store(["arxiv/md/2301/a.md"])
    when = datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc)
    client.modified["arxiv/md/2301/a.md"] = when
    assert list(store.iter_objects("md")) == [("arxiv/md/2301/a.md", 10, when)]


def test_bucket_present_does_not_create_the_bucket():
    store, client = _store()
    assert store.bucket_present() is False
    assert client.buckets == set()


# --- the sync itself ---------------------------------------------------------------------
def test_marks_papers_done_from_the_bucket(tmp_path):
    store, client = _store(_objects(IDS))
    when = datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc)
    for name in client.objects:
        client.modified[name] = when

    with _manifest(tmp_path) as m:
        report = sync_from_bucket(m, store)
        assert report.newly_marked == len(IDS)
        assert m.stats()[DONE] == len(IDS)
        row = m.conn.execute(
            "SELECT * FROM papers WHERE arxiv_id = ?", ("hep-th/9901001",)).fetchone()
        assert row["remote_only"] == 1
        assert row["md_bytes"] == 10
        # The object's own timestamp, not the moment the sync happened to run.
        assert row["completed_at"] == "2024-05-01T12:00:00+00:00"


def test_md_without_meta_is_reported_but_not_marked(tmp_path):
    """An upload killed between the md and the meta must stay outstanding for someone."""
    names = _objects(IDS[:2]) + _objects(IDS[2:], kinds=("md",))
    store, _ = _store(names)
    with _manifest(tmp_path) as m:
        report = sync_from_bucket(m, store)
        assert report.newly_marked == 2
        assert report.md_without_meta == 2
        assert m.stats()[PENDING] == 2


def test_require_meta_can_be_switched_off(tmp_path):
    store, _ = _store(_objects(IDS, kinds=("md",)))
    with _manifest(tmp_path) as m:
        report = sync_from_bucket(m, store, require_meta=False)
        assert report.newly_marked == len(IDS)
        assert report.md_without_meta == 0


def test_dry_run_changes_nothing(tmp_path):
    store, _ = _store(_objects(IDS))
    with _manifest(tmp_path) as m:
        report = sync_from_bucket(m, store, dry_run=True)
        assert report.newly_marked == len(IDS)
        assert m.stats().get(DONE, 0) == 0
        assert m.stats()[PENDING] == len(IDS)


def test_in_flight_rows_are_left_alone(tmp_path):
    """A concurrent local run owns these; the writer would overwrite the sync anyway."""
    store, _ = _store(_objects(IDS))
    with _manifest(tmp_path) as m:
        m.conn.execute("UPDATE papers SET status = ? WHERE arxiv_id = ?",
                       (IN_FLIGHT, "2301.00001"))
        m.conn.commit()
        report = sync_from_bucket(m, store)
        assert report.in_flight_skipped == 1
        assert report.newly_marked == len(IDS) - 1
        status = m.conn.execute(
            "SELECT status FROM papers WHERE arxiv_id = ?", ("2301.00001",)).fetchone()[0]
        assert status == IN_FLIGHT


def test_a_no_pdf_paper_converted_elsewhere_is_recovered(tmp_path):
    store, _ = _store(_objects(IDS))
    with _manifest(tmp_path) as m:
        m.conn.execute("UPDATE papers SET status = ? WHERE arxiv_id = ?",
                       (NO_PDF, "2301.00002"))
        m.conn.commit()
        report = sync_from_bucket(m, store)
        assert report.no_pdf_recovered == 1
        assert m.stats()[DONE] == len(IDS)


def test_already_done_papers_are_counted_separately_from_absent_ones(tmp_path):
    store, _ = _store(_objects(IDS + ["2301.99999"]))
    # The extra paper shares a shard with the others, so it is actually listed: an
    # object in a shard this manifest has no rows in is never reached at all.
    with _manifest(tmp_path) as m:
        m.conn.execute("UPDATE papers SET status = ? WHERE arxiv_id = ?",
                       (DONE, "2301.00001"))
        m.conn.commit()
        report = sync_from_bucket(m, store)
        assert report.already_done == 1
        assert report.absent_from_manifest == 1
        assert report.newly_marked == len(IDS) - 1


def test_unrecognised_names_are_counted_not_crashed_on(tmp_path):
    store, client = _store(_objects(IDS))
    client.objects["arxiv/md/2301/README"] = b"notes"
    with _manifest(tmp_path) as m:
        report = sync_from_bucket(m, store)
        assert report.unrecognised == 1
        assert report.newly_marked == len(IDS)


def test_completed_shards_are_not_listed_at_all(tmp_path):
    store, client = _store(_objects(IDS))
    with _manifest(tmp_path) as m:
        m.conn.execute("UPDATE papers SET status = ? WHERE shard = ?", (DONE, "9901"))
        m.conn.commit()
        client.listings.clear()
        report = sync_from_bucket(m, store)
        assert report.shards_skipped == 1
        assert not any("9901" in (p or "") for p in client.listings)


def test_a_second_sync_marks_nothing_new(tmp_path):
    store, _ = _store(_objects(IDS))
    with _manifest(tmp_path) as m:
        sync_from_bucket(m, store)
        again = sync_from_bucket(m, store)
        assert again.newly_marked == 0
        assert again.shards_scanned == 0         # every shard is complete now


def test_an_empty_bucket_is_a_no_op(tmp_path):
    store, _ = _store()
    with _manifest(tmp_path) as m:
        report = sync_from_bucket(m, store)
        assert report.newly_marked == 0
        assert m.stats()[PENDING] == len(IDS)


# --- device markers ----------------------------------------------------------------------
def test_device_name_prefers_the_environment(monkeypatch):
    monkeypatch.setenv("ARXIV_CRAWLER_DEVICE", "mac studio/01")
    assert device_name(None) == "mac-studio-01"      # sanitised for a key


def test_device_name_falls_back_to_config_then_hostname(monkeypatch):
    monkeypatch.delenv("ARXIV_CRAWLER_DEVICE", raising=False)

    class Cfg:
        device = "linux-box"

    assert device_name(Cfg()) == "linux-box"
    assert device_name(None)                          # the hostname, whatever it is


def test_marker_round_trips_through_the_bucket(tmp_path):
    from src.utils.sync import SyncReport

    store, _ = _store()
    local = tmp_path / "sync-state.json"
    publish_marker(store, "dev-a", SyncReport(newly_marked=7), partition=(2, 1),
                   local_path=local)
    markers = read_markers(store)
    assert len(markers) == 1
    assert markers[0]["device"] == "dev-a"
    assert (markers[0]["devices"], markers[0]["device_index"]) == (2, 1)
    assert json.loads(local.read_text())["report"]["newly_marked"] == 7
    assert marker_name("dev-a", prefix="arxiv") == "arxiv/_state/sync/dev-a.json"


def test_an_unwritable_marker_is_not_fatal(tmp_path):
    from src.utils.sync import SyncReport

    store, client = _store()
    client.fail_on.add(marker_name("dev-a", prefix="arxiv"))
    publish_marker(store, "dev-a", SyncReport(), local_path=tmp_path / "s.json")


def _marker(device, devices, index, *, hours_ago=0.0):
    when = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return {"device": device, "devices": devices, "device_index": index,
            "synced_at": when.isoformat(timespec="seconds")}


def test_a_shared_device_index_is_flagged():
    problems = check_partition_agreement(
        [_marker("dev-b", 2, 0)], "dev-a", (2, 0), announce=lambda *a: None)
    assert len(problems) == 1 and "device-index 0" in problems[0]


def test_a_disagreement_about_the_device_count_is_flagged():
    problems = check_partition_agreement(
        [_marker("dev-b", 3, 1)], "dev-a", (2, 0), announce=lambda *a: None)
    assert len(problems) == 1 and "do not cover" in problems[0]


def test_a_complementary_slice_is_fine():
    assert not check_partition_agreement(
        [_marker("dev-b", 2, 1)], "dev-a", (2, 0), announce=lambda *a: None)


def test_a_stale_marker_is_not_evidence_of_anything():
    assert not check_partition_agreement(
        [_marker("dev-b", 2, 0, hours_ago=72)], "dev-a", (2, 0), announce=lambda *a: None)


def test_this_device_does_not_clash_with_itself():
    assert not check_partition_agreement(
        [_marker("dev-a", 2, 0)], "dev-a", (2, 0), announce=lambda *a: None)
