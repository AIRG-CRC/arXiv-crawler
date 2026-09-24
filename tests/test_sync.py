"""Reading the bucket back into the manifest.

The properties worth protecting: a paper is only believed finished when both its md and
its meta are in the bucket; a row another device is holding `in_flight` is never touched;
`--dry-run` changes nothing; and the report distinguishes "already done here" from "not in
this manifest at all", because those mean completely different things.
"""

from __future__ import annotations

import json
import time
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
    assert json.loads(local.read_text())["sync"]["newly_marked"] == 7
    assert marker_name("dev-a", prefix="arxiv") == "arxiv/_state/sync/dev-a.json"


def test_a_progress_marker_carries_the_run(tmp_path):
    store, _ = _store()
    publish_marker(store, "dev-a", partition=(2, 0), state="running",
                   run={"done": 120, "slice_done": 120, "slice_total": 500,
                        "papers_per_min": 2.5})
    marker = read_markers(store)[0]
    assert marker["run"]["state"] == "running"
    assert marker["run"]["done"] == 120
    assert marker["updated_at"]
    assert "sync" not in marker           # a heartbeat says nothing about the last sync


def test_the_heartbeat_publishes_until_it_is_finished(tmp_path):
    from src.utils.sync import Heartbeat

    store, _ = _store()
    counter = {"n": 0}

    def snapshot():
        counter["n"] += 1
        return {"done": counter["n"]}

    hb = Heartbeat(store, "dev-a", snapshot=snapshot, interval=0.02, partition=(2, 0))
    hb.start()
    # Wait for the beats rather than sleeping a fixed time: a `sleep` long enough to be
    # reliable under load is a slow test, and one short enough to be quick is a flaky one.
    deadline = time.monotonic() + 5.0
    while counter["n"] < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert counter["n"] >= 2, "the heartbeat never beat twice"

    hb.finish("interrupted")
    assert not hb.is_alive()
    marker = read_markers(store)[0]
    assert marker["run"]["state"] == "interrupted"


def test_a_heartbeat_survives_a_broken_snapshot(tmp_path):
    from src.utils.sync import Heartbeat

    store, _ = _store()

    def snapshot():
        raise RuntimeError("counters moved")

    hb = Heartbeat(store, "dev-a", snapshot=snapshot, interval=0.05)
    hb.start()
    hb.finish()
    assert not hb.is_alive()
    assert read_markers(store) == []       # nothing published, nothing crashed


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


# --- the shared allocation ---------------------------------------------------------------
def test_a_plan_round_trips_and_validates():
    from src.utils.sync import plan_name, plan_partition, read_plan, write_plan

    store, _ = _store()
    assert read_plan(store) is None                  # absent is not an error

    plan = write_plan(store, 4, {"a": 0, "b": 1, "c": 2}, by="a")
    assert plan["devices"] == 4
    assert read_plan(store) == plan
    assert plan_name("arxiv") == "arxiv/_state/partition.json"
    assert plan_partition(plan, "b") == (4, 1)
    assert plan_partition(plan, "unknown") is None   # never guesses a slice
    assert plan_partition(None, "b") is None


@pytest.mark.parametrize("devices,assignments,fragment", [
    (0, {}, "at least 1"),
    (2, {"a": 5}, "does not exist"),
    (2, {"a": 0, "b": 0}, "both assigned"),
])
def test_an_impossible_plan_is_refused(devices, assignments, fragment):
    from src.utils.sync import write_plan

    store, _ = _store()
    with pytest.raises(ValueError, match=fragment):
        write_plan(store, devices, assignments)


def test_unparseable_plans_are_ignored_not_fatal():
    from src.utils.sync import plan_name, read_plan

    store, client = _store()
    client.objects[plan_name("arxiv")] = b"{not json"
    assert read_plan(store) is None
    client.objects[plan_name("arxiv")] = b'{"assignments": {"a": 0}}'
    assert read_plan(store) is None                  # no device count, so no plan


def test_changing_the_count_reallocates_every_device():
    """The point of the feature: one write moves the whole fleet."""
    from src.utils.sync import plan_partition, write_plan

    store, _ = _store()
    names = ["mac-studio", "linux-box", "CIT"]

    two = write_plan(store, 2, {"mac-studio": 0, "linux-box": 1})
    assert plan_partition(two, "mac-studio") == (2, 0)
    assert plan_partition(two, "CIT") is None

    three = write_plan(store, 3, {name: i for i, name in enumerate(names)})
    assert [plan_partition(three, n) for n in names] == [(3, 0), (3, 1), (3, 2)]
    # and the slices still cover everything, which is the property that matters
    from src.utils.partition import bucket_for

    owners = {n: {p for p in (f"2301.{i:05d}" for i in range(300))
                  if bucket_for(p) % 3 == plan_partition(three, n)[1]} for n in names}
    union = set().union(*owners.values())
    assert len(union) == 300
    assert sum(len(v) for v in owners.values()) == 300      # no overlap


def _marker(device, devices, index, *, hours_ago=0.0, state="finished"):
    when = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return {"device": device, "devices": devices, "device_index": index,
            "updated_at": when.isoformat(timespec="seconds"),
            "run": {"state": state}}


def test_a_machine_yet_to_restart_is_a_notice_not_a_problem():
    """Immediately after the count changes, every other marker disagrees. That is a
    rollout, and calling it a fault is what made the warning confusing."""
    plan = {"devices": 3, "assignments": {"a": 0, "b": 1, "c": 2}}
    said: list[str] = []
    problems = check_partition_agreement(
        [_marker("b", 2, 1, state="finished")], "a", (3, 0), plan=plan,
        announce=lambda msg, *args: said.append(msg % args))
    assert problems == []
    assert any("will pick up slice 2 of 3" in line for line in said)
    assert said[0].strip().startswith("·")


def test_a_machine_running_on_the_old_split_is_a_problem():
    plan = {"devices": 3, "assignments": {"a": 0, "b": 1, "c": 2}}
    problems = check_partition_agreement(
        [_marker("b", 2, 1, state="running")], "a", (3, 0), plan=plan,
        announce=lambda *a: None)
    assert len(problems) == 1
    assert "right now" in problems[0]


def test_an_unassigned_slice_is_a_problem():
    plan = {"devices": 4, "assignments": {"a": 0, "b": 1}}
    problems = check_partition_agreement([], "a", (4, 0), plan=plan, announce=lambda *a: None)
    assert len(problems) == 1
    assert "will not be crawled by anyone" in problems[0]


def test_a_device_outside_the_plan_is_flagged():
    plan = {"devices": 2, "assignments": {"a": 0, "b": 1}}
    said: list[str] = []
    check_partition_agreement([_marker("stray", 1, 0)], "a", (2, 0), plan=plan,
                              announce=lambda msg, *args: said.append(msg % args))
    assert any("not in the allocation" in line for line in said)


def test_without_a_plan_the_pairwise_check_still_applies():
    problems = check_partition_agreement(
        [_marker("b", 2, 0)], "a", (2, 0), announce=lambda *a: None)
    assert len(problems) == 1 and "device-index 0" in problems[0]


def test_scaling_down_reindexes_contiguously():
    """Dropping a machine must leave a valid allocation, not a hole where it was.

    Lowering the count alone cannot work: whoever held the top slice is left pointing at a
    slice that no longer exists, which `write_plan` refuses. Reindexing is what makes
    scaling down expressible at all.
    """
    from src.utils.sync import plan_partition, write_plan

    store, _ = _store()
    fleet = ["CIT", "gx10-df62", "linux-box", "mac-studio"]
    write_plan(store, 4, {name: i for i, name in enumerate(fleet)})

    # the naive lowering is refused, and for the right reason
    with pytest.raises(ValueError, match="does not exist"):
        write_plan(store, 3, {name: i for i, name in enumerate(fleet)})

    remaining = [n for n in fleet if n != "mac-studio"]
    three = write_plan(store, 3, {name: i for i, name in enumerate(remaining)})
    assert [plan_partition(three, n) for n in remaining] == [(3, 0), (3, 1), (3, 2)]
    assert plan_partition(three, "mac-studio") is None


def test_the_three_remaining_slices_still_cover_the_corpus():
    from src.utils.partition import bucket_for

    papers = [f"2301.{i:05d}" for i in range(600)]
    owners = [{p for p in papers if bucket_for(p) % 3 == i} for i in range(3)]
    assert set().union(*owners) == set(papers)
    assert sum(len(o) for o in owners) == len(papers)      # disjoint
