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
from src.utils.partition import BUCKETS, Slice, bucket_for
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
    from src.utils.sync import SyncReport, marker_range

    store, _ = _store()
    local = tmp_path / "sync-state.json"
    publish_marker(store, "dev-a", SyncReport(newly_marked=7), partition=Slice.equal(2, 1),
                   local_path=local)
    markers = read_markers(store)
    assert len(markers) == 1
    assert markers[0]["device"] == "dev-a"
    assert marker_range(markers[0]) == Slice.equal(2, 1)
    assert json.loads(local.read_text())["sync"]["newly_marked"] == 7
    assert marker_name("dev-a", prefix="arxiv") == "arxiv/_state/sync/dev-a.json"


def test_an_old_style_marker_reads_as_its_equal_slice():
    from src.utils.sync import WHOLE, marker_range

    assert marker_range({"devices": 4, "device_index": 2}) == Slice.equal(4, 2)
    assert marker_range({}) == WHOLE
    assert marker_range({"range": "garbage"}) == WHOLE


def test_a_progress_marker_carries_the_run(tmp_path):
    store, _ = _store()
    publish_marker(store, "dev-a", partition=Slice.equal(2, 0), state="running",
                   run={"done": 120, "slice_done": 120, "slice_total": 500,
                        "papers_per_min": 2.5})
    marker = read_markers(store)[0]
    assert marker["run"]["state"] == "running"
    assert marker["run"]["done"] == 120
    assert marker["updated_at"]
    assert "sync" not in marker           # a heartbeat says nothing about the last sync


def _wait_for(predicate, timeout=5.0):
    # Wait for the beats rather than sleeping a fixed time: a `sleep` long enough to be
    # reliable under load is a slow test, and one short enough to be quick is a flaky one.
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    return predicate()


def test_the_heartbeat_publishes_until_it_is_finished(tmp_path):
    from src.utils.sync import Heartbeat

    store, _ = _store()
    counter = {"n": 0}

    def snapshot():
        counter["n"] += 1
        return {"done": counter["n"]}

    hb = Heartbeat(store, "dev-a", snapshot=snapshot, interval=0.02,
                   partition=Slice.equal(2, 0))
    hb.start()
    assert _wait_for(lambda: counter["n"] >= 2), "the heartbeat never beat twice"

    hb.finish("interrupted")
    assert not hb.is_alive()
    marker = read_markers(store)[0]
    assert marker["run"]["state"] == "interrupted"


def test_the_heartbeat_moves_a_running_crawl_to_its_new_share(tmp_path):
    """The live half of `devices --share`: no restart, one interval."""
    from src.utils.partition import LiveSlice
    from src.utils.sync import Heartbeat, marker_range, write_plan

    store, _ = _store()
    write_plan(store, weights={"dev-a": 1, "dev-b": 1}, order=["dev-a", "dev-b"])
    live = LiveSlice(Slice(0, 128))
    hb = Heartbeat(store, "dev-a", snapshot=dict, interval=0.02, live=live)
    hb.start()
    try:
        time.sleep(0.1)
        assert live.version == 0                       # unchanged plan, nothing to do
        write_plan(store, weights={"dev-a": 3, "dev-b": 1}, order=["dev-a", "dev-b"])
        assert _wait_for(lambda: live.version == 1)
        assert live.current == Slice(0, 192)
    finally:
        hb.finish()
    assert marker_range(read_markers(store)[0]) == Slice(0, 192)


def test_a_heartbeat_keeps_its_slice_when_the_plan_drops_it(tmp_path):
    from src.utils.partition import LiveSlice
    from src.utils.sync import Heartbeat, write_plan

    store, _ = _store()
    write_plan(store, weights={"dev-b": 1}, order=["dev-b"])
    live = LiveSlice(Slice(0, 128))
    hb = Heartbeat(store, "dev-a", snapshot=dict, interval=0.02, live=live)
    hb.start()
    time.sleep(0.1)
    hb.finish()
    assert live.current == Slice(0, 128) and live.version == 0


def test_the_heartbeat_flushes_the_journal(tmp_path):
    from src.utils.sync import DoneJournal, Heartbeat

    store, client = _store()
    journal = DoneJournal(store, "dev-a")
    journal.record("2301.00001")
    hb = Heartbeat(store, "dev-a", snapshot=dict, interval=0.02, journal=journal)
    hb.start()
    assert _wait_for(lambda: any("/_state/done/dev-a/" in n for n in client.objects))
    hb.finish()


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


def test_an_overlapping_slice_is_flagged():
    problems = check_partition_agreement(
        [_marker("dev-b", 2, 0)], "dev-a", Slice.equal(2, 0), announce=lambda *a: None)
    assert len(problems) == 1 and "overlaps" in problems[0]


def test_a_disagreement_about_the_device_count_is_flagged():
    problems = check_partition_agreement(
        [_marker("dev-b", 3, 0)], "dev-a", Slice.equal(2, 0), announce=lambda *a: None)
    assert len(problems) == 1 and "overlaps" in problems[0]


def test_a_complementary_slice_is_fine():
    assert not check_partition_agreement(
        [_marker("dev-b", 2, 1)], "dev-a", Slice.equal(2, 0), announce=lambda *a: None)


def test_a_stale_marker_is_not_evidence_of_anything():
    assert not check_partition_agreement(
        [_marker("dev-b", 2, 0, hours_ago=72)], "dev-a", Slice.equal(2, 0),
        announce=lambda *a: None)


def test_this_device_does_not_clash_with_itself():
    assert not check_partition_agreement(
        [_marker("dev-a", 2, 0)], "dev-a", Slice.equal(2, 0), announce=lambda *a: None)


# --- the shared allocation ---------------------------------------------------------------
def test_a_plan_round_trips_and_validates():
    from src.utils.sync import plan_name, plan_partition, read_plan, write_plan

    store, _ = _store()
    assert read_plan(store) is None                  # absent is not an error

    plan = write_plan(store, weights={"a": 1, "b": 1, "c": 2}, order=["a", "b", "c"],
                      by="a")
    assert plan["version"] == 2
    assert read_plan(store) == plan
    assert plan_name("arxiv") == "arxiv/_state/partition.json"
    assert plan_partition(plan, "a") == Slice(0, 64)
    assert plan_partition(plan, "c") == Slice(128, 256)
    assert plan_partition(plan, "unknown") is None   # never guesses a slice
    assert plan_partition(None, "b") is None


def test_a_version_one_plan_still_reads_as_its_equal_split():
    from src.utils.sync import plan_partition

    plan = {"devices": 4, "assignments": {"a": 0, "b": 3}}
    assert plan_partition(plan, "b") == Slice.equal(4, 3)


@pytest.mark.parametrize("weights,fragment", [
    ({"a": 0}, "above 0"),
    ({"a": -1, "b": 1}, "negative"),
])
def test_an_impossible_plan_is_refused(weights, fragment):
    from src.utils.sync import write_plan

    store, _ = _store()
    with pytest.raises(ValueError, match=fragment):
        write_plan(store, weights=weights, order=list(weights))


def test_overlapping_ranges_are_refused():
    from src.utils.sync import write_plan

    store, _ = _store()
    with pytest.raises(ValueError, match="overlap"):
        write_plan(store, ranges={"a": Slice(0, 130), "b": Slice(128, 256)})


def test_unparseable_plans_are_ignored_not_fatal():
    from src.utils.sync import plan_name, read_plan

    store, client = _store()
    client.objects[plan_name("arxiv")] = b"{not json"
    assert read_plan(store) is None
    client.objects[plan_name("arxiv")] = b'{"assignments": {"a": 0}}'
    assert read_plan(store) is None                  # no device count, so no plan


# --- shares ------------------------------------------------------------------------------
def test_uneven_shares_cover_the_corpus_exactly_once():
    from src.utils.partition import ranges_from_weights

    ranges = ranges_from_weights(["DGX-1", "DGX-2"], {"DGX-1": 60, "DGX-2": 40})
    assert ranges == {"DGX-1": Slice(0, 154), "DGX-2": Slice(154, 256)}
    papers = [f"2301.{i:05d}" for i in range(2000)]
    owners = {n: {p for p in papers if r.contains(bucket_for(p))} for n, r in ranges.items()}
    assert set().union(*owners.values()) == set(papers)
    assert sum(len(v) for v in owners.values()) == len(papers)      # disjoint


def test_a_zero_share_parks_a_machine_and_a_tiny_one_still_gets_a_bucket():
    from src.utils.partition import ranges_from_weights

    ranges = ranges_from_weights(["a", "b", "c"], {"a": 0, "b": 1000, "c": 0.001})
    assert ranges["a"].empty
    assert ranges["c"].hi - ranges["c"].lo == 1
    assert sum(r.hi - r.lo for r in ranges.values()) == BUCKETS


def test_rebalancing_two_machines_hands_over_only_the_difference():
    """50/50 -> 60/40 moves 10% of the corpus, not most of it as `bucket % n` would."""
    from src.utils.partition import ranges_from_weights

    order = ["DGX-1", "DGX-2"]
    before = ranges_from_weights(order, {"DGX-1": 1, "DGX-2": 1})
    after = ranges_from_weights(order, {"DGX-1": 60, "DGX-2": 40})
    moved = [b for b in range(BUCKETS)
             if before["DGX-1"].contains(b) != after["DGX-1"].contains(b)]
    assert moved == list(range(128, 154))


def test_merge_shares_edits_one_machine_and_keeps_the_rest():
    from src.utils.sync import merge_shares, plan_weights

    existing = {"version": 2, "ranges": {"DGX-1": [0, 128], "DGX-2": [128, 256]},
                "weights": {"DGX-1": 1, "DGX-2": 1}}
    weights, order = merge_shares(existing, {"DGX-1": 3})
    assert weights == {"DGX-1": 3.0, "DGX-2": 1.0}
    assert order == ["DGX-1", "DGX-2"]

    weights, order = merge_shares(existing, {"DGX-3": 1})
    assert order == ["DGX-1", "DGX-2", "DGX-3"]      # a newcomer goes on the end

    weights, order = merge_shares(existing, drop=["DGX-1"])
    assert weights == {"DGX-2": 1.0} and order == ["DGX-2"]

    weights, order = merge_shares(existing, {"DGX-2": 2}, equal=["DGX-1", "DGX-2", "x"])
    assert weights == {"DGX-1": 1.0, "DGX-2": 2.0, "x": 1.0}

    # a plan without stored weights reads as its widths
    assert plan_weights({"ranges": {"a": [0, 64], "b": [64, 256]}}) == {"a": 64.0, "b": 192.0}

    with pytest.raises(ValueError, match="negative"):
        merge_shares(existing, {"DGX-1": -1})


def _marker(device, devices, index, *, hours_ago=0.0, state="finished"):
    when = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return {"device": device, "range": Slice.equal(devices, index).to_json(),
            "updated_at": when.isoformat(timespec="seconds"),
            "run": {"state": state}}


THREE = {"version": 2, "ranges": {"a": Slice.equal(3, 0).to_json(),
                                  "b": Slice.equal(3, 1).to_json(),
                                  "c": Slice.equal(3, 2).to_json()}}


def test_a_machine_yet_to_restart_is_a_notice_not_a_problem():
    """Immediately after a share changes, the other markers disagree. That is a rollout,
    and calling it a fault is what made the warning confusing."""
    said: list[str] = []
    problems = check_partition_agreement(
        [_marker("b", 2, 1, state="finished")], "a", Slice.equal(3, 0), plan=THREE,
        announce=lambda msg, *args: said.append(msg % args))
    assert problems == []
    assert any("will pick up buckets 85-169" in line for line in said)
    assert said[0].strip().startswith("·")


def test_a_machine_running_on_the_old_split_is_a_problem():
    problems = check_partition_agreement(
        [_marker("b", 2, 1, state="running")], "a", Slice.equal(3, 0), plan=THREE,
        announce=lambda *a: None)
    assert len(problems) == 1
    assert "right now" in problems[0]


def test_an_unassigned_share_is_a_problem():
    plan = {"version": 2, "ranges": {"a": [0, 64], "b": [64, 128]}}
    problems = check_partition_agreement([], "a", Slice(0, 64), plan=plan,
                                         announce=lambda *a: None)
    assert len(problems) == 1
    assert "will not be crawled by anyone" in problems[0]


def test_a_device_outside_the_plan_is_flagged():
    plan = {"version": 2, "ranges": {"a": [0, 128], "b": [128, 256]}}
    said: list[str] = []
    check_partition_agreement([_marker("stray", 1, 0)], "a", Slice(0, 128), plan=plan,
                              announce=lambda msg, *args: said.append(msg % args))
    assert any("not in the allocation" in line for line in said)


def test_without_a_plan_the_pairwise_check_still_applies():
    problems = check_partition_agreement(
        [_marker("b", 2, 0)], "a", Slice.equal(2, 0), announce=lambda *a: None)
    assert len(problems) == 1 and "overlaps" in problems[0]


def test_forgetting_a_device_deletes_its_marker():
    """`--unassign` parks a machine; `--forget` retires it.

    The distinction is load-bearing: `--auto` rebuilds the fleet from the markers, so while
    a retired machine's marker survives, every `--auto` resurrects the machine that was just
    removed — which is a loop you cannot escape with `--unassign` alone.
    """
    from src.utils.sync import forget_device, marker_name, read_markers

    store, client = _store()
    for device in ("keep", "retire"):
        publish_marker(store, device, partition=Slice.equal(2, 0), run={"done": 1})
    assert {m["device"] for m in read_markers(store)} == {"keep", "retire"}

    assert forget_device(store, "retire") is True
    assert {m["device"] for m in read_markers(store)} == {"keep"}
    assert marker_name("retire", prefix="arxiv") not in client.objects

    # retiring twice is not an error -- it is the same end state
    assert forget_device(store, "retire") is False


def test_a_long_dead_device_outside_the_plan_stops_nagging():
    """A marker nobody refreshes should not produce a notice for ever."""
    plan = {"devices": 2, "assignments": {"a": 0, "b": 1}}
    said: list[str] = []
    check_partition_agreement([_marker("ghost", 1, 0, hours_ago=72)], "a", Slice.equal(2, 0),
                              plan=plan, announce=lambda msg, *a: said.append(msg % a))
    assert said == []

    check_partition_agreement([_marker("ghost", 1, 0, hours_ago=1)], "a", Slice.equal(2, 0),
                              plan=plan, announce=lambda msg, *a: said.append(msg % a))
    assert any("not in the allocation" in line for line in said)
    assert any("--forget ghost" in line for line in said)


def test_remove_object_reports_whether_it_was_there():
    store, client = _store(["arxiv/md/2301/a.md"])
    assert store.remove_object("arxiv/md/2301/a.md") is True
    assert store.remove_object("arxiv/md/2301/a.md") is False
    assert "arxiv/md/2301/a.md" not in client.objects


# --- the done-journal --------------------------------------------------------------------
def _done(m, ids, *, remote_only=0, when="2026-01-01T00:00:00+00:00"):
    m.conn.executemany(
        "UPDATE papers SET status = ?, remote_only = ?, completed_at = ? WHERE arxiv_id = ?",
        [(DONE, remote_only, when, i) for i in ids])
    m.conn.commit()


def test_journal_ids_snapshot_everything_then_only_own_new_work(tmp_path):
    m = _manifest(tmp_path)
    _done(m, IDS[:2])
    _done(m, IDS[2:3], remote_only=1, when="2026-06-01T00:00:00+00:00")
    _done(m, IDS[3:], when="2026-06-01T00:00:00+00:00")
    assert sorted(m.journal_ids(None)) == sorted(IDS)                # first snapshot: all
    # later: only this device's own conversions, never echoing another device's
    assert m.journal_ids("2026-03-01T00:00:00+00:00") == [IDS[3]]
    m.close()


def test_one_device_journal_replaces_the_scan_for_another(tmp_path):
    """The 8-hour start, gone: device B learns A's work from A's journal alone."""
    from src.utils.sync import DoneJournal, pull_journal

    store, client = _store()
    a = Manifest(tmp_path / "a.db")
    a.add_papers([PaperRow(arxiv_id=i, version="v1", shard="2301") for i in IDS])
    _done(a, IDS[:2])
    ja = DoneJournal(store, "dev-a", tmp_path / "a.json")
    assert ja.catch_up(a) == 2                       # the snapshot
    assert ja.catch_up(a) == 0                       # nothing new since
    ja.record(IDS[2])
    assert ja.flush() == 1                           # as-you-go

    (tmp_path / "b").mkdir()
    b = _manifest(tmp_path / "b")
    jb = DoneJournal(store, "dev-b", tmp_path / "b.json")
    first = pull_journal(b, store, jb)
    assert first.mode == "journal"
    assert first.segments_read == 2 and first.newly_marked == 3
    assert b.stats()[DONE] == 3
    assert not any(n.startswith("arxiv/md/") for n in client.listings if n)   # no scan

    again = pull_journal(b, store, DoneJournal(store, "dev-b", tmp_path / "b.json"))
    assert again.segments_read == 0                  # the cursor survived the restart
    a.close()
    b.close()


def test_a_torn_segment_is_skipped_not_fatal(tmp_path):
    from src.utils.sync import DoneJournal, pull_journal

    store, client = _store()
    client.objects["arxiv/_state/done/dev-a/0000000000.txt.gz"] = b"not gzip"
    m = _manifest(tmp_path)
    report = pull_journal(m, store, DoneJournal(store, "dev-b"))
    assert report.segments_read == 1 and report.newly_marked == 0
    m.close()


def test_a_lost_state_file_never_reuses_a_segment_number(tmp_path):
    from src.utils.sync import DoneJournal

    store, client = _store()
    DoneJournal(store, "dev-a").publish(["x"])
    DoneJournal(store, "dev-a").publish(["y"])       # no state file either time
    names = sorted(n for n in client.objects if "/_state/done/dev-a/" in n)
    assert [n.rsplit("/", 1)[1] for n in names] == ["0000000000.txt.gz", "0000000001.txt.gz"]


def test_run_sync_defaults_to_the_journal(tmp_path):
    from src.config import Config, Paths
    from src.utils.sync import run_sync

    cfg = Config()
    cfg.paths = Paths(data_dir=tmp_path)
    cfg.sync.device = "dev-b"
    store, client = _store(_objects(IDS[:1]))
    m = _manifest(tmp_path)
    report, journal = run_sync(cfg, m, store, announce=lambda *a: None)
    assert report.mode == "journal" and journal.device == "dev-b"
    assert m.stats().get(DONE, 0) == 0               # the bucket was not listed

    report, _ = run_sync(cfg, m, store, full=True, announce=lambda *a: None)
    assert report.mode == "full" and report.newly_marked == 1
    m.close()
