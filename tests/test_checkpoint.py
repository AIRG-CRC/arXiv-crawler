import json
import threading

import pytest

from src.utils.checkpoint import (
    RUN_FILE, SUMMARY_FILE, CheckpointStore, WorkerState,
)


@pytest.fixture()
def store(tmp_path):
    return CheckpointStore(tmp_path / "checkpoints")


def test_begin_on_a_clean_directory_starts_from_zero(store):
    carried = store.begin(target=100)
    assert carried.processed == 0 and carried.workers == 0
    assert (store.root / RUN_FILE).exists()


def test_each_bump_increments_that_worker_by_exactly_one(store):
    store.begin(target=10)
    for _ in range(3):
        store.bump(pid=111, status="done", arxiv_id="2301.00001")
    state = store.bump(pid=111, status="done", arxiv_id="2301.00002")
    assert state.processed == 4 and state.done == 4
    assert store.totals().processed == 4


def test_workers_get_separate_files_and_never_share_a_counter(store):
    store.begin(target=10)
    for pid in (101, 102, 103):
        store.bump(pid=pid, status="done", arxiv_id=f"2301.{pid}")
    store.bump(pid=101, status="failed_convert", arxiv_id="2301.999")

    files = sorted(p.name for p in store.root.glob("worker-*.json"))
    assert files == ["worker-00.json", "worker-01.json", "worker-02.json"]

    totals = store.totals()
    assert totals.processed == 4 and totals.done == 3 and totals.failed == 1
    assert totals.workers == 3


def test_outcomes_are_counted_separately(store):
    store.begin(target=10)
    store.bump(pid=1, status="done", arxiv_id="a")
    store.bump(pid=1, status="no_pdf", arxiv_id="b")
    store.bump(pid=1, status="failed_download", arxiv_id="c")
    t = store.totals()
    assert (t.processed, t.done, t.no_pdf, t.failed) == (3, 1, 1, 1)


def test_checkpoint_files_are_valid_json_and_readable(store):
    store.begin(target=10)
    store.bump(pid=7, status="done", arxiv_id="2301.00007")
    data = json.loads((store.root / "worker-00.json").read_text())
    assert data["processed"] == 1 and data["last_paper"] == "2301.00007"
    assert data["slot"] == 0 and data["pid"] == 7


def test_an_interrupted_run_is_resumed_not_restarted(tmp_path):
    """The whole point: a resumed run continues the count instead of going back to 0."""
    root = tmp_path / "checkpoints"
    first = CheckpointStore(root)
    first.begin(target=100)
    for i in range(12):
        first.bump(pid=100 + (i % 3), status="done", arxiv_id=f"2301.{i:05d}")
    # ...process is killed here; no finalize() call.

    resumed = CheckpointStore(root)
    carried = resumed.begin(target=88)
    assert carried.processed == 12
    assert carried.workers == 3

    resumed.bump(pid=100, status="done", arxiv_id="2301.00099")
    assert resumed.totals().processed == 13      # continues, does not restart


def test_finalize_collapses_to_one_summary_and_clears_the_debris(store):
    store.begin(target=3)
    for pid in (1, 2):
        store.bump(pid=pid, status="done", arxiv_id="x")
    store.finalize({"processed": 2, "done": 2})

    assert not list(store.root.glob("worker-*.json"))
    assert not (store.root / RUN_FILE).exists()
    summary = json.loads((store.root / SUMMARY_FILE).read_text())
    assert summary["totals"]["processed"] == 2
    assert len(summary["per_worker"]) == 2


def test_a_finished_run_is_history_not_carried_progress(store):
    store.begin(target=3)
    store.bump(pid=1, status="done", arxiv_id="x")
    store.finalize()

    fresh = CheckpointStore(store.root).begin(target=5)
    assert fresh.processed == 0        # a completed run must not seed the next bar


def test_a_torn_checkpoint_costs_one_tally_not_the_run(store):
    store.begin(target=10)
    store.bump(pid=1, status="done", arxiv_id="a")
    store.bump(pid=2, status="done", arxiv_id="b")
    (store.root / "worker-00.json").write_text("{ this is not json")

    recovered = CheckpointStore(store.root).begin(target=10)
    assert recovered.processed == 1     # the intact worker's tally survives


def test_bump_is_safe_under_concurrent_callers(store):
    store.begin(target=400)

    def worker(pid: int):
        for i in range(50):
            store.bump(pid=pid, status="done", arxiv_id=f"{pid}-{i}")

    threads = [threading.Thread(target=worker, args=(200 + n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    totals = store.totals()
    assert totals.processed == 200 and totals.workers == 4
    assert all(s.processed == 50 for s in store.load_workers())


def test_describe_reports_progress_and_survives_an_empty_store(store):
    assert "no checkpoints" in store.describe()
    store.begin(target=5)
    store.bump(pid=1, status="done", arxiv_id="2301.00001")
    text = store.describe()
    assert "2301.00001" in text and "total" in text and "resumable" in text
