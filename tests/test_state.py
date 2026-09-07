import threading

import pytest

from src.utils.state import (
    DONE, FAILED_CONVERT, FAILED_DOWNLOAD, IN_FLIGHT, PENDING,
    Manifest, ManifestWriter, PaperRow, TaskResult,
)


def _rows(n, start=0):
    return [
        PaperRow(arxiv_id=f"2301.{i:05d}", version="v1", shard="2301", title=f"Paper {i}")
        for i in range(start, start + n)
    ]


@pytest.fixture()
def manifest(tmp_path):
    with Manifest(tmp_path / "m.db") as m:
        yield m


def test_add_papers_is_idempotent(manifest):
    assert manifest.add_papers(_rows(10)) == 10
    assert manifest.add_papers(_rows(10)) == 0          # same rows, nothing new
    assert manifest.add_papers(_rows(5, start=10)) == 5  # widening the scope adds work
    assert manifest.stats()["total"] == 15


def test_claim_batch_marks_in_flight(manifest):
    manifest.add_papers(_rows(10))
    claimed = manifest.claim_batch(4)
    assert len(claimed) == 4
    assert all(isinstance(r, PaperRow) for r in claimed)
    assert manifest.stats()[IN_FLIGHT] == 4
    assert manifest.stats()[PENDING] == 6


def test_claim_batch_never_hands_the_same_paper_to_two_workers(tmp_path):
    db = tmp_path / "m.db"
    with Manifest(db) as m:
        m.add_papers(_rows(200))

    seen, lock = [], threading.Lock()

    def worker():
        mm = Manifest(db)
        try:
            while True:
                batch = mm.claim_batch(7)
                if not batch:
                    return
                with lock:
                    seen.extend(p.arxiv_id for p in batch)
        finally:
            mm.close()

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(seen) == 200
    assert len(set(seen)) == 200        # the whole point: no duplicates


def test_reset_stale_recovers_a_crashed_run(manifest):
    manifest.add_papers(_rows(10))
    manifest.claim_batch(6)
    assert manifest.reset_stale() == 6
    assert manifest.stats()[PENDING] == 10


def test_reset_failed_respects_the_attempt_ceiling(tmp_path):
    db = tmp_path / "m.db"
    with Manifest(db) as m:
        m.add_papers(_rows(4))
        m.conn.execute(
            "UPDATE papers SET status=?, attempts=9 WHERE arxiv_id=?", (FAILED_DOWNLOAD, "2301.00000"))
        m.conn.execute(
            "UPDATE papers SET status=?, attempts=1 WHERE arxiv_id=?", (FAILED_DOWNLOAD, "2301.00001"))
        m.conn.execute(
            "UPDATE papers SET status=?, attempts=1 WHERE arxiv_id=?", (FAILED_CONVERT, "2301.00002"))
        m.conn.commit()

        assert m.reset_failed("download", max_attempts=4) == 1   # the exhausted one stays put
        assert m.reset_failed("convert", max_attempts=4) == 1


def _fail(m, arxiv_id, status=FAILED_CONVERT, attempts=1):
    m.conn.execute("UPDATE papers SET status=?, attempts=? WHERE arxiv_id=?",
                   (status, attempts, arxiv_id))
    m.conn.commit()


def test_claim_batch_honours_the_attempt_ceiling(manifest):
    manifest.add_papers(_rows(3))
    _fail(manifest, "2301.00000", FAILED_CONVERT, attempts=1)
    _fail(manifest, "2301.00001", FAILED_DOWNLOAD, attempts=4)

    claimed = manifest.claim_batch(10, (FAILED_CONVERT, FAILED_DOWNLOAD), max_attempts=4)
    assert [r.arxiv_id for r in claimed] == ["2301.00000"]   # the exhausted one is left


def test_claimable_ids_and_counts_agree(manifest):
    manifest.add_papers(_rows(4))
    _fail(manifest, "2301.00000", FAILED_CONVERT, attempts=1)
    _fail(manifest, "2301.00001", FAILED_DOWNLOAD, attempts=2)
    _fail(manifest, "2301.00002", FAILED_CONVERT, attempts=4)

    statuses = (FAILED_DOWNLOAD, FAILED_CONVERT)
    ids = manifest.claimable_ids(statuses, max_attempts=4)
    assert sorted(ids) == ["2301.00000", "2301.00001"]
    assert manifest.count_claimable(statuses, max_attempts=4) == 2
    assert manifest.count_claimable(statuses) == 3            # no ceiling: all three


def test_claim_ids_skips_rows_that_moved_on(manifest):
    """The retry snapshot is taken up front, so a row may change status underneath it."""
    manifest.add_papers(_rows(2))
    _fail(manifest, "2301.00000", FAILED_CONVERT)
    _fail(manifest, "2301.00001", FAILED_CONVERT)
    snapshot = ["2301.00000", "2301.00001"]

    manifest.conn.execute("UPDATE papers SET status=? WHERE arxiv_id=?", (DONE, "2301.00001"))
    manifest.conn.commit()

    claimed = manifest.claim_ids(snapshot)
    assert [r.arxiv_id for r in claimed] == ["2301.00000"]
    assert manifest.stats()[DONE] == 1


def test_a_paper_that_fails_again_is_not_retried_twice_in_one_run(manifest):
    """One attempt per paper per run: the snapshot is consumed, never re-queried."""
    manifest.add_papers(_rows(1))
    _fail(manifest, "2301.00000", FAILED_CONVERT, attempts=1)

    worklist = manifest.claimable_ids((FAILED_DOWNLOAD, FAILED_CONVERT), max_attempts=4)
    assert len(manifest.claim_ids(worklist[:1])) == 1
    del worklist[:1]

    # it fails again mid-run, and is back below the ceiling...
    _fail(manifest, "2301.00000", FAILED_CONVERT, attempts=2)
    assert worklist == []                                   # ...but the run is done with it


def test_writer_thread_applies_results(tmp_path):
    db = tmp_path / "m.db"
    with Manifest(db) as m:
        m.add_papers(_rows(3))
        claimed = m.claim_batch(3)

        writer = ManifestWriter(db, batch_size=2, flush_interval=0.1)
        writer.start()
        for row in claimed:
            writer.submit(TaskResult(
                arxiv_id=row.arxiv_id, status=DONE,
                pdf_bytes=1000, md_bytes=100, n_tables=2, n_pages=8,
                low_text=False, count_attempt=True,
            ))
        writer.stop()

        assert writer.applied == 3
        assert m.stats()[DONE] == 3
        row = m.conn.execute("SELECT * FROM papers LIMIT 1").fetchone()
        assert row["md_bytes"] == 100 and row["attempts"] == 1
        assert row["completed_at"] is not None


def test_writer_preserves_values_it_was_not_given(tmp_path):
    """A convert-stage result must not wipe the pdf_bytes recorded at download time."""
    db = tmp_path / "m.db"
    with Manifest(db) as m:
        m.add_papers(_rows(1))
        writer = ManifestWriter(db, batch_size=1, flush_interval=0.05)
        writer.start()
        writer.submit(TaskResult(arxiv_id="2301.00000", status=IN_FLIGHT, pdf_bytes=4242))
        writer.submit(TaskResult(arxiv_id="2301.00000", status=DONE, md_bytes=17))
        writer.stop()
        row = m.conn.execute("SELECT * FROM papers").fetchone()
        assert row["pdf_bytes"] == 4242 and row["md_bytes"] == 17
