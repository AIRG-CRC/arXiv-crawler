"""Splitting the corpus between devices.

The property the whole feature rests on is in `test_slices_are_disjoint_and_exhaustive`:
every paper belongs to exactly one device. Everything else here guards the ways that
property has a habit of breaking -- an unstable hash, a forgotten backfill, a partition
filter that silently does nothing.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys

import pytest

from src.utils.partition import (
    BUCKETS,
    DEVICES_ENV,
    INDEX_ENV,
    bucket_for,
    describe,
    resolve_partition,
)
from src.utils.state import DONE, PENDING, Manifest, PaperRow

IDS = [f"2301.{i:05d}" for i in range(2000)] + [f"hep-th/99{i:05d}" for i in range(500)]


# --- the key ---------------------------------------------------------------------------
def test_bucket_for_is_stable():
    """Literal values, deliberately.

    The builtin `hash()` is randomised per process for `str`, so a switch to it would
    repartition the corpus on every run and the two devices would silently overlap. Pinning
    the actual numbers is what turns that into a test failure instead of lost work.
    """
    assert bucket_for("2301.00001") == 199
    assert bucket_for("hep-th/9901001") == 234
    assert bucket_for("") == 0
    assert 0 <= bucket_for("math.GT/0309136") < BUCKETS


def test_bucket_for_survives_a_fresh_interpreter():
    """The same id must hash the same way in another process -- and on the other device."""
    code = "from src.utils.partition import bucket_for; print(bucket_for('2301.00001'))"
    env = {**os.environ, "PYTHONHASHSEED": "random"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=env, cwd=os.getcwd())
    assert out.stdout.strip() == "199", out.stderr


def test_buckets_are_reasonably_even():
    counts = [0] * BUCKETS
    for paper_id in IDS:
        counts[bucket_for(paper_id)] += 1
    assert min(counts) > 0
    assert max(counts) < 4 * (len(IDS) / BUCKETS)


@pytest.mark.parametrize("devices", [2, 3, 4, 7])
def test_slices_are_disjoint_and_exhaustive(devices):
    seen: dict[str, int] = {}
    for index in range(devices):
        for paper_id in IDS:
            if bucket_for(paper_id) % devices == index:
                assert paper_id not in seen, f"{paper_id} claimed by two devices"
                seen[paper_id] = index
    assert set(seen) == set(IDS)


# --- resolving the flags ---------------------------------------------------------------
def test_no_devices_means_the_whole_corpus():
    assert resolve_partition(None, None) is None
    assert resolve_partition(1, 0) is None       # one device needs no filter at all


def test_environment_supplies_the_defaults(monkeypatch):
    monkeypatch.setenv(DEVICES_ENV, "3")
    monkeypatch.setenv(INDEX_ENV, "2")
    assert resolve_partition(None, None) == (3, 2)
    assert resolve_partition(4, 1) == (4, 1)     # an explicit flag still wins


def test_a_lone_index_is_taken_as_one_device_of_one(monkeypatch):
    monkeypatch.delenv(DEVICES_ENV, raising=False)
    assert resolve_partition(None, 0) is None


@pytest.mark.parametrize("devices,index", [(2, 2), (2, -1), (0, 0), (BUCKETS + 1, 0)])
def test_impossible_partitions_are_rejected(devices, index):
    with pytest.raises(ValueError):
        resolve_partition(devices, index)


def test_a_non_numeric_environment_value_is_rejected(monkeypatch):
    monkeypatch.setenv(DEVICES_ENV, "two")
    with pytest.raises(ValueError):
        resolve_partition(None, None)


def test_describe():
    assert describe(None) == "the whole corpus"
    assert describe((2, 0)) == "slice 1 of 2"


# --- the manifest ----------------------------------------------------------------------
def _manifest(tmp_path, ids=IDS):
    m = Manifest(tmp_path / "m.db")
    m.add_papers([PaperRow(arxiv_id=i, version="v1", shard="2301") for i in ids])
    return m


def test_claims_from_two_devices_never_overlap(tmp_path):
    a = _manifest(tmp_path)
    b = Manifest(tmp_path / "m.db")              # the same file, two claimers
    got_a = {r.arxiv_id for r in a.claim_batch(len(IDS), partition=(2, 0))}
    got_b = {r.arxiv_id for r in b.claim_batch(len(IDS), partition=(2, 1))}
    assert not (got_a & got_b)
    assert got_a | got_b == set(IDS)
    assert got_a and got_b
    a.close()
    b.close()


def test_an_unpartitioned_claim_still_takes_everything(tmp_path):
    m = _manifest(tmp_path)
    assert len(m.claim_batch(len(IDS))) == len(IDS)
    m.close()


def test_counts_and_id_lists_honour_the_partition(tmp_path):
    m = _manifest(tmp_path)
    whole = m.count_claimable((PENDING,))
    half = m.count_claimable((PENDING,), partition=(2, 0))
    assert whole == len(IDS)
    assert 0 < half < whole
    assert len(m.claimable_ids((PENDING,), partition=(2, 0))) == half
    assert all(bucket_for(i) % 2 == 0 for i in m.claimable_ids((PENDING,), partition=(2, 0)))
    m.close()


def test_the_partition_does_not_disturb_the_attempt_ceiling(tmp_path):
    m = _manifest(tmp_path, ids=IDS[:50])
    m.conn.execute("UPDATE papers SET status = ?, attempts = 9", (DONE,))
    m.conn.commit()
    assert m.count_claimable((DONE,), max_attempts=4, partition=(2, 0)) == 0
    assert m.count_claimable((DONE,), partition=(2, 0)) > 0
    m.close()


def test_rows_written_before_the_column_existed_are_backfilled(tmp_path):
    """The migration path: an existing manifest has no `bucket` and no index over it."""
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE papers (arxiv_id TEXT PRIMARY KEY, version TEXT, shard TEXT, "
        "title TEXT, authors TEXT, categories TEXT, primary_category TEXT, doi TEXT, "
        "date_released TEXT, date_updated TEXT, status TEXT NOT NULL DEFAULT 'pending', "
        "pdf_bytes INTEGER, pdf_sha256 TEXT, md_bytes INTEGER, tables_bytes INTEGER, "
        "n_pages INTEGER, n_tables INTEGER, n_chars INTEGER, "
        "low_text INTEGER NOT NULL DEFAULT 0, attempts INTEGER NOT NULL DEFAULT 0, "
        "error TEXT, completed_at TEXT)"
    )
    conn.executemany("INSERT INTO papers (arxiv_id, version, shard) VALUES (?, ?, ?)",
                     [(i, "v1", "2301") for i in IDS[:100]])
    conn.commit()
    conn.close()

    with Manifest(db) as m:
        columns = {r[1] for r in m.conn.execute("PRAGMA table_info(papers)")}
        assert {"bucket", "remote_only"} <= columns
        assert m.conn.execute("SELECT COUNT(*) FROM papers WHERE bucket < 0").fetchone()[0] == 0
        stored = dict(m.conn.execute("SELECT arxiv_id, bucket FROM papers"))
        assert all(stored[i] == bucket_for(i) for i in stored)

    with Manifest(db) as m:                      # reopening must be a no-op
        assert m.count_claimable((PENDING,)) == 100


# --- two runs, one corpus ----------------------------------------------------------------
def _pipeline_cfg(tmp_path, monkeypatch, converted):
    """A pipeline whose download and conversion are stubs, recording what it handled."""
    from concurrent.futures import ThreadPoolExecutor

    from src.config import Config, Paths
    from src.utils import crawler as C
    from src.utils.state import TaskResult

    cfg = Config()
    cfg.paths = Paths(data_dir=tmp_path)
    cfg.convert.workers = 1
    cfg.crawl.workers = 1
    cfg.crawl.cooldown_seconds = 0          # never wait in a test
    cfg.paths.ensure()

    monkeypatch.setattr(
        C, "download_one",
        lambda row, session, limiter, data_dir, stop: C.DownloadOutcome(
            path=tmp_path / "tmp" / f"{row.arxiv_id}.pdf", size=10, sha256="x"))
    monkeypatch.setattr(C, "ProcessPoolExecutor", lambda n, **kw: ThreadPoolExecutor(n))

    def convert(row, pdf, data_dir, convert_cfg, **kw):
        converted.append(row.arxiv_id)
        return TaskResult(arxiv_id=row.arxiv_id, status=DONE, md_bytes=10, n_pages=1,
                          n_tables=0, n_chars=10, count_attempt=True, worker_id=1,
                          converter="pymupdf")

    monkeypatch.setattr(C, "convert_and_write", convert)
    return cfg, C


def test_two_runs_over_one_manifest_never_convert_the_same_paper(tmp_path, monkeypatch):
    """The property the whole feature exists for, end to end through `run_pipeline`.

    Both runs share a manifest here, which is the *harder* case: on two real devices each
    has its own copy and only the bucket connects them. If the slices overlap, this is
    where it shows up.
    """
    ids = [f"2301.{i:05d}" for i in range(60)]
    first: list[str] = []
    cfg, C = _pipeline_cfg(tmp_path, monkeypatch, first)
    with Manifest(cfg.paths.manifest_db) as m:
        m.add_papers([PaperRow(arxiv_id=i, version="v1", shard="2301") for i in ids])

    C.run_pipeline(cfg, partition=(2, 0))

    second: list[str] = []
    cfg2, C2 = _pipeline_cfg(tmp_path, monkeypatch, second)
    C2.run_pipeline(cfg2, partition=(2, 1))

    assert first and second
    assert not set(first) & set(second)          # no paper crawled twice
    assert set(first) | set(second) == set(ids)  # and none missed
    assert all(bucket_for(i) % 2 == 0 for i in first)
    with Manifest(cfg.paths.manifest_db) as m:
        assert m.stats()[DONE] == len(ids)


def test_an_unpartitioned_run_still_takes_the_whole_corpus(tmp_path, monkeypatch):
    ids = [f"2301.{i:05d}" for i in range(20)]
    converted: list[str] = []
    cfg, C = _pipeline_cfg(tmp_path, monkeypatch, converted)
    with Manifest(cfg.paths.manifest_db) as m:
        m.add_papers([PaperRow(arxiv_id=i, version="v1", shard="2301") for i in ids])

    C.run_pipeline(cfg)
    assert set(converted) == set(ids)


def _with_fake_bucket(monkeypatch, objects=()):
    """Point `run_pipeline`'s MinIO path at a fake client holding `objects`."""
    from src.utils import objectstore as O
    from tests.test_objectstore import FakeClient

    client = FakeClient()
    for name in objects:
        client.objects[name] = b"x" * 10
    real = O.MinioStore
    monkeypatch.setenv("MINIO_ACCESS_KEY", "k")
    monkeypatch.setenv("MINIO_SECRET_KEY", "s")
    monkeypatch.setattr(O, "MinioStore", lambda settings, **kw: real(settings, client=client))
    return client


def test_claim_any_takes_the_other_slice_but_not_its_finished_papers(tmp_path, monkeypatch):
    """The tail of a run: this device's slice is empty, the other device's is not.

    Stealing has to skip whatever is already in the bucket, or the point of partitioning --
    not doing the same work twice -- is lost exactly when it is cheapest to check.
    """
    from src.utils.objectstore import object_name

    ids = [f"2301.{i:05d}" for i in range(40)]
    mine = [i for i in ids if bucket_for(i) % 2 == 0]
    theirs = [i for i in ids if bucket_for(i) % 2 == 1]
    assert mine and theirs

    # Half of the other device's slice is already converted and in the bucket.
    done_elsewhere = theirs[: len(theirs) // 2]
    objects = [object_name(k, i, prefix="arxiv")
               for i in done_elsewhere for k in ("md", "meta")]

    converted: list[str] = []
    cfg, C = _pipeline_cfg(tmp_path, monkeypatch, converted)
    _with_fake_bucket(monkeypatch, objects)
    with Manifest(cfg.paths.manifest_db) as m:
        m.add_papers([PaperRow(arxiv_id=i, version="v1", shard="2301") for i in ids])

    tallies = C.run_pipeline(cfg, partition=(2, 0), to_minio=True, claim_any=True,
                             sync_bucket=False)

    # Its own slice, plus the part of the other slice nobody had done.
    assert set(converted) == set(mine) | (set(theirs) - set(done_elsewhere))
    assert not set(converted) & set(done_elsewhere)      # never re-downloaded
    assert tallies["already_elsewhere"] == len(done_elsewhere)
    with Manifest(cfg.paths.manifest_db) as m:
        assert m.stats()[DONE] == len(ids)               # the corpus is complete either way


def test_without_claim_any_a_device_stops_at_its_own_slice(tmp_path, monkeypatch):
    ids = [f"2301.{i:05d}" for i in range(40)]
    converted: list[str] = []
    cfg, C = _pipeline_cfg(tmp_path, monkeypatch, converted)
    _with_fake_bucket(monkeypatch)
    with Manifest(cfg.paths.manifest_db) as m:
        m.add_papers([PaperRow(arxiv_id=i, version="v1", shard="2301") for i in ids])

    C.run_pipeline(cfg, partition=(2, 0), to_minio=True, sync_bucket=False)
    assert set(converted) == {i for i in ids if bucket_for(i) % 2 == 0}


def test_the_start_of_run_sync_marks_papers_the_other_device_finished(tmp_path, monkeypatch):
    """`run-minio` must not crawl what the bucket already holds."""
    from src.utils.objectstore import object_name

    ids = [f"2301.{i:05d}" for i in range(20)]
    already = ids[:8]
    objects = [object_name(k, i, prefix="arxiv") for i in already for k in ("md", "meta")]

    converted: list[str] = []
    cfg, C = _pipeline_cfg(tmp_path, monkeypatch, converted)
    _with_fake_bucket(monkeypatch, objects)
    with Manifest(cfg.paths.manifest_db) as m:
        m.add_papers([PaperRow(arxiv_id=i, version="v1", shard="2301") for i in ids])

    C.run_pipeline(cfg, to_minio=True, sync_bucket=True)

    assert not set(converted) & set(already)
    assert set(converted) == set(ids) - set(already)
    with Manifest(cfg.paths.manifest_db) as m:
        assert m.stats()[DONE] == len(ids)
