"""What the progress bar claims, and why it has to come from the manifest.

Every number on the bar has been wrong at some point, always for the same reason: it was
derived from the checkpoint files. Those only accumulate for runs that were *interrupted*
-- `finalize` deletes them on a clean finish -- so they drift and never self-correct. On a
real manifest the bar opened at 108,492 against 221,056 papers actually converted, and
reported `fail=10` with no failures in the manifest at all.

The bar now reads `done` and `total` straight from the manifest, so it cannot disagree
with what `status` prints.
"""

from concurrent.futures import ThreadPoolExecutor

import pytest

from src.config import Config, Paths
from src.utils import crawler as C
from src.utils.state import (
    DONE, FAILED_CONVERT, NO_PDF, PENDING, Manifest, PaperRow, TaskResult,
)


@pytest.fixture()
def pipeline(tmp_path, monkeypatch):
    cfg = Config()
    cfg.paths = Paths(data_dir=tmp_path)
    cfg.convert.workers, cfg.crawl.workers = 2, 2
    cfg.retry.in_run = False            # keep attempt counting out of these assertions
    cfg.paths.ensure()
    monkeypatch.setattr(
        C, "download_one",
        lambda row, s, l, d, stop: C.DownloadOutcome(
            path=tmp_path / "x.pdf", size=1, sha256="x"))
    monkeypatch.setattr(C, "ProcessPoolExecutor", lambda n, **kw: ThreadPoolExecutor(n))
    return cfg


def _seed(cfg, n, done=0):
    rows = [PaperRow(arxiv_id=f"2301.{i:05d}", version="v1", shard="2301", title="t")
            for i in range(n)]
    with Manifest(cfg.paths.manifest_db) as m:
        m.add_papers(rows)
        for i in range(done):                    # papers finished by an earlier run
            m.conn.execute("UPDATE papers SET status = ? WHERE arxiv_id = ?",
                           (DONE, f"2301.{i:05d}"))
        m.conn.commit()


def _outcome(statuses: dict[str, str]):
    def convert_and_write(row, pdf, data_dir, convert_cfg, **kw):
        status = statuses.get(row.arxiv_id, DONE)
        if status == DONE:
            return TaskResult(arxiv_id=row.arxiv_id, status=DONE, md_bytes=1, n_pages=1,
                              n_tables=0, n_chars=1, count_attempt=True, worker_id=1)
        return TaskResult(arxiv_id=row.arxiv_id, status=status, error="boom",
                          count_attempt=True, worker_id=1)
    return convert_and_write


def _captured_bar(monkeypatch):
    """Record the tqdm instance the pipeline builds, so its numbers can be inspected."""
    made = []
    real = C.tqdm

    def spy(*args, **kwargs):
        bar = real(*args, **kwargs)
        if kwargs.get("position") == 0:
            made.append(bar)
        return bar

    monkeypatch.setattr(C, "tqdm", spy)
    return made


def test_the_bar_starts_at_the_manifest_done_count_over_every_paper(pipeline, monkeypatch):
    monkeypatch.setattr(C, "convert_and_write", _outcome({}))
    made = _captured_bar(monkeypatch)
    _seed(pipeline, 20, done=12)                 # 12 already converted, 8 to go

    C.run_pipeline(pipeline)

    bar = made[0]
    assert bar.total == 20, "denominator is every paper in the manifest"
    assert bar.n == 20, "all 20 are converted by the end"
    with Manifest(pipeline.paths.manifest_db) as m:
        assert m.stats()[DONE] == bar.n, "the bar and the manifest must agree"


def test_only_a_converted_paper_advances_the_bar(pipeline, monkeypatch):
    """A failed or withdrawn paper has no markdown on disk; moving the bar would claim
    otherwise. They are reported in the postfix and by `status` instead."""
    monkeypatch.setattr(C, "convert_and_write", _outcome({
        "2301.00000": FAILED_CONVERT,
        "2301.00001": NO_PDF,
    }))
    made = _captured_bar(monkeypatch)
    _seed(pipeline, 5)

    tallies = C.run_pipeline(pipeline)

    assert tallies["processed"] == 5
    assert made[0].n == 3, "five handled, but only three converted"
    with Manifest(pipeline.paths.manifest_db) as m:
        stats = m.stats()
        assert stats[DONE] == 3 and stats[FAILED_CONVERT] == 1 and stats[NO_PDF] == 1


def test_the_pending_readout_matches_the_manifest_when_the_run_ends(pipeline, monkeypatch):
    monkeypatch.setattr(C, "convert_and_write", _outcome({}))
    _seed(pipeline, 30)

    C.run_pipeline(pipeline, limit=10)

    with Manifest(pipeline.paths.manifest_db) as m:
        stats = m.stats()
    assert stats[PENDING] == 20, "ten claimed, twenty still waiting"
    assert stats[DONE] == 10
