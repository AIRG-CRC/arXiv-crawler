"""Retrying a failed paper inside the run that failed it.

The cross-run pass (`retry.on_start`) only helps on the *next* invocation. These cover
the immediate path: a transient failure -- a GPU under momentary pressure, a truncated
download, a worker that died -- should be recovered before the run finishes, without
letting one pathological paper cycle forever.
"""

from concurrent.futures import ThreadPoolExecutor

import pytest

from src.config import Config, Paths
from src.utils import crawler as C
from src.utils.state import DONE, FAILED_CONVERT, Manifest, PaperRow, TaskResult


@pytest.fixture()
def pipeline(tmp_path, monkeypatch):
    """A pipeline whose download and conversion are both stubs.

    The conversion pool is swapped for threads so the stub is actually used -- a real
    process pool would fork and re-import the unpatched module.
    """
    cfg = Config()
    cfg.paths = Paths(data_dir=tmp_path)
    cfg.convert.workers = 1
    cfg.crawl.workers = 2
    cfg.paths.ensure()

    monkeypatch.setattr(
        C, "download_one",
        lambda row, session, limiter, data_dir, stop: C.DownloadOutcome(
            path=tmp_path / "tmp" / f"{row.arxiv_id}.pdf", size=10, sha256="x"),
    )
    monkeypatch.setattr(C, "ProcessPoolExecutor", lambda n, **kw: ThreadPoolExecutor(n))
    return cfg


def _seed(cfg, ids, **columns):
    rows = [PaperRow(arxiv_id=i, version="v1", shard="2301", title=i) for i in ids]
    with Manifest(cfg.paths.manifest_db) as m:
        m.add_papers(rows)
        for key, value in columns.items():
            m.conn.execute(f"UPDATE papers SET {key} = ?", (value,))
        m.conn.commit()


def _converter(fail_times: dict[str, int], calls: dict[str, int]):
    """A conversion stub that fails each paper a set number of times, then succeeds."""
    def convert_and_write(row, pdf, data_dir, convert_cfg, **kw):
        calls[row.arxiv_id] = n = calls.get(row.arxiv_id, 0) + 1
        if n <= fail_times.get(row.arxiv_id, 0):
            return TaskResult(arxiv_id=row.arxiv_id, status=FAILED_CONVERT,
                              error="transient", count_attempt=True, worker_id=1)
        return TaskResult(arxiv_id=row.arxiv_id, status=DONE, md_bytes=10,
                          n_pages=1, n_tables=0, n_chars=10,
                          count_attempt=True, worker_id=1, converter="pymupdf")
    return convert_and_write


def test_a_transient_failure_is_recovered_inside_the_same_run(pipeline, monkeypatch):
    calls: dict[str, int] = {}
    monkeypatch.setattr(C, "convert_and_write", _converter({"2301.00001": 1}, calls))
    _seed(pipeline, ["2301.00001", "2301.00002"])

    tallies = C.run_pipeline(pipeline)

    assert calls == {"2301.00001": 2, "2301.00002": 1}
    assert tallies["in_run_retries"] == 1
    # The retry is the same paper coming round again, so the bar must not count it twice.
    assert tallies["processed"] == 2 and tallies["done"] == 2
    with Manifest(pipeline.paths.manifest_db) as m:
        assert m.stats()[DONE] == 2


def test_the_per_run_budget_bounds_a_paper_that_keeps_failing(pipeline, monkeypatch):
    calls: dict[str, int] = {}
    monkeypatch.setattr(C, "convert_and_write", _converter({"2301.00001": 99}, calls))
    pipeline.retry.in_run_attempts = 1
    _seed(pipeline, ["2301.00001"])

    tallies = C.run_pipeline(pipeline)

    assert calls["2301.00001"] == 2          # the original try plus one retry, then stop
    assert tallies["failed"] == 1
    with Manifest(pipeline.paths.manifest_db) as m:
        assert m.stats()[FAILED_CONVERT] == 1


def test_the_lifetime_ceiling_still_wins_over_the_per_run_budget(pipeline, monkeypatch):
    """A paper already near `max_attempts` gets no in-run retry, however generous the
    per-run budget is -- otherwise a broken paper would burn a download on every run."""
    calls: dict[str, int] = {}
    monkeypatch.setattr(C, "convert_and_write", _converter({"2301.00001": 99}, calls))
    pipeline.retry.in_run_attempts = 5
    pipeline.retry.max_attempts = 4
    _seed(pipeline, ["2301.00001"], attempts=3)   # one try left in its whole lifetime

    C.run_pipeline(pipeline)

    assert calls["2301.00001"] == 1
    with Manifest(pipeline.paths.manifest_db) as m:
        row = m.conn.execute("SELECT attempts FROM papers").fetchone()
        assert row["attempts"] == 4               # spent, and no longer claimable


def test_in_run_retry_can_be_switched_off(pipeline, monkeypatch):
    calls: dict[str, int] = {}
    monkeypatch.setattr(C, "convert_and_write", _converter({"2301.00001": 1}, calls))
    pipeline.retry.in_run = False
    _seed(pipeline, ["2301.00001"])

    tallies = C.run_pipeline(pipeline)

    assert calls["2301.00001"] == 1
    assert tallies["failed"] == 1 and "in_run_retries" not in tallies


# --- the cross-run pass, and its attempt ceiling -----------------------------------------
def _failed(cfg, arxiv_id, attempts):
    with Manifest(cfg.paths.manifest_db) as m:
        m.conn.execute("UPDATE papers SET status = ?, attempts = ? WHERE arxiv_id = ?",
                       (FAILED_CONVERT, attempts, arxiv_id))
        m.conn.commit()


def test_failed_papers_are_retried_first_on_the_next_run(pipeline, monkeypatch):
    order: list[str] = []
    monkeypatch.setattr(C, "convert_and_write", _converter({}, {}))
    real_download = C.download_one
    monkeypatch.setattr(C, "download_one", lambda row, *a: (
        order.append(row.arxiv_id), real_download(row, *a))[1])

    _seed(pipeline, ["2301.00001", "2301.00002", "2301.00003"])
    _failed(pipeline, "2301.00003", attempts=1)      # the only failure, seeded last

    tallies = C.run_pipeline(pipeline)

    assert order[0] == "2301.00003", "the failed paper must go before pending work"
    assert tallies["retried"] == 1
    with Manifest(pipeline.paths.manifest_db) as m:
        assert m.stats()[DONE] == 3


def test_a_paper_out_of_attempts_is_left_alone_by_default(pipeline, monkeypatch):
    monkeypatch.setattr(C, "convert_and_write", _converter({}, {}))
    _seed(pipeline, ["2301.00001"])
    _failed(pipeline, "2301.00001", attempts=4)      # at the default ceiling

    tallies = C.run_pipeline(pipeline)

    assert tallies["processed"] == 0 and tallies["retried"] == 0
    with Manifest(pipeline.paths.manifest_db) as m:
        assert m.stats()[FAILED_CONVERT] == 1


def test_retry_all_ignores_the_ceiling(pipeline, monkeypatch):
    """`--retry-all`: every failure gets another go, however many it has already had."""
    monkeypatch.setattr(C, "convert_and_write", _converter({}, {}))
    _seed(pipeline, ["2301.00001"])
    _failed(pipeline, "2301.00001", attempts=9)

    tallies = C.run_pipeline(pipeline, retry_all=True)

    assert tallies["retried"] == 1 and tallies["done"] == 1
    with Manifest(pipeline.paths.manifest_db) as m:
        assert m.stats()[DONE] == 1


def test_a_null_ceiling_in_config_means_always_retry(pipeline, monkeypatch):
    monkeypatch.setattr(C, "convert_and_write", _converter({}, {}))
    pipeline.retry.max_attempts = None
    _seed(pipeline, ["2301.00001"])
    _failed(pipeline, "2301.00001", attempts=9)

    assert C.run_pipeline(pipeline)["retried"] == 1
