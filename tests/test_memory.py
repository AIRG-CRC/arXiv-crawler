"""The memory defences.

Grounded in a measurement: a docling worker's RSS climbed 2.70 -> 3.68 GB over four
papers and kept going, because glibc holds freed arenas rather than returning them.
Four such workers reach the OOM killer, which does not kill the process at fault -- it
kills whatever is biggest and convenient, i.e. the editor.
"""

import pytest

from src.utils import memory
from src.utils.memory import MemoryGuard, human, release_memory, resolve_floor


# --- reading the machine --------------------------------------------------------------
def test_available_and_total_are_plausible():
    available, total = memory.available_bytes(), memory.total_bytes()
    if total is None:                       # not Linux; the module declines to guess
        assert available is None
        return
    assert 0 < available <= total
    assert total > 256 * 2**20              # any machine running this has more than 256M


def test_rss_of_this_process_is_positive():
    rss = memory.rss_bytes()
    assert rss is None or rss > 0


def test_release_memory_is_safe_to_call_and_reports_whether_it_ran():
    assert release_memory() in (True, False)
    assert release_memory() in (True, False)      # repeatable, no state left behind


# --- floor parsing ---------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("4GB", 4 * 2**30), ("512MB", 512 * 2**20), ("2 gb", 2 * 2**30),
    ("1TB", 2**40), ("4GiB", 4 * 2**30),
])
def test_absolute_sizes_are_understood(text, expected):
    assert resolve_floor(text) == expected


def test_a_fraction_is_read_as_a_share_of_total_ram():
    total = memory.total_bytes()
    if total is None:
        pytest.skip("not Linux")
    assert resolve_floor(0.1) == pytest.approx(total * 0.1, rel=1e-6)


def test_a_plain_number_above_one_is_already_bytes():
    assert resolve_floor(4 * 2**30) == 4 * 2**30


@pytest.mark.parametrize("value", [None, "", False])
def test_the_guard_can_be_switched_off(value):
    assert resolve_floor(value) is None
    assert MemoryGuard(resolve_floor(value)).enabled is False


def test_nonsense_disables_the_guard_rather_than_crashing_the_run():
    assert resolve_floor("plenty please") is None


# --- back-pressure ----------------------------------------------------------------------
def test_a_disabled_guard_always_allows_dispatch():
    guard = MemoryGuard(None)
    assert guard.has_headroom() is True
    assert guard.pauses == 0


def test_dispatch_pauses_below_the_floor_and_resumes_above_it(monkeypatch):
    free = [10 * 2**30]
    monkeypatch.setattr(memory, "available_bytes", lambda: free[0])
    seen = []
    guard = MemoryGuard(4 * 2**30, on_pause=lambda f, fl: seen.append(f))

    assert guard.has_headroom() is True
    free[0] = 1 * 2**30
    assert guard.has_headroom() is False
    assert seen == [1 * 2**30]

    # Still short: it stays paused, but does not re-announce on every single loop.
    assert guard.has_headroom() is False
    assert len(seen) == 1

    free[0] = 10 * 2**30
    assert guard.has_headroom() is True
    assert guard.pauses == 1


def test_an_unreadable_meminfo_does_not_stall_the_run(monkeypatch):
    """Unknowable is not the same as exhausted -- refusing to dispatch would hang."""
    monkeypatch.setattr(memory, "available_bytes", lambda: None)
    assert MemoryGuard(4 * 2**30).has_headroom() is True


def test_human_is_readable():
    assert human(4 * 2**30) == "4.0 GB"
    assert human(None) == "?"


# --- the guard inside the pipeline ------------------------------------------------------
def test_a_paused_guard_delays_the_run_but_never_truncates_it(tmp_path, monkeypatch):
    """The bug this guards: with dispatch paused, the last in-flight paper finishing
    empties both pools, which the loop would otherwise read as "queue drained, done" --
    silently stopping short of the target instead of waiting for memory."""
    from concurrent.futures import ThreadPoolExecutor

    from src.config import Config, Paths
    from src.utils import crawler as C
    from src.utils.state import DONE, Manifest, PaperRow, TaskResult

    cfg = Config()
    cfg.paths = Paths(data_dir=tmp_path)
    cfg.convert.workers, cfg.crawl.workers = 1, 1
    cfg.convert.memory_floor = "4GB"
    cfg.paths.ensure()

    # Short of memory for the first few checks, then plenty.
    calls = [0]

    def fake_available():
        calls[0] += 1
        return 1 * 2**30 if calls[0] <= 3 else 32 * 2**30

    monkeypatch.setattr(memory, "available_bytes", fake_available)
    monkeypatch.setattr(
        C, "download_one",
        lambda row, s, l, d, stop: C.DownloadOutcome(path=tmp_path / "x.pdf", size=1, sha256="x"))
    monkeypatch.setattr(
        C, "convert_and_write",
        lambda row, pdf, d, cfg_, **kw: TaskResult(
            arxiv_id=row.arxiv_id, status=DONE, md_bytes=1, n_pages=1,
            n_tables=0, n_chars=1, count_attempt=True, worker_id=1))
    monkeypatch.setattr(C, "ProcessPoolExecutor", lambda n, **kw: ThreadPoolExecutor(n))

    ids = [f"2301.0000{i}" for i in range(1, 5)]
    with Manifest(cfg.paths.manifest_db) as m:
        m.add_papers([PaperRow(arxiv_id=i, version="v1", shard="2301", title=i) for i in ids])

    tallies = C.run_pipeline(cfg)

    assert tallies["processed"] == 4 and tallies["done"] == 4
    with Manifest(cfg.paths.manifest_db) as m:
        assert m.stats()[DONE] == 4
