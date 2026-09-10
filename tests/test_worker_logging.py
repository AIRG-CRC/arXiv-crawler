"""A conversion worker must never write to the terminal.

`convert.max_tasks_per_child` forces the pool onto the **spawn** start method (CPython:
"Requires a non-'fork' mp_context start method. When given, we default to using
'spawn'"). A spawned worker inherits no logging configuration, so every ERROR fell
through to `logging.lastResort` -- an unformatted stderr writer -- and printed full
tracebacks over the progress bar while leaving the log file empty.

The worker function here is module level because spawn has to pickle it by reference.
"""

import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor

from src.utils.logging_setup import quiet_worker_logging


def _noisy(_ignored):
    """Every channel a real backend uses to make noise."""
    print("advice from a library, the way pymupdf4llm does")
    logging.getLogger("docling").error("Stage table failed", exc_info=ValueError("boom"))
    logging.getLogger("noisy").warning("a warning nobody needs")
    os.write(2, b"a raw write from a C extension\n")
    return "ok"


def test_a_spawned_worker_logs_to_the_file_and_not_the_terminal(tmp_path, capfd):
    log = tmp_path / "crawler.log"

    with ProcessPoolExecutor(
        1, initializer=quiet_worker_logging, initargs=(str(log),), max_tasks_per_child=5
    ) as pool:
        assert pool.submit(_noisy, 1).result() == "ok"

    captured = capfd.readouterr()
    assert captured.err == "", f"worker leaked to stderr: {captured.err!r}"
    assert "Stage table failed" not in captured.out
    assert "advice from a library" not in captured.out

    written = log.read_text()
    assert "Stage table failed" in written          # logging, at ERROR
    assert "ValueError: boom" in written            # with its traceback
    assert "a raw write from a C extension" in written   # fd 2, which logging cannot see
    assert "advice from a library" in written            # fd 1, likewise
    assert "a warning nobody needs" not in written  # below ERROR: dropped, as intended


def test_lastresort_is_removed_so_nothing_can_fall_through_to_stderr(monkeypatch):
    """Belt and braces: with no handler configured, logging writes to stderr by default.
    In a worker that means the progress bar, so the fallback is taken away entirely."""
    monkeypatch.setattr(logging, "lastResort", logging.StreamHandler(sys.stderr))
    quiet_worker_logging(None)
    assert logging.lastResort is None


def test_an_unwritable_log_path_does_not_kill_the_worker():
    """A worker that cannot open its log should still convert papers."""
    quiet_worker_logging("/proc/definitely/not/writable/crawler.log")
