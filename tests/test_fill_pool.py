"""`fill_pool`: the conversion pool runs at full strength, not on demand."""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from src.utils.crawler import fill_pool


def _pid() -> int:
    import os
    return os.getpid()


def test_fill_pool_starts_every_worker_up_front():
    pool = ProcessPoolExecutor(3, mp_context=multiprocessing.get_context("spawn"),
                               max_tasks_per_child=1)
    try:
        assert len(pool._processes) == 0          # stock behaviour: nothing until asked
        assert fill_pool(pool) == 3
        assert pool.submit(_pid).result(timeout=60) > 0
    finally:
        pool.shutdown(wait=True)


def test_fill_pool_keeps_full_strength_through_recycling():
    pool = ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn"),
                               max_tasks_per_child=1)
    try:
        fill_pool(pool)
        for _ in range(3):                        # every task retires its worker
            pool.submit(_pid).result(timeout=60)
            assert fill_pool(pool) == 2
    finally:
        pool.shutdown(wait=True)


def test_fill_pool_after_shutdown_starts_nothing():
    pool = ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn"))
    pool.shutdown(wait=True)
    assert fill_pool(pool) == 0
