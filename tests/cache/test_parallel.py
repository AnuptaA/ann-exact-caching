#!/usr/bin/env python3
"""Tests for utils.cache.parallel.worker_pool."""

import os

import numpy as np
import pytest

from utils.cache.parallel import worker_pool


def _square_plus(task, offset, table):
    return task * task + offset + int(table[task % len(table)])


def _boom(task, flag):
    if task == 2:
        raise RuntimeError("worker failed")
    return task


def test_results_are_ordered_and_shared_inputs_visible_in_workers():
    table = np.arange(5)
    tasks = list(range(8))
    want = [_square_plus(t, 10, table) for t in tasks]
    for workers in (1, 3):
        with worker_pool(_square_plus, workers, len(tasks), offset=10, table=table) as pool:
            assert pool.map(tasks) == want
            assert list(pool.imap(tasks)) == want


def test_zero_and_single_task_run_in_process():
    with worker_pool(_square_plus, 4, 0, offset=0, table=np.arange(2)) as pool:
        assert pool.map([]) == []
    with worker_pool(_square_plus, 4, 1, offset=1, table=np.arange(2)) as pool:
        assert pool.map([3]) == [3 * 3 + 1 + 1]


def test_forked_workers_use_other_processes_and_errors_propagate():
    def pid(task, **_):
        return os.getpid()
    with worker_pool(pid, 2, 4) as pool:
        assert set(pool.map(range(4))) != {os.getpid()}
    with worker_pool(_boom, 2, 4, flag=True) as pool:
        with pytest.raises(RuntimeError, match="worker failed"):
            pool.map(range(4))
    with worker_pool(_boom, 1, 4, flag=True) as pool:                 # in-process path too
        with pytest.raises(RuntimeError, match="worker failed"):
            pool.map(range(4))
