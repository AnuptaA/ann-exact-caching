#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Run a function over tasks in forked workers that share large read-only inputs."""

import multiprocessing as mp
from contextlib import contextmanager

_STATE = {}

#-------------------------------------------------------------------------------


def _call(task):
    return _STATE["fn"](task, **_STATE["shared"])


class _Local:
    def __init__(self, fn, shared):
        self.fn, self.shared = fn, shared

    def imap(self, tasks):
        return (self.fn(t, **self.shared) for t in tasks)

    def map(self, tasks):
        return [self.fn(t, **self.shared) for t in tasks]


class _Forked:
    def __init__(self, pool):
        self.pool = pool

    def imap(self, tasks):
        return self.pool.imap(_call, tasks)

    def map(self, tasks):
        return self.pool.map(_call, tasks)


@contextmanager
def worker_pool(fn, workers, n_tasks, **shared):
    """Yield an object with imap(tasks) and map(tasks) calling fn(task, **shared).

    The shared inputs are inherited through fork rather than pickled per task.
    With workers <= 1 or a single task everything runs in this process.
    """
    if workers <= 1 or n_tasks <= 1:
        yield _Local(fn, shared)
        return
    _STATE["fn"], _STATE["shared"] = fn, shared
    with mp.get_context("fork").Pool(min(workers, n_tasks)) as pool:
        yield _Forked(pool)
