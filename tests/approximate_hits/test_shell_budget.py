#!/usr/bin/env python3
"""Tests for the min-n bound, deeper shells, budget prefixes, cal study."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from simulations.approximate_hits import eval_shell_budget as eb
from simulations.approximate_hits import eval_shell_size as ess
from utils.cache.calibration import select
from utils.cache.shell import shell_ids
from utils.cache.workloads import RunData


def test_min30_skips_tiny_groups_and_default_unchanged():
    # top 3 scores have perfect retention, the rest average well below target
    s = np.arange(100, dtype=float)[::-1]
    y = np.r_[np.ones(3), np.full(97, 0.5)]
    grid = s[2:].copy()                 # strictest threshold admits the top 3
    # normal_fixed: the top 3 have sd 0 -> lower bound 1 -> admits them
    assert select("normal_fixed", s, y, grid, 0.9, 0.05) == 97.0
    # min30: groups below 30 are skipped; the first tested group fails -> nothing
    assert select("normal_fixed_min30", s, y, grid, 0.9, 0.05) is None


def test_shell_ids_uses_extended_lists_beyond_500():
    run = RunData(k=10, k_values=[10, 500], db=None, cand_q=None, sets={}, topk={},
                  cached_ids=[np.arange(10), np.arange(500), np.arange(10), np.arange(500)],
                  radius=None, cand_ext=np.tile(np.arange(2000), (2, 1)))
    assert np.array_equal(shell_ids(run, 1, 10, 500), np.arange(10, 500))
    assert np.array_equal(shell_ids(run, 1, 10, 2000), np.arange(10, 2000))


def test_cumulative_cache_and_prefix():
    run = RunData(k=10, k_values=[10], db=np.zeros((100, 2)), cand_q=None, sets={}, topk={},
                  cached_ids=[np.array([0, 1, 2]), np.array([2, 3]), np.array([4, 5, 6, 7])],
                  radius=None)
    selected = [0, 1, 2]
    cum = eb.cumulative_cache_size(run, selected)
    assert cum.tolist() == [3, 4, 8]
    assert eb.prefix_for(0.04, 100, selected, cum) == [0, 1]
    assert eb.prefix_for(0.02, 100, selected, cum) == []


def test_cal_study_shapes():
    rng = np.random.default_rng(0)
    s = rng.random(6000)
    y = np.clip(s + 0.05 * rng.standard_normal(6000), 0, 1)
    grid = np.unique(np.quantile(s, np.linspace(0, 0.99, 50)))[::-1]
    out = ess.cal_study(s, y, grid, 0.9, [500, 7000], 10, ["normal_fixed", "eb_fixed"],
                        np.random.default_rng(1))
    assert set(out["normal_fixed"]) == {"500"}
    assert 0 <= out["eb_fixed"]["500"]["frac_below_T"] <= 1
