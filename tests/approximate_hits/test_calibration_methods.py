#!/usr/bin/env python3
"""Tests for threshold-selection procedures in eval_calibration_methods."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from simulations.approximate_hits.threshold_selection import (
    METHODS, grid_stats, select,
)


def test_grid_stats_matches_brute_force():
    rng = np.random.default_rng(0)
    s = rng.random(500)
    y = rng.random(500)
    grid = np.sort(rng.random(30))[::-1]
    n, mean, sd = grid_stats(s, y, grid)
    for lam, ni, mi, si in zip(grid, n, mean, sd):
        adm = y[s >= lam]
        assert ni == len(adm)
        if len(adm) >= 2:
            assert mi == pytest.approx(adm.mean())
            assert si == pytest.approx(adm.std(ddof=1))


def _population(rng, n):
    """Scores in [0, 1]; retention in {0, .1, ..., 1}, higher for high scores."""
    s = rng.random(n)
    p = 0.7 + 0.3 * s
    y = rng.binomial(10, p) / 10.0
    return s, y


@pytest.mark.parametrize("target", [0.9, 0.95])
def test_guaranteed_methods_rarely_miss_target(target):
    rng = np.random.default_rng(1)
    s_pop, y_pop = _population(rng, 400000)
    grid_desc = np.quantile(s_pop, np.linspace(0, 0.99, 100))[::-1]
    true_mean = {}
    order = np.argsort(-s_pop)
    csum = np.cumsum(y_pop[order])
    for lam in grid_desc:
        n = int((s_pop >= lam).sum())
        true_mean[lam] = csum[n - 1] / n
    trials = 300
    fails = {m: 0 for m in METHODS}
    for _ in range(trials):
        idx = rng.choice(len(s_pop), size=3000, replace=False)
        for m in METHODS:
            lam = select(m, s_pop[idx], y_pop[idx], grid_desc, target, 0.05, rng)
            if lam is not None and true_mean[lam] < target:
                fails[m] += 1
    for m in METHODS:
        if m != "naive":
            # allow sampling slack around delta = 0.05 for 300 trials
            assert fails[m] / trials <= 0.09, (m, fails[m] / trials)
    assert fails["naive"] / trials > 0.2  # naive often misses (selection effect)
