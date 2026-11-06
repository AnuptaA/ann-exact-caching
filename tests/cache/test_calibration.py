#!/usr/bin/env python3
"""Tests for threshold-selection procedures in eval calibration."""


import numpy as np
import pytest

from utils.cache.calibration import (
    METHODS, calibrated_lambda, grid_stats, hoeffding_upper, lcb_eb, lcb_normal, select, threshold_grid,
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


def test_hoeffding_upper_matches_formula():
    up = hoeffding_upper(np.full(100, 0.8), 0.05)
    assert np.isclose(up, 0.8 + np.sqrt(np.log(20) / 200))


def test_select_rejects_bad_inputs():
    s, y = np.linspace(0, 1, 50), np.ones(50)
    grid = threshold_grid(s)
    with pytest.raises(ValueError, match="unknown method"):
        select("typo_fixed", s, y, grid, 0.9, 0.05)
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError, match="delta"):
            select("normal_fixed", s, y, grid, 0.9, bad)
    with pytest.raises(ValueError, match="scores but"):
        select("normal_fixed", s, y[:-1], grid, 0.9, 0.05)
    with pytest.raises(ValueError, match="NaN"):
        select("normal_fixed", np.r_[s[:-1], np.nan], y, grid, 0.9, 0.05)


def test_select_on_empty_calibration_set_admits_nothing():
    assert select("normal_fixed", np.array([]), np.array([]), np.array([0.5]), 0.9, 0.05) is None
    assert select("naive", np.array([]), np.array([]), np.array([0.5]), 0.9, 0.05) is None


def test_select_with_all_equal_scores_and_perfect_retention():
    s, y = np.zeros(100), np.ones(100)
    grid = threshold_grid(s)
    assert grid.tolist() == [0.0]
    assert select("normal_fixed", s, y, grid, 0.9, 0.05) == 0.0


def test_threshold_grid_validates_and_is_strict_to_lenient():
    with pytest.raises(ValueError):
        threshold_grid(np.array([]))
    with pytest.raises(ValueError):
        threshold_grid(np.array([0.1, np.nan]))
    g = threshold_grid(np.random.default_rng(0).random(500))
    assert (np.diff(g) < 0).all() and len(g) == 100


def test_lcb_bounds_below_mean_and_tolerate_tiny_samples():
    assert lcb_normal(100, 0.9, 0.1, 0.05) < 0.9
    assert lcb_eb(100, 0.9, 0.1, 0.05) < 0.9
    with np.errstate(all="ignore"):
        assert not lcb_eb(1, 1.0, 0.0, 0.05) >= 0.9          # n = 1: bound is -inf, never passes
        assert np.isnan(lcb_normal(0, np.nan, np.nan, 0.05)) or lcb_normal(0, 1.0, 0.0, 0.05) != 1.0


def test_hoeffding_validates_arguments():
    with pytest.raises(ValueError):
        hoeffding_upper([], 0.05)
    for bad in (0.0, 1.0):
        with pytest.raises(ValueError):
            hoeffding_upper([0.5], bad)


def test_calibrated_lambda_uses_cal_split_with_fit_grid():
    rng = np.random.default_rng(0)
    s = {"fit": rng.random(500), "cal": rng.random(500)}
    y = {"cal": np.clip(s["cal"] + 0.05 * rng.standard_normal(500), 0, 1)}
    idx = np.arange(500)
    lam = calibrated_lambda(s, y, idx, 0.9)
    assert lam == select("normal_fixed", s["cal"], y["cal"], threshold_grid(s["fit"]), 0.9, 0.05)
    assert calibrated_lambda(s, y, idx, 0.9, 0.05, "naive") is not None

