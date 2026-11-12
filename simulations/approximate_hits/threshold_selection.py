#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Choose an admission threshold with a probabilistic retention guarantee.

Admit a query iff its score >= lambda. Given calibration queries (score,
true retention), pick lambda so that, with probability ~>= 1 - delta, the
true mean retention over admitted queries is >= target.

Per-threshold lower bounds:
  normal  one-sided normal-approximation bound (as in SUPG)
  eb      empirical Bernstein, finite-sample, values in [0, 1]
Many thresholds:
  fixed       fixed-order testing (BARGAIN Alg. 1): test from strict to lenient
              at the full level, stop at the first failure
  bonferroni  test every grid threshold at level delta / |grid| (SUPG Alg. 3)
  split       choose on one half of the calibration set, test once on the other

Default: normal + fixed order.
"""

from statistics import NormalDist
from typing import Optional

import numpy as np

METHODS = ["naive", "eb_bonferroni", "normal_bonferroni", "normal_fixed",
           "eb_fixed", "normal_split", "normal_fixed_min30"]
DEFAULT_METHOD = "normal_fixed"
MIN_N = 30  # normal_fixed_min30: thresholds admitting < MIN_N cal requests are skipped


def lcb_normal(n, mean, sd, level):
    """One-sided normal-approximation lower bound (holds w.p. ~1 - level)."""
    z = NormalDist().inv_cdf(1.0 - level)
    return mean - z * sd / np.sqrt(n)


def lcb_eb(n, mean, sd, level):
    """Empirical Bernstein lower bound for values in [0, 1] (Maurer-Pontil)."""
    lt = np.log(2.0 / level)
    return mean - np.sqrt(2.0 * sd ** 2 * lt / n) - 7.0 * lt / (3.0 * (n - 1))


def grid_stats(s, y, grid):
    """n, mean, sd of y over {s >= lam} for each lam in grid (vectorized)."""
    order = np.argsort(-s)
    s_sorted = s[order]
    y_sorted = y[order]
    c1 = np.concatenate([[0.0], np.cumsum(y_sorted)])
    c2 = np.concatenate([[0.0], np.cumsum(y_sorted ** 2)])
    n = np.searchsorted(-s_sorted, -grid, side="right")
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = c1[n] / n
        var = (c2[n] - n * mean ** 2) / (n - 1)
    return n, mean, np.sqrt(np.maximum(var, 0.0))


def threshold_grid(fit_scores: np.ndarray, n_points: int = 100) -> np.ndarray:
    """Candidate thresholds, strict to lenient, from quantiles of FIT scores.

    Built from data independent of the calibration set, so the grid (and the
    fixed testing order) is set before calibration labels are seen.
    """
    return np.unique(np.quantile(fit_scores, np.linspace(0.0, 0.99, n_points)))[::-1]


def _passes(n, mean, sd, level, kind, target):
    lcb = lcb_normal(n, mean, sd, level) if kind == "normal" else lcb_eb(n, mean, sd, level)
    return (n >= 2) & np.nan_to_num(lcb >= target, nan=False)


def select(method: str, s_cal: np.ndarray, y_cal: np.ndarray,
           grid_desc: np.ndarray, target: float, delta: float,
           rng: Optional[np.random.Generator] = None) -> Optional[float]:
    """Chosen lambda (None = admit nothing). grid_desc is ordered strict -> lenient.

    s_cal, y_cal are request-level (repeat a query once per occurrence).
    """
    if method == "normal_split":
        rng = rng if rng is not None else np.random.default_rng(0)
        half = rng.permutation(len(s_cal))
        a, b = half[: len(half) // 2], half[len(half) // 2:]
        n, mean, sd = grid_stats(s_cal[a], y_cal[a], grid_desc)
        ok = _passes(n, mean, sd, delta, "normal", target)
        if not ok.any():
            return None
        lam = grid_desc[np.flatnonzero(ok)[-1]]
        n, mean, sd = grid_stats(s_cal[b], y_cal[b], np.array([lam]))
        return lam if _passes(n, mean, sd, delta, "normal", target)[0] else None

    n, mean, sd = grid_stats(s_cal, y_cal, grid_desc)
    if method == "naive":
        ok = (n >= 1) & np.nan_to_num(mean >= target, nan=False)
    elif method.endswith("bonferroni"):
        ok = _passes(n, mean, sd, delta / len(grid_desc), method.split("_")[0], target)
    elif "fixed" in method:
        # skipping depends only on calibration scores, not labels
        tested = n >= MIN_N if method.endswith("_min30") else np.ones(len(n), dtype=bool)
        ok = _passes(n, mean, sd, delta, method.split("_")[0], target) & tested
        fails = np.flatnonzero(tested & ~ok)
        if len(fails):
            ok = ok.copy()
            ok[fails[0]:] = False
    else:
        raise ValueError(f"unknown method {method}")
    if not ok.any():
        return None
    return float(grid_desc[np.flatnonzero(ok)[-1]])


def start_from_fit(s_fit, y_fit, target, delta, n_points=100):
    """Starting threshold for the fixed-order walk, chosen on the FIT split (independent of the
    calibration labels, so the walk's guarantee is unchanged; doc 9p): the grid threshold whose
    fit-set one-sided normal lower bound on mean retention is highest, i.e. the point the
    independent data says is easiest to certify. Thresholds stricter than it are not tested (a
    tiny, noisy top group can otherwise veto every threshold). None if the fit split is empty."""
    s_fit, y_fit = np.asarray(s_fit, float), np.asarray(y_fit, float)
    ok = np.isfinite(s_fit) & np.isfinite(y_fit)
    s_fit, y_fit = s_fit[ok], y_fit[ok]
    if len(s_fit) < 2:
        return None
    grid = threshold_grid(s_fit, n_points)
    order = np.argsort(-s_fit)
    s, y = s_fit[order], y_fit[order]
    cum = lambda v: np.concatenate([[0.0], np.cumsum(v)])
    S, SS = cum(y), cum(y * y)
    n = np.searchsorted(-s, -grid, side="right")
    z = NormalDist().inv_cdf(1.0 - delta)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = S[n] / n
        var = (SS[n] - n * mean ** 2) / np.maximum(n - 1, 1)
        lcb = np.where(n >= 2, mean - z * np.sqrt(np.maximum(var, 0.0) / n), -np.inf)
    return float(grid[int(np.nanargmax(lcb))])


def select_weighted(s_cal, y_cal, w_cal, grid_desc, target, delta, start=None):
    """normal_fixed with inverse-probability weights (a query labeled with
    probability pi gets weight 1 / pi). Per threshold: ratio estimate of the
    mean retention over {s >= lam}, sum w y / sum w, with variance
    sum w^2 (y - mean)^2 / (sum w)^2 (times n / (n - 1)); equals the
    unweighted bound when all weights are 1. Fixed order, stop at the first
    failure. Returns lambda or None (admit nothing)."""
    if start is not None:
        # walk from the fit split's starting threshold (start_from_fit); stricter ones are skipped
        keep = grid_desc <= start
        grid_desc = grid_desc[keep] if keep.any() else grid_desc[-1:]
    order = np.argsort(-s_cal)
    s, y, w = s_cal[order], y_cal[order], w_cal[order].astype(np.float64)
    cum = lambda v: np.concatenate([[0.0], np.cumsum(v)])
    W, WY, W2, W2Y, W2YY = cum(w), cum(w * y), cum(w * w), cum(w * w * y), cum(w * w * y * y)
    n = np.searchsorted(-s, -grid_desc, side="right")
    z = NormalDist().inv_cdf(1.0 - delta)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = WY[n] / W[n]
        var = (W2YY[n] - 2 * mean * W2Y[n] + mean ** 2 * W2[n]) / W[n] ** 2 * n / (n - 1)
        lcb = mean - z * np.sqrt(np.maximum(var, 0.0))
    ok = (n >= 2) & np.nan_to_num(lcb >= target, nan=False)
    fails = np.flatnonzero(~ok)
    if len(fails):
        ok[fails[0]:] = False
    if not ok.any():
        return None
    return float(grid_desc[np.flatnonzero(ok)[-1]])
