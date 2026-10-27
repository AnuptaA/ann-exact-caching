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
           "eb_fixed", "normal_split"]
DEFAULT_METHOD = "normal_fixed"


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
    elif method.endswith("fixed"):
        ok = _passes(n, mean, sd, delta, method.split("_")[0], target)
        fails = np.flatnonzero(~ok)
        if len(fails):
            ok = ok.copy()
            ok[fails[0]:] = False
    else:
        raise ValueError(f"unknown method {method}")
    if not ok.any():
        return None
    return float(grid_desc[np.flatnonzero(ok)[-1]])
