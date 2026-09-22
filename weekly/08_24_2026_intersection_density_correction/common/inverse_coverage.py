#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Inverse-coverage counting: estimates a test query's own true K-NN
radius using only a cache entry's own stored (K+Delta) neighbors and
their known distances - no full-corpus access at query time, unlike the
deprecated LSH-HT/median-halving-tree line (see ../POSTMORTEM.md).

For a cache entry with pole q_c storing K+Delta neighbors, rho_c is the
distance to its (K+Delta)-th neighbor (free - the largest already-stored
distance). For a test query q at distance delta from q_c, and a stored
point at distance t from q: by the spherical law of cosines, the
fraction of the sphere-shell at distance t from q that also lies within
rho_c of q_c is `coverage(t)`. Weighting each stored point by
1/coverage(t) and accumulating in ascending order of t until the
weighted sum reaches K gives r_hat, an estimate of q's own true K-NN
radius - built entirely from the entry's own stored neighbors."""

import sys
import math
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np

from primitives import single_cap_fraction

#-------------------------------------------------------------------------------


def coverage(t: np.ndarray, delta: float, rho_c: float, d: int) -> np.ndarray:
    """Fraction of the sphere-shell at angular distance t from the test
    query q that also lies within rho_c of the cache entry's pole q_c,
    given the poles are separated by delta. Vectorized over t.

    d is the ambient dimension (D-dimensional unit vectors on S^(D-1));
    the cross-section at fixed t is itself a (D-2)-sphere, so
    single_cap_fraction is evaluated at d-1 per the spherical law of
    cosines derivation.

    Degenerate case (t=0, or delta=0): the shell collapses to a single
    point (t=0: the point coincides with q itself; delta=0: q coincides
    with q_c, so every shell is trivially either fully in or fully out).
    Handled by direct comparison rather than a 0/0 division."""
    t = np.asarray(t, dtype=np.float64)
    sin_delta = math.sin(delta)
    sin_t = np.sin(t)
    denom = sin_delta * sin_t

    cov = np.empty_like(t)
    degenerate = np.abs(denom) < 1e-12
    normal = ~degenerate

    if np.any(normal):
        cos_arg = (math.cos(rho_c) - math.cos(delta) * np.cos(t[normal])) / denom[normal]
        cos_arg = np.clip(cos_arg, -1.0, 1.0)
        angle = np.arccos(cos_arg)
        cov[normal] = _single_cap_fraction_vec(angle, d - 1)

    if np.any(degenerate):
        if abs(delta) < 1e-12:
            # q coincides with q_c: covered iff this shell's own radius is within rho_c
            cov[degenerate] = (t[degenerate] <= rho_c).astype(np.float64)
        else:
            # t=0: the "shell" is the single point q, covered iff q itself is within rho_c of q_c
            cov[degenerate] = 1.0 if delta <= rho_c else 0.0

    return cov


def _single_cap_fraction_vec(theta: np.ndarray, d: int) -> np.ndarray:
    """Vectorized single_cap_fraction (the scalar version in primitives.py
    only takes a float theta)."""
    return np.array([single_cap_fraction(float(th), d) for th in theta])

#-------------------------------------------------------------------------------


def accumulate_to_k(sorted_t: np.ndarray, sorted_coverage: np.ndarray, K: int) -> tuple:
    """sorted_t/sorted_coverage: a cache entry's stored-neighbor distances
    from the test query (ascending) and their coverage() values. Weights
    each point by 1/coverage(t) and accumulates until the running total
    reaches K. Returns (r_hat, min_coverage, n_used - how many reserve
    points were consumed to reach K); r_hat is inf if the stored
    neighbors' total weight never reaches K (insufficient reserve -
    flagged, not silently understated)."""
    if len(sorted_t) == 0:
        return float('inf'), 0.0, 0
    weights = 1.0 / np.clip(sorted_coverage, 1e-12, 1.0)
    cum_weights = np.cumsum(weights)
    idx = np.searchsorted(cum_weights, K, side='left')
    if idx >= len(sorted_t):
        return float('inf'), (float(sorted_coverage.min()) if len(sorted_coverage) else 0.0), len(sorted_t)
    r_hat = float(sorted_t[idx])
    min_coverage = float(sorted_coverage[:idx + 1].min())
    return r_hat, min_coverage, idx + 1


def recall_hat_from_r_hat(r_hat: float, entry_top_k_distances: np.ndarray) -> float:
    """Fraction of the cache entry's own true top-K (not K+Delta) whose
    distance to the test query is within r_hat."""
    if not np.isfinite(r_hat):
        return 0.0
    return float(np.mean(entry_top_k_distances <= r_hat))
