#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""M4: hybrid estimator. Exact count over the candidate's own stored points
(query-relative distances, known exactly since coordinates are legitimately
cached), plus M3's LSH-based count curve for the residual beyond what the
stored set can tell us, combined into one r_hat and the same
recall_hat_from_r_hat formula M1 uses.

The exact/LSH boundary is the entry's own MAX query-relative distance
(t_sorted[-1]), not rho_c. rho_c bounds the stored points relative to the
entry's POLE, not the query - since pole and query are `delta` apart, stored
points can sit up to rho_c+delta from the query, so using rho_c as the
exact/estimated cutoff falsely assumes full coverage earlier than actually
achieved (confirmed directly: this produced a severe near-1.0 overestimation
bias, up to +0.42 mean bias, before the fix).

The inside step reads real cached coordinates (via GuardedCoordinatePool,
restricted to the entry's own stored IDs, in the calling script). The
outside step touches only the LSH codes/hyperplanes/LUT (global summary) via
bayesian_lsh_estimator - never a raw corpus coordinate outside the entry's
own cache."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np

from bayesian_lsh_estimator import estimate_count_curve
from inverse_coverage import recall_hat_from_r_hat

#-------------------------------------------------------------------------------


def estimate(query: np.ndarray, cached_points: np.ndarray, K: int,
             codes: np.ndarray, hyperplanes: np.ndarray, lut: np.ndarray,
             r_grid: np.ndarray) -> tuple:
    """cached_points: the entry's own stored points' coordinates (already
    read through the guard by the caller - this function takes them as a
    plain array, it does not itself touch the coordinate pool).

    Returns (recall_hat, r_hat, response_idx_order) where response_idx_order
    is the ascending-distance order over the entry's stored points (caller
    maps this back to real point IDs for the top-K response set)."""
    t = np.arccos(np.clip(cached_points @ query, -1.0, 1.0))
    order = np.argsort(t)
    t_sorted = t[order]
    boundary = float(t_sorted[-1])  # max query-relative distance the stored set can tell us about

    curve = estimate_count_curve(query, codes, hyperplanes, lut, r_grid)
    n_at_boundary = float(np.interp(boundary, r_grid, curve))
    inside_total = len(t_sorted)  # all stored points fall within [0, boundary] by definition

    def total_count(r: float) -> float:
        if r <= boundary:
            return float(np.searchsorted(t_sorted, r, side='right'))
        outside_est = max(0.0, float(np.interp(r, r_grid, curve)) - n_at_boundary)
        return inside_total + outside_est

    counts_at_grid = np.array([total_count(r) for r in r_grid])
    idx = np.searchsorted(counts_at_grid, K, side='left')
    r_hat = float(r_grid[idx]) if idx < len(r_grid) else float('inf')

    recall_hat = recall_hat_from_r_hat(r_hat, t_sorted[:K]) if inside_total >= K else 0.0
    return recall_hat, r_hat, order
