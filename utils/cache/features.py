#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""The 7-feature retention inputs: per-ball offset and size, plus the shell intruder counts."""

import numpy as np

from utils.base.knn import knn_dists, pairwise_dist, sq_norms
from utils.cache.entries import entry_kmax

N_BALLS = 3
POOR_FIT = 10.0
EPS = 1e-12

#-------------------------------------------------------------------------------


def balls_of(run, entries):
    """One ball per cached past query (its largest K): centers, radii."""
    cand, K = entry_kmax(entries, run.k_values)
    n_kv = len(run.k_values)
    t = np.array([run.k_values.index(int(k)) for k in K])
    rad = run.radius[cand * n_kv + t]
    cq = np.asarray(run.cand_q[cand], dtype=np.float64)
    return {"cand": cand, "K": K, "c": cq, "c_sq": sq_norms(cq), "r": rad}


def fit_balls(nr, de, radius, n, fill):
    """Relative offset (de / r) and size (nr / r) of the n best-fitting balls.

    nr: neighborhood radius per query; de: distance to every ball center
    (queries x balls); radius: ball radii. Balls are ranked by the worst-case
    fit (nr + de) / r; columns of missing balls (fewer than n) hold `fill`.
    Shared by the offline evaluation and the online simulator.
    """
    radius = np.maximum(radius, EPS)  # duplicate-vector balls can have radius 0
    F = np.full((len(nr), 2 * n), fill, dtype=np.float64)
    m = min(n, de.shape[1])
    fit = (nr[:, None] + de) / radius[None, :]
    cand = (np.argpartition(fit, m - 1, axis=1)[:, :m] if m < fit.shape[1]
            else np.tile(np.arange(m), (len(nr), 1)))
    order = np.take_along_axis(cand, np.argsort(np.take_along_axis(fit, cand, axis=1),
                                                axis=1), axis=1)
    rows = np.arange(len(nr))
    for t in range(m):
        j = order[:, t]
        F[:, 2 * t] = de[rows, j] / radius[j]
        F[:, 2 * t + 1] = nr / radius[j]
    return F


def ball_features(q, nr, bl, n):
    """Relative offset and size for the n best-fitting balls (NaN where missing)."""
    out = np.empty((len(q), 2 * n))
    for s in range(0, len(q), 2048):
        d = pairwise_dist(q[s:s + 2048], bl["c"], bl["c_sq"])
        out[s:s + 2048] = fit_balls(nr[s:s + 2048], d, bl["r"], n, np.nan)
    return out


def neighborhood_radius(q, st, k, eps=1e-12):
    """Distance from each query to its k-th nearest cached vector (floored at eps)."""
    if len(st["db_c"]) < k:
        raise ValueError(f"only {len(st['db_c'])} cached vectors, fewer than k={k}")
    return np.maximum(knn_dists(q, st["db_c"], st["db_c_sq"], k)[:, k - 1], eps)

