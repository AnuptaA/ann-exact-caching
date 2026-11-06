#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""The 7-feature retention inputs: per-ball offset and size, plus the shell intruder counts."""

import numpy as np

from utils.base.knn import pairwise_dist, sq_norms
from utils.cache.entries import entry_kmax

N_BALLS = 3
POOR_FIT = 10.0

#-------------------------------------------------------------------------------


def balls_of(run, entries):
    """One ball per cached past query (its largest K): centers, radii."""
    cand, K = entry_kmax(entries, run.k_values)
    n_kv = len(run.k_values)
    t = np.array([run.k_values.index(int(k)) for k in K])
    rad = run.radius[cand * n_kv + t]
    cq = np.asarray(run.cand_q[cand], dtype=np.float64)
    return {"cand": cand, "K": K, "c": cq, "c_sq": sq_norms(cq), "r": rad}


def ball_features(q, nr, bl, n):
    """Relative offset and size for the n best-fitting balls."""
    F = np.full((len(q), 2 * n), np.nan)
    m = min(n, len(bl["r"]))
    for s in range(0, len(q), 2048):
        d = pairwise_dist(q[s:s + 2048], bl["c"], bl["c_sq"])
        c = nr[s:s + 2048]
        fit = (c[:, None] + d) / bl["r"][None, :]
        part = (np.argpartition(fit, m - 1, axis=1)[:, :m] if m < fit.shape[1]
                else np.tile(np.arange(fit.shape[1]), (len(d), 1)))
        order = np.take_along_axis(part, np.argsort(np.take_along_axis(fit, part, axis=1),
                                                    axis=1), axis=1)
        rows = np.arange(len(d))
        for t in range(m):
            j = order[:, t]
            F[s:s + 2048, 2 * t] = d[rows, j] / bl["r"][j]
            F[s:s + 2048, 2 * t + 1] = c / bl["r"][j]
    return F
