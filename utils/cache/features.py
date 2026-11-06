#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""The 7-feature retention inputs: per-ball offset and size, plus the shell intruder counts."""

from dataclasses import dataclass

import numpy as np

from utils.base.knn import knn_dists, pairwise_dist, sq_norms
from utils.cache.entries import entry_kmax, state_tables
from utils.cache.shell import shell_features, shell_sample

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


@dataclass
class SplitInputs:
    """One cache state evaluated on every split (dicts keyed by split name)."""
    st: dict      # state tables
    bl: dict      # balls
    y: dict       # retention per distinct query
    w: dict       # request weights per distinct query
    nr: dict      # neighborhood radius
    B: dict       # ball features, missing balls set to POOR_FIT


def split_inputs(run, entries, splits, n_balls=N_BALLS):
    """Labels, weights, neighborhood radius and ball features per split.

    n_balls=None skips the ball features (B is empty). Returns None when the cache holds
    no more than k vectors (nothing to answer from).
    """
    st = state_tables(run, entries)
    if len(st["cached"]) <= run.k:
        return None
    bl = balls_of(run, entries)
    nr = {sp: neighborhood_radius(run.sets[sp][0], st, run.k) for sp in splits}
    return SplitInputs(
        st=st, bl=bl, nr=nr,
        y={sp: st["in_cache"][run.topk[sp]].mean(axis=1) for sp in splits},
        w={sp: run.sets[sp][1].astype(np.float64) for sp in splits},
        B={} if n_balls is None else {
            sp: np.nan_to_num(ball_features(run.sets[sp][0], nr[sp], bl, n_balls), nan=POOR_FIT)
            for sp in splits})


def shell_block(run, inp, p, R, bits, columns=None):
    """Shell sample and its features per split.

    columns: indices into the shell feature vector, e.g. [SAMPLE_J] for the
    1.0 x radius intruder count; None keeps all columns.
    """
    samp = shell_sample(run, inp.bl, inp.st["in_cache"], p, R, bits)
    S = {}
    for sp in inp.nr:
        F = shell_features(run.sets[sp][0], inp.nr[sp], samp)
        S[sp] = F if columns is None else F[:, columns]
    return samp, S

