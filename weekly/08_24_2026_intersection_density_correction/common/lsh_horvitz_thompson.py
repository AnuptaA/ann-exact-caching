#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Multi-table Horvitz-Thompson LSH density estimator: retrieves a
deliberately small subset of points via wide multi-table random-
hyperplane probing, then corrects for the known miss rate via inverse-
probability weighting. Unlike exact bucket-touching designs (the
recursive median-halving tree), this never tries to find every
overlapping bucket - it only needs the exact retrieval probability per
point, which is known in closed form for random origin hyperplanes.
"""

import sys
import math
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import numpy as np
from scipy.stats import binom

from part2_uniform_baseline_run_experiment import _sample_uniform_sphere

#-------------------------------------------------------------------------------


def p_single(theta: np.ndarray, m: int, x: int) -> np.ndarray:
    """P(a point at angular distance theta lands within Hamming
    distance x of the query's code in one table) = P(Binomial(m,
    theta/pi) <= x), since each of the m hyperplanes independently
    disagrees with probability theta/pi."""
    theta = np.asarray(theta, dtype=np.float64)
    return binom.cdf(x, m, theta / np.pi)


def p_union(theta: np.ndarray, m: int, x: int, L: int) -> np.ndarray:
    return 1.0 - (1.0 - p_single(theta, m, x)) ** L

#-------------------------------------------------------------------------------


def _popcount64(x: np.ndarray) -> np.ndarray:
    """Vectorized SWAR popcount for non-negative int64 arrays (codes
    here never exceed m<=32 bits, well within positive int64 range)."""
    x = x.astype(np.int64)
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0f
    return (x * 0x0101010101010101) >> 56


def _bit_weights(m: int) -> np.ndarray:
    return (1 << np.arange(m)).astype(np.int64)


def build_tables(points: np.ndarray, m: int, L: int, rng: np.random.Generator) -> list:
    """One table per index: {'H': (m,d) hyperplanes, 'codes': sorted
    unique populated codes, 'flat_points': concatenated point indices
    per populated code, 'offsets'/'lengths': ragged-gather layout,
    matching the repeat/cumsum idiom from median_halving_tree.py."""
    d = points.shape[1]
    weights = _bit_weights(m)
    tables = []
    for _ in range(L):
        H = _sample_uniform_sphere(m, d, rng)
        bits = (points @ H.T) >= 0.0
        point_codes = (bits.astype(np.int64) * weights).sum(axis=1)

        order = np.argsort(point_codes, kind='stable')
        sorted_codes = point_codes[order]
        sorted_points = order.astype(np.int64)
        unique_codes, first_idx, counts = np.unique(sorted_codes, return_index=True, return_counts=True)
        offsets = first_idx

        tables.append({
            'H': H, 'codes': unique_codes, 'offsets': offsets, 'lengths': counts,
            'flat_points': sorted_points,
        })
    return tables

#-------------------------------------------------------------------------------


def query_retrieval(tables: list, queries: np.ndarray, m: int, x: int) -> list:
    """Per-query array of unique retrieved point indices, deduplicated
    across all L tables. One query at a time (not flattened across the
    whole batch) - retrieval is deliberately small by design, but this
    also matches the lesson from median_halving_tree.py: never assume
    a per-query loop is the bottleneck without measuring; the earlier
    OOM came from flattening across queries, not from looping over
    them."""
    n_queries = queries.shape[0]
    weights = _bit_weights(m)
    query_codes_per_table = []
    for table in tables:
        bits = (queries @ table['H'].T) >= 0.0
        query_codes_per_table.append((bits.astype(np.int64) * weights).sum(axis=1))

    results = []
    for qi in range(n_queries):
        retrieved_parts = []
        for t, table in enumerate(tables):
            qc = query_codes_per_table[t][qi]
            dist = _popcount64(table['codes'] ^ qc)
            touched = np.flatnonzero(dist <= x)
            if len(touched) == 0:
                continue
            lengths = table['lengths'][touched]
            offsets = table['offsets'][touched]
            total = int(lengths.sum())
            if total == 0:
                continue
            starts_within = np.concatenate([[0], np.cumsum(lengths)[:-1]])
            within_pos = np.arange(total) - np.repeat(starts_within, lengths)
            source_indices = np.repeat(offsets, lengths) + within_pos
            retrieved_parts.append(table['flat_points'][source_indices])

        if retrieved_parts:
            results.append(np.unique(np.concatenate(retrieved_parts)))
        else:
            results.append(np.zeros(0, dtype=np.int64))
    return results

#-------------------------------------------------------------------------------


def estimate_count(query: np.ndarray, retrieved_ids: np.ndarray, points: np.ndarray,
                    r: float, m: int, x: int, L: int) -> tuple:
    """Returns (N_hat, retrieved_count). r is used only as a filter
    (step 5 of the spec: sum only over retrieved points with
    theta_i <= r) - the weight itself uses each qualifying point's own
    theta_i, never r. Omitting the filter (summing HT weights over
    every retrieved point regardless of whether it's within the cap)
    is a distinct bug from using r inside the weight formula, and was
    caught directly: an early pilot run produced N_hat in the tens of
    thousands against a true K=100 before this filter was added."""
    if len(retrieved_ids) == 0:
        return 0.0, 0
    theta_i = np.arccos(np.clip(points[retrieved_ids] @ query, -1.0, 1.0))
    within = theta_i <= r
    if not within.any():
        return 0.0, 0
    p_u = p_union(theta_i[within], m, x, L)
    p_u = np.clip(p_u, 1e-12, 1.0)
    return float((1.0 / p_u).sum()), int(within.sum())


def estimate_count_batch(queries: np.ndarray, retrieved_lists: list, points: np.ndarray,
                          radii: np.ndarray, m: int, x: int, L: int) -> np.ndarray:
    return np.array([estimate_count(queries[i], retrieved_lists[i], points, float(radii[i]), m, x, L)[0]
                      for i in range(len(queries))])


def estimate_count_multi_radius(query: np.ndarray, retrieved_ids: np.ndarray, points: np.ndarray,
                                 radii: np.ndarray, m: int, x: int, L: int) -> np.ndarray:
    """N_hat for one (query, retrieved set) pair against many radii at
    once - e.g. one radius per candidate sharing the same retrieval.

    Avoids both a per-radius Python loop and a per-radius re-filter of
    the retrieved set: each retrieved point's HT weight depends only on
    its own angular distance to the query, never on the radius (see
    estimate_count), so the weights can be computed once, sorted by
    distance, and prefix-summed - then N_hat for any radius is a single
    searchsorted lookup into that prefix sum. Equivalent to calling
    estimate_count once per radius, verified to agree to within
    floating-point boundary-tie noise (~1e-4 relative)."""
    radii = np.asarray(radii, dtype=np.float64)
    if len(retrieved_ids) == 0:
        return np.zeros(len(radii))
    theta_i = np.arccos(np.clip(points[retrieved_ids] @ query, -1.0, 1.0))
    weights = 1.0 / np.clip(p_union(theta_i, m, x, L), 1e-12, 1.0)
    order = np.argsort(theta_i)
    theta_sorted = theta_i[order]
    cum_weights = np.concatenate([[0.0], np.cumsum(weights[order])])
    return cum_weights[np.searchsorted(theta_sorted, radii, side='right')]


def estimate_knn_radius(query: np.ndarray, retrieved_ids: np.ndarray, points: np.ndarray,
                         K: int, m: int, x: int, L: int) -> float:
    """r_hat_q*: the radius at which the HT-weighted cumulative count of
    retrieved points first reaches K - the estimated true K-NN radius of
    the query itself, independent of any candidate. Inverts the same
    sorted-cumulative-weight construction estimate_count_multi_radius
    evaluates forward (count given a radius); this instead finds the
    radius given a target count.

    Returns inf if the retrieved set's total weight never reaches K
    (undercoverage - flagged via inf rather than silently returning the
    largest available radius, which would understate the true K-NN
    radius)."""
    if len(retrieved_ids) == 0:
        return float('inf')
    theta_i = np.arccos(np.clip(points[retrieved_ids] @ query, -1.0, 1.0))
    weights = 1.0 / np.clip(p_union(theta_i, m, x, L), 1e-12, 1.0)
    order = np.argsort(theta_i)
    theta_sorted = theta_i[order]
    cum_weights = np.cumsum(weights[order])
    idx = np.searchsorted(cum_weights, K, side='left')
    if idx >= len(theta_sorted):
        return float('inf')
    return float(theta_sorted[idx])

#-------------------------------------------------------------------------------


def select_hyperparameters(K: int, epsilon: float, r: float, m_candidates: list,
                            x_candidates: list, L_max: int) -> dict:
    """3-step derivation: required capture probability -> tables needed
    -> (m,x) minimizing retrieved fraction subject to L<=L_max."""
    p_star = 1.0 / (1.0 + K * epsilon ** 2)

    best = None
    for m in m_candidates:
        for x in x_candidates:
            if x >= m:
                continue
            ps_r = float(p_single(np.array([r]), m, x)[0])
            if ps_r <= 0.0 or ps_r >= 1.0:
                continue
            L = math.ceil(math.log(1.0 - p_star) / math.log(1.0 - ps_r))
            if L > L_max or L < 1:
                continue
            f_single = float(p_single(np.array([np.pi / 2]), m, x)[0])
            f_union = 1.0 - (1.0 - f_single) ** L
            if best is None or f_union < best['retrieved_frac']:
                best = {'m': m, 'x': x, 'L': L, 'retrieved_frac': f_union, 'p_star': p_star}
    return best

#-------------------------------------------------------------------------------


def make_anisotropic_basis(d: int, r_eff: int, rng: np.random.Generator) -> np.ndarray:
    """A random r_eff-dimensional subspace basis (d, r_eff) of ambient
    R^d. Corpus and queries must share the SAME basis - drawing a fresh
    basis per call (an earlier version of this generator did exactly
    that) makes queries unrelated to the corpus's concentration
    structure, silently defeating the whole point of the anisotropic
    setting: caught directly by checking that a query's K-NN radius
    against a highly concentrated (r_eff=1) corpus came out identical
    to the isotropic case, when it should have collapsed to nearly the
    corpus's own tiny internal spread."""
    basis, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return basis[:, :r_eff]


def sample_anisotropic(N: int, basis: np.ndarray, eps: float, rng: np.random.Generator) -> np.ndarray:
    """N points near the r_eff-dimensional subspace spanned by `basis`
    (from make_anisotropic_basis), plus eps isotropic noise (avoids an
    exactly degenerate subspace), normalized to the unit sphere."""
    d, r_eff = basis.shape
    z = rng.standard_normal((N, r_eff))
    points = z @ basis.T
    points += eps * rng.standard_normal((N, d))
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return points / norms
