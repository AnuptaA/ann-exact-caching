#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Sigma-relaxed lemma implementations.

Only has implementations for algorithms we still expect to use.
Each certifies a hit once a relaxed threshold is reached instead of requiring 
the full N. sigma=1.0 reproduces the corresponding exact function's behavior exactly.
"""

import math

import numpy as np
from scipy.special import betainc
from typing import Callable, List, Optional, Tuple

from .distance_metrics import get_distance_function, DistanceMetric
from .kv_cache import CacheEntry
from .lemmas import (
    _batch_distances,
    _cig_distances,
    _entry_vector_id,
    _sort_vectors_by_distance,
)

#-------------------------------------------------------------------------------

def lemma1_no_union_sigma(
    query: np.ndarray,
    cache_entries: List[CacheEntry],
    N: int,
    sigma: float,
    metric: DistanceMetric = "angular",
    distance_tracker: Optional[Callable[..., float]] = None,
    precomputed_d_Q: Optional[np.ndarray] = None,
    entry_radii: Optional[np.ndarray] = None,
    base_hat: Optional[np.ndarray] = None,
    base_vectors: Optional[np.ndarray] = None,
) -> Tuple[Optional[List[np.ndarray]], bool, dict]:
    """CIG without union, certified when a single entry's passing count is
    >= ceil(sigma * N).

    Identical to lemma1_no_union except the per-entry threshold is
    ceil(sigma * N) instead of N.

    Args:
        query: Test query vector.
        cache_entries: Cached entries to search.
        N: Number of results requested (K_test).
        sigma: Recall threshold in (0, 1]. sigma=1.0 reproduces exact behavior.
        metric: Distance metric.
        distance_tracker: Optional distance call tracker.
        precomputed_d_Q: Optional (n_entries,) array of D(query, entry.query)
            for every entry, e.g. from KVCache.get_d_Q_batch(query). When
            given, the O1 gate check uses this instead of calling
            distance_func once per entry.
        entry_radii: Optional (n_entries,) array of each entry's r_Q, e.g.
            from KVCache.get_entry_radii(). When given together with
            precomputed_d_Q, the O1 gate (d_Q <= r_Q) is vectorized across
            all entries once up front.
        base_hat: Optional shared, pre-normalized base-vector matrix used for
            distance computation on index-only entries (entries whose
            top_k_matrix/top_k_vectors are empty).
        base_vectors: Optional shared, original-scale base-vector matrix used
            to look up actual (non-normalized) result vectors on index-only
            entries.

    Returns:
        (result_vectors, is_hit, metadata). On a miss result_vectors is None.
    """
    threshold = math.ceil(sigma * N)
    raw_distance_func = get_distance_function(metric)
    distance_func = distance_tracker if distance_tracker is not None else raw_distance_func
    is_fast_path = distance_tracker is None

    q_norm = float(np.linalg.norm(query))
    q_hat = (query / q_norm).astype(np.float32) if q_norm != 0.0 else query.copy().astype(np.float32)

    precomputed_base_dist = None
    if is_fast_path and base_hat is not None:
        dots = np.clip(base_hat @ q_hat, -1.0, 1.0)
        if metric == "angular":
            precomputed_base_dist = np.arccos(dots) / np.pi
        else:
            precomputed_base_dist = np.sqrt(np.maximum(2.0 - 2.0 * dots, 0.0))

    metadata = {
        "algorithm": "lemma1_no_union",
        "sigma": sigma,
        "threshold": threshold,
        "checked_entries": 0,
        "cache_hit": False,
    }

    if precomputed_d_Q is not None and entry_radii is not None:
        candidate_idx = np.flatnonzero(precomputed_d_Q <= entry_radii)
    else:
        candidate_idx = range(len(cache_entries))

    for entry_idx in candidate_idx:
        entry = cache_entries[entry_idx]
        r_Q = entry.get_radius()
        d_Q = (
            precomputed_d_Q[entry_idx] if precomputed_d_Q is not None
            else distance_func(query, entry.query)
        )

        # O1: skip if CIG condition impossible (already applied above when
        # vectorized; re-checked here for the fallback path)
        if d_Q > r_Q:
            continue

        distances_to_q = _cig_distances(
            query, entry, metric, distance_func, q_hat, base_hat,
            precomputed_base_dist=precomputed_base_dist,
        )

        # O2: direct threshold mask
        threshold_dist = r_Q - d_Q
        passing = np.where(distances_to_q <= threshold_dist)[0]

        if len(passing) >= threshold:
            passing_dists = distances_to_q[passing]
            order = np.argsort(passing_dists)[:min(N, len(passing))]
            result_positions = passing[order]
            if entry.top_k_vectors:
                result = [entry.top_k_vectors[int(i)] for i in result_positions]
            else:
                result = [base_vectors[entry.top_k_indices[int(i)]] for i in result_positions]
            metadata["result_indices"] = [
                _entry_vector_id(entry, int(i)) for i in result_positions
            ]
            metadata["certified_indices"] = [
                _entry_vector_id(entry, int(i)) for i in passing
            ]
            metadata["checked_entries"] = int(entry_idx) + 1
            metadata["cache_hit"] = True
            metadata["hit_entry_index"] = int(entry_idx)
            return result, True, metadata

    metadata["checked_entries"] = len(cache_entries)
    metadata["certified_indices"] = []
    return None, False, metadata

#-------------------------------------------------------------------------------


def lemma2_no_union_sigma(
    query: np.ndarray,
    cache_entries: List[CacheEntry],
    N: int,
    sigma: float,
    metric: DistanceMetric = "angular",
    distance_tracker: Optional[Callable[..., float]] = None,
    precomputed_d_Q: Optional[np.ndarray] = None,
    base_hat: Optional[np.ndarray] = None,
    base_vectors: Optional[np.ndarray] = None,
) -> Tuple[Optional[List[np.ndarray]], bool, dict]:
    """HGG without union, certified when d_Q < half_gap and K >= ceil(sigma * N).

    Identical to lemma2_no_union except the per-entry K requirement is relaxed
    from K >= N to K >= ceil(sigma * N). 

    Args:
        query: Test query vector.
        cache_entries: Cached entries to search.
        N: Number of results requested (K_test).
        sigma: Recall threshold in (0, 1]. sigma=1.0 reproduces exact behavior.
        metric: Distance metric.
        distance_tracker: Optional distance call tracker.
        precomputed_d_Q: Optional (n_entries,) array of D(query, entry.query)
            for every entry, e.g. from KVCache.get_d_Q_batch(query).
        base_hat: Optional shared, pre-normalized base-vector matrix used for
            distance computation on index-only entries.
        base_vectors: Optional shared, original-scale base-vector matrix used
            to look up actual (non-normalized) result vectors on index-only
            entries.

    Returns:
        (result_vectors, is_hit, metadata). On a miss result_vectors is None.
    """
    threshold = math.ceil(sigma * N)
    raw_distance_func = get_distance_function(metric)
    distance_func = distance_tracker if distance_tracker is not None else raw_distance_func

    metadata = {
        "algorithm": "lemma2_no_union",
        "sigma": sigma,
        "threshold": threshold,
        "checked_entries": 0,
        "cache_hit": False,
    }

    for entry_idx, entry in enumerate(cache_entries):
        metadata["checked_entries"] += 1
        Q, half_gap = entry.query, entry.half_gap
        if entry.top_k_matrix is not None:
            R_Q_dist = entry.top_k_matrix
        elif entry.top_k_vectors:
            R_Q_dist = entry.top_k_vectors
        else:
            R_Q_dist = base_hat[np.asarray(entry.top_k_indices)]
        R_Q_result = (
            entry.top_k_vectors if entry.top_k_vectors
            else base_vectors[np.asarray(entry.top_k_indices)]
        )
        d_Q = (
            precomputed_d_Q[entry_idx] if precomputed_d_Q is not None
            else distance_func(query, Q)
        )
        K = len(R_Q_dist)

        if d_Q < half_gap and K >= threshold:
            distances_to_q = _batch_distances(query, R_Q_dist, metric, raw_distance_func)
            sorted_indices = np.argsort(distances_to_q)
            result_positions = [int(sorted_indices[i]) for i in range(min(N, K))]
            result = [R_Q_result[i] for i in result_positions]
            metadata["result_indices"] = [
                _entry_vector_id(entry, i) for i in result_positions
            ]
            metadata["certified_indices"] = [
                _entry_vector_id(entry, i) for i in range(K)
            ]
            metadata["cache_hit"] = True
            metadata["hit_entry_index"] = entry_idx
            return result, True, metadata

    metadata["certified_indices"] = []
    return None, False, metadata

#-------------------------------------------------------------------------------

def _chord_distance_from_angular_scalar(r: float) -> float:
    """Get Euclidean chord distance, this should be moved to a helper module becuase
    it is currently duplicated"""
    return 2.0 * math.sin(r * math.pi / 2.0)


def _chord_distance_from_angular_array(r: np.ndarray) -> np.ndarray:
    """Batched version of _chord_distance_from_angular_scalar, for vectorized calls."""
    r = np.asarray(r, dtype=np.float64)
    return 2.0 * np.sin(r * np.pi / 2.0)


def _solid_ball_intersection_fraction_array(
    dist: np.ndarray, R1: np.ndarray, R2: float, n: int,
) -> np.ndarray:
    """Score the intersection of two solid balls in R^n. Scores
    every entry in one batched call. Should be moved to a shared helper"""
    dist = np.asarray(dist, dtype=np.float64)
    R1 = np.broadcast_to(np.asarray(R1, dtype=np.float64), dist.shape).copy()
    score = np.zeros_like(dist)

    mask_zero = dist >= R1 + R2
    mask_one = (dist + R2 <= R1) & ~mask_zero
    mask_ratio = (dist + R1 <= R2) & ~mask_zero & ~mask_one
    mask_degenerate = (dist < 1e-12) & ~mask_zero & ~mask_one & ~mask_ratio
    mask_general = ~(mask_zero | mask_one | mask_ratio | mask_degenerate)

    ratio_pow = (R1 / R2) ** n
    score[mask_one] = 1.0
    score[mask_ratio] = ratio_pow[mask_ratio]
    score[mask_degenerate] = np.where(
        R1[mask_degenerate] >= R2, 1.0, ratio_pow[mask_degenerate],
    )

    if np.any(mask_general):
        dg = dist[mask_general]
        r1g = R1[mask_general]
        d1 = (dg ** 2 + r1g ** 2 - R2 ** 2) / (2.0 * dg)
        d2 = dg - d1

        def J(R, di):
            x = np.clip(1.0 - (di / R) ** 2, 0.0, 1.0)
            a = (n + 1) / 2.0
            ib = betainc(a, 0.5, x)
            return np.where(di >= 0, 0.5 * ib, 1.0 - 0.5 * ib)

        score[mask_general] = ratio_pow[mask_general] * J(r1g, d1) + J(R2, d2)

    return score

def fit_r_t_hat(train_r_t: np.ndarray, percentile: float) -> float:
    """Workload-calibrated top-K radius estimate for
    overlap_threshold_hit_policy."""
    return float(np.percentile(train_r_t, percentile))


def overlap_threshold_hit_policy(
    query: np.ndarray,
    cache_entries: List[CacheEntry],
    N: int,
    sigma: float,
    metric: DistanceMetric = "angular",
    distance_tracker: Optional[Callable[..., float]] = None,
    precomputed_d_Q: Optional[np.ndarray] = None,
    base_hat: Optional[np.ndarray] = None,
    base_vectors: Optional[np.ndarray] = None,
    r_t_hat: Optional[float] = None,
    threshold: float = 0.9,
    precomputed_r_c: Optional[np.ndarray] = None,
) -> Tuple[Optional[List[np.ndarray]], bool, dict]:
    """Prototype hit policy that scores every cache entry by the solid-ball intersection fraction
    between the entry's own radius and a FIXED radius estimate (r_t_hat), reports a hit
    if the best entry's score exceeds `threshold`, and returns that entry's cached vectors as the answer."""
    if r_t_hat is None:
        raise ValueError("overlap_threshold_hit_policy requires r_t_hat (see fit_r_t_hat)")
    if sigma != 1.0:
        raise ValueError("overlap_threshold_hit_policy does not support sigma-relaxation")
    if metric != "angular":
        raise ValueError(f"overlap_threshold_hit_policy requires metric='angular', got {metric!r}")

    raw_distance_func = get_distance_function(metric)
    distance_func = distance_tracker if distance_tracker is not None else raw_distance_func

    metadata = {
        "algorithm": "overlap_threshold",
        "threshold": threshold,
        "r_t_hat": r_t_hat,
        "checked_entries": 0,
        "cache_hit": False,
        "best_score": None,
        "hit_entry_index": None,
    }

    if not cache_entries:
        metadata["certified_indices"] = []
        return None, False, metadata

    d = query.shape[0]
    R2 = _chord_distance_from_angular_scalar(r_t_hat)

    r_c_array = (
        np.asarray(precomputed_r_c, dtype=np.float64) if precomputed_r_c is not None
        else np.array([e.get_radius() for e in cache_entries], dtype=np.float64)
    )
    if precomputed_d_Q is not None:
        D_array = np.asarray(precomputed_d_Q, dtype=np.float64)
    else:
        D_array = np.array(
            [distance_func(query, e.query) for e in cache_entries], dtype=np.float64,
        )
    metadata["checked_entries"] = len(cache_entries)

    R1_array = _chord_distance_from_angular_array(r_c_array)
    dist_array = _chord_distance_from_angular_array(D_array)
    scores = _solid_ball_intersection_fraction_array(dist_array, R1_array, R2, d)
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    metadata["best_score"] = best_score
    metadata["hit_entry_index"] = best_idx

    if best_score > threshold:
        entry = cache_entries[best_idx]
        # Nearest-N-by-actual-distance
        if entry.top_k_matrix is not None:
            R_Q_dist = entry.top_k_matrix
        elif entry.top_k_vectors:
            R_Q_dist = entry.top_k_vectors
        else:
            R_Q_dist = base_hat[np.asarray(entry.top_k_indices)]
        R_Q_result = (
            entry.top_k_vectors if entry.top_k_vectors
            else base_vectors[np.asarray(entry.top_k_indices)]
        )
        distances_to_q = _batch_distances(query, R_Q_dist, metric, raw_distance_func)
        n_available = len(R_Q_dist)
        n_take = min(N, n_available)
        
        if n_take < n_available:
            nearest = np.argpartition(distances_to_q, n_take - 1)[:n_take]
            sorted_indices = nearest[np.argsort(distances_to_q[nearest])]
        else:
            sorted_indices = np.argsort(distances_to_q)
        result_positions = [int(sorted_indices[i]) for i in range(n_take)]
        result = [R_Q_result[i] for i in result_positions]
        metadata["cache_hit"] = True
        metadata["certified_indices"] = [_entry_vector_id(entry, i) for i in result_positions]
        metadata["result_indices"] = list(metadata["certified_indices"])
        return result, True, metadata

    metadata["certified_indices"] = []
    return None, False, metadata
