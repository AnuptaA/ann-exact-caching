#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Sigma-relaxed lemma implementations.

lemma1_sigma mirrors lemma1_circular_inclusion (union) from lemmas.py, kept for
backward compatibility -- the canonical algorithms going forward are the
no-union variants: lemma1_no_union_sigma (CIG) and lemma2_no_union_sigma (HGG).
Each certifies a hit once its relaxed per-entry threshold is reached instead of
requiring the full N, enabling sub-100% recall certification. sigma=1.0
reproduces the corresponding exact (non-sigma) function's behavior exactly.

The helper functions (_batch_distances, _cig_distances, etc.) are imported from
lemmas.py -- only the hit-detection threshold changes.
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


def lemma1_sigma(
    query: np.ndarray,
    cache_entries: List[CacheEntry],
    N: int,
    sigma: float,
    metric: DistanceMetric = "angular",
    distance_tracker: Optional[Callable[..., float]] = None,
    precomputed_d_Q: Optional[np.ndarray] = None,
) -> Tuple[Optional[List[np.ndarray]], bool, dict]:
    """CIG with union, certified when |R_q| >= ceil(sigma * N).

    Identical to lemma1_circular_inclusion except the termination threshold is
    ceil(sigma * N) instead of N. On both hit and miss paths, certified_indices
    is set in metadata so callers can compute actual recall.

    Args:
        query: Test query vector.
        cache_entries: Cached entries to search.
        N: Number of results requested (K_test).
        sigma: Recall threshold in (0, 1]. sigma=1.0 reproduces exact behavior.
        metric: Distance metric.
        distance_tracker: Optional distance call tracker.
        precomputed_d_Q: Optional array of shape (n_entries,) containing pre-computed
            D(query, Q_i) for all cache entries. When provided, avoids per-entry scalar
            distance calls (O4). Must align with cache_entries order.

    Returns:
        (result_vectors, is_hit, metadata). On a miss result_vectors is None.
        metadata always contains 'certified_indices'.
    """
    threshold = math.ceil(sigma * N)
    raw_distance_func = get_distance_function(metric)
    distance_func = distance_tracker if distance_tracker is not None else raw_distance_func

    # Pre-normalize query once for the fast matmul path (O3)
    q_norm = float(np.linalg.norm(query))
    q_hat = (query / q_norm).astype(np.float32) if q_norm != 0.0 else query.copy().astype(np.float32)

    R_q: List[np.ndarray] = []
    R_q_indices: List[int] = []
    R_q_set: set = set()

    metadata = {
        "algorithm": "lemma1",
        "sigma": sigma,
        "threshold": threshold,
        "checked_entries": 0,
        "vectors_added": 0,
        "cache_hit": False,
        "hit_entry_index": None,
        "theta_values": [],
    }

    for entry_idx, entry in enumerate(cache_entries):
        metadata["checked_entries"] += 1

        r_Q = entry.get_radius()
        if precomputed_d_Q is not None:
            d_Q = float(precomputed_d_Q[entry_idx])
        else:
            d_Q = distance_func(query, entry.query)

        # O1: if d_Q > r_Q, CIG condition D(q,v)+d_Q <= r_Q is impossible for all v
        if d_Q > r_Q:
            metadata["theta_values"].append(-1)
            continue

        distances_to_q = _cig_distances(query, entry, metric, distance_func, q_hat)

        # O2: direct threshold mask instead of sort + binary search
        threshold_dist = r_Q - d_Q
        passing = np.where(distances_to_q <= threshold_dist)[0]
        metadata["theta_values"].append(int(len(passing)) - 1)

        for original_i in passing:
            vec_id = _entry_vector_id(entry, int(original_i))
            if vec_id not in R_q_set:
                R_q.append(entry.top_k_vectors[int(original_i)])
                R_q_indices.append(vec_id)
                R_q_set.add(vec_id)
                metadata["vectors_added"] += 1

        if len(R_q) >= threshold:
            metadata["cache_hit"] = True
            metadata["hit_entry_index"] = entry_idx
            metadata["vectors_added"] = len(R_q)
            metadata["certified_indices"] = list(R_q_indices)
            result, result_indices = _sort_vectors_by_distance(
                query, R_q, R_q_indices, raw_distance_func, N, metric
            )
            metadata["result_indices"] = result_indices
            return result, True, metadata

    metadata["certified_indices"] = list(R_q_indices)
    return None, False, metadata

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
    ceil(sigma * N) instead of N. On hit, returns the closest min(N, len(passing))
    vectors from the certifying entry. On both hit and miss, metadata contains
    'certified_indices' (empty on miss, since no_union does not accumulate
    partial credit across entries -- a miss means no single entry reached
    threshold on its own).

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
            all entries once up front instead of a per-entry Python
            comparison -- entries failing the gate never reach the loop
            body at all, exact same condition, just batched.
        base_hat: Optional shared, pre-normalized base-vector matrix used for
            distance computation on index-only entries (entries whose
            top_k_matrix/top_k_vectors are empty -- see CacheEntry).
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

    # Precompute distance-to-query for the entire base once per query
    # (instead of once per index-only cache entry, whose R_c sets overlap
    # heavily by design) -- same fast-path condition _cig_distances uses
    # internally, so distance-call-tracking semantics are unaffected.
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

    # Vectorized O1 gate: when both arrays are available, compute which
    # entries pass (d_Q <= r_Q) once up front instead of a per-entry Python
    # comparison -- flatnonzero preserves ascending order, so iterating over
    # candidate_idx is identical to filtering the original enumerate() loop.
    # Falls back to the per-entry loop (unvectorized gate) when either array
    # is missing, e.g. a distance_tracker is active (per-call counting must
    # stay exact) or the function is called standalone without them.
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

        # At threshold=0 (N=0, e.g. population's K_test=0 containment
        # mode), len(passing) >= 0 is always true once an entry has already
        # passed the O1 containment gate above -- this correctly reduces the
        # hit condition to "contained alone is sufficient", with no
        # additional special-casing needed here.
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
            # checked_entries matches the original per-entry loop's count:
            # every entry up to and including this one would have been
            # reached (regardless of which failed O1), not just the ones
            # that passed the vectorized gate.
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
    from K >= N to K >= ceil(sigma * N). The HGG guarantee means that once
    d_Q < half_gap, the entry's entire cached top-K is exactly correct -- so
    fewer than N of them can still be returned as a certified partial-recall
    hit. On hit, returns the closest min(N, K) vectors and metadata
    ['certified_indices'] is the entry's full top-K (all guaranteed correct).
    On miss, certified_indices is empty (no partial credit across entries).

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
        # Symmetric with _cig_distances' priority order (utils/base/lemmas.py):
        # a real per-entry top_k_matrix/top_k_vectors always wins over the
        # shared base_hat/base_vectors lookup, regardless of which of the two
        # per-entry fields happens to be set.
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
# Prototype: geometric-overlap threshold policy (uncertified, probabilistic --
# not CIG/HGG). See simulations/population/debug_intersection_calibration.py
# for the calibration study this is based on: intersection fraction between a
# cache entry's own radius and a fixed radius estimate is a strong predictor
# of oracle-hit likelihood (AUC 0.92-1.0 across all tested configs).
#-------------------------------------------------------------------------------


def _chord_distance_from_angular_scalar(r: float) -> float:
    """Euclidean chord distance for angular distance r (normalized units,
    angle/pi): chord = 2*sin(r*pi/2). Duplicated from
    debug_intersection_calibration.py's chord_distance_from_angular
    (array-native there; this module needs a plain-scalar version and
    importing from simulations/ into utils/base/ would invert the
    normal dependency direction). Consolidate into a shared location if
    this prototype graduates past experimental status."""
    return 2.0 * math.sin(r * math.pi / 2.0)


def _chord_distance_from_angular_array(r: np.ndarray) -> np.ndarray:
    """Array-native twin of _chord_distance_from_angular_scalar -- same
    formula, np.sin instead of math.sin so it broadcasts over an entire
    array of angular distances in one call instead of one Python call
    per element."""
    r = np.asarray(r, dtype=np.float64)
    return 2.0 * np.sin(r * np.pi / 2.0)


def _solid_ball_intersection_fraction_scalar(dist: float, R1: float, R2: float, n: int) -> float:
    """Fraction of solid ball 2's (radius R2) volume lying within solid
    ball 1 (radius R1) in R^n. Duplicated from
    debug_intersection_calibration.py's solid_ball_intersection_fraction
    -- see that function's docstring for the derivation; same caveat as
    _chord_distance_from_angular_scalar above about why it's copied
    rather than imported."""
    if dist >= R1 + R2:
        return 0.0
    if dist + R2 <= R1:
        return 1.0
    if dist + R1 <= R2:
        return (R1 / R2) ** n
    if dist < 1e-12:
        return 1.0 if R1 >= R2 else (R1 / R2) ** n

    d1 = (dist ** 2 + R1 ** 2 - R2 ** 2) / (2.0 * dist)
    d2 = dist - d1

    def J(R, di):
        x = min(1.0, max(0.0, 1.0 - (di / R) ** 2))
        a = (n + 1) / 2.0
        ib = betainc(a, 0.5, x)
        return 0.5 * ib if di >= 0 else 1.0 - 0.5 * ib

    return (R1 / R2) ** n * J(R1, d1) + J(R2, d2)


def _solid_ball_intersection_fraction_array(
    dist: np.ndarray, R1: np.ndarray, R2: float, n: int,
) -> np.ndarray:
    """Vectorized twin of _solid_ball_intersection_fraction_scalar: scores
    every entry (dist/R1 vary per entry, R2 fixed) in one batched call via
    np.where branch masks instead of a Python loop calling the scalar
    version once per entry -- betainc is natively array-vectorized in
    scipy, so the branches were the only thing forcing a per-element
    Python call. Bit-identical to calling the scalar version elementwise
    (see the parity test in tests/utils/test_lemmas.py)."""
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
    overlap_threshold_hit_policy: a single percentile of the *cache/
    training* queries' own top-K radii (never the test query's -- that
    would be oracle knowledge unavailable at real decision time)."""
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
    """Prototype hit policy (not CIG/HGG -- uncertified, probabilistic):
    scores every cache entry by the solid-ball intersection fraction
    between the entry's own radius and a FIXED radius estimate
    (r_t_hat), reports a hit if the best entry's score exceeds
    `threshold`, and returns that entry's cached vectors as the answer.

    r_t_hat must be precomputed offline via fit_r_t_hat on cache/training
    query radii -- this function never has access to (and must never be
    given) the test query's own true top-K radius; that's exactly the
    thing a real hit policy can't know in advance. metric must be
    "angular" (the intersection-fraction math assumes it; ValueError
    otherwise). sigma must be 1.0 -- this policy doesn't support
    sigma-relaxation.

    precomputed_r_c: optional array of every cache_entries[i].get_radius()
    -- an entry's own radius never changes across queries, so a caller
    scoring many queries against the same cache_entries (every real
    caller) should compute this array once and pass it in, rather than
    re-fetching it from each entry on every query.

    Returns: (result_vectors, is_hit, metadata). On a miss,
    result_vectors is None and metadata['certified_indices'] is empty
    (uncertified policy -- no partial-credit accumulation across
    entries, matching the no_union lemmas' convention)."""
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

    # Vectorized scoring across all entries at once (see
    # _solid_ball_intersection_fraction_array's docstring) instead of a
    # per-entry Python loop calling scalar math -- the dominant cost at
    # real scale (hundreds of entries x hundreds of test queries).
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
        # Nearest-N-by-actual-distance, not array order: entry.top_k_indices
        # here is the full R_c set (all base vectors within r_c), not
        # pre-sorted by distance to this query -- same pattern as
        # lemma2_no_union_sigma's R_Q_dist/R_Q_result split.
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
        # R_Q_dist is the full R_c set (often >> N), so a full O(n log n)
        # argsort to keep only the nearest n_take is wasted work --
        # argpartition finds the n_take smallest in O(n), then only that
        # small slice needs sorting.
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
