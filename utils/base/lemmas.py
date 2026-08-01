#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import numpy as np
from typing import Callable, List, Optional, Tuple
from .distance_metrics import get_distance_function, DistanceMetric
from .kv_cache import CacheEntry

#-------------------------------------------------------------------------------

def _batch_distances(
    query: np.ndarray,
    vectors,
    metric: DistanceMetric,
    distance_func: Callable,
) -> np.ndarray:
    """
    Compute distances from query to each vector in vectors.

    Uses a vectorized numpy path when distance_func is the untracked distance
    function (i.e., no distance_tracker is active). Falls back to per-call
    iteration when a tracker wraps the function, preserving count accuracy.

    If vectors is a 2D numpy array (pre-stacked matrix), it is used directly
    without copying. Otherwise it is a list of arrays and np.array() is called.

    Returns a numpy array of distances.
    """
    raw_func = get_distance_function(metric)
    if distance_func is raw_func:
        if isinstance(vectors, np.ndarray) and vectors.ndim == 2:
            mat = vectors
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            mat_hat = mat / norms
        else:
            mat = np.array(vectors)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            mat_hat = mat / norms
        q_norm = float(np.linalg.norm(query))
        q_hat = query / q_norm if q_norm != 0.0 else query.copy()
        if metric == "angular":
            dots = np.clip(mat_hat @ q_hat, -1.0, 1.0)
            return np.arccos(dots) / np.pi
        else:
            return np.linalg.norm(mat_hat - q_hat, axis=1)
    return np.asarray([distance_func(query, v) for v in vectors])

#-------------------------------------------------------------------------------

def _entry_vector_id(entry: CacheEntry, position: int) -> int:
    """Return the stable vector index for a cached vector."""
    if entry.top_k_indices is not None:
        return int(entry.top_k_indices[position])
    return id(entry.top_k_vectors[position])

def _sort_vectors_by_distance(
    query: np.ndarray,
    vectors: List[np.ndarray],
    vector_indices: List[int],
    distance_func: Callable[..., float],
    N: int,
    metric: DistanceMetric = "angular",
) -> Tuple[List[np.ndarray], List[int]]:
    distances = _batch_distances(query, vectors, metric, distance_func)
    sorted_positions = np.argsort(distances)[:N]
    return (
        [vectors[i] for i in sorted_positions],
        [int(vector_indices[i]) for i in sorted_positions]
    )

#-------------------------------------------------------------------------------

def binary_search_last_index(
    sorted_distances,
    r_Q: float,
    d_q: float
) -> int:
    """
    Binary search to find the last index where D(q, E_i) + d_q < r_Q.

    Kept for backward compatibility / tests. The hot-path lemma functions use
    a direct threshold mask (np.where) instead of sorting + binary search.
    """
    left, right = 0, len(sorted_distances) - 1
    result = -1
    while left <= right:
        mid = (left + right) // 2
        if sorted_distances[mid] + d_q < r_Q:
            result = mid
            left = mid + 1
        else:
            right = mid - 1
    return result

#-------------------------------------------------------------------------------

def _cig_distances(
    query: np.ndarray,
    entry: CacheEntry,
    metric: DistanceMetric,
    distance_func: Callable,
    q_hat: Optional[np.ndarray],
    base_hat: Optional[np.ndarray] = None,
    precomputed_base_dist: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute distances from query to all vectors in a cache entry.

    Uses entry.top_k_matrix (pre-stacked, pre-normalized) when available for
    a single matmul without any copying. Otherwise, when the entry is
    index-only (no private per-entry matrix -- the common case for candidates
    with large, overlapping R_c sets, where a private copy would duplicate
    the same base vectors across many entries), gathers from
    precomputed_base_dist when the caller has already computed
    distance-to-query for the entire base once for this query (the common
    case -- avoids every entry independently repeating an O(rc_size * dim)
    matmul against heavily-overlapping index sets). Falls back to a transient
    fancy-index into a shared, pre-normalized base_hat matrix
    (base_hat[entry.top_k_indices], not retained) followed by a per-entry
    matmul when precomputed_base_dist isn't available. Falls back further to
    _batch_distances(entry.top_k_vectors, ...) when neither a per-entry
    matrix nor a shared base_hat is available.
    """
    is_fast_path = distance_func is get_distance_function(metric)
    if entry.top_k_matrix is not None and is_fast_path:
        mat = entry.top_k_matrix
    elif (
        not entry.top_k_vectors and base_hat is not None
        and entry.top_k_indices is not None and is_fast_path
    ):
        # index-only entry (no real top_k_vectors/top_k_matrix of its own) --
        # only then fall back to the shared base_hat lookup. An entry with
        # real top_k_vectors (e.g. built against a dummy/placeholder base_vectors
        # array, as in the ESCI precomputed-GT pipeline) must use its own
        # vectors, not indices into an unrelated shared matrix.
        if precomputed_base_dist is not None:
            # Row-independent gather: base_hat[i] @ q_hat (and the pointwise
            # arccos/sqrt that follows) doesn't depend on any other row, so
            # selecting a subset of rows before or after that computation
            # gives byte-identical values -- this is exact, not approximate.
            return precomputed_base_dist[np.asarray(entry.top_k_indices)]
        mat = base_hat[np.asarray(entry.top_k_indices)]
    else:
        return _batch_distances(query, entry.top_k_vectors, metric, distance_func)

    # q_hat must be provided by the caller when using either fast path above
    dots = np.clip(mat @ q_hat, -1.0, 1.0)
    if metric == "angular":
        return np.arccos(dots) / np.pi
    else:
        # For euclidean on pre-normalized vectors: ||a-b|| = sqrt(2 - 2*dot)
        return np.sqrt(np.maximum(2.0 - 2.0 * dots, 0.0))

#-------------------------------------------------------------------------------

# algorithm 1: CIG with union
def lemma1_circular_inclusion(
    query: np.ndarray,
    cache_entries: List[CacheEntry],
    N: int,
    metric: DistanceMetric = "euclidean",
    distance_tracker: Optional[Callable[..., float]] = None
) -> Tuple[Optional[List[np.ndarray]], bool, dict]:
    """
    Implement Circular Inclusion Guarantee with union.

    Args:
        query: New query vector q
        cache_entries: List of cached queries and their results
        N: Number of top-N results to return
        metric: Distance metric
        distance_tracker: Optional distance calculation tracker

    Returns:
        Tuple of (result_vectors, is_cache_hit, metadata):
        - result_vectors: List of top-N result vectors, sorted by distance to q
        - is_cache_hit: Whether the cache was hit
        - metadata: Dictionary with algorithm metadata
    """
    raw_distance_func = get_distance_function(metric)
    distance_func = distance_tracker if distance_tracker is not None else raw_distance_func

    # Pre-normalize query once for the fast matmul path (O3)
    q_norm = float(np.linalg.norm(query))
    q_hat = (query / q_norm).astype(np.float32) if q_norm != 0.0 else query.copy().astype(np.float32)

    R_q = []
    R_q_indices = []
    R_q_set = set()

    metadata = {
        "algorithm": "1: CIG + Union",
        "checked_entries": 0,
        "vectors_added": 0,
        "cache_hit": False,
        "hit_entry_index": None,
        "theta_values": []
    }

    for entry_idx, entry in enumerate(cache_entries):
        metadata["checked_entries"] += 1

        r_Q = entry.get_radius()
        d_Q = distance_func(query, entry.query)

        # O1: if d_Q > r_Q, D(q,v)+d_Q > r_Q for all v (since D(q,v)>=0) -- skip
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

        if len(R_q) >= N:
            metadata["cache_hit"] = True
            metadata["hit_entry_index"] = entry_idx
            metadata["vectors_added"] = len(R_q)
            result, result_indices = _sort_vectors_by_distance(
                query, R_q, R_q_indices, raw_distance_func, N, metric
            )
            metadata["result_indices"] = result_indices
            return result, True, metadata

    # cache miss
    return None, False, metadata

#-------------------------------------------------------------------------------

# algorithm 2: CIG without union
def lemma1_no_union(
    query: np.ndarray,
    cache_entries: List[CacheEntry],
    N: int,
    metric: DistanceMetric = "euclidean",
    distance_tracker: Optional[Callable[..., float]] = None
) -> Tuple[Optional[List[np.ndarray]], bool, dict]:
    """
    Implement Circular Inclusion Guarantee without union.

    Args:
        query: New query vector q
        cache_entries: List of cached queries and their results
        N: Number of top-N results to return
        metric: Distance metric
        distance_tracker: Optional distance calculation tracker

    Returns:
        Tuple of (result_vectors, is_cache_hit, metadata):
        - result_vectors: List of top-N result vectors, sorted by distance to q
        - is_cache_hit: Whether the cache was hit
        - metadata: Dictionary with algorithm metadata
    """
    raw_distance_func = get_distance_function(metric)
    distance_func = distance_tracker if distance_tracker is not None else raw_distance_func

    q_norm = float(np.linalg.norm(query))
    q_hat = (query / q_norm).astype(np.float32) if q_norm != 0.0 else query.copy().astype(np.float32)

    metadata = {
        "algorithm": "2: CIG + No Union",
        "checked_entries": 0,
        "cache_hit": False
    }

    for entry_idx, entry in enumerate(cache_entries):
        metadata["checked_entries"] += 1
        r_Q = entry.get_radius()
        d_Q = distance_func(query, entry.query)

        # O1: skip if CIG condition impossible
        if d_Q > r_Q:
            continue

        distances_to_q = _cig_distances(query, entry, metric, distance_func, q_hat)

        # O2: direct threshold mask
        threshold_dist = r_Q - d_Q
        passing = np.where(distances_to_q <= threshold_dist)[0]

        if len(passing) >= N:
            # Sort just the passing subset to return the N closest
            passing_dists = distances_to_q[passing]
            order = np.argsort(passing_dists)[:N]
            result_positions = passing[order]
            R_Q = entry.top_k_vectors
            result = [R_Q[int(i)] for i in result_positions]
            metadata["result_indices"] = [
                _entry_vector_id(entry, int(i)) for i in result_positions
            ]
            metadata["cache_hit"] = True
            metadata["hit_entry_index"] = entry_idx
            return result, True, metadata

    # cache miss
    return None, False, metadata

#-------------------------------------------------------------------------------

# algorithm 3: HGG with union
def lemma2_half_gap(
    query: np.ndarray,
    cache_entries: List[CacheEntry],
    N: int,
    metric: DistanceMetric = "euclidean",
    distance_tracker: Optional[Callable[..., float]] = None
) -> Tuple[Optional[List[np.ndarray]], bool, dict]:
    """
    Implement Half-Gap Guarantee with union.

    Args:
        query: New query vector q
        cache_entries: List of cached queries and their results
        N: Number of top-N results to return
        metric: Distance metric
        distance_tracker: Optional distance calculation tracker

    Returns:
        Tuple of (result_vectors, is_cache_hit, metadata):
        - result_vectors: List of top-N result vectors, sorted by distance to q
        - is_cache_hit: Whether the cache was hit
        - metadata: Dictionary with algorithm metadata
    """
    raw_distance_func = get_distance_function(metric)
    distance_func = distance_tracker if distance_tracker is not None else raw_distance_func

    R_q = []
    R_q_indices = []
    R_q_set = set()

    metadata = {
        "algorithm": "3: HGG + Union",
        "checked_entries": 0,
        "vectors_added": 0,
        "cache_hit": False,
        "hit_entry_index": None
    }

    for entry_idx, entry in enumerate(cache_entries):
        metadata["checked_entries"] += 1

        Q, R_Q = entry.query, entry.top_k_vectors
        d_Q, half_gap = distance_func(query, Q), entry.get_half_gap()
        K = len(R_Q)

        if d_Q < half_gap:
            if K >= N:
                metadata["cache_hit"] = True
                metadata["hit_entry_index"] = entry_idx
                result, result_indices = _sort_vectors_by_distance(
                    query, R_Q, [_entry_vector_id(entry, i) for i in range(K)],
                    raw_distance_func, N, metric,
                )
                metadata["result_indices"] = result_indices
                return result, True, metadata

            for i in range(K):
                vec_id = _entry_vector_id(entry, i)
                if vec_id not in R_q_set:
                    R_q.append(R_Q[i])
                    R_q_indices.append(vec_id)
                    R_q_set.add(vec_id)
                    metadata["vectors_added"] += 1

            if len(R_q) >= N:
                metadata["cache_hit"] = True
                metadata["vectors_added"] = len(R_q)
                metadata["hit_entry_index"] = entry_idx
                result, result_indices = _sort_vectors_by_distance(
                    query, R_q, R_q_indices, raw_distance_func, N, metric,
                )
                metadata["result_indices"] = result_indices
                return result, True, metadata

    # cache miss
    return None, False, metadata

#-------------------------------------------------------------------------------

# algorithm 4: HGG without union
def lemma2_no_union(
    query: np.ndarray,
    cache_entries: List[CacheEntry],
    N: int,
    metric: DistanceMetric = "euclidean",
    distance_tracker: Optional[Callable[..., float]] = None
) -> Tuple[Optional[List[np.ndarray]], bool, dict]:
    """
    Implement Half-Gap Guarantee without union.

    Args:
        query: New query vector q
        cache_entries: List of cached queries and their results
        N: Number of top-N results to return
        metric: Distance metric
        distance_tracker: Optional distance calculation tracker

    Returns:
        Tuple of (result_vectors, is_cache_hit, metadata):
        - result_vectors: List of top-N result vectors, sorted by distance to q
        - is_cache_hit: Whether the cache was hit
        - metadata: Dictionary with algorithm metadata
    """
    raw_distance_func = get_distance_function(metric)
    distance_func = distance_tracker if distance_tracker is not None else raw_distance_func

    metadata = {
        "algorithm": "4: HGG + No Union",
        "checked_entries": 0,
        "cache_hit": False
    }

    for entry_idx, entry in enumerate(cache_entries):
        metadata["checked_entries"] += 1
        Q, R_Q, half_gap = entry.query, entry.top_k_vectors, entry.half_gap
        d_Q = distance_func(query, Q)

        if d_Q < half_gap and len(R_Q) >= N:
            distances_to_q = _batch_distances(query, R_Q, metric, raw_distance_func)
            sorted_indices = np.argsort(distances_to_q)
            result_positions = [int(sorted_indices[i]) for i in range(N)]
            result = [R_Q[i] for i in result_positions]
            metadata["result_indices"] = [
                _entry_vector_id(entry, i) for i in result_positions
            ]
            metadata["cache_hit"] = True
            metadata["hit_entry_index"] = entry_idx
            return result, True, metadata

    # cache miss
    return None, False, metadata

#-------------------------------------------------------------------------------

# algorithm 5: Combined algorithm with union
def combined_algorithm(
    query: np.ndarray,
    cache_entries: List[CacheEntry],
    N: int,
    metric: DistanceMetric = "euclidean",
    distance_tracker: Optional[Callable[..., float]] = None
) -> Tuple[Optional[List[np.ndarray]], bool, dict]:
    """
    Implement Combined algorithm with union by trying HGG first, then fallback to CIG.

    Args:
        query: New query vector q
        cache_entries: List of cached queries and their results
        N: Number of top-N results to return
        metric: Distance metric
        distance_tracker: Optional distance calculation tracker

    Returns:
        Tuple of (result_vectors, is_cache_hit, metadata):
        - result_vectors: List of top-N result vectors, sorted by distance to q
        - is_cache_hit: Whether the cache was hit
        - metadata: Dictionary with algorithm metadata
    """
    raw_distance_func = get_distance_function(metric)
    distance_func = distance_tracker if distance_tracker is not None else raw_distance_func

    q_norm = float(np.linalg.norm(query))
    q_hat = (query / q_norm).astype(np.float32) if q_norm != 0.0 else query.copy().astype(np.float32)

    R_q = []
    R_q_indices = []
    R_q_set = set()

    metadata = {
        "algorithm": "5: Combined + Union",
        "checked_entries": 0,
        "lemma2_attempts": 0,
        "lemma2_successes": 0,
        "lemma1_attempts": 0,
        "vectors_added": 0,
        "cache_hit": False,
        "hit_entry_index": None,
        "hit_lemma": None
    }

    for entry_idx, entry in enumerate(cache_entries):
        metadata["checked_entries"] += 1

        Q, R_Q = entry.query, entry.top_k_vectors
        d_Q, K = distance_func(query, Q), len(R_Q)

        # try HGG first
        half_gap = entry.get_half_gap()
        if d_Q < half_gap:
            metadata["lemma2_attempts"] += 1

            if K >= N:
                metadata["cache_hit"] = True
                metadata["hit_entry_index"] = entry_idx
                metadata["hit_lemma"] = "lemma2"
                metadata["lemma2_successes"] += 1
                result, result_indices = _sort_vectors_by_distance(
                    query, R_Q, [_entry_vector_id(entry, i) for i in range(K)],
                    raw_distance_func, N, metric,
                )
                metadata["result_indices"] = result_indices
                return result, True, metadata

            for i in range(K):
                vec_id = _entry_vector_id(entry, i)
                if vec_id not in R_q_set:
                    R_q.append(R_Q[i])
                    R_q_indices.append(vec_id)
                    R_q_set.add(vec_id)
                    metadata["vectors_added"] += 1

            if len(R_q) >= N:
                metadata["cache_hit"] = True
                metadata["vectors_added"] = len(R_q)
                metadata["hit_entry_index"] = entry_idx
                result, result_indices = _sort_vectors_by_distance(
                    query, R_q, R_q_indices, raw_distance_func, N, metric,
                )
                metadata["result_indices"] = result_indices
                return result, True, metadata

            continue

        # fallback to CIG
        metadata["lemma1_attempts"] += 1
        r_Q = entry.get_radius()

        # O1: skip if CIG condition impossible
        if d_Q > r_Q:
            continue

        distances_to_q = _cig_distances(query, entry, metric, distance_func, q_hat)

        # O2: direct threshold mask
        threshold_dist = r_Q - d_Q
        passing = np.where(distances_to_q <= threshold_dist)[0]

        for original_i in passing:
            vec_id = _entry_vector_id(entry, int(original_i))
            if vec_id not in R_q_set:
                R_q.append(R_Q[int(original_i)])
                R_q_indices.append(vec_id)
                R_q_set.add(vec_id)
                metadata["vectors_added"] += 1

        if len(R_q) >= N:
            metadata["cache_hit"] = True
            metadata["hit_entry_index"] = entry_idx
            metadata["hit_lemma"] = "lemma1"
            metadata["vectors_added"] = len(R_q)
            result, result_indices = _sort_vectors_by_distance(
                query, R_q, R_q_indices, raw_distance_func, N, metric,
            )
            metadata["result_indices"] = result_indices
            return result, True, metadata

    # cache miss
    return None, False, metadata

#-------------------------------------------------------------------------------
