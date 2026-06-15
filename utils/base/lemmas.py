#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import numpy as np
from typing import Callable, List, Optional, Tuple
from .distance_metrics import get_distance_function, DistanceMetric
from .kv_cache import CacheEntry

#-------------------------------------------------------------------------------

def _batch_distances(
    query: np.ndarray,
    vectors: List[np.ndarray],
    metric: DistanceMetric,
    distance_func: Callable,
) -> List[float]:
    """
    Compute distances from query to each vector in vectors.

    Uses a vectorized numpy path when distance_func is the untracked distance
    function (i.e., no distance_tracker is active). Falls back to per-call
    iteration when a tracker wraps the function, preserving count accuracy.
    """
    raw_func = get_distance_function(metric)
    if distance_func is raw_func:
        mat = np.array(vectors)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        mat_hat = mat / norms
        q_norm = float(np.linalg.norm(query))
        q_hat = query / q_norm if q_norm != 0.0 else query.copy()
        if metric == "angular":
            dots = np.clip(mat_hat @ q_hat, -1.0, 1.0)
            return (np.arccos(dots) / np.pi).tolist()
        else:
            return np.linalg.norm(mat_hat - q_hat, axis=1).tolist()
    return [distance_func(query, v) for v in vectors]

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
    sorted_distances: List[float],
    r_Q: float,
    d_q: float
) -> int:
    """
    Binary search to find the last index where D(q, E_i) + d_q < r_Q.
    
    Since sorted_distances is sorted in ascending order by D(q, E_i),
    this finds the rightmost index satisfying the circular inclusion condition.
    
    Args:
        sorted_distances: Distances from q to each vector, sorted ascending
        r_Q: Cache radius (distance from Q to its K-th result)
        d_q: Distance from q to Q
        
    Returns:
        Last index satisfying the condition, or -1 if none satisfy
    """
    left, right = 0, len(sorted_distances) - 1
    result = -1
    
    while left <= right:
        mid = (left + right) // 2
        if sorted_distances[mid] + d_q < r_Q:
            # found vector satisfying lemma, try to find a later index
            result = mid
            left = mid + 1
        else:
            # not satisfied, search earlier
            right = mid - 1
    
    return result

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
        
        Q, R_Q = entry.query, entry.top_k_vectors
        r_Q, d_Q = entry.get_radius(), distance_func(query, Q)   # D(Q, E_K), D(q, Q)

        # sort R_Q by distance to query q and find last index using binary search
        distances_to_q = _batch_distances(query, R_Q, metric, distance_func)
        sorted_indices = np.argsort(distances_to_q)
        sorted_R_Q = [R_Q[i] for i in sorted_indices]
        sorted_distances = [distances_to_q[i] for i in sorted_indices]
        theta = binary_search_last_index(sorted_distances, r_Q, d_Q)
        metadata["theta_values"].append(theta)

        # unite certified vectors and handle duplicates by stable vector index
        for i in range(theta + 1):
            original_i = int(sorted_indices[i])
            vec_id = _entry_vector_id(entry, original_i)
            if vec_id not in R_q_set:
                R_q.append(sorted_R_Q[i])
                R_q_indices.append(vec_id)
                R_q_set.add(vec_id)
                metadata["vectors_added"] += 1

        # check if we have enough certified vectors for a full cache hit
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
    distance_func = get_distance_function(metric)
    if distance_tracker is not None:
        distance_func = distance_tracker
    
    metadata = {
        "algorithm": "2: CIG + No Union",
        "checked_entries": 0,
        "cache_hit": False
    }
    
    for entry_idx, entry in enumerate(cache_entries):
        metadata["checked_entries"] += 1
        Q, R_Q = entry.query, entry.top_k_vectors
        r_Q, d_Q = entry.get_radius(), distance_func(query, Q)   # D(Q, E_K), D(q, Q)
        
        # sort R_Q by distance to query q
        distances_to_q = _batch_distances(query, R_Q, metric, distance_func)
        sorted_indices = np.argsort(distances_to_q)
        sorted_distances = [distances_to_q[i] for i in sorted_indices]

        # find last index using binary search
        theta = binary_search_last_index(sorted_distances, r_Q, d_Q)
        
        # if the cached query has enough certified vectors, we have a full cache hit
        if theta >= 0 and theta + 1 >= N:
            result_positions = [int(sorted_indices[i]) for i in range(N)]
            result = [R_Q[i] for i in result_positions]
            metadata["result_indices"] = [
                _entry_vector_id(entry, i) for i in result_positions
            ]
            metadata["cache_hit"], metadata["hit_entry_index"] = True, entry_idx
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
        
        # check if the new query is within the half-gap of the cached query
        if d_Q < half_gap:
            # we have a full cache hit
            if K == N:
                metadata["cache_hit"] = True
                metadata["hit_entry_index"] = entry_idx
                result, result_indices = _sort_vectors_by_distance(
                    query, R_Q, [_entry_vector_id(entry, i) for i in range(K)],
                    raw_distance_func, N, metric,
                )
                metadata["result_indices"] = result_indices
                return result, True, metadata

            # we have a full cache hit but need to return only N vectors
            elif K > N:
                metadata["cache_hit"] = True
                metadata["hit_entry_index"] = entry_idx
                result, result_indices = _sort_vectors_by_distance(
                    query, R_Q, [_entry_vector_id(entry, i) for i in range(K)],
                    raw_distance_func, N, metric,
                )
                metadata["result_indices"] = result_indices
                return result, True, metadata

            # we have a partial cache hit and need to union R_Q into R_q
            else:
                for i in range(K):
                    vec_id = _entry_vector_id(entry, i)
                    if vec_id not in R_q_set:
                        R_q.append(R_Q[i])
                        R_q_indices.append(vec_id)
                        R_q_set.add(vec_id)
                        metadata["vectors_added"] += 1

                # check if we have enough certified vectors for a full cache hit
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
        
        # check if the new query is within the half-gap of the cached query
        # and if the cached query has enough certified vectors
        if d_Q < half_gap and len(R_Q) >= N:
            distances_to_q = _batch_distances(query, R_Q, metric, raw_distance_func)
            sorted_indices = np.argsort(distances_to_q)
            result_positions = [int(sorted_indices[i]) for i in range(N)]
            result = [R_Q[i] for i in result_positions]
            metadata["result_indices"] = [
                _entry_vector_id(entry, i) for i in result_positions
            ]
            metadata["cache_hit"], metadata["hit_entry_index"] = True, entry_idx
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
            
            # we have a full cache hit
            if K == N:
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

            # we have a full cache hit but need to return only N vectors
            elif K > N:
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

            # we have a partial cache hit and need to union R_Q into R_q
            else:
                for i in range(K):
                    vec_id = _entry_vector_id(entry, i)
                    if vec_id not in R_q_set:
                        R_q.append(R_Q[i])
                        R_q_indices.append(vec_id)
                        R_q_set.add(vec_id)
                        metadata["vectors_added"] += 1

                # check if we have enough certified vectors for a full cache hit
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

        # sort R_Q by distance to query q for CIG certification
        distances_to_q = _batch_distances(query, R_Q, metric, distance_func)
        sorted_indices = np.argsort(distances_to_q)
        sorted_R_Q = [R_Q[i] for i in sorted_indices]
        sorted_distances = [distances_to_q[i] for i in sorted_indices]

        # find the farthest certified index using binary search
        theta = binary_search_last_index(sorted_distances, r_Q, d_Q)

        # unite certified vectors and handle duplicates by stable vector index
        for i in range(theta + 1):
            original_i = int(sorted_indices[i])
            vec_id = _entry_vector_id(entry, original_i)
            if vec_id not in R_q_set:
                R_q.append(sorted_R_Q[i])
                R_q_indices.append(vec_id)
                R_q_set.add(vec_id)
                metadata["vectors_added"] += 1

        # check if we have enough certified vectors for a full cache hit
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
