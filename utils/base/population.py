#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from .distance_metrics import DistanceMetric, pairwise_angular_distance
from .kv_cache import CacheEntry

#-------------------------------------------------------------------------------

@dataclass
class CandidateEntry:
    query: np.ndarray
    r_Q: float
    alpha: Optional[float]             # this is deprecated and should be removed
    r_c: float                         # used as cache radius at hit time
    R_c: object                        # bool array (len n_base) when store_rc=True, else None
    rc_size: int                       # R_c.sum(); always set regardless of store_rc
    covered_train_indices: object      # bool array (len n_train_queries), observed indices satisfying CIG
    covered_test_indices: object       # bool array (len n_test_queries), held-out test indices satisfying CIG
    top_k_vectors: List[np.ndarray]
    top_k_indices: List[int]
    half_gap: float
    source_index: Optional[int] = None  # index into observed_queries this candidate was proposed from
    n: Optional[int] = None             # neighbor rank (1..n_max) this candidate's sphere was built from

    def to_cache_entry(self) -> CacheEntry:
        # CIG soundness requires the stored set to be the complete set of base vectors
        # within r_c of the query
        if self.R_c is None:
            raise ValueError(
                "to_cache_entry requires R_c; use store_rc=True in propose_candidates"
            )
        indices = np.flatnonzero(self.R_c).astype(np.int32)
    
        return CacheEntry(
            query=self.query,
            top_k_vectors=[],
            r_Q=self.r_c,
            half_gap=self.half_gap,
            top_k_indices=indices,
            top_k_matrix=None,
        )

#-------------------------------------------------------------------------------
# Shared helpers
#-------------------------------------------------------------------------------

def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vecs / norms


def _cig_mask(
    q_hat: np.ndarray,
    Q_hat: np.ndarray,
    r_t: np.ndarray,
    r_c: float,
    metric: DistanceMetric,
) -> np.ndarray:
    if metric == "angular":
        dists = pairwise_angular_distance(Q_hat, q_hat)
    else:
        dists = np.linalg.norm(Q_hat - q_hat, axis=1)
    return (dists + r_t) <= r_c


def _raw_pairwise_dists(vecs_hat: np.ndarray, metric: DistanceMetric) -> np.ndarray:
    """Full (n, n) distance matrix among pre-normalized vecs_hat.
    """
    if metric == "angular":
        return pairwise_angular_distance(vecs_hat, vecs_hat)
    n = len(vecs_hat)
    dists = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        dists[i] = np.linalg.norm(vecs_hat - vecs_hat[i], axis=1)
    return dists


def _pairwise_cache_dists(
    cache_queries: np.ndarray,
    metric: DistanceMetric,
) -> tuple:
    """Pairwise distances among cache queries, sorted per row, self excluded.

    Returns (sorted_dists, sorted_indices) both shape (n_cache, n_cache - 1).
    sorted_dists[c, n-1] = distance from query c to its n-th nearest neighbor.
    sorted_indices[c, n-1] = index of that neighbor.
    """
    cache_hat = _normalize(cache_queries.copy())
    dists = _raw_pairwise_dists(cache_hat, metric).copy()
    np.fill_diagonal(dists, np.inf)
    order = np.argsort(dists, axis=1)
    sorted_dists = np.take_along_axis(dists, order, axis=1)

    # Remove the inf column (self): after sort, inf is last; take all but last
    sorted_dists = sorted_dists[:, :-1]
    sorted_indices = order[:, :-1]
    return sorted_dists, sorted_indices


def _base_dists_for_query(
    q_hat: np.ndarray,
    base_hat: np.ndarray,
    metric: DistanceMetric,
) -> np.ndarray:
    if metric == "angular":
        return pairwise_angular_distance(base_hat, q_hat)
    return np.linalg.norm(base_hat - q_hat, axis=1)


def _precompute_kth_base_dist(
    cache_hat: np.ndarray,
    base_hat: np.ndarray,
    k: int,
    metric: DistanceMetric,
    batch_size: int = 32,
) -> np.ndarray:
    """k-th nearest base distance for each cache query, computed in batches."""
    assert k >= 1, f"_precompute_kth_base_dist requires k >= 1, got k={k}"
    n_cache = len(cache_hat)
    d_kth = np.empty(n_cache, dtype=np.float32)
    for start in range(0, n_cache, batch_size):
        end = min(start + batch_size, n_cache)
        batch = cache_hat[start:end]
        if metric == "angular":
            d_batch = pairwise_angular_distance(batch, base_hat)
        else:
            d_batch = np.array([
                np.linalg.norm(base_hat - batch[j], axis=1)
                for j in range(len(batch))
            ], dtype=np.float32)
        # partition is cheaper than full sort for large base
        d_kth[start:end] = np.partition(d_batch, k - 1, axis=1)[:, k - 1]
    return d_kth

#-------------------------------------------------------------------------------

def propose_candidates(
    observed_queries: np.ndarray,
    observed_r_Q: np.ndarray,
    observed_top_k_vectors: List[List[np.ndarray]],
    observed_top_k_indices: List[List[int]],
    observed_half_gaps: np.ndarray,
    train_queries: np.ndarray,
    train_r_t: np.ndarray,
    test_queries: np.ndarray,
    test_r_t: np.ndarray,
    base_vectors: np.ndarray,
    metric: DistanceMetric,
    alpha_values: List[float] = (),
    store_rc: bool = True,
) -> List[CandidateEntry]:
    """
    Generate candidate cache entries by expanding each observed query ball.

    For each (observed_query, alpha) pair:
      - r_c = alpha * r_Q
      - R_c = base vectors within r_c
      - covered_train_indices = observed queries satisfying D(t, q) + r_t <= r_c (oracle CIG)
      - covered_test_indices  = held-out test queries satisfying the same condition

    A candidate is kept only if covered_train_indices is non-empty (source query excluded).
    """
    train_hat = _normalize(train_queries.copy())
    test_hat = _normalize(test_queries.copy())

    base_norms = np.linalg.norm(base_vectors, axis=1, keepdims=True)
    base_norms[base_norms == 0.0] = 1.0
    base_hat = base_vectors / base_norms

    candidates: List[CandidateEntry] = []

    for i in range(len(observed_queries)):
        r_Q = float(observed_r_Q[i])

        q_norm = float(np.linalg.norm(observed_queries[i]))
        q_hat = observed_queries[i] / q_norm if q_norm != 0.0 else observed_queries[i].copy()

        all_dists = _base_dists_for_query(q_hat, base_hat, metric)

        for alpha in alpha_values:
            r_c = alpha * r_Q

            R_c_mask = all_dists <= r_c
            rc_size = int(R_c_mask.sum())
            R_c = R_c_mask if store_rc else None

            train_covered = _cig_mask(q_hat, train_hat, train_r_t, r_c, metric)
            train_covered[i] = False
            if not train_covered.any():
                continue

            test_covered = _cig_mask(q_hat, test_hat, test_r_t, r_c, metric)

            candidates.append(CandidateEntry(
                query=observed_queries[i],
                r_Q=r_Q,
                alpha=alpha,
                r_c=r_c,
                R_c=R_c,
                rc_size=rc_size,
                covered_train_indices=train_covered,
                covered_test_indices=test_covered,
                top_k_vectors=observed_top_k_vectors[i],
                top_k_indices=observed_top_k_indices[i],
                half_gap=float(observed_half_gaps[i]),
            ))

    return candidates

#-------------------------------------------------------------------------------

def precompute_ball_invariants(
    observed_queries: np.ndarray,
    train_queries: np.ndarray,
    test_queries: np.ndarray,
    base_vectors: np.ndarray,
    metric: DistanceMetric,
) -> dict:
    """K-independent invariants shared across a K-sweep for the same
    observed/train/test/base arrays.
    """
    obs_hat = _normalize(observed_queries.copy())
    base_hat = _normalize(base_vectors.copy())

    raw_cache_dists = _raw_pairwise_dists(obs_hat, metric)
    dists_for_sort = raw_cache_dists.copy()
    np.fill_diagonal(dists_for_sort, np.inf)
    order = np.argsort(dists_for_sort, axis=1)
    sorted_cache_dists = np.take_along_axis(dists_for_sort, order, axis=1)[:, :-1]
    sorted_cache_indices = order[:, :-1]

    def _dist_matrix(a_hat: np.ndarray, b_hat: np.ndarray) -> np.ndarray:
        dots = np.clip(a_hat @ b_hat.T, -1.0, 1.0)
        if metric == "angular":
            d = np.arccos(dots) / np.pi
        else:
            d = np.sqrt(np.maximum(2.0 - 2.0 * dots, 0.0))
        return d.astype(np.float32)

    if train_queries is observed_queries or np.array_equal(train_queries, observed_queries):
        train_dist_only = raw_cache_dists.astype(np.float32)
    else:
        train_hat = _normalize(train_queries.copy())
        train_dist_only = _dist_matrix(obs_hat, train_hat)

    test_hat = _normalize(test_queries.copy())

    return dict(
        sorted_cache_dists=sorted_cache_dists,
        sorted_cache_indices=sorted_cache_indices,
        all_dists=_dist_matrix(obs_hat, base_hat),
        train_dist_only=train_dist_only,
        test_dist_only=_dist_matrix(obs_hat, test_hat),
    )


def _propose_nested_spheres(
    observed_queries: np.ndarray,
    observed_r_Q: np.ndarray,
    observed_top_k_vectors: List[List[np.ndarray]],
    observed_top_k_indices: List[List[int]],
    observed_half_gaps: np.ndarray,
    train_queries: np.ndarray,
    train_r_t: np.ndarray,
    test_queries: np.ndarray,
    test_r_t: np.ndarray,
    base_vectors: np.ndarray,
    metric: DistanceMetric,
    n_max: int,
    margin_lookup: np.ndarray,
    store_rc: bool,
    invariants: Optional[dict] = None,
) -> List[CandidateEntry]:
    """
    Shared nested-sphere candidate construction for propose_candidates_ball
    """
    n_cache = len(observed_queries)
    if n_max >= n_cache:
        raise ValueError(f"n_max={n_max} must be < n_cache={n_cache}")

    if invariants is not None:
        sorted_cache_dists = invariants['sorted_cache_dists']
        sorted_cache_indices = invariants['sorted_cache_indices']
        all_dists_matrix = invariants['all_dists']
        train_dist_matrix = invariants['train_dist_only']
        test_dist_matrix = invariants['test_dist_only']
    else:
        train_hat = _normalize(train_queries.copy())
        test_hat = _normalize(test_queries.copy())
        base_hat = _normalize(base_vectors.copy())
        obs_hat = _normalize(observed_queries.copy())
        sorted_cache_dists, sorted_cache_indices = _pairwise_cache_dists(observed_queries, metric)

    candidates: List[CandidateEntry] = []

    for i in range(n_cache):
        # invariants across all n_max spheres for this query computed once per query
        if invariants is not None:
            all_dists = all_dists_matrix[i]
            train_thresh = train_dist_matrix[i] + train_r_t
            test_thresh = test_dist_matrix[i] + test_r_t
        else:
            q_hat = obs_hat[i]
            all_dists = _base_dists_for_query(q_hat, base_hat, metric)
            train_thresh = _base_dists_for_query(q_hat, train_hat, metric) + train_r_t
            test_thresh = _base_dists_for_query(q_hat, test_hat, metric) + test_r_t

        for n in range(1, n_max + 1):
            nn_index = int(sorted_cache_indices[i, n - 1])
            nn_dist = float(sorted_cache_dists[i, n - 1])
            r_c = nn_dist + float(margin_lookup[nn_index])

            R_c_mask = all_dists <= r_c
            rc_size = int(R_c_mask.sum())
            R_c = R_c_mask if store_rc else None

            train_covered = train_thresh <= r_c
            # neighbor_n's own coverage uses margin_lookup[nn_index] directly
            train_covered[nn_index] = nn_dist + float(margin_lookup[nn_index]) <= r_c
            train_covered[i] = False
            if not train_covered.any():
                continue

            test_covered = test_thresh <= r_c

            candidates.append(CandidateEntry(
                query=observed_queries[i],
                r_Q=float(observed_r_Q[i]),
                alpha=None,
                r_c=r_c,
                R_c=R_c,
                rc_size=rc_size,
                covered_train_indices=train_covered,
                covered_test_indices=test_covered,
                top_k_vectors=observed_top_k_vectors[i],
                top_k_indices=observed_top_k_indices[i],
                half_gap=float(observed_half_gaps[i]),
                source_index=i,
                n=n,
            ))

    return candidates


def propose_candidates_ball(
    observed_queries: np.ndarray,
    observed_r_Q: np.ndarray,
    observed_top_k_vectors: List[List[np.ndarray]],
    observed_top_k_indices: List[List[int]],
    observed_half_gaps: np.ndarray,
    train_queries: np.ndarray,
    train_r_t: np.ndarray,
    test_queries: np.ndarray,
    test_r_t: np.ndarray,
    base_vectors: np.ndarray,
    metric: DistanceMetric,
    n_max: int = 20,
    K: int = 5,
    store_rc: bool = True,
    invariants: Optional[dict] = None,
) -> List[CandidateEntry]:
    """
    Ball-like proposal: for each observed query, propose up to n_max nested-
    sphere candidates. Sphere n's radius is r_c = D(Q, q_n) + D(q_n, E_K),
    the n-th nearest cache neighbor's distance plus the K-th nearest base
    distance from that neighbor, sized to fully contain q_n's own K-ball.
    """
    if K == 0:
        margin_lookup = np.zeros(len(observed_queries), dtype=np.float32)
    else:
        margin_lookup = observed_r_Q.astype(np.float32)
    return _propose_nested_spheres(
        observed_queries, observed_r_Q, observed_top_k_vectors, observed_top_k_indices,
        observed_half_gaps, train_queries, train_r_t, test_queries, test_r_t,
        base_vectors, metric, n_max, margin_lookup=margin_lookup, store_rc=store_rc,
        invariants=invariants,
    )
