#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import numpy as np
from dataclasses import dataclass
from typing import List

from .distance_metrics import DistanceMetric
from .kv_cache import CacheEntry

#-------------------------------------------------------------------------------

@dataclass
class CandidateEntry:
    query: np.ndarray
    r_Q: float
    alpha: float
    r_c: float                        # = alpha * r_Q; used as cache radius at hit time
    R_c: object                       # set[int] when store_rc=True, else None
    rc_size: int                      # len(R_c); always set regardless of store_rc
    covered_train_indices: set        # set[int] — observed indices satisfying CIG
    covered_test_indices: set         # set[int] — held-out test indices satisfying CIG
    top_k_vectors: List[np.ndarray]
    top_k_indices: List[int]
    half_gap: float

    def to_cache_entry(self) -> CacheEntry:
        # r_c (expanded radius) replaces r_Q so hit checks use the correct expanded ball
        return CacheEntry(
            query=self.query,
            top_k_vectors=self.top_k_vectors,
            r_Q=self.r_c,
            half_gap=self.half_gap,
            top_k_indices=self.top_k_indices,
        )

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
    alpha_values: List[float],
    metric: DistanceMetric,
    store_rc: bool = True,
) -> List[CandidateEntry]:
    """
    Generate candidate cache entries by expanding each observed query ball.

    For each (observed_query, alpha) pair:
      - r_c = alpha * r_Q
      - R_c = base vectors within r_c
      - covered_train_indices = observed queries satisfying D(t, q) + r_t < r_c (oracle CIG)
      - covered_test_indices  = held-out test queries satisfying the same condition

    A candidate is kept only if covered_train_indices is non-empty (source query excluded).
    """
    def _normalize(vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return vecs / norms

    def _cig_mask(
        q_hat: np.ndarray,
        Q_hat: np.ndarray,
        r_t: np.ndarray,
        r_c: float,
    ) -> np.ndarray:
        if metric == "angular":
            dots = np.clip(Q_hat @ q_hat, -1.0, 1.0)
            dists = np.arccos(dots) / np.pi
        else:
            dists = np.linalg.norm(Q_hat - q_hat, axis=1)
        return (dists + r_t) < r_c

    train_hat = _normalize(train_queries.copy())
    test_hat = _normalize(test_queries.copy())

    # precompute normalized base matrix once — avoids re-normalizing per query
    base_norms = np.linalg.norm(base_vectors, axis=1, keepdims=True)
    base_norms[base_norms == 0.0] = 1.0
    base_hat = base_vectors / base_norms  # (M, d)

    candidates: List[CandidateEntry] = []

    for i in range(len(observed_queries)):
        r_Q = float(observed_r_Q[i])

        q_norm = float(np.linalg.norm(observed_queries[i]))
        q_hat = observed_queries[i] / q_norm if q_norm != 0.0 else observed_queries[i].copy()

        if metric == "angular":
            all_dists = np.arccos(np.clip(base_hat @ q_hat, -1.0, 1.0)) / np.pi
        else:
            all_dists = np.linalg.norm(base_hat - q_hat, axis=1)

        for alpha in alpha_values:
            r_c = alpha * r_Q

            R_c_arr = np.where(all_dists <= r_c)[0].astype(np.int64)
            rc_size = len(R_c_arr)
            R_c = set(R_c_arr.tolist()) if store_rc else None

            train_covered = set(
                np.where(_cig_mask(q_hat, train_hat, train_r_t, r_c))[0].tolist()
            )
            train_covered.discard(i)
            if not train_covered:
                continue

            test_covered = set(
                np.where(_cig_mask(q_hat, test_hat, test_r_t, r_c))[0].tolist()
            )

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
