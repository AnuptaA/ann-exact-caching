#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
LSH-bucket radius estimator.

Estimates the distance to the K-th nearest neighbor (the "radius") of a test
query by averaging the ground-truth radii of train queries that fall into
nearby LSH buckets (within a Hamming distance threshold).

Uses plain random-hyperplane LSH (data-independent): each bit is sign(w . x)
for a fixed Gaussian w, chosen without looking at the data.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

#-------------------------------------------------------------------------------


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms


def true_radius(queries: np.ndarray, data: np.ndarray, k: int) -> np.ndarray:
    """Exact distance to the k-th nearest data vector for each query.

    Euclidean distance on raw vectors. Returned radius is the distance to the
    k-th neighbor (1-indexed: k=1 is the nearest neighbor).
    """
    n = len(queries)
    radii = np.empty(n, dtype=np.float64)
    # chunk to bound memory on large data
    chunk = max(1, min(n, 2048))
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        # (chunk, M) pairwise distances
        diff = queries[start:end, None, :] - data[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # k-th smallest (1-indexed) -> index k-1
        part = np.partition(dists, k - 1, axis=1)[:, k - 1]
        radii[start:end] = part
    return radii

#-------------------------------------------------------------------------------


@dataclass
class LSHRadiusEstimator:
    """Random-hyperplane LSH radius estimator.

    Buckets are identified by an n_bits binary code (sign of random projections).
    Per-bucket we store the sum and count of train-query true radii so that a
    count-weighted average over a Hamming ball can be formed at query time.
    """

    n_bits: int
    dim: int
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        # random hyperplanes: (dim, n_bits)
        self.planes = rng.standard_normal((self.dim, self.n_bits))
        # bit value i contributes 2**i to the integer bucket code
        self._pow2 = (1 << np.arange(self.n_bits)).astype(np.int64)
        # per-bucket accumulators, keyed by integer code
        self._sum: dict = {}
        self._count: dict = {}

    def _codes(self, vectors: np.ndarray) -> np.ndarray:
        """Integer bucket code for each vector."""
        bits = (vectors @ self.planes) > 0.0  # (n, n_bits) bool
        return bits.astype(np.int64) @ self._pow2

    def fit(self, train_queries: np.ndarray, train_radii: np.ndarray) -> None:
        """Store count-weighted radius accumulators per bucket."""
        codes = self._codes(train_queries)
        self._sum.clear()
        self._count.clear()
        for code, r in zip(codes.tolist(), train_radii.tolist()):
            self._sum[code] = self._sum.get(code, 0.0) + r
            self._count[code] = self._count.get(code, 0) + 1

    def _neighbors_within_hamming(self, code: int, max_h: int):
        """Yield all bucket codes within Hamming distance max_h of code.

        For small n_bits and small max_h this enumerates flips directly.
        """
        yield code
        if max_h <= 0:
            return
        from itertools import combinations
        bit_positions = range(self.n_bits)
        for h in range(1, max_h + 1):
            for flips in combinations(bit_positions, h):
                m = code
                for b in flips:
                    m ^= (1 << b)
                yield m

    def estimate(
        self,
        test_queries: np.ndarray,
        hamming: int,
        expand_fallback: bool = True,
        global_mean: Optional[float] = None,
    ):
        """Estimate radius for each test query.

        Count-weighted average of stored bucket radii within Hamming distance
        `hamming` of the test query's bucket. When no train queries are found
        and expand_fallback is True, the Hamming threshold is expanded until a
        populated bucket is found (recorded as hamming_used). If a populated
        bucket is never found, falls back to global_mean (or NaN if None).

        Returns a dict of parallel arrays:
          estimate, hamming_used, n_train_avg (train queries averaged over),
          n_buckets (populated buckets averaged over)
        """
        codes = self._codes(test_queries)
        n = len(test_queries)
        est = np.full(n, np.nan, dtype=np.float64)
        h_used = np.full(n, -1, dtype=np.int64)
        n_train = np.zeros(n, dtype=np.int64)
        n_buckets = np.zeros(n, dtype=np.int64)

        for i, code in enumerate(codes.tolist()):
            h = hamming
            while True:
                total_sum = 0.0
                total_count = 0
                buckets_hit = 0
                for m in self._neighbors_within_hamming(code, h):
                    c = self._count.get(m, 0)
                    if c > 0:
                        total_sum += self._sum[m]
                        total_count += c
                        buckets_hit += 1
                if total_count > 0:
                    est[i] = total_sum / total_count
                    h_used[i] = h
                    n_train[i] = total_count
                    n_buckets[i] = buckets_hit
                    break
                if not expand_fallback or h >= self.n_bits:
                    if global_mean is not None:
                        est[i] = global_mean
                        h_used[i] = self.n_bits + 1  # sentinel: global fallback
                    break
                h += 1

        return {
            "estimate": est,
            "hamming_used": h_used,
            "n_train_avg": n_train,
            "n_buckets": n_buckets,
        }
