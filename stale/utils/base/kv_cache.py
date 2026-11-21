#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import numpy as np
from typing import List, Optional
from .distance_metrics import DistanceMetric, pairwise_angular_distance
from .main_memory import MainMemory

#-------------------------------------------------------------------------------

class CacheEntry:
    """
    Represents a single cache entry.
    """

    def __init__(
        self,
        query: np.ndarray,
        top_k_vectors: List[np.ndarray],
        r_Q: float,
        half_gap: float,
        top_k_indices: Optional[List[int]] = None,
        top_k_matrix: Optional[np.ndarray] = None,
    ):
        """
        Initialize a cache entry.

        Args:
            query: Query vector (key)
            top_k_vectors: Top-K closest vectors
            r_Q: Distance to the K-th (farthest cached) vector
            half_gap: Half of the difference between K-th and (K+1)-th vector
            top_k_indices: Stable vector indices corresponding to top_k_vectors
            top_k_matrix: Pre-stacked, pre-normalized (|R_c|, dim) matrix for
                fast angular distance computation via a single matmul. Built by
                to_cache_entry(); None when created via add_entry directly.
        """
        self.query = query
        self.top_k_vectors = top_k_vectors
        self.r_Q = r_Q
        self.half_gap = half_gap
        self.top_k_indices = top_k_indices
        self.top_k_matrix = top_k_matrix

    def get_radius(self) -> float:
        """
        Get the distance to the K-th (farthest cached) vector.

        Returns:
            Caching radius (r_Q)
        """
        return self.r_Q
    
    def get_half_gap(self) -> float:
        """
        Get half of the difference between K-th and (K+1)-th vector.
        
        Returns:
            Half gap value (half_gap)
        """
        return self.half_gap

#-------------------------------------------------------------------------------

class KVCache:
    """
    Key-Value cache for storing query results.
    Keys are query embeddings, values are (top_k_vectors, radius, half_gap).
    """
    
    def __init__(self, metric: DistanceMetric = "euclidean"):
        """
        Initialize the KV cache.

        Args:
            metric: Distance metric to use
        """
        self.cache: List[CacheEntry] = []
        self.metric = metric
        # O4: stacked normalized query matrix and radii, built lazily on first use
        self._query_matrix: Optional[np.ndarray] = None  # (n_entries, dim), float32
        self._entry_radii: Optional[np.ndarray] = None   # (n_entries,), float32

    def _invalidate_query_matrix(self) -> None:
        self._query_matrix = None
        self._entry_radii = None

    def build_query_matrix(self) -> None:
        """Build (or rebuild) the stacked normalized query matrix and radii array."""
        if not self.cache:
            dim = self.cache[0].query.shape[0] if self.cache else 1
            self._query_matrix = np.empty((0, dim), dtype=np.float32)
            self._entry_radii = np.empty((0,), dtype=np.float32)
            return
        queries = np.array([e.query for e in self.cache], dtype=np.float32)
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self._query_matrix = queries / norms
        self._entry_radii = np.array([e.r_Q for e in self.cache], dtype=np.float32)

    def get_d_Q_batch(self, query: np.ndarray) -> np.ndarray:
        """Compute D(query, Q_i) for all cache entries in one matmul."""
        if self._query_matrix is None:
            self.build_query_matrix()
        if len(self.cache) == 0:
            return np.empty((0,), dtype=np.float32)
        q_norm = float(np.linalg.norm(query))
        q_hat = (query / q_norm).astype(np.float32) if q_norm != 0.0 else query.astype(np.float32)
        if self.metric == "angular":
            return pairwise_angular_distance(self._query_matrix, q_hat)
        return np.linalg.norm(self._query_matrix - q_hat, axis=1)

    def get_entry_radii(self) -> np.ndarray:
        """(n_entries,) array of each cache entry's r_Q, for vectorizing the
        O1 gate (d_Q <= r_Q) across all entries at once instead of a
        per-entry Python comparison. Same lazy-build as get_d_Q_batch."""
        if self._entry_radii is None:
            self.build_query_matrix()
        return self._entry_radii

    def add_entry(
        self,
        query: np.ndarray,
        top_k_vectors: List[np.ndarray],
        r_Q: float,
        half_gap: float,
        top_k_indices: Optional[List[int]] = None,
        top_k_matrix: Optional[np.ndarray] = None,
    ):
        """
        Add a new entry to the cache.

        Args:
            query: Query vector
            top_k_vectors: Top-K closest vectors
            r_Q: Distance to the K-th (farthest cached) vector
            half_gap: Half of the difference between K-th and (K+1)-th vector
            top_k_indices: Stable vector indices corresponding to top_k_vectors
            top_k_matrix: Optional pre-normalized matrix for fast distance (O3)
        """
        entry = CacheEntry(query, top_k_vectors, r_Q, half_gap, top_k_indices, top_k_matrix)
        self.cache.append(entry)
        self._invalidate_query_matrix()

    def populate_from_queries(
        self,
        queries: np.ndarray,
        main_memory: MainMemory,
        K: int
    ):
        """
        Populate cache by running brute-force top-K search for each query.
        
        Args:
            queries: Array of query vectors to cache
            main_memory: MainMemory instance to search
            K: Number of vectors to cache per query
        """
        for query in queries:
            top_k_vecs, top_k_dists, gap, top_k_indices = main_memory.top_k_search(
                query,
                K,
                self.metric,
                return_indices=True
            )
            self.add_entry(query, top_k_vecs, float(top_k_dists[-1]), gap / 2.0, top_k_indices)
    
    def get_all_entries(self) -> List[CacheEntry]:
        """
        Get all cache entries.
        
        Returns:
            List of cache entries
        """
        return self.cache
    
    def clear(self) -> None:
        """
        Clear all cache entries.
        """
        self.cache.clear()
    
    def size(self) -> int:
        """
        Get the number of entries in the cache.
        
        Returns:
            Number of cache entries
        """
        return len(self.cache)
    
    def __repr__(self) -> str:
        return f"KVCache(metric={self.metric}, size={self.size()})"
