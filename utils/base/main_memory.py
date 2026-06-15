#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import numpy as np
from typing import Callable, List, Optional, Tuple, Union
from .distance_metrics import DistanceMetric, get_distance_function

#-------------------------------------------------------------------------------

class MainMemory:
    """
    Represents main memory containing M vectors of dimension d.
    """

    def __init__(
        self, 
        M: int = None, 
        d: int = None, 
        seed: int = None,
        vectors: List[np.ndarray] = None
    ):
        """
        Initialize main memory.
        
        Args:
            M: Number of vectors, required if vectors not provided
            d: Dimensionality of each vector, required if vectors not provided
            seed: Random seed for reproducibility
            vectors: Pre-existing vectors to load (optional)
        """
        if vectors is not None:
            if not vectors:
                raise ValueError("Vectors cannot be empty.")
            self.vectors = vectors
            self.M = len(vectors)
            self.d = vectors[0].shape[0]
            self.seed = None
        else:
            if M is None or d is None:
                raise ValueError("M and d must be provided if vectors is None.")
            
            self.M = M
            self.d = d
            self.seed = seed
            
            if seed is not None:
                np.random.seed(seed)

            self.vectors = self._generate_random_vectors()

    def _generate_random_vectors(self) -> List[np.ndarray]:
        """Generate M random vectors of dimensionality d."""
        vectors = []
        for _ in range(self.M):
            vec = np.random.randn(self.d)
            vectors.append(vec)
        return vectors
        
    def top_k_search(
        self, 
        query: np.ndarray, 
        k: int, 
        metric: DistanceMetric = "euclidean",
        distance_tracker: Optional[Callable[..., float]] = None,
        return_indices: bool = False
    ) -> Union[
        Tuple[List[np.ndarray], List[float], float],
        Tuple[List[np.ndarray], List[float], float, List[int]]
    ]:
        """
        Find the top-k closest vectors to the query.
        
        Args:
            query: Query vector
            k: Number of closest vectors to return
            metric: Distance metric
            distance_tracker: Optional function to track distance calculations
            return_indices: Whether to include vector indices in the return value
            
        Returns:
            Tuple of (top_k_vectors, top_k_distances, gap), or the same tuple
            plus top_k_indices when return_indices is True.
        """
        distance_func = get_distance_function(metric)
        if distance_tracker is not None:
            distance_func = distance_tracker
        
        distances = []
        for i, vec in enumerate(self.vectors):
            dist = distance_func(query, vec)
            distances.append((dist, i))
        
        # sort by distance and get top k
        distances.sort(key=lambda x: x[0])
        top_k = distances[:k]
        
        top_k_vectors = [self.vectors[i] for _, i in top_k]
        top_k_distances = [d for d, _ in top_k]
        top_k_indices = [i for _, i in top_k]
        
        # compute gap
        if len(distances) > k:
            gap = distances[k][0] - distances[k-1][0]
        else:
            gap = 0.0
        
        if return_indices:
            return top_k_vectors, top_k_distances, gap, top_k_indices

        return top_k_vectors, top_k_distances, gap
    
    def get_all_vectors(self) -> List[np.ndarray]:
        """
        Get all vectors in main memory.
        
        Returns:
            List of all vectors
        """
        return self.vectors
    
    def __repr__(self) -> str:
        return f"MainMemory(M={self.M}, d={self.d})"

#-------------------------------------------------------------------------------

def range_search(
    query: np.ndarray,
    radius: float,
    base_vectors: np.ndarray,
    metric: DistanceMetric,
) -> np.ndarray:
    """
    Return indices of base_vectors within radius of query.

    Vectorized: computes all distances in a single matrix operation.
    Supports 'angular' and 'euclidean' metrics (both on normalized vectors).
    """
    q_norm = float(np.linalg.norm(query))
    if q_norm == 0.0:
        return np.empty(0, dtype=np.int64)
    q_hat = query / q_norm

    row_norms = np.linalg.norm(base_vectors, axis=1, keepdims=True)
    row_norms[row_norms == 0.0] = 1.0
    B_hat = base_vectors / row_norms  # (M, d)

    if metric == "angular":
        dots = B_hat @ q_hat  # (M,)
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots) / np.pi
    elif metric == "euclidean":
        diff = B_hat - q_hat  # (M, d)
        dists = np.linalg.norm(diff, axis=1)
    else:
        raise ValueError(f"range_search: unsupported metric '{metric}'")

    return np.where(dists <= radius)[0].astype(np.int64)
