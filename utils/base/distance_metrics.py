#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import numpy as np
from typing import Callable, Literal

#-------------------------------------------------------------------------------

DistanceMetric = Literal["euclidean", "cosine", "angular"]

#-------------------------------------------------------------------------------

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Normalize vectors and compute Euclidean distance between two vectors.
    Satisfies triangle inequality.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Euclidean distance between normalized vectors
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    a_normalized = a / norm_a if norm_a != 0 else a
    b_normalized = b / norm_b if norm_b != 0 else b

    return float(np.linalg.norm(a_normalized - b_normalized))

#-------------------------------------------------------------------------------

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine distance between two vectors. Does not satisfy triangle 
    inequality.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine distance (1 - cosine_similarity)
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 1.0
    
    cosine_sim = dot_product / (norm_a * norm_b)
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
    
    return float(1.0 - cosine_sim)

#-------------------------------------------------------------------------------

def angular_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute angular distance between two vectors. Satisfies triangle inequality.
    
    Distance = arccos(cosine_similarity) / pi
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Angular distance (normalized by pi, range [0, 1])
    """

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 1.0
    
    cosine_sim = dot_product / (norm_a * norm_b)
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
    
    # return angular distance normalized by pi
    return float(np.arccos(cosine_sim) / np.pi)

#-------------------------------------------------------------------------------

def pairwise_angular_distance(a_hat: np.ndarray, b_hat: np.ndarray) -> np.ndarray:
    """Pairwise angular distance matrix between two sets of pre-normalized
    (unit-norm) vectors: arccos(clip(a_hat @ b_hat.T, -1, 1)) / pi."""
    dots = np.clip(a_hat @ b_hat.T, -1.0, 1.0)
    return np.arccos(dots) / np.pi

#-------------------------------------------------------------------------------

def get_distance_function(metric: DistanceMetric) -> Callable[[np.ndarray, np.ndarray], float]:
    """
    Get the distance function for a given metric.
    
    Args:
        metric: Distance metric name
        
    Returns:
        Distance function
    """
    if metric == "euclidean":
        return euclidean_distance
    elif metric == "cosine":
        return cosine_distance
    elif metric == "angular":
        return angular_distance
    else:
        raise ValueError(f"Unknown distance metric: {metric}")
