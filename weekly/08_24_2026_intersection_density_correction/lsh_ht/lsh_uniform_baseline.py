#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Uniform-sampling baseline for the Horvitz-Thompson LSH comparison:
sample n points uniformly at random (no tables, no hyperplanes), weight
by N/n - itself a valid HT estimator, since every point has equal
inclusion probability n/N. Answers whether the LSH structure earns its
existence over the trivial alternative."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'common'))

import numpy as np

#-------------------------------------------------------------------------------


def uniform_sample_estimate(points: np.ndarray, query: np.ndarray, r: float, n: int,
                             rng: np.random.Generator) -> float:
    N = points.shape[0]
    sample_idx = rng.choice(N, size=n, replace=False)
    theta = np.arccos(np.clip(points[sample_idx] @ query, -1.0, 1.0))
    k = int((theta <= r).sum())
    return k * (N / n)


def uniform_sample_estimate_batch(points: np.ndarray, queries: np.ndarray, radii: np.ndarray,
                                   n: int, rng: np.random.Generator) -> np.ndarray:
    """One fresh uniform sample per query (matches how each E3 query
    gets independently retrieved leaves/tables - a shared sample across
    queries would understate variance)."""
    return np.array([uniform_sample_estimate(points, queries[i], float(radii[i]), n, rng)
                      for i in range(len(queries))])
