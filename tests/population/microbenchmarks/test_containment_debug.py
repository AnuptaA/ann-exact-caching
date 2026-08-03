#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Distance invariant checks for the containment microbenchmark's debug data,
replacing the sanity_check() report-generation code that used to live in
containment_debug.py."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import pytest

from datasets.dataloaders import load_sift_benchmark
from simulations.population.microbenchmarks.microbenchmark_containment import (
    _normalize, DATASET_CONFIG,
)

#-------------------------------------------------------------------------------

BENCHMARK_DIR = Path('datasets/sift/sift_b50k_c256k100_t512n20_s42')


def _load_cache_and_test_dists():
    if not (BENCHMARK_DIR / 'cache_queries.npy').exists():
        pytest.skip(f'{BENCHMARK_DIR.name} not prepared')
    base, cache_q, test_q, _ = load_sift_benchmark(str(BENCHMARK_DIR))
    cache_norm = _normalize(cache_q)
    test_norm = _normalize(test_q)
    test_dists = np.arccos(np.clip(test_norm @ cache_norm.T, -1.0, 1.0)) / np.pi
    return cache_norm, test_dists


def test_cache_queries_are_unit_normalized():
    cache_norm, _ = _load_cache_and_test_dists()
    self_sims = np.diag(cache_norm @ cache_norm.T)
    assert np.abs(self_sims - 1.0).max() < 1e-5


def test_test_to_cache_distances_are_non_negative():
    _, test_dists = _load_cache_and_test_dists()
    assert (test_dists >= 0).all()


def test_min_distance_flags_known_near_duplicate():
    """Construct a test query that's a near-exact duplicate of a cache query
    and confirm the minimum test-to-cache distance correctly detects it."""
    rng = np.random.RandomState(0)
    dim = 16
    cache_q = rng.randn(8, dim).astype(np.float32)
    test_q = rng.randn(5, dim).astype(np.float32)
    test_q[0] = cache_q[3] + rng.randn(dim).astype(np.float32) * 1e-6

    cache_norm = _normalize(cache_q)
    test_norm = _normalize(test_q)
    test_dists = np.arccos(np.clip(test_norm @ cache_norm.T, -1.0, 1.0)) / np.pi

    assert test_dists.min() < 1e-4
