#!/usr/bin/env python3
"""
Helper functions for ESCI population pipeline tests.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import json
import numpy as np
import tempfile
import shutil
from utils.base import MainMemory

#-------------------------------------------------------------------------------

POPULATION_K = 10   # mini K for testing

#-------------------------------------------------------------------------------

def _brute_force_knn_angular(queries, base, K):
    """Brute-force top-K angular KNN. Returns list of (indices, distances, gap)."""
    q = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    b = base / np.linalg.norm(base, axis=1, keepdims=True)
    kp1 = min(K + 1, len(base))
    results = []
    for i in range(len(queries)):
        scores = q[i] @ b.T
        top_idx = np.argsort(-scores)[:kp1]
        top_scores = scores[top_idx]
        dists = (np.arccos(np.clip(top_scores, -1.0, 1.0)) / np.pi).astype(np.float32)
        idx = top_idx[:K]
        d = dists[:K]
        gap = float(dists[K] - dists[K - 1]) if kp1 > K else 0.0
        results.append((idx, d, gap))
    return results

#-------------------------------------------------------------------------------

def create_mini_esci_benchmark(benchmark_dir: Path, seed: int = 42) -> dict:
    """
    Create a minimal ESCI-style benchmark for population testing.

    Files created (ESCI population format):
      base_vectors.npy              100 x 384, normalized
      cache_queries.npy             20 x 384, normalized
      test_queries.npy              10 x 384, normalized
      cache_gt_population_K10.npz  top-10 of cache_queries on base (indices + gap)
      config.json
    """
    np.random.seed(seed)

    dimension = 384
    num_base = 100
    num_cache = 20
    num_test = 10
    K = POPULATION_K

    base_vectors = np.random.randn(num_base, dimension).astype(np.float32)
    base_vectors /= np.linalg.norm(base_vectors, axis=1, keepdims=True)

    cache_queries = np.random.randn(num_cache, dimension).astype(np.float32)
    cache_queries /= np.linalg.norm(cache_queries, axis=1, keepdims=True)

    test_queries = np.random.randn(num_test, dimension).astype(np.float32)
    test_queries /= np.linalg.norm(test_queries, axis=1, keepdims=True)

    benchmark_dir.mkdir(parents=True, exist_ok=True)
    np.save(benchmark_dir / 'base_vectors.npy', base_vectors)
    np.save(benchmark_dir / 'cache_queries.npy', cache_queries)
    np.save(benchmark_dir / 'test_queries.npy', test_queries)

    # cache GT — unified format: indices + r_Q_i + half_gap_i
    results = _brute_force_knn_angular(cache_queries, base_vectors, K)
    save_dict = {}
    for i, (idx, d, gap) in enumerate(results):
        save_dict[f'indices_{i}'] = idx
        save_dict[f'r_Q_{i}'] = np.float32(d[-1])
        save_dict[f'half_gap_{i}'] = np.float32(gap / 2.0)
    np.savez(benchmark_dir / f'cache_gt_population_K{K}.npz', **save_dict)

    config = {
        'dataset': 'ESCI_test_mini',
        'benchmark_name': benchmark_dir.name,
        'num_base_vectors': num_base,
        'num_cache_queries': num_cache,
        'num_test_queries': num_test,
        'cache_K': K,
        'test_N': 5,
        'dimension': dimension,
        'seed': seed,
        'description': f'Mini ESCI population test: {num_base} base, {num_cache} cache (K={K}), {num_test} test',
    }
    with open(benchmark_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    return config

#-------------------------------------------------------------------------------

def create_mini_esci_benchmark_with_population_data(benchmark_dir: Path, seed: int = 42) -> dict:
    """
    Create a minimal ESCI benchmark with population data files.

    Calls create_mini_esci_benchmark then adds:
      cache_r_Q.npy, test_gt_population_K{K}.npz, test_r_t.npy,
      cache_gt_population_K1.npz, cache_gt_population_K5.npz (for K-microbenchmark tests)
    and patches config.json with has_population_data=true.
    """
    config = create_mini_esci_benchmark(benchmark_dir, seed=seed)
    K = config['cache_K']
    num_cache = config['num_cache_queries']
    num_test = config['num_test_queries']

    base = np.load(benchmark_dir / 'base_vectors.npy')
    cache_q = np.load(benchmark_dir / 'cache_queries.npy')
    test_q = np.load(benchmark_dir / 'test_queries.npy')

    # cache_r_Q from existing population cache GT
    cache_gt = np.load(benchmark_dir / f'cache_gt_population_K{K}.npz')
    r_Q = np.array(
        [float(cache_gt[f'r_Q_{i}']) for i in range(num_cache)],
        dtype=np.float32,
    )
    np.save(benchmark_dir / 'cache_r_Q.npy', r_Q)

    # test GT
    test_results = _brute_force_knn_angular(test_q, base, K)
    test_gt_data = {}
    for i, (idx, d, gap) in enumerate(test_results):
        test_gt_data[f'indices_{i}'] = idx
        test_gt_data[f'r_Q_{i}'] = np.float32(d[-1])
        test_gt_data[f'half_gap_{i}'] = np.float32(gap / 2.0)
    np.savez(benchmark_dir / f'test_gt_population_K{K}.npz', **test_gt_data)

    # test_r_t
    r_t = np.array(
        [float(test_gt_data[f'r_Q_{i}']) for i in range(num_test)],
        dtype=np.float32,
    )
    np.save(benchmark_dir / 'test_r_t.npy', r_t)

    # per-K GT files for K-microbenchmark tests
    for k in [1, 5]:
        k_results = _brute_force_knn_angular(cache_q, base, k)
        k_dict = {}
        for i, (idx, d, gap) in enumerate(k_results):
            k_dict[f'indices_{i}'] = idx
            k_dict[f'r_Q_{i}'] = np.float32(d[-1])
            k_dict[f'half_gap_{i}'] = np.float32(gap / 2.0)
        np.savez(benchmark_dir / f'cache_gt_population_K{k}.npz', **k_dict)

    config['has_population_data'] = True
    with open(benchmark_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    return config

#-------------------------------------------------------------------------------

def get_temp_dir(prefix: str = "esci_test_") -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))

def cleanup_temp_dir(temp_dir: Path) -> None:
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
