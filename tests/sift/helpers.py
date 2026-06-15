#!/usr/bin/env python3
"""
Helper functions for SIFT tests.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import json
import tempfile
import shutil
from utils.base import MainMemory

#-------------------------------------------------------------------------------

def create_mini_sift_benchmark(benchmark_dir: Path, seed=42):
    """
    Create minimal SIFT-like benchmark for testing.
    
    Args:
        benchmark_dir: directory to save benchmark files
        seed: random seed for reproducibility
        
    Returns:
        config dict with benchmark parameters
    """
    np.random.seed(seed)
    
    dimension = 128
    num_base = 100
    num_cache = 20
    num_test = 10
    K = 10
    N = 5
    
    base_vectors = np.random.randn(num_base, dimension).astype(np.float32)
    base_vectors /= np.linalg.norm(base_vectors, axis=1, keepdims=True)
    
    cache_queries = np.random.randn(num_cache, dimension).astype(np.float32)
    cache_queries /= np.linalg.norm(cache_queries, axis=1, keepdims=True)
    
    test_queries = np.random.randn(num_test, dimension).astype(np.float32)
    test_queries /= np.linalg.norm(test_queries, axis=1, keepdims=True)
    
    
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    np.save(benchmark_dir / "base_vectors.npy", base_vectors)
    np.save(benchmark_dir / "cache_queries.npy", cache_queries)
    np.save(benchmark_dir / "test_queries.npy", test_queries)

    # precompute angular cache ground truth (mirrors prepare_benchmark.py)
    mm = MainMemory(vectors=list(base_vectors))
    save_dict = {}
    for i, q in enumerate(cache_queries):
        _, top_k_dists, gap, indices = mm.top_k_search(
            q, k=K, metric='angular', return_indices=True
        )
        save_dict[f'indices_{i}'] = np.array(indices)
        save_dict[f'r_Q_{i}'] = np.float32(top_k_dists[-1])
        save_dict[f'half_gap_{i}'] = np.float32(gap / 2.0)
    np.savez(benchmark_dir / f'cache_gt_population_K{K}.npz', **save_dict)

    config = {
        'dataset': 'test_mini',
        'num_base_vectors': num_base,
        'num_cache_queries': num_cache,
        'num_test_queries': num_test,
        'cache_K': K,
        'test_N': N,
        'dimension': dimension,
        'seed': seed,
        'description': f'Mini test benchmark: {num_base} base, {num_cache} cache (K={K}), {num_test} test (N={N})',
    }

    with open(benchmark_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    return config

#-------------------------------------------------------------------------------

def create_mini_sift_benchmark_with_population_data(benchmark_dir: Path, seed: int = 42) -> dict:
    """
    Create a minimal SIFT-like benchmark including population data files.

    Calls create_mini_sift_benchmark() then adds:
      cache_r_Q.npy, test_gt_population_K{K}.npz, test_r_t.npy,
      cache_gt_population_K1.npz, cache_gt_population_K5.npz (for K-microbenchmark tests)
    and patches config.json with has_population_data=true.
    """
    config = create_mini_sift_benchmark(benchmark_dir, seed=seed)
    K = config['cache_K']
    num_cache = config['num_cache_queries']
    num_test = config['num_test_queries']

    base = np.load(benchmark_dir / 'base_vectors.npy')
    cache_q = np.load(benchmark_dir / 'cache_queries.npy')
    test_q = np.load(benchmark_dir / 'test_queries.npy')

    # extract cache_r_Q from existing cache GT
    cache_gt = np.load(benchmark_dir / f'cache_gt_population_K{K}.npz')
    r_Q = np.array(
        [float(cache_gt[f'r_Q_{i}']) for i in range(num_cache)],
        dtype=np.float32,
    )
    np.save(benchmark_dir / 'cache_r_Q.npy', r_Q)

    # compute test GT
    mm = MainMemory(vectors=list(base))
    test_gt_data = {}
    for i, q in enumerate(test_q):
        _, top_k_dists, gap, indices = mm.top_k_search(
            q, k=K, metric='angular', return_indices=True
        )
        test_gt_data[f'indices_{i}'] = np.array(indices)
        test_gt_data[f'r_Q_{i}'] = np.float32(top_k_dists[-1])
        test_gt_data[f'half_gap_{i}'] = np.float32(gap / 2.0)
    np.savez(benchmark_dir / f'test_gt_population_K{K}.npz', **test_gt_data)

    # extract test_r_t
    r_t = np.array(
        [float(test_gt_data[f'r_Q_{i}']) for i in range(num_test)],
        dtype=np.float32,
    )
    np.save(benchmark_dir / 'test_r_t.npy', r_t)

    # per-K GT files for K-microbenchmark tests
    for k in [1, 5]:
        k_dict = {}
        for i, q in enumerate(cache_q):
            _, top_k_dists, gap, indices = mm.top_k_search(
                q, k=k, metric='angular', return_indices=True
            )
            k_dict[f'indices_{i}'] = np.array(indices)
            k_dict[f'r_Q_{i}'] = np.float32(top_k_dists[-1])
            k_dict[f'half_gap_{i}'] = np.float32(gap / 2.0)
        np.savez(benchmark_dir / f'cache_gt_population_K{k}.npz', **k_dict)

    # patch config
    config['has_population_data'] = True
    with open(benchmark_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    return config

#-------------------------------------------------------------------------------

def get_temp_dir(prefix="sift_test_"):
    """Create and return temporary directory."""
    return Path(tempfile.mkdtemp(prefix=prefix))

#-------------------------------------------------------------------------------

def cleanup_temp_dir(temp_dir: Path):
    """Clean up temporary directory."""
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
