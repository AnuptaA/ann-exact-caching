#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Edge case tests for the ESCI simulation workflow.

ESCI-specific notes vs SIFT edge cases (tests/test_edge_cases_sift.py):
  - Exercises run_esci.py's precomputed-GT pipeline (load_cache_ground_truth /
    populate_cache_from_precomputed), not raw CacheSimulator construction --
    a genuinely different code path from SIFT's edge cases, not a mirror pair.
  - The simulator is built with a dummy zero base_vectors array on purpose:
    run_esci.py never does a real base search (the full 1.82M catalog is
    never loaded), it only certifies hits via precomputed cache GT.
  - Angular metric only, no multi-metric tests.
  - Tests use synthetic 384-dim L2-normalized vectors (no real ESCI data needed).
"""

import json

import numpy as np
import pytest
from datasets.dataloaders import (
    load_esci_benchmark,
    load_cache_ground_truth_full as load_cache_ground_truth,
    populate_cache_from_precomputed_full as populate_cache_from_precomputed,
)
from simulations.core.simulate import CacheSimulator
from tests.helpers import create_mini_benchmark, get_temp_dir, cleanup_temp_dir

#-------------------------------------------------------------------------------

def test_empty_cache():
    """Test handling when cache is populated but has zero matching entries."""
    tmpdir = get_temp_dir("edge_empty_cache_")
    try:
        benchmark_dir = tmpdir / "test_benchmark"
        create_mini_benchmark('esci', benchmark_dir)

        _, cache_q, test_q, cfg = load_esci_benchmark(str(benchmark_dir))
        cache_gt = load_cache_ground_truth(benchmark_dir, cfg['cache_K'])

        # use a single test query that is very unlikely to match any cache entry
        single_test = test_q[:1]
        # build a simulator with the mini cache populated
        dummy_base = np.zeros((1, cfg['dimension']), dtype=np.float32)
        simulator = CacheSimulator(
            base_vectors=dummy_base,
            cache_queries=cache_q,
            test_queries=single_test,
            K=cfg['cache_K'],
            N=cfg['test_N'],
            metric='angular'
        )
        populate_cache_from_precomputed(simulator, cache_q, cache_gt)

        result = simulator.run_query(
            query_id=0,
            query=single_test[0],
            algorithm='lemma1_no_union'
        )

        # result may be hit or miss; important thing is no exception is raised
        assert result.error is None, f"Unexpected error: {result.error}"

    finally:
        cleanup_temp_dir(tmpdir)

#-------------------------------------------------------------------------------

def test_lemma_algorithms_run():
    """Test that both canonical no-union lemma algorithms run without error.

    lemma1/lemma2/combined (union-based) are deprecated -- see
    simulations/core/simulate.py.
    """
    tmpdir = get_temp_dir("edge_algos_")
    try:
        benchmark_dir = tmpdir / "test_benchmark"
        create_mini_benchmark('esci', benchmark_dir)

        _, cache_q, test_q, cfg = load_esci_benchmark(str(benchmark_dir))
        cache_gt = load_cache_ground_truth(benchmark_dir, cfg['cache_K'])

        dummy_base = np.zeros((1, cfg['dimension']), dtype=np.float32)

        algorithms = ['lemma1_no_union', 'lemma2_no_union']

        for algo in algorithms:
            simulator = CacheSimulator(
                base_vectors=dummy_base,
                cache_queries=cache_q,
                test_queries=test_q[:1],
                K=cfg['cache_K'],
                N=cfg['test_N'],
                metric='angular'
            )
            populate_cache_from_precomputed(simulator, cache_q, cache_gt)

            result = simulator.run_query(0, test_q[0], algo)
            assert result is not None, f"None result for {algo}"
            assert result.error is None, f"{algo} raised an error: {result.error}"

    finally:
        cleanup_temp_dir(tmpdir)

#-------------------------------------------------------------------------------

def test_invalid_algorithm():
    """Test that an invalid algorithm raises ValueError."""
    tmpdir = get_temp_dir("edge_invalid_algo_")
    try:
        benchmark_dir = tmpdir / "test_benchmark"
        create_mini_benchmark('esci', benchmark_dir)

        _, cache_q, test_q, cfg = load_esci_benchmark(str(benchmark_dir))
        cache_gt = load_cache_ground_truth(benchmark_dir, cfg['cache_K'])

        dummy_base = np.zeros((1, cfg['dimension']), dtype=np.float32)
        simulator = CacheSimulator(
            base_vectors=dummy_base,
            cache_queries=cache_q,
            test_queries=test_q,
            K=cfg['cache_K'],
            N=cfg['test_N'],
            metric='angular'
        )
        populate_cache_from_precomputed(simulator, cache_q, cache_gt)

        with pytest.raises(ValueError, match="Unknown algorithm"):
            simulator.run_query(0, test_q[0], 'invalid_algorithm')

    finally:
        cleanup_temp_dir(tmpdir)

#-------------------------------------------------------------------------------

def test_cache_gt_loading():
    """Test that cache GT loads correctly and has expected structure."""
    tmpdir = get_temp_dir("edge_gt_load_")
    try:
        benchmark_dir = tmpdir / "test_benchmark"
        create_mini_benchmark('esci', benchmark_dir)

        _, cache_q, test_q, cfg = load_esci_benchmark(str(benchmark_dir))
        cache_gt = load_cache_ground_truth(benchmark_dir, cfg['cache_K'])

        assert len(cache_gt) == cfg['num_cache_queries'], \
            f"Expected {cfg['num_cache_queries']} GT entries, got {len(cache_gt)}"

        for i in range(min(5, len(cache_gt))):
            indices, vectors, distances, gap = cache_gt[i]
            assert len(indices) == cfg['cache_K'], f"Entry {i}: wrong indices length"
            assert vectors.shape == (cfg['cache_K'], cfg['dimension']), \
                f"Entry {i}: wrong vectors shape {vectors.shape}"
            assert len(distances) == cfg['cache_K'], f"Entry {i}: wrong distances length"
            assert isinstance(gap, float), f"Entry {i}: gap should be float"

    finally:
        cleanup_temp_dir(tmpdir)

#-------------------------------------------------------------------------------

def test_cache_population():
    """Test that KVCache is populated correctly from precomputed GT."""
    tmpdir = get_temp_dir("edge_cache_pop_")
    try:
        benchmark_dir = tmpdir / "test_benchmark"
        create_mini_benchmark('esci', benchmark_dir)

        _, cache_q, test_q, cfg = load_esci_benchmark(str(benchmark_dir))
        cache_gt = load_cache_ground_truth(benchmark_dir, cfg['cache_K'])

        dummy_base = np.zeros((1, cfg['dimension']), dtype=np.float32)
        simulator = CacheSimulator(
            base_vectors=dummy_base,
            cache_queries=cache_q,
            test_queries=test_q,
            K=cfg['cache_K'],
            N=cfg['test_N'],
            metric='angular'
        )
        populate_cache_from_precomputed(simulator, cache_q, cache_gt)

        assert simulator.cache is not None, "Cache should be initialized"
        assert simulator.cache.size() == cfg['num_cache_queries'], \
            f"Expected {cfg['num_cache_queries']} cache entries, got {simulator.cache.size()}"

    finally:
        cleanup_temp_dir(tmpdir)

#-------------------------------------------------------------------------------

def test_missing_gt_file():
    """Test that missing GT file raises FileNotFoundError."""
    tmpdir = get_temp_dir("edge_missing_gt_")
    try:
        benchmark_dir = tmpdir / "test_benchmark"
        create_mini_benchmark('esci', benchmark_dir)

        try:
            load_cache_ground_truth(benchmark_dir, K=999)
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass

    finally:
        cleanup_temp_dir(tmpdir)

#-------------------------------------------------------------------------------

def test_n_larger_than_k():
    """Test behavior when N > K (requires union across entries to answer query)."""
    tmpdir = get_temp_dir("edge_n_gt_k_")
    try:
        benchmark_dir = tmpdir / "test_benchmark"
        # K=5, N=10: each cache entry covers only 5 neighbors, need union for N=10
        np.random.seed(42)
        dimension = 384
        num_cache = 10
        num_test = 5
        K = 5
        N = 10
        num_products = 100

        def rand_unit(n, d):
            v = np.random.randn(n, d).astype(np.float32)
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            return v

        product_pool = rand_unit(num_products, dimension)
        cache_q = rand_unit(num_cache, dimension)
        test_q = rand_unit(num_test, dimension)

        benchmark_dir.mkdir(parents=True, exist_ok=True)
        np.save(benchmark_dir / "base_vectors.npy", product_pool)
        np.save(benchmark_dir / "cache_queries.npy", cache_q)
        np.save(benchmark_dir / "test_queries.npy", test_q)

        save_dict = {}
        for i in range(num_cache):
            indices = np.random.choice(num_products, size=K, replace=False).astype(np.int64)
            vectors = product_pool[indices]
            distances = np.sort(np.random.uniform(0.1, 1.2, K).astype(np.float32))
            save_dict[f'indices_{i}'] = indices
            save_dict[f'vectors_{i}'] = vectors
            save_dict[f'distances_{i}'] = distances
            save_dict[f'gap_{i}'] = np.float32(distances[-1])

        np.savez_compressed(benchmark_dir / f"cache_gt_K{K}.npz", **save_dict)

        config = {
            'dataset': 'ESCI', 'benchmark_name': benchmark_dir.name,
            'num_cache_queries': num_cache, 'num_test_queries': num_test,
            'cache_K': K, 'test_N': N, 'dimension': dimension,
            'num_base_vectors': num_products, 'seed': 42, 'description': 'N>K test',
        }
        with open(benchmark_dir / "config.json", 'w') as f:
            json.dump(config, f)

        _, cache_q_l, test_q_l, cfg = load_esci_benchmark(str(benchmark_dir))
        cache_gt = load_cache_ground_truth(benchmark_dir, K)

        dummy_base = np.zeros((1, dimension), dtype=np.float32)
        simulator = CacheSimulator(
            base_vectors=dummy_base,
            cache_queries=cache_q_l,
            test_queries=test_q_l,
            K=K,
            N=N,
            metric='angular'
        )
        populate_cache_from_precomputed(simulator, cache_q_l, cache_gt)

        for algo in ['lemma1_no_union', 'lemma2_no_union']:
            result = simulator.run_query(0, test_q_l[0], algo)
            assert result is not None
            assert result.error is None, f"{algo} raised an error: {result.error}"

    finally:
        cleanup_temp_dir(tmpdir)
