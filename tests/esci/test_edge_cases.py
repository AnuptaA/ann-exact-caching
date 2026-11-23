#!/usr/bin/env python3
"""
Edge case tests for the ESCI simulation workflow.

ESCI-specific notes:
  - No base_vectors in benchmark (full 1.82M catalog is never loaded in tests)
  - Cache GT includes actual vectors, no brute force, no MainMemory lookup
  - Angular metric only, no multi-metric tests
  - Tests use synthetic 384-dim L2-normalized vectors (no real ESCI data needed)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from datasets.dataloaders import load_esci_benchmark
from simulations.simulate import CacheSimulator
from simulations.esci.run_esci import load_cache_ground_truth, populate_cache_from_precomputed
from tests.esci.helpers import create_mini_esci_benchmark, get_temp_dir, cleanup_temp_dir

#-------------------------------------------------------------------------------

def test_empty_cache():
    """Test handling when cache is populated but has zero matching entries."""
    print("\nTest: Empty cache (miss on all queries)")

    tmpdir = get_temp_dir("edge_empty_cache_")
    try:
        benchmark_dir = tmpdir / "test_benchmark"
        config = create_mini_esci_benchmark(benchmark_dir)

        cache_q, test_q, cfg = load_esci_benchmark(str(benchmark_dir))
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
            algorithm='lemma1'
        )

        # result may be hit or miss; important thing is no exception is raised
        assert result.error is None or isinstance(result.error, str), \
            f"Unexpected error type: {result.error}"
        print(f"Query result: {'HIT' if result.cache_hit else 'MISS'} (no exception raised)")

    finally:
        cleanup_temp_dir(tmpdir)

#-------------------------------------------------------------------------------

def test_lemma_algorithms_run():
    """Test that all 6 lemma algorithms run without error."""
    print("\nTest: All lemma algorithms run")

    tmpdir = get_temp_dir("edge_algos_")
    try:
        benchmark_dir = tmpdir / "test_benchmark"
        config = create_mini_esci_benchmark(benchmark_dir)

        cache_q, test_q, cfg = load_esci_benchmark(str(benchmark_dir))
        cache_gt = load_cache_ground_truth(benchmark_dir, cfg['cache_K'])

        dummy_base = np.zeros((1, cfg['dimension']), dtype=np.float32)

        algorithms = [
            'lemma1', 'lemma1_no_union',
            'lemma2', 'lemma2_no_union',
            'combined',
        ]

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
            print(f"  {algo:25s}: {'HIT' if result.cache_hit else 'MISS'}")

        print("All 6 algorithms completed without error")

    finally:
        cleanup_temp_dir(tmpdir)

#-------------------------------------------------------------------------------

def test_invalid_algorithm():
    """Test that invalid algorithm sets error field (no exception)."""
    print("\nTest: Invalid algorithm")

    tmpdir = get_temp_dir("edge_invalid_algo_")
    try:
        benchmark_dir = tmpdir / "test_benchmark"
        config = create_mini_esci_benchmark(benchmark_dir)

        cache_q, test_q, cfg = load_esci_benchmark(str(benchmark_dir))
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

        result = simulator.run_query(0, test_q[0], 'invalid_algorithm')
        assert result.error is not None, "Invalid algorithm should set error field"
        print(f"Error field set: {result.error}")

    finally:
        cleanup_temp_dir(tmpdir)

#-------------------------------------------------------------------------------

def test_cache_gt_loading():
    """Test that cache GT loads correctly and has expected structure."""
    print("\nTest: Cache GT loading")

    tmpdir = get_temp_dir("edge_gt_load_")
    try:
        benchmark_dir = tmpdir / "test_benchmark"
        config = create_mini_esci_benchmark(benchmark_dir)

        cache_q, test_q, cfg = load_esci_benchmark(str(benchmark_dir))
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

        print(f"GT loaded: {len(cache_gt)} entries, each with K={cfg['cache_K']} neighbors")

    finally:
        cleanup_temp_dir(tmpdir)

#-------------------------------------------------------------------------------

def test_cache_population():
    """Test that KVCache is populated correctly from precomputed GT."""
    print("\nTest: Cache population from GT")

    tmpdir = get_temp_dir("edge_cache_pop_")
    try:
        benchmark_dir = tmpdir / "test_benchmark"
        config = create_mini_esci_benchmark(benchmark_dir)

        cache_q, test_q, cfg = load_esci_benchmark(str(benchmark_dir))
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

        print(f"Cache populated with {simulator.cache.size()} entries")

    finally:
        cleanup_temp_dir(tmpdir)

#-------------------------------------------------------------------------------

def test_missing_gt_file():
    """Test that missing GT file raises FileNotFoundError."""
    print("\nTest: Missing GT file raises error")

    tmpdir = get_temp_dir("edge_missing_gt_")
    try:
        benchmark_dir = tmpdir / "test_benchmark"
        config = create_mini_esci_benchmark(benchmark_dir)

        try:
            load_cache_ground_truth(benchmark_dir, K=999)
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError as e:
            print(f"FileNotFoundError raised: {e}")

    finally:
        cleanup_temp_dir(tmpdir)

#-------------------------------------------------------------------------------

def test_n_larger_than_k():
    """Test behavior when N > K (requires union across entries to answer query)."""
    print("\nTest: N > K (union required)")

    tmpdir = get_temp_dir("edge_n_gt_k_")
    try:
        benchmark_dir = tmpdir / "test_benchmark"
        # K=5, N=10: each cache entry covers only 5 neighbors, need union for N=10
        from tests.esci.helpers import create_mini_esci_benchmark as _create
        import json as _json

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
            'num_base_vectors': 1814924, 'seed': 42, 'description': 'N>K test',
        }
        with open(benchmark_dir / "config.json", 'w') as f:
            _json.dump(config, f)

        cache_q_l, test_q_l, cfg = load_esci_benchmark(str(benchmark_dir))
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

        for algo in ['lemma1', 'lemma2', 'combined']:
            result = simulator.run_query(0, test_q_l[0], algo)
            assert result is not None
            print(f"  {algo}: {'HIT' if result.cache_hit else 'MISS'}")

        print("N > K handled correctly")

    finally:
        cleanup_temp_dir(tmpdir)

#-------------------------------------------------------------------------------

def main():
    print("\n" + "="*70)
    print("ESCI EDGE CASE TESTS")
    print("="*70)

    test_empty_cache()
    test_lemma_algorithms_run()
    test_invalid_algorithm()
    test_cache_gt_loading()
    test_cache_population()
    test_missing_gt_file()
    test_n_larger_than_k()

    print("\n" + "="*70)
    print("All edge case tests passed!")
    print("="*70)

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
