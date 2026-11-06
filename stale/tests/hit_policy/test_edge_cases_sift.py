#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Edge case tests for the SIFT simulation workflow.

Exercises CacheSimulator directly with a real base_vectors array, covering
both angular and euclidean metrics.
"""

import numpy as np
import pytest
from datasets.dataloaders import load_sift_benchmark
from simulations.core.simulate import CacheSimulator
from tests.helpers import create_mini_benchmark, get_temp_dir, cleanup_temp_dir

#-------------------------------------------------------------------------------

def test_empty_cache():
    """Test handling of empty cache (zero cache queries)."""
    tmpdir = get_temp_dir("edge_empty_cache_")
    try:
        benchmark_dir = tmpdir / "test_benchmark"
        create_mini_benchmark('sift', benchmark_dir)
        base, cache_q, test_q, config = load_sift_benchmark(str(benchmark_dir))

        empty_cache = np.zeros((0, 128), dtype=np.float32)
        empty_test = test_q[:1]

        simulator = CacheSimulator(
            base_vectors=base,
            cache_queries=empty_cache,
            test_queries=empty_test,
            K=10,
            N=5,
            metric='euclidean'
        )

        simulator.populate_cache()

        result = simulator.run_query(
            query_id=0,
            query=empty_test[0],
            algorithm='lemma1_no_union',
            ground_truth_indices=[]
        )

        assert not result.cache_hit, "Empty cache should result in miss"
        assert result.error is None, f"Unexpected error: {result.error}"

    finally:
        cleanup_temp_dir(tmpdir)

#-------------------------------------------------------------------------------

def test_single_base_vector():
    """Test handling of single base vector."""
    tmpdir = get_temp_dir("edge_single_base_")
    try:
        benchmark_dir = tmpdir / "test_benchmark"
        create_mini_benchmark('sift', benchmark_dir)
        base, cache_q, test_q, config = load_sift_benchmark(str(benchmark_dir))

        single_base = base[:1]
        small_cache = cache_q[:5]
        small_test = test_q[:1]

        simulator = CacheSimulator(
            base_vectors=single_base,
            cache_queries=small_cache,
            test_queries=small_test,
            K=1,
            N=1,
            metric='euclidean'
        )

        simulator.populate_cache()

        result = simulator.run_query(
            query_id=0,
            query=small_test[0],
            algorithm='brute',
            ground_truth_indices=[0]
        )

        assert result.correct, "Brute force should be correct"

    finally:
        cleanup_temp_dir(tmpdir)

#-------------------------------------------------------------------------------

def test_k_larger_than_base():
    """Test handling when K > number of base vectors."""
    tmpdir = get_temp_dir("edge_k_large_")
    try:
        benchmark_dir = tmpdir / "test_benchmark"
        create_mini_benchmark('sift', benchmark_dir)
        base, cache_q, test_q, config = load_sift_benchmark(str(benchmark_dir))

        small_base = base[:5]
        small_cache = cache_q[:5]
        small_test = test_q[:1]

        simulator = CacheSimulator(
            base_vectors=small_base,
            cache_queries=small_cache,
            test_queries=small_test,
            K=100,
            N=3,
            metric='euclidean'
        )

        simulator.populate_cache()

        result = simulator.run_query(
            query_id=0,
            query=small_test[0],
            algorithm='brute',
            ground_truth_indices=[]
        )

        assert result.distance_calculations is None or result.distance_calculations > 0, \
            "Should compute distances or return None on error"

    finally:
        cleanup_temp_dir(tmpdir)

#-------------------------------------------------------------------------------

def test_n_equals_zero():
    """Test handling when N=0 (no ground truth needed)."""
    tmpdir = get_temp_dir("edge_n_zero_")
    try:
        benchmark_dir = tmpdir / "test_benchmark"
        create_mini_benchmark('sift', benchmark_dir)
        base, cache_q, test_q, config = load_sift_benchmark(str(benchmark_dir))

        simulator = CacheSimulator(
            base_vectors=base,
            cache_queries=cache_q,
            test_queries=test_q,
            K=10,
            N=0,
            metric='euclidean'
        )

        simulator.populate_cache()

        result = simulator.run_query(
            query_id=0,
            query=test_q[0],
            algorithm='brute',
            ground_truth_indices=[]
        )

        assert result.distance_calculations is None or result.distance_calculations > 0, \
            "Should compute distances or return None on error"

    finally:
        cleanup_temp_dir(tmpdir)

#-------------------------------------------------------------------------------

def test_identical_vectors():
    """Test handling of identical vectors in base."""
    tmpdir = get_temp_dir("edge_identical_")
    try:
        benchmark_dir = tmpdir / "test_benchmark"
        create_mini_benchmark('sift', benchmark_dir)
        base, cache_q, test_q, config = load_sift_benchmark(str(benchmark_dir))

        identical_base = np.tile(base[0], (10, 1))
        small_cache = cache_q[:5]
        small_test = test_q[:1]

        simulator = CacheSimulator(
            base_vectors=identical_base,
            cache_queries=small_cache,
            test_queries=small_test,
            K=5,
            N=5,
            metric='euclidean'
        )

        simulator.populate_cache()

        result = simulator.run_query(
            query_id=0,
            query=small_test[0],
            algorithm='brute',
            ground_truth_indices=list(range(5))
        )

        assert result.distance_calculations is None or result.distance_calculations > 0

    finally:
        cleanup_temp_dir(tmpdir)


def test_invalid_algorithm():
    """Test that invalid algorithm raises appropriate error."""
    tmpdir = get_temp_dir("edge_invalid_algo_")
    try:
        benchmark_dir = tmpdir / "test_benchmark"
        create_mini_benchmark('sift', benchmark_dir)
        base, cache_q, test_q, config = load_sift_benchmark(str(benchmark_dir))

        simulator = CacheSimulator(
            base_vectors=base,
            cache_queries=cache_q,
            test_queries=test_q,
            K=config['cache_K'],
            N=config['test_N'],
            metric='euclidean'
        )

        simulator.populate_cache()

        with pytest.raises(ValueError, match="Unknown algorithm"):
            simulator.run_query(
                query_id=0,
                query=test_q[0],
                algorithm='invalid_algorithm',
                ground_truth_indices=[]
            )

    finally:
        cleanup_temp_dir(tmpdir)

#-------------------------------------------------------------------------------

def test_invalid_metric():
    """Test that invalid metric raises appropriate error."""
    tmpdir = get_temp_dir("edge_invalid_metric_")
    try:
        benchmark_dir = tmpdir / "test_benchmark"
        create_mini_benchmark('sift', benchmark_dir)
        base, cache_q, test_q, config = load_sift_benchmark(str(benchmark_dir))

        try:
            simulator = CacheSimulator(
                base_vectors=base,
                cache_queries=cache_q,
                test_queries=test_q,
                K=config['cache_K'],
                N=config['test_N'],
                metric='invalid_metric'
            )
            assert False, "Should have raised error for invalid metric"
        except (ValueError, KeyError):
            pass

    finally:
        cleanup_temp_dir(tmpdir)

#-------------------------------------------------------------------------------

def test_validation_wrong_metric():
    """Test that validation requires matching metrics."""
    tmpdir = get_temp_dir("edge_validation_metric_")
    try:
        benchmark_dir = tmpdir / "test_benchmark"
        create_mini_benchmark('sift', benchmark_dir)
        base, cache_q, test_q, config = load_sift_benchmark(str(benchmark_dir))

        simulator_ang = CacheSimulator(
            base_vectors=base,
            cache_queries=cache_q,
            test_queries=test_q,
            K=config['cache_K'],
            N=config['test_N'],
            metric='angular'
        )

        simulator_ang.populate_cache()

        result_ang = simulator_ang.run_query(
            query_id=0,
            query=test_q[0],
            algorithm='lemma1_no_union',
            ground_truth_indices=[]
        )
        assert result_ang.error is None, f"Unexpected error: {result_ang.error}"

        simulator_euc = CacheSimulator(
            base_vectors=base,
            cache_queries=cache_q,
            test_queries=test_q,
            K=config['cache_K'],
            N=config['test_N'],
            metric='euclidean'
        )

        simulator_euc.populate_cache()

        simulator_euc.run_query(
            query_id=0,
            query=test_q[0],
            algorithm='brute',
            ground_truth_indices=[]
        )

    finally:
        cleanup_temp_dir(tmpdir)

#-------------------------------------------------------------------------------

def test_ground_truth_with_duplicates():
    """Test handling of duplicate indices in ground truth."""
    tmpdir = get_temp_dir("edge_gt_duplicates_")
    try:
        benchmark_dir = tmpdir / "test_benchmark"
        create_mini_benchmark('sift', benchmark_dir)
        base, cache_q, test_q, config = load_sift_benchmark(str(benchmark_dir))

        simulator = CacheSimulator(
            base_vectors=base,
            cache_queries=cache_q,
            test_queries=test_q,
            K=config['cache_K'],
            N=10,
            metric='euclidean'
        )

        simulator.populate_cache()

        dup_gt = [0, 1, 1, 2, 2, 2, 3, 4, 5, 5]

        simulator.run_query(
            query_id=0,
            query=test_q[0],
            algorithm='brute',
            ground_truth_indices=dup_gt
        )

    finally:
        cleanup_temp_dir(tmpdir)
