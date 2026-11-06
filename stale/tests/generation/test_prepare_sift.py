#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Unit tests for datasets/sift/prepare_benchmark.py's own logic: benchmark
naming, partial-GT resume, and the optional full-base cache GT builder.
Uses small dummy arrays throughout; no real SIFT data needed."""

import numpy as np

from datasets.sift.prepare_benchmark import (
    create_benchmark_name,
    _check_partial_gt,
    build_cache_gt_full,
    _compute_single_query_gt,
    _compute_single_query_gt_multi_k,
)
from tests.helpers import get_temp_dir, cleanup_temp_dir

#-------------------------------------------------------------------------------


def test_create_benchmark_name_format():
    name = create_benchmark_name(50_000, 1024, 512, 100, 20, seed=42)
    assert name == "sift_b50k_c1024k100_t512n20_s42"


def test_create_benchmark_name_million_base():
    name = create_benchmark_name(1_000_000, 9000, 1000, 100, 10, seed=42)
    assert name == "sift_b1m_c9000k100_t1000n10_s42"


def test_create_benchmark_name_small_base():
    """num_base < 1000 is not abbreviated with a 'k'/'m' suffix."""
    name = create_benchmark_name(500, 100, 50, 20, 10, seed=1)
    assert name == "sift_b500_c100k20_t50n10_s1"

#-------------------------------------------------------------------------------


def test_check_partial_gt_missing_file():
    tmp = get_temp_dir()
    try:
        gt, completed = _check_partial_gt(tmp / "does_not_exist.npz", expected_count=5)
        assert gt == {}
        assert completed == set()
    finally:
        cleanup_temp_dir(tmp)


def test_check_partial_gt_partial_completion():
    """Only entries with both indices_i and r_Q_i present count as completed."""
    tmp = get_temp_dir()
    try:
        gt_file = tmp / "partial.npz"
        save_dict = {}
        for i in range(3):
            save_dict[f'indices_{i}'] = np.array([i, i + 1, i + 2])
            save_dict[f'r_Q_{i}'] = np.float32(0.1 * i)
            save_dict[f'half_gap_{i}'] = np.float32(0.01 * i)
        np.savez_compressed(gt_file, **save_dict)

        gt, completed = _check_partial_gt(gt_file, expected_count=5)
        assert completed == {0, 1, 2}
        assert set(gt.keys()) == {0, 1, 2}
        indices, r_Q, half_gap = gt[1]
        assert list(indices) == [1, 2, 3]
        assert abs(r_Q - 0.1) < 1e-6
    finally:
        cleanup_temp_dir(tmp)


def test_check_partial_gt_fully_complete():
    tmp = get_temp_dir()
    try:
        gt_file = tmp / "full.npz"
        save_dict = {}
        for i in range(4):
            save_dict[f'indices_{i}'] = np.array([i])
            save_dict[f'r_Q_{i}'] = np.float32(1.0)
            save_dict[f'half_gap_{i}'] = np.float32(0.0)
        np.savez_compressed(gt_file, **save_dict)

        gt, completed = _check_partial_gt(gt_file, expected_count=4)
        assert completed == {0, 1, 2, 3}
    finally:
        cleanup_temp_dir(tmp)

#-------------------------------------------------------------------------------


def _brute_force_euclidean_topk(query, base, k):
    dists = np.linalg.norm(base - query, axis=1)
    order = np.argsort(dists)[:k + 1]
    return order, dists[order]


def test_build_cache_gt_full_matches_brute_force():
    """Verify indices/vectors/distances/gap match a direct brute-force Euclidean
    recomputation from small dummy base/query/groundtruth arrays."""
    rng = np.random.RandomState(0)
    dim = 8
    n_base = 50
    n_queries = 5
    cache_K = 10

    base = rng.randn(n_base, dim).astype(np.float32)
    queries = rng.randn(n_queries, dim).astype(np.float32)

    # precomputed top-100-style groundtruth: here top-n_base (sorted by true distance)
    groundtruth = np.zeros((n_queries, n_base), dtype=np.int64)
    for qi in range(n_queries):
        dists = np.linalg.norm(base - queries[qi], axis=1)
        groundtruth[qi] = np.argsort(dists)

    tmp = get_temp_dir()
    try:
        cache_query_indices = np.array([0, 2, 4])
        build_cache_gt_full(
            cache_query_indices=cache_query_indices,
            groundtruth=groundtruth,
            full_base_vectors=base,
            query_vectors=queries,
            cache_K=cache_K,
            output_dir=tmp,
        )

        gt_file = tmp / f"cache_gt_K{cache_K}.npz"
        assert gt_file.exists()
        data = np.load(gt_file)

        for local_i, global_i in enumerate(cache_query_indices):
            expected_order, expected_dists = _brute_force_euclidean_topk(
                queries[global_i], base, cache_K
            )
            ids = data[f'indices_{local_i}']
            vectors = data[f'vectors_{local_i}']
            distances = data[f'distances_{local_i}']
            gap = float(data[f'gap_{local_i}'])

            assert list(ids) == list(expected_order[:cache_K])
            assert np.allclose(distances, expected_dists[:cache_K], atol=1e-4)
            assert np.allclose(vectors, base[ids])
            expected_gap = float(expected_dists[cache_K] - expected_dists[cache_K - 1])
            assert abs(gap - expected_gap) < 1e-4
    finally:
        cleanup_temp_dir(tmp)


def test_build_cache_gt_full_rejects_k_ge_100():
    tmp = get_temp_dir()
    try:
        dummy = np.zeros((1, 1), dtype=np.float32)
        try:
            build_cache_gt_full(
                cache_query_indices=np.array([0]),
                groundtruth=np.zeros((1, 100), dtype=np.int64),
                full_base_vectors=dummy,
                query_vectors=dummy,
                cache_K=100,
                output_dir=tmp,
            )
            assert False, "expected AssertionError for cache_K=100"
        except AssertionError as e:
            assert "cache_K must be < 100" in str(e)
    finally:
        cleanup_temp_dir(tmp)


def test_multi_k_gt_matches_per_k_top_k_search():
    """_compute_single_query_gt_multi_k (one shared brute-force pass, sliced
    per K) must be bit-identical to calling _compute_single_query_gt (which
    wraps MainMemory.top_k_search) once per K. The multi-K dedup relies on
    slicing from a single max-K pass being exact, not an approximation.
    Checked across several queries and K values."""
    rng = np.random.RandomState(11)
    dim = 6
    n_base = 40
    n_queries = 4
    K_list = [1, 5, 20]

    base = rng.randn(n_base, dim).astype(np.float32)
    queries = rng.randn(n_queries, dim).astype(np.float32)

    for qi in range(n_queries):
        _, multi_k_results = _compute_single_query_gt_multi_k(
            (qi, queries[qi], base, K_list, "angular")
        )
        for k in K_list:
            _, expected_idx, expected_dists, expected_gap = _compute_single_query_gt(
                (qi, queries[qi], base, k, "angular")
            )
            idx_k, r_Q_k, half_gap_k = multi_k_results[k]

            assert np.array_equal(idx_k, expected_idx), f"qi={qi} k={k}: indices differ"
            assert r_Q_k == float(expected_dists[-1]), f"qi={qi} k={k}: r_Q differs"
            assert half_gap_k == float(expected_gap) / 2.0, f"qi={qi} k={k}: half_gap differs"


def test_build_cache_gt_full_skips_if_exists():
    """If the output file already exists, the function returns without rewriting."""
    tmp = get_temp_dir()
    try:
        gt_file = tmp / "cache_gt_K5.npz"
        np.savez_compressed(gt_file, sentinel=np.array([1, 2, 3]))
        mtime_before = gt_file.stat().st_mtime

        build_cache_gt_full(
            cache_query_indices=np.array([0]),
            groundtruth=np.zeros((1, 100), dtype=np.int64),
            full_base_vectors=np.zeros((1, 4), dtype=np.float32),
            query_vectors=np.zeros((1, 4), dtype=np.float32),
            cache_K=5,
            output_dir=tmp,
        )

        data = np.load(gt_file)
        assert 'sentinel' in data.files, "existing file should not have been overwritten"
        assert gt_file.stat().st_mtime == mtime_before
    finally:
        cleanup_temp_dir(tmp)

#-------------------------------------------------------------------------------
