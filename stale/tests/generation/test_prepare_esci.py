#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Unit tests for datasets/esci/prepare_benchmark.py's own logic: benchmark
naming, base-vector sampling, and the lean cache GT builder. Uses small dummy
arrays throughout; no real ESCI data needed."""

import numpy as np

from datasets.esci.prepare_benchmark import (
    create_benchmark_name,
    _sample_base_vectors,
    build_cache_gt,
    _brute_force_knn_angular,
    _brute_force_knn_angular_multi_k,
    RESEARCH_BENCHMARKS,
    _POPULATION_SOURCES,
)
from tests.helpers import get_temp_dir, cleanup_temp_dir

#-------------------------------------------------------------------------------


def test_create_benchmark_name_format():
    name = create_benchmark_name(num_cache=1024, cache_K=99, num_test=512, test_N=20, seed=42)
    assert name == "esci_c1024k99_t512n20_s42"


def test_create_benchmark_name_varies_by_param():
    a = create_benchmark_name(256, 20, 256, 50, 43)
    b = create_benchmark_name(256, 20, 256, 99, 43)
    assert a != b
    assert a == "esci_c256k20_t256n50_s43"
    assert b == "esci_c256k20_t256n99_s43"

#-------------------------------------------------------------------------------


def test_sample_base_vectors_deterministic():
    """Fixed seed must produce identical samples across calls."""
    rng = np.random.RandomState(0)
    embeddings = rng.randn(1000, 16).astype(np.float32)

    vecs1, idx1 = _sample_base_vectors(embeddings, n=100, seed=7)
    vecs2, idx2 = _sample_base_vectors(embeddings, n=100, seed=7)

    assert np.array_equal(idx1, idx2)
    assert np.allclose(vecs1, vecs2)


def test_sample_base_vectors_shape_and_no_duplicates():
    rng = np.random.RandomState(1)
    embeddings = rng.randn(500, 8).astype(np.float32)

    vecs, idx = _sample_base_vectors(embeddings, n=50, seed=1)

    assert vecs.shape == (50, 8)
    assert vecs.dtype == np.float32
    assert len(set(idx.tolist())) == 50, "sampled indices should be unique"
    assert list(idx) == sorted(idx), "indices should be sorted ascending"


def test_sample_base_vectors_different_seeds_differ():
    rng = np.random.RandomState(2)
    embeddings = rng.randn(1000, 8).astype(np.float32)

    _, idx_a = _sample_base_vectors(embeddings, n=100, seed=1)
    _, idx_b = _sample_base_vectors(embeddings, n=100, seed=2)

    assert not np.array_equal(idx_a, idx_b)

#-------------------------------------------------------------------------------


def test_population_preset_matches_population_sources():
    """The 'population' preset in main() (see prepare_benchmark.py's
    preset_configs) filters RESEARCH_BENCHMARKS down to exactly
    _POPULATION_SOURCES, the 12 K=99 benchmarks run_population.py's
    K=100 population benchmarks actually derive from, so that
    esci-population-prepare's --preset population doesn't waste time
    preparing the K/N-relationship or c4096 benchmarks population work
    never uses. Replicates main()'s filter expression directly (the dict
    itself is local to main(), not importable) rather than invoking main()
    with argv, which would perform real file I/O."""
    population_preset = [
        b for b in RESEARCH_BENCHMARKS
        if create_benchmark_name(
            b['num_cache'], b['cache_K'], b['num_test'], b['test_N'], b['seed']
        ) in _POPULATION_SOURCES
    ]
    names = {
        create_benchmark_name(b['num_cache'], b['cache_K'], b['num_test'], b['test_N'], b['seed'])
        for b in population_preset
    }
    assert names == set(_POPULATION_SOURCES)
    assert len(population_preset) == len(_POPULATION_SOURCES), "no duplicate matches"


def test_multi_k_gt_matches_per_k_brute_force():
    """_brute_force_knn_angular_multi_k (one shared O(n_queries*n_base) pass,
    sliced per K) must be bit-identical to calling _brute_force_knn_angular
    once per K. The multi-K dedup relies on slicing from a single max-K pass
    being exact, not an approximation."""
    rng = np.random.RandomState(21)
    dim = 6
    n_base = 40
    n_queries = 5
    K_list = [1, 5, 20]

    base = rng.randn(n_base, dim).astype(np.float32)
    queries = rng.randn(n_queries, dim).astype(np.float32)

    multi_k_results = _brute_force_knn_angular_multi_k(queries, base, K_list)
    for k in K_list:
        expected = _brute_force_knn_angular(queries, base, k)
        for qi in range(n_queries):
            idx, d, gap = multi_k_results[k][qi]
            expected_idx, expected_d, expected_gap = expected[qi]

            assert np.array_equal(idx, expected_idx), f"qi={qi} k={k}: indices differ"
            assert np.array_equal(d, expected_d), f"qi={qi} k={k}: distances differ"
            assert gap == expected_gap, f"qi={qi} k={k}: gap differs"


def test_build_cache_gt_schema_and_values():
    """Verify indices/vectors/distances/gap match a direct recomputation from
    dummy topk_ids/topk_scores/product_embeddings arrays."""
    rng = np.random.RandomState(3)
    dim = 8
    n_products = 200
    n_queries = 5
    cache_K = 10

    product_embeddings = rng.randn(n_products, dim).astype(np.float32)
    # 100 neighbors per query, sorted by descending cosine similarity (as real shards are)
    topk_ids = np.zeros((n_queries, 100), dtype=np.int64)
    topk_scores = np.zeros((n_queries, 100), dtype=np.float32)
    for qi in range(n_queries):
        scores = rng.uniform(-1.0, 1.0, n_products).astype(np.float32)
        order = np.argsort(-scores)[:100]
        topk_ids[qi] = order
        topk_scores[qi] = scores[order]

    tmp = get_temp_dir()
    try:
        cache_query_indices = np.array([0, 2, 4])
        build_cache_gt(
            cache_query_indices=cache_query_indices,
            topk_ids=topk_ids,
            topk_scores=topk_scores,
            product_embeddings=product_embeddings,
            cache_K=cache_K,
            output_dir=tmp,
        )

        gt_file = tmp / f"cache_gt_K{cache_K}.npz"
        assert gt_file.exists()
        data = np.load(gt_file)

        for local_i, global_i in enumerate(cache_query_indices):
            ids = data[f'indices_{local_i}']
            vectors = data[f'vectors_{local_i}']
            distances = data[f'distances_{local_i}']
            gap = float(data[f'gap_{local_i}'])

            expected_ids = topk_ids[global_i, :cache_K]
            assert list(ids) == list(expected_ids)
            assert np.allclose(vectors, product_embeddings[expected_ids])

            scores_kp1 = topk_scores[global_i, :cache_K + 1]
            expected_dists_kp1 = np.arccos(np.clip(scores_kp1, -1.0, 1.0)) / np.pi
            assert np.allclose(distances, expected_dists_kp1[:cache_K], atol=1e-5)

            expected_gap = float(expected_dists_kp1[cache_K] - expected_dists_kp1[cache_K - 1])
            assert abs(gap - expected_gap) < 1e-5
    finally:
        cleanup_temp_dir(tmp)


def test_build_cache_gt_rejects_k_ge_100():
    tmp = get_temp_dir()
    try:
        dummy = np.zeros((1, 1), dtype=np.float32)
        try:
            build_cache_gt(
                cache_query_indices=np.array([0]),
                topk_ids=np.zeros((1, 100), dtype=np.int64),
                topk_scores=np.zeros((1, 100), dtype=np.float32),
                product_embeddings=dummy,
                cache_K=100,
                output_dir=tmp,
            )
            assert False, "expected AssertionError for cache_K=100"
        except AssertionError as e:
            assert "cache_K must be < 100" in str(e)
    finally:
        cleanup_temp_dir(tmp)


def test_build_cache_gt_skips_if_exists():
    tmp = get_temp_dir()
    try:
        gt_file = tmp / "cache_gt_K5.npz"
        np.savez_compressed(gt_file, sentinel=np.array([1, 2, 3]))
        mtime_before = gt_file.stat().st_mtime

        build_cache_gt(
            cache_query_indices=np.array([0]),
            topk_ids=np.zeros((1, 100), dtype=np.int64),
            topk_scores=np.zeros((1, 100), dtype=np.float32),
            product_embeddings=np.zeros((1, 4), dtype=np.float32),
            cache_K=5,
            output_dir=tmp,
        )

        data = np.load(gt_file)
        assert 'sentinel' in data.files, "existing file should not have been overwritten"
        assert gt_file.stat().st_mtime == mtime_before
    finally:
        cleanup_temp_dir(tmp)

#-------------------------------------------------------------------------------
