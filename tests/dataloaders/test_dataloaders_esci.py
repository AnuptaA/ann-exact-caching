#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""ESCI-format-specific dataloader tests (queries.npz/topk_100shards.npz/
product_embeddings.npy raw format, benchmark save/load round-trip). Symmetric
to tests/dataloaders/test_dataloaders_sift.py's coverage for SIFT's fvecs/ivecs
format. Uses small synthetic fixtures written to a temp dir -- no real ESCI
data needed (real ESCI raw files are large)."""

import json

import numpy as np

from datasets.dataloaders import (
    load_esci_data,
    save_esci_benchmark,
    load_esci_benchmark,
)
from tests.helpers import get_temp_dir, cleanup_temp_dir

#-------------------------------------------------------------------------------


def _write_esci_raw_fixture(esci_dir, n_queries=20, n_products=50, dim=16, seed=0):
    """Write small synthetic queries.npz/topk_100shards.npz/product_embeddings.npy
    mirroring the real ESCI raw data layout load_esci_data() expects."""
    rng = np.random.RandomState(seed)
    esci_dir.mkdir(parents=True, exist_ok=True)

    query_vectors = rng.randn(n_queries, dim).astype(np.float32)
    np.savez_compressed(esci_dir / "queries.npz", query_vectors=query_vectors)

    topk_ids = np.zeros((n_queries, 100), dtype=np.int64)
    topk_scores = np.zeros((n_queries, 100), dtype=np.float32)
    for qi in range(n_queries):
        scores = rng.uniform(-1.0, 1.0, n_products).astype(np.float32)
        order = np.argsort(-scores)
        # pad/repeat indices up to 100 columns since n_products < 100 here
        padded_order = np.resize(order, 100)
        topk_ids[qi] = padded_order
        topk_scores[qi] = scores[padded_order % n_products]
    np.savez_compressed(esci_dir / "topk_100shards.npz", topk_ids=topk_ids, topk_scores=topk_scores)

    product_embeddings = rng.randn(n_products, dim).astype(np.float32)
    np.save(esci_dir / "product_embeddings.npy", product_embeddings)

    return query_vectors, topk_ids, topk_scores, product_embeddings

#-------------------------------------------------------------------------------


def test_load_esci_data_without_embeddings():
    tmp = get_temp_dir()
    try:
        query_vectors, topk_ids, topk_scores, _ = _write_esci_raw_fixture(tmp)

        product_embeddings, loaded_queries, loaded_ids, loaded_scores = load_esci_data(
            str(tmp), load_embeddings=False
        )

        assert product_embeddings is None
        assert np.allclose(loaded_queries, query_vectors)
        assert np.array_equal(loaded_ids, topk_ids)
        assert np.allclose(loaded_scores, topk_scores)
    finally:
        cleanup_temp_dir(tmp)


def test_load_esci_data_with_embeddings():
    tmp = get_temp_dir()
    try:
        _, _, _, product_embeddings = _write_esci_raw_fixture(tmp)

        loaded_embeddings, _, _, _ = load_esci_data(str(tmp), load_embeddings=True)

        assert loaded_embeddings is not None
        assert np.allclose(loaded_embeddings, product_embeddings)
        assert loaded_embeddings.shape == product_embeddings.shape
    finally:
        cleanup_temp_dir(tmp)

#-------------------------------------------------------------------------------


def test_save_load_esci_benchmark_roundtrip():
    """save_esci_benchmark() does not write base_vectors.npy (the full catalog
    is the fixed base, not embedded per-benchmark) but load_esci_benchmark()
    unconditionally reads one -- write it manually here to exercise the
    round-trip, matching how prepare_benchmark.py's population-extension step
    supplies it in the real pipeline."""
    tmp = get_temp_dir()
    try:
        rng = np.random.RandomState(1)
        dim = 12
        cache_queries = rng.randn(5, dim).astype(np.float32)
        test_queries = rng.randn(3, dim).astype(np.float32)
        config = {
            "dataset": "ESCI", "benchmark_name": "esci_c5k4_t3n2_s1",
            "num_cache_queries": 5, "num_test_queries": 3,
            "cache_K": 4, "test_N": 2, "dimension": dim, "seed": 1,
        }

        save_esci_benchmark(str(tmp), cache_queries, test_queries, config)

        assert (tmp / "cache_queries.npy").exists()
        assert (tmp / "test_queries.npy").exists()
        assert (tmp / "config.json").exists()
        assert not (tmp / "base_vectors.npy").exists(), (
            "save_esci_benchmark should not write base_vectors.npy"
        )

        base_vectors = rng.randn(30, dim).astype(np.float32)
        np.save(tmp / "base_vectors.npy", base_vectors)

        loaded_base, loaded_cache_q, loaded_test_q, loaded_config = load_esci_benchmark(str(tmp))

        assert np.allclose(loaded_base, base_vectors)
        assert np.allclose(loaded_cache_q, cache_queries)
        assert np.allclose(loaded_test_q, test_queries)
        assert loaded_config == config
    finally:
        cleanup_temp_dir(tmp)


def test_save_esci_benchmark_config_roundtrip_types():
    """config.json round-trips through JSON -- confirm no int/float coercion bugs."""
    tmp = get_temp_dir()
    try:
        dim = 4
        cache_queries = np.zeros((2, dim), dtype=np.float32)
        test_queries = np.zeros((1, dim), dtype=np.float32)
        config = {"cache_K": 99, "seed": 42, "description": "test"}

        save_esci_benchmark(str(tmp), cache_queries, test_queries, config)

        with open(tmp / "config.json") as f:
            loaded = json.load(f)
        assert loaded == config
        assert isinstance(loaded["cache_K"], int)
    finally:
        cleanup_temp_dir(tmp)

#-------------------------------------------------------------------------------
