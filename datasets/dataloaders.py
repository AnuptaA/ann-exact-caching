#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import struct
import json
from typing import Dict

#-------------------------------------------------------------------------------

def load_fvecs(filepath: str) -> np.ndarray:
    """
    Load fvecs file and return as numpy array.
    
    Args:
        filepath: Path to fvecs file

    Returns:
        Numpy array of shape (n_vectors, vector_dim)
    """
    with open(filepath, "rb") as f:
        data = f.read()

    dim = struct.unpack('i', data[:4])[0]
    bytes_per_vector = 4 * (dim + 1)
    n_vectors = len(data) // bytes_per_vector

    vectors = np.empty((n_vectors, dim), dtype=np.float32)
    offset = 0

    for i in range(n_vectors):
        offset += 4 
        vector = struct.unpack('f' * dim, data[offset:offset + 4 * dim])
        vectors[i] = np.array(vector, dtype=np.float32)
        offset += 4 * dim

    return vectors

#-------------------------------------------------------------------------------

def load_ivecs(filepath: str) -> np.ndarray:
    """
    Load ivecs file and return as numpy array.
    
    Args:
        filepath: Path to ivecs file

    Returns:
        Numpy array of shape (n_vectors, vector_dim)
    """
    with open(filepath, "rb") as f:
        data = f.read()

    dim = struct.unpack('i', data[:4])[0]
    bytes_per_vector = 4 * (dim + 1)
    n_vectors = len(data) // bytes_per_vector

    vectors = np.empty((n_vectors, dim), dtype=np.int32)
    offset = 0

    for i in range(n_vectors):
        offset += 4
        vector = struct.unpack('i' * dim, data[offset:offset + 4 * dim])
        vectors[i] = np.array(vector, dtype=np.int32)
        offset += 4 * dim

    return vectors

#-------------------------------------------------------------------------------

def load_sift_dataset(
    sift_dir: str = "datasets/sift"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load SIFT1M dataset.
    
    Args:
        sift_dir: Directory containing SIFT files
        
    Returns:
        Tuple of (base_vectors, learn_vectors, query_vectors, groundtruth)
    """
    sift_path = Path(sift_dir)

    print("Loading SIFT dataset...")
    base = load_fvecs(str(sift_path / "sift_base.fvecs"))
    learn = load_fvecs(str(sift_path / "sift_learn.fvecs"))
    query = load_fvecs(str(sift_path / "sift_query.fvecs"))
    groundtruth = load_ivecs(str(sift_path / "sift_groundtruth.ivecs"))

    print(f"  Base vectors: {base.shape}")
    print(f"  Learn vectors: {learn.shape}")
    print(f"  Query vectors: {query.shape}")
    print(f"  Ground truth: {groundtruth.shape}")

    return base, learn, query, groundtruth

#-------------------------------------------------------------------------------

def create_sift_benchmark_split(
    base_vectors: np.ndarray,
    query_vectors: np.ndarray,
    num_cache_queries: int = 9000,
    num_test_queries: int = 1000,
    cache_K: int = 100,
    test_N: int = 10,
    seed: int = 42,
    test_start: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create train/test split for SIFT benchmark.

    Args:
        base_vectors: 1M base vectors (main memory)
        query_vectors: 10k query vectors
        num_cache_queries: Number of queries for cache population (9000)
        num_test_queries: Number of queries for testing (1000)
        cache_K: Result set size for cache queries (100)
        test_N: Result set size for test queries (10)
        seed: Random seed for reproducibility
        test_start: Index in the permutation where test queries begin.
                    Defaults to num_cache_queries (standard non-overlapping split).
                    Set to a fixed value (e.g. 2048) to use the SAME test queries
                    across benchmarks that differ only in cache size (Set 2).

    Returns:
        Tuple of (base_vectors, cache_queries, test_queries)
    """
    if test_start is None:
        test_start = num_cache_queries

    if test_start + num_test_queries > len(query_vectors):
        raise ValueError(
            f"test_start={test_start} + num_test={num_test_queries} exceeds "
            f"available queries ({len(query_vectors)})"
        )
    if num_cache_queries > test_start:
        raise ValueError(
            f"num_cache_queries={num_cache_queries} would overlap with "
            f"test queries starting at test_start={test_start}"
        )

    np.random.seed(seed)

    # randomly split the query set
    indices = np.random.permutation(len(query_vectors))
    cache_indices = indices[:num_cache_queries]
    test_indices = indices[test_start:test_start + num_test_queries]

    cache_queries = query_vectors[cache_indices]
    test_queries = query_vectors[test_indices]
    
    print(f"\nBenchmark split created:")
    print(f"  Main memory: {base_vectors.shape}")
    print(f"  Cache queries: {cache_queries.shape} (K={cache_K})")
    print(f"  Test queries: {test_queries.shape} (N={test_N})")
    
    return base_vectors, cache_queries, test_queries

#-------------------------------------------------------------------------------

def save_sift_benchmark(
    output_dir: str,
    base_vectors: np.ndarray,
    cache_queries: np.ndarray,
    test_queries: np.ndarray,
    config: Dict
):
    """
    Save SIFT benchmark split to disk for reuse.
    
    Args:
        output_dir: Directory to save benchmark
        base_vectors: Main memory vectors
        cache_queries: Cache population queries
        test_queries: Test queries
        config: Benchmark configuration
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving benchmark to: {output_path}")
    
    # save vectors
    np.save(output_path / "base_vectors.npy", base_vectors)
    np.save(output_path / "cache_queries.npy", cache_queries)
    np.save(output_path / "test_queries.npy", test_queries)
    
    # save configuration
    with open(output_path / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("  Saved base vectors, queries, and config")

#-------------------------------------------------------------------------------

def load_sift_benchmark(benchmark_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Load previously saved SIFT benchmark.

    Args:
        benchmark_dir: Directory containing benchmark

    Returns:
        Tuple of (base_vectors, cache_queries, test_queries, config)
    """
    path = Path(benchmark_dir)

    print(f"Loading benchmark from: {path}")

    base_vectors = np.load(path / "base_vectors.npy")
    cache_queries = np.load(path / "cache_queries.npy")
    test_queries = np.load(path / "test_queries.npy")

    with open(path / "config.json", 'r') as f:
        config = json.load(f)

    print(f"  Loaded {base_vectors.shape[0]} base vectors")
    print(f"  Loaded {cache_queries.shape[0]} cache queries")
    print(f"  Loaded {test_queries.shape[0]} test queries")

    return base_vectors, cache_queries, test_queries, config

#-------------------------------------------------------------------------------

def load_esci_data(
    esci_dir: str,
    load_embeddings: bool = False
) -> Tuple:
    """
    Load ESCI dataset files.

    Args:
        esci_dir: Directory containing ESCI files (queries.npz, topk_100shards.npz, etc.)
        load_embeddings: If True, also load product_embeddings.npy (2.8GB)

    Returns:
        If load_embeddings=True: (product_embeddings, query_vectors, topk_ids, topk_scores)
        If load_embeddings=False: (None, query_vectors, topk_ids, topk_scores)
    """
    path = Path(esci_dir)

    queries = np.load(path / "queries.npz")
    query_vectors = queries['query_vectors']  # (130652, 384)

    topk = np.load(path / "topk_100shards.npz")
    topk_ids = topk['topk_ids']      # (130652, 100)
    topk_scores = topk['topk_scores'] # (130652, 100) cosine similarity

    if load_embeddings:
        product_embeddings = np.load(path / "product_embeddings.npy")
        print(f"  Loaded product embeddings: {product_embeddings.shape}")
    else:
        product_embeddings = None

    print(f"  Loaded query vectors: {query_vectors.shape}")
    print(f"  Loaded topk_ids: {topk_ids.shape}, topk_scores: {topk_scores.shape}")

    return product_embeddings, query_vectors, topk_ids, topk_scores

#-------------------------------------------------------------------------------

def save_esci_benchmark(
    output_dir: str,
    cache_queries: np.ndarray,
    test_queries: np.ndarray,
    config: Dict
):
    """
    Save ESCI benchmark split to disk.

    Unlike SIFT, ESCI does not save base_vectors -- the full 1.82M product catalog
    is the fixed base and is not embedded in the benchmark directory.

    Args:
        output_dir: Directory to save benchmark
        cache_queries: Cache population query vectors
        test_queries: Test query vectors
        config: Benchmark configuration dict
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(output_path / "cache_queries.npy", cache_queries)
    np.save(output_path / "test_queries.npy", test_queries)

    with open(output_path / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print(f"  Saved ESCI benchmark to {output_path}")

#-------------------------------------------------------------------------------

def load_esci_benchmark(benchmark_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Load a previously prepared ESCI benchmark.

    Args:
        benchmark_dir: Directory containing benchmark files

    Returns:
        Tuple of (base_vectors, cache_queries, test_queries, config)
    """
    path = Path(benchmark_dir)

    base_vectors = np.load(path / "base_vectors.npy")
    cache_queries = np.load(path / "cache_queries.npy")
    test_queries = np.load(path / "test_queries.npy")

    with open(path / "config.json", 'r') as f:
        config = json.load(f)

    print(f"Loading ESCI benchmark from: {path}")
    print(f"  Loaded {base_vectors.shape[0]} base vectors")
    print(f"  Loaded {cache_queries.shape[0]} cache queries")
    print(f"  Loaded {test_queries.shape[0]} test queries")

    return base_vectors, cache_queries, test_queries, config

#-------------------------------------------------------------------------------