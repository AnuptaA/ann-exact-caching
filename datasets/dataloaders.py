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

def load_fvecs_fast(filepath: str, max_n: Optional[int] = None) -> np.ndarray:
    """
    Vectorized fvecs reader. Same output as load_fvecs.

    Args:
        filepath: Path to fvecs file
        max_n: Read at most this many vectors (all if None)

    Returns:
        Numpy array of shape (n_vectors, vector_dim)
    """
    with open(filepath, "rb") as f:
        dim = int(np.fromfile(f, dtype=np.int32, count=1)[0])
    row = dim + 1
    count = -1 if max_n is None else max_n * row
    raw = np.fromfile(filepath, dtype=np.int32, count=count).reshape(-1, row)
    return np.ascontiguousarray(raw[:, 1:].view(np.float32))

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

    ESCI does not save base_vectors: the full 1.82M product catalog
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