#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import json
import numpy as np
import argparse
from datasets.dataloaders import load_esci_data, save_esci_benchmark

#-------------------------------------------------------------------------------

ESCI_DIR = Path("datasets/esci")

RESEARCH_BENCHMARKS = [
    # Set 1: Baseline -- 1024 cache x 3 seeds
    # test_start=5000 matches Set 2 so 1024 shared with Set 2 uses identical test queries
    dict(num_cache=1024, num_test=512, cache_K=99, test_N=20, seed=42, test_start=5000),
    dict(num_cache=1024, num_test=512, cache_K=99, test_N=20, seed=43, test_start=5000),
    dict(num_cache=1024, num_test=512, cache_K=99, test_N=20, seed=44, test_start=5000),
    # Set 2: Cache size scaling -- 5 sizes x 3 seeds
    # test_start=5000 anchors test queries beyond max cache size (4096) so all sizes use
    # identical test queries (perm[5000:5512]). 1024 is shared with Set 1.
    dict(num_cache=256,  num_test=512, cache_K=99, test_N=20, seed=42, test_start=5000),
    dict(num_cache=256,  num_test=512, cache_K=99, test_N=20, seed=43, test_start=5000),
    dict(num_cache=256,  num_test=512, cache_K=99, test_N=20, seed=44, test_start=5000),
    dict(num_cache=512,  num_test=512, cache_K=99, test_N=20, seed=42, test_start=5000),
    dict(num_cache=512,  num_test=512, cache_K=99, test_N=20, seed=43, test_start=5000),
    dict(num_cache=512,  num_test=512, cache_K=99, test_N=20, seed=44, test_start=5000),
    # 1024 already covered in Set 1
    dict(num_cache=2048, num_test=512, cache_K=99, test_N=20, seed=42, test_start=5000),
    dict(num_cache=2048, num_test=512, cache_K=99, test_N=20, seed=43, test_start=5000),
    dict(num_cache=2048, num_test=512, cache_K=99, test_N=20, seed=44, test_start=5000),
    dict(num_cache=4096, num_test=512, cache_K=99, test_N=20, seed=42, test_start=5000),
    dict(num_cache=4096, num_test=512, cache_K=99, test_N=20, seed=43, test_start=5000),
    dict(num_cache=4096, num_test=512, cache_K=99, test_N=20, seed=44, test_start=5000),
    # Set 2b: cache density investigation (8192, single seed) -- test_start
    # must clear 8192 (can't reuse Set 2's test_start=5000, which would
    # overlap an 8192-sized cache range), so this uses a different held-out
    # test set than the 256-4096 rows above.
    dict(num_cache=8192, num_test=512, cache_K=99, test_N=20, seed=42, test_start=8692),
    # Set 3: K/N relationship -- 9 pairs x 3 seeds; (K,N) in {20,50,99}^2
    dict(num_cache=1024, num_test=256, cache_K=20,  test_N=20,  seed=42),
    dict(num_cache=1024, num_test=256, cache_K=20,  test_N=20,  seed=43),
    dict(num_cache=1024, num_test=256, cache_K=20,  test_N=20,  seed=44),
    dict(num_cache=1024, num_test=256, cache_K=20,  test_N=50,  seed=42),
    dict(num_cache=1024, num_test=256, cache_K=20,  test_N=50,  seed=43),
    dict(num_cache=1024, num_test=256, cache_K=20,  test_N=50,  seed=44),
    dict(num_cache=1024, num_test=256, cache_K=20,  test_N=99,  seed=42),
    dict(num_cache=1024, num_test=256, cache_K=20,  test_N=99,  seed=43),
    dict(num_cache=1024, num_test=256, cache_K=20,  test_N=99,  seed=44),
    dict(num_cache=1024, num_test=256, cache_K=50,  test_N=20,  seed=42),
    dict(num_cache=1024, num_test=256, cache_K=50,  test_N=20,  seed=43),
    dict(num_cache=1024, num_test=256, cache_K=50,  test_N=20,  seed=44),
    dict(num_cache=1024, num_test=256, cache_K=50,  test_N=50,  seed=42),
    dict(num_cache=1024, num_test=256, cache_K=50,  test_N=50,  seed=43),
    dict(num_cache=1024, num_test=256, cache_K=50,  test_N=50,  seed=44),
    dict(num_cache=1024, num_test=256, cache_K=50,  test_N=99,  seed=42),
    dict(num_cache=1024, num_test=256, cache_K=50,  test_N=99,  seed=43),
    dict(num_cache=1024, num_test=256, cache_K=50,  test_N=99,  seed=44),
    dict(num_cache=1024, num_test=256, cache_K=99,  test_N=20,  seed=42),
    dict(num_cache=1024, num_test=256, cache_K=99,  test_N=20,  seed=43),
    dict(num_cache=1024, num_test=256, cache_K=99,  test_N=20,  seed=44),
    dict(num_cache=1024, num_test=256, cache_K=99,  test_N=50,  seed=42),
    dict(num_cache=1024, num_test=256, cache_K=99,  test_N=50,  seed=43),
    dict(num_cache=1024, num_test=256, cache_K=99,  test_N=50,  seed=44),
    dict(num_cache=1024, num_test=256, cache_K=99,  test_N=99,  seed=42),
    dict(num_cache=1024, num_test=256, cache_K=99,  test_N=99,  seed=43),
    dict(num_cache=1024, num_test=256, cache_K=99,  test_N=99,  seed=44),
]

#-------------------------------------------------------------------------------

def create_benchmark_name(num_cache: int, cache_K: int, num_test: int, test_N: int, seed: int) -> str:
    """
    Create benchmark directory name.
    Format: esci_c{cache}k{K}_t{test}n{N}_s{seed}
    """
    return f"esci_c{num_cache}k{cache_K}_t{num_test}n{test_N}_s{seed}"

#-------------------------------------------------------------------------------

def build_cache_gt(
    cache_query_indices: np.ndarray,
    topk_ids: np.ndarray,
    topk_scores: np.ndarray,
    product_embeddings: np.ndarray,
    cache_K: int,
    output_dir: Path
):
    """
    Build and save cache query ground truth from precomputed topk data.

    Converts cosine similarity scores to angular distances and saves the
    top-K product vectors (needed for KVCache population at simulation time).

    Args:
        cache_query_indices: indices into the full query set (130K) for cache queries
        topk_ids: precomputed top-100 product indices, shape (num_queries, 100)
        topk_scores: precomputed top-100 cosine scores, shape (num_queries, 100)
        product_embeddings: full product embedding array, shape (1.82M, 384)
        cache_K: number of neighbors to use (must be <= 100)
        output_dir: benchmark directory to save cache_gt_K{K}.npz
    """
    assert cache_K < 100, f"cache_K must be < 100 to derive gap from precomputed top-100 shards (got {cache_K}); use cache_K=99 instead of 100"

    gt_file = output_dir / f"cache_gt_K{cache_K}.npz"
    if gt_file.exists():
        print(f"  Cache GT already exists: {gt_file.name}")
        return

    print(f"  Building cache GT for {len(cache_query_indices)} cache queries (K={cache_K})...")

    save_dict = {}
    for local_i, global_i in enumerate(cache_query_indices):
        # fetch K+1 entries from precomputed top-100 to compute true gap D(Q,e_{K+1}) - D(Q,e_K)
        # cache_K < 100 is asserted above, so index cache_K is always in bounds
        ids = topk_ids[global_i, :cache_K]               # (K,) product indices
        scores_kp1 = topk_scores[global_i, :cache_K + 1] # (K+1,) cosine similarity

        # convert cosine similarity -> angular distance, normalized by pi -> [0, 1]
        # matches angular_distance() in utils/base/distance_metrics.py
        distances_kp1 = (np.arccos(np.clip(scores_kp1, -1.0, 1.0)) / np.pi).astype(np.float32)
        distances = distances_kp1[:cache_K]              # (K,) distances for cache entry

        # true gap: D(Q, e_{K+1}) - D(Q, e_K)
        gap = float(distances_kp1[cache_K] - distances_kp1[cache_K - 1])

        # actual product vectors needed for KVCache
        vectors = product_embeddings[ids]  # (K, D)

        save_dict[f'indices_{local_i}'] = ids
        save_dict[f'vectors_{local_i}'] = vectors
        save_dict[f'distances_{local_i}'] = distances
        save_dict[f'gap_{local_i}'] = np.float32(gap)

    np.savez_compressed(gt_file, **save_dict)
    print(f"  Saved cache GT: {gt_file.name}")

#-------------------------------------------------------------------------------

def prepare_esci_benchmark(
    num_cache_queries: int = 1024,
    num_test_queries: int = 512,
    cache_K: int = 99,
    test_N: int = 20,
    seed: int = 42,
    force: bool = False,
    test_start: int = None,
):
    """
    Prepare a single ESCI benchmark.

    Samples (num_cache_queries + num_test_queries) queries disjointly from the
    full 130K query set, then slices top-K GT from topk_100shards.npz.
    Saves: cache_queries.npy, test_queries.npy, cache_gt_K{K}.npz, config.json.

    Args:
        num_cache_queries: number of cache population queries
        num_test_queries: number of test queries
        cache_K: K for cache entries (top-K neighbors); must be < 100
        test_N: N for test queries (top-N to retrieve at query time)
        seed: random seed
        force: overwrite existing benchmark
    """
    benchmark_name = create_benchmark_name(num_cache_queries, cache_K, num_test_queries, test_N, seed)
    output_dir = ESCI_DIR / benchmark_name
    gt_file = output_dir / f"cache_gt_K{cache_K}.npz"

    if output_dir.exists() and gt_file.exists() and not force:
        print(f"Benchmark already exists: {benchmark_name}")
        return str(output_dir)

    print(f"\n{'='*80}")
    print(f"Preparing ESCI benchmark: {benchmark_name}")
    print(f"{'='*80}")
    print(f"  Cache queries: {num_cache_queries:,} (K={cache_K})")
    print(f"  Test queries:  {num_test_queries:,} (N={test_N})")
    print(f"  Seed:          {seed}")

    total_queries_needed = num_cache_queries + num_test_queries
    print(f"\nLoading ESCI data (need {total_queries_needed:,} of 130,652 queries)...")

    # Load embeddings -- needed once to extract top-K vectors for cache GT
    product_embeddings, query_vectors, topk_ids, topk_scores = load_esci_data(
        str(ESCI_DIR), load_embeddings=True
    )

    if total_queries_needed > len(query_vectors):
        raise ValueError(
            f"Not enough queries: need {total_queries_needed}, have {len(query_vectors)}"
        )

    # sample disjoint cache + test query indices
    # test_start fixes the anchor for test queries so all cache-size variants
    # share identical test queries (important for Set 2 cache-scaling comparison)
    if test_start is None:
        test_start = num_cache_queries
    np.random.seed(seed)
    perm = np.random.permutation(len(query_vectors))
    cache_indices = perm[:num_cache_queries]
    test_indices = perm[test_start:test_start + num_test_queries]
    if test_start < num_cache_queries:
        raise ValueError(
            f"test_start={test_start} < num_cache_queries={num_cache_queries}: "
            "test queries would overlap with cache queries"
        )

    cache_queries = query_vectors[cache_indices]
    test_queries = query_vectors[test_indices]

    config = {
        "dataset": "ESCI",
        "benchmark_name": benchmark_name,
        "num_cache_queries": num_cache_queries,
        "num_test_queries": num_test_queries,
        "cache_K": cache_K,
        "test_N": test_N,
        "dimension": int(query_vectors.shape[1]),
        "num_base_vectors": int(product_embeddings.shape[0]),
        "seed": seed,
        "description": (
            f"ESCI benchmark: full 1.82M base, "
            f"{num_cache_queries} cache (K={cache_K}), "
            f"{num_test_queries} test (N={test_N}), seed={seed}"
        ),
    }

    # save benchmark split
    save_esci_benchmark(str(output_dir), cache_queries, test_queries, config)

    # build and save cache GT (sliced from topk_100shards.npz)
    build_cache_gt(
        cache_query_indices=cache_indices,
        topk_ids=topk_ids,
        topk_scores=topk_scores,
        product_embeddings=product_embeddings,
        cache_K=cache_K,
        output_dir=output_dir,
    )

    print(f"\nBenchmark ready: {output_dir}")
    return str(output_dir)

#-------------------------------------------------------------------------------
# Population pipeline extension
#-------------------------------------------------------------------------------

K_VALUES = [1, 5, 10, 20]        # per-K GT files for K-microbenchmark

POPULATION_BASE_N = 50000        # number of product vectors to sample
POPULATION_BASE_SEED = 0         # fixed seed for reproducible sample
POPULATION_K = 100               # brute-force KNN for population GT

# k99 source benchmarks used to create k100 population counterparts
_POPULATION_SOURCES = [
    "esci_c256k99_t512n20_s42",
    "esci_c256k99_t512n20_s43",
    "esci_c256k99_t512n20_s44",
    "esci_c512k99_t512n20_s42",
    "esci_c512k99_t512n20_s43",
    "esci_c512k99_t512n20_s44",
    "esci_c1024k99_t512n20_s42",
    "esci_c1024k99_t512n20_s43",
    "esci_c1024k99_t512n20_s44",
    "esci_c2048k99_t512n20_s42",
    "esci_c2048k99_t512n20_s43",
    "esci_c2048k99_t512n20_s44",
    # Cache density investigation (single seed only)
    "esci_c4096k99_t512n20_s42",
    "esci_c8192k99_t512n20_s42",
]

# k100 population benchmark directories (brute-force GT, used by run scripts)
POPULATION_BENCHMARKS = [
    "esci_c256k100_t512n20_s42",
    "esci_c256k100_t512n20_s43",
    "esci_c256k100_t512n20_s44",
    "esci_c512k100_t512n20_s42",
    "esci_c512k100_t512n20_s43",
    "esci_c512k100_t512n20_s44",
    "esci_c1024k100_t512n20_s42",
    "esci_c1024k100_t512n20_s43",
    "esci_c1024k100_t512n20_s44",
    "esci_c2048k100_t512n20_s42",
    "esci_c2048k100_t512n20_s43",
    "esci_c2048k100_t512n20_s44",
    # Cache density investigation (single seed only)
    "esci_c4096k100_t512n20_s42",
    "esci_c8192k100_t512n20_s42",
]

#-------------------------------------------------------------------------------

def _sample_base_vectors(
    product_embeddings: np.ndarray,
    n: int = POPULATION_BASE_N,
    seed: int = POPULATION_BASE_SEED,
) -> np.ndarray:
    """Sample n product vectors with a fixed seed. Returns (n, D) float32 array."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(product_embeddings), n, replace=False)
    idx.sort()
    return product_embeddings[idx].astype(np.float32), idx

#-------------------------------------------------------------------------------

def _brute_force_knn_angular(
    queries: np.ndarray,
    base: np.ndarray,
    K: int,
) -> list:
    """
    Brute-force top-K angular KNN of queries against base.

    Returns list of (indices, distances, gap) per query, where distances are
    angular (arccos / pi) and gap = dist[K] - dist[K-1].
    """
    q = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    b = base / np.linalg.norm(base, axis=1, keepdims=True)

    kp1 = min(K + 1, len(base))
    results = []
    batch = 256  # process in batches to limit peak memory
    for start in range(0, len(queries), batch):
        qb = q[start:start + batch]
        scores = qb @ b.T  # (batch, N_base)
        top_idx = np.argsort(-scores, axis=1)[:, :kp1]
        top_scores = np.take_along_axis(scores, top_idx, axis=1)
        dists = (np.arccos(np.clip(top_scores, -1.0, 1.0)) / np.pi).astype(np.float32)
        for i in range(len(qb)):
            idx = top_idx[i, :K]
            d = dists[i, :K]
            gap = float(dists[i, K] - dists[i, K - 1]) if kp1 > K else 0.0
            results.append((idx, d, gap))
    return results

#-------------------------------------------------------------------------------

def _save_population_gt(
    queries: np.ndarray,
    base: np.ndarray,
    K: int,
    out_path: Path,
) -> None:
    """Compute brute-force KNN and save GT (indices_i, r_Q_i, half_gap_i)."""
    if out_path.exists():
        print(f"  Population GT already exists: {out_path.name}")
        return
    print(f"  Computing population GT for {len(queries)} queries (K={K}) on {len(base)} base vectors...")
    results = _brute_force_knn_angular(queries, base, K)
    save_dict = {}
    for i, (idx, d, gap) in enumerate(results):
        save_dict[f'indices_{i}'] = idx
        save_dict[f'r_Q_{i}'] = np.float32(float(d[-1]))
        save_dict[f'half_gap_{i}'] = np.float32(gap / 2.0)
    np.savez_compressed(out_path, **save_dict)
    print(f"  Saved: {out_path.name}")

#-------------------------------------------------------------------------------

def _brute_force_knn_angular_multi_k(
    queries: np.ndarray,
    base: np.ndarray,
    K_list: list,
) -> dict:
    """
    Brute-force top-K_max angular KNN of queries against base, sliced per K
    in K_list -- one shared O(n_queries * n_base) matmul pass instead of one
    pass per K (top-1 subset top-5 subset ... subset top-k_max are nested).
    Identical math to _brute_force_knn_angular, just parameterized by
    k_max = max(K_list) and sliced afterward (verified bit-identical in
    tests/datasets/test_multi_k_gt.py).

    Returns {k: [(indices, distances, gap), ...] per query} for k in K_list.
    """
    k_max = max(K_list)
    q = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    b = base / np.linalg.norm(base, axis=1, keepdims=True)

    kp1 = min(k_max + 1, len(base))
    results = {k: [] for k in K_list}
    batch = 256
    for start in range(0, len(queries), batch):
        qb = q[start:start + batch]
        scores = qb @ b.T
        top_idx = np.argsort(-scores, axis=1)[:, :kp1]
        top_scores = np.take_along_axis(scores, top_idx, axis=1)
        dists = (np.arccos(np.clip(top_scores, -1.0, 1.0)) / np.pi).astype(np.float32)
        for k in K_list:
            kp1_k = min(k + 1, len(base))
            for i in range(len(qb)):
                idx = top_idx[i, :k]
                d = dists[i, :k]
                gap = float(dists[i, k] - dists[i, k - 1]) if kp1_k > k else 0.0
                results[k].append((idx, d, gap))
    return results


def _save_population_gt_multi_k(
    queries: np.ndarray,
    base: np.ndarray,
    K_list: list,
    output_dir: Path,
    file_prefix: str,
) -> None:
    """Save {file_prefix}_gt_population_K{k}.npz for every k in K_list from
    one shared brute-force pass. Skips any K whose file already exists;
    skips the whole pass if every K is already complete."""
    pending_k = [
        k for k in K_list
        if not (output_dir / f"{file_prefix}_gt_population_K{k}.npz").exists()
    ]
    if not pending_k:
        print(f"  All {file_prefix} multi-K population GT files already exist, skipping.")
        return

    print(f"  Computing {file_prefix} population GT for {len(queries)} queries "
          f"(K={pending_k}) on {len(base)} base vectors...")
    results_by_k = _brute_force_knn_angular_multi_k(queries, base, pending_k)

    for k in pending_k:
        save_dict = {}
        for i, (idx, d, gap) in enumerate(results_by_k[k]):
            save_dict[f'indices_{i}'] = idx
            save_dict[f'r_Q_{i}'] = np.float32(float(d[-1]))
            save_dict[f'half_gap_{i}'] = np.float32(gap / 2.0)
        out_path = output_dir / f"{file_prefix}_gt_population_K{k}.npz"
        np.savez_compressed(out_path, **save_dict)
        print(f"  Saved: {out_path.name}")

#-------------------------------------------------------------------------------

def create_esci_population_benchmark(
    source_dir: Path,
    product_embeddings: np.ndarray,
) -> None:
    """
    Create a K=100 population benchmark directory from an existing K=99 benchmark.

    The new directory shares cache_queries.npy and test_queries.npy with the source
    but uses K=100 brute-force GT against the 50K base sample, so all K+1 neighbors
    are guaranteed to exist.
    """
    config_file = source_dir / 'config.json'
    with open(config_file) as f:
        source_config = json.load(f)

    num_cache = source_config['num_cache_queries']
    num_test = source_config['num_test_queries']
    test_N = source_config['test_N']
    seed = source_config['seed']
    benchmark_name = create_benchmark_name(num_cache, POPULATION_K, num_test, test_N, seed)
    output_dir = ESCI_DIR / benchmark_name

    cache_gt_file = output_dir / f'cache_gt_population_K{POPULATION_K}.npz'
    test_gt_file = output_dir / f'test_gt_population_K{POPULATION_K}.npz'
    k_gts_exist = all(
        (output_dir / f'cache_gt_population_K{k}.npz').exists() for k in K_VALUES
    )
    test_k_gts_exist = all(
        (output_dir / f'test_gt_population_K{k}.npz').exists() for k in K_VALUES
    )
    if (cache_gt_file.exists() and test_gt_file.exists()
            and (output_dir / 'test_r_t.npy').exists()
            and k_gts_exist and test_k_gts_exist):
        print(f"  Population benchmark already complete: {benchmark_name}")
        return

    print(f"\n{'='*80}")
    print(f"Creating K=100 population benchmark: {benchmark_name}")
    print(f"{'='*80}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_q = np.load(source_dir / 'cache_queries.npy')
    test_q = np.load(source_dir / 'test_queries.npy')
    np.save(output_dir / 'cache_queries.npy', cache_q)
    np.save(output_dir / 'test_queries.npy', test_q)

    config = dict(source_config)
    config['cache_K'] = POPULATION_K
    config['benchmark_name'] = benchmark_name
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # base_vectors.npy — 50K sample, fixed seed
    base_file = output_dir / 'base_vectors.npy'
    idx_file = output_dir / 'base_sample_indices.npy'
    if base_file.exists():
        print("  base_vectors.npy already exists, loading...")
        base = np.load(base_file)
    else:
        print(f"  Sampling {POPULATION_BASE_N} base vectors (seed={POPULATION_BASE_SEED})...")
        base, sample_idx = _sample_base_vectors(product_embeddings, POPULATION_BASE_N, POPULATION_BASE_SEED)
        np.save(base_file, base)
        np.save(idx_file, sample_idx)
        print(f"  Saved: base_vectors.npy  shape={base.shape}")

    # cache_gt_population_K100.npz
    _save_population_gt(cache_q, base, POPULATION_K, cache_gt_file)

    # cache_r_Q.npy
    r_Q_file = output_dir / 'cache_r_Q.npy'
    if r_Q_file.exists():
        print("  cache_r_Q.npy already exists, skipping.")
    else:
        data = np.load(cache_gt_file)
        r_Q = np.array([float(data[f'r_Q_{i}']) for i in range(len(cache_q))], dtype=np.float32)
        np.save(r_Q_file, r_Q)
        print(f"  Saved: cache_r_Q.npy  range=[{r_Q.min():.4f}, {r_Q.max():.4f}]")

    # test_gt_population_K100.npz
    _save_population_gt(test_q, base, POPULATION_K, test_gt_file)

    # test_r_t.npy
    r_t_file = output_dir / 'test_r_t.npy'
    if r_t_file.exists():
        print("  test_r_t.npy already exists, skipping.")
    else:
        data = np.load(test_gt_file)
        r_t = np.array([float(data[f'r_Q_{i}']) for i in range(len(test_q))], dtype=np.float32)
        np.save(r_t_file, r_t)
        print(f"  Saved: test_r_t.npy  range=[{r_t.min():.4f}, {r_t.max():.4f}]")

    # per-K GT files for K-microbenchmark (K=[1,5,10,20]): one shared pass
    # at max(K_VALUES), sliced per K, instead of one pass per K.
    _save_population_gt_multi_k(cache_q, base, K_VALUES, output_dir, "cache")

    # per-K test GT files for K-override runs
    _save_population_gt_multi_k(test_q, base, K_VALUES, output_dir, "test")

    config['has_population_data'] = True
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Done: {benchmark_name}")

#-------------------------------------------------------------------------------

def extend_benchmark_with_population_data(
    benchmark_dir: Path,
    product_embeddings: np.ndarray = None,
) -> None:
    """
    Extend an existing ESCI benchmark with population-pipeline data files:
      base_vectors.npy, cache_gt_population_K100.npz, cache_r_Q.npy,
      test_gt_population_K100.npz, test_r_t.npy

    All files are guarded by existence checks; safe to re-run on a partially
    extended benchmark. Patches config.json with has_population_data=true.

    product_embeddings can be passed in to avoid reloading 1.82M vectors per call.
    """
    config_file = benchmark_dir / 'config.json'
    if not config_file.exists():
        raise FileNotFoundError(f"No config.json in {benchmark_dir}")
    with open(config_file) as f:
        config = json.load(f)

    cache_q = np.load(benchmark_dir / 'cache_queries.npy')
    test_q = np.load(benchmark_dir / 'test_queries.npy')

    print(f"\n{'='*80}")
    print(f"Extending with population data: {benchmark_dir.name}")
    print(f"{'='*80}")

    # base_vectors.npy — 50K sample, fixed seed
    base_file = benchmark_dir / 'base_vectors.npy'
    idx_file = benchmark_dir / 'base_sample_indices.npy'
    if base_file.exists():
        print("  base_vectors.npy already exists, loading...")
        base = np.load(base_file)
    else:
        if product_embeddings is None:
            print("  Loading product embeddings (1.82M vectors)...")
            product_embeddings = np.load(ESCI_DIR / 'product_embeddings.npy')
        print(f"  Sampling {POPULATION_BASE_N} base vectors (seed={POPULATION_BASE_SEED})...")
        base, sample_idx = _sample_base_vectors(product_embeddings, POPULATION_BASE_N, POPULATION_BASE_SEED)
        np.save(base_file, base)
        np.save(idx_file, sample_idx)
        print(f"  Saved: base_vectors.npy  shape={base.shape}")

    # cache_gt_population_K100.npz
    cache_gt_file = benchmark_dir / f'cache_gt_population_K{POPULATION_K}.npz'
    _save_population_gt(cache_q, base, POPULATION_K, cache_gt_file)

    # cache_r_Q.npy
    r_Q_file = benchmark_dir / 'cache_r_Q.npy'
    if r_Q_file.exists():
        print("  cache_r_Q.npy already exists, skipping.")
    else:
        data = np.load(cache_gt_file)
        r_Q = np.array(
            [float(data[f'r_Q_{i}']) for i in range(len(cache_q))],
            dtype=np.float32,
        )
        np.save(r_Q_file, r_Q)
        print(f"  Saved: cache_r_Q.npy  range=[{r_Q.min():.4f}, {r_Q.max():.4f}]")

    # test_gt_population_K100.npz
    test_gt_file = benchmark_dir / f'test_gt_population_K{POPULATION_K}.npz'
    _save_population_gt(test_q, base, POPULATION_K, test_gt_file)

    # test_r_t.npy
    r_t_file = benchmark_dir / 'test_r_t.npy'
    if r_t_file.exists():
        print("  test_r_t.npy already exists, skipping.")
    else:
        data = np.load(test_gt_file)
        r_t = np.array(
            [float(data[f'r_Q_{i}']) for i in range(len(test_q))],
            dtype=np.float32,
        )
        np.save(r_t_file, r_t)
        print(f"  Saved: test_r_t.npy  range=[{r_t.min():.4f}, {r_t.max():.4f}]")

    # per-K GT files for K-microbenchmark (K=[1,5,10,20]): one shared pass
    # at max(K_VALUES), sliced per K, instead of one pass per K.
    _save_population_gt_multi_k(cache_q, base, K_VALUES, benchmark_dir, "cache")

    # per-K test GT files for K-override runs
    _save_population_gt_multi_k(test_q, base, K_VALUES, benchmark_dir, "test")

    # patch config
    if not config.get('has_population_data'):
        config['has_population_data'] = True
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print("  Patched config.json with has_population_data=true.")

    print(f"  Done: {benchmark_dir.name}")

#-------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare ESCI benchmarks (full 1.82M base, GT from topk_100shards.npz)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--all',
        action='store_true',
        help='Prepare all research benchmarks'
    )
    group.add_argument(
        '--preset',
        type=str,
        choices=['baseline', 'cache_scaling', 'kn_relationship', 'population'],
        help='Prepare benchmarks for one test set'
    )
    group.add_argument(
        '--extend-population',
        action='store_true',
        help='Extend all population benchmarks with base_vectors + population GT files'
    )

    parser.add_argument('--cache', type=int, default=1024, help='Number of cache queries')
    parser.add_argument('--test', type=int, default=512, help='Number of test queries')
    parser.add_argument('--cache-k', type=int, default=99, help='K for cache entries (must be < 100)')
    parser.add_argument('--test-n', type=int, default=20, help='N for test queries')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--force', action='store_true', help='Overwrite existing benchmarks')

    args = parser.parse_args()

    # define preset subsets
    preset_configs = {
        'baseline': [b for b in RESEARCH_BENCHMARKS
                     if b['num_cache'] in (512, 1024, 2048) and b['cache_K'] == 99 and b['test_N'] == 20],
        'cache_scaling': [b for b in RESEARCH_BENCHMARKS
                          if b['cache_K'] == 99 and b['test_N'] == 20],
        'kn_relationship': [b for b in RESEARCH_BENCHMARKS
                            if b['num_cache'] == 1024 and b['num_test'] == 256],
        # exactly the K=99 source benchmarks _POPULATION_SOURCES derives the
        # K=100 population benchmarks from -- avoids --all's wasted prep of
        # the K/N-relationship and c4096 benchmarks population work never uses.
        'population': [b for b in RESEARCH_BENCHMARKS
                       if create_benchmark_name(
                           b['num_cache'], b['cache_K'], b['num_test'], b['test_N'], b['seed']
                       ) in _POPULATION_SOURCES],
    }

    if args.extend_population:
        print(f"\n{'='*80}")
        print(f"Creating K=100 population benchmarks from {len(_POPULATION_SOURCES)} source benchmarks")
        print(f"{'='*80}\n")
        product_embeddings = None
        failed = []
        for name in _POPULATION_SOURCES:
            source_dir = ESCI_DIR / name
            if not source_dir.exists():
                print(f"Skipping {name}: source benchmark not prepared yet")
                failed.append(name)
                continue
            try:
                if product_embeddings is None:
                    print("Loading product embeddings (1.82M vectors)...")
                    product_embeddings = np.load(ESCI_DIR / 'product_embeddings.npy')
                create_esci_population_benchmark(source_dir, product_embeddings)
            except Exception as e:
                print(f"\nERROR: {name} failed: {e}")
                failed.append(name)
        print(f"\n{'='*80}")
        print(f"Done: {len(_POPULATION_SOURCES) - len(failed)}/{len(_POPULATION_SOURCES)} succeeded")
        if failed:
            for b in failed:
                print(f"  {b}")
        print(f"{'='*80}")
        import sys
        sys.exit(1 if failed else 0)

    if args.all:
        configs = RESEARCH_BENCHMARKS
    elif args.preset:
        configs = preset_configs[args.preset]
    else:
        # single benchmark from CLI args
        configs = [dict(
            num_cache=args.cache,
            num_test=args.test,
            cache_K=args.cache_k,
            test_N=args.test_n,
            seed=args.seed,
        )]

    print(f"\n{'='*80}")
    print(f"ESCI Benchmark Preparation")
    print(f"{'='*80}")
    print(f"Benchmarks to prepare: {len(configs)}")
    print(f"{'='*80}\n")

    failed = []
    for cfg in configs:
        try:
            prepare_esci_benchmark(
                num_cache_queries=cfg['num_cache'],
                num_test_queries=cfg['num_test'],
                cache_K=cfg['cache_K'],
                test_N=cfg['test_N'],
                seed=cfg['seed'],
                force=args.force,
                test_start=cfg.get('test_start'),
            )
        except Exception as e:
            name = create_benchmark_name(cfg['num_cache'], cfg['cache_K'], cfg['num_test'], cfg['test_N'], cfg['seed'])
            print(f"\nERROR: {name} failed: {e}")
            failed.append(name)

    print(f"\n{'='*80}")
    print(f"Done: {len(configs) - len(failed)}/{len(configs)} benchmarks succeeded")
    if failed:
        print("Failed:")
        for b in failed:
            print(f"  {b}")
    print(f"{'='*80}")

    import sys
    sys.exit(1 if failed else 0)

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#-------------------------------------------------------------------------------
