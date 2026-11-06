#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import json
import numpy as np
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from datasets.dataloaders import (
    load_sift_dataset,
    load_sift_benchmark,
    create_sift_benchmark_split,
    save_sift_benchmark,
    sift_query_split_indices,
)
from utils.base import MainMemory

#-------------------------------------------------------------------------------

K_VALUES = [1, 5, 10, 20]   # per-K GT files for K-microbenchmark

RESEARCH_BENCHMARKS = [
    # Set 1: Baseline (1024 cache x 3 seeds; shared with Set 2)
    dict(num_base_vectors=50_000, num_cache_queries=1024, num_test_queries=512,
         cache_K=100, test_N=20, seed=42, test_start=2048),
    dict(num_base_vectors=50_000, num_cache_queries=1024, num_test_queries=512,
         cache_K=100, test_N=20, seed=43, test_start=2048),
    dict(num_base_vectors=50_000, num_cache_queries=1024, num_test_queries=512,
         cache_K=100, test_N=20, seed=44, test_start=2048),
    # Set 2: Cache size scaling (3 sizes x 3 seeds; 1024 shared with Set 1)
    dict(num_base_vectors=50_000, num_cache_queries=256,  num_test_queries=512,
         cache_K=100, test_N=20, seed=42, test_start=2048),
    dict(num_base_vectors=50_000, num_cache_queries=256,  num_test_queries=512,
         cache_K=100, test_N=20, seed=43, test_start=2048),
    dict(num_base_vectors=50_000, num_cache_queries=256,  num_test_queries=512,
         cache_K=100, test_N=20, seed=44, test_start=2048),
    dict(num_base_vectors=50_000, num_cache_queries=512,  num_test_queries=512,
         cache_K=100, test_N=20, seed=42, test_start=2048),
    dict(num_base_vectors=50_000, num_cache_queries=512,  num_test_queries=512,
         cache_K=100, test_N=20, seed=43, test_start=2048),
    dict(num_base_vectors=50_000, num_cache_queries=512,  num_test_queries=512,
         cache_K=100, test_N=20, seed=44, test_start=2048),
    dict(num_base_vectors=50_000, num_cache_queries=2048, num_test_queries=512,
         cache_K=100, test_N=20, seed=42, test_start=2048),
    dict(num_base_vectors=50_000, num_cache_queries=2048, num_test_queries=512,
         cache_K=100, test_N=20, seed=43, test_start=2048),
    dict(num_base_vectors=50_000, num_cache_queries=2048, num_test_queries=512,
         cache_K=100, test_N=20, seed=44, test_start=2048),
    # Set 3: cache density investigation (4096/8192, single seed), share
    # test_start=8192 between the two so they use the identical held-out
    # test set for a controlled comparison; can't reuse Set 1/2's
    # test_start=2048 since cache_queries would then overlap the test range.
    dict(num_base_vectors=50_000, num_cache_queries=4096, num_test_queries=512,
         cache_K=100, test_N=20, seed=42, test_start=8192),
    dict(num_base_vectors=50_000, num_cache_queries=8192, num_test_queries=512,
         cache_K=100, test_N=20, seed=42, test_start=8192),
]

#-------------------------------------------------------------------------------

def create_benchmark_name(num_base: int, num_cache: int, num_test: int, cache_K: int, test_N: int, seed: int = 42) -> str:
    """
    Create unique benchmark directory name from parameters.

    Format: sift_b{base}_c{cache}k{K}_t{test}n{N}_s{seed}
    Example: sift_b50k_c1024k100_t512n20_s42
    """
    base_str = f"{num_base//1000}k" if num_base >= 1000 else str(num_base)
    if num_base >= 1_000_000:
        base_str = f"{num_base//1_000_000}m"

    return f"sift_b{base_str}_c{num_cache}k{cache_K}_t{num_test}n{test_N}_s{seed}"

#-------------------------------------------------------------------------------

def _compute_single_query_gt(args):
    """helper for parallel ground truth computation."""
    query_id, query, base_vectors, K, metric = args

    mm = MainMemory(vectors=list(base_vectors))
    _, top_k_dists, gap, top_k_indices = mm.top_k_search(
        query, k=K, metric=metric, return_indices=True
    )

    return query_id, np.array(top_k_indices), np.array(top_k_dists), gap

#-------------------------------------------------------------------------------

def _compute_single_query_gt_multi_k(args):
    """helper for parallel multi-K ground truth computation.

    Optimally duplicates MainMemory.top_k_search's O(n_base * D) matmul
    pass for the max K, then slices the results

    Returns (query_id, {k: (indices, r_Q, half_gap) for k in K_list}).
    """
    query_id, query, base_vectors, K_list, metric = args
    k_max = max(K_list)

    mm = MainMemory(vectors=list(base_vectors))
    q_norm = float(np.linalg.norm(query))
    q_hat = query / q_norm if q_norm != 0.0 else query.copy()
    if metric == "angular":
        dots = np.clip(mm._B_hat @ q_hat, -1.0, 1.0)
        dists = np.arccos(dots) / np.pi
    else:
        dists = np.linalg.norm(mm._B_hat - q_hat, axis=1)

    n = len(dists)
    kp1 = min(k_max + 1, n)
    part = np.argpartition(dists, min(k_max, n - 1))[:kp1]
    top_idx = part[np.argsort(dists[part])]
    top_dists = dists[top_idx]

    results = {}
    for k in K_list:
        kp1_k = min(k + 1, n)
        idx_k = top_idx[:k]
        d_k = top_dists[:k]
        gap = float(top_dists[k] - top_dists[k - 1]) if kp1_k > k else 0.0
        results[k] = (idx_k, float(d_k[-1]), gap / 2.0)

    return query_id, results

#-------------------------------------------------------------------------------

def _check_partial_gt(gt_file: Path, expected_count: int) -> tuple[dict, set]:
    """Load partial ground truth from disk if it exists."""
    if not gt_file.exists():
        return {}, set()

    try:
        data = np.load(gt_file)
        ground_truth = {}
        completed = set()

        for i in range(expected_count):
            if f'indices_{i}' in data and f'r_Q_{i}' in data:
                ground_truth[i] = (
                    data[f'indices_{i}'],
                    float(data[f'r_Q_{i}']),
                    float(data[f'half_gap_{i}']),
                )
                completed.add(i)

        if completed:
            print(f"  Found partial ground truth: {len(completed)}/{expected_count} completed")

        return ground_truth, completed
    except Exception as e:
        print(f"  Warning: could not load partial ground truth: {e}")
        return {}, set()

#-------------------------------------------------------------------------------

def precompute_cache_ground_truth(
    base_vectors: np.ndarray,
    cache_queries: np.ndarray,
    K: int,
    output_dir: Path,
    num_workers: int = None,
    checkpoint_interval: int = 100
) -> dict:
    """
    Precompute top-K ground truth for cache queries using angular metric.
    Results are saved to output_dir/cache_gt_population_K{K}.npz.
    Supports resuming from partial completion.

    Stored format per entry: indices_i (K ints), r_Q_i (float), half_gap_i (float).

    Args:
        base_vectors: base vector array (M x D)
        cache_queries: cache query vectors (C x D)
        K: number of neighbors per cache entry
        output_dir: benchmark directory to save ground truth
        num_workers: parallel workers (default: cpu_count - 1)
        checkpoint_interval: save to disk every N queries

    Returns:
        dict mapping query_id -> (indices, r_Q, half_gap)
    """
    gt_file = output_dir / f"cache_gt_population_K{K}.npz"
    ground_truth, completed = _check_partial_gt(gt_file, len(cache_queries))

    if len(completed) == len(cache_queries):
        print(f"Cache ground truth already complete ({len(cache_queries)} queries).")
        return ground_truth

    remaining = len(cache_queries) - len(completed)
    print(f"\nPrecomputing cache ground truth: {remaining}/{len(cache_queries)} remaining...")
    print(f"  Metric: angular, K: {K}")

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    print(f"  Using {num_workers} parallel workers")
    print(f"  Checkpointing every {checkpoint_interval} queries")

    args_list = [
        (i, cache_queries[i], base_vectors, K, "angular")
        for i in range(len(cache_queries))
        if i not in completed
    ]

    with Pool(num_workers) as pool:
        for chunk_start in range(0, len(args_list), checkpoint_interval):
            chunk_end = min(chunk_start + checkpoint_interval, len(args_list))
            chunk = args_list[chunk_start:chunk_end]

            print(f"\n  Processing chunk {chunk_start // checkpoint_interval + 1}/"
                  f"{(len(args_list) + checkpoint_interval - 1) // checkpoint_interval}...")

            results = list(tqdm(
                pool.imap(_compute_single_query_gt, chunk),
                total=len(chunk),
                desc="  Computing cache GT",
                file=sys.stderr
            ))

            for query_id, indices, distances, gap in results:
                ground_truth[query_id] = (indices, float(distances[-1]), float(gap) / 2.0)

            print(f"  Saving checkpoint... ({len(ground_truth)}/{len(cache_queries)} complete)")
            save_dict = {}
            for i, (indices, r_Q, half_gap) in ground_truth.items():
                save_dict[f'indices_{i}'] = indices
                save_dict[f'r_Q_{i}'] = np.float32(r_Q)
                save_dict[f'half_gap_{i}'] = np.float32(half_gap)
            np.savez_compressed(gt_file, **save_dict)

    print(f"\nCache ground truth complete: {gt_file}")
    return ground_truth

#-------------------------------------------------------------------------------

def precompute_multi_k_ground_truth(
    base_vectors: np.ndarray,
    queries: np.ndarray,
    K_list: list,
    output_dir: Path,
    file_prefix: str,
    num_workers: int = None,
    checkpoint_interval: int = 100,
) -> None:
    """
    Precompute {file_prefix}_gt_population_K{k}.npz for every k in K_list in
    one brute-force pass per query. Skips any K whose file already
    exists; skips the whole pass if every K is already complete.
    """
    pending_k = sorted({
        k for k in K_list
        if not (output_dir / f"{file_prefix}_gt_population_K{k}.npz").exists()
    })
    if not pending_k:
        print(f"  All {file_prefix} multi-K ground truth files already exist, skipping.")
        return

    print(f"\nPrecomputing {file_prefix} ground truth for K={pending_k} "
          f"({len(queries)} queries, one shared brute-force pass)...")

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    ground_truth_by_k = {k: {} for k in pending_k}
    args_list = [(i, queries[i], base_vectors, pending_k, "angular") for i in range(len(queries))]

    with Pool(num_workers) as pool:
        for chunk_start in range(0, len(args_list), checkpoint_interval):
            chunk_end = min(chunk_start + checkpoint_interval, len(args_list))
            chunk = args_list[chunk_start:chunk_end]

            print(f"\n  Processing chunk {chunk_start // checkpoint_interval + 1}/"
                  f"{(len(args_list) + checkpoint_interval - 1) // checkpoint_interval}...")

            results = list(tqdm(
                pool.imap(_compute_single_query_gt_multi_k, chunk),
                total=len(chunk),
                desc=f"  Computing {file_prefix} multi-K GT",
                file=sys.stderr,
            ))

            for query_id, per_k in results:
                for k, (indices, r_Q, half_gap) in per_k.items():
                    ground_truth_by_k[k][query_id] = (indices, r_Q, half_gap)

    for k in pending_k:
        save_dict = {}
        for i, (indices, r_Q, half_gap) in ground_truth_by_k[k].items():
            save_dict[f'indices_{i}'] = indices
            save_dict[f'r_Q_{i}'] = np.float32(r_Q)
            save_dict[f'half_gap_{i}'] = np.float32(half_gap)
        gt_file = output_dir / f"{file_prefix}_gt_population_K{k}.npz"
        np.savez_compressed(gt_file, **save_dict)
        print(f"  Saved: {gt_file}")

#-------------------------------------------------------------------------------

def build_cache_gt_full(
    cache_query_indices: np.ndarray,
    groundtruth: np.ndarray,
    full_base_vectors: np.ndarray,
    query_vectors: np.ndarray,
    cache_K: int,
    output_dir: Path,
):
    """
    Build and save cache query ground truth from SIFT's precomputed top-100
    groundtruth (sift_groundtruth.ivecs), which is computed with EUCLIDEAN
    distance against the true full 1M base.

    Args:
        cache_query_indices: indices into the full 10K query set for cache queries
        groundtruth: precomputed top-100 neighbor indices, shape (10000, 100)
        full_base_vectors: the full 1M SIFT base vectors (not a subsample)
        query_vectors: the full 10K SIFT query vectors
        cache_K: number of neighbors to use (must be < 100)
        output_dir: benchmark directory to save cache_gt_K{K}.npz
    """
    assert cache_K < 100, f"cache_K must be < 100 to derive gap from precomputed top-100 groundtruth (got {cache_K}); use cache_K=99 instead of 100"

    gt_file = output_dir / f"cache_gt_K{cache_K}.npz"
    if gt_file.exists():
        print(f"  Cache GT (full-base) already exists: {gt_file.name}")
        return

    print(f"  Building full-base cache GT for {len(cache_query_indices)} cache queries (K={cache_K})...")

    save_dict = {}
    for local_i, global_i in enumerate(cache_query_indices):
        # K+1 entries from precomputed top-100 to compute true gap D(Q,e_{K+1}) - D(Q,e_K)
        ids = groundtruth[global_i, :cache_K]                 # (K,) base indices
        ids_kp1 = groundtruth[global_i, :cache_K + 1]         # (K+1,) base indices
        query = query_vectors[global_i]

        neighbor_vectors_kp1 = full_base_vectors[ids_kp1]     # (K+1, D)
        distances_kp1 = np.linalg.norm(neighbor_vectors_kp1 - query, axis=1).astype(np.float32)
        distances = distances_kp1[:cache_K]                   # (K,) Euclidean distances

        gap = float(distances_kp1[cache_K] - distances_kp1[cache_K - 1])
        vectors = full_base_vectors[ids]                      # (K, D)

        save_dict[f'indices_{local_i}'] = ids
        save_dict[f'vectors_{local_i}'] = vectors
        save_dict[f'distances_{local_i}'] = distances
        save_dict[f'gap_{local_i}'] = np.float32(gap)

    np.savez_compressed(gt_file, **save_dict)
    print(f"  Saved full-base cache GT: {gt_file.name}")

#-------------------------------------------------------------------------------

def prepare_sift_benchmark(
    num_base_vectors: int = 1_000_000,
    num_cache_queries: int = 9000,
    num_test_queries: int = 1000,
    cache_K: int = 100,
    test_N: int = 10,
    seed: int = 42,
    force: bool = False,
    num_workers: int = None,
    test_start: int = None,
    full_base: bool = False,
):
    """
    Prepare SIFT benchmark with specified parameters.

    Computes and saves cache query ground truth (angular metric) to the benchmark
    directory so run_sift.py can load it directly without any brute-force computation.

    Args:
        num_base_vectors: number of base vectors (max 1M from SIFT1M)
        num_cache_queries: number of cache queries
        num_test_queries: number of test queries
        cache_K: K for cache queries (top-K)
        test_N: N for test queries (top-N)
        seed: random seed for reproducibility
        force: overwrite existing benchmark if True
        num_workers: parallel workers for ground truth computation
        full_base: if True, also build a lean cache_gt_K{K}.npz from SIFT's
                    precomputed groundtruth (Euclidean, validated against the
                    true full 1M base), mirroring ESCI's optional full-catalog
                    GT capability. Off by default -- research benchmarks use
                    the 50K-sample population GT only.
    """

    # load full SIFT dataset
    print("Loading SIFT dataset...")
    base, _, query, groundtruth = load_sift_dataset("datasets/sift")

    # validate parameters
    if num_base_vectors > len(base):
        print(f"Warning: requested {num_base_vectors:,} base vectors but SIFT only has {len(base):,}")
        print(f"Using all {len(base):,} vectors")
        num_base_vectors = len(base)

    if num_cache_queries + num_test_queries > len(query):
        raise ValueError(
            f"Not enough queries: requested {num_cache_queries + num_test_queries}, "
            f"available {len(query)}"
        )

    # create benchmark name
    benchmark_name = create_benchmark_name(
        num_base_vectors, num_cache_queries, num_test_queries, cache_K, test_N, seed
    )

    output_dir = Path("datasets/sift") / benchmark_name

    # check if benchmark split already exists
    gt_file = output_dir / f"cache_gt_population_K{cache_K}.npz"
    if output_dir.exists() and gt_file.exists() and not force:
        print(f"\n{'='*80}")
        print(f"Benchmark already exists: {output_dir}")
        print(f"{'='*80}")
        return str(output_dir)

    print(f"\n{'='*80}")
    print("Preparing SIFT benchmark with the following configuration:")
    print(f"{'='*80}")
    print(f"\nConfiguration:")
    print(f"  Base vectors:     {num_base_vectors:>10,}")
    print(f"  Cache queries:    {num_cache_queries:>10,} (K={cache_K})")
    print(f"  Test queries:     {num_test_queries:>10,} (N={test_N})")
    print(f"  Dimension:        {base.shape[1]:>10,}")
    print(f"  Seed:             {seed:>10}")
    print(f"\nOutput: {output_dir}")

    # subsample base vectors if needed
    if num_base_vectors < len(base):
        print(f"\nSubsampling base vectors: {num_base_vectors:,} / {len(base):,}")
        base_subset = base[:num_base_vectors]
    else:
        base_subset = base

    # create config
    config = {
        "dataset": "SIFT1M",
        "benchmark_name": benchmark_name,
        "num_base_vectors": num_base_vectors,
        "num_cache_queries": num_cache_queries,
        "num_test_queries": num_test_queries,
        "cache_K": cache_K,
        "test_N": test_N,
        "dimension": int(base.shape[1]),
        "seed": seed,
        "description": (
            f"SIFT benchmark: {num_base_vectors:,} base, "
            f"{num_cache_queries:,} cache (K={cache_K}), "
            f"{num_test_queries:,} test (N={test_N})"
        )
    }

    # create benchmark split
    print("\nCreating benchmark split...")
    if test_start is not None:
        print(f"  test_start={test_start} (fixed test queries across cache sizes)")
    base_out, cache_q, test_q = create_sift_benchmark_split(
        base_subset, query,
        num_cache_queries=num_cache_queries,
        num_test_queries=num_test_queries,
        cache_K=cache_K,
        test_N=test_N,
        seed=seed,
        test_start=test_start,
    )

    # save benchmark vectors and config
    if not output_dir.exists() or force:
        print(f"\nSaving benchmark to {output_dir}...")
        save_sift_benchmark(str(output_dir), base_out, cache_q, test_q, config)

    # compute and save cache query ground truth (angular, needed by run_sift.py)
    print("\n[GT] Computing cache query ground truth (angular metric)...")
    precompute_cache_ground_truth(
        base_vectors=base_out,
        cache_queries=cache_q,
        K=cache_K,
        output_dir=output_dir,
        num_workers=num_workers
    )

    if full_base:
        print("\n[GT] Building full-base cache ground truth (Euclidean metric)...")
        full_base_cache_K = min(cache_K, 99)
        cache_indices, _ = sift_query_split_indices(
            len(query), num_cache_queries, num_test_queries, seed, test_start,
        )
        build_cache_gt_full(
            cache_query_indices=cache_indices,
            groundtruth=groundtruth,
            full_base_vectors=base,
            query_vectors=query,
            cache_K=full_base_cache_K,
            output_dir=output_dir,
        )

    print(f"\n{'='*80}")
    print("Benchmark prepared.")
    print(f"{'='*80}")
    print(f"Location: {output_dir}")

    return str(output_dir)

#-------------------------------------------------------------------------------

def extract_cache_r_Q(benchmark_dir: Path, K: int, num_cache_queries: int) -> np.ndarray:
    """Extract r_Q values from the population GT file."""
    data = np.load(benchmark_dir / f"cache_gt_population_K{K}.npz")
    r_Q = np.array(
        [float(data[f'r_Q_{i}']) for i in range(num_cache_queries)],
        dtype=np.float32,
    )
    return r_Q

#-------------------------------------------------------------------------------

def precompute_test_ground_truth(
    base_vectors: np.ndarray,
    test_queries: np.ndarray,
    K: int,
    output_dir: Path,
    num_workers: int = None,
    checkpoint_interval: int = 100,
) -> dict:
    """
    Precompute top-K ground truth for test queries using angular metric.
    Mirrors precompute_cache_ground_truth; writes test_gt_population_K{K}.npz.
    """
    gt_file = output_dir / f"test_gt_population_K{K}.npz"
    ground_truth, completed = _check_partial_gt(gt_file, len(test_queries))

    if len(completed) == len(test_queries):
        print(f"Test ground truth already complete ({len(test_queries)} queries).")
        return ground_truth

    remaining = len(test_queries) - len(completed)
    print(f"\nPrecomputing test ground truth: {remaining}/{len(test_queries)} remaining...")
    print(f"  Metric: angular, K: {K}")

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    print(f"  Using {num_workers} parallel workers")
    print(f"  Checkpointing every {checkpoint_interval} queries")

    args_list = [
        (i, test_queries[i], base_vectors, K, "angular")
        for i in range(len(test_queries))
        if i not in completed
    ]

    with Pool(num_workers) as pool:
        for chunk_start in range(0, len(args_list), checkpoint_interval):
            chunk_end = min(chunk_start + checkpoint_interval, len(args_list))
            chunk = args_list[chunk_start:chunk_end]

            print(f"\n  Processing chunk {chunk_start // checkpoint_interval + 1}/"
                  f"{(len(args_list) + checkpoint_interval - 1) // checkpoint_interval}...")

            results = list(tqdm(
                pool.imap(_compute_single_query_gt, chunk),
                total=len(chunk),
                desc="  Computing test GT",
                file=sys.stderr,
            ))

            for query_id, indices, distances, gap in results:
                ground_truth[query_id] = (indices, float(distances[-1]), float(gap) / 2.0)

            print(f"  Saving checkpoint... ({len(ground_truth)}/{len(test_queries)} complete)")
            save_dict = {}
            for i, (indices, r_Q, half_gap) in ground_truth.items():
                save_dict[f'indices_{i}'] = indices
                save_dict[f'r_Q_{i}'] = np.float32(r_Q)
                save_dict[f'half_gap_{i}'] = np.float32(half_gap)
            np.savez_compressed(gt_file, **save_dict)

    print(f"\nTest ground truth complete: {gt_file}")
    return ground_truth

#-------------------------------------------------------------------------------

def extract_test_r_t(benchmark_dir: Path, K: int, num_test_queries: int) -> np.ndarray:
    """Extract r_t values from the population test GT file."""
    data = np.load(benchmark_dir / f"test_gt_population_K{K}.npz")
    r_t = np.array(
        [float(data[f'r_Q_{i}']) for i in range(num_test_queries)],
        dtype=np.float32,
    )
    return r_t

#-------------------------------------------------------------------------------

def extend_benchmark_with_population_data(
    benchmark_dir: Path,
    num_workers: int = None,
) -> None:
    """
    Extend an existing benchmark with population-pipeline data files:
      cache_r_Q.npy, test_gt_angular_K{K}.npz, test_r_t.npy

    All files are guarded by existence checks; safe to re-run on a partially
    extended benchmark. Patches config.json with has_population_data=true after
    all three files are present.
    """
    config_file = benchmark_dir / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"No config.json in {benchmark_dir}")

    with open(config_file) as f:
        config = json.load(f)

    K = config["cache_K"]
    num_cache = config["num_cache_queries"]
    num_test = config["num_test_queries"]

    cache_gt_file = benchmark_dir / f"cache_gt_population_K{K}.npz"
    if not cache_gt_file.exists():
        raise FileNotFoundError(
            f"Cache ground truth not found: {cache_gt_file}\n"
            f"Run prepare_benchmark.py --all first."
        )

    print(f"\n{'='*80}")
    print(f"Extending benchmark with population data: {benchmark_dir.name}")
    print(f"{'='*80}")

    base, cache_q, test_q, _ = load_sift_benchmark(str(benchmark_dir))

    # cache_r_Q.npy
    r_Q_file = benchmark_dir / "cache_r_Q.npy"
    if r_Q_file.exists():
        print("cache_r_Q.npy already exists, skipping.")
    else:
        print("Extracting cache_r_Q.npy...")
        r_Q = extract_cache_r_Q(benchmark_dir, K, num_cache)
        np.save(r_Q_file, r_Q)
        print(f"  Saved: shape={r_Q.shape}, range=[{r_Q.min():.4f}, {r_Q.max():.4f}]")

    # test_gt_population_K{K}.npz, if a partial (interrupted) file exists,
    # finish it via the single-K resumable path; otherwise it's covered by
    # the multi-K pass below which includes K in its K_list.
    test_gt_file = benchmark_dir / f"test_gt_population_K{K}.npz"
    if test_gt_file.exists():
        _, completed = _check_partial_gt(test_gt_file, num_test)
        if len(completed) < num_test:
            precompute_test_ground_truth(base, test_q, K, benchmark_dir, num_workers)

    # per-K cache GT files for K-microbenchmark (K=[1,5,10,20])
    precompute_multi_k_ground_truth(base, cache_q, K_VALUES, benchmark_dir, "cache", num_workers)

    # per-K test GT files for K-override runs, plus the main cache_K test
    # file (already handled above if partially complete)
    test_k_list = sorted(set(K_VALUES) | {K})
    precompute_multi_k_ground_truth(base, test_q, test_k_list, benchmark_dir, "test", num_workers)

    # test_r_t.npy
    r_t_file = benchmark_dir / "test_r_t.npy"
    if r_t_file.exists():
        print("test_r_t.npy already exists, skipping.")
    else:
        print("Extracting test_r_t.npy...")
        r_t = extract_test_r_t(benchmark_dir, K, num_test)
        np.save(r_t_file, r_t)
        print(f"  Saved: shape={r_t.shape}, range=[{r_t.min():.4f}, {r_t.max():.4f}]")

    # patch config.json
    if not config.get("has_population_data"):
        config["has_population_data"] = True
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        print("Patched config.json with has_population_data=true.")

    print(f"\nDone: {benchmark_dir.name}")

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare SIFT benchmark with configurable parameters",
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
        choices=['tiny'],
        help='Use preset configuration'
    )

    parser.add_argument(
        '--extend-population',
        action='store_true',
        help='Extend all existing benchmarks with population data (cache_r_Q, test_gt, test_r_t)'
    )

    parser.add_argument('--base', type=int, help='Number of base vectors')
    parser.add_argument('--cache', type=int, help='Number of cache queries')
    parser.add_argument('--test', type=int, help='Number of test queries')
    parser.add_argument('--cache-k', type=int, help='K for cache queries')
    parser.add_argument('--test-n', type=int, default=20, help='N for test queries')
    parser.add_argument('--test-start', type=int, default=None,
                        help='Fixed start index for test queries (Set 2 cache scaling)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--force', action='store_true', help='Overwrite existing benchmark')
    parser.add_argument('--workers', type=int, default=None, help='Parallel workers for GT computation')
    parser.add_argument('--full-base', action='store_true',
                        help='Also build cache_gt_K{K}.npz from precomputed groundtruth '
                             '(Euclidean, validated against the true full 1M base)')

    args = parser.parse_args()

    presets = {
        'tiny': {
            'num_base_vectors': 10_000,
            'num_cache_queries': 100,
            'cache_K': 20,
            'num_test_queries': 50,
            'test_N': 10,
            'description': 'Tiny test (10K base, 100 cache, 50 test)'
        }
    }

    if args.all:
        print(f"\n{'='*80}")
        print(f"Preparing all {len(RESEARCH_BENCHMARKS)} SIFT research benchmarks")
        print(f"{'='*80}\n")
        failed = []
        for i, bench_params in enumerate(RESEARCH_BENCHMARKS):
            print(f"\n[{i+1}/{len(RESEARCH_BENCHMARKS)}] {bench_params}")
            try:
                p = dict(bench_params)
                p['force'] = args.force
                p['num_workers'] = args.workers
                prepare_sift_benchmark(**p)
            except Exception as e:
                print(f"ERROR: {e}")
                failed.append(bench_params)
        if failed:
            print(f"\nFailed: {len(failed)} benchmarks")
            sys.exit(1)
    elif args.preset:
        params = presets[args.preset].copy()
        print(f"Using preset '{args.preset}': {params.pop('description')}")
        params['seed'] = args.seed
        params['force'] = args.force
        params['num_workers'] = args.workers
        params['full_base'] = args.full_base
        prepare_sift_benchmark(**params)
    elif not args.extend_population:
        params = {
            'num_base_vectors': args.base or 50_000,
            'num_cache_queries': args.cache or 1024,
            'num_test_queries': args.test or 512,
            'cache_K': args.cache_k or 100,
            'test_N': args.test_n,
            'seed': args.seed,
            'force': args.force,
            'num_workers': args.workers,
            'test_start': args.test_start,
            'full_base': args.full_base,
        }
        prepare_sift_benchmark(**params)

    if args.extend_population:
        print(f"\n{'='*80}")
        print(f"Extending all {len(RESEARCH_BENCHMARKS)} benchmarks with population data")
        print(f"{'='*80}\n")
        failed = []
        for bench_params in RESEARCH_BENCHMARKS:
            benchmark_name = create_benchmark_name(
                bench_params['num_base_vectors'],
                bench_params['num_cache_queries'],
                bench_params['num_test_queries'],
                bench_params['cache_K'],
                bench_params['test_N'],
                bench_params['seed'],
            )
            benchmark_dir = Path("datasets/sift") / benchmark_name
            if not benchmark_dir.exists():
                print(f"Skipping {benchmark_name}: not prepared yet")
                continue
            try:
                extend_benchmark_with_population_data(benchmark_dir, args.workers)
            except Exception as e:
                print(f"ERROR extending {benchmark_name}: {e}")
                failed.append(benchmark_name)
        if failed:
            print(f"\nFailed: {len(failed)} benchmarks")
            sys.exit(1)

#-------------------------------------------------------------------------------
