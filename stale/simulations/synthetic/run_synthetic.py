#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import csv
import numpy as np
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from datasets.synthetic.generator import load_synthetic_dataset
from simulations.core.simulate import CacheSimulator, SimulationResult
from utils.base import MainMemory, KVCache
from datetime import datetime

#-------------------------------------------------------------------------------

def _compute_single_query_gt(args):
    query_id, query, base_vectors, N, metric = args
    
    mm = MainMemory(vectors=list(base_vectors))
    _, top_n_dists, gap, top_n_indices = mm.top_k_search(
        query, k=N, metric=metric, return_indices=True
    )
    
    return query_id, np.array(top_n_indices), np.array(top_n_dists), gap

#-------------------------------------------------------------------------------

def precompute_ground_truth(
    base_vectors: np.ndarray,
    queries: np.ndarray,
    N: int,
    metric: str,
    cache_path: Path,
    query_type: str = "test",
    num_workers: int = None
) -> dict:
    """
    Precompute ground truth top-N results for queries in parallel.
    """
    cache_file = cache_path / f"ground_truth_{query_type}_{metric}_N{N}.npz"
    
    if cache_file.exists():
        try:
            print(f"Loading cached ground truth from {cache_file}")
            data = np.load(cache_file)
            ground_truth = {
                i: (data[f'indices_{i}'], data[f'distances_{i}'], float(data[f'gap_{i}']))
                for i in range(len(queries))
            }
            print(f"Loaded ground truth for {len(ground_truth)} queries")
            return ground_truth

        except (EOFError, ValueError, KeyError) as e:
            print(f"Warning: Corrupted cache file ({e}), recomputing...")
            cache_file.unlink()  # delete corrupted file
    
    print(f"Precomputing {query_type} ground truth for {len(queries)} queries...")
    print(f"  Metric: {metric}, N: {N}")
    
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    print(f"Using {num_workers} parallel workers")
    
    args_list = [
        (i, query, base_vectors, N, metric)
        for i, query in enumerate(queries)
    ]
    
    ground_truth = {}
    if num_workers == 1:
        results = [
            _compute_single_query_gt(args)
            for args in tqdm(args_list, total=len(queries), desc=f"Computing {query_type} GT")
        ]
    else:
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(_compute_single_query_gt, args_list),
                total=len(queries),
                desc=f"Computing {query_type} GT"
            ))
    
    for query_id, indices, distances, gap in results:
        ground_truth[query_id] = (indices, distances, gap)
    
    print(f"Saving ground truth to {cache_file}")
    cache_path.mkdir(parents=True, exist_ok=True)
    
    save_dict = {}
    for i, (indices, distances, gap) in ground_truth.items():
        save_dict[f'indices_{i}'] = indices
        save_dict[f'distances_{i}'] = distances
        save_dict[f'gap_{i}'] = gap
    
    # use atomic write: np.savez_compressed adds .npz automatically
    # write to temp.npz, then rename to cache_file
    temp_base = str(cache_file)[:-4]  # remove .npz extension
    np.savez_compressed(temp_base + '.tmp', **save_dict)

    # now temp_base + '.tmp.npz' exists, rename it
    Path(temp_base + '.tmp.npz').rename(cache_file)
    print("Ground truth cached")
    
    return ground_truth

#-------------------------------------------------------------------------------

def populate_cache_from_precomputed(
    simulator: CacheSimulator,
    cache_queries: np.ndarray,
    cache_ground_truth: dict,
    base_vectors: np.ndarray
):
    """Populate cache using precomputed ground truth results."""
    if simulator.cache is None:
        simulator.cache = KVCache(metric=simulator.metric)
    
    print(f"Populating cache from precomputed results...")
    
    for i, query in enumerate(tqdm(cache_queries, desc="Populating cache", file=sys.stderr)):
        top_k_indices, top_k_distances, gap = cache_ground_truth[i]
        top_k_vectors = [base_vectors[idx] for idx in top_k_indices]
        
        simulator.cache.add_entry(
            query,
            top_k_vectors,
            float(top_k_distances[-1]),
            gap / 2.0,
            [int(idx) for idx in top_k_indices]
        )
    
    print(f"Cache populated with {simulator.cache.size()} entries")

#-------------------------------------------------------------------------------

def run_synthetic_simulations(
    dataset_path: str,
    algorithms: list = None,
    metrics: list = None,
    output_dir: str = "simulations/synthetic/raw",
    use_cache: bool = True,
    num_workers: int = None
):
    """
    Run cache simulations on synthetic dataset.
    
    Args:
        dataset_path: Path to synthetic dataset
        algorithms: Algorithms to test
        metrics: Distance metrics to test
        output_dir: Where to save results
        use_cache: Whether to use cached ground truth
        num_workers: Number of parallel workers
    """
    if algorithms is None:
        # lemma1/lemma2/combined (union-based) are deprecated -- see
        # simulations/core/simulate.py.
        algorithms = ["lemma1_no_union", "lemma2_no_union", "brute"]
    
    if metrics is None:
        metrics = ["euclidean", "angular"]
    
    print("="*80)
    print(f"Simulating synthetic dataset: {Path(dataset_path).name}")
    print("="*80)
    
    # load dataset
    base, cache_q, test_q, config = load_synthetic_dataset(dataset_path)
    dataset_name = Path(dataset_path).name
    
    print(f"\nDataset configuration:")
    print(f"Base vectors: {len(base):,}")
    print(f"Cache queries: {len(cache_q):,} (K={config['K']})")
    print(f"Test queries: {len(test_q):,} (N={config['N']})")
    print(f"Dimension: {config['dimension']}")
    
    # set 4: variable N per query, GT precomputed at k=K, per-query N read from config
    is_variable_n = config['N'] == 0 and config.get('N_range') is not None
    test_n_values = config.get('test_N_values')  # list[int] or None
    gt_n = config['K'] if is_variable_n else config['N']
    
    if is_variable_n:
        if test_n_values is None:
            raise ValueError(f"Dataset {dataset_name} has N_range but missing test_N_values in config, REGEN DUMMY.")
        print(f"Variable-N dataset: GT precomputed at k={gt_n}, per-query N in [{config['N_range'][0]}, {config['N_range'][1]}]")
    
    output_path = Path(output_dir) / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    logs_dir = output_path / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    cache_path = Path(dataset_path) / "ground_truth_cache"
    
    all_results = []
    dataset_log_file = logs_dir / f"run.log"
    
    with open(dataset_log_file, 'w') as dataset_log:
        for metric in metrics:
            print(f"\n{'='*80}")
            print(f"Testing metric: {metric}")
            print(f"{'='*80}")
            
            # precompute ground truth for cache and test queries
            if use_cache:
                print("\n[1/2] Precomputing cache query results...")
                cache_ground_truth = precompute_ground_truth(
                    base, cache_q, config['K'], metric, cache_path,
                    query_type="cache", num_workers=num_workers
                )
                
                print("\n[2/2] Precomputing test query results...")
                test_ground_truth = precompute_ground_truth(
                    base, test_q, gt_n, metric, cache_path,
                    query_type="test", num_workers=num_workers
                )
            else:
                cache_ground_truth = None
                test_ground_truth = None
            
            for algorithm in algorithms:
                print(f"\n{'-'*80}")
                print(f"Running algorithm: {algorithm}")
                print(f"{'-'*80}")
                
                simulator = CacheSimulator(
                    base_vectors=base,
                    cache_queries=cache_q,
                    test_queries=test_q,
                    K=config['K'],
                    N=gt_n,
                    metric=metric
                )
                
                # populate cache from precomputed results
                if use_cache and cache_ground_truth is not None:
                    populate_cache_from_precomputed(
                        simulator, cache_q, cache_ground_truth, base
                    )
                else:
                    print("Populating cache (computing on-the-fly)...")
                    simulator.populate_cache()
                    print(f"Cache size: {simulator.cache.size()} entries")
                
                print(f"Running {len(test_q)} test queries...")
                
                query_results = []
                cache_hits = 0
                correct_results = 0
                total_distance_calcs = None
                total_time_us = None

                lemma1_hits = 0
                lemma2_hits = 0

                for i, query in enumerate(tqdm(test_q, desc=f"{algorithm:>10}", file=sys.stderr)):
                    ground_truth_indices = None
                    override_N = None
                    
                    if test_ground_truth is not None:
                        gt_indices, _, _ = test_ground_truth[i]
                        if is_variable_n:
                            # slice GT to per-query N and tell the algorithm to use that N
                            query_n = test_n_values[i]
                            ground_truth_indices = list(gt_indices[:query_n])
                            override_N = query_n
                        else:
                            ground_truth_indices = list(gt_indices)
                    
                    result = simulator.run_query(
                        i, query, algorithm,
                        ground_truth_indices=ground_truth_indices,
                        override_N=override_N
                    )
                    query_results.append(result)
                    
                    if result.cache_hit:
                        cache_hits += 1
                        if result.hit_source == "lemma1":
                            lemma1_hits += 1
                        elif result.hit_source == "lemma2":
                            lemma2_hits += 1
                    
                    if result.correct:
                        correct_results += 1
                    
                    if result.distance_calculations is not None:
                        total_distance_calcs = (total_distance_calcs or 0) + result.distance_calculations
                    if result.time_us is not None:
                        total_time_us = (total_time_us or 0.0) + result.time_us
                
                result = SimulationResult(
                    dataset_name=dataset_name,
                    algorithm=algorithm,
                    metric=metric,
                    total_queries=len(test_q),
                    cache_hits=cache_hits,
                    correct_results=correct_results,
                    total_distance_calcs=total_distance_calcs,
                    total_time_us=total_time_us,
                    lemma1_hits=lemma1_hits,
                    lemma2_hits=lemma2_hits,
                    query_results=query_results
                )
                
                print(f"\nResults:")
                print(f"Cache hits: {result.cache_hits}/{result.total_queries} ({result.hit_rate:.1%})")
                print(f"Accuracy: {result.accuracy:.1%}")
                dc = f"{result.avg_distance_calcs:.1f}" if result.avg_distance_calcs is not None else "N/A"
                t = f"{result.avg_time_us:.2f} us" if result.avg_time_us is not None else "N/A"
                print(f"Avg distance calcs: {dc}")
                print(f"Avg time: {t}")
                if algorithm == "combined":
                    print(f"Lemma 1 hits: {lemma1_hits}")
                    print(f"Lemma 2 hits: {lemma2_hits}")
                
                all_results.append(result)

                # save per-algorithm JSON
                algo_result_file = output_path / f"{algorithm}_{metric}.json"
                with open(algo_result_file, 'w') as f:
                    json.dump(result.to_dict(), f, indent=2)

                # write per-simulation log
                sim_log_file = logs_dir / f"{algorithm}_{metric}.log"
                with open(sim_log_file, 'w') as sim_log:
                    for qr in result.query_results:
                        if qr.cache_hit:
                            sim_log.write(f"Query {qr.query_id}: HIT via {qr.hit_source} | dist_calcs={qr.distance_calculations}\n")
                        else:
                            if qr.error:
                                sim_log.write(f"Query {qr.query_id}: MISS | error={qr.error} | dist_calcs={qr.distance_calculations}\n")
                            else:
                                sim_log.write(f"Query {qr.query_id}: MISS | dist_calcs={qr.distance_calculations}\n")

                # append to dataset level log
                dataset_log.write(f"--- {algorithm} / {metric} ---\n")
                with open(sim_log_file) as sim_log_r:
                    for line in sim_log_r:
                        dataset_log.write(line)
                dataset_log.write("\n")
    
    # write compact per-query CSV for analysis
    per_query_file = output_path / "per_query.csv"
    with open(per_query_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'algorithm', 'metric', 'query_id',
                         'cache_hit', 'hit_source', 'distance_calcs', 'time_us'])
        for sim_res in all_results:
            for qr in (sim_res.query_results or []):
                writer.writerow([
                    sim_res.dataset_name,
                    sim_res.algorithm,
                    sim_res.metric,
                    qr.query_id,
                    int(qr.cache_hit),
                    qr.hit_source or '',
                    qr.distance_calculations,
                    round(qr.time_us, 3) if qr.time_us is not None else 0
                ])

    # build consolidated summary JSON
    summary = {
        "dataset": dataset_name,
        "config": config,
        "results": [r.to_dict() for r in all_results],
        "validations": {}
    }
    
    for r in summary["results"]:
        r.pop('query_results', None)
    
    summary_file = output_path / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved summary to: {summary_file}")
    print(f"Logs saved to: {logs_dir}")
    print(f"{'='*80}\n")
    
    return all_results

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"simulations/synthetic/raw/{RUN_ID}"
    datasets_dir = Path("./datasets/synthetic/data")
    datasets = sorted(d for d in datasets_dir.iterdir() if d.is_dir())

    print(f"Run ID:  {RUN_ID}")
    print(f"Output:  {output_dir}")
    print(f"Found {len(datasets)} synthetic datasets")
    print()

    for dataset_path in datasets:
        run_synthetic_simulations(
            str(dataset_path),
            algorithms=["lemma1_no_union", "lemma2_no_union", "brute"],
            metrics=["euclidean", "angular"],
            output_dir=output_dir,
            use_cache=True,
            num_workers=None
        )

    print(f"\nAll done running synthetic simulations.")

#-------------------------------------------------------------------------------