#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import json
import csv
import numpy as np
import logging
from tqdm import tqdm

from datasets.dataloaders import load_sift_benchmark
from utils.base.distance_metrics import angular_distance
from simulations.simulate import CacheSimulator, SimulationResult
from utils.base import KVCache
from utils.base.population import propose_candidates
from utils.base.selection import select_candidates

#-------------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

#-------------------------------------------------------------------------------

ALPHA_VALUES = [1.05, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5]
BUDGET_VALUES = [250, 500, 1000, 2500, 5000, 10000, 20000, 30000, 40000, 50000]
STRATEGIES = ['random', 'greedy_naive', 'greedy_cost_static', 'greedy_cost']
ALGORITHMS = ['lemma1', 'lemma2', 'combined']
METRIC = 'angular'
SEED = 42
BENCHMARK_BASE_DIR = Path('datasets/sift')
OUTPUT_DIR = Path('simulations/sift/raw/population')

RESEARCH_BENCHMARKS = [
    "sift_b50k_c1024k100_t512n20_s42",
    "sift_b50k_c1024k100_t512n20_s43",
    "sift_b50k_c1024k100_t512n20_s44",
    "sift_b50k_c256k100_t512n20_s42",
    "sift_b50k_c256k100_t512n20_s43",
    "sift_b50k_c256k100_t512n20_s44",
    "sift_b50k_c512k100_t512n20_s42",
    "sift_b50k_c512k100_t512n20_s43",
    "sift_b50k_c512k100_t512n20_s44",
    "sift_b50k_c2048k100_t512n20_s42",
    "sift_b50k_c2048k100_t512n20_s43",
    "sift_b50k_c2048k100_t512n20_s44",
]

QUICK_BENCHMARKS = [
    "sift_b50k_c256k100_t512n20_s42",
    "sift_b50k_c512k100_t512n20_s42",
    "sift_b50k_c1024k100_t512n20_s42",
    "sift_b50k_c2048k100_t512n20_s42",
]

#-------------------------------------------------------------------------------

def _load_cache_gt_arrays(
    benchmark_dir: Path,
    K: int,
    n_cache: int,
    base_vectors: np.ndarray,
) -> tuple:
    """Load cache GT as parallel lists of top_k_vectors, top_k_indices, half_gaps."""
    data = np.load(benchmark_dir / f"cache_gt_population_K{K}.npz")
    tv_all, ti_all, hg_all = [], [], []
    for i in range(n_cache):
        indices = data[f'indices_{i}'].tolist()
        tv_all.append([base_vectors[idx] for idx in indices])
        ti_all.append(indices)
        hg_all.append(float(data[f'half_gap_{i}']))
    return tv_all, ti_all, np.array(hg_all, dtype=np.float32)

#-------------------------------------------------------------------------------

def _run_simulation(
    kv_cache,
    base_vectors: np.ndarray,
    test_queries: np.ndarray,
    K: int,
    algorithms: list,
    benchmark_name: str,
    strategy: str,
    budget_B: int,
    eval_mode: str,
) -> list:
    """Run lemma simulation for all algorithms on test_queries. Returns SimulationResult list."""
    n_test = len(test_queries)
    simulator = CacheSimulator(
        base_vectors=base_vectors,
        cache_queries=test_queries[:1],  # placeholder; populate_cache not called
        test_queries=test_queries,
        K=K,
        N=K,
        metric=METRIC,
    )
    simulator.cache = kv_cache

    all_results = []
    for algorithm in algorithms:
        cache_hits = 0
        total_distance_calcs = 0
        total_time_us = 0.0
        lemma1_hits = 0
        lemma2_hits = 0
        query_results = []

        for i, query in enumerate(tqdm(
            test_queries, desc=f"  [{eval_mode}] {algorithm:>20}", file=sys.stderr
        )):
            result = simulator.run_query(i, query, algorithm, ground_truth_indices=[])
            query_results.append(result)

            if result.cache_hit:
                cache_hits += 1
                if result.hit_source == 'lemma1':
                    lemma1_hits += 1
                elif result.hit_source == 'lemma2':
                    lemma2_hits += 1
            if result.distance_calculations is not None:
                total_distance_calcs += result.distance_calculations
            if result.time_us is not None:
                total_time_us += result.time_us

        sim_result = SimulationResult(
            dataset_name=f'{benchmark_name}/B{budget_B}/{eval_mode}/{strategy}',
            algorithm=algorithm,
            metric=METRIC,
            total_queries=n_test,
            cache_hits=cache_hits,
            correct_results=0,
            total_distance_calcs=total_distance_calcs,
            total_time_us=total_time_us,
            lemma1_hits=lemma1_hits,
            lemma2_hits=lemma2_hits,
            query_results=query_results,
        )
        all_results.append(sim_result)

        hit_rate = sim_result.hit_rate
        avg_dc = f"{sim_result.avg_distance_calcs:.1f}" if sim_result.avg_distance_calcs else "N/A"
        print(f"  [{eval_mode}] Hit rate: {hit_rate:.2%}  |  Avg dist calcs: {avg_dc}")

    return all_results

#-------------------------------------------------------------------------------

def _write_outputs(
    out_base: Path,
    benchmark_name: str,
    strategy: str,
    config: dict,
    all_results: list,
) -> None:
    """Write per_query.csv, per-algorithm JSON logs, run.log, and summary.json."""
    logs_dir = out_base / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)

    for sim_result in all_results:
        algo = sim_result.algorithm

        algo_dict = sim_result.to_dict()
        algo_dict.pop('query_results', None)
        with open(out_base / f"{algo}_{METRIC}.json", 'w') as f:
            json.dump(algo_dict, f, indent=2)

        log_file = logs_dir / f"{algo}_{METRIC}.log"
        with open(log_file, 'w') as lf:
            for qr in sim_result.query_results:
                if qr.cache_hit:
                    lf.write(
                        f"Query {qr.query_id}: HIT via {qr.hit_source} | "
                        f"dist_calcs={qr.distance_calculations}\n"
                    )
                else:
                    err = f" | error={qr.error}" if qr.error else ""
                    lf.write(
                        f"Query {qr.query_id}: MISS{err} | "
                        f"dist_calcs={qr.distance_calculations}\n"
                    )

    with open(out_base / 'per_query.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'algorithm', 'metric', 'query_id',
                         'cache_hit', 'hit_source', 'distance_calcs', 'time_us'])
        for sr in all_results:
            for qr in (sr.query_results or []):
                writer.writerow([
                    sr.dataset_name,
                    sr.algorithm,
                    sr.metric,
                    qr.query_id,
                    int(qr.cache_hit),
                    qr.hit_source or '',
                    qr.distance_calculations,
                    round(qr.time_us, 3) if qr.time_us is not None else None,
                ])

    summary = {
        "dataset": benchmark_name,
        "strategy": strategy,
        "config": config,
        "results": [r.to_dict() for r in all_results],
    }
    for r in summary["results"]:
        r.pop('query_results', None)
    with open(out_base / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    with open(logs_dir / 'run.log', 'w') as out_log:
        for algo in ALGORITHMS:
            log_file = logs_dir / f"{algo}_{METRIC}.log"
            if log_file.exists():
                out_log.write(f"--- {algo} / {METRIC} ---\n")
                out_log.write(log_file.read_text())
                out_log.write("\n")

#-------------------------------------------------------------------------------

def _write_selection_log(
    out_path: Path,
    selected: list,
    trajectory: list,
) -> None:
    """Write per-step candidate detail to a JSON file alongside the trajectory."""
    log = []
    cached_before: set = set()
    prev_queries = []
    for step, (cand, traj_entry) in enumerate(zip(selected, trajectory)):
        new_vecs = len(cand.R_c - cached_before)
        cached_before |= cand.R_c
        rc_size = cand.rc_size
        overlap_pct = round((rc_size - new_vecs) / max(rc_size, 1), 4)
        min_dist = round(float(min(
            angular_distance(cand.query, pq) for pq in prev_queries
        )), 6) if prev_queries else 0.0
        prev_queries.append(cand.query)
        log.append({
            "step": step + 1,
            "alpha": round(float(cand.alpha), 4),
            "r_Q": round(float(cand.r_Q), 6),
            "r_c": round(float(cand.r_c), 6),
            "vectors_in_R_c": rc_size,
            "new_vectors_added": new_vecs,
            "overlap_pct": overlap_pct,
            "min_dist_to_prev": min_dist,
            "queries_covered_total": len(cand.covered_train_indices),
            "new_queries_covered": traj_entry.get("new_train_coverage", 0),
            "cig_coverage": round(traj_entry["cig_train_coverage"], 6),
        })
    with open(out_path, 'w') as f:
        json.dump(log, f, indent=2)

#-------------------------------------------------------------------------------

def run_population_benchmark(
    benchmark_name: str,
    output_dir: Path = OUTPUT_DIR,
    benchmark_base_dir: Path = BENCHMARK_BASE_DIR,
    alpha_values: list = None,
    budget_values: list = None,
    strategies: list = None,
    algorithms: list = None,
    seed: int = SEED,
    run_ts: str = None,
    override_K: int = None,
) -> None:
    """Run the full population pipeline for a single SIFT benchmark.

    Proposes candidates once (using the maximum budget) from all available
    cache queries, then sweeps over all budget_values running every strategy.
    Both oracle (test = observed) and non-oracle (test = held-out t512) are
    evaluated for each (budget, strategy) combination.
    """
    if alpha_values is None:
        alpha_values = ALPHA_VALUES
    if budget_values is None:
        budget_values = BUDGET_VALUES
    if strategies is None:
        strategies = STRATEGIES
    if algorithms is None:
        algorithms = ALGORITHMS
    if run_ts is None:
        from datetime import datetime
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    benchmark_dir = benchmark_base_dir / benchmark_name

    if not (benchmark_dir / 'cache_r_Q.npy').exists():
        logger.warning(f"Skipping {benchmark_name}: run prepare_benchmark.py --extend-population first")
        return

    print(f"\n{'='*80}")
    print(f"POPULATION BENCHMARK: {benchmark_name}")
    print(f"{'='*80}")

    base, cache_q, test_q, config = load_sift_benchmark(str(benchmark_dir))
    config_K = config['cache_K']
    n_cache = len(cache_q)
    n_test = len(test_q)

    target_K = override_K if override_K is not None else config_K

    if target_K != config_K:
        gt_k = np.load(benchmark_dir / f'cache_gt_population_K{target_K}.npz')
        obs_r_Q = np.array(
            [float(gt_k[f'r_Q_{i}']) for i in range(n_cache)], dtype=np.float32
        )
        test_gt_k = np.load(benchmark_dir / f'test_gt_population_K{target_K}.npz')
        test_r_t = np.array(
            [float(test_gt_k[f'r_Q_{i}']) for i in range(n_test)], dtype=np.float32
        )
    else:
        obs_r_Q = np.load(benchmark_dir / 'cache_r_Q.npy')
        test_r_t = np.load(benchmark_dir / 'test_r_t.npy')

    K = target_K

    # use all available cache queries (no subsampling)
    obs_queries = cache_q
    obs_n = n_cache

    logger.info(f"Loading cache GT arrays ({n_cache} entries, K={K})...")
    obs_tv, obs_ti, obs_hg = _load_cache_gt_arrays(benchmark_dir, K, n_cache, base)

    logger.info(f"Proposing candidates: {obs_n} queries x {len(alpha_values)} alphas...")
    candidates = propose_candidates(
        obs_queries, obs_r_Q, obs_tv, obs_ti, obs_hg,
        obs_queries, obs_r_Q,
        test_q, test_r_t,
        base, alpha_values, METRIC,
    )
    logger.info(f"Proposed {len(candidates)} candidates")
    alpha_counts: dict = {}
    for c in candidates:
        alpha_counts[c.alpha] = alpha_counts.get(c.alpha, 0) + 1
    for a in alpha_values:
        logger.info(f"  alpha={a}: {alpha_counts.get(a, 0)} candidates")

    effective_name = (
        benchmark_name.replace(f'k{config_K}', f'k{target_K}')
        if target_K != config_K else benchmark_name
    )

    for budget_B in budget_values:
        print(f"\n{'='*60}")
        print(f"Budget B={budget_B}")
        print(f"{'='*60}")

        b_dir = output_dir / run_ts / effective_name / f'B{budget_B}'

        for strategy in strategies:
            print(f"\n{'-'*40}")
            print(f"Strategy: {strategy}")
            print(f"{'-'*40}")

            selected, trajectory = select_candidates(
                candidates, budget_B, strategy,
                n_train_queries=obs_n,
                n_test_queries=n_test,
                seed=seed,
            )
            logger.info(f"B={budget_B} {strategy}: {len(selected)} entries, "
                        f"trajectory length {len(trajectory)}")

            # trajectory (oracle-agnostic)
            traj_file = b_dir / f'{strategy}_trajectory.json'
            traj_file.parent.mkdir(parents=True, exist_ok=True)
            with open(traj_file, 'w') as f:
                json.dump(trajectory, f, indent=2)

            # per-step candidate detail
            sel_file = b_dir / f'{strategy}_selection.json'
            _write_selection_log(sel_file, selected, trajectory)

            # build KVCache
            kv_cache = KVCache(metric=METRIC)
            for cand in selected:
                entry = cand.to_cache_entry()
                kv_cache.add_entry(
                    entry.query, entry.top_k_vectors,
                    entry.r_Q, entry.half_gap, entry.top_k_indices,
                )
            assert kv_cache.size() == len(selected), "KVCache size mismatch"

            # oracle simulation: test on the observed queries
            oracle_results = _run_simulation(
                kv_cache, base, obs_queries, K, algorithms,
                effective_name, strategy, budget_B, 'oracle',
            )
            out_oracle = b_dir / 'oracle' / strategy
            _write_outputs(out_oracle, effective_name, strategy, config, oracle_results)

            # non-oracle simulation: test on held-out t512 queries
            non_oracle_results = _run_simulation(
                kv_cache, base, test_q, K, algorithms,
                effective_name, strategy, budget_B, 'non_oracle',
            )
            out_non_oracle = b_dir / 'non_oracle' / strategy
            _write_outputs(out_non_oracle, effective_name, strategy, config, non_oracle_results)

            logger.info(f"Results written to {b_dir}")

#-------------------------------------------------------------------------------

def main() -> None:
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(
        description="Run SIFT population pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--benchmark', type=str, default=None,
                        help='Single benchmark name (default: all)')
    parser.add_argument('--quick', action='store_true',
                        help='Run only QUICK_BENCHMARKS (4 benchmarks, seed=42)')
    parser.add_argument('--budgets', type=int, nargs='+', default=None,
                        help='Budget values to sweep (default: BUDGET_VALUES)')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--K', type=int, default=None,
                        help='Override K (requires cache_gt_population_K{K}.npz and '
                             'test_gt_population_K{K}.npz in each benchmark dir)')
    args = parser.parse_args()

    if args.quick:
        benchmarks = QUICK_BENCHMARKS
    elif args.benchmark:
        benchmarks = [args.benchmark]
    else:
        benchmarks = RESEARCH_BENCHMARKS

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for name in benchmarks:
        try:
            run_population_benchmark(
                name,
                budget_values=args.budgets,
                seed=args.seed,
                run_ts=run_ts,
                override_K=args.K,
            )
        except Exception as e:
            logger.error(f"Failed {name}: {e}")
            raise

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
