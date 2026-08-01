#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
import json
import csv
import logging
import numpy as np
from tqdm import tqdm
from datasets.dataloaders import (
    load_sift_benchmark,
    load_esci_benchmark,
    load_cache_ground_truth_population,
    load_cache_ground_truth_full,
    populate_cache_from_precomputed_population,
    populate_cache_from_precomputed_full,
)
from simulations.core.simulate import CacheSimulator, SimulationResult

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

#-------------------------------------------------------------------------------

SIFT_RESEARCH_BENCHMARKS = [
    # Set 1: Baseline (1024 cache x 3 seeds; shared with Set 2)
    "sift_b50k_c1024k100_t512n20_s42",
    "sift_b50k_c1024k100_t512n20_s43",
    "sift_b50k_c1024k100_t512n20_s44",
    # Set 2: Cache size scaling (3 sizes x 3 seeds; 1024 shared with Set 1)
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

# All 42 ESCI research benchmarks (3 Set-1 + 12 Set-2 + 27 Set-3; 1024 shared)
ESCI_RESEARCH_BENCHMARKS = [
    # Set 1: Baseline (1024 cache x 3 seeds)
    "esci_c1024k99_t512n20_s42",
    "esci_c1024k99_t512n20_s43",
    "esci_c1024k99_t512n20_s44",
    # Set 2: Cache size scaling (5 sizes x 3 seeds; 1024 shared with Set 1)
    "esci_c256k99_t512n20_s42",
    "esci_c256k99_t512n20_s43",
    "esci_c256k99_t512n20_s44",
    "esci_c512k99_t512n20_s42",
    "esci_c512k99_t512n20_s43",
    "esci_c512k99_t512n20_s44",
    "esci_c2048k99_t512n20_s42",
    "esci_c2048k99_t512n20_s43",
    "esci_c2048k99_t512n20_s44",
    "esci_c4096k99_t512n20_s42",
    "esci_c4096k99_t512n20_s43",
    "esci_c4096k99_t512n20_s44",
    # Set 3: K/N relationship (K, N in {20, 50, 99} x 3 seeds)
    "esci_c1024k20_t256n20_s42",
    "esci_c1024k20_t256n20_s43",
    "esci_c1024k20_t256n20_s44",
    "esci_c1024k20_t256n50_s42",
    "esci_c1024k20_t256n50_s43",
    "esci_c1024k20_t256n50_s44",
    "esci_c1024k20_t256n99_s42",
    "esci_c1024k20_t256n99_s43",
    "esci_c1024k20_t256n99_s44",
    "esci_c1024k50_t256n20_s42",
    "esci_c1024k50_t256n20_s43",
    "esci_c1024k50_t256n20_s44",
    "esci_c1024k50_t256n50_s42",
    "esci_c1024k50_t256n50_s43",
    "esci_c1024k50_t256n50_s44",
    "esci_c1024k50_t256n99_s42",
    "esci_c1024k50_t256n99_s43",
    "esci_c1024k50_t256n99_s44",
    "esci_c1024k99_t256n20_s42",
    "esci_c1024k99_t256n20_s43",
    "esci_c1024k99_t256n20_s44",
    "esci_c1024k99_t256n50_s42",
    "esci_c1024k99_t256n50_s43",
    "esci_c1024k99_t256n50_s44",
    "esci_c1024k99_t256n99_s42",
    "esci_c1024k99_t256n99_s43",
    "esci_c1024k99_t256n99_s44",
]

#-------------------------------------------------------------------------------

def _sift_base_for_simulator(base: np.ndarray, dim: int) -> np.ndarray:
    """SIFT's population GT needs real base vectors for index lookup."""
    return base

def _esci_base_for_simulator(base: np.ndarray, dim: int) -> np.ndarray:
    """
    ESCI's full-schema GT embeds vectors directly, so CacheSimulator's
    base_vectors is never accessed (no brute force, no fallback) -- use a
    dummy array so a plain (non-population-extended) benchmark still works.
    """
    return np.zeros((1, dim), dtype=np.float32)

def _sift_base_description(config: dict) -> str:
    return f"{config['num_base_vectors']:,} vectors"

def _esci_base_description(config: dict) -> str:
    return "full 1.82M ESCI products"

DATASET_CONFIG = {
    'sift': dict(
        label='SIFT',
        load_benchmark_fn=load_sift_benchmark,
        load_gt_fn=load_cache_ground_truth_population,
        populate_fn=lambda simulator, cache_q, gt, base: (
            populate_cache_from_precomputed_population(simulator, cache_q, gt, base)
        ),
        base_for_simulator=_sift_base_for_simulator,
        base_description=_sift_base_description,
        benchmark_base_dir=Path('datasets/sift'),
        output_base=Path('simulations/sift/raw'),
        research_benchmarks=SIFT_RESEARCH_BENCHMARKS,
        prepare_hint='prepare_benchmark.py',
    ),
    'esci': dict(
        label='ESCI',
        load_benchmark_fn=load_esci_benchmark,
        load_gt_fn=load_cache_ground_truth_full,
        populate_fn=lambda simulator, cache_q, gt, base: (
            populate_cache_from_precomputed_full(simulator, cache_q, gt)
        ),
        base_for_simulator=_esci_base_for_simulator,
        base_description=_esci_base_description,
        benchmark_base_dir=Path('datasets/esci'),
        output_base=Path('simulations/esci/raw'),
        research_benchmarks=ESCI_RESEARCH_BENCHMARKS,
        prepare_hint='datasets/esci/prepare_benchmark.py',
    ),
}

#-------------------------------------------------------------------------------

def run_benchmark(
    dataset: str,
    benchmark_name: str,
    algorithms: list = None,
    output_dir: str = None,
    debug: bool = False,
    benchmark_base_dir: str = None,
):
    """
    Run cache simulations on a single benchmark (angular metric only).
    No brute force, no correctness validation -- only hit rate, distance calcs, timing.

    Args:
        dataset: 'sift' or 'esci'
        benchmark_name: benchmark directory name under the dataset's benchmark dir
        algorithms: algorithms to test (default: all lemma variants)
        output_dir: root output directory (default: DATASET_CONFIG's output_base)
        debug: limit to 5 test queries for quick verification
        benchmark_base_dir: override the dataset's default benchmark directory
    """
    cfg = DATASET_CONFIG[dataset]
    if algorithms is None:
        # lemma1/lemma2/combined (union-based) are deprecated -- see
        # simulations/core/simulate.py.
        algorithms = ["lemma1_no_union", "lemma2_no_union"]

    metric = "angular"
    benchmark_dir = Path(benchmark_base_dir or cfg['benchmark_base_dir']) / benchmark_name
    output_path = Path(output_dir or cfg['output_base']) / benchmark_name
    output_path.mkdir(parents=True, exist_ok=True)
    logs_dir = output_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"{cfg['label']} BENCHMARK SIMULATION: {benchmark_name}")
    print("=" * 80)

    base, cache_q, test_q, config = cfg['load_benchmark_fn'](str(benchmark_dir))

    print(f"\nBenchmark configuration:")
    print(f"  Base: {cfg['base_description'](config)}")
    print(f"  Cache queries: {config['num_cache_queries']:,} (K={config['cache_K']})")
    print(f"  Test queries: {config['num_test_queries']:,} (N={config['test_N']})")
    print(f"  Dimension: {config['dimension']}")
    print(f"  Metric: {metric}")

    # load precomputed cache ground truth (computed during prepare)
    cache_ground_truth = cfg['load_gt_fn'](benchmark_dir, config['cache_K'])

    all_results = []

    for algorithm in algorithms:
        print(f"\n{'-'*80}")
        print(f"Running algorithm: {algorithm}")
        print(f"{'-'*80}")

        # simulator's own base_vectors is never accessed (no brute force, no fallback)
        simulator = CacheSimulator(
            base_vectors=cfg['base_for_simulator'](base, config['dimension']),
            cache_queries=cache_q,
            test_queries=test_q,
            K=config['cache_K'],
            N=config['test_N'],
            metric=metric
        )

        cfg['populate_fn'](simulator, cache_q, cache_ground_truth, base)

        queries_to_run = test_q[:5] if debug else test_q
        if debug:
            logger.info(f"Debug mode: limiting to {len(queries_to_run)} queries")

        cache_hits = 0
        total_distance_calcs = 0
        total_time_us = 0.0
        lemma1_hits = 0
        lemma2_hits = 0
        query_results = []

        for i, query in enumerate(tqdm(queries_to_run, desc=f"{algorithm:>20}", file=sys.stderr)):
            result = simulator.run_query(i, query, algorithm, ground_truth_indices=[])
            query_results.append(result)

            if result.cache_hit:
                cache_hits += 1
                if result.hit_source == "lemma1":
                    lemma1_hits += 1
                elif result.hit_source == "lemma2":
                    lemma2_hits += 1

            if result.distance_calculations is not None:
                total_distance_calcs += result.distance_calculations
            if result.time_us is not None:
                total_time_us += result.time_us

        sim_result = SimulationResult(
            dataset_name=benchmark_name,
            algorithm=algorithm,
            metric=metric,
            total_queries=len(queries_to_run),
            cache_hits=cache_hits,
            correct_results=0,      # not tracked for SIFT/ESCI
            total_distance_calcs=total_distance_calcs,
            total_time_us=total_time_us,
            lemma1_hits=lemma1_hits,
            lemma2_hits=lemma2_hits,
            query_results=query_results
        )

        print(f"\n  Results:")
        print(f"    Cache hits: {sim_result.cache_hits}/{sim_result.total_queries} ({sim_result.hit_rate:.1%})")
        dc = f"{sim_result.avg_distance_calcs:.1f}" if sim_result.avg_distance_calcs is not None else "N/A"
        t = f"{sim_result.avg_time_us:.2f} us" if sim_result.avg_time_us is not None else "N/A"
        print(f"    Avg distance calcs: {dc}")
        print(f"    Avg time: {t}")
        if algorithm == "combined":
            print(f"    Lemma 1 hits: {lemma1_hits}")
            print(f"    Lemma 2 hits: {lemma2_hits}")

        all_results.append(sim_result)

        # per-algorithm JSON
        algo_file = output_path / f"{algorithm}_{metric}.json"
        with open(algo_file, 'w') as f:
            result_dict = sim_result.to_dict()
            result_dict.pop('query_results', None)
            json.dump(result_dict, f, indent=2)

        # per-algorithm query log
        sim_log_file = logs_dir / f"{algorithm}_{metric}.log"
        with open(sim_log_file, 'w') as sim_log:
            for qr in sim_result.query_results:
                if qr.cache_hit:
                    sim_log.write(
                        f"Query {qr.query_id}: HIT via {qr.hit_source} | "
                        f"dist_calcs={qr.distance_calculations}\n"
                    )
                else:
                    error_str = f" | error={qr.error}" if qr.error else ""
                    sim_log.write(
                        f"Query {qr.query_id}: MISS{error_str} | "
                        f"dist_calcs={qr.distance_calculations}\n"
                    )

    # consolidated output
    print(f"\n{'='*80}")
    print("Saving consolidated results...")

    per_query_file = output_path / "per_query.csv"
    with open(per_query_file, 'w', newline='') as f:
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
        "config": config,
        "results": [r.to_dict() for r in all_results],
    }
    for r in summary["results"]:
        r.pop('query_results', None)

    summary_file = output_path / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # combined run log
    dataset_log = logs_dir / "run.log"
    with open(dataset_log, 'w') as out_log:
        for algo in algorithms:
            sim_log_file = logs_dir / f"{algo}_{metric}.log"
            if sim_log_file.exists():
                out_log.write(f"--- {algo} / {metric} ---\n")
                with open(sim_log_file) as s:
                    out_log.writelines(s.readlines())
                out_log.write("\n")

    print(f"Saved summary to: {summary_file}")
    print(f"Logs saved to: {logs_dir}")
    print(f"{'='*80}")

    return all_results

#-------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run cache simulations (angular metric, lemma variants only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--dataset',
        type=str,
        choices=['sift', 'esci'],
        required=True,
        help='dataset to run'
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--benchmark',
        type=str,
        metavar='NAME',
        help='single benchmark directory name to run'
    )
    group.add_argument(
        '--all',
        action='store_true',
        default=False,
        help='run all research benchmarks'
    )

    parser.add_argument(
        '--algorithms',
        nargs='+',
        default=None,
        help='algorithms to test (default: all lemma variants)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='root output directory (default: simulations/<dataset>/raw/<timestamp>)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='enable debug logging and limit to 5 queries per benchmark'
    )
    parser.add_argument(
        '--benchmark-base-dir',
        type=str,
        default=None,
        help='directory containing prepared benchmark folders (default: datasets/<dataset>)'
    )
    parser.add_argument(
        '--benchmark-list',
        type=str,
        default=None,
        help='optional text file with one benchmark name per line'
    )

    args = parser.parse_args()
    cfg = DATASET_CONFIG[args.dataset]

    RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or str(cfg['output_base'] / RUN_ID)
    benchmark_base_dir = Path(args.benchmark_base_dir or cfg['benchmark_base_dir'])

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.info("Debug mode enabled -- will process only 5 queries per benchmark")

    # resolve which benchmarks to run
    if args.benchmark:
        benchmarks = [args.benchmark]
    elif args.benchmark_list:
        with open(args.benchmark_list) as f:
            benchmarks = [
                line.strip()
                for line in f
                if line.strip() and not line.lstrip().startswith("#")
            ]
    else:
        # --all or default (no flag) -> run all research benchmarks
        benchmarks = cfg['research_benchmarks']

    # validate benchmark directories exist
    missing = [b for b in benchmarks if not (benchmark_base_dir / b).exists()]
    if missing:
        print("Error: the following benchmark directories do not exist:")
        for b in missing:
            print(f"  {benchmark_base_dir / b}")
        print(f"\nRun {cfg['prepare_hint']} to create them.")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f"{cfg['label']} Simulation Suite")
    print(f"{'='*80}")
    print(f"Run ID:     {RUN_ID}")
    print(f"Output:     {output_dir}")
    print(f"Input:      {benchmark_base_dir}")
    print(f"Benchmarks: {len(benchmarks)}")
    print(f"Algorithms: {args.algorithms or 'all lemma variants'}")
    print(f"{'='*80}\n")

    failed = []
    for benchmark_name in benchmarks:
        try:
            run_benchmark(
                dataset=args.dataset,
                benchmark_name=benchmark_name,
                algorithms=args.algorithms,
                output_dir=output_dir,
                debug=args.debug,
                benchmark_base_dir=str(benchmark_base_dir),
            )
        except Exception as e:
            print(f"\nERROR: {benchmark_name} failed: {e}")
            failed.append(benchmark_name)

    print(f"\n{'='*80}")
    print(f"Done: {len(benchmarks) - len(failed)}/{len(benchmarks)} benchmarks succeeded")
    if failed:
        print("Failed:")
        for b in failed:
            print(f"  {b}")
    print(f"{'='*80}")

    sys.exit(1 if failed else 0)

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#-------------------------------------------------------------------------------
