#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Run ball-like population proposals for a benchmark dataset.
"""

import sys
import functools
import logging
import concurrent.futures
from datetime import datetime
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from datasets.dataloaders import load_esci_benchmark, load_sift_benchmark
from simulations.population.population_runner import run_population_benchmark as _run_benchmark
from utils.base.population import propose_candidates_ball

#-------------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

#-------------------------------------------------------------------------------

METRIC = 'angular'
ALGORITHMS = ['lemma1_no_union']
SEED = 42

BUDGET_FRACTIONS = [
    0.05, 0.10, 0.15, 0.30, 0.45, 0.50, 0.55, 0.60, 0.65, 0.75, 0.85, 0.90, 0.95, 0.98, 1.0,
]

DATASET_CONFIG = {
    'esci': dict(
        load_benchmark_fn=load_esci_benchmark,
        benchmark_base_dir=Path('datasets/esci'),
        output_dir=Path('simulations/esci/raw/population_proposal'),
        research_benchmarks=[
            'esci_c256k100_t512n20_s42',
            'esci_c512k100_t512n20_s42',
            'esci_c1024k100_t512n20_s42',
            'esci_c2048k100_t512n20_s42',
        ],
    ),
    'sift': dict(
        load_benchmark_fn=load_sift_benchmark,
        benchmark_base_dir=Path('datasets/sift'),
        output_dir=Path('simulations/sift/raw/population_proposal'),
        research_benchmarks=[
            'sift_b50k_c256k100_t512n20_s42',
            'sift_b50k_c512k100_t512n20_s42',
            'sift_b50k_c1024k100_t512n20_s42',
            'sift_b50k_c2048k100_t512n20_s42',
        ],
    ),
}

DEFAULT_N_MAX = 20
DEFAULT_K_TEST = [0, 1, 5, 10]
STRATEGIES = ['greedy_cost']

#-------------------------------------------------------------------------------

def _budget_values(benchmark_dir: Path) -> list:
    """Compute budget values as fractions of the base dataset size."""
    import numpy as np
    base = Path(benchmark_dir) / 'base_vectors.npy'
    n_base = np.load(base, mmap_mode='r').shape[0]
    return [max(1, int(f * n_base)) for f in BUDGET_FRACTIONS]


def _run_ball(
    benchmark_name: str,
    n_max: int,
    k_test: int,
    strategies: list,
    seed: int,
    run_ts: str,
    output_dir: Path,
    benchmark_base_dir: Path,
    load_benchmark_fn,
    sigma_values: list = None,
) -> None:
    K = k_test
    benchmark_dir = benchmark_base_dir / benchmark_name
    budgets = _budget_values(benchmark_dir)
    propose_fn = functools.partial(propose_candidates_ball, n_max=n_max, K=K)
    sub_dir = output_dir / run_ts / 'ball' / f'nmax{n_max}_k{k_test}'

    k_test_override = 0 if k_test == 0 else None
    override_K = k_test if k_test > 0 else None

    _run_benchmark(
        benchmark_name=benchmark_name,
        load_benchmark_fn=load_benchmark_fn,
        metric=METRIC,
        output_dir=sub_dir,
        benchmark_base_dir=benchmark_base_dir,
        propose_fn=propose_fn,
        budget_values=budgets,
        strategies=strategies,
        algorithms=ALGORITHMS,
        seed=seed,
        run_ts='.',
        override_K=override_K,
        k_test_override=k_test_override,
        sigma_values=sigma_values,
    )

#-------------------------------------------------------------------------------

def _dispatch(item, strategies, seed, run_ts, output_dir, benchmark_base_dir,
              load_benchmark_fn, sigma_values):
    bm, n_max, k_test = item
    logger.info(f"[ball] {bm} n_max={n_max} K={k_test}")
    _run_ball(bm, n_max, k_test, strategies, seed, run_ts, output_dir,
              benchmark_base_dir, load_benchmark_fn, sigma_values)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the proposal-based population pipeline for a dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--dataset', choices=['esci', 'sift'], required=True)
    parser.add_argument('--n-max', type=int, default=DEFAULT_N_MAX, dest='n_max',
                        help='Largest sphere index to propose per query '
                             '(all n in [1, n_max] are proposed, not swept)')
    parser.add_argument('--k-test', type=int, nargs='+', default=None,
                        dest='k_test',
                        help='K_test values (default: 0 1 5 10). Also sets '
                             'the candidate ball-margin K (K_cache = '
                             'K_test always, including K=0/containment '
                             'mode).')
    parser.add_argument('--benchmark', type=str, default=None,
                        help='Single benchmark name (default: all)')
    parser.add_argument('--strategies', type=str, nargs='+', default=None,
                        help='Selection strategies (default: greedy_cost)')
    parser.add_argument('--sigma', type=float, nargs='+', default=None,
                        help='Recall thresholds to sweep (default: [1.0], exact CIG). '
                             'Only meaningful with the lemma1 algorithm used here; '
                             'more than one value nests output under sigma_{val}/.')
    parser.add_argument('--workers', type=int, default=1,
                        help='Parallel worker processes (default: 1)')
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmarks = [args.benchmark] if args.benchmark else cfg['research_benchmarks']
    strategies = args.strategies if args.strategies else STRATEGIES
    sigma_values = args.sigma if args.sigma else [1.0]

    k_test_vals = args.k_test if args.k_test else DEFAULT_K_TEST
    work = [
        (bm, args.n_max, k_test)
        for bm in benchmarks
        for k_test in k_test_vals
    ]

    dispatch = functools.partial(
        _dispatch,
        strategies=strategies,
        seed=args.seed,
        run_ts=run_ts,
        output_dir=cfg['output_dir'],
        benchmark_base_dir=cfg['benchmark_base_dir'],
        load_benchmark_fn=cfg['load_benchmark_fn'],
        sigma_values=sigma_values,
    )

    if args.workers == 1:
        for item in work:
            dispatch(item)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(dispatch, item): item for item in work}
            for f in concurrent.futures.as_completed(futs):
                f.result()

#-------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
