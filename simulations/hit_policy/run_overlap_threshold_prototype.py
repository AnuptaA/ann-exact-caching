#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Evaluate the overlap_threshold_hit_policy prototype (see
utils/base/lemmas_sigma.py) on a real benchmark: real proposed/selected
candidates, real test queries, real hit_rate/mean_recall via
population_runner.py's existing simulation harness (_run_simulation) --
no new recall-measurement code, reusing what exists.

r_t_hat (the fixed workload-calibrated top-K radius estimate) is fit
from *cache/training* query radii only, never from test queries --
sweeps the median/p95/p99 percentiles side by side, since the
percentile choice trades off hit-rate against recall (higher
percentile -> larger assumed radius -> smaller intersection fractions
-> fewer, more conservative hits).
"""

import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from simulations.population import debug_common as dc
from simulations.population.population_runner import _run_simulation
from utils.base import KVCache
from utils.base.selection import select_candidates
from utils.base.lemmas_sigma import fit_r_t_hat

#-------------------------------------------------------------------------------

PERCENTILES = [50.0, 95.0, 99.0]


def _build_kv_cache(selected: list, metric: str) -> KVCache:
    kv_cache = KVCache(metric=metric)
    for cand in selected:
        entry = cand.to_cache_entry()
        kv_cache.add_entry(
            entry.query, entry.top_k_vectors,
            entry.r_Q, entry.half_gap, entry.top_k_indices,
            entry.top_k_matrix,
        )
    return kv_cache


def run_prototype(
    dataset: str, benchmark_name: str, K: int, budget_fraction: float,
    thresholds: list = None,
) -> list:
    thresholds = thresholds if thresholds is not None else [0.9]
    base, cache_q, test_q, bdir = dc.load_benchmark(dataset, benchmark_name)
    n_cache, n_test = len(cache_q), len(test_q)
    train_r_t, test_r_t = dc.load_gt_r_t(bdir, K, n_cache, n_test)
    n_base = len(base)

    candidates = dc.propose_full_pool(base, cache_q, test_q, train_r_t, test_r_t, K)
    budget_B = max(1, int(budget_fraction * n_base))
    selected, _traj = select_candidates(
        candidates, budget_B, strategy=dc.STRATEGY,
        n_train_queries=n_cache, n_test_queries=n_test, seed=dc.SEED,
    )
    print(f"{dataset} {benchmark_name} K={K} budget={budget_fraction*100:.0f}%: "
          f"{len(selected)} selected candidates")

    kv_cache = _build_kv_cache(selected, metric='angular')
    test_gt_indices = dc.load_gt_indices(bdir, K, n_test)

    rows = []
    for pct in PERCENTILES:
        r_t_hat = fit_r_t_hat(train_r_t, pct)
        for threshold in thresholds:
            results = _run_simulation(
                kv_cache, base, test_q, K, ['overlap_threshold'],
                benchmark_name, dc.STRATEGY, budget_B, 'non_oracle', 'angular',
                test_gt_indices=test_gt_indices,
                r_t_hat=r_t_hat, overlap_threshold=threshold,
            )
            sim_result = results[0]
            rows.append({
                'percentile': pct, 'r_t_hat': r_t_hat, 'threshold': threshold,
                'hit_rate': sim_result.hit_rate, 'mean_recall': sim_result.mean_recall,
            })
            print(f"  p{pct:.0f} (r_t_hat={r_t_hat:.4f}) threshold={threshold}: "
                  f"hit_rate={sim_result.hit_rate:.2%} mean_recall={sim_result.mean_recall:.2%}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['sift', 'esci'], default='sift')
    parser.add_argument('--benchmark', type=str, required=True)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--budget-fraction', type=float, default=0.5)
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.9])
    args = parser.parse_args()

    run_prototype(args.dataset, args.benchmark, args.k, args.budget_fraction, args.thresholds)


if __name__ == '__main__':
    main()
