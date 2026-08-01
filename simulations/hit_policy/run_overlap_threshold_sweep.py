#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Batch runner for the overlap_threshold_hit_policy prototype (see
utils/base/lemmas_sigma.py): sweeps dataset x benchmark x K x budget
x r_t_hat percentile across the full matrix and writes one CSV row per
(config, percentile, threshold).

Score computation (the expensive per-query, per-entry scan) happens
ONCE per (benchmark, K, budget, percentile) -- not once per threshold.
overlap_threshold_hit_policy is called per query with
threshold=min(THRESHOLDS) (the lowest threshold in the grid, not -1.0/
always-hit -- see _score_and_recall_per_query's docstring for why this
is exact, not an approximation), which returns metadata['best_score']
and the resulting top-N vectors for every query that clears even the
most lenient threshold we'll ever check. Recall against ground truth is
computed once per query from that result. The threshold grid is then a
free post-processing step: hit_rate(t) = mean(best_score > t),
mean_recall(t) = mean(recall for queries with best_score > t). This
avoids re-scanning every cache entry once per threshold value, which
would multiply the dominant cost of the sweep by the grid size for no
reason -- threshold only gates which queries count as hits, it doesn't
change which candidate is best or that candidate's would-be recall.
"""

import sys
import argparse
import csv
import json
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from simulations.population import debug_common as dc
from utils.base.selection import select_candidates
from utils.base.lemmas_sigma import fit_r_t_hat, overlap_threshold_hit_policy
from utils.base import KVCache

#-------------------------------------------------------------------------------

K_VALUES = [1, 5, 10, 20]
BUDGET_FRACTIONS = [0.05, 0.10, 0.15, 0.30, 0.45, 0.50, 0.55, 0.60, 0.65, 0.75, 0.85, 0.90, 0.95, 0.98]
PERCENTILES = [25.0, 50.0, 95.0, 99.0]
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

CSV_COLUMNS = [
    'dataset', 'benchmark', 'K', 'budget_fraction', 'n_selected',
    'percentile', 'r_t_hat', 'threshold', 'hit_rate', 'mean_recall',
]

#-------------------------------------------------------------------------------


def _score_and_recall_per_query(
    kv_cache: KVCache, base: np.ndarray, base_hat: np.ndarray, test_q: np.ndarray, K: int,
    r_t_hat: float, test_gt_indices: list,
) -> list:
    """Runs overlap_threshold_hit_policy once per test query and returns
    [(best_score, recall), ...], one pair per query -- the shared
    computation the threshold grid below is derived from without
    re-scanning.

    Calls the policy with threshold=min(THRESHOLDS) rather than -1.0
    (always-hit, an earlier version of this function). Profiling found
    the policy's "build the hit result" branch (_batch_distances +
    gathering/sorting the entry's candidate vectors) dominates its cost
    (~93% of one budget's runtime, worst case measured) -- and it's pure
    waste for any query whose best_score doesn't clear even the lowest
    threshold in THRESHOLDS, since _aggregate_at_thresholds below only
    ever reads a query's recall when best_score > threshold, and no
    threshold in the grid is lower than min(THRESHOLDS). Skipping that
    branch for those queries (score=miss, recall left at the 0.0
    certified_indices already produces) is mathematically exact -- it
    changes zero output values, only which queries pay for a result
    that would've been discarded regardless. Matters most for ESCI,
    where hit rates are low enough that most queries fall in this
    always-discarded bucket. base_hat (normalized base vectors) is
    passed in rather than computed here -- it doesn't depend on
    percentile, budget, or K, only on the dataset's base array, so the
    caller computes it once per (benchmark, K) instead of once per
    percentile call (4x per budget otherwise, ~22ms wasted each time on
    a 50K-row base)."""
    cache_entries = kv_cache.get_all_entries()
    r_c_array = kv_cache.get_entry_radii()
    min_threshold = min(THRESHOLDS)

    records = []
    for i, query in enumerate(test_q):
        d_Q = kv_cache.get_d_Q_batch(query)
        _, _, metadata = overlap_threshold_hit_policy(
            query, cache_entries, N=K, sigma=1.0, metric='angular',
            precomputed_d_Q=d_Q, precomputed_r_c=r_c_array,
            base_hat=base_hat, base_vectors=base,
            r_t_hat=r_t_hat, threshold=min_threshold,
        )
        true_gt = set(test_gt_indices[i])
        certified = set(metadata['certified_indices'])
        recall = len(certified & true_gt) / K
        # best_score is None when cache_entries is empty (e.g. a tiny
        # budget on a small benchmark selects zero candidates) -- treat
        # as a definite miss at every threshold, not a crash.
        best_score = metadata['best_score'] if metadata['best_score'] is not None else -1.0
        records.append((best_score, recall))
    return records


def _aggregate_at_thresholds(records: list, thresholds: list) -> list:
    scores = np.array([r[0] for r in records])
    recalls = np.array([r[1] for r in records])
    rows = []
    for threshold in thresholds:
        is_hit = scores > threshold
        hit_rate = float(np.mean(is_hit))
        mean_recall = float(np.mean(recalls[is_hit])) if is_hit.any() else 0.0
        rows.append({'threshold': threshold, 'hit_rate': hit_rate, 'mean_recall': mean_recall})
    return rows


def run_bm_k(dataset: str, benchmark_name: str, K: int, budget_fractions: list) -> tuple:
    """Runs every budget in budget_fractions for one (dataset, benchmark,
    K), across all PERCENTILES and THRESHOLDS. load_benchmark/
    propose_full_pool/load_gt_r_t/load_gt_indices are budget-independent
    and run ONCE here, not once per budget -- an earlier version of this
    function (run_combo, one call per budget) redid all of that on every
    one of the 14 swept budgets, ~14x more than necessary (the same
    inefficiency found and fixed in compute_oracle_hit_rate.py's
    oracle_hit_rates_for_bm_k -- this mirrors that fix). Only
    select_candidates + KVCache build + scoring are genuinely
    budget-dependent and still run per budget -- wrapped per-budget in
    try/except so one bad budget (e.g. a tiny budget selecting zero
    candidates) doesn't lose every other budget's already-computed rows
    for this (benchmark, K), matching the per-combo failure isolation
    the old one-call-per-budget design had for free.

    Returns (rows, failed_budgets) where failed_budgets is a list of
    (budget_fraction, error_message) pairs."""
    base, cache_q, test_q, bdir = dc.load_benchmark(dataset, benchmark_name)
    n_cache, n_test = len(cache_q), len(test_q)
    train_r_t, test_r_t = dc.load_gt_r_t(bdir, K, n_cache, n_test)
    n_base = len(base)
    base_hat = base / np.linalg.norm(base, axis=1, keepdims=True)

    candidates = dc.propose_full_pool(base, cache_q, test_q, train_r_t, test_r_t, K)
    test_gt_indices = dc.load_gt_indices(bdir, K, n_test)

    rows = []
    failed_budgets = []
    for budget_fraction in budget_fractions:
        try:
            budget_B = max(1, int(budget_fraction * n_base))
            selected, _traj = select_candidates(
                candidates, budget_B, strategy=dc.STRATEGY,
                n_train_queries=n_cache, n_test_queries=n_test, seed=dc.SEED,
            )

            kv_cache = KVCache(metric='angular')
            for cand in selected:
                entry = cand.to_cache_entry()
                kv_cache.add_entry(
                    entry.query, entry.top_k_vectors,
                    entry.r_Q, entry.half_gap, entry.top_k_indices,
                    entry.top_k_matrix,
                )

            for percentile in PERCENTILES:
                r_t_hat = fit_r_t_hat(train_r_t, percentile)
                records = _score_and_recall_per_query(kv_cache, base, base_hat, test_q, K, r_t_hat, test_gt_indices)
                for entry in _aggregate_at_thresholds(records, THRESHOLDS):
                    rows.append({
                        'dataset': dataset, 'benchmark': benchmark_name, 'K': K,
                        'budget_fraction': budget_fraction, 'n_selected': len(selected),
                        'percentile': percentile, 'r_t_hat': r_t_hat, **entry,
                    })
        except Exception as e:
            failed_budgets.append((budget_fraction, str(e)))
    return rows, failed_budgets


def run_combo(dataset: str, benchmark_name: str, K: int, budget_fraction: float) -> list:
    """Single-budget convenience wrapper around run_bm_k, kept for
    callers (tests, ad hoc single-combo checks) that only want one
    budget -- main()'s real sweep calls run_bm_k directly to get the
    propose-once benefit across all swept budgets. Raises if that one
    budget failed (matching the old run_combo's behavior -- no
    per-budget isolation needed when there's only one budget)."""
    rows, failed_budgets = run_bm_k(dataset, benchmark_name, K, [budget_fraction])
    if failed_budgets:
        raise RuntimeError(failed_budgets[0][1])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=['sift', 'esci'], default=['sift', 'esci'])
    parser.add_argument('--k-values', type=int, nargs='+', default=K_VALUES, dest='k_values')
    parser.add_argument('--budget-fractions', type=float, nargs='+', default=BUDGET_FRACTIONS, dest='budget_fractions')
    parser.add_argument('--benchmarks', nargs='+', default=None, help='override benchmark list per dataset (applies to all datasets given)')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path('simulations/hit_policy/reports/overlap_threshold') / RUN_ID
    out_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        'run_id': RUN_ID, 'datasets': args.datasets, 'k_values': args.k_values,
        'budget_fractions': args.budget_fractions, 'percentiles': PERCENTILES,
        'thresholds': THRESHOLDS,
    }
    with open(out_dir / 'run_config.json', 'w') as f:
        json.dump(run_config, f, indent=2)

    all_rows = []
    failed = []
    for dataset in args.datasets:
        benchmarks = args.benchmarks or dc.DATASET_CONFIG[dataset]['benchmarks']
        for benchmark_name in benchmarks:
            for K in args.k_values:
                cell = f"{dataset}/{benchmark_name}/K={K}"
                try:
                    rows, failed_budgets = run_bm_k(dataset, benchmark_name, K, args.budget_fractions)
                    all_rows.extend(rows)
                    print(f"OK   {cell}: {len(rows)} rows, {len(failed_budgets)} failed budgets")
                    for budget_fraction, err in failed_budgets:
                        print(f"FAIL {cell}/budget={budget_fraction}: {err}")
                        failed.append(f"{cell}/budget={budget_fraction}")
                except Exception as e:
                    print(f"FAIL {cell}: {e}")
                    failed.append(cell)

    csv_path = out_dir / 'results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(all_rows, f, indent=2)

    print(f"\nWrote {len(all_rows)} rows to {csv_path}")
    print(f"Done: {len(failed)} failed combos")
    if failed:
        for combo in failed:
            print(f"  FAILED: {combo}")
        sys.exit(1)


if __name__ == '__main__':
    main()
