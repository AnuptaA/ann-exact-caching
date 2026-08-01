#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Oracle union hit-rate ceiling for run_overlap_threshold_sweep.py's
sweep: for each (dataset, benchmark, K, budget_fraction) combo, the
fraction of test queries whose full top-K ground-truth set is entirely
contained in the UNION of all selected candidates' R_c sets -- the best
any hit policy could possibly achieve on that candidate set, regardless
of which single candidate it picks. Independent of percentile/threshold
(a property of the selected candidate set and budget alone).

Proposal happens ONCE per (dataset, benchmark, K) and is reused across
all budget_fractions -- only select_candidates repeats per budget --
same efficient structure as debug_vector_coverage.py's compute_combo
(that script computes the same vector-coverage ceiling alongside the
CIG-certified hit rate; this one is scoped to just the ceiling, for the
uncertified overlap_threshold_hit_policy's sweep instead). Re-proposing
per budget (an earlier draft of this script did that) is pure waste --
proposal is budget-independent.

Same union-coverage pattern verified by
tests/population/test_debug_intersection_calibration.py's
test_union_covered_at_least_as_permissive_as_per_candidate_oracle_hit
and debug_vector_coverage.py's compute_combo.
"""

import sys
import argparse
import csv
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from simulations.population import debug_common as dc
from simulations.hit_policy.run_overlap_threshold_sweep import K_VALUES, BUDGET_FRACTIONS
from utils.base.selection import select_candidates

#-------------------------------------------------------------------------------

CSV_COLUMNS = ['dataset', 'benchmark', 'K', 'budget_fraction', 'oracle_hit_rate']

#-------------------------------------------------------------------------------


def oracle_hit_rates_for_bm_k(
    dataset: str, benchmark_name: str, K: int, budget_fractions: list,
) -> dict:
    """Proposes once, then returns {budget_fraction: oracle_hit_rate}
    across all budget_fractions against that same candidate pool."""
    base, cache_q, test_q, bdir = dc.load_benchmark(dataset, benchmark_name)
    n_cache, n_test = len(cache_q), len(test_q)
    train_r_t, test_r_t = dc.load_gt_r_t(bdir, K, n_cache, n_test)
    n_base = len(base)

    candidates = dc.propose_full_pool(base, cache_q, test_q, train_r_t, test_r_t, K)
    test_gt_indices = dc.load_gt_indices(bdir, K, n_test)
    gt_matrix = np.array(test_gt_indices, dtype=np.int32)

    rates = {}
    for budget_fraction in budget_fractions:
        budget_B = max(1, int(budget_fraction * n_base))
        selected, _traj = select_candidates(
            candidates, budget_B, strategy=dc.STRATEGY,
            n_train_queries=n_cache, n_test_queries=n_test, seed=dc.SEED,
        )
        cached_mask = np.zeros(n_base, dtype=bool)
        for cand in selected:
            cached_mask |= cand.R_c
        covered = cached_mask[gt_matrix].all(axis=1)
        rates[budget_fraction] = float(covered.mean())
    return rates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=['sift', 'esci'], default=['sift', 'esci'])
    parser.add_argument('--k-values', type=int, nargs='+', default=K_VALUES, dest='k_values')
    parser.add_argument('--budget-fractions', type=float, nargs='+', default=BUDGET_FRACTIONS, dest='budget_fractions')
    parser.add_argument('--benchmarks', nargs='+', default=None)
    parser.add_argument('--output-dir', type=str, required=True,
                         help='Existing sweep run directory to write oracle_hit_rate.csv into')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    failed = []
    for dataset in args.datasets:
        benchmarks = args.benchmarks or dc.DATASET_CONFIG[dataset]['benchmarks']
        for benchmark_name in benchmarks:
            for K in args.k_values:
                combo = f"{dataset}/{benchmark_name}/K={K}"
                try:
                    rates = oracle_hit_rates_for_bm_k(dataset, benchmark_name, K, args.budget_fractions)
                    for budget_fraction, rate in rates.items():
                        rows.append({
                            'dataset': dataset, 'benchmark': benchmark_name, 'K': K,
                            'budget_fraction': budget_fraction, 'oracle_hit_rate': rate,
                        })
                    print(f"OK   {combo}: {len(rates)} budgets")
                except Exception as e:
                    print(f"FAIL {combo}: {e}")
                    failed.append(combo)

    csv_path = out_dir / 'oracle_hit_rate.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {csv_path}")
    print(f"Done: {len(failed)} failed combos")
    if failed:
        for combo in failed:
            print(f"  FAILED: {combo}")
        sys.exit(1)


if __name__ == '__main__':
    main()
