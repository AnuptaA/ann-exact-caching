#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""False-positive dilution cost of relaxed certification: how many extra
base vectors fall inside a cached candidate's ball as its radius is
inflated by the slack a relaxed CIG scheme would need to certify more
queries.

Standalone, read-only analysis script (same pattern as
debug_distance_ratio.py). For a representative budget fraction, proposes
and selects candidates once (reusing debug_common's fast helpers), then
for each selected candidate computes how many additional base vectors
fall within its ball as the radius scales by slack multipliers -- a
direct, concrete measure of what a relaxed-tolerance guarantee would
actually admit, complementing debug_vector_coverage.py's recovery-vs-
slack curves (which show how much hit rate relaxation buys) with the
cost side of that tradeoff.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent.parent))

from simulations.population import debug_common as dc
from simulations.population.debug_common import parse_benchmark_sizes
from utils.base.selection import select_candidates

#-------------------------------------------------------------------------------

K_VALUES = [1, 5, 10]
BUDGET_FRACTION = 0.65
SLACK_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
OUT_DIR = Path('simulations/debug/population_dilution_cost')

#-------------------------------------------------------------------------------


def compute_dilution(dataset: str, benchmark_name: str, K: int) -> np.ndarray:
    """Per-candidate median (over SLACK_LEVELS) extra-points-admitted
    ratio: (# base points within radius r_c*(1+slack)) / (# base points
    within r_c), for each selected candidate at BUDGET_FRACTION. Returns
    an array of shape (len(SLACK_LEVELS), n_selected) -- dilution ratio
    per candidate at each slack level."""
    base, cache_q, test_q, bdir = dc.load_benchmark(dataset, benchmark_name)
    n_cache, n_test = len(cache_q), len(test_q)
    train_r_t, test_r_t = dc.load_gt_r_t(bdir, K, n_cache, n_test)
    n_base = len(base)

    candidates = dc.propose_full_pool(base, cache_q, test_q, train_r_t, test_r_t, K)
    if not candidates:
        return None

    budget_B = max(1, int(BUDGET_FRACTION * n_base))
    selected, _traj = select_candidates(
        candidates, budget_B, strategy=dc.STRATEGY,
        n_train_queries=n_cache, n_test_queries=n_test, seed=dc.SEED,
    )
    if not selected:
        return None

    base_hat = base / np.linalg.norm(base, axis=1, keepdims=True)
    cand_centers = np.array([c.query for c in selected])
    cand_centers_hat = cand_centers / np.linalg.norm(cand_centers, axis=1, keepdims=True)
    cand_rc = np.array([c.r_c for c in selected])

    D = dc.pairwise_dist(cand_centers_hat, base_hat, dc.METRIC)

    base_count = (D <= cand_rc[:, None]).sum(axis=1)
    base_count = np.maximum(base_count, 1)

    dilution = np.zeros((len(SLACK_LEVELS), len(selected)), dtype=np.float64)
    for i, slack in enumerate(SLACK_LEVELS):
        scaled_count = (D <= (cand_rc[:, None] * (1.0 + slack))).sum(axis=1)
        dilution[i] = scaled_count / base_count
    return dilution


def plot_dataset_dilution(dataset: str, data_by_bm_k: dict, k_values: list) -> None:
    benchmarks = sorted(data_by_bm_k.keys(), key=lambda bm: parse_benchmark_sizes(bm)[0])
    n_rows, n_cols = len(benchmarks), len(k_values)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.8 * n_rows), squeeze=False)

    for r, bm in enumerate(benchmarks):
        train_n, _ = parse_benchmark_sizes(bm)
        for c, K in enumerate(k_values):
            ax = axes[r][c]
            dilution = data_by_bm_k[bm].get(K)
            if dilution is None:
                ax.axis('off')
                continue
            median = np.median(dilution, axis=1)
            q25 = np.percentile(dilution, 25, axis=1)
            q75 = np.percentile(dilution, 75, axis=1)
            ax.plot(SLACK_LEVELS, median, marker='o', markersize=3, color='tab:purple',
                     label='median extra-points-admitted ratio')
            ax.fill_between(SLACK_LEVELS, q25, q75, alpha=0.2, color='tab:purple',
                             label='IQR')
            ax.axhline(1.0, linestyle=':', color='gray', linewidth=0.8, alpha=0.5)

            if r == 0:
                ax.set_title(f'$K={K}$', fontsize=10)
            if c == 0:
                ax.set_ylabel(f'{train_n} Train Queries\nbase points admitted / original')
            if r == n_rows - 1:
                ax.set_xlabel('relaxation slack s')

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               fontsize=8, bbox_to_anchor=(0.5, -0.02 / n_rows))
    fig.suptitle(
        f'{dataset.upper()}: False-Positive Dilution vs. Relaxation Slack '
        f'(budget={BUDGET_FRACTION*100:.0f}%)', fontsize=13,
    )
    plt.tight_layout()

    outdir = OUT_DIR / dataset
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / 'dilution_vs_slack.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['sift', 'esci'], default=None,
                         help='Default: both')
    parser.add_argument('--k-values', type=int, nargs='+', default=None, dest='k_values')
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    k_values = args.k_values or K_VALUES
    datasets = [args.dataset] if args.dataset else list(dc.DATASET_CONFIG.keys())

    for dataset in datasets:
        print(f"\n=== {dataset.upper()} ===")
        cfg = dc.DATASET_CONFIG[dataset]
        data_by_bm_k = {}
        for benchmark_name in cfg['benchmarks']:
            bdir = cfg['base_dir'] / benchmark_name
            if not (bdir / 'cache_queries.npy').exists():
                print(f"  {benchmark_name}: not prepared, skipping")
                continue
            print(f"  {benchmark_name}")
            data_by_bm_k[benchmark_name] = {}
            for K in k_values:
                dilution = compute_dilution(dataset, benchmark_name, K)
                data_by_bm_k[benchmark_name][K] = dilution
                if dilution is not None:
                    median_at_1 = np.median(dilution[-1])
                    print(f"    K={K}: median dilution at slack={SLACK_LEVELS[-1]}: "
                          f"{median_at_1:.2f}x")
        plot_dataset_dilution(dataset, data_by_bm_k, k_values)


if __name__ == '__main__':
    main()
