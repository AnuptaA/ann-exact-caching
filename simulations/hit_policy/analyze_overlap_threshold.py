#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Plots for run_overlap_threshold_sweep.py's output CSV: descriptive
(one row per config already), not the pair-level inferential stats used
for the intersection-fraction calibration study (debug_intersection_calibration.py).

Grid layout mirrors debug_intersection_calibration.py's convention:
benchmark (cache size) as grid rows, K as grid columns, one figure per
dataset -- budget_fraction isn't even on the same absolute scale across
datasets (each has its own n_base), and different cache sizes admit
different candidate pools at the same budget_fraction, so pooling
across either would average together numbers that aren't comparable.
Only percentile stays pooled as colored lines within each subplot --
that pooling is the actual comparison being made.

hit_rate_vs_budget/recall_vs_budget are written once per swept
threshold value (same convention as e.g. the intersection-fraction
study's per-budget scatter plots) rather than picking one
"representative" threshold -- the sweep computed all of them."""

import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import re

import pandas as pd
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------

PERCENTILE_COLORS = {25.0: 'tab:red', 50.0: 'tab:blue', 95.0: 'tab:orange', 99.0: 'tab:green'}
ORACLE_COLOR = 'black'


def _short_label(benchmark_name: str) -> str:
    """Cache-size token, e.g. 'c256k100' -- benchmark names aren't
    positionally consistent across datasets (SIFT has an extra 'b50k'
    base-size token ESCI doesn't: sift_b50k_c256k100_... vs
    esci_c256k100_...), so split('_')[1] silently grabs the wrong token
    for SIFT. Match the cNNNkNNN pattern directly instead."""
    m = re.search(r'c\d+k\d+', benchmark_name)
    return m.group(0) if m else benchmark_name


def _cache_size(benchmark_name: str) -> int:
    """Numeric cache-query count for sorting rows in ascending cache-size
    order -- plain sorted() on benchmark names is lexicographic, so
    c1024/c2048 would sort before c256/c512."""
    m = re.search(r'c(\d+)k\d+', benchmark_name)
    return int(m.group(1)) if m else 0


def _sorted_benchmarks(benchmark_names) -> list:
    return sorted(set(benchmark_names), key=_cache_size)


def _grid(benchmarks: list, k_values: list, sharey: bool = True):
    n_rows, n_cols = len(benchmarks), len(k_values)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.6 * n_cols, 3.2 * n_rows),
        sharex=True, sharey=sharey, squeeze=False,
    )
    return fig, axes


def plot_hit_rate_vs_budget(df: pd.DataFrame, out_dir: Path, oracle_df: pd.DataFrame = None) -> None:
    for threshold in sorted(df['threshold'].unique()):
        sub = df[df['threshold'] == threshold]
        for dataset, dset_df in sub.groupby('dataset'):
            benchmarks = _sorted_benchmarks(dset_df['benchmark'].unique())
            k_values = sorted(dset_df['K'].unique())
            fig, axes = _grid(benchmarks, k_values)
            for r, bm in enumerate(benchmarks):
                for c, K in enumerate(k_values):
                    ax = axes[r][c]
                    cell = dset_df[(dset_df['benchmark'] == bm) & (dset_df['K'] == K)]
                    for pct, group in cell.groupby('percentile'):
                        agg = group.groupby('budget_fraction')['hit_rate'].mean().sort_index()
                        ax.plot(agg.index, agg.values, marker='o', markersize=3, label=f'p{int(pct)}',
                                color=PERCENTILE_COLORS.get(pct))
                    if oracle_df is not None:
                        ocell = oracle_df[
                            (oracle_df['dataset'] == dataset) & (oracle_df['benchmark'] == bm)
                            & (oracle_df['K'] == K)
                        ]
                        oagg = ocell.groupby('budget_fraction')['oracle_hit_rate'].mean().sort_index()
                        if not oagg.empty:
                            ax.plot(oagg.index, oagg.values, marker='x', markersize=4,
                                    linestyle='--', color=ORACLE_COLOR, label='union ceiling')
                    ax.set_ylim(0, 1)
                    if r == 0:
                        ax.set_title(f'K={K}')
                    if c == 0:
                        ax.set_ylabel(f'{_short_label(bm)}\nhit rate')
                    if r == len(benchmarks) - 1:
                        ax.set_xlabel('budget fraction')
                    ax.grid(alpha=0.3)
            axes[0][0].legend(fontsize=7)
            fig.suptitle(f'{dataset.upper()}: overlap-threshold hit rate vs. budget fraction (threshold={threshold:.2f})')
            fig.tight_layout()
            dset_dir = out_dir / dataset
            dset_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(dset_dir / f'hit_rate_vs_budget_threshold{threshold:.2f}.png', dpi=150)
            plt.close(fig)


def plot_recall_vs_budget(df: pd.DataFrame, out_dir: Path) -> None:
    """Scatter, not connected line+marker or bars -- budget_fraction
    points are unevenly spaced (0.05, 0.10, 0.15, 0.30, ...) so
    connecting them implies a continuity that isn't there (a bar-chart
    version was tried and reverted -- harder to read than scatter)."""
    base = df[df['K'] > 1]
    if base.empty:
        return
    for threshold in sorted(base['threshold'].unique()):
        sub = base[base['threshold'] == threshold]
        for dataset, dset_df in sub.groupby('dataset'):
            benchmarks = _sorted_benchmarks(dset_df['benchmark'].unique())
            k_values = sorted(dset_df['K'].unique())
            fig, axes = _grid(benchmarks, k_values)
            for r, bm in enumerate(benchmarks):
                for c, K in enumerate(k_values):
                    ax = axes[r][c]
                    cell = dset_df[(dset_df['benchmark'] == bm) & (dset_df['K'] == K)]
                    for pct, group in cell.groupby('percentile'):
                        agg = group.groupby('budget_fraction')['mean_recall'].mean().sort_index()
                        ax.scatter(agg.index, agg.values, s=16, label=f'p{int(pct)}',
                                   color=PERCENTILE_COLORS.get(pct))
                    ax.set_ylim(0, 1)
                    if r == 0:
                        ax.set_title(f'K={K}')
                    if c == 0:
                        ax.set_ylabel(f'{_short_label(bm)}\nmean recall')
                    if r == len(benchmarks) - 1:
                        ax.set_xlabel('budget fraction')
                    ax.grid(alpha=0.3)
            axes[0][0].legend(fontsize=7)
            fig.suptitle(f'{dataset.upper()}: overlap-threshold mean recall vs. budget fraction (threshold={threshold:.2f})')
            fig.tight_layout()
            dset_dir = out_dir / dataset
            dset_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(dset_dir / f'recall_vs_budget_threshold{threshold:.2f}.png', dpi=150)
            plt.close(fig)


def plot_recall_vs_hit_rate_tradeoff(df: pd.DataFrame, out_dir: Path) -> None:
    """One file per budget (pooling across budgets here was misleading
    -- different budgets have very different hit-rate/recall scales),
    scatter only (no connecting lines between thresholds -- threshold
    is the swept parameter, not a continuous path), both axes fixed to
    (0, 1) rather than autoscaled/zoomed."""
    base = df[df['K'] > 1]
    if base.empty:
        return
    for budget_fraction in sorted(base['budget_fraction'].unique()):
        sub = base[base['budget_fraction'] == budget_fraction]
        for dataset, dset_df in sub.groupby('dataset'):
            benchmarks = _sorted_benchmarks(dset_df['benchmark'].unique())
            k_values = sorted(dset_df['K'].unique())
            fig, axes = _grid(benchmarks, k_values, sharey=True)
            for r, bm in enumerate(benchmarks):
                for c, K in enumerate(k_values):
                    ax = axes[r][c]
                    cell = dset_df[(dset_df['benchmark'] == bm) & (dset_df['K'] == K)]
                    for pct, group in cell.groupby('percentile'):
                        agg = group.groupby('threshold')[['hit_rate', 'mean_recall']].mean()
                        ax.scatter(agg['hit_rate'], agg['mean_recall'], s=16, label=f'p{int(pct)}',
                                   color=PERCENTILE_COLORS.get(pct))
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    if r == 0:
                        ax.set_title(f'K={K}')
                    if c == 0:
                        ax.set_ylabel(f'{_short_label(bm)}\nmean recall')
                    if r == len(benchmarks) - 1:
                        ax.set_xlabel('hit rate')
                    ax.grid(alpha=0.3)
            axes[0][0].legend(fontsize=7)
            fig.suptitle(f'{dataset.upper()}: recall vs. hit-rate tradeoff (budget={budget_fraction*100:.0f}%, threshold varies per point)')
            fig.tight_layout()
            dset_dir = out_dir / dataset
            dset_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(dset_dir / f'recall_vs_hit_rate_tradeoff_budget{budget_fraction*100:.0f}pct.png', dpi=150)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results_csv', type=str)
    parser.add_argument('--oracle-csv', type=str, default=None,
                         help='oracle_hit_rate.csv from compute_oracle_hit_rate.py; '
                              'defaults to oracle_hit_rate.csv next to results_csv if present')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    csv_path = Path(args.results_csv)
    df = pd.read_csv(csv_path)
    out_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    oracle_path = Path(args.oracle_csv) if args.oracle_csv else csv_path.parent / 'oracle_hit_rate.csv'
    oracle_df = pd.read_csv(oracle_path) if oracle_path.exists() else None
    if oracle_df is None:
        print(f"No oracle_hit_rate.csv found at {oracle_path} -- hit_rate_vs_budget will skip the union ceiling line")

    plot_hit_rate_vs_budget(df, out_dir, oracle_df=oracle_df)
    plot_recall_vs_budget(df, out_dir)
    plot_recall_vs_hit_rate_tradeoff(df, out_dir)
    print(f"Plots written to {out_dir}")


if __name__ == '__main__':
    main()
