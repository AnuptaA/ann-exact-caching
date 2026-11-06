#!/usr/bin/env python3
#-------------------------------------------------------------------------------
""" Plot for experimental overlap hit policy
Grid layout mirrors debug_intersection_calibration.py's convention:
benchmark as grid rows, K as grid columns, one figure per
dataset."""

import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import re

import pandas as pd
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------

PERCENTILE_COLORS = {25.0: 'tab:red', 50.0: 'tab:blue', 95.0: 'tab:orange', 99.0: 'tab:green'}

def _short_label(benchmark_name: str) -> str:
    """Short label for grid cell titles and y-axis labels for nice display"""
    m = re.search(r'c\d+k\d+', benchmark_name)
    return m.group(0) if m else benchmark_name


def _cache_size(benchmark_name: str) -> int:
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

def plot_hit_rate_vs_budget(df: pd.DataFrame, out_dir: Path) -> None:
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
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    csv_path = Path(args.results_csv)
    df = pd.read_csv(csv_path)
    out_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_hit_rate_vs_budget(df, out_dir)
    plot_recall_vs_budget(df, out_dir)
    plot_recall_vs_hit_rate_tradeoff(df, out_dir)
    print(f"Plots written to {out_dir}")


if __name__ == '__main__':
    main()
