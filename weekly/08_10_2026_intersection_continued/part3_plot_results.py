#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Plots for the r_cache diagnostic: fixed cache pole, fixed delta ratios,
r_cache swept via target coverage fraction.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#-------------------------------------------------------------------------------

RESULTS_BASE = Path(__file__).parent / 'results'
OUT_BASE = Path(__file__).parent / 'plots'
PART_DIR = 'part3'
DIMS = [2, 3, 8, 16, 32, 64, 128, 256, 384]
K_VALUES = [1, 5, 10, 25, 50, 100]
DELTA_RATIOS = [0.5, 1.0, 1.5]
METHOD_COLORS = {'cap': 'tab:green', 'solid': 'tab:orange'}
METHOD_LABELS = {'cap': 'cap intersection', 'solid': 'solid-ball intersection'}
N_BINS = 20

#-------------------------------------------------------------------------------

def latest_run_dir(base: Path) -> Path:
    runs = sorted(p for p in base.glob('*') if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"no run directories under {base}")
    return runs[-1]


def load_dimension(run_dir: Path, d: int, k_values: list) -> dict:
    out = {}
    for K in k_values:
        path = run_dir / PART_DIR / f'{d}d' / 'raw' / f'K{K}.npz'
        arr = np.load(path)
        out[K] = {key: arr[key] for key in arr.files}
    return out


def _bin_stats(x: np.ndarray, y: np.ndarray, n_bins: int = N_BINS):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    means = np.full(n_bins, np.nan)
    sems = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    for b in range(n_bins):
        mask = idx == b
        counts[b] = mask.sum()
        if counts[b] > 0:
            means[b] = y[mask].mean()
            sems[b] = y[mask].std(ddof=1) / np.sqrt(counts[b]) if counts[b] > 1 else 0.0
    return centers, means, sems, counts


def _grid(n_cols: int, n_rows: int = 1, figsize_scale=(3.6, 3.4)):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_scale[0] * n_cols, figsize_scale[1] * n_rows),
                              squeeze=False)
    return fig, axes

#-------------------------------------------------------------------------------
# 1. Recall (and predicted fraction) vs. coverage target, the primary
# diagnostic plot: does recall track the formula smoothly across the sweep?
#-------------------------------------------------------------------------------

def plot_recall_vs_coverage(d: int, data: dict, out_dir: Path) -> None:
    fig, axes = _grid(len(K_VALUES))
    colors = cm.viridis(np.linspace(0, 1, len(DELTA_RATIOS)))
    for c, K in enumerate(K_VALUES):
        ax = axes[0][c]
        sel = data[K]
        for ratio, color in zip(DELTA_RATIOS, colors):
            mask = sel['delta_ratio'] == ratio
            targets = np.unique(sel['coverage_target'][mask])
            recall_means = [sel['recall'][mask & (sel['coverage_target'] == t)].mean() for t in targets]
            cap_means = [sel['cap_fraction'][mask & (sel['coverage_target'] == t)].mean() for t in targets]
            ax.plot(targets, recall_means, marker='o', markersize=3, linestyle='-',
                     color=color, label=f'recall, delta={ratio}x')
            ax.plot(targets, cap_means, marker='x', markersize=3, linestyle='--',
                     color=color, alpha=0.6, label=f'cap fraction, delta={ratio}x')
        ax.set_xscale('log')
        ax.set_xlabel('coverage target (r_cache)')
        ax.set_title(f'$K={K}$', fontsize=10)
        if c == 0:
            ax.set_ylabel('mean value')
            ax.legend(fontsize=6, ncol=1)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    fig.suptitle(f'{d}D: Recall and Predicted Fraction vs. Coverage-Fraction Target (by delta ratio)')
    plt.tight_layout()
    out_path = out_dir / f'{d}d_recall_vs_coverage.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")

#-------------------------------------------------------------------------------
# 2. Binned calibration, pooled across delta ratio and coverage target
#-------------------------------------------------------------------------------

def plot_binned_calibration(d: int, data: dict, out_dir: Path) -> None:
    fig, axes = _grid(len(K_VALUES))
    for c, K in enumerate(K_VALUES):
        ax = axes[0][c]
        sel = data[K]
        for method in ('cap', 'solid'):
            centers, means, sems, counts = _bin_stats(sel[f'{method}_fraction'], sel['recall'])
            valid = counts > 0
            ax.errorbar(centers[valid], means[valid], yerr=sems[valid], marker='o', markersize=3,
                        linestyle='-', color=METHOD_COLORS[method], label=METHOD_LABELS[method], capsize=2)
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1, label='y=x')
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f'$K={K}$', fontsize=10)
        ax.set_xlabel('predicted intersection fraction')
        if c == 0:
            ax.set_ylabel('mean actual recall')
            ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle(f'{d}D: Binned Calibration (pooled across delta ratio, coverage target, backgrounds)')
    plt.tight_layout()
    out_path = out_dir / f'{d}d_binned_calibration.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")

#-------------------------------------------------------------------------------
# 3. Cap vs. solid-ball fraction, colored by coverage target
#-------------------------------------------------------------------------------

def plot_method_comparison_scatter(d: int, data: dict, out_dir: Path) -> None:
    fig, axes = _grid(len(K_VALUES))
    for c, K in enumerate(K_VALUES):
        ax = axes[0][c]
        sel = data[K]
        sc = ax.scatter(sel['cap_fraction'], sel['solid_fraction'], c=np.log10(sel['coverage_target']),
                        cmap='plasma', s=4, alpha=0.3, linewidths=0)
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1, label='y=x')
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f'$K={K}$', fontsize=10)
        ax.set_xlabel('cap fraction')
        if c == 0:
            ax.set_ylabel('solid-ball fraction')
        if c == len(K_VALUES) - 1:
            plt.colorbar(sc, ax=ax, label='log10(coverage target)')
        ax.grid(alpha=0.3)
    fig.suptitle(f'{d}D: Solid-Ball vs. Cap Fraction (colored by log10 coverage target)')
    plt.tight_layout()
    out_path = out_dir / f'{d}d_method_comparison_scatter.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")

#-------------------------------------------------------------------------------

def main() -> None:
    global K_VALUES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=str, default=None, help='defaults to latest run under results/')
    parser.add_argument('--dims', type=int, nargs='+', default=DIMS)
    parser.add_argument('--k-values', type=int, nargs='+', default=K_VALUES, dest='k_values')
    parser.add_argument('--out-dir', type=str, default=None,
                         help='write directly into this exact directory (no new timestamp minted), '
                              'e.g. to combine output with other part plot runs')
    args = parser.parse_args()

    K_VALUES = sorted(args.k_values)

    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir(RESULTS_BASE)
    print(f"Loading from: {run_dir}")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = OUT_BASE / ts
    out_dir = out_dir / PART_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing plots to: {out_dir}\n")

    for d in args.dims:
        print(f"=== d={d} ===")
        data = load_dimension(run_dir, d, args.k_values)
        plot_recall_vs_coverage(d, data, out_dir)
        plot_binned_calibration(d, data, out_dir)
        plot_method_comparison_scatter(d, data, out_dir)
        print()

    print(f"Done: {out_dir}")

if __name__ == '__main__':
    main()
