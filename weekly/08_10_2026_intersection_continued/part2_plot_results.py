#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Plots for the controlled-construction intersection-fraction study.

Each K's data is one flat array (no argmax/selection step, since there is
a single fixed cache candidate).
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

#-------------------------------------------------------------------------------

RESULTS_BASE = Path(__file__).parent / 'results'
OUT_BASE = Path(__file__).parent / 'plots'
PART_DIR = 'part2'
DIMS = [2, 3, 8, 16, 32, 64, 128, 256, 384]
K_VALUES = [1, 5, 10, 25, 50, 100]
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
# 1. Binned calibration plot
#-------------------------------------------------------------------------------

def plot_binned_calibration(d: int, data: dict, out_dir: Path) -> None:
    fig, axes = _grid(len(K_VALUES))
    for c, K in enumerate(K_VALUES):
        ax = axes[0][c]
        rec = data[K]
        for method in ('cap', 'solid'):
            centers, means, sems, counts = _bin_stats(rec[f'{method}_fraction'], rec['recall'])
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
    fig.suptitle(f'{d}D: Binned Calibration (controlled delta sweep, pooled across backgrounds)')
    plt.tight_layout()
    out_path = out_dir / f'{d}d_binned_calibration.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")


def plot_calibration_scatter(d: int, data: dict, out_dir: Path) -> None:
    """Unbinned companion to plot_binned_calibration."""
    rng = np.random.default_rng(0)
    fig, axes = _grid(len(K_VALUES))
    for c, K in enumerate(K_VALUES):
        ax = axes[0][c]
        rec = data[K]
        for method in ('cap', 'solid'):
            fraction = rec[f'{method}_fraction']
            jitter = rng.uniform(-0.015, 0.015, size=len(fraction))
            ax.scatter(fraction, rec['recall'] + jitter, s=4, alpha=0.25,
                      color=METHOD_COLORS[method], label=METHOD_LABELS[method], linewidths=0)
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1, label='y=x')
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f'$K={K}$', fontsize=10)
        ax.set_xlabel('predicted intersection fraction')
        if c == 0:
            ax.set_ylabel('actual recall (jittered)')
            ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle(f'{d}D: Calibration Scatter, Unbinned')
    plt.tight_layout()
    out_path = out_dir / f'{d}d_calibration_scatter.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")

#-------------------------------------------------------------------------------
# 2. Cap vs. solid-ball fraction agreement, colored by delta (the swept
# controlled variable, in place of r_cache which is now nearly constant per K)
#-------------------------------------------------------------------------------

def plot_method_comparison_scatter(d: int, data: dict, out_dir: Path) -> None:
    fig, axes = _grid(len(K_VALUES))
    for c, K in enumerate(K_VALUES):
        ax = axes[0][c]
        rec = data[K]
        sc = ax.scatter(rec['cap_fraction'], rec['solid_fraction'], c=rec['delta'], cmap='plasma',
                        s=4, alpha=0.3, linewidths=0)
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1, label='y=x')
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f'$K={K}$', fontsize=10)
        ax.set_xlabel('cap fraction')
        if c == 0:
            ax.set_ylabel('solid-ball fraction')
        if c == len(K_VALUES) - 1:
            plt.colorbar(sc, ax=ax, label='delta (rad)')
        ax.grid(alpha=0.3)
    fig.suptitle(f'{d}D: Solid-Ball vs. Cap Fraction (colored by delta)')
    plt.tight_layout()
    out_path = out_dir / f'{d}d_method_comparison_scatter.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")

#-------------------------------------------------------------------------------
# 3. Aggregate summary: mean/median/p90/p95 of r_cache, r_test, delta by K
#-------------------------------------------------------------------------------

def plot_aggregate_summary(d: int, data: dict, out_dir: Path) -> None:
    fig, axes = _grid(3, figsize_scale=(4.2, 4.0))
    quantities = ['r_cache', 'r_test', 'delta']
    stats_fns = [('mean', np.mean), ('median', np.median),
                 ('p90', lambda x: np.percentile(x, 90)), ('p95', lambda x: np.percentile(x, 95))]
    x = np.arange(len(K_VALUES))
    for c, qty in enumerate(quantities):
        ax = axes[0][c]
        for stat_name, fn in stats_fns:
            values = [fn(data[K][qty]) for K in K_VALUES]
            ax.plot(x, values, marker='o', markersize=4, label=stat_name)
        ax.set_xticks(x)
        ax.set_xticklabels([str(K) for K in K_VALUES])
        ax.set_xlabel('K')
        ax.set_ylabel(f'{qty} (rad)')
        ax.set_title(qty)
        ax.grid(alpha=0.3)
        if c == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f'{d}D: Aggregate Summary of r_cache, r_test, delta by K')
    plt.tight_layout()
    out_path = out_dir / f'{d}d_aggregate_summary.png'
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
                              "e.g. to combine output with another part's run")
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
        plot_binned_calibration(d, data, out_dir)
        plot_calibration_scatter(d, data, out_dir)
        plot_method_comparison_scatter(d, data, out_dir)
        plot_aggregate_summary(d, data, out_dir)
        print()

    print(f"Done: {out_dir}")

if __name__ == '__main__':
    main()
