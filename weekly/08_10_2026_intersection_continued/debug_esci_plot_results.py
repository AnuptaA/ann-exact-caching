#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Plots for the ESCI validation of the pool+argmax calibration experiment.

Each K's data has two parts: 'selection' (per test query, winning
candidate's own fraction/recall) and 'pairwise' (a raw subsample of the
full test_q x cache_q cross product before any argmax).
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
PART_DIR = 'debug_esci'
D = 384
K_VALUES = [10, 25, 50, 100]
N_BINS = 20

#-------------------------------------------------------------------------------

def latest_run_dir(base: Path) -> Path:
    runs = sorted(p for p in base.glob('*') if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"no run directories under {base}")
    return runs[-1]

def load_data(run_dir: Path, k_values: list) -> dict:
    out = {}
    for K in k_values:
        out[K] = {}
        for part in ('selection', 'pairwise'):
            path = run_dir / PART_DIR / f'{D}d' / 'raw' / f'K{K}_{part}.npz'
            arr = np.load(path)
            out[K][part] = {key: arr[key] for key in arr.files}
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

def plot_binned_calibration(data: dict, out_dir: Path) -> None:
    fig, axes = _grid(len(K_VALUES))
    for c, K in enumerate(K_VALUES):
        ax = axes[0][c]
        sel = data[K]['selection']
        centers, means, sems, counts = _bin_stats(sel['fraction'], sel['recall'])
        valid = counts > 0
        ax.errorbar(centers[valid], means[valid], yerr=sems[valid], marker='o', markersize=3,
                    linestyle='-', color='tab:green', label='cap intersection', capsize=2)
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1, label='y=x')
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f'$K={K}$', fontsize=10)
        ax.set_xlabel('intersection fraction (best candidate)')
        if c == 0:
            ax.set_ylabel('mean actual recall')
            ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle(f'ESCI (d={D}): Binned Calibration (best-candidate intersection fraction vs. actual recall, pooled across seeds)')
    plt.tight_layout()
    out_path = out_dir / f'{D}d_binned_calibration.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")


def plot_calibration_scatter(data: dict, out_dir: Path) -> None:
    fig, axes = _grid(len(K_VALUES))
    rng = np.random.default_rng(0)
    for c, K in enumerate(K_VALUES):
        ax = axes[0][c]
        sel = data[K]['selection']
        fraction = sel['fraction']
        jitter = rng.uniform(-0.015, 0.015, size=len(fraction))
        ax.scatter(fraction, sel['recall'] + jitter, s=4, alpha=0.25,
                  color='tab:green', label='cap intersection', linewidths=0)
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1, label='y=x')
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f'$K={K}$', fontsize=10)
        ax.set_xlabel('intersection fraction (best candidate)')
        if c == 0:
            ax.set_ylabel('actual recall (jittered)')
            ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle(f'ESCI (d={D}): Calibration Scatter, Unbinned')
    plt.tight_layout()
    out_path = out_dir / f'{D}d_calibration_scatter.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")

#-------------------------------------------------------------------------------

def main() -> None:
    global K_VALUES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=str, default=None, help='defaults to latest run under results/')
    parser.add_argument('--k-values', type=int, nargs='+', default=K_VALUES, dest='k_values')
    parser.add_argument('--out-dir', type=str, default=None,
                         help='write directly into this exact directory (no new timestamp minted)')
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

    data = load_data(run_dir, args.k_values)
    plot_binned_calibration(data, out_dir)
    plot_calibration_scatter(data, out_dir)

    print(f"\nDone: {out_dir}")

if __name__ == '__main__':
    main()
