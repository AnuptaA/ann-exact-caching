#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Plots for the fixed-test-query real-data sweep."""

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
K_VALUES = [10, 25, 50, 100]
N_BINS = 20

#-------------------------------------------------------------------------------

def latest_run_dir(base: Path) -> Path:
    runs = sorted(p for p in base.glob('*') if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"no run directories under {base}")
    return runs[-1]

def load_data(run_dir: Path, part_dir: str, d: int, k_values: list) -> dict:
    out = {}
    for K in k_values:
        path = run_dir / part_dir / f'{d}d' / 'raw' / f'K{K}.npz'
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

#-------------------------------------------------------------------------------

def plot_per_anchor(data: dict, dataset_name: str, d: int, out_dir: Path) -> None:
    n_anchors = int(data[K_VALUES[0]]['anchor_id'].max()) + 1
    fig, axes = plt.subplots(n_anchors, len(K_VALUES), figsize=(3.6 * len(K_VALUES), 3.0 * n_anchors), squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 1, n_anchors))
    for r in range(n_anchors):
        for c, K in enumerate(K_VALUES):
            ax = axes[r][c]
            sel = data[K]['anchor_id'] == r
            fraction = data[K]['fraction'][sel]
            recall = data[K]['recall'][sel]
            centers, means, sems, counts = _bin_stats(fraction, recall)
            valid = counts > 0
            ax.errorbar(centers[valid], means[valid], yerr=sems[valid], marker='o', markersize=3,
                        linestyle='-', color=colors[r], capsize=2)
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            if r == 0:
                ax.set_title(f'$K={K}$', fontsize=10)
            if c == 0:
                ax.set_ylabel(f'anchor {r}\nrecall')
            if r == n_anchors - 1:
                ax.set_xlabel('intersection fraction')
            ax.grid(alpha=0.3)
    fig.suptitle(f'{dataset_name} (d={d}): Fixed-Query Sweep, Per Anchor')
    plt.tight_layout()
    out_path = out_dir / f'{d}d_per_anchor.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")


def plot_anchor_overlay(data: dict, dataset_name: str, d: int, out_dir: Path) -> None:
    n_anchors = int(data[K_VALUES[0]]['anchor_id'].max()) + 1
    colors = plt.cm.tab10(np.linspace(0, 1, n_anchors))
    fig, axes = plt.subplots(1, len(K_VALUES), figsize=(3.6 * len(K_VALUES), 3.4), squeeze=False)
    for c, K in enumerate(K_VALUES):
        ax = axes[0][c]
        for r in range(n_anchors):
            sel = data[K]['anchor_id'] == r
            fraction = data[K]['fraction'][sel]
            recall = data[K]['recall'][sel]
            centers, means, sems, counts = _bin_stats(fraction, recall)
            valid = counts > 0
            ax.errorbar(centers[valid], means[valid], yerr=sems[valid], marker='o', markersize=3,
                        linestyle='-', color=colors[r], label=f'anchor {r}', capsize=2)
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1, label='y=x')
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f'$K={K}$', fontsize=10)
        ax.set_xlabel('intersection fraction')
        if c == 0:
            ax.set_ylabel('mean actual recall')
            ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle(f'{dataset_name} (d={d}): All Anchors Overlaid')
    plt.tight_layout()
    out_path = out_dir / f'{d}d_anchor_overlay.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")

def plot_log_scale_scatter(data: dict, dataset_name: str, d: int, out_dir: Path) -> None:
    n_anchors = int(data[K_VALUES[0]]['anchor_id'].max()) + 1
    colors = plt.cm.tab10(np.linspace(0, 1, n_anchors))
    fig, axes = plt.subplots(1, len(K_VALUES), figsize=(3.6 * len(K_VALUES), 3.4), squeeze=False)
    for c, K in enumerate(K_VALUES):
        ax = axes[0][c]
        fraction = data[K]['fraction']
        recall = data[K]['recall']
        anchor_id = data[K]['anchor_id']
        floor = fraction[fraction > 0].min() * 0.5 if (fraction > 0).any() else 1e-300
        x = np.clip(fraction, floor, None)
        for r in range(n_anchors):
            sel = anchor_id == r
            ax.scatter(x[sel], recall[sel], s=6, alpha=0.4, color=colors[r], label=f'anchor {r}', linewidths=0)
        ax.set_xscale('log')
        ax.set_title(f'$K={K}$', fontsize=10)
        ax.set_xlabel('intersection fraction (log)')
        if c == 0:
            ax.set_ylabel('actual recall')
            ax.legend(fontsize=7, markerscale=2)
        ax.grid(alpha=0.3)
    fig.suptitle(f'{dataset_name} (d={d}): Fraction vs. Recall, Log-Scale X')
    plt.tight_layout()
    out_path = out_dir / f'{d}d_log_scale_scatter.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")

#-------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['sift', 'esci'], required=True)
    parser.add_argument('--run-dir', type=str, default=None)
    parser.add_argument('--out-dir', type=str, default=None)
    args = parser.parse_args()

    part_dir = f'debug_{args.dataset}_fixed_query_sweep'
    d = 128 if args.dataset == 'sift' else 384
    dataset_name = args.dataset.upper()

    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir(RESULTS_BASE)
    print(f"Loading from: {run_dir}")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = OUT_BASE / ts
    out_dir = out_dir / part_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing plots to: {out_dir}\n")

    data = load_data(run_dir, part_dir, d, K_VALUES)
    plot_per_anchor(data, dataset_name, d, out_dir)
    plot_anchor_overlay(data, dataset_name, d, out_dir)
    plot_log_scale_scatter(data, dataset_name, d, out_dir)

    print(f"\nDone: {out_dir}")

if __name__ == '__main__':
    main()
