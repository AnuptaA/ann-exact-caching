#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Plots for the clustered-synthetic experiment 1 (pool+argmax) and
experiment 2 (fixed-cache/swept-test) reruns, plus overlays against the
original uniform-synthetic part1/part2 results.
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
K_VALUES = [10, 25, 50, 100]
N_BINS = 20
D = 128

#-------------------------------------------------------------------------------

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


def _overlay_plot(series: list, title: str, out_path: Path) -> None:
    """series: list of (label, color, {K: (fraction, recall)})."""
    fig, axes = plt.subplots(1, len(K_VALUES), figsize=(3.6 * len(K_VALUES), 3.4), squeeze=False)
    for c, K in enumerate(K_VALUES):
        ax = axes[0][c]
        for label, color, data in series:
            if K not in data:
                continue
            fraction, recall = data[K]
            centers, means, sems, counts = _bin_stats(fraction, recall)
            valid = counts > 0
            ax.errorbar(centers[valid], means[valid], yerr=sems[valid], marker='o', markersize=3,
                        linestyle='-', color=color, label=label, capsize=2)
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1, label='y=x')
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f'$K={K}$', fontsize=10)
        ax.set_xlabel('intersection fraction')
        if c == 0:
            ax.set_ylabel('mean actual recall')
            ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")

#-------------------------------------------------------------------------------

def load_exp1(run_dir: Path, part_dir: str) -> dict:
    out = {}
    for K in K_VALUES:
        path = run_dir / part_dir / f'{D}d' / 'raw' / f'K{K}_selection.npz'
        if not path.exists():
            continue
        arr = np.load(path)
        out[K] = (arr['fraction'], arr['recall'])
    return out


def load_exp2(run_dir: Path, part_dir: str) -> dict:
    out = {}
    for K in K_VALUES:
        path = run_dir / part_dir / f'{D}d' / 'raw' / f'K{K}.npz'
        if not path.exists():
            continue
        arr = np.load(path)
        out[K] = (arr['fraction'], arr['recall'])
    return out

#-------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp1-clustered-run-dir', type=str, required=True)
    parser.add_argument('--exp2-clustered-run-dir', type=str, required=True)
    parser.add_argument('--exp1-uniform-run-dir', type=str, default=None)
    parser.add_argument('--exp2-uniform-run-dir', type=str, default=None)
    parser.add_argument('--out-dir', type=str, default=None)
    args = parser.parse_args()

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = OUT_BASE / ts
    out_dir = out_dir / 'clustered_reruns'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing plots to: {out_dir}\n")

    exp1_clustered = load_exp1(Path(args.exp1_clustered_run_dir), 'part1_clustered')
    exp2_clustered = load_exp2(Path(args.exp2_clustered_run_dir), 'part2_clustered')

    series1 = [('clustered', 'tab:red', exp1_clustered)]
    series2 = [('clustered', 'tab:red', exp2_clustered)]

    if args.exp1_uniform_run_dir:
        exp1_uniform = load_exp1(Path(args.exp1_uniform_run_dir), 'part1')
        if exp1_uniform:
            series1.insert(0, ('uniform', 'tab:blue', exp1_uniform))
    if args.exp2_uniform_run_dir:
        exp2_uniform = load_exp2(Path(args.exp2_uniform_run_dir), 'part2')
        if exp2_uniform:
            series2.insert(0, ('uniform', 'tab:blue', exp2_uniform))

    _overlay_plot(series1, f'Experiment 1 (pool+argmax), d={D}: Clustered vs. Uniform',
                  out_dir / 'exp1_clustered_vs_uniform.png')
    _overlay_plot(series2, f'Experiment 2 (fixed-cache/swept-test), d={D}: Clustered vs. Uniform',
                  out_dir / 'exp2_clustered_vs_uniform.png')

    print(f"\nDone: {out_dir}")

if __name__ == '__main__':
    main()
