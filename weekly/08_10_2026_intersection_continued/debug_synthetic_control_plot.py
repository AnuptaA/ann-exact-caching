#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Overlays synthetic uniform-sphere calibration (Part 1, matched
candidate/test-query scale) against SIFT/ESCI debug validation
calibration, per K.
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

OUT_BASE = Path(__file__).parent / 'plots'
PART_DIR = 'debug_synthetic_control'
K_VALUES = [10, 25, 50, 100]
N_BINS = 20

DATASETS = [
    ('SIFT', 128, 'debug_sift'),
    ('ESCI', 384, 'debug_esci'),
]

#-------------------------------------------------------------------------------

def latest_run_dir(base: Path) -> Path:
    runs = sorted(p for p in base.glob('*') if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"no run directories under {base}")
    return runs[-1]

def load_real(run_dir: Path, part_dir: str, d: int, k_values: list) -> dict:
    out = {}
    for K in k_values:
        path = run_dir / part_dir / f'{d}d' / 'raw' / f'K{K}_selection.npz'
        arr = np.load(path)
        out[K] = {'fraction': arr['fraction'], 'recall': arr['recall']}
    return out

def load_synthetic(run_dir: Path, d: int, k_values: list) -> dict:
    out = {}
    for K in k_values:
        path = run_dir / 'part1' / f'{d}d' / 'raw' / f'K{K}_selection.npz'
        arr = np.load(path)
        out[K] = {'fraction': arr['cap_fraction'], 'recall': arr['cap_recall']}
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

def plot_overlay(real_data: dict, synthetic_data: dict, dataset_name: str, d: int, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(K_VALUES), figsize=(3.6 * len(K_VALUES), 3.4), squeeze=False)
    for c, K in enumerate(K_VALUES):
        ax = axes[0][c]

        centers, means, sems, counts = _bin_stats(real_data[K]['fraction'], real_data[K]['recall'])
        valid = counts > 0
        ax.errorbar(centers[valid], means[valid], yerr=sems[valid], marker='o', markersize=3,
                    linestyle='-', color='tab:green', label=dataset_name, capsize=2)

        centers, means, sems, counts = _bin_stats(synthetic_data[K]['fraction'], synthetic_data[K]['recall'])
        valid = counts > 0
        ax.errorbar(centers[valid], means[valid], yerr=sems[valid], marker='s', markersize=3,
                    linestyle='-', color='tab:blue', label='synthetic uniform sphere', capsize=2)

        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1, label='y=x')
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f'$K={K}$', fontsize=10)
        ax.set_xlabel('intersection fraction (best candidate)')
        if c == 0:
            ax.set_ylabel('mean actual recall')
            ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle(f'{dataset_name} (d={d}) vs. Synthetic Control: Binned Calibration, Matched Scale')
    plt.tight_layout()
    out_path = out_dir / f'{d}d_{dataset_name.lower()}_vs_synthetic.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")

#-------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic-run-dir', type=str, required=True,
                         help='part1_run_experiment.py output directory (the matched-scale control run)')
    parser.add_argument('--sift-run-dir', type=str, default=None, help='defaults to latest under debug_sift results')
    parser.add_argument('--esci-run-dir', type=str, default=None, help='defaults to latest under debug_esci results')
    parser.add_argument('--out-dir', type=str, default=None)
    args = parser.parse_args()

    synthetic_run_dir = Path(args.synthetic_run_dir)

    results_base = Path(__file__).parent / 'results'
    sift_run_dir = Path(args.sift_run_dir) if args.sift_run_dir else latest_run_dir(results_base)
    esci_run_dir = Path(args.esci_run_dir) if args.esci_run_dir else latest_run_dir(results_base)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = OUT_BASE / ts
    out_dir = out_dir / PART_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing plots to: {out_dir}\n")

    run_dirs = {'debug_sift': sift_run_dir, 'debug_esci': esci_run_dir}

    for dataset_name, d, part_dir in DATASETS:
        real_data = load_real(run_dirs[part_dir], part_dir, d, K_VALUES)
        synthetic_data = load_synthetic(synthetic_run_dir, d, K_VALUES)
        plot_overlay(real_data, synthetic_data, dataset_name, d, out_dir)

    print(f"\nDone: {out_dir}")

if __name__ == '__main__':
    main()
