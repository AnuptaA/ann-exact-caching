#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Plots IF vs. recall for the synthetic angular-sweep validation
(argmax-selected candidate per test query), plus IF coverage."""

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
K_VALUES = [25, 50, 100]
DATASET_DIMS = {'sift': 128, 'esci': 384}
N_IF_BINS = 20
IF_BIN_EDGES = np.linspace(0.0, 1.0, N_IF_BINS + 1)

#-------------------------------------------------------------------------------

def latest_run_dir(base: Path) -> Path:
    runs = sorted(p for p in base.glob('*') if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"no run directories under {base}")
    return runs[-1]


def load_dataset(run_dir: Path, dataset: str, d: int, k_values: list) -> dict:
    part_dir = f'{dataset}_synth_sweep'
    out = {}
    for K in k_values:
        path = run_dir / part_dir / f'{d}d' / 'raw' / f'K{K}.npz'
        arr = np.load(path)
        out[K] = {key: arr[key] for key in arr.files}
    return out


def balance_by_if_bin(fraction: np.ndarray, recall: np.ndarray, rng: np.random.Generator) -> tuple:
    """Downsample to equal count per 0.05-wide IF bin (uniform random,
    no replacement), N = smallest non-empty bin count. Returns
    (balanced_fraction, balanced_recall, n_per_bin, raw_counts)."""
    bin_idx = np.clip(np.digitize(fraction, IF_BIN_EDGES[1:-1]), 0, N_IF_BINS - 1)
    raw_counts = np.array([np.sum(bin_idx == b) for b in range(N_IF_BINS)])
    n_per_bin = int(raw_counts[raw_counts > 0].min())

    selected = []
    for b in range(N_IF_BINS):
        idx_in_bin = np.flatnonzero(bin_idx == b)
        if len(idx_in_bin) == 0:
            continue
        chosen = rng.choice(idx_in_bin, size=n_per_bin, replace=False)
        selected.append(chosen)
    selected = np.concatenate(selected)
    return fraction[selected], recall[selected], n_per_bin, raw_counts


def _ols_fit(x: np.ndarray, y: np.ndarray) -> tuple:
    """Ordinary least squares recall ~ slope*IF + intercept. Returns
    (slope, intercept, r_squared)."""
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return float(slope), float(intercept), float(r_squared)


def _concordance_correlation_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's CCC: agreement with the identity line y=x, penalizing both
    correlation strength and any shift in location/scale (unlike
    Pearson r, which is invariant to those shifts)."""
    mean_x, mean_y = x.mean(), y.mean()
    var_x, var_y = x.var(), y.var()
    covariance = np.mean((x - mean_x) * (y - mean_y))
    return float(2 * covariance / (var_x + var_y + (mean_x - mean_y) ** 2))


def _fixed_bin_stats(x: np.ndarray, y: np.ndarray):
    """Mean/sem of y within each fixed 0.05-wide IF bin."""
    centers = (IF_BIN_EDGES[:-1] + IF_BIN_EDGES[1:]) / 2.0
    means = np.full(N_IF_BINS, np.nan)
    sems = np.full(N_IF_BINS, np.nan)
    counts = np.zeros(N_IF_BINS, dtype=int)
    idx = np.clip(np.digitize(x, IF_BIN_EDGES[1:-1]), 0, N_IF_BINS - 1)
    for b in range(N_IF_BINS):
        mask = idx == b
        counts[b] = mask.sum()
        if counts[b] > 0:
            means[b] = y[mask].mean()
            sems[b] = y[mask].std(ddof=1) / np.sqrt(counts[b]) if counts[b] > 1 else 0.0
    return centers, means, sems, counts


def _grid(n_cols: int, figsize_scale=(3.8, 3.6)):
    fig, axes = plt.subplots(1, n_cols, figsize=(figsize_scale[0] * n_cols, figsize_scale[1]), squeeze=False)
    return fig, axes[0]

#-------------------------------------------------------------------------------
# 1. IF coverage histogram (raw, pre-downsampling), independent of recall
#-------------------------------------------------------------------------------

def plot_if_coverage(dataset: str, d: int, data: dict, out_dir: Path) -> None:
    fig, axes = _grid(len(K_VALUES))
    for c, K in enumerate(K_VALUES):
        ax = axes[c]
        fraction = data[K]['fraction']
        ax.hist(fraction, bins=IF_BIN_EDGES, color='tab:blue', alpha=0.75)
        ax.set_xlim(-0.02, 1.02)
        ax.set_title(f'$K={K}$ (n={len(fraction)}, raw)', fontsize=10)
        ax.set_xlabel('intersection fraction (argmax candidate)')
        if c == 0:
            ax.set_ylabel('count')
        ax.grid(alpha=0.3)
    fig.suptitle(f'{dataset.upper()} ({d}D): Raw IF Coverage, Pre-Downsampling (before any recall comparison)')
    plt.tight_layout()
    out_path = out_dir / f'{dataset}_{d}d_if_coverage.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")

#-------------------------------------------------------------------------------
# 2. Recall vs. IF: scatter and fixed-bin average, on the balanced
# (downsampled) dataset
#-------------------------------------------------------------------------------

def plot_scatter(dataset: str, d: int, balanced: dict, out_dir: Path) -> None:
    fig, axes = _grid(len(K_VALUES))
    rng = np.random.default_rng(0)
    for c, K in enumerate(K_VALUES):
        ax = axes[c]
        fraction, recall = balanced[K]['fraction'], balanced[K]['recall']
        jitter = rng.uniform(-0.01, 0.01, size=len(recall))
        ax.scatter(fraction, recall + jitter, s=4, alpha=0.15, color='tab:blue', linewidths=0)
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1, label='y=x')
        slope, intercept, r2 = balanced[K]['ols']
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f'$K={K}$ (n={len(fraction)}, balanced)\n'
                     f'$R^2$={r2:.3f}, slope={slope:.3f}, intercept={intercept:.3f}', fontsize=9)
        ax.set_xlabel('intersection fraction (argmax candidate)')
        if c == 0:
            ax.set_ylabel('actual recall (jittered)')
            ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle(f'{dataset.upper()} ({d}D): IF vs. Recall, Balanced Scatter (equal count per 0.05 IF bin)')
    plt.tight_layout()
    out_path = out_dir / f'{dataset}_{d}d_scatter.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")


def plot_binned(dataset: str, d: int, balanced: dict, out_dir: Path) -> None:
    fig, axes = _grid(len(K_VALUES))
    for c, K in enumerate(K_VALUES):
        ax = axes[c]
        fraction, recall = balanced[K]['fraction'], balanced[K]['recall']
        centers, means, sems, counts = _fixed_bin_stats(fraction, recall)
        valid = counts > 0
        ax.errorbar(centers[valid], means[valid], yerr=sems[valid], marker='o', markersize=3,
                    linestyle='-', color='tab:blue', capsize=2, label='binned mean recall')
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1, label='y=x')
        _, _, r2_binned = _ols_fit(centers[valid], means[valid])
        ccc = balanced[K]['ccc']
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f'$K={K}$\nbinned $R^2$={r2_binned:.3f} (inflated by averaging), CCC={ccc:.3f}', fontsize=9)
        ax.set_xlabel('intersection fraction (argmax candidate)')
        if c == 0:
            ax.set_ylabel('mean actual recall')
            ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle(f'{dataset.upper()} ({d}D): IF vs. Recall, Fixed 0.05-Bin Average (balanced dataset)')
    plt.tight_layout()
    out_path = out_dir / f'{dataset}_{d}d_binned.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")

#-------------------------------------------------------------------------------

def main() -> None:
    global K_VALUES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=str, choices=['sift', 'esci'], required=True)
    parser.add_argument('--run-dir', type=str, default=None, help='defaults to latest run under results/')
    parser.add_argument('--k-values', type=int, nargs='+', default=K_VALUES, dest='k_values')
    parser.add_argument('--out-dir', type=str, default=None,
                         help='write directly into this exact directory (no new timestamp minted)')
    parser.add_argument('--seed', type=int, default=0, help='downsampling RNG seed')
    args = parser.parse_args()

    K_VALUES = sorted(args.k_values)
    d = DATASET_DIMS[args.dataset]

    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir(RESULTS_BASE)
    print(f"Loading from: {run_dir}")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = OUT_BASE / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing plots to: {out_dir}\n")

    data = load_dataset(run_dir, args.dataset, d, K_VALUES)
    plot_if_coverage(args.dataset, d, data, out_dir)

    rng = np.random.default_rng(args.seed)
    balanced = {}
    print(f"\n  Balancing (downsampling to equal count per 0.05 IF bin), {args.dataset.upper()} ({d}D):")
    for K in K_VALUES:
        fraction, recall, n_per_bin, raw_counts = balance_by_if_bin(data[K]['fraction'], data[K]['recall'], rng)
        slope, intercept, r2 = _ols_fit(fraction, recall)
        ccc = _concordance_correlation_coefficient(fraction, recall)
        home_rate = float(data[K]['is_home_winner'].mean()) if 'is_home_winner' in data[K] else float('nan')
        balanced[K] = {'fraction': fraction, 'recall': recall, 'ols': (slope, intercept, r2), 'ccc': ccc}
        print(f"    K={K}: raw n={len(data[K]['fraction'])}, balanced N/bin={n_per_bin}, "
              f"balanced total={len(fraction)}, is_home_winner rate={home_rate:.4f}")
        print(f"      R^2={r2:.4f}, slope={slope:.4f}, intercept={intercept:.4f}, CCC={ccc:.4f}")

    plot_scatter(args.dataset, d, balanced, out_dir)
    plot_binned(args.dataset, d, balanced, out_dir)

    print(f"\nDone: {out_dir}")

if __name__ == '__main__':
    main()
