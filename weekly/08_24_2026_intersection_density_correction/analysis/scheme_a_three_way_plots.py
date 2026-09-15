#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Plots for the Scheme A three-way comparison: IF (r_hat_q*),
per_cache_query_recall_hat, per_test_query_recall_hat - all from one
unified run (sift_scheme_a_recall_hat_run_experiment.py), so all three
predictors share the same background/candidate/query rows by
construction."""

import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'common'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scheme_corpus_comparison_plot_results import _ols_fit, _concordance_correlation_coefficient

#-------------------------------------------------------------------------------

D = 128
K_VALUES = [1, 5, 10, 25, 50, 100]
CORPORA = ('real', 'uniform')
CORPUS_LABELS = {'real': 'real corpus', 'uniform': 'uniform corpus'}
PART_DIRS = {'real': 'sift_part2_real_data_recall_hat', 'uniform': 'sift_part2_real_anchor_on_uniform_recall_hat'}

PREDICTORS = [
    ('if_fair', 'IF (r_hat_q*)', lambda d: d['if_fair']),
    ('pcq', 'per_cache_query_recall_hat', lambda d: np.clip(d['per_cache_query_recall_hat'], 0.0, 1.0)),
    ('ptq', 'per_test_query_recall_hat', lambda d: d['per_test_query_recall_hat']),
]

PRED_BIN_EDGES = np.arange(0.0, 1.05, 0.05)
N_PRED_BINS = len(PRED_BIN_EDGES) - 1

OUT_DIR = Path(__file__).parent.parent / 'plots' / 'scheme_a_three_way'

#-------------------------------------------------------------------------------


def _pred_binned_mean(pred: np.ndarray, actual: np.ndarray) -> tuple:
    centers = (PRED_BIN_EDGES[:-1] + PRED_BIN_EDGES[1:]) / 2.0
    means = np.full(N_PRED_BINS, np.nan)
    sems = np.full(N_PRED_BINS, np.nan)
    counts = np.zeros(N_PRED_BINS, dtype=int)
    idx = np.clip(np.digitize(pred, PRED_BIN_EDGES) - 1, 0, N_PRED_BINS - 1)
    for b in range(N_PRED_BINS):
        mask = idx == b
        counts[b] = mask.sum()
        if counts[b] > 0:
            means[b] = actual[mask].mean()
            sems[b] = actual[mask].std(ddof=1) / np.sqrt(counts[b]) if counts[b] > 1 else 0.0
    return centers, means, sems, counts


def load_all(run_dir: Path) -> dict:
    data = {}
    for corpus in CORPORA:
        data[corpus] = {}
        for K in K_VALUES:
            data[corpus][K] = dict(np.load(run_dir / PART_DIRS[corpus] / f'{D}d' / 'raw' / f'K{K}.npz'))
    return data


def plot_scatter(data: dict, out_dir: Path) -> None:
    """Predicted vs actual scatter, y=x reference - one file per corpus,
    rows=predictor, cols=K."""
    for corpus in CORPORA:
        fig, axes = plt.subplots(3, len(K_VALUES), figsize=(3.0 * len(K_VALUES), 9), squeeze=False)
        for col, K in enumerate(K_VALUES):
            d = data[corpus][K]
            actual = d['recall']
            for row, (key, label, extract) in enumerate(PREDICTORS):
                pred = extract(d)
                ax = axes[row][col]
                finite = np.isfinite(pred)
                ax.scatter(pred[finite], actual[finite], s=3, alpha=0.1, color='tab:blue', linewidths=0)
                ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
                ax.set_xlim(-0.02, 1.02)
                ax.set_ylim(-0.05, 1.05)
                ax.set_title(f'{label}, K={K}', fontsize=8)
                if row == 2:
                    ax.set_xlabel('predicted')
                if col == 0:
                    ax.set_ylabel('actual recall')
                ax.grid(alpha=0.3)
        fig.suptitle(f'Scheme A three-way: predicted vs. actual recall, SIFT (128D), {CORPUS_LABELS[corpus]}')
        plt.tight_layout()
        out_path = out_dir / f'scatter_{corpus}.png'
        fig.savefig(out_path, dpi=110, bbox_inches='tight')
        plt.close(fig)
        print(f"wrote {out_path}")


def plot_calibration_bins(data: dict, out_dir: Path) -> None:
    """Binned calibration: mean actual recall per 0.05-wide predicted bin,
    against y=x."""
    for corpus in CORPORA:
        fig, axes = plt.subplots(3, len(K_VALUES), figsize=(3.0 * len(K_VALUES), 9), squeeze=False)
        for col, K in enumerate(K_VALUES):
            d = data[corpus][K]
            actual = d['recall']
            for row, (key, label, extract) in enumerate(PREDICTORS):
                pred = extract(d)
                finite = np.isfinite(pred)
                centers, means, sems, counts = _pred_binned_mean(pred[finite], actual[finite])
                valid = counts > 0
                ax = axes[row][col]
                ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
                ax.errorbar(centers[valid], means[valid], yerr=sems[valid], marker='o', markersize=3,
                            color='tab:blue', capsize=2)
                ax.set_xlim(-0.02, 1.02)
                ax.set_ylim(-0.05, 1.05)
                ax.set_title(f'{label}, K={K}', fontsize=8)
                if row == 2:
                    ax.set_xlabel('predicted (bin center)')
                if col == 0:
                    ax.set_ylabel('mean actual recall')
                ax.grid(alpha=0.3)
        fig.suptitle(f'Scheme A three-way: calibration, SIFT (128D), {CORPUS_LABELS[corpus]}')
        plt.tight_layout()
        out_path = out_dir / f'calibration_bins_{corpus}.png'
        fig.savefig(out_path, dpi=110, bbox_inches='tight')
        plt.close(fig)
        print(f"wrote {out_path}")


def plot_mean_bias_bars(data: dict, out_dir: Path) -> None:
    """Mean signed bias (actual - predicted) per predictor, grouped by K -
    one figure, 2 panels (real/uniform), the headline summary number."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), squeeze=False)
    x = np.arange(len(K_VALUES))
    width = 0.25
    colors = {'if_fair': 'tab:orange', 'pcq': 'tab:green', 'ptq': 'tab:blue'}
    for col, corpus in enumerate(CORPORA):
        ax = axes[0][col]
        for j, (key, label, extract) in enumerate(PREDICTORS):
            biases = []
            for K in K_VALUES:
                d = data[corpus][K]
                pred = extract(d)
                finite = np.isfinite(pred)
                biases.append((d['recall'][finite] - pred[finite]).mean())
            ax.bar(x + (j - 1) * width, biases, width, label=label, color=colors[key])
        ax.axhline(0.0, color='black', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'K={K}' for K in K_VALUES])
        ax.set_ylabel('mean bias (actual - predicted)')
        ax.set_title(CORPUS_LABELS[corpus])
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Scheme A three-way: mean calibration bias, SIFT (128D)')
    plt.tight_layout()
    out_path = out_dir / 'mean_bias_bars.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_path}")

#-------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=str, required=True)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_all(Path(args.run_dir))

    plot_scatter(data, OUT_DIR)
    plot_calibration_bins(data, OUT_DIR)
    plot_mean_bias_bars(data, OUT_DIR)


if __name__ == '__main__':
    main()
