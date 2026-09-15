#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Plots for the Scheme B three-way argmax-selection comparison: IF
(r_hat_q*), per_cache_query_recall_hat, per_test_query_recall_hat - all
from one unified run (sift_scheme_b_recall_hat_run_experiment.py), so all
three predictors share the same candidate pool/queries/corpus by
construction. Each predictor does its own independent argmax over the
full pooled candidate population per test query."""

import sys
import math
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'common'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------

D = 128
K_VALUES = [25, 50, 100]
CORPORA = ('real', 'uniform')
CORPUS_LABELS = {'real': 'real corpus', 'uniform': 'uniform corpus'}
PART_DIRS = {'real': 'sift_synth_sweep_recall_hat', 'uniform': 'sift_synth_sweep_on_uniform_recall_hat'}

PREDICTORS = [
    ('if_fair', 'recall_if', 'is_home_if', 'IF (r_hat_q*)', 'tab:orange'),
    ('per_cache_query_recall_hat', 'recall_pcq', 'is_home_pcq', 'per_cache_query_recall_hat', 'tab:green'),
    ('per_test_query_recall_hat', 'recall_ptq', 'is_home_ptq', 'per_test_query_recall_hat', 'tab:blue'),
]

DELTA_BIN_EDGES = np.array([0.0, 1e-4, 1e-2, 0.1, 0.3, 0.6, 1.0, 2.0, np.pi])
DELTA_BIN_LABELS = ['0', '(0,1e-4]', '(1e-4,1e-2]', '(1e-2,0.1]', '(0.1,0.3]',
                     '(0.3,0.6]', '(0.6,1.0]', '(1.0,2.0]', '(2.0,pi]']
N_DELTA_BINS = len(DELTA_BIN_LABELS)

PRED_BIN_EDGES = np.arange(0.0, 1.05, 0.05)
N_PRED_BINS = len(PRED_BIN_EDGES) - 1

OUT_DIR = Path(__file__).parent.parent / 'plots' / 'scheme_b_three_way'

#-------------------------------------------------------------------------------


def _delta_bin_index(delta: np.ndarray) -> np.ndarray:
    idx = np.zeros(len(delta), dtype=np.int64)
    for b in range(1, len(DELTA_BIN_EDGES)):
        idx[delta > DELTA_BIN_EDGES[b - 1]] = b
    return idx


def _delta_bin_centers() -> np.ndarray:
    centers = np.zeros(N_DELTA_BINS)
    centers[1] = DELTA_BIN_EDGES[1] / 2.0
    for b in range(2, N_DELTA_BINS):
        centers[b] = math.sqrt(DELTA_BIN_EDGES[b - 1] * DELTA_BIN_EDGES[b])
    return centers


def _binned_mean(x: np.ndarray, y: np.ndarray, bin_index_fn, n_bins: int) -> tuple:
    idx = bin_index_fn(x)
    means = np.full(n_bins, np.nan)
    sems = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = idx == b
        counts[b] = mask.sum()
        if counts[b] > 0:
            means[b] = y[mask].mean()
            sems[b] = y[mask].std(ddof=1) / np.sqrt(counts[b]) if counts[b] > 1 else 0.0
    return means, sems, counts


def _pred_bin_index(pred: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(pred, PRED_BIN_EDGES) - 1, 0, N_PRED_BINS - 1)


def load_all(run_dir: Path) -> dict:
    data = {}
    for corpus in CORPORA:
        data[corpus] = {}
        for K in K_VALUES:
            data[corpus][K] = dict(np.load(run_dir / PART_DIRS[corpus] / f'{D}d' / 'raw' / f'K{K}.npz'))
    return data


def plot_mean_recall_bars(data: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), squeeze=False)
    x = np.arange(len(K_VALUES))
    width = 0.25
    for col, corpus in enumerate(CORPORA):
        ax = axes[0][col]
        for j, (pred_f, recall_f, home_f, label, color) in enumerate(PREDICTORS):
            means = [data[corpus][K][recall_f].mean() for K in K_VALUES]
            ax.bar(x + (j - 1) * width, means, width, label=label, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels([f'K={K}' for K in K_VALUES])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('mean actual recall')
        ax.set_title(CORPUS_LABELS[corpus])
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Scheme B three-way: mean recall of full-population argmax selection, SIFT (128D)')
    plt.tight_layout()
    out_path = out_dir / 'mean_recall_bars.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_home_winner_bars(data: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), squeeze=False)
    x = np.arange(len(K_VALUES))
    width = 0.25
    for col, corpus in enumerate(CORPORA):
        ax = axes[0][col]
        for j, (pred_f, recall_f, home_f, label, color) in enumerate(PREDICTORS):
            rates = [data[corpus][K][home_f].mean() for K in K_VALUES]
            ax.bar(x + (j - 1) * width, rates, width, label=label, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels([f'K={K}' for K in K_VALUES])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('is_home_winner rate')
        ax.set_title(CORPUS_LABELS[corpus])
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Scheme B three-way: rate the argmax selects the near-identical "home" candidate, SIFT (128D)')
    plt.tight_layout()
    out_path = out_dir / 'is_home_winner_bars.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_recall_vs_delta(data: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    centers = _delta_bin_centers()
    plot_x = np.where(centers == 0, centers[1] / 10.0, centers)
    for row, corpus in enumerate(CORPORA):
        for col, K in enumerate(K_VALUES):
            ax = axes[row][col]
            d = data[corpus][K]
            for pred_f, recall_f, home_f, label, color in PREDICTORS:
                means, sems, counts = _binned_mean(d['delta'], d[recall_f], _delta_bin_index, N_DELTA_BINS)
                valid = counts > 0
                ax.errorbar(plot_x[valid], means[valid], yerr=sems[valid], marker='o', markersize=4,
                            label=label, color=color, capsize=2)
            ax.set_xscale('symlog', linthresh=1e-4)
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f'{CORPUS_LABELS[corpus]}, K={K}', fontsize=10)
            ax.set_xlabel('delta')
            if col == 0:
                ax.set_ylabel('mean actual recall')
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
    fig.suptitle('Scheme B three-way: selection quality vs. query-anchor distance, SIFT (128D)')
    plt.tight_layout()
    out_path = out_dir / 'recall_vs_delta.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_calibration_bins(data: dict, out_dir: Path) -> None:
    for corpus in CORPORA:
        fig, axes = plt.subplots(3, len(K_VALUES), figsize=(3.2 * len(K_VALUES), 9), squeeze=False)
        for col, K in enumerate(K_VALUES):
            d = data[corpus][K]
            for row, (pred_f, recall_f, home_f, label, color) in enumerate(PREDICTORS):
                pred = np.clip(d[pred_f], 0.0, 1.0) if pred_f != 'if_fair' else d[pred_f]
                actual = d[recall_f]
                finite = np.isfinite(pred)
                means, sems, counts = _binned_mean(pred[finite], actual[finite], _pred_bin_index, N_PRED_BINS)
                centers = (PRED_BIN_EDGES[:-1] + PRED_BIN_EDGES[1:]) / 2.0
                valid = counts > 0
                ax = axes[row][col]
                ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
                ax.errorbar(centers[valid], means[valid], yerr=sems[valid], marker='o', markersize=3,
                            color=color, capsize=2)
                ax.set_xlim(-0.02, 1.02)
                ax.set_ylim(-0.05, 1.05)
                ax.set_title(f'{label}, K={K}', fontsize=8)
                if row == 2:
                    ax.set_xlabel('predicted (bin center)')
                if col == 0:
                    ax.set_ylabel('mean actual recall')
                ax.grid(alpha=0.3)
        fig.suptitle(f'Scheme B three-way: calibration of the argmax winner, SIFT (128D), {CORPUS_LABELS[corpus]}')
        plt.tight_layout()
        out_path = out_dir / f'calibration_bins_{corpus}.png'
        fig.savefig(out_path, dpi=110, bbox_inches='tight')
        plt.close(fig)
        print(f"wrote {out_path}")


def plot_mean_bias_bars(data: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), squeeze=False)
    x = np.arange(len(K_VALUES))
    width = 0.25
    for col, corpus in enumerate(CORPORA):
        ax = axes[0][col]
        for j, (pred_f, recall_f, home_f, label, color) in enumerate(PREDICTORS):
            biases = []
            for K in K_VALUES:
                d = data[corpus][K]
                pred = np.clip(d[pred_f], 0.0, 1.0) if pred_f != 'if_fair' else d[pred_f]
                finite = np.isfinite(pred)
                biases.append((d[recall_f][finite] - pred[finite]).mean())
            ax.bar(x + (j - 1) * width, biases, width, label=label, color=color)
        ax.axhline(0.0, color='black', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'K={K}' for K in K_VALUES])
        ax.set_ylabel('mean bias (actual - predicted)')
        ax.set_title(CORPUS_LABELS[corpus])
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Scheme B three-way: mean calibration bias of the argmax winner, SIFT (128D)')
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

    plot_mean_recall_bars(data, OUT_DIR)
    plot_home_winner_bars(data, OUT_DIR)
    plot_recall_vs_delta(data, OUT_DIR)
    plot_calibration_bins(data, OUT_DIR)
    plot_mean_bias_bars(data, OUT_DIR)

    print("\n=== Summary ===")
    for corpus in CORPORA:
        for K in K_VALUES:
            d = data[corpus][K]
            vals = ' '.join(f"{label}={d[recall_f].mean():.4f}" for _, recall_f, _, label, _ in PREDICTORS)
            print(f"{corpus} K={K}: {vals}")


if __name__ == '__main__':
    main()
