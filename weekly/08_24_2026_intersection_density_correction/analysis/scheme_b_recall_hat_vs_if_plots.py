#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Plots for the Scheme B full-population argmax comparison: IF-argmax
selection (existing overnight_sift/esci data) vs recall_hat-argmax
selection (sift_scheme_b_recall_hat_run_experiment.py output)."""

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
OLD_PART_DIRS = {'real': 'sift_synth_sweep', 'uniform': 'sift_synth_sweep_on_uniform'}
NEW_PART_DIRS = {'real': 'sift_synth_sweep_recall_hat', 'uniform': 'sift_synth_sweep_on_uniform_recall_hat'}

# Log-scale delta bins - deltas span from ~1e-6 (near self-match) up to
# ~1.5-2 radians; linear bins would collapse almost everything into the
# first bin, the same reasoning behind every other log-delta/log-IF bin
# scheme used elsewhere this session.
DELTA_BIN_EDGES = np.array([0.0, 1e-4, 1e-2, 0.1, 0.3, 0.6, 1.0, 2.0, np.pi])
DELTA_BIN_LABELS = ['0', '(0,1e-4]', '(1e-4,1e-2]', '(1e-2,0.1]', '(0.1,0.3]',
                     '(0.3,0.6]', '(0.6,1.0]', '(1.0,2.0]', '(2.0,pi]']
N_DELTA_BINS = len(DELTA_BIN_LABELS)

OUT_DIR = Path(__file__).parent.parent / 'plots' / 'scheme_b_recall_hat_vs_if'

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


def _binned_mean(delta: np.ndarray, recall: np.ndarray) -> tuple:
    centers = _delta_bin_centers()
    means = np.full(N_DELTA_BINS, np.nan)
    sems = np.full(N_DELTA_BINS, np.nan)
    counts = np.zeros(N_DELTA_BINS, dtype=int)
    idx = _delta_bin_index(delta)
    for b in range(N_DELTA_BINS):
        mask = idx == b
        counts[b] = mask.sum()
        if counts[b] > 0:
            means[b] = recall[mask].mean()
            sems[b] = recall[mask].std(ddof=1) / np.sqrt(counts[b]) if counts[b] > 1 else 0.0
    return centers, means, sems, counts


# Linear 0.05-wide bins on the *predicted* value, for calibration - unlike
# the log-scale delta/IF bins used elsewhere, a calibration plot needs
# even coverage of [0,1] since that's the axis being checked against y=x.
PRED_BIN_EDGES = np.arange(0.0, 1.05, 0.05)
N_PRED_BINS = len(PRED_BIN_EDGES) - 1


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

#-------------------------------------------------------------------------------


def load_all(old_run_dir: Path, new_run_dir: Path) -> dict:
    data = {}
    for corpus in CORPORA:
        data[corpus] = {}
        for K in K_VALUES:
            old = np.load(old_run_dir / OLD_PART_DIRS[corpus] / f'{D}d' / 'raw' / f'K{K}.npz')
            new = np.load(new_run_dir / NEW_PART_DIRS[corpus] / f'{D}d' / 'raw' / f'K{K}.npz')
            data[corpus][K] = {
                'delta': old['delta'], 'recall_if': old['recall'], 'is_home_if': old['is_home_winner'],
                'fraction': old['fraction'],
                'recall_rh': new['recall'], 'is_home_rh': new['is_home_winner'],
                'recall_hat': new['recall_hat'],
            }
    return data


def plot_mean_recall_bars(data: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), squeeze=False)
    x = np.arange(len(K_VALUES))
    width = 0.35
    for col, corpus in enumerate(CORPORA):
        ax = axes[0][col]
        if_means = [data[corpus][K]['recall_if'].mean() for K in K_VALUES]
        rh_means = [data[corpus][K]['recall_rh'].mean() for K in K_VALUES]
        ax.bar(x - width / 2, if_means, width, label='IF-argmax', color='tab:orange')
        ax.bar(x + width / 2, rh_means, width, label='recall_hat-argmax', color='tab:blue')
        ax.set_xticks(x)
        ax.set_xticklabels([f'K={K}' for K in K_VALUES])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('mean actual recall')
        ax.set_title(CORPUS_LABELS[corpus])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Scheme B: mean recall of full-population argmax selection, SIFT (128D)')
    plt.tight_layout()
    out_path = out_dir / 'mean_recall_bars.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_home_winner_bars(data: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), squeeze=False)
    x = np.arange(len(K_VALUES))
    width = 0.35
    for col, corpus in enumerate(CORPORA):
        ax = axes[0][col]
        if_rates = [data[corpus][K]['is_home_if'].mean() for K in K_VALUES]
        rh_rates = [data[corpus][K]['is_home_rh'].mean() for K in K_VALUES]
        ax.bar(x - width / 2, if_rates, width, label='IF-argmax', color='tab:orange')
        ax.bar(x + width / 2, rh_rates, width, label='recall_hat-argmax', color='tab:blue')
        ax.set_xticks(x)
        ax.set_xticklabels([f'K={K}' for K in K_VALUES])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('is_home_winner rate')
        ax.set_title(CORPUS_LABELS[corpus])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Scheme B: rate the argmax selects the near-identical "home" candidate, SIFT (128D)')
    plt.tight_layout()
    out_path = out_dir / 'is_home_winner_bars.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_recall_vs_delta(data: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    centers = _delta_bin_centers()
    plot_x = np.where(centers == 0, centers[1] / 10.0, centers)  # keep bin 0 visible on log axis
    for row, corpus in enumerate(CORPORA):
        for col, K in enumerate(K_VALUES):
            ax = axes[row][col]
            d = data[corpus][K]
            _, means_if, sems_if, counts = _binned_mean(d['delta'], d['recall_if'])
            _, means_rh, sems_rh, _ = _binned_mean(d['delta'], d['recall_rh'])
            valid = counts > 0
            ax.errorbar(plot_x[valid], means_if[valid], yerr=sems_if[valid], marker='o', markersize=4,
                        label='IF-argmax', color='tab:orange', capsize=2)
            ax.errorbar(plot_x[valid], means_rh[valid], yerr=sems_rh[valid], marker='s', markersize=4,
                        label='recall_hat-argmax', color='tab:blue', capsize=2)
            ax.set_xscale('symlog', linthresh=1e-4)
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f'{CORPUS_LABELS[corpus]}, K={K}', fontsize=10)
            ax.set_xlabel('delta (anchor-to-query angular distance)')
            if col == 0:
                ax.set_ylabel('mean actual recall')
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
    fig.suptitle('Scheme B: selection quality vs. query-anchor distance, SIFT (128D)')
    plt.tight_layout()
    out_path = out_dir / 'recall_vs_delta.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_path}")

def plot_calibration_bins(data: dict, out_dir: Path) -> None:
    """Binned calibration check: mean actual recall per 0.05-wide bin of
    the winner's predicted value, against y=x. Averages out the dense
    scatter's visual noise so a systematic bias (or lack of one) is
    unambiguous - unlike the raw scatter, this directly answers "is the
    predictor calibrated," not just "does it rank well.\""""
    for corpus in CORPORA:
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), squeeze=False)
        for col, K in enumerate(K_VALUES):
            d = data[corpus][K]
            recall_hat_clipped = np.clip(d['recall_hat'], 0.0, 1.0)
            for row, (label, pred, actual) in enumerate((
                ('IF', d['fraction'], d['recall_if']),
                ('recall_hat (clipped)', recall_hat_clipped, d['recall_rh']),
            )):
                ax = axes[row][col]
                centers, means, sems, counts = _pred_binned_mean(pred, actual)
                valid = counts > 0
                ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1, label='y=x')
                ax.errorbar(centers[valid], means[valid], yerr=sems[valid], marker='o', markersize=4,
                            color='tab:blue', capsize=2)
                ax.set_xlim(-0.02, 1.02)
                ax.set_ylim(-0.05, 1.05)
                ax.set_title(f'{label}, K={K}', fontsize=10)
                ax.set_xlabel('predicted recall (bin center)')
                if col == 0:
                    ax.set_ylabel('mean actual recall')
                if row == 0 and col == 0:
                    ax.legend(fontsize=8)
                ax.grid(alpha=0.3)
        fig.suptitle(f'Scheme B: calibration (binned mean actual vs. predicted), SIFT (128D), {CORPUS_LABELS[corpus]}')
        plt.tight_layout()
        out_path = out_dir / f'calibration_bins_{corpus}.png'
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"wrote {out_path}")


def plot_bias_vs_delta(data: dict, out_dir: Path) -> None:
    """Calibration bias (mean actual - mean predicted) as a function of
    delta - tests whether the bias is structural (roughly flat across
    delta) or concentrated in a specific regime."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)
    centers = _delta_bin_centers()
    plot_x = np.where(centers == 0, centers[1] / 10.0, centers)
    for col, K in enumerate(K_VALUES):
        ax = axes[0][col]
        for corpus, color in (('real', 'tab:red'), ('uniform', 'tab:green')):
            d = data[corpus][K]
            recall_hat_clipped = np.clip(d['recall_hat'], 0.0, 1.0)
            _, mean_actual, _, counts = _binned_mean(d['delta'], d['recall_rh'])
            _, mean_pred, _, _ = _binned_mean(d['delta'], recall_hat_clipped)
            bias = mean_actual - mean_pred
            valid = counts > 0
            ax.plot(plot_x[valid], bias[valid], marker='o', markersize=4, color=color,
                    label=f'{corpus} corpus')
        ax.axhline(0.0, linestyle='--', color='gray', linewidth=1)
        ax.set_xscale('symlog', linthresh=1e-4)
        ax.set_ylim(-0.1, 0.6)
        ax.set_title(f'K={K}', fontsize=10)
        ax.set_xlabel('delta (anchor-to-query angular distance)')
        if col == 0:
            ax.set_ylabel('calibration bias (mean actual - mean predicted recall_hat)')
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle('Scheme B: recall_hat calibration bias vs. query-anchor distance, SIFT (128D)')
    plt.tight_layout()
    out_path = out_dir / 'bias_vs_delta.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_predicted_vs_actual(data: dict, out_dir: Path) -> None:
    """Scatter of the winning candidate's predicted recall (IF, or
    recall_hat clipped to [0,1]) against its actual recall, against
    y=x - one file per corpus, rows=predictor, cols=K."""
    for corpus in CORPORA:
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), squeeze=False)
        for col, K in enumerate(K_VALUES):
            d = data[corpus][K]
            recall_hat_clipped = np.clip(d['recall_hat'], 0.0, 1.0)
            for row, (label, pred, actual) in enumerate((
                ('IF', d['fraction'], d['recall_if']),
                ('recall_hat (clipped)', recall_hat_clipped, d['recall_rh']),
            )):
                ax = axes[row][col]
                ax.scatter(pred, actual, s=4, alpha=0.1, color='tab:blue', linewidths=0)
                ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
                ax.set_xlim(-0.02, 1.02)
                ax.set_ylim(-0.05, 1.05)
                ax.set_title(f'{label}, K={K}', fontsize=10)
                ax.set_xlabel('predicted recall')
                if col == 0:
                    ax.set_ylabel('actual recall')
                ax.grid(alpha=0.3)
        fig.suptitle(f'Scheme B: predicted vs. actual recall of the argmax winner, SIFT (128D), {CORPUS_LABELS[corpus]}')
        plt.tight_layout()
        out_path = out_dir / f'predicted_vs_actual_{corpus}.png'
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"wrote {out_path}")

#-------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-run-dir', type=str, default=str(Path(__file__).parent.parent / 'results' / 'overnight_sift'))
    parser.add_argument('--new-run-dir', type=str, required=True)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_all(Path(args.old_run_dir), Path(args.new_run_dir))

    plot_mean_recall_bars(data, OUT_DIR)
    plot_home_winner_bars(data, OUT_DIR)
    plot_recall_vs_delta(data, OUT_DIR)
    plot_predicted_vs_actual(data, OUT_DIR)
    plot_calibration_bins(data, OUT_DIR)
    plot_bias_vs_delta(data, OUT_DIR)

    print("\n=== Summary ===")
    for corpus in CORPORA:
        for K in K_VALUES:
            d = data[corpus][K]
            print(f"{corpus} K={K}: IF={d['recall_if'].mean():.4f}, recall_hat={d['recall_rh'].mean():.4f}, "
                  f"delta={d['recall_rh'].mean() - d['recall_if'].mean():+.4f}")


if __name__ == '__main__':
    main()
