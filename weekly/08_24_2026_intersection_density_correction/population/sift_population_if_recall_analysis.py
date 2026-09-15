#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Analysis of the pooled population-selected IF-vs-recall dataset:
realized-distribution check, primary recall-vs-IF plot, IF=1 vs IF<1
fraction, comparison against the existing Scheme B SIFT results, and the
secondary inflation-factor plot."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'common'))
sys.path.append(str(Path(__file__).parent.parent / 'analysis'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scheme_corpus_comparison_plot_results import _ols_fit, _concordance_correlation_coefficient
from overnight_analysis import load_cell as load_scheme_b_cell, fine_balance_by_if_bin

#-------------------------------------------------------------------------------

HERE = Path(__file__).parent.parent
POOLED_PATH = HERE / 'results' / 'sift_population_if_recall' / 'pooled.npz'
PLOTS_DIR = HERE / 'plots' / 'sift_population_if_recall'

SCHEME_B_RUN_DIR = HERE / 'results' / 'overnight_sift'
SCHEME_B_PART_DIR = 'sift_synth_sweep'
SCHEME_B_D = 128
SCHEME_B_K = 100

# Everything below 0.01 behaves as a single "near zero" mass (recall stays
# ~0 there regardless of how many decades it spans), so it collapses into
# one bin instead of the log-scale decades used elsewhere in this branch.
# Above that, bins are equal-width (0.1) since that's where IF and recall
# actually vary; exact 0 and exact 1 get their own bins.
BIN_EDGES = [0.0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
BIN_LABELS = ['0', '(0,0.01]', '(0.01,0.1]', '(0.1,0.2]', '(0.2,0.3]', '(0.3,0.4]',
              '(0.4,0.5]', '(0.5,0.6]', '(0.6,0.7]', '(0.7,0.8]', '(0.8,0.9]', '(0.9,1)']
N_BINS = len(BIN_LABELS) + 1  # + exact-1 bin

#-------------------------------------------------------------------------------


def bin_index(values: np.ndarray) -> np.ndarray:
    """0 for exact zero, 1..len(BIN_EDGES)-1 for the ranges above,
    last index for exact 1."""
    idx = np.zeros(len(values), dtype=np.int64)
    for b in range(1, len(BIN_EDGES)):
        idx[values > BIN_EDGES[b - 1]] = b
    idx[values >= 1.0 - 1e-9] = N_BINS - 1
    return idx


def binned_stats(fraction: np.ndarray, recall: np.ndarray) -> tuple:
    """Bin center is the median IF within the bin, not the edge midpoint:
    more honest for the collapsed near-zero bin, whose contents are
    heavily skewed rather than uniformly spread across its nominal range."""
    if_centers = np.full(N_BINS, np.nan)
    means = np.full(N_BINS, np.nan)
    q25 = np.full(N_BINS, np.nan)
    q75 = np.full(N_BINS, np.nan)
    counts = np.zeros(N_BINS, dtype=int)
    idx = bin_index(fraction)
    for b in range(N_BINS):
        mask = idx == b
        counts[b] = mask.sum()
        if counts[b] > 0:
            if_centers[b] = np.median(fraction[mask])
            means[b] = recall[mask].mean()
            q25[b] = np.percentile(recall[mask], 25)
            q75[b] = np.percentile(recall[mask], 75)
    return if_centers, means, q25, q75, counts

#-------------------------------------------------------------------------------


def plot_primary(fraction: np.ndarray, recall: np.ndarray) -> Path:
    """Categorical bin-index x-axis: equal visual width per bin regardless
    of its underlying IF value. A continuous log-x axis here would spend
    almost its entire width on the 1e-100..1e-5 range, where recall is
    uniformly ~0, squeezing the actually-informative transition near IF=1
    into a sliver. The IF bin center is plotted as its own line so
    agreement with y=x is read as the two lines overlapping."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    centers, means, q25, q75, counts = binned_stats(fraction, recall)
    mask = counts > 0
    positions = np.arange(N_BINS)[mask]
    labels = [BIN_LABELS[b] if b < len(BIN_LABELS) else '1' for b in np.arange(N_BINS)[mask]]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(positions, means[mask], 'o-', color='tab:blue', label='mean recall')
    ax.fill_between(positions, q25[mask], q75[mask], color='tab:blue', alpha=0.2, label='recall IQR')
    ax.plot(positions, centers[mask], 'k--', marker='x',
            label='y=x (each bin\'s own median IF, plotted at that bin\'s position)')

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('IF bin')
    ax.set_ylabel('recall')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('SIFT, population-selected cache (pooled across 11 budgets)\nrecall vs IF, equal-width bins')
    ax.legend(fontsize=8)
    plt.tight_layout()
    out_path = PLOTS_DIR / 'primary_recall_vs_if.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_inflation_divergence(fraction: np.ndarray, recall: np.ndarray, inflation: np.ndarray) -> Path:
    divergence = recall - fraction
    edges = np.quantile(inflation, np.linspace(0.0, 1.0, 9))
    edges[0] -= 1e-9
    idx = np.digitize(inflation, edges[1:-1], right=True)

    centers, means, sems = [], [], []
    for b in range(len(edges) - 1):
        mask = idx == b
        if mask.sum() == 0:
            continue
        centers.append(inflation[mask].mean())
        means.append(divergence[mask].mean())
        sems.append(divergence[mask].std(ddof=1) / np.sqrt(mask.sum()) if mask.sum() > 1 else 0.0)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(centers, means, yerr=sems, fmt='o-', color='tab:purple', capsize=3)
    ax.axhline(0.0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('inflation factor (r_cache_used / bare K-NN radius)')
    ax.set_ylabel('recall - IF (divergence)')
    ax.set_title('SIFT population-selected cache: divergence vs inflation factor')
    plt.tight_layout()
    out_path = PLOTS_DIR / 'secondary_divergence_vs_inflation.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out_path

#-------------------------------------------------------------------------------


def compare_against_scheme_b(fraction: np.ndarray, recall: np.ndarray, rng: np.random.Generator) -> None:
    finite = np.isfinite(fraction) & np.isfinite(recall)
    fraction, recall = fraction[finite], recall[finite]
    pop_r2, pop_ccc = _ols_fit(fraction, recall)[2], _concordance_correlation_coefficient(fraction, recall)
    print(f"Population-selected pooled: raw R2={pop_r2:.4f}, raw CCC={pop_ccc:.4f}, n={len(fraction)}")

    sb_cell = load_scheme_b_cell(SCHEME_B_RUN_DIR, SCHEME_B_PART_DIR, SCHEME_B_D, SCHEME_B_K)
    if sb_cell is None:
        print("Scheme B comparison data not found.")
        return
    sb_fraction, sb_recall = sb_cell['fraction'], sb_cell['recall']
    sb_raw_r2, sb_raw_ccc = _ols_fit(sb_fraction, sb_recall)[2], _concordance_correlation_coefficient(sb_fraction, sb_recall)
    bal_fraction, bal_recall, n_per_bin = fine_balance_by_if_bin(sb_fraction, sb_recall, rng)
    sb_bal_r2, sb_bal_ccc = _ols_fit(bal_fraction, bal_recall)[2], _concordance_correlation_coefficient(bal_fraction, bal_recall)
    print(f"Scheme B (bare K-NN, anchor-perturbation, SIFT K={SCHEME_B_K}): "
          f"raw R2={sb_raw_r2:.4f}, raw CCC={sb_raw_ccc:.4f}, "
          f"balanced R2={sb_bal_r2:.4f}, balanced CCC={sb_bal_ccc:.4f} (n/bin={n_per_bin})")

    closer = pop_ccc > max(sb_raw_ccc, sb_bal_ccc)
    print(f"Verdict: population-selected divergence from y=x is "
          f"{'SMALLER (closer to y=x)' if closer else 'NOT smaller'} than Scheme B's, "
          f"comparing CCC {pop_ccc:.4f} vs {sb_raw_ccc:.4f}/{sb_bal_ccc:.4f}")

#-------------------------------------------------------------------------------


def main() -> None:
    data = np.load(POOLED_PATH)
    fraction, recall, inflation = data['fraction'], data['recall'], data['inflation_factor']
    n = len(fraction)
    print(f"Loaded {n} pooled (IF, recall, inflation_factor) points across "
          f"{len(np.unique(data['budget_fraction']))} budgets.\n")

    print("=== Realized IF distribution (before choosing bins) ===")
    print("percentiles:", np.percentile(fraction, [0, 1, 5, 25, 50, 75, 95, 99, 100]))
    # Exact-zero fast path returns bit-exact 0.0; near-1 values can overshoot
    # slightly due to floating-point error in the quadrature/ratio paths.
    frac_1 = (fraction >= 1.0 - 1e-9).mean()
    frac_0 = (fraction <= 0.0).mean()
    frac_mid = 1.0 - frac_1 - frac_0
    print(f"fraction of queries at IF=1 exactly: {frac_1:.4f}")
    print(f"fraction of queries at IF=0 exactly: {frac_0:.4f}")
    print(f"fraction of queries with genuine partial overlap (0<IF<1): {frac_mid:.4f}\n")

    primary_path = plot_primary(fraction, recall)
    print(f"Wrote {primary_path}")

    print("\n=== Comparison against Scheme B (bare K-NN) ===")
    rng = np.random.default_rng(42)
    compare_against_scheme_b(fraction, recall, rng)

    secondary_path = plot_inflation_divergence(fraction, recall, inflation)
    print(f"\nWrote {secondary_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
