#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Analysis of the setup (a) overnight run (uniform cache candidates,
real-query test set) on SIFT and ESCI. Checks the realized IF
distribution before choosing bin edges, produces recall-vs-IF plots,
and compares agreement-with-y=x against the anchor-perturbation
design's real-corpus results (Scheme A/B)."""

import re
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'common'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scheme_corpus_comparison_plot_results import _ols_fit, _concordance_correlation_coefficient
from overnight_analysis import load_cell as load_scheme_b_cell, fine_balance_by_if_bin

#-------------------------------------------------------------------------------

HERE = Path(__file__).parent.parent
SUMMARY_LOG = HERE / 'overnight_setup_a_summary.log'
PLOTS_DIR = HERE / 'plots' / 'setup_a'

K_VALUES = [25, 50, 100]

DATASETS = {
    'sift': {'d': 128, 'run_dir': HERE / 'results' / 'overnight_setup_a_sift', 'part_dir': 'sift_setup_a',
             'scheme_b_run_dir': HERE / 'results' / 'overnight_sift', 'scheme_b_part_dir': 'sift_synth_sweep'},
    'esci': {'d': 384, 'run_dir': HERE / 'results' / 'overnight_setup_a_esci', 'part_dir': 'esci_setup_a',
              'scheme_b_run_dir': HERE / 'results' / 'overnight_esci', 'scheme_b_part_dir': 'esci_synth_sweep'},
}

N_QUANTILE_BINS = 10

#-------------------------------------------------------------------------------


def load_cell(run_dir: Path, part_dir: str, d: int, K: int) -> dict:
    path = run_dir / part_dir / f'{d}d' / 'raw' / f'K{K}.npz'
    if not path.exists():
        return None
    arr = np.load(path)
    return {'fraction': arr['fraction'], 'recall': arr['recall']}


def quantile_bin_stats(fraction: np.ndarray, recall: np.ndarray, n_bins: int = N_QUANTILE_BINS) -> dict:
    """Equal-count bins by fraction quantile, matched to the actual
    realized data range rather than an assumed log-scale span."""
    edges = np.quantile(fraction, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] -= 1e-12
    bin_idx = np.digitize(fraction, edges[1:-1], right=True)

    centers = np.full(n_bins, np.nan)
    means = np.full(n_bins, np.nan)
    sems = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = bin_idx == b
        counts[b] = mask.sum()
        if counts[b] > 0:
            centers[b] = fraction[mask].mean()
            means[b] = recall[mask].mean()
            sems[b] = recall[mask].std(ddof=1) / np.sqrt(counts[b]) if counts[b] > 1 else 0.0
    return {'edges': edges, 'centers': centers, 'means': means, 'sems': sems, 'counts': counts}


def analyze_cell(fraction: np.ndarray, recall: np.ndarray) -> dict:
    finite = np.isfinite(fraction) & np.isfinite(recall)
    fraction, recall = fraction[finite], recall[finite]

    raw_slope, raw_intercept, raw_r2 = _ols_fit(fraction, recall)
    raw_ccc = _concordance_correlation_coefficient(fraction, recall)
    binned = quantile_bin_stats(fraction, recall)

    return {
        'n': len(fraction), 'fraction': fraction, 'recall': recall,
        'raw_r2': raw_r2, 'raw_ccc': raw_ccc, 'raw_slope': raw_slope, 'raw_intercept': raw_intercept,
        'binned': binned,
    }


def analyze_scheme_b_cell(fraction: np.ndarray, recall: np.ndarray, rng: np.random.Generator) -> dict:
    """Scheme B real-corpus comparison numbers: raw (all argmax
    winners) and fine-bin-balanced, matching overnight_analysis.py's
    convention so the CCC/R2 values are directly comparable."""
    finite = np.isfinite(fraction) & np.isfinite(recall)
    fraction, recall = fraction[finite], recall[finite]
    raw_r2, raw_ccc = _ols_fit(fraction, recall)[2], _concordance_correlation_coefficient(fraction, recall)

    bal_fraction, bal_recall, n_per_bin = fine_balance_by_if_bin(fraction, recall, rng)
    if len(bal_fraction) >= 2:
        bal_r2 = _ols_fit(bal_fraction, bal_recall)[2]
        bal_ccc = _concordance_correlation_coefficient(bal_fraction, bal_recall)
    else:
        bal_r2 = bal_ccc = float('nan')
    return {'raw_r2': raw_r2, 'raw_ccc': raw_ccc, 'bal_r2': bal_r2, 'bal_ccc': bal_ccc, 'bal_n_per_bin': n_per_bin}

#-------------------------------------------------------------------------------


def print_distribution_check(dataset: str, K: int, fraction: np.ndarray) -> None:
    pct = np.percentile(fraction, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    labels = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    pct_str = ', '.join(f'p{p}={v:.4f}' for p, v in zip(labels, pct))
    print(f"  {dataset} K={K}: n={len(fraction)}, realized fraction range: {pct_str}")


def plot_dataset_row(axes_row: np.ndarray, dataset: str, results: dict) -> None:
    for col, K in enumerate(K_VALUES):
        ax = axes_row[col]
        cell = results[K]
        fraction, recall = cell['fraction'], cell['recall']
        binned = cell['binned']

        ax.scatter(fraction, recall, s=3, alpha=0.05, color='tab:blue', label='raw')
        mask = binned['counts'] > 0
        ax.errorbar(binned['centers'][mask], binned['means'][mask], yerr=binned['sems'][mask],
                     fmt='o-', color='tab:orange', capsize=3, label='quantile-binned mean')

        lo, hi = fraction.min(), fraction.max()
        ref = np.linspace(lo, hi, 100)
        ax.plot(ref, ref, 'k--', linewidth=1, label='y=x')

        ax.set_title(f'{dataset.upper()}, K={K}\nraw R2={cell["raw_r2"]:.3f}, CCC={cell["raw_ccc"]:.3f}')
        ax.set_xlabel('IF (best candidate)')
        ax.set_ylabel('recall')
        ax.set_xlim(lo * 0.95, hi * 1.05)
        if col == 0:
            ax.legend(fontsize=7, loc='upper left')


def make_plots(all_results: dict) -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(DATASETS), len(K_VALUES),
                              figsize=(4.5 * len(K_VALUES), 4.2 * len(DATASETS)), squeeze=False)
    for row, dataset in enumerate(DATASETS):
        plot_dataset_row(axes[row], dataset, all_results[dataset])
    fig.suptitle('Setup (a): uniform cache candidates, real-query test set. Recall vs IF.')
    plt.tight_layout()
    out_path = PLOTS_DIR / 'setup_a_recall_vs_if.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out_path


def parse_summary_log() -> str:
    if not SUMMARY_LOG.exists():
        return "  (no summary log found)"
    return SUMMARY_LOG.read_text().strip()

#-------------------------------------------------------------------------------


def main() -> None:
    rng = np.random.default_rng(42)
    all_results = {}
    all_scheme_b = {}

    print("=== Setup (a) analysis ===\n")
    for dataset, cfg in DATASETS.items():
        print(f"--- {dataset} (d={cfg['d']}) ---")
        results = {}
        scheme_b_results = {}
        for K in K_VALUES:
            cell = load_cell(cfg['run_dir'], cfg['part_dir'], cfg['d'], K)
            if cell is None:
                print(f"  {dataset} K={K}: MISSING (run did not produce this file)")
                continue
            print_distribution_check(dataset, K, cell['fraction'])
            results[K] = analyze_cell(cell['fraction'], cell['recall'])
            print(f"    raw R2={results[K]['raw_r2']:.4f}, raw CCC={results[K]['raw_ccc']:.4f}, "
                  f"slope={results[K]['raw_slope']:.4f}, intercept={results[K]['raw_intercept']:.4f}")

            sb_cell = load_scheme_b_cell(cfg['scheme_b_run_dir'], cfg['scheme_b_part_dir'], cfg['d'], K)
            if sb_cell is not None:
                scheme_b_results[K] = analyze_scheme_b_cell(sb_cell['fraction'], sb_cell['recall'], rng)
                print(f"    Scheme B (anchor-perturbation, real corpus) comparison: "
                      f"raw R2={scheme_b_results[K]['raw_r2']:.4f}, raw CCC={scheme_b_results[K]['raw_ccc']:.4f}, "
                      f"balanced R2={scheme_b_results[K]['bal_r2']:.4f}, balanced CCC={scheme_b_results[K]['bal_ccc']:.4f} "
                      f"(n/bin={scheme_b_results[K]['bal_n_per_bin']})")
            else:
                print(f"    Scheme B comparison data missing for K={K}")
        all_results[dataset] = results
        all_scheme_b[dataset] = scheme_b_results
        print()

    plot_path = make_plots(all_results)
    print(f"Wrote {plot_path}")

    print("\n=== Prediction check: does setup (a) track y=x more closely than Scheme B? ===")
    for dataset in DATASETS:
        for K in K_VALUES:
            if K not in all_results[dataset] or K not in all_scheme_b[dataset]:
                continue
            setup_a_ccc = all_results[dataset][K]['raw_ccc']
            scheme_b_ccc = all_scheme_b[dataset][K]['raw_ccc']
            scheme_b_bal_ccc = all_scheme_b[dataset][K]['bal_ccc']
            closer = setup_a_ccc > max(scheme_b_ccc, scheme_b_bal_ccc)
            verdict = "CLOSER to y=x" if closer else "NOT closer to y=x"
            print(f"  {dataset} K={K}: setup(a) CCC={setup_a_ccc:.4f} vs Scheme B raw CCC={scheme_b_ccc:.4f} / "
                  f"balanced CCC={scheme_b_bal_ccc:.4f} -> {verdict}")

    print("\n=== Timing summary (overnight_setup_a_summary.log) ===")
    print(parse_summary_log())

    print("\nDone.")


if __name__ == '__main__':
    main()
