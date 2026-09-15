#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Head-to-head comparison of IF vs recall_hat=K/N_hat as recall
predictors on Scheme A (sift_part2_real_data_run_experiment.py), reusing
the existing OLS/R2/CCC fit methodology from
scheme_corpus_comparison_plot_results.py."""

import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'common'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scheme_corpus_comparison_plot_results import (
    load_cell, analyze, _log_bin_index, _log_bin_centers, N_IF_BINS, IF_BIN_LABELS,
)
from part2_uniform_baseline_run_experiment import K_VALUES

#-------------------------------------------------------------------------------

D = 128
OLD_RUN_DIR = Path(__file__).parent.parent / 'results' / '20260829_193502'
OLD_PART_DIRS = {'real': 'sift_part2_real_data', 'uniform': 'sift_part2_real_anchor_on_uniform'}
NEW_PART_DIRS = {'real': 'sift_part2_real_data_recall_hat', 'uniform': 'sift_part2_real_anchor_on_uniform_recall_hat'}
CORPUS_LABELS = {'real': 'real corpus', 'uniform': 'uniform corpus (same anchor/queries)'}

OUT_DIR = Path(__file__).parent.parent / 'plots' / 'scheme_a_recall_hat_vs_if'

#-------------------------------------------------------------------------------


def sanity_check(old_data: dict, new_data: dict, corpus: str) -> None:
    print(f"\n  Sanity check ({corpus}): max|recall_new - recall_old| per K")
    for K in K_VALUES:
        old_recall = old_data[K]['recall']
        new_recall = new_data[K]['recall']
        if len(old_recall) != len(new_recall):
            print(f"    K={K}: row count mismatch (old={len(old_recall)}, new={len(new_recall)})")
            continue
        max_diff = float(np.max(np.abs(old_recall - new_recall)))
        print(f"    K={K}: max_diff={max_diff:.6f}")


def clipped_copy(data: dict) -> dict:
    out = {}
    for K, arrays in data.items():
        out[K] = {'fraction': np.clip(arrays['fraction'], 0.0, 1.0), 'recall': arrays['recall']}
    return out


def print_head_to_head(corpus: str, if_results: dict, rh_raw_results: dict, rh_clipped_results: dict) -> None:
    print(f"\n=== {CORPUS_LABELS[corpus]} ===")
    header = f"{'K':>5} | {'IF raw_R2':>10} {'IF raw_CCC':>11} {'IF bal_R2':>10} {'IF bal_CCC':>11} | " \
             f"{'rh raw_R2':>10} {'rh raw_CCC':>11} {'rh bal_R2':>10} {'rh bal_CCC':>11} | " \
             f"{'rhc bal_R2':>11} {'rhc bal_CCC':>12}"
    print(header)
    print('-' * len(header))
    for K in K_VALUES:
        a, b, c = if_results[K], rh_raw_results[K], rh_clipped_results[K]
        print(f"{K:>5} | {a['raw_r2']:>10.3f} {a['raw_ccc']:>11.3f} {a['bal_r2']:>10.3f} {a['bal_ccc']:>11.3f} | "
              f"{b['raw_r2']:>10.3f} {b['raw_ccc']:>11.3f} {b['bal_r2']:>10.3f} {b['bal_ccc']:>11.3f} | "
              f"{c['bal_r2']:>11.3f} {c['bal_ccc']:>12.3f}")


def plot_comparison(corpus: str, if_results: dict, rh_results: dict, out_dir: Path) -> None:
    for K in K_VALUES:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), squeeze=False)
        for col, (label, r) in enumerate((('IF', if_results[K]), ('recall_hat (clipped)', rh_results[K]))):
            ax = axes[0][col]
            ax.scatter(r['bal_fraction'], r['bal_recall'], s=4, alpha=0.15, color='tab:blue', linewidths=0)
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f"{label}\nbalanced R2={r['bal_r2']:.3f}, CCC={r['bal_ccc']:.3f} (n/bin={r['bal_n_per_bin']})",
                         fontsize=9)
            ax.set_xlabel('predicted recall')
            if col == 0:
                ax.set_ylabel('actual recall')
            ax.grid(alpha=0.3)
        fig.suptitle(f'Scheme A, SIFT ({D}D), K={K}, {CORPUS_LABELS[corpus]}')
        plt.tight_layout()
        out_path = out_dir / f'sift_{D}d_K{K}_{corpus}_if_vs_recall_hat.png'
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"    wrote {out_path}")

#-------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-run-dir', type=str, default=str(OLD_RUN_DIR), dest='old_run_dir')
    parser.add_argument('--new-run-dir', type=str, required=True, dest='new_run_dir')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    old_run_dir = Path(args.old_run_dir)
    new_run_dir = Path(args.new_run_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    summary_bal_r2 = {'IF': [], 'recall_hat_clipped': []}
    for corpus in ('real', 'uniform'):
        old_data = load_cell(old_run_dir, OLD_PART_DIRS[corpus], D, K_VALUES, 'cap_fraction')
        new_data_raw = load_cell(new_run_dir, NEW_PART_DIRS[corpus], D, K_VALUES, 'recall_hat')

        sanity_check(old_data, new_data_raw, corpus)

        new_data_clipped = clipped_copy(new_data_raw)

        if_results = analyze(old_data, rng)
        rh_raw_results = analyze(new_data_raw, rng)
        rh_clipped_results = analyze(new_data_clipped, rng)

        print_head_to_head(corpus, if_results, rh_raw_results, rh_clipped_results)
        plot_comparison(corpus, if_results, rh_clipped_results, OUT_DIR)

        for K in K_VALUES:
            summary_bal_r2['IF'].append(if_results[K]['bal_r2'])
            summary_bal_r2['recall_hat_clipped'].append(rh_clipped_results[K]['bal_r2'])

    mean_if_r2 = np.nanmean(summary_bal_r2['IF'])
    mean_rh_r2 = np.nanmean(summary_bal_r2['recall_hat_clipped'])
    print(f"\n=== Bottom line (mean balanced R2 across corpora x K) ===")
    print(f"  IF:                  {mean_if_r2:.4f}")
    print(f"  recall_hat (clipped): {mean_rh_r2:.4f}")
    if mean_rh_r2 > mean_if_r2:
        print(f"  recall_hat beats IF by {mean_rh_r2 - mean_if_r2:.4f} (balanced R2)")
    else:
        print(f"  recall_hat does NOT beat IF (delta={mean_rh_r2 - mean_if_r2:.4f}) - "
              f"investigate before trusting this result.")


if __name__ == '__main__':
    main()
