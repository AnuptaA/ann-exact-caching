#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Tests whether binary cap-touch (IF > 0) alone predicts recall on
real ESCI data, using true offline radii."""

import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from primitives import exact_intersection_fraction_batch
from part2_uniform_baseline_run_experiment import _chord_to_angle, _bruteforce_knn
from esci_synth_sweep_run_experiment import (
    _normalize, precompute_candidates,
    K_VALUES, K_MAX, D, BACKGROUND_PATH,
)
from density_correction_common import _BOUNDARY_EPS
from sift_touch_coverage_test import (
    _angular_distance_matrix, _log_bin_index, evaluate_touching_pairs,
    LOG_BIN_EDGES, LOG_BIN_LABELS,
)

#-------------------------------------------------------------------------------

QUERY_PATH = 'datasets/esci/queries.npz'
N_CANDIDATES = 2000  # real queries used as cache entries/candidates
N_TEST_QUERIES = 5000  # real queries used as test queries; disjoint from the candidates
SEED = 42

OUT_DIR = Path(__file__).parent / 'results' / 'touch_coverage'
PLOTS_DIR = Path(__file__).parent / 'plots' / 'touch_coverage'

#-------------------------------------------------------------------------------


def report_and_save(results: dict, dataset_label: str, d: int) -> None:
    print(f"\n=== Touch coverage/recall, {dataset_label} (d={d}) ===")
    for K in K_VALUES:
        cov = results[K]['coverage']
        n_pairs = len(results[K]['if_value'])
        print(f"  K={K}: coverage={cov:.4f}, touching pairs={n_pairs}")
        if n_pairs == 0:
            continue
        bin_idx = _log_bin_index(results[K]['if_value'])
        for b, label in enumerate(LOG_BIN_LABELS):
            mask = bin_idx == b
            n = int(mask.sum())
            if n == 0:
                continue
            recall_b = results[K]['recall'][mask]
            print(f"    IF bin {label:>16}: n={n}, recall mean={recall_b.mean():.4f}, "
                  f"median={np.median(recall_b):.4f}")

    out_dir = OUT_DIR / dataset_label / f'{d}d'
    out_dir.mkdir(parents=True, exist_ok=True)
    for K in K_VALUES:
        path = out_dir / f'K{K}.npz'
        np.savez_compressed(path, if_value=results[K]['if_value'], recall=results[K]['recall'],
                             coverage=results[K]['coverage'])
        print(f"    wrote {path}")


def plot_boxplots(results: dict, dataset_label: str, d: int) -> None:
    plots_dir = PLOTS_DIR / dataset_label
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(K_VALUES), figsize=(4.5 * len(K_VALUES), 4.5), squeeze=False)
    for col, K in enumerate(K_VALUES):
        ax = axes[0, col]
        if_values = results[K]['if_value']
        recall = results[K]['recall']
        if len(if_values) == 0:
            ax.set_title(f'K={K} (no touching pairs)')
            continue
        bin_idx = _log_bin_index(if_values)
        box_data, box_labels = [], []
        for b, label in enumerate(LOG_BIN_LABELS):
            mask = bin_idx == b
            if mask.sum() == 0:
                continue
            box_data.append(recall[mask])
            box_labels.append(f'{label}\n(n={int(mask.sum())})')
        ax.boxplot(box_data, tick_labels=box_labels, showfliers=False)
        ax.set_title(f'K={K}, coverage={results[K]["coverage"]:.3f}')
        ax.set_ylabel('recall')
        ax.tick_params(axis='x', labelrotation=45, labelsize=7)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle(f'{dataset_label.upper()} (d={d}): recall by IF magnitude, all touching pairs '
                 '(true radii, offline-only)')
    plt.tight_layout()
    out_path = plots_dir / f'{dataset_label}_{d}d_touch_coverage_boxplot.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")

#-------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-candidates', type=int, default=N_CANDIDATES, dest='n_candidates')
    parser.add_argument('--n-test-queries', type=int, default=N_TEST_QUERIES, dest='n_test_queries')
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent.parent
    bg_path = repo_root / BACKGROUND_PATH
    background = _normalize(np.load(bg_path).astype(np.float64))

    query_path = repo_root / QUERY_PATH
    query_data = np.load(query_path)
    all_queries = _normalize(query_data['query_vectors'].astype(np.float64))
    print(f"Loaded {len(all_queries)} real ESCI query embeddings from {query_path}")

    # Disjoint split of the same real query pool into candidates and test queries.
    total_needed = args.n_candidates + args.n_test_queries
    if total_needed > len(all_queries):
        raise ValueError(f"n_candidates + n_test_queries ({total_needed}) exceeds "
                          f"available queries ({len(all_queries)})")
    perm = np.random.default_rng(args.seed).permutation(len(all_queries))
    candidates = all_queries[perm[:args.n_candidates]]
    queries = all_queries[perm[args.n_candidates:total_needed]]
    print(f"  candidates (real cache queries): {len(candidates)}, test queries: {len(queries)}")

    cand_data = precompute_candidates(background, candidates)

    print(f"Evaluating {len(queries)} test queries against the candidate pool...")
    results = evaluate_touching_pairs(queries, background, candidates, cand_data, K_VALUES, K_MAX, D)

    report_and_save(results, 'esci', D)
    plot_boxplots(results, 'esci', D)

    print("\nDone.")


if __name__ == '__main__':
    main()
