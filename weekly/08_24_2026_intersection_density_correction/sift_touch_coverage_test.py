#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Tests whether binary cap-touch (IF > 0) alone predicts recall on
real SIFT data, using true offline radii."""

import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datasets.dataloaders import load_fvecs, sift_query_split_indices
from primitives import exact_intersection_fraction_batch
from part2_uniform_baseline_run_experiment import _chord_to_angle, _bruteforce_knn
from sift_synth_sweep_run_experiment import (
    _normalize, precompute_candidates,
    K_VALUES, K_MAX, D, BACKGROUND_PATH,
)
from density_correction_common import _BOUNDARY_EPS

#-------------------------------------------------------------------------------

QUERY_PATH = 'datasets/sift/sift_query.fvecs'
N_CANDIDATES = 2000  # real queries used as cache entries/candidates
N_TEST_QUERIES = 8000  # real queries used as test queries; disjoint from the candidates
SEED = 42

OUT_DIR = Path(__file__).parent / 'results' / 'touch_coverage'
PLOTS_DIR = Path(__file__).parent / 'plots' / 'touch_coverage'

# Log-scale IF bins: touching-pair IF spans ~1e-24 to near 1 on the
# real corpus, so linear [0,1] bins would collapse everything into one.
LOG_BIN_EDGES = [0.0, 1e-20, 1e-10, 1e-5, 1e-2, 1.0]
LOG_BIN_LABELS = ['0', '(0,1e-20]', '(1e-20,1e-10]', '(1e-10,1e-5]', '(1e-5,1e-2]', '(1e-2,1]']

#-------------------------------------------------------------------------------


def _angular_distance_matrix(candidates: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """(n_cand, n_queries) angular distance via the chord-then-angle
    formula, consistent with r_test/r_cache elsewhere in this branch."""
    sims = candidates @ queries.T
    chord = np.sqrt(np.clip(2.0 - 2.0 * sims, 0.0, None))
    return 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))


def _log_bin_index(if_values: np.ndarray) -> np.ndarray:
    """0 for exact zero, 1..5 for the log-scale ranges above."""
    idx = np.zeros(len(if_values), dtype=np.int64)
    for b in range(1, len(LOG_BIN_EDGES)):
        idx[if_values > LOG_BIN_EDGES[b - 1]] = b
    return idx


def evaluate_touching_pairs(queries: np.ndarray, background: np.ndarray, candidates: np.ndarray,
                             cand_data: dict, k_values: list, k_max: int, d: int,
                             chunk_size: int = 200) -> dict:
    """Every touching (query, candidate) pair, per K. Returns
    {K: {'if_value': array, 'recall': array, 'coverage': float}}."""
    n_queries = queries.shape[0]
    dists_test, idx_test = _bruteforce_knn(background, queries, k_max)

    out = {K: {'if_value': [], 'recall': []} for K in k_values}
    covered_any = {K: np.zeros(n_queries, dtype=bool) for K in k_values}

    for start in range(0, n_queries, chunk_size):
        end = min(start + chunk_size, n_queries)
        chunk = queries[start:end]
        delta_qc = _angular_distance_matrix(candidates, chunk)  # (n_cand, chunk_n)

        for K in k_values:
            r_test_all = _chord_to_angle(dists_test[start:end, K - 1])  # (chunk_n,)
            r_cache = cand_data[K]['r_cache']  # (n_cand,)
            resp_idx = cand_data[K]['idx']  # (n_cand, K)

            for j in range(chunk.shape[0]):
                q_global = start + j
                r_test = float(r_test_all[j])
                touching = delta_qc[:, j] <= (r_cache + r_test + _BOUNDARY_EPS)
                if not np.any(touching):
                    continue
                covered_any[K][q_global] = True

                touch_idx = np.where(touching)[0]
                IFs = exact_intersection_fraction_batch(r_cache, r_test, delta_qc[:, j], d)
                true_topk = set(idx_test[q_global, :K].tolist())
                for cand_i in touch_idx:
                    resp = set(resp_idx[cand_i].tolist())
                    recall = len(true_topk & resp) / K
                    out[K]['if_value'].append(float(IFs[cand_i]))
                    out[K]['recall'].append(recall)

    for K in k_values:
        out[K]['if_value'] = np.array(out[K]['if_value'])
        out[K]['recall'] = np.array(out[K]['recall'])
        out[K]['coverage'] = float(covered_any[K].mean())
    return out


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
    all_queries = _normalize(load_fvecs(str(query_path)).astype(np.float64))
    print(f"Loaded {len(all_queries)} real SIFT queries from {query_path}")

    # Disjoint split of the same real query pool into candidates and test queries.
    cand_idx, test_idx = sift_query_split_indices(
        len(all_queries), args.n_candidates, args.n_test_queries, seed=args.seed)
    candidates = all_queries[cand_idx]
    queries = all_queries[test_idx]
    print(f"  candidates (real cache queries): {len(candidates)}, test queries: {len(queries)}")

    cand_data = precompute_candidates(background, candidates)

    print(f"Evaluating {len(queries)} test queries against the candidate pool...")
    results = evaluate_touching_pairs(queries, background, candidates, cand_data, K_VALUES, K_MAX, D)

    report_and_save(results, 'sift', D)
    plot_boxplots(results, 'sift', D)

    print("\nDone.")


if __name__ == '__main__':
    main()
