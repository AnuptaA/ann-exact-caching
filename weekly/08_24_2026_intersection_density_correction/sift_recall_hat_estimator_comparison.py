#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Compares IF against two LSH-based recall_hat = K/N_q_hat estimators
(raw and exclusion-corrected) as predictors of true recall, SIFT only,
K in {25, 50}, real and uniform corpora."""

import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from primitives import exact_intersection_fraction_batch
import sift_synth_sweep_run_experiment as sw
from density_correction_common import (
    _angular_distance_consistent, covered_mask, build_lsh_tables, estimate_range_count,
)
from scheme_corpus_comparison_plot_results import (
    balance_by_if_bin, _ols_fit, _concordance_correlation_coefficient, _fixed_bin_stats,
)

#-------------------------------------------------------------------------------

K_VALUES = [25, 50]
K_MAX = 50
D = sw.D
CORPUS_SIZE = 50000  # matches base_vectors_50k.npy
L_TABLES = 16
N_ANCHORS_FAST = 25
N_DIRECTIONS_FAST = 5
SEED = 42

PLOTS_DIR = Path(__file__).parent / 'plots' / 'recall_hat_estimator'

#-------------------------------------------------------------------------------


def k_bits_for(k: int) -> int:
    return max(1, round(math.log2(CORPUS_SIZE / k)))


def run_cell(background: np.ndarray, uniform_corpus: np.ndarray, n_anchors: int, n_directions: int,
             seed: int) -> dict:
    """Generates fresh Scheme B candidates/queries and scores IF/recall_hat
    (raw+excl)/true_recall/true_N for the IF-argmax-selected candidate,
    per K, per corpus. Returns {corpus: {K: {field: array}}}."""
    sw.K_VALUES = K_VALUES
    sw.K_MAX = K_MAX

    rng = np.random.default_rng(seed)
    anchor_idx = rng.choice(len(background), size=n_anchors, replace=False)
    anchors = background[anchor_idx]

    print("  building candidate swarm...")
    candidates, home_indices = sw.build_candidate_pool(background, anchors, rng, D)
    print(f"  candidate pool size: {len(candidates)}")

    cand_data = {'real': sw.precompute_candidates(background, candidates),
                 'uniform': sw.precompute_candidates(uniform_corpus, candidates)}
    corpora = {'real': background, 'uniform': uniform_corpus}

    print("  building LSH tables...")
    lsh_tables = {}
    for corpus_name, corpus in corpora.items():
        lsh_tables[corpus_name] = {}
        for K in K_VALUES:
            lsh_tables[corpus_name][K] = build_lsh_tables(
                corpus, k_bits_for(K), L_TABLES, np.random.default_rng(999 + K), D)

    fields = ('if_value', 'true_recall', 'recall_hat_raw', 'recall_hat_excl',
              'n_q_hat_raw', 'n_q_hat_excl', 'true_n')
    rows = {corpus_name: {K: {f: [] for f in fields} for K in K_VALUES} for corpus_name in corpora}

    for a_id in range(n_anchors):
        anchor = anchors[a_id]
        home_idx = int(home_indices[a_id])
        r_cache_by_K = {K: float(cand_data['real'][K]['r_cache'][home_idx]) for K in K_VALUES}
        deltas = sw._anchor_delta_grid(r_cache_by_K, D, sw.IF_BIN_CENTERS)

        for delta in deltas:
            directions = np.array([sw._random_tangent(anchor, rng, D) for _ in range(n_directions)])
            queries = sw._move_along_geodesic(anchor, np.full(n_directions, delta), directions)

            for corpus_name, corpus in corpora.items():
                dists_test, idx_test = sw._bruteforce_knn(corpus, queries, K_MAX)
                delta_qc = np.arccos(np.clip(candidates @ queries.T, -1.0, 1.0))  # (n_cand, n_queries)

                for K in K_VALUES:
                    r_test_all = sw._chord_to_angle(dists_test[:, K - 1])
                    r_cache = cand_data[corpus_name][K]['r_cache']
                    resp_idx = cand_data[corpus_name][K]['idx']

                    for i in range(n_directions):
                        q = queries[i]
                        r_test = float(r_test_all[i])
                        IFs = exact_intersection_fraction_batch(r_cache, r_test, delta_qc[:, i], D)
                        best = int(IFs.argmax())
                        if_value = float(IFs[best])

                        true_topk = set(idx_test[i, :K].tolist())
                        c_response = resp_idx[best]
                        true_recall = len(true_topk & set(c_response.tolist())) / K

                        c_points = corpus[c_response]
                        r_q_proxy = float(_angular_distance_consistent(c_points, q).max())

                        raw_est, extra_est = estimate_range_count(
                            lsh_tables[corpus_name][K], q, exclude_points=c_points)
                        n_q_hat_raw = raw_est
                        n_q_hat_excl = K + extra_est
                        recall_hat_raw = K / n_q_hat_raw if n_q_hat_raw > 0 else float('nan')
                        recall_hat_excl = K / n_q_hat_excl

                        true_n = int(covered_mask(corpus, q, r_q_proxy).sum())

                        dst = rows[corpus_name][K]
                        dst['if_value'].append(if_value)
                        dst['true_recall'].append(true_recall)
                        dst['recall_hat_raw'].append(recall_hat_raw)
                        dst['recall_hat_excl'].append(recall_hat_excl)
                        dst['n_q_hat_raw'].append(n_q_hat_raw)
                        dst['n_q_hat_excl'].append(n_q_hat_excl)
                        dst['true_n'].append(true_n)

        if (a_id + 1) % 5 == 0 or a_id == n_anchors - 1:
            print(f"  anchor {a_id + 1}/{n_anchors}...")

    for corpus_name in corpora:
        for K in K_VALUES:
            for f in fields:
                rows[corpus_name][K][f] = np.array(rows[corpus_name][K][f])
    return rows


def report_ground_truth_diagnostic(rows: dict) -> None:
    print("\n=== N_q_hat vs. true (brute-force) count diagnostic, exact, every row ===")
    for corpus_name in ('real', 'uniform'):
        for K in K_VALUES:
            r = rows[corpus_name][K]
            true_n = r['true_n'].astype(np.float64)
            for label, est in (('raw', r['n_q_hat_raw']), ('excl', r['n_q_hat_excl'])):
                err = est - true_n
                rel_err = err / np.maximum(true_n, 1.0)
                ss_res = np.sum((true_n - est) ** 2)
                ss_tot = np.sum((true_n - true_n.mean()) ** 2)
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
                print(f"  corpus={corpus_name:>7} K={K:>3} N_q_hat_{label:>4}: "
                      f"mean_abs_err={np.abs(err).mean():.2f}, mean_rel_err={rel_err.mean():+.3f}, "
                      f"R2_vs_true_count={r2:.3f}")


def report_recall_hat_ranges(rows: dict) -> None:
    print("\n=== recall_hat range sanity check (excl guaranteed in (0,1], raw not) ===")
    for corpus_name in ('real', 'uniform'):
        for K in K_VALUES:
            r = rows[corpus_name][K]
            print(f"  corpus={corpus_name:>7} K={K:>3}: "
                  f"recall_hat_raw range=[{r['recall_hat_raw'].min():.4f}, {r['recall_hat_raw'].max():.4f}], "
                  f"recall_hat_excl range=[{r['recall_hat_excl'].min():.4f}, {r['recall_hat_excl'].max():.4f}]")


def plot_estimator_comparison(rows: dict, rng: np.random.Generator) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    estimators = [('if_value', 'IF', 'tab:orange'), ('recall_hat_raw', 'recall_hat_raw', 'tab:blue'),
                  ('recall_hat_excl', 'recall_hat_excl', 'tab:green')]

    for K in K_VALUES:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), squeeze=False)
        axes = axes[0]
        for col, corpus_name in enumerate(('real', 'uniform')):
            ax = axes[col]
            r = rows[corpus_name][K]
            true_recall = r['true_recall']
            for field, label, color in estimators:
                x = r[field]
                finite = np.isfinite(x) & np.isfinite(true_recall)
                x_f, y_f = x[finite], true_recall[finite]
                bal_x, bal_y, n_per_bin = balance_by_if_bin(x_f, y_f, rng)
                if len(bal_x) < 2:
                    continue
                slope, intercept, r2 = _ols_fit(bal_x, bal_y)
                ccc = _concordance_correlation_coefficient(bal_x, bal_y)
                centers, means, sems, counts = _fixed_bin_stats(bal_x, bal_y)
                valid = counts > 0
                ax.errorbar(centers[valid], means[valid], yerr=sems[valid], marker='o', markersize=3,
                           color=color, capsize=2, label=f"{label} (R2={r2:.2f}, CCC={ccc:.2f})")
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f'{corpus_name} corpus', fontsize=9)
            ax.set_xlabel('estimator value')
            if col == 0:
                ax.set_ylabel('mean true recall')
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
        fig.suptitle(f'SIFT ({D}D), K={K}: IF vs recall_hat_raw vs recall_hat_excl, binned')
        plt.tight_layout()
        out_path = PLOTS_DIR / f'sift_{D}d_K{K}_estimator_comparison_binned.png'
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"    wrote {out_path}")

#-------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-anchors', type=int, default=N_ANCHORS_FAST, dest='n_anchors')
    parser.add_argument('--n-directions', type=int, default=N_DIRECTIONS_FAST, dest='n_directions')
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent.parent
    bg_path = repo_root / sw.BACKGROUND_PATH
    background = sw._normalize(np.load(bg_path).astype(np.float64))
    uniform_rng = np.random.default_rng(sw.UNIFORM_CORPUS_SEED)
    uniform_corpus = sw._normalize(uniform_rng.standard_normal((len(background), D)))

    print(f"=== recall_hat estimator comparison, SIFT (d={D}), K={K_VALUES} ===")
    print(f"n_anchors={args.n_anchors}, n_directions={args.n_directions}\n")

    rows = run_cell(background, uniform_corpus, args.n_anchors, args.n_directions, args.seed)

    report_ground_truth_diagnostic(rows)
    report_recall_hat_ranges(rows)

    rng = np.random.default_rng(123)
    plot_estimator_comparison(rows, rng)

    print("\nDone.")


if __name__ == '__main__':
    main()
