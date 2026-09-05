#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Density-correction stage for Scheme A (SIFT): runs Part 1 (overlap-
count sanity check) and Part 2 (local-density estimator)."""

import sys
import math
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from primitives import exact_intersection_fraction_batch_theta2
from part2_uniform_baseline_run_experiment import (
    _sample_uniform_sphere, _bruteforce_knn, _construct_test_poles, _chord_to_angle,
    K_VALUES, K_MAX, N_BACKGROUNDS, N_BACKGROUND_POINTS, N_THETA, M_PER_THETA, GRID_MARGIN,
)
from sift_part2_real_data_run_experiment import (
    _normalize, _householder_e1_to, _real_anchor, D, BACKGROUND_PATH, UNIFORM_CORPUS_SEED,
)
from density_correction_common import (
    independent_overlap_count, compute_corrected_recall_exact, true_recall_consistent,
    build_lsh_tables, compute_corrected_recall_estimated,
)

#-------------------------------------------------------------------------------

K_BITS = 12   # corpus_size=50000 * (1/2)^12 ~= 12 expected uniform-null bucket occupancy
L_TABLES = 12

#-------------------------------------------------------------------------------

def regenerate_background_trial(d: int, background_id: int, n_points: int, n_theta: int, m_per_theta: int,
                                 seed: int) -> dict:
    """Reproduces run_background_real's position generation, returning
    actual positions (anchor and relocated test poles) instead of
    derived scalars."""
    rng = np.random.default_rng(seed)
    synthetic_background = _sample_uniform_sphere(n_points, d, rng)

    e1 = np.zeros(d)
    e1[0] = 1.0
    dists_cache, _ = _bruteforce_knn(synthetic_background, e1[None, :], K_MAX)
    r_cache_by_K_synthetic = {K: float(_chord_to_angle(dists_cache[0, K - 1])) for K in K_VALUES}
    grid_max = min(GRID_MARGIN * 2.0 * max(r_cache_by_K_synthetic.values()), math.pi)
    delta_grid = np.linspace(0.0, grid_max, n_theta)

    real_anchor = _real_anchor(background_id=background_id, real_background=REAL_BACKGROUND, d=d)
    H = _householder_e1_to(real_anchor)

    per_delta = []
    for delta in delta_grid:
        delta = float(delta)
        test_poles = _construct_test_poles(delta, m_per_theta, d, rng)
        relocated = _normalize(test_poles @ H)
        per_delta.append((delta, relocated))

    return {'real_anchor': real_anchor, 'per_delta': per_delta}


def main() -> None:
    global REAL_BACKGROUND

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--background-path', type=str, default=BACKGROUND_PATH, dest='background_path')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--part1-only', action='store_true')
    parser.add_argument('--diagnostic-rows', type=int, default=0,
                         help='if >0, only process this many rows per (K, corpus) for Part 2 (diagnostic mode)')
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent.parent
    bg_path = Path(args.background_path)
    if not bg_path.is_absolute():
        bg_path = repo_root / bg_path
    REAL_BACKGROUND = _normalize(np.load(bg_path).astype(np.float64))

    uniform_rng = np.random.default_rng(UNIFORM_CORPUS_SEED)
    UNIFORM_BACKGROUND = _normalize(uniform_rng.standard_normal((len(REAL_BACKGROUND), D)))
    corpora = {'real': REAL_BACKGROUND, 'uniform': UNIFORM_BACKGROUND}

    est_rng = np.random.default_rng(123)

    print(f"=== Density correction, Scheme A, SIFT (d={D}) ===")
    print(f"K_bits={K_BITS}, L_tables={L_TABLES}\n")

    lsh_tables = {}
    if not args.part1_only:
        for corpus_name, corpus in corpora.items():
            print(f"  building LSH tables for {corpus_name} corpus...")
            lsh_tables[corpus_name] = build_lsh_tables(corpus, K_BITS, L_TABLES,
                                                        np.random.default_rng(999), D)

    part1_summary = {}
    part2_rows = {corpus: {K: {'corrected_recall': [], 'if_value': [], 'true_recall': []}
                            for K in K_VALUES} for corpus in corpora}

    for bg_id in range(N_BACKGROUNDS):
        print(f"  background {bg_id + 1}/{N_BACKGROUNDS}...")
        gen = regenerate_background_trial(D, bg_id, N_BACKGROUND_POINTS, N_THETA, M_PER_THETA,
                                           seed=args.seed + bg_id)
        real_anchor = gen['real_anchor']

        # r_cache per K, per corpus (fresh KNN of the anchor against each corpus)
        r_cache_by_corpus_K = {}
        for corpus_name, corpus in corpora.items():
            dists_anchor, _ = _bruteforce_knn(corpus, real_anchor[None, :], K_MAX)
            r_cache_by_corpus_K[corpus_name] = {K: float(_chord_to_angle(dists_anchor[0, K - 1]))
                                                 for K in K_VALUES}

        for delta, relocated in gen['per_delta']:
            if delta < 1e-12:
                # q coincides exactly with the anchor here, a degenerate tie
                # case excluded elsewhere in this branch too.
                continue
            for corpus_name, corpus in corpora.items():
                dists_test, idx_test = _bruteforce_knn(corpus, relocated, K_MAX)
                for K in K_VALUES:
                    r_cache = r_cache_by_corpus_K[corpus_name][K]
                    r_test_all = _chord_to_angle(dists_test[:, K - 1])
                    if_all = exact_intersection_fraction_batch_theta2(r_cache, r_test_all, delta, D)

                    for i in range(relocated.shape[0]):
                        q = relocated[i]
                        r_test = float(r_test_all[i])
                        if_value = float(if_all[i])
                        true_recall = true_recall_consistent(corpus, idx_test[i, :K], real_anchor, r_cache)

                        # Part 1
                        overlap_count = independent_overlap_count(corpus, q, real_anchor, r_test, r_cache)
                        corrected_exact = compute_corrected_recall_exact(overlap_count, K, if_value, r_test, D)
                        key = (corpus_name, K)
                        part1_summary.setdefault(key, []).append(abs(corrected_exact - true_recall))

                        # Part 2 (skip once diagnostic cap reached, if set)
                        if args.part1_only:
                            continue
                        if args.diagnostic_rows and len(part2_rows[corpus_name][K]['corrected_recall']) >= args.diagnostic_rows:
                            continue
                        corrected_est = compute_corrected_recall_estimated(
                            lsh_tables[corpus_name], q, real_anchor, r_test, r_cache, if_value, est_rng)
                        part2_rows[corpus_name][K]['corrected_recall'].append(corrected_est)
                        part2_rows[corpus_name][K]['if_value'].append(if_value)
                        part2_rows[corpus_name][K]['true_recall'].append(true_recall)

    print("\n  Part 1 (independent sanity check), max/mean |corrected_recall - recall|"
          " (excluding IF=0 rows, where the density ratio is undefined by construction, not a deviation):")
    for (corpus_name, K), diffs in sorted(part1_summary.items()):
        diffs = np.array(diffs)
        undefined_frac = np.isnan(diffs).mean()
        finite = diffs[np.isfinite(diffs)]
        if len(finite) == 0:
            print(f"    corpus={corpus_name:>7} K={K:>4}: n={len(diffs)}, all undefined (IF=0 for every row)")
            continue
        flag = " [FLAG: > 1e-6]" if finite.max() > 1e-6 else ""
        print(f"    corpus={corpus_name:>7} K={K:>4}: n={len(diffs)}, undefined_frac={undefined_frac:.3f}, "
              f"max={finite.max():.2e}, mean={finite.mean():.2e}{flag}")

    if not args.part1_only:
        print("\n  Part 2 (local-density estimator), summary:")
        for corpus_name in corpora:
            for K in K_VALUES:
                cr = np.array(part2_rows[corpus_name][K]['corrected_recall'])
                nan_frac = np.isnan(cr).mean() if len(cr) else float('nan')
                finite = cr[np.isfinite(cr)]
                print(f"    corpus={corpus_name:>7} K={K:>4}: n={len(cr)}, nan_frac={nan_frac:.3f}, "
                      f"corrected_recall range=[{finite.min() if len(finite) else float('nan'):.4f}, "
                      f"{finite.max() if len(finite) else float('nan'):.4f}], "
                      f"mean={finite.mean() if len(finite) else float('nan'):.4f}")

        out_dir = Path(__file__).parent / 'results' / 'density_correction'
        out_dir.mkdir(parents=True, exist_ok=True)
        for corpus_name in corpora:
            for K in K_VALUES:
                path = out_dir / f'sift_scheme_a_{corpus_name}_K{K}.npz'
                np.savez_compressed(path, **{k: np.array(v) for k, v in part2_rows[corpus_name][K].items()})
                print(f"    wrote {path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
