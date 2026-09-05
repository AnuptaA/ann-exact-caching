#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Debug why the LSH-based Part 2 estimator's corrected_recall is much
lower on the real corpus than the uniform corpus."""

import sys
import math
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from primitives import single_cap_fraction, sample_cap_points_direct
from part2_uniform_baseline_run_experiment import (
    _sample_uniform_sphere, _bruteforce_knn, _construct_test_poles, _chord_to_angle,
    K_VALUES, K_MAX, N_BACKGROUND_POINTS, GRID_MARGIN,
)
from sift_part2_real_data_run_experiment import (
    _normalize, _householder_e1_to, _real_anchor, D, BACKGROUND_PATH, UNIFORM_CORPUS_SEED,
)
from density_correction_common import (
    _angular_distance_consistent, compute_corrected_recall_exact,
    build_lsh_tables, hash_count_at_point, covered_mask,
)

#-------------------------------------------------------------------------------

SEED = 42
N_THETA = 8
M_PER_THETA = 5
N_PAIRS_PER_K = 8
K_SWEEP = [(6, 12), (8, 12), (12, 12), (12, 48)]

#-------------------------------------------------------------------------------


def per_table_hash_counts(tables: list, point: np.ndarray) -> np.ndarray:
    counts = []
    for table in tables:
        bits = (table['hyperplanes'] @ point) >= 0.0
        weights = (1 << np.arange(len(bits)))
        key = int((bits.astype(np.int64) * weights).sum())
        counts.append(table['key_to_count'].get(key, 0))
    return np.array(counts)


def nearest_corpus_dist(background: np.ndarray, point: np.ndarray) -> float:
    d = _angular_distance_consistent(background, point)
    return float(d.min())


def find_overlap_sample_point_instrumented(q, c_star, r_test, r_cache, rng,
                                            max_batches=20, batch_size=200):
    if r_test <= r_cache:
        small_pole, small_radius = q, r_test
    else:
        small_pole, small_radius = c_star, r_cache

    for batch_idx in range(max_batches):
        candidates = sample_cap_points_direct(small_pole, small_radius, batch_size, rng)
        ok = covered_mask(candidates, q, r_test) & covered_mask(candidates, c_star, r_cache)
        if np.any(ok):
            return candidates[ok][0], batch_idx + 1
    return None, max_batches


def main() -> None:
    repo_root = Path(__file__).parent.parent.parent
    bg_path = Path(BACKGROUND_PATH)
    if not bg_path.is_absolute():
        bg_path = repo_root / bg_path
    real_background = _normalize(np.load(bg_path).astype(np.float64))

    uniform_rng = np.random.default_rng(UNIFORM_CORPUS_SEED)
    uniform_background = _normalize(uniform_rng.standard_normal((len(real_background), D)))
    corpora = {'real': real_background, 'uniform': uniform_background}

    rng = np.random.default_rng(SEED)
    synthetic_background = _sample_uniform_sphere(N_BACKGROUND_POINTS, D, rng)
    e1 = np.zeros(D)
    e1[0] = 1.0
    dists_cache, _ = _bruteforce_knn(synthetic_background, e1[None, :], K_MAX)
    r_cache_by_K_synth = {K: float(_chord_to_angle(dists_cache[0, K - 1])) for K in K_VALUES}
    grid_max = min(GRID_MARGIN * 2.0 * max(r_cache_by_K_synth.values()), math.pi)
    delta_grid = np.linspace(0.0, grid_max, N_THETA)

    real_anchor = _real_anchor(background_id=0, real_background=real_background, d=D)
    H = _householder_e1_to(real_anchor)

    per_delta = []
    for delta in delta_grid:
        delta = float(delta)
        if delta < 1e-12:
            continue
        test_poles = _construct_test_poles(delta, M_PER_THETA, D, rng)
        relocated = _normalize(test_poles @ H)
        per_delta.append((delta, relocated))

    r_cache_by_corpus_K = {}
    for corpus_name, corpus in corpora.items():
        dists_anchor, _ = _bruteforce_knn(corpus, real_anchor[None, :], K_MAX)
        r_cache_by_corpus_K[corpus_name] = {K: float(_chord_to_angle(dists_anchor[0, K - 1]))
                                             for K in K_VALUES}

    print("=== Building LSH tables at default (K_BITS=12, L_TABLES=12) for diagnostics 1-4 ===")
    lsh_tables = {name: build_lsh_tables(corpus, 12, 12, np.random.default_rng(999), D)
                  for name, corpus in corpora.items()}

    est_rng = np.random.default_rng(123)

    for corpus_name, corpus in corpora.items():
        print(f"\n=== Corpus: {corpus_name} (size={len(corpus)}) ===")
        collected = 0
        for K in K_VALUES:
            r_cache = r_cache_by_corpus_K[corpus_name][K]
            for delta, relocated in per_delta:
                dists_test, idx_test = _bruteforce_knn(corpus, relocated, K_MAX)
                r_test_all = _chord_to_angle(dists_test[:, K - 1])
                for i in range(relocated.shape[0]):
                    if collected >= N_PAIRS_PER_K:
                        break
                    q = relocated[i]
                    r_test = float(r_test_all[i])

                    overlap_point, n_batches = find_overlap_sample_point_instrumented(
                        q, real_anchor, r_test, r_cache, est_rng)
                    if overlap_point is None:
                        continue

                    counts_q = per_table_hash_counts(lsh_tables[corpus_name], q)
                    counts_overlap = per_table_hash_counts(lsh_tables[corpus_name], overlap_point)

                    nn_q = nearest_corpus_dist(corpus, q)
                    nn_overlap = nearest_corpus_dist(corpus, overlap_point)

                    area_q = single_cap_fraction(r_test, D)
                    exact_density_q = K / area_q if area_q > 0 else float('nan')
                    lsh_density_q = hash_count_at_point(lsh_tables[corpus_name], q)

                    dist_q_to_overlap = float(np.arccos(np.clip(q @ overlap_point, -1.0, 1.0)))
                    dist_c_to_overlap = float(np.arccos(np.clip(real_anchor @ overlap_point, -1.0, 1.0)))

                    print(f"  K={K:>3} delta={delta:.4f} row={i}: "
                          f"batches_to_accept={n_batches}, "
                          f"counts_q(per-table)={counts_q.tolist()}, "
                          f"counts_overlap(per-table)={counts_overlap.tolist()}, "
                          f"nn_dist_q={nn_q:.4f}, nn_dist_overlap={nn_overlap:.4f}, "
                          f"exact_density_q={exact_density_q:.2f}, lsh_hash_count_q={lsh_density_q:.2f}, "
                          f"dist_q_to_overlap={dist_q_to_overlap:.4f} (r_test={r_test:.4f}), "
                          f"dist_c_to_overlap={dist_c_to_overlap:.4f} (r_cache={r_cache:.4f})")
                    collected += 1
                if collected >= N_PAIRS_PER_K:
                    break
            if collected >= N_PAIRS_PER_K:
                break

    print("\n=== Diagnostic 5: K_BITS/L_TABLES sweep ===")
    for K_BITS, L_TABLES in K_SWEEP:
        print(f"\n  K_BITS={K_BITS}, L_TABLES={L_TABLES}:")
        sweep_tables = {name: build_lsh_tables(corpus, K_BITS, L_TABLES, np.random.default_rng(999), D)
                        for name, corpus in corpora.items()}
        for corpus_name, corpus in corpora.items():
            for K in K_VALUES:
                r_cache = r_cache_by_corpus_K[corpus_name][K]
                cr_vals = []
                hc_q_vals = []
                hc_overlap_vals = []
                collected = 0
                for delta, relocated in per_delta:
                    dists_test, idx_test = _bruteforce_knn(corpus, relocated, K_MAX)
                    r_test_all = _chord_to_angle(dists_test[:, K - 1])
                    for i in range(relocated.shape[0]):
                        if collected >= N_PAIRS_PER_K:
                            break
                        q = relocated[i]
                        r_test = float(r_test_all[i])
                        overlap_point, _ = find_overlap_sample_point_instrumented(
                            q, real_anchor, r_test, r_cache, est_rng)
                        if overlap_point is None:
                            continue
                        hc_q = hash_count_at_point(sweep_tables[corpus_name], q)
                        hc_overlap = hash_count_at_point(sweep_tables[corpus_name], overlap_point)
                        if hc_q <= 0:
                            continue
                        if_value = 1.0
                        cr_vals.append(if_value * hc_overlap / hc_q)
                        hc_q_vals.append(hc_q)
                        hc_overlap_vals.append(hc_overlap)
                        collected += 1
                    if collected >= N_PAIRS_PER_K:
                        break
                if cr_vals:
                    print(f"    corpus={corpus_name:>7} K={K:>3}: n={len(cr_vals)}, "
                          f"mean_hash_count_q={np.mean(hc_q_vals):.2f}, "
                          f"mean_hash_count_overlap={np.mean(hc_overlap_vals):.2f}, "
                          f"mean(hc_overlap/hc_q)={np.mean(cr_vals):.4f}")
                else:
                    print(f"    corpus={corpus_name:>7} K={K:>3}: n=0 (no accepted overlap points)")

    print("\nDone.")


if __name__ == '__main__':
    main()
