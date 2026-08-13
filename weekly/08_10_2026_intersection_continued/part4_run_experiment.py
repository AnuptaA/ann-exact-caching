#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Enlarged-radius pool+argmax: Part 1 experiment's random-sampling/argmax
structure, with each cache candidate's naive K-th-NN radius replaced by a
shared coverage-fraction-target radius.
"""

import sys
import math
import argparse
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent))

import numpy as np
from scipy.optimize import brentq

from primitives import (
    exact_intersection_fraction,
    solid_ball_intersection_fraction,
    chord_distance_from_angular,
    single_cap_fraction,
)

#-------------------------------------------------------------------------------

K_VALUES = [10, 25, 50, 100]
K_MAX = max(K_VALUES)
N_BACKGROUNDS = 5
N_BACKGROUND_POINTS = 100000
MAX_PAIRWISE_SAMPLE = 20000
DIMS = [2, 3, 8, 16, 32, 64, 128, 256, 384]
METHODS = ('cap', 'solid')

N_CANDIDATES_DEFAULT = 100
N_TEST_QUERIES_DEFAULT = 1000

COVERAGE_TARGETS = [0.10, 0.25, 0.50, 0.65, 0.80, 0.90, 0.95]


OUT_DIR = Path(__file__).parent / 'results'
PART_DIR = 'part4'
_RUN_TIMESTAMP = None


def _run_dir() -> Path:
    """One timestamp per process run."""
    global _RUN_TIMESTAMP
    if _RUN_TIMESTAMP is None:
        _RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
    return OUT_DIR / _RUN_TIMESTAMP

#-------------------------------------------------------------------------------


def _chord_to_angle(chord: np.ndarray) -> np.ndarray:
    """Inverse of chord_distance_from_angular: theta = 2*arcsin(chord/2)."""
    return 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))


def _sample_uniform_sphere(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """n uniform random points on S^(d-1)."""
    pts = rng.standard_normal((n, d))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _bruteforce_knn(background: np.ndarray, queries: np.ndarray, k: int, chunk_size: int = 1000) -> tuple:
    """Top-k nearest neighbors of each query against background, by chord
    distance, equivalently largest dot product since both live on the unit
    sphere. Chunked over queries to bound memory."""
    n_queries = queries.shape[0]
    all_dists = np.empty((n_queries, k), dtype=np.float64)
    all_idx = np.empty((n_queries, k), dtype=np.int64)
    for start in range(0, n_queries, chunk_size):
        end = min(start + chunk_size, n_queries)
        sims = queries[start:end] @ background.T
        top_idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        top_sims = np.take_along_axis(sims, top_idx, axis=1)
        order = np.argsort(-top_sims, axis=1)
        sorted_idx = np.take_along_axis(top_idx, order, axis=1)
        sorted_sims = np.take_along_axis(top_sims, order, axis=1)
        all_dists[start:end] = np.sqrt(np.clip(2.0 - 2.0 * sorted_sims, 0.0, None))
        all_idx[start:end] = sorted_idx
    return all_dists, all_idx


def _analytic_radius(coverage: float, d: int) -> float:
    """Radius r solving single_cap_fraction(r, d) == coverage. Monotone
    increasing and continuous on [0, pi], so brentq applies over the whole
    domain."""
    coverage = min(max(coverage, 1e-12), 1.0 - 1e-12)
    return brentq(lambda r: single_cap_fraction(r, d) - coverage, 1e-9, math.pi - 1e-9)

#-------------------------------------------------------------------------------


def run_background(d: int, background_id: int, n_points: int, n_candidates: int, n_test_queries: int,
                    coverage_targets: list, seed: int) -> dict:
    """Generates one background, one candidate pool, one test-query pool
    (independent random draws), and computes every (coverage_target, K)
    combination's selection/pairwise data from them.

    cache_q positions and test_q are drawn once per background and reused
    across every (target, K) combination; only the cache radius and r_test
    change.

    Returns {(target, K): {'selection': {...}, 'pairwise': {...}}}.
    """
    rng = np.random.default_rng(seed)
    background = _sample_uniform_sphere(n_points, d, rng)

    cache_q = _sample_uniform_sphere(n_candidates, d, rng)
    test_q = _sample_uniform_sphere(n_test_queries, d, rng)

    dists_test, idx_test = _bruteforce_knn(background, test_q, K_MAX)

    # every test query against every candidate: (n_test_queries, n_candidates), reused across every (target, K).
    delta_matrix = np.arccos(np.clip(test_q @ cache_q.T, -1.0, 1.0))

    per_combo = {}
    for target in coverage_targets:
        r_cache_scalar = _analytic_radius(target, d)
        r_cache_chord_scalar = chord_distance_from_angular(r_cache_scalar)

        for K in K_VALUES:
            r_test = _chord_to_angle(dists_test[:, K - 1])  # (n_test_queries,)

            cap_matrix = np.array([
                [exact_intersection_fraction(r_cache_scalar, float(r_test[i]), float(delta_matrix[i, j]), d)
                 for j in range(n_candidates)]
                for i in range(n_test_queries)
            ])

            r_test_chord = np.array([chord_distance_from_angular(float(r)) for r in r_test])
            delta_chord = np.array([
                [chord_distance_from_angular(float(delta_matrix[i, j])) for j in range(n_candidates)]
                for i in range(n_test_queries)
            ])
            solid_matrix = np.array([
                [solid_ball_intersection_fraction(r_cache_chord_scalar, float(r_test_chord[i]), float(delta_chord[i, j]), d)
                 for j in range(n_candidates)]
                for i in range(n_test_queries)
            ])

            selection = {
                'background_id': np.full(n_test_queries, background_id, dtype=np.int32),
                'coverage_target': np.full(n_test_queries, target, dtype=np.float64),
                'r_test': r_test,
            }
            method_matrices = {'cap': cap_matrix, 'solid': solid_matrix}
            for method, matrix in method_matrices.items():
                w_idx = matrix.argmax(axis=1)
                winner_delta = delta_matrix[np.arange(n_test_queries), w_idx]
                winner_fraction = matrix[np.arange(n_test_queries), w_idx]

                neighbor_coords = background[idx_test[:, :K]]  # (n_test_queries, K, d)
                winner_cache_q = cache_q[w_idx]  # (n_test_queries, d)
                dist_to_winner = np.linalg.norm(neighbor_coords - winner_cache_q[:, None, :], axis=2)
                # every candidate shares the same radius for a given target, no per-winner indexing needed.
                covered = dist_to_winner <= r_cache_chord_scalar
                recall = covered.mean(axis=1)

                selection[f'{method}_winner_idx'] = w_idx.astype(np.int32)
                selection[f'{method}_r_cache'] = np.full(n_test_queries, r_cache_scalar, dtype=np.float64)
                selection[f'{method}_delta'] = winner_delta
                selection[f'{method}_fraction'] = winner_fraction
                selection[f'{method}_recall'] = recall

            n_total = n_test_queries * n_candidates
            n_sample = min(n_total, MAX_PAIRWISE_SAMPLE)
            flat_i, flat_j = np.divmod(rng.choice(n_total, size=n_sample, replace=False), n_candidates)
            pairwise = {
                'background_id': np.full(n_sample, background_id, dtype=np.int32),
                'coverage_target': np.full(n_sample, target, dtype=np.float64),
                'r_cache': np.full(n_sample, r_cache_scalar, dtype=np.float64),
                'r_test': r_test[flat_i],
                'delta': delta_matrix[flat_i, flat_j],
                'cap_fraction': cap_matrix[flat_i, flat_j],
                'solid_fraction': solid_matrix[flat_i, flat_j],
            }

            per_combo[(target, K)] = {'selection': selection, 'pairwise': pairwise}

    return per_combo


def run_dimension(d: int, n_backgrounds: int, n_points: int, n_candidates: int, n_test_queries: int,
                   coverage_targets: list, base_seed: int) -> dict:
    """Runs every background for one dimension and pools their per-(target,
    K) selection/pairwise arrays into one array per field."""
    selection_fields = ('background_id', 'coverage_target', 'r_test') + tuple(
        f'{m}_{f}' for m in METHODS for f in ('winner_idx', 'r_cache', 'delta', 'fraction', 'recall')
    )
    pairwise_fields = ('background_id', 'coverage_target', 'r_cache', 'r_test', 'delta', 'cap_fraction', 'solid_fraction')

    combos = [(t, K) for t in coverage_targets for K in K_VALUES]
    pooled = {c: {'selection': {f: [] for f in selection_fields}, 'pairwise': {f: [] for f in pairwise_fields}}
              for c in combos}

    for bg_id in range(n_backgrounds):
        print(f"  background {bg_id + 1}/{n_backgrounds} (d={d})...")
        per_combo = run_background(d, bg_id, n_points, n_candidates, n_test_queries, coverage_targets,
                                    seed=base_seed + bg_id)
        for c in combos:
            for field in selection_fields:
                pooled[c]['selection'][field].append(per_combo[c]['selection'][field])
            for field in pairwise_fields:
                pooled[c]['pairwise'][field].append(per_combo[c]['pairwise'][field])

    for c in combos:
        for part in ('selection', 'pairwise'):
            for field in pooled[c][part]:
                pooled[c][part][field] = np.concatenate(pooled[c][part][field])
    return pooled


def save_dimension(d: int, pooled: dict, run_dir: Path) -> None:
    """One selection .npz and one pairwise .npz per K persisted, pooling
    every coverage_target into that K's file via the 'coverage_target'
    field."""
    dim_dir = run_dir / PART_DIR / f'{d}d' / 'raw'
    dim_dir.mkdir(parents=True, exist_ok=True)

    by_K = {K: {'selection': {}, 'pairwise': {}} for K in K_VALUES}
    for (target, K), parts in pooled.items():
        for part_name, arrays in parts.items():
            for field, values in arrays.items():
                by_K[K][part_name].setdefault(field, []).append(values)

    for K, parts in by_K.items():
        for part_name, fields in parts.items():
            merged = {field: np.concatenate(values) for field, values in fields.items()}
            path = dim_dir / f'K{K}_{part_name}.npz'
            np.savez_compressed(path, d=d, K=K, **merged)
            print(f"    wrote {path}")


def bin_count_diagnostic(d: int, pooled: dict, coverage_targets: list, min_count: int = 30) -> None:
    """Prints per-(target, K) informative-fraction and selection agreement,
    the direct answer to "does this coverage target actually rescue random
    sampling at this dimension"."""
    print(f"\n  Informative fraction and selection agreement (d={d}):")
    for target in coverage_targets:
        for K in K_VALUES:
            sel = pooled[(target, K)]['selection']
            cap_frac, solid_frac = sel['cap_fraction'], sel['solid_fraction']
            cap_idx, solid_idx = sel['cap_winner_idx'], sel['solid_winner_idx']
            informative = (cap_frac > 0) | (solid_frac > 0)
            informative_frac = informative.mean()
            real_agree = (
                (cap_idx[informative] == solid_idx[informative]).mean()
                if informative.any() else float('nan')
            )
            mean_recall = sel['cap_recall'].mean()
            print(f"    target={target:.2f}, K={K}: informative={informative_frac:.4f}, "
                  f"agreement={real_agree:.4f}, mean_cap_recall={mean_recall:.4f}")


def main() -> None:
    global K_VALUES, K_MAX, OUT_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dims', type=int, nargs='+', default=DIMS)
    parser.add_argument('--k-values', type=int, nargs='+', default=K_VALUES, dest='k_values')
    parser.add_argument('--coverage-targets', type=float, nargs='+', default=COVERAGE_TARGETS, dest='coverage_targets')
    parser.add_argument('--n-backgrounds', type=int, default=N_BACKGROUNDS, dest='n_backgrounds')
    parser.add_argument('--n-background-points', type=int, default=N_BACKGROUND_POINTS, dest='n_background_points')
    parser.add_argument('--n-candidates', type=int, default=None, dest='n_candidates',
                         help='overrides the per-dimension tier for every dimension if given')
    parser.add_argument('--n-test-queries', type=int, default=None, dest='n_test_queries',
                         help='overrides the per-dimension tier for every dimension if given')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--run-dir', type=str, default=None,
                         help='write directly into this exact directory (no new timestamp minted), '
                              "e.g. to combine output with another part's run")
    args = parser.parse_args()

    K_VALUES = sorted(args.k_values)
    K_MAX = max(K_VALUES)
    if args.output_dir:
        OUT_DIR = Path(args.output_dir)

    run_dir = Path(args.run_dir) if args.run_dir else _run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")
    print(f"K values: {K_VALUES}, coverage_targets={args.coverage_targets}, n_backgrounds={args.n_backgrounds}, "
          f"n_background_points={args.n_background_points}\n")

    for d in args.dims:
        n_candidates = args.n_candidates if args.n_candidates is not None else N_CANDIDATES_DEFAULT
        n_test_queries = args.n_test_queries if args.n_test_queries is not None else N_TEST_QUERIES_DEFAULT
        print(f"=== d={d} (n_candidates={n_candidates}, n_test_queries={n_test_queries}) ===")
        pooled = run_dimension(d, args.n_backgrounds, args.n_background_points, n_candidates,
                                n_test_queries, args.coverage_targets, base_seed=args.seed)
        save_dimension(d, pooled, run_dir)
        bin_count_diagnostic(d, pooled, args.coverage_targets)
        print()

    print(f"Done: {run_dir}")


if __name__ == '__main__':
    main()
