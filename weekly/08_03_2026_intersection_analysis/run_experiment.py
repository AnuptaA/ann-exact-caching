#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Synthetic 2D/3D experiment generator for the intersection-fraction
calibration study.

Three independent vector pools per background:

  - background: the data vectors, uniform on the unit circle or sphere
  - cache_q: a pool of N_CANDIDATES points, shared across every test query in a
    background.
  - test_q: N_TEST_QUERIES points, independent of cache_q and the background

For every test query and every K, both methods score it against EVERY
candidate in the pool and take the argmax separately per method (cap's
winner and solid-ball's winner may differ for the same test query),
then compute the exact empirical recall against each method's winning
candidate.

Two datasets are saved per (dimension, K):
  - selection: per test query, per method, the winning candidate's own
    r_cache, delta, predicted fraction, actual recall, and which
    candidate index won (measure cap/solid-ball selection agreement).
  - pairwise: a subsample of the raw (test_q, cache_q) cross product,
    before any argmax, for debugging.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from scipy.spatial import cKDTree

from primitives import exact_intersection_fraction, solid_ball_intersection_fraction, chord_distance_from_angular

#-------------------------------------------------------------------------------

K_VALUES = [1, 5, 10, 25, 50, 100]
K_MAX = max(K_VALUES)
N_BACKGROUNDS = 5
N_BACKGROUND_POINTS = 100000
N_CANDIDATES = 500
N_TEST_QUERIES = 5000
MAX_PAIRWISE_SAMPLE = 20000
DIMS = [2, 3]
METHODS = ('cap', 'solid')

OUT_DIR = Path(__file__).parent / 'results'
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


def run_background(d: int, background_id: int, n_points: int, n_candidates: int, n_test_queries: int, seed: int) -> dict:
    """Generates one background (data corpus, a shared candidate pool, and a
    set of test queries) and computes every K stratum's selection and
    pairwise data from it.

    cache_q (the pool) and test_q are drawn ONCE per background and
    reused across every K in K_VALUES.

    Returns {K: {'selection': {...}, 'pairwise': {...}}} for K in K_VALUES.
    """
    rng = np.random.default_rng(seed)
    background = _sample_uniform_sphere(n_points, d, rng)
    tree = cKDTree(background)

    cache_q = _sample_uniform_sphere(n_candidates, d, rng)
    test_q = _sample_uniform_sphere(n_test_queries, d, rng)

    dists_cache, _ = tree.query(cache_q, k=K_MAX)
    dists_test, idx_test = tree.query(test_q, k=K_MAX)

    # every test query against every candidate: (n_test_queries, n_candidates).
    delta_matrix = np.arccos(np.clip(test_q @ cache_q.T, -1.0, 1.0))

    per_k = {}
    for K in K_VALUES:
        r_cache = _chord_to_angle(dists_cache[:, K - 1])  # (n_candidates,)
        r_test = _chord_to_angle(dists_test[:, K - 1])  # (n_test_queries,)

        cap_matrix = np.array([
            [exact_intersection_fraction(float(r_cache[j]), float(r_test[i]), float(delta_matrix[i, j]), d)
             for j in range(n_candidates)]
            for i in range(n_test_queries)
        ])

        r_cache_chord = np.array([chord_distance_from_angular(float(r)) for r in r_cache])
        r_test_chord = np.array([chord_distance_from_angular(float(r)) for r in r_test])
        delta_chord = np.array([
            [chord_distance_from_angular(float(delta_matrix[i, j])) for j in range(n_candidates)]
            for i in range(n_test_queries)
        ])
        solid_matrix = np.array([
            [solid_ball_intersection_fraction(float(r_cache_chord[j]), float(r_test_chord[i]), float(delta_chord[i, j]), d)
             for j in range(n_candidates)]
            for i in range(n_test_queries)
        ])

        selection = {'background_id': np.full(n_test_queries, background_id, dtype=np.int32), 'r_test': r_test}
        method_matrices = {'cap': cap_matrix, 'solid': solid_matrix}
        for method, matrix in method_matrices.items():
            w_idx = matrix.argmax(axis=1)
            winner_r_cache = r_cache[w_idx]
            winner_delta = delta_matrix[np.arange(n_test_queries), w_idx]
            winner_fraction = matrix[np.arange(n_test_queries), w_idx]

            # find recall of test query wiith respect to best cache candidate selected by method
            neighbor_coords = background[idx_test[:, :K]]  # (n_test_queries, K, d)
            winner_cache_q = cache_q[w_idx]  # (n_test_queries, d)
            dist_to_winner = np.linalg.norm(neighbor_coords - winner_cache_q[:, None, :], axis=2)
            winner_r_cache_chord = dists_cache[w_idx, K - 1]
            covered = dist_to_winner <= winner_r_cache_chord[:, None]
            recall = covered.mean(axis=1)

            selection[f'{method}_winner_idx'] = w_idx.astype(np.int32)
            selection[f'{method}_r_cache'] = winner_r_cache
            selection[f'{method}_delta'] = winner_delta
            selection[f'{method}_fraction'] = winner_fraction
            selection[f'{method}_recall'] = recall

        # store raw pairwise subsample, before argmax
        n_total = n_test_queries * n_candidates
        n_sample = min(n_total, MAX_PAIRWISE_SAMPLE)
        flat_i, flat_j = np.divmod(rng.choice(n_total, size=n_sample, replace=False), n_candidates)
        pairwise = {
            'background_id': np.full(n_sample, background_id, dtype=np.int32),
            'r_cache': r_cache[flat_j],
            'r_test': r_test[flat_i],
            'delta': delta_matrix[flat_i, flat_j],
            'cap_fraction': cap_matrix[flat_i, flat_j],
            'solid_fraction': solid_matrix[flat_i, flat_j],
        }

        per_k[K] = {'selection': selection, 'pairwise': pairwise}

    return per_k


def run_dimension(d: int, n_backgrounds: int, n_points: int, n_candidates: int, n_test_queries: int, base_seed: int) -> dict:
    """Runs every background for one dimension and pools their per-K
    selection/pairwise arrays into one array per field."""
    selection_fields = ('background_id', 'r_test') + tuple(
        f'{m}_{f}' for m in METHODS for f in ('winner_idx', 'r_cache', 'delta', 'fraction', 'recall')
    )
    pairwise_fields = ('background_id', 'r_cache', 'r_test', 'delta', 'cap_fraction', 'solid_fraction')

    pooled = {K: {'selection': {f: [] for f in selection_fields}, 'pairwise': {f: [] for f in pairwise_fields}}
              for K in K_VALUES}

    for bg_id in range(n_backgrounds):
        print(f"  background {bg_id + 1}/{n_backgrounds} (d={d})...")
        per_k = run_background(d, bg_id, n_points, n_candidates, n_test_queries, seed=base_seed + bg_id)
        for K in K_VALUES:
            for field in selection_fields:
                pooled[K]['selection'][field].append(per_k[K]['selection'][field])
            for field in pairwise_fields:
                pooled[K]['pairwise'][field].append(per_k[K]['pairwise'][field])

    for K in K_VALUES:
        for part in ('selection', 'pairwise'):
            for field in pooled[K][part]:
                pooled[K][part][field] = np.concatenate(pooled[K][part][field])
    return pooled


def save_dimension(d: int, pooled: dict, run_dir: Path) -> None:
    """One selection .npz and one pairwise .npz per K persisted."""
    dim_dir = run_dir / f'{d}d' / 'raw'
    dim_dir.mkdir(parents=True, exist_ok=True)
    for K, parts in pooled.items():
        for part_name, arrays in parts.items():
            path = dim_dir / f'K{K}_{part_name}.npz'
            np.savez_compressed(path, d=d, K=K, **arrays)
            print(f"    wrote {path}")


def bin_count_diagnostic(d: int, pooled: dict, min_count: int = 30) -> None:
    """Prints per-K bin population counts (5% predicted-fraction
    increments, cap method's selection data) and flags sparse or empty
    bins, plus per-K cap/solid-ball selection agreement."""
    bin_edges = np.arange(0.0, 1.05, 0.05)
    print(f"\n  Bin-count diagnostic (d={d}, 5% predicted-fraction bins, cap method, selected candidate):")
    for K in K_VALUES:
        fractions = pooled[K]['selection']['cap_fraction']
        counts, _ = np.histogram(fractions, bins=bin_edges)
        sparse = [(bin_edges[i], bin_edges[i + 1], c) for i, c in enumerate(counts) if 0 < c < min_count]
        empty = int(np.sum(counts == 0))
        print(f"    K={K}: total={len(fractions)}, min bin count={counts.min()}, "
              f"empty bins={empty}, sparse bins (<{min_count})={len(sparse)}")
        if sparse:
            ranges = ', '.join(f'[{lo:.2f},{hi:.2f}):{c}' for lo, hi, c in sparse[:6])
            more = '' if len(sparse) <= 6 else f' (+{len(sparse)-6} more)'
            print(f"      FLAG sparse bins: {ranges}{more}")

    print(f"\n  Selection agreement (d={d})")
    for K in K_VALUES:
        sel = pooled[K]['selection']
        informative = (sel['cap_fraction'] > 0) | (sel['solid_fraction'] > 0)
        informative_frac = informative.mean()
        real_agree = (
            (sel['cap_winner_idx'][informative] == sel['solid_winner_idx'][informative]).mean()
            if informative.any() else float('nan')
        )
        print(f"    K={K}: informative={informative_frac:.4f}, agreement (informative only)={real_agree:.4f}")


def main() -> None:
    global K_VALUES, K_MAX, OUT_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dims', type=int, nargs='+', default=DIMS)
    parser.add_argument('--k-values', type=int, nargs='+', default=K_VALUES, dest='k_values')
    parser.add_argument('--n-backgrounds', type=int, default=N_BACKGROUNDS, dest='n_backgrounds')
    parser.add_argument('--n-background-points', type=int, default=N_BACKGROUND_POINTS, dest='n_background_points')
    parser.add_argument('--n-candidates', type=int, default=N_CANDIDATES, dest='n_candidates')
    parser.add_argument('--n-test-queries', type=int, default=N_TEST_QUERIES, dest='n_test_queries')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    K_VALUES = sorted(args.k_values)
    K_MAX = max(K_VALUES)
    if args.output_dir:
        OUT_DIR = Path(args.output_dir)

    run_dir = _run_dir()
    print(f"Run dir: {run_dir}")
    print(f"K values: {K_VALUES}, n_backgrounds={args.n_backgrounds}, "
          f"n_background_points={args.n_background_points}, n_candidates={args.n_candidates}, "
          f"n_test_queries={args.n_test_queries}\n")

    for d in args.dims:
        print(f"=== d={d} ===")
        pooled = run_dimension(d, args.n_backgrounds, args.n_background_points, args.n_candidates,
                                args.n_test_queries, base_seed=args.seed)
        save_dimension(d, pooled, run_dir)
        bin_count_diagnostic(d, pooled)
        print()

    print(f"Done: {run_dir}")


if __name__ == '__main__':
    main()
