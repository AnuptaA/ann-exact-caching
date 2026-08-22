#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Pool+argmax intersection-fraction calibration experiment on clustered
synthetic data (background, candidates, and test queries all drawn from
sample_clustered_sphere), cap intersection only.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from primitives import exact_intersection_fraction
from clustered_sphere_generator import sample_clustered_sphere

#-------------------------------------------------------------------------------

K_VALUES = [10, 25, 50, 100]
K_MAX = max(K_VALUES)
D = 128
N_BACKGROUNDS = 3
N_BACKGROUND_POINTS = 20000
N_CANDIDATES = 1000
N_TEST_QUERIES = 1000
N_CLUSTERS = 20
DENSE_KAPPA = 500.0
SPARSE_KAPPA = 3.0
DENSE_FRACTION = 0.5
MAX_PAIRWISE_SAMPLE = 20000

OUT_DIR = Path(__file__).parent / 'results'
PART_DIR = 'part1_clustered'
_RUN_TIMESTAMP = None


def _run_dir() -> Path:
    global _RUN_TIMESTAMP
    if _RUN_TIMESTAMP is None:
        _RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
    return OUT_DIR / _RUN_TIMESTAMP

#-------------------------------------------------------------------------------

def _chord_to_angle(chord: np.ndarray) -> np.ndarray:
    return 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))

def _bruteforce_knn(background: np.ndarray, queries: np.ndarray, k: int, chunk_size: int = 1000) -> tuple:
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

#-------------------------------------------------------------------------------

def run_background(background_id: int, n_points: int, n_candidates: int, n_test_queries: int, n_clusters: int,
                    dense_kappa: float, sparse_kappa: float, dense_fraction: float, seed: int) -> dict:
    """One clustered background, candidate pool, and test-query pool;
    computes every K stratum's selection and pairwise data.

    Returns {K: {'selection': {...}, 'pairwise': {...}}} for K in K_VALUES.
    """
    rng = np.random.default_rng(seed)
    background, _, _ = sample_clustered_sphere(n_points, D, n_clusters, dense_kappa, sparse_kappa,
                                                dense_fraction, rng)
    cache_q, _, _ = sample_clustered_sphere(n_candidates, D, n_clusters, dense_kappa, sparse_kappa,
                                             dense_fraction, rng)
    test_q, _, _ = sample_clustered_sphere(n_test_queries, D, n_clusters, dense_kappa, sparse_kappa,
                                            dense_fraction, rng)

    dists_cache, _ = _bruteforce_knn(background, cache_q, K_MAX)
    dists_test, idx_test = _bruteforce_knn(background, test_q, K_MAX)

    delta_matrix = np.arccos(np.clip(test_q @ cache_q.T, -1.0, 1.0))

    per_k = {}
    for K in K_VALUES:
        r_cache = _chord_to_angle(dists_cache[:, K - 1])
        r_test = _chord_to_angle(dists_test[:, K - 1])

        cap_matrix = np.array([
            [exact_intersection_fraction(float(r_cache[j]), float(r_test[i]), float(delta_matrix[i, j]), D)
             for j in range(n_candidates)]
            for i in range(n_test_queries)
        ])

        w_idx = cap_matrix.argmax(axis=1)
        winner_r_cache = r_cache[w_idx]
        winner_delta = delta_matrix[np.arange(n_test_queries), w_idx]
        winner_fraction = cap_matrix[np.arange(n_test_queries), w_idx]

        neighbor_coords = background[idx_test[:, :K]]
        winner_cache_q = cache_q[w_idx]
        dist_to_winner = np.linalg.norm(neighbor_coords - winner_cache_q[:, None, :], axis=2)
        winner_r_cache_chord = dists_cache[w_idx, K - 1]
        covered = dist_to_winner <= winner_r_cache_chord[:, None]
        recall = covered.mean(axis=1)

        selection = {
            'background_id': np.full(n_test_queries, background_id, dtype=np.int32),
            'r_test': r_test,
            'winner_idx': w_idx.astype(np.int32),
            'r_cache': winner_r_cache,
            'delta': winner_delta,
            'fraction': winner_fraction,
            'recall': recall,
        }

        n_total = n_test_queries * n_candidates
        n_sample = min(n_total, MAX_PAIRWISE_SAMPLE)
        flat_i, flat_j = np.divmod(rng.choice(n_total, size=n_sample, replace=False), n_candidates)
        pairwise = {
            'background_id': np.full(n_sample, background_id, dtype=np.int32),
            'r_cache': r_cache[flat_j],
            'r_test': r_test[flat_i],
            'delta': delta_matrix[flat_i, flat_j],
            'fraction': cap_matrix[flat_i, flat_j],
        }

        per_k[K] = {'selection': selection, 'pairwise': pairwise}

    return per_k


def run_all_backgrounds(n_backgrounds: int, n_points: int, n_candidates: int, n_test_queries: int, n_clusters: int,
                         dense_kappa: float, sparse_kappa: float, dense_fraction: float, base_seed: int) -> dict:
    selection_fields = ('background_id', 'r_test', 'winner_idx', 'r_cache', 'delta', 'fraction', 'recall')
    pairwise_fields = ('background_id', 'r_cache', 'r_test', 'delta', 'fraction')

    pooled = {K: {'selection': {f: [] for f in selection_fields}, 'pairwise': {f: [] for f in pairwise_fields}}
              for K in K_VALUES}

    for bg_id in range(n_backgrounds):
        print(f"  background {bg_id + 1}/{n_backgrounds}...")
        per_k = run_background(bg_id, n_points, n_candidates, n_test_queries, n_clusters, dense_kappa,
                                sparse_kappa, dense_fraction, seed=base_seed + bg_id)
        for K in K_VALUES:
            for f in selection_fields:
                pooled[K]['selection'][f].append(per_k[K]['selection'][f])
            for f in pairwise_fields:
                pooled[K]['pairwise'][f].append(per_k[K]['pairwise'][f])

    for K in K_VALUES:
        for part in ('selection', 'pairwise'):
            for f in pooled[K][part]:
                pooled[K][part][f] = np.concatenate(pooled[K][part][f])
    return pooled


def save_results(pooled: dict, run_dir: Path) -> None:
    dim_dir = run_dir / PART_DIR / f'{D}d' / 'raw'
    dim_dir.mkdir(parents=True, exist_ok=True)
    for K, parts in pooled.items():
        for part_name, arrays in parts.items():
            path = dim_dir / f'K{K}_{part_name}.npz'
            np.savez_compressed(path, d=D, K=K, **arrays)
            print(f"    wrote {path}")


def bin_count_diagnostic(pooled: dict, min_count: int = 30) -> None:
    bin_edges = np.arange(0.0, 1.05, 0.05)
    print(f"\n  Bin-count diagnostic (clustered synthetic d={D}):")
    for K in K_VALUES:
        fractions = pooled[K]['selection']['fraction']
        counts, _ = np.histogram(fractions, bins=bin_edges)
        empty = int(np.sum(counts == 0))
        print(f"    K={K}: total={len(fractions)}, min bin count={counts.min()}, empty bins={empty}, "
              f"informative={float((fractions > 0).mean()):.4f}, "
              f"fraction range=[{fractions.min():.4f}, {fractions.max():.4f}]")

#-------------------------------------------------------------------------------

def main() -> None:
    global K_VALUES, K_MAX, OUT_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--k-values', type=int, nargs='+', default=K_VALUES, dest='k_values')
    parser.add_argument('--n-backgrounds', type=int, default=N_BACKGROUNDS, dest='n_backgrounds')
    parser.add_argument('--n-background-points', type=int, default=N_BACKGROUND_POINTS, dest='n_background_points')
    parser.add_argument('--n-candidates', type=int, default=N_CANDIDATES, dest='n_candidates')
    parser.add_argument('--n-test-queries', type=int, default=N_TEST_QUERIES, dest='n_test_queries')
    parser.add_argument('--n-clusters', type=int, default=N_CLUSTERS, dest='n_clusters')
    parser.add_argument('--dense-kappa', type=float, default=DENSE_KAPPA, dest='dense_kappa')
    parser.add_argument('--sparse-kappa', type=float, default=SPARSE_KAPPA, dest='sparse_kappa')
    parser.add_argument('--dense-fraction', type=float, default=DENSE_FRACTION, dest='dense_fraction')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--run-dir', type=str, default=None)
    args = parser.parse_args()

    K_VALUES = sorted(args.k_values)
    K_MAX = max(K_VALUES)
    if args.output_dir:
        OUT_DIR = Path(args.output_dir)

    run_dir = Path(args.run_dir) if args.run_dir else _run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")
    print(f"K values: {K_VALUES}, n_backgrounds={args.n_backgrounds}, n_candidates={args.n_candidates}, "
          f"n_test_queries={args.n_test_queries}, n_clusters={args.n_clusters}, "
          f"dense_kappa={args.dense_kappa}, sparse_kappa={args.sparse_kappa}\n")

    pooled = run_all_backgrounds(args.n_backgrounds, args.n_background_points, args.n_candidates,
                                  args.n_test_queries, args.n_clusters, args.dense_kappa, args.sparse_kappa,
                                  args.dense_fraction, base_seed=args.seed)
    save_results(pooled, run_dir)
    bin_count_diagnostic(pooled)

    print(f"\nDone: {run_dir}")


if __name__ == '__main__':
    main()
