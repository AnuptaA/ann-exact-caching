#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Controlled-construction intersection-fraction calibration on a
clustered synthetic background: one fixed cache candidate at
e_1 = (1, 0, ..., 0), test poles at exact controlled angular distances,
same as part2_run_experiment.py, but background drawn from
sample_clustered_sphere instead of uniformly.
"""

import sys
import math
import argparse
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from primitives import exact_intersection_fraction, chord_distance_from_angular
from clustered_sphere_generator import sample_clustered_sphere, _sample_uniform_sphere

#-------------------------------------------------------------------------------

K_VALUES = [10, 25, 50, 100]
K_MAX = max(K_VALUES)
D = 128
N_BACKGROUNDS = 3
N_BACKGROUND_POINTS = 20000
N_THETA = 40
M_PER_THETA = 50
N_CLUSTERS = 20
DENSE_KAPPA = 500.0
SPARSE_KAPPA = 3.0
DENSE_FRACTION = 0.5
GRID_MARGIN = 1.2

OUT_DIR = Path(__file__).parent / 'results'
PART_DIR = 'part2_clustered'
_RUN_TIMESTAMP = None


def _run_dir() -> Path:
    global _RUN_TIMESTAMP
    if _RUN_TIMESTAMP is None:
        _RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
    return OUT_DIR / _RUN_TIMESTAMP

#-------------------------------------------------------------------------------

def _chord_to_angle(chord: np.ndarray) -> np.ndarray:
    return 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))

def _bruteforce_knn(background: np.ndarray, queries: np.ndarray, k: int, chunk_size: int = 2000) -> tuple:
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

def _construct_test_poles(delta: float, m: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """m points at exactly angular distance delta from e_1 = (1, 0, ..., 0)."""
    perp = _sample_uniform_sphere(m, d - 1, rng)
    poles = np.empty((m, d), dtype=np.float64)
    poles[:, 0] = math.cos(delta)
    poles[:, 1:] = math.sin(delta) * perp
    return poles

#-------------------------------------------------------------------------------

def run_background(background_id: int, n_points: int, n_theta: int, m_per_theta: int, n_clusters: int,
                    dense_kappa: float, sparse_kappa: float, dense_fraction: float, seed: int) -> dict:
    """One clustered background: fixes the cache pole, computes its K-th-NN
    radius per K against the clustered background, sweeps delta, and
    computes predicted fraction + real recall for m_per_theta controlled
    test poles per delta.

    Returns {K: {field: array}} pooling over every delta bin.
    """
    rng = np.random.default_rng(seed)
    background, _, _ = sample_clustered_sphere(n_points, D, n_clusters, dense_kappa, sparse_kappa,
                                                dense_fraction, rng)

    cache_pole = np.zeros(D)
    cache_pole[0] = 1.0

    dists_cache, _ = _bruteforce_knn(background, cache_pole[None, :], K_MAX)
    r_cache_by_K = {K: float(_chord_to_angle(dists_cache[0, K - 1])) for K in K_VALUES}

    grid_max = min(GRID_MARGIN * 2.0 * max(r_cache_by_K.values()), math.pi)
    delta_grid = np.linspace(0.0, grid_max, n_theta)

    fields = ('background_id', 'delta', 'r_cache', 'r_test', 'fraction', 'recall')
    out = {K: {f: [] for f in fields} for K in K_VALUES}

    for delta in delta_grid:
        delta = float(delta)
        test_poles = _construct_test_poles(delta, m_per_theta, D, rng)
        dists_test, idx_test = _bruteforce_knn(background, test_poles, K_MAX)

        for K in K_VALUES:
            r_test = _chord_to_angle(dists_test[:, K - 1])
            relevant_coords = background[idx_test[:, :K]]
            dist_to_cache = np.linalg.norm(relevant_coords - cache_pole[None, None, :], axis=2)

            r_cache = r_cache_by_K[K]
            r_cache_chord = chord_distance_from_angular(r_cache)

            covered = dist_to_cache <= r_cache_chord
            recall = covered.mean(axis=1)

            fraction = np.array([
                exact_intersection_fraction(r_cache, float(r_test[i]), delta, D)
                for i in range(m_per_theta)
            ])

            dst = out[K]
            dst['background_id'].append(np.full(m_per_theta, background_id, dtype=np.int32))
            dst['delta'].append(np.full(m_per_theta, delta, dtype=np.float64))
            dst['r_cache'].append(np.full(m_per_theta, r_cache, dtype=np.float64))
            dst['r_test'].append(r_test)
            dst['fraction'].append(fraction)
            dst['recall'].append(recall)

    for K in K_VALUES:
        for f in fields:
            out[K][f] = np.concatenate(out[K][f])

    return out


def run_all_backgrounds(n_backgrounds: int, n_points: int, n_theta: int, m_per_theta: int, n_clusters: int,
                         dense_kappa: float, sparse_kappa: float, dense_fraction: float, base_seed: int) -> dict:
    fields = ('background_id', 'delta', 'r_cache', 'r_test', 'fraction', 'recall')
    pooled = {K: {f: [] for f in fields} for K in K_VALUES}

    for bg_id in range(n_backgrounds):
        print(f"  background {bg_id + 1}/{n_backgrounds}...")
        per_bg = run_background(bg_id, n_points, n_theta, m_per_theta, n_clusters, dense_kappa, sparse_kappa,
                                 dense_fraction, seed=base_seed + bg_id)
        for K in K_VALUES:
            for field in fields:
                pooled[K][field].append(per_bg[K][field])

    for K in K_VALUES:
        for field in fields:
            pooled[K][field] = np.concatenate(pooled[K][field])
    return pooled


def save_results(pooled: dict, run_dir: Path) -> None:
    dim_dir = run_dir / PART_DIR / f'{D}d' / 'raw'
    dim_dir.mkdir(parents=True, exist_ok=True)
    for K, arrays in pooled.items():
        path = dim_dir / f'K{K}.npz'
        np.savez_compressed(path, d=D, K=K, **arrays)
        print(f"    wrote {path}")


def bin_count_diagnostic(pooled: dict, min_count: int = 30) -> None:
    bin_edges = np.arange(0.0, 1.05, 0.05)
    print(f"\n  Bin-count diagnostic (clustered synthetic d={D}):")
    for K in K_VALUES:
        fractions = pooled[K]['fraction']
        counts, _ = np.histogram(fractions, bins=bin_edges)
        empty = int(np.sum(counts == 0))
        print(f"    K={K}: total={len(fractions)}, min bin count={counts.min()}, empty bins={empty}, "
              f"fraction range=[{fractions.min():.4f}, {fractions.max():.4f}]")

#-------------------------------------------------------------------------------

def main() -> None:
    global K_VALUES, K_MAX, OUT_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--k-values', type=int, nargs='+', default=K_VALUES, dest='k_values')
    parser.add_argument('--n-backgrounds', type=int, default=N_BACKGROUNDS, dest='n_backgrounds')
    parser.add_argument('--n-background-points', type=int, default=N_BACKGROUND_POINTS, dest='n_background_points')
    parser.add_argument('--n-theta', type=int, default=N_THETA, dest='n_theta')
    parser.add_argument('--m-per-theta', type=int, default=M_PER_THETA, dest='m_per_theta')
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
    print(f"K values: {K_VALUES}, n_backgrounds={args.n_backgrounds}, n_background_points={args.n_background_points}, "
          f"n_theta={args.n_theta}, m_per_theta={args.m_per_theta}, n_clusters={args.n_clusters}, "
          f"dense_kappa={args.dense_kappa}, sparse_kappa={args.sparse_kappa}\n")

    pooled = run_all_backgrounds(args.n_backgrounds, args.n_background_points, args.n_theta, args.m_per_theta,
                                  args.n_clusters, args.dense_kappa, args.sparse_kappa, args.dense_fraction,
                                  base_seed=args.seed)
    save_results(pooled, run_dir)
    bin_count_diagnostic(pooled)

    print(f"\nDone: {run_dir}")


if __name__ == '__main__':
    main()
