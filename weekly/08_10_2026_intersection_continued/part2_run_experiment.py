#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Controlled-construction intersection-fraction calibration: one fixed
cache candidate at e_1 = (1, 0, ..., 0), test poles placed at exact
controlled angular distances instead of drawn at random.
"""

import sys
import math
import argparse
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent))

import numpy as np

from primitives import (
    exact_intersection_fraction,
    solid_ball_intersection_fraction,
    chord_distance_from_angular,
)

#-------------------------------------------------------------------------------

K_VALUES = [1, 5, 10, 25, 50, 100]
K_MAX = max(K_VALUES)
N_BACKGROUNDS = 5
N_BACKGROUND_POINTS = 100000
N_THETA = 40
M_PER_THETA = 50
DIMS = [2, 3, 8, 16, 32, 64, 128, 256, 384]
GRID_MARGIN = 1.2  # sweep out to GRID_MARGIN * 2 * r_cache, capped at pi

OUT_DIR = Path(__file__).parent / 'results'
PART_DIR = 'part2'
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


def _bruteforce_knn(background: np.ndarray, queries: np.ndarray, k: int, chunk_size: int = 2000) -> tuple:
    """Top-k nearest neighbors of each query against background, by chord
    distance, equivalently largest dot product since both live on the
    unit sphere. Returns (dists, indices), each (n_queries, k), dists
    ascending."""
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
    """m points at exactly angular distance delta from e_1 = (1, 0, ..., 0),
    via a random direction in the orthogonal (d-2)-sphere. Every point at
    that angular distance is statistically equivalent by rotational
    symmetry, so this gives m independent, valid samples per delta bin."""
    perp = _sample_uniform_sphere(m, d - 1, rng)  # (m, d-1), on S^(d-2)
    poles = np.empty((m, d), dtype=np.float64)
    poles[:, 0] = math.cos(delta)
    poles[:, 1:] = math.sin(delta) * perp
    return poles

#-------------------------------------------------------------------------------


def run_background(d: int, background_id: int, n_points: int, n_theta: int, m_per_theta: int, seed: int) -> dict:
    """One background: samples the corpus, fixes the cache pole, computes
    its K-th-NN radius per K, sweeps delta, and computes predicted
    fraction + real recall for m_per_theta controlled test poles per
    delta.

    Returns {K: {field: array}} pooling over every delta bin.
    """
    rng = np.random.default_rng(seed)
    background = _sample_uniform_sphere(n_points, d, rng)

    cache_pole = np.zeros(d)
    cache_pole[0] = 1.0

    dists_cache, _ = _bruteforce_knn(background, cache_pole[None, :], K_MAX)
    r_cache_by_K = {K: float(_chord_to_angle(dists_cache[0, K - 1])) for K in K_VALUES}

    grid_max = min(GRID_MARGIN * 2.0 * max(r_cache_by_K.values()), math.pi)
    delta_grid = np.linspace(0.0, grid_max, n_theta)

    fields = ('background_id', 'delta', 'r_cache', 'r_test', 'cap_fraction', 'solid_fraction', 'recall')
    out = {K: {f: [] for f in fields} for K in K_VALUES}

    for delta in delta_grid:
        delta = float(delta)
        test_poles = _construct_test_poles(delta, m_per_theta, d, rng)
        dists_test, idx_test = _bruteforce_knn(background, test_poles, K_MAX)
        delta_chord = chord_distance_from_angular(delta)

        for K in K_VALUES:
            r_test = _chord_to_angle(dists_test[:, K - 1])  # (m,)
            r_test_chord = np.array([chord_distance_from_angular(float(r)) for r in r_test])
            relevant_coords = background[idx_test[:, :K]]  # (m, K, d)
            dist_to_cache = np.linalg.norm(relevant_coords - cache_pole[None, None, :], axis=2)  # (m, K)

            r_cache = r_cache_by_K[K]
            r_cache_chord = chord_distance_from_angular(r_cache)

            covered = dist_to_cache <= r_cache_chord
            recall = covered.mean(axis=1)  # (m,)

            cap_fraction = np.array([
                exact_intersection_fraction(r_cache, float(r_test[i]), delta, d)
                for i in range(m_per_theta)
            ])
            solid_fraction = np.array([
                solid_ball_intersection_fraction(r_cache_chord, float(r_test_chord[i]), delta_chord, d)
                for i in range(m_per_theta)
            ])

            dst = out[K]
            dst['background_id'].append(np.full(m_per_theta, background_id, dtype=np.int32))
            dst['delta'].append(np.full(m_per_theta, delta, dtype=np.float64))
            dst['r_cache'].append(np.full(m_per_theta, r_cache, dtype=np.float64))
            dst['r_test'].append(r_test)
            dst['cap_fraction'].append(cap_fraction)
            dst['solid_fraction'].append(solid_fraction)
            dst['recall'].append(recall)

    for K in K_VALUES:
        for f in fields:
            out[K][f] = np.concatenate(out[K][f])

    return out


def run_dimension(d: int, n_backgrounds: int, n_points: int, n_theta: int, m_per_theta: int, base_seed: int) -> dict:
    """Runs every background for one dimension and pools their per-K
    arrays into one array per field."""
    fields = ('background_id', 'delta', 'r_cache', 'r_test', 'cap_fraction', 'solid_fraction', 'recall')
    pooled = {K: {f: [] for f in fields} for K in K_VALUES}

    for bg_id in range(n_backgrounds):
        print(f"  background {bg_id + 1}/{n_backgrounds} (d={d})...")
        per_bg = run_background(d, bg_id, n_points, n_theta, m_per_theta, seed=base_seed + bg_id)
        for K in K_VALUES:
            for field in fields:
                pooled[K][field].append(per_bg[K][field])

    for K in K_VALUES:
        for field in fields:
            pooled[K][field] = np.concatenate(pooled[K][field])
    return pooled


def save_dimension(d: int, pooled: dict, run_dir: Path) -> None:
    """One .npz per K persisted."""
    dim_dir = run_dir / PART_DIR / f'{d}d' / 'raw'
    dim_dir.mkdir(parents=True, exist_ok=True)
    for K, arrays in pooled.items():
        path = dim_dir / f'K{K}.npz'
        np.savez_compressed(path, d=d, K=K, **arrays)
        print(f"    wrote {path}")


def bin_count_diagnostic(d: int, pooled: dict, min_count: int = 30) -> None:
    """Prints per-K bin population counts (5% predicted-fraction
    increments) and flags sparse or empty bins."""
    bin_edges = np.arange(0.0, 1.05, 0.05)
    print(f"\n  Bin-count diagnostic (d={d}, 5% predicted-fraction bins, cap method):")
    for K in K_VALUES:
        fractions = pooled[K]['cap_fraction']
        counts, _ = np.histogram(fractions, bins=bin_edges)
        sparse = [(bin_edges[i], bin_edges[i + 1], c) for i, c in enumerate(counts) if 0 < c < min_count]
        empty = int(np.sum(counts == 0))
        print(f"    K={K}: total={len(fractions)}, min bin count={counts.min()}, "
              f"empty bins={empty}, sparse bins (<{min_count})={len(sparse)}")
        if sparse:
            ranges = ', '.join(f'[{lo:.2f},{hi:.2f}):{c}' for lo, hi, c in sparse[:6])
            more = '' if len(sparse) <= 6 else f' (+{len(sparse)-6} more)'
            print(f"      FLAG sparse bins: {ranges}{more}")


def main() -> None:
    global K_VALUES, K_MAX, OUT_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dims', type=int, nargs='+', default=DIMS)
    parser.add_argument('--k-values', type=int, nargs='+', default=K_VALUES, dest='k_values')
    parser.add_argument('--n-backgrounds', type=int, default=N_BACKGROUNDS, dest='n_backgrounds')
    parser.add_argument('--n-background-points', type=int, default=N_BACKGROUND_POINTS, dest='n_background_points')
    parser.add_argument('--n-theta', type=int, default=N_THETA, dest='n_theta')
    parser.add_argument('--m-per-theta', type=int, default=M_PER_THETA, dest='m_per_theta')
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
    print(f"K values: {K_VALUES}, n_backgrounds={args.n_backgrounds}, "
          f"n_background_points={args.n_background_points}, "
          f"n_theta={args.n_theta}, m_per_theta={args.m_per_theta}\n")

    for d in args.dims:
        print(f"=== d={d} ===")
        pooled = run_dimension(d, args.n_backgrounds, args.n_background_points,
                                args.n_theta, args.m_per_theta, base_seed=args.seed)
        save_dimension(d, pooled, run_dir)
        bin_count_diagnostic(d, pooled)
        print()

    print(f"Done: {run_dir}")


if __name__ == '__main__':
    main()
