#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""r_cache diagnostic: fixed cache pole, fixed delta ratios, r_cache swept
via target coverage fraction instead of the K-th-NN mechanism.
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

K_VALUES = [1, 5, 10, 25, 50, 100]
K_MAX = max(K_VALUES)
N_BACKGROUNDS = 5
N_BACKGROUND_POINTS = 100000
M_PER_CONFIG = 50
DIMS = [2, 3, 8, 16, 32, 64, 128, 256, 384]
DELTA_RATIOS = [0.5, 1.0, 1.5]  # relative to the typical r_test scale for a given K
COVERAGE_TARGETS = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50]

OUT_DIR = Path(__file__).parent / 'results'
PART_DIR = 'part3'
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


def _analytic_radius(coverage: float, d: int) -> float:
    """Radius r solving single_cap_fraction(r, d) == coverage. Monotone
    increasing and continuous on [0, pi], so brentq applies over the
    whole domain."""
    coverage = min(max(coverage, 1e-12), 1.0 - 1e-12)
    return brentq(lambda r: single_cap_fraction(r, d) - coverage, 1e-9, math.pi - 1e-9)


def _construct_test_poles(delta: float, m: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """m points at exactly angular distance delta from e_1 = (1, 0, ..., 0),
    via a random direction in the orthogonal (d-2)-sphere. Every point at
    that angular distance is statistically equivalent by rotational
    symmetry, so this gives m independent, valid samples per delta."""
    perp = _sample_uniform_sphere(m, d - 1, rng)  # (m, d-1), on S^(d-2)
    poles = np.empty((m, d), dtype=np.float64)
    poles[:, 0] = math.cos(delta)
    poles[:, 1:] = math.sin(delta) * perp
    return poles


def _coverage_targets_for_K(K: int, n_points: int) -> list:
    """Coverage-fraction targets that exceed K/n_points, guaranteeing the
    resulting r_cache exceeds the naive K-th-NN r_test scale."""
    floor = K / n_points
    return [t for t in COVERAGE_TARGETS if t > floor]

#-------------------------------------------------------------------------------


def run_background(d: int, background_id: int, n_points: int, m_per_config: int, seed: int) -> dict:
    """One background: samples the corpus, fixes the cache pole, computes a
    per-K reference radius (used both as the delta scale and as the r_test
    ground truth), then for each K sweeps fixed delta ratios and coverage-
    fraction-based r_cache values.

    Returns {K: {field: array}} pooling over every (delta_ratio,
    coverage_target) combination.
    """
    rng = np.random.default_rng(seed)
    background = _sample_uniform_sphere(n_points, d, rng)

    cache_pole = np.zeros(d)
    cache_pole[0] = 1.0

    dists_cache, _ = _bruteforce_knn(background, cache_pole[None, :], K_MAX)
    r_typical_by_K = {K: float(_chord_to_angle(dists_cache[0, K - 1])) for K in K_VALUES}

    fields = ('background_id', 'delta_ratio', 'delta', 'coverage_target',
              'r_cache', 'r_test', 'cap_fraction', 'solid_fraction', 'recall')
    out = {K: {f: [] for f in fields} for K in K_VALUES}

    for K in K_VALUES:
        r_typical = r_typical_by_K[K]
        coverage_targets = _coverage_targets_for_K(K, n_points)

        for delta_ratio in DELTA_RATIOS:
            delta = min(delta_ratio * r_typical, math.pi * 0.999)
            test_poles = _construct_test_poles(delta, m_per_config, d, rng)
            dists_test, idx_test = _bruteforce_knn(background, test_poles, K)
            delta_chord = chord_distance_from_angular(delta)

            r_test = _chord_to_angle(dists_test[:, K - 1])  # (m,)
            r_test_chord = np.array([chord_distance_from_angular(float(r)) for r in r_test])
            relevant_coords = background[idx_test[:, :K]]  # (m, K, d)
            dist_to_cache = np.linalg.norm(relevant_coords - cache_pole[None, None, :], axis=2)  # (m, K)

            for target in coverage_targets:
                r_cache = _analytic_radius(target, d)
                r_cache_chord = chord_distance_from_angular(r_cache)

                covered = dist_to_cache <= r_cache_chord
                recall = covered.mean(axis=1)  # (m,)

                cap_fraction = np.array([
                    exact_intersection_fraction(r_cache, float(r_test[i]), delta, d)
                    for i in range(m_per_config)
                ])
                solid_fraction = np.array([
                    solid_ball_intersection_fraction(r_cache_chord, float(r_test_chord[i]), delta_chord, d)
                    for i in range(m_per_config)
                ])

                dst = out[K]
                dst['background_id'].append(np.full(m_per_config, background_id, dtype=np.int32))
                dst['delta_ratio'].append(np.full(m_per_config, delta_ratio, dtype=np.float64))
                dst['delta'].append(np.full(m_per_config, delta, dtype=np.float64))
                dst['coverage_target'].append(np.full(m_per_config, target, dtype=np.float64))
                dst['r_cache'].append(np.full(m_per_config, r_cache, dtype=np.float64))
                dst['r_test'].append(r_test)
                dst['cap_fraction'].append(cap_fraction)
                dst['solid_fraction'].append(solid_fraction)
                dst['recall'].append(recall)

    for K in K_VALUES:
        for f in fields:
            out[K][f] = np.concatenate(out[K][f])

    return out


def run_dimension(d: int, n_backgrounds: int, n_points: int, m_per_config: int, base_seed: int) -> dict:
    """Runs every background for one dimension and pools their per-K arrays
    into one array per field."""
    fields = ('background_id', 'delta_ratio', 'delta', 'coverage_target',
              'r_cache', 'r_test', 'cap_fraction', 'solid_fraction', 'recall')
    pooled = {K: {f: [] for f in fields} for K in K_VALUES}

    for bg_id in range(n_backgrounds):
        print(f"  background {bg_id + 1}/{n_backgrounds} (d={d})...")
        per_bg = run_background(d, bg_id, n_points, m_per_config, seed=base_seed + bg_id)
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


def constraint_diagnostic(d: int, pooled: dict) -> None:
    """Confirms r_cache > r_test holds everywhere (the requested
    constraint), and prints the coverage-fraction -> recall transition
    summary per K to inform part4's target selection."""
    print(f"\n  r_cache > r_test constraint check (d={d}):")
    for K in K_VALUES:
        sel = pooled[K]
        violations = int(np.sum(sel['r_cache'] <= sel['r_test']))
        print(f"    K={K}: {violations} violations out of {len(sel['r_cache'])}")

    print(f"\n  Coverage target -> mean recall by K (d={d}, pooled across delta_ratio and backgrounds):")
    for K in K_VALUES:
        sel = pooled[K]
        targets = np.unique(sel['coverage_target'])
        print(f"    K={K}:")
        for t in targets:
            mask = sel['coverage_target'] == t
            print(f"      target={t:.4f}: mean_recall={sel['recall'][mask].mean():.4f}, "
                  f"mean_cap_fraction={sel['cap_fraction'][mask].mean():.4f}")


def main() -> None:
    global K_VALUES, K_MAX, OUT_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dims', type=int, nargs='+', default=DIMS)
    parser.add_argument('--k-values', type=int, nargs='+', default=K_VALUES, dest='k_values')
    parser.add_argument('--n-backgrounds', type=int, default=N_BACKGROUNDS, dest='n_backgrounds')
    parser.add_argument('--n-background-points', type=int, default=N_BACKGROUND_POINTS, dest='n_background_points')
    parser.add_argument('--m-per-config', type=int, default=M_PER_CONFIG, dest='m_per_config')
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
          f"n_background_points={args.n_background_points}, m_per_config={args.m_per_config}\n")

    for d in args.dims:
        print(f"=== d={d} ===")
        pooled = run_dimension(d, args.n_backgrounds, args.n_background_points,
                                args.m_per_config, base_seed=args.seed)
        save_dimension(d, pooled, run_dir)
        constraint_diagnostic(d, pooled)
        print()

    print(f"Done: {run_dir}")


if __name__ == '__main__':
    main()
