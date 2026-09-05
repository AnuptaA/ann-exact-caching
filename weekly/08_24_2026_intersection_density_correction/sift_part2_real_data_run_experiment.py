#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Reruns 08_10 Part 2's controlled query sweep against a real SIFT
anchor and corpus instead of a synthetic one."""

import sys
import math
import argparse
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from primitives import exact_intersection_fraction_batch_theta2
from part2_uniform_baseline_run_experiment import (
    _sample_uniform_sphere, _bruteforce_knn, _construct_test_poles, _chord_to_angle,
    bin_count_diagnostic, K_VALUES, K_MAX, N_BACKGROUNDS, N_BACKGROUND_POINTS,
    N_THETA, M_PER_THETA, GRID_MARGIN,
)
from density_correction_common import covered_mask

#-------------------------------------------------------------------------------

D = 128
BACKGROUND_PATH = 'datasets/sift/base_vectors_50k.npy'
ANCHOR_SEED_BASE = 900000  # independent of the generation-critical rng stream
UNIFORM_CORPUS_SEED = 800000  # independent of both the above
HOME_DELTA = 1e-6  # tiny perturbation off the exact background point, avoids self-match

OUT_DIR = Path(__file__).parent / 'results'
PART_DIR = 'sift_part2_real_data'
PART_DIR_UNIFORM = 'sift_part2_real_anchor_on_uniform'
_RUN_TIMESTAMP = None


def _run_dir() -> Path:
    global _RUN_TIMESTAMP
    if _RUN_TIMESTAMP is None:
        _RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
    return OUT_DIR / _RUN_TIMESTAMP

#-------------------------------------------------------------------------------

def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vecs / norms


def _householder_e1_to(target: np.ndarray) -> np.ndarray:
    """Symmetric orthogonal matrix H with H @ e_1 == target, preserving
    all angular distances."""
    d = len(target)
    e1 = np.zeros(d)
    e1[0] = 1.0
    v = e1 - target
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-12:
        return np.eye(d)
    v = v / norm_v
    return np.eye(d) - 2.0 * np.outer(v, v)


def _real_anchor(real_background: np.ndarray, background_id: int, d: int) -> np.ndarray:
    """A real background point perturbed by a tiny fixed delta, avoiding
    an exact self-match against its own source background."""
    anchor_rng = np.random.default_rng(ANCHOR_SEED_BASE + background_id)
    idx = int(anchor_rng.integers(len(real_background)))
    base_point = real_background[idx]
    direction = anchor_rng.standard_normal(d)
    direction -= (direction @ base_point) * base_point
    direction /= np.linalg.norm(direction)
    anchor = math.cos(HOME_DELTA) * base_point + math.sin(HOME_DELTA) * direction
    return _normalize(anchor[None, :])[0]

#-------------------------------------------------------------------------------

def _score_poles(corpus: np.ndarray, relocated: np.ndarray, anchor: np.ndarray, r_cache_by_K: dict,
                  delta: float, background_id: int, m_per_theta: int, d: int) -> dict:
    """Scores an already-fixed set of relocated poles against one
    corpus: r_test, cap_fraction, recall."""
    dists_test, idx_test = _bruteforce_knn(corpus, relocated, K_MAX)
    row = {K: {} for K in K_VALUES}
    for K in K_VALUES:
        r_test = _chord_to_angle(dists_test[:, K - 1])
        relevant_coords = corpus[idx_test[:, :K]]
        r_cache = r_cache_by_K[K]
        covered = covered_mask(relevant_coords, anchor, r_cache)
        recall = covered.mean(axis=1)
        cap_fraction = exact_intersection_fraction_batch_theta2(r_cache, r_test, delta, d)
        row[K] = {
            'background_id': np.full(m_per_theta, background_id, dtype=np.int32),
            'delta': np.full(m_per_theta, delta, dtype=np.float64),
            'r_cache': np.full(m_per_theta, r_cache, dtype=np.float64),
            'r_test': r_test, 'cap_fraction': cap_fraction, 'recall': recall,
        }
    return row


def run_background_real(d: int, background_id: int, real_background: np.ndarray, uniform_corpus: np.ndarray,
                         n_points: int, n_theta: int, m_per_theta: int, seed: int) -> tuple:
    """Reproduces 08_10 Part 2's delta grid and test poles, relocated
    onto a real anchor, scored against both the real and a size-matched
    uniform corpus."""
    rng = np.random.default_rng(seed)  # identical seeding to the baseline
    synthetic_background = _sample_uniform_sphere(n_points, d, rng)  # only to reproduce grid_max/delta_grid exactly

    e1 = np.zeros(d)
    e1[0] = 1.0
    dists_cache, _ = _bruteforce_knn(synthetic_background, e1[None, :], K_MAX)
    r_cache_by_K_synthetic = {K: float(_chord_to_angle(dists_cache[0, K - 1])) for K in K_VALUES}
    grid_max = min(GRID_MARGIN * 2.0 * max(r_cache_by_K_synthetic.values()), math.pi)
    delta_grid = np.linspace(0.0, grid_max, n_theta)

    real_anchor = _real_anchor(real_background, background_id, d)
    H = _householder_e1_to(real_anchor)

    dists_anchor_real, _ = _bruteforce_knn(real_background, real_anchor[None, :], K_MAX)
    r_cache_by_K_real = {K: float(_chord_to_angle(dists_anchor_real[0, K - 1])) for K in K_VALUES}

    dists_anchor_uniform, _ = _bruteforce_knn(uniform_corpus, real_anchor[None, :], K_MAX)
    r_cache_by_K_uniform = {K: float(_chord_to_angle(dists_anchor_uniform[0, K - 1])) for K in K_VALUES}

    fields = ('background_id', 'delta', 'r_cache', 'r_test', 'cap_fraction', 'recall')
    out_real = {K: {f: [] for f in fields} for K in K_VALUES}
    out_uniform = {K: {f: [] for f in fields} for K in K_VALUES}

    for delta in delta_grid:
        delta = float(delta)
        test_poles = _construct_test_poles(delta, m_per_theta, d, rng)  # same call/rng state as the baseline
        relocated = _normalize(test_poles @ H)  # H symmetric; preserves each pole's delta from real_anchor exactly

        row_real = _score_poles(real_background, relocated, real_anchor, r_cache_by_K_real, delta,
                                 background_id, m_per_theta, d)
        row_uniform = _score_poles(uniform_corpus, relocated, real_anchor, r_cache_by_K_uniform, delta,
                                    background_id, m_per_theta, d)
        for K in K_VALUES:
            for f in fields:
                out_real[K][f].append(row_real[K][f])
                out_uniform[K][f].append(row_uniform[K][f])

    for K in K_VALUES:
        for f in fields:
            out_real[K][f] = np.concatenate(out_real[K][f])
            out_uniform[K][f] = np.concatenate(out_uniform[K][f])
    return out_real, out_uniform


def run_dimension_real(d: int, real_background: np.ndarray, n_backgrounds: int, n_points: int, n_theta: int,
                        m_per_theta: int, base_seed: int) -> tuple:
    uniform_corpus_rng = np.random.default_rng(UNIFORM_CORPUS_SEED)
    uniform_corpus = _sample_uniform_sphere(len(real_background), d, uniform_corpus_rng)

    fields = ('background_id', 'delta', 'r_cache', 'r_test', 'cap_fraction', 'recall')
    pooled_real = {K: {f: [] for f in fields} for K in K_VALUES}
    pooled_uniform = {K: {f: [] for f in fields} for K in K_VALUES}

    for bg_id in range(n_backgrounds):
        print(f"  background {bg_id + 1}/{n_backgrounds} (d={d})...")
        per_bg_real, per_bg_uniform = run_background_real(d, bg_id, real_background, uniform_corpus, n_points,
                                                            n_theta, m_per_theta, seed=base_seed + bg_id)
        for K in K_VALUES:
            for field in fields:
                pooled_real[K][field].append(per_bg_real[K][field])
                pooled_uniform[K][field].append(per_bg_uniform[K][field])

    for K in K_VALUES:
        for field in fields:
            pooled_real[K][field] = np.concatenate(pooled_real[K][field])
            pooled_uniform[K][field] = np.concatenate(pooled_uniform[K][field])
    return pooled_real, pooled_uniform


def save_dimension(d: int, pooled: dict, run_dir: Path, part_dir: str) -> None:
    dim_dir = run_dir / part_dir / f'{d}d' / 'raw'
    dim_dir.mkdir(parents=True, exist_ok=True)
    for K, arrays in pooled.items():
        path = dim_dir / f'K{K}.npz'
        np.savez_compressed(path, d=d, K=K, **arrays)
        print(f"    wrote {path}")

#-------------------------------------------------------------------------------

def main() -> None:
    global K_VALUES, K_MAX, OUT_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-backgrounds', type=int, default=N_BACKGROUNDS, dest='n_backgrounds')
    parser.add_argument('--n-background-points', type=int, default=N_BACKGROUND_POINTS, dest='n_background_points')
    parser.add_argument('--n-theta', type=int, default=N_THETA, dest='n_theta')
    parser.add_argument('--m-per-theta', type=int, default=M_PER_THETA, dest='m_per_theta')
    parser.add_argument('--background-path', type=str, default=BACKGROUND_PATH, dest='background_path')
    parser.add_argument('--seed', type=int, default=42, help='must match the baseline run to reproduce its queries')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--run-dir', type=str, default=None)
    args = parser.parse_args()

    if args.output_dir:
        OUT_DIR = Path(args.output_dir)

    run_dir = Path(args.run_dir) if args.run_dir else _run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")
    print(f"K values: {K_VALUES}, n_backgrounds={args.n_backgrounds}, "
          f"n_background_points={args.n_background_points}, n_theta={args.n_theta}, "
          f"m_per_theta={args.m_per_theta}\n")

    repo_root = Path(__file__).parent.parent.parent
    bg_path = Path(args.background_path)
    if not bg_path.is_absolute():
        bg_path = repo_root / bg_path
    real_background = _normalize(np.load(bg_path).astype(np.float64))

    print(f"=== d={D} (SIFT real corpus + size-matched uniform corpus, same anchor/queries) ===")
    pooled_real, pooled_uniform = run_dimension_real(D, real_background, args.n_backgrounds,
                                                       args.n_background_points, args.n_theta, args.m_per_theta,
                                                       base_seed=args.seed)
    save_dimension(D, pooled_real, run_dir, PART_DIR)
    print("  [real corpus]")
    bin_count_diagnostic(D, pooled_real)
    save_dimension(D, pooled_uniform, run_dir, PART_DIR_UNIFORM)
    print("  [uniform corpus, same anchor/queries]")
    bin_count_diagnostic(D, pooled_uniform)

    print(f"\nDone: {run_dir}")


if __name__ == '__main__':
    main()
