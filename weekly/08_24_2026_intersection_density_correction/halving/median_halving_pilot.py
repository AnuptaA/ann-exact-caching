#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Baseline and adversarial pilots for the median-halving tree, before
committing to the full 352-config grid. See the plan for exact scope."""

import sys
import time
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'common'))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import numpy as np

from datasets.dataloaders import load_fvecs
from median_halving_tree import (
    build_direction_pool, build_tree, touching_leaves_batch, counts_from_hits,
    leaf_touch_counts_from_hits, brute_force_count, tree_memory_bytes,
)
from part2_uniform_baseline_run_experiment import _bruteforce_knn
from primitives import sample_cap_points_direct

#-------------------------------------------------------------------------------

N = 50000
K = 100
POOL_SIZE = 256

#-------------------------------------------------------------------------------


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vecs / norms


def load_sift_points_and_queries(n_queries: int, seed: int) -> tuple:
    repo_root = Path(__file__).parent.parent.parent.parent
    base = _normalize(np.load(repo_root / 'datasets/sift/base_vectors_50k.npy').astype(np.float64))
    all_queries = _normalize(load_fvecs(str(repo_root / 'datasets/sift/sift_query.fvecs')).astype(np.float64))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_queries), size=n_queries, replace=False)
    queries = all_queries[idx]
    return base, queries


def query_radii(base: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """K-th NN angular radius via direct arccos(dot), matching
    count_in_radius/brute_force_count's own distance formula exactly.
    Using _chord_to_angle's chord-then-arcsin path here instead
    introduces a ULP-level discrepancy that lands right at the K-th
    neighbor's boundary distance (since the radius is defined to be
    exactly that neighbor's distance), spuriously off-by-one-ing the
    ground truth itself, not just the tree's estimate."""
    _, idx = _bruteforce_knn(base, queries, k)
    kth_points = base[idx[:, k - 1]]
    dots = np.clip(np.einsum('ij,ij->i', queries, kth_points), -1.0, 1.0)
    return np.arccos(dots)


def run_pilot_config(points: np.ndarray, queries: np.ndarray, radii: np.ndarray,
                      C: int, halving: bool, pool: np.ndarray, seed: int) -> dict:
    d = points.shape[1]
    rng = np.random.default_rng(seed)

    t0 = time.perf_counter()
    tree = build_tree(points, C=C, halving=halving, direction_pool=pool, rng=rng)
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    hit_q, hit_n = touching_leaves_batch(tree, queries, radii)
    t_traverse = time.perf_counter() - t0

    t0 = time.perf_counter()
    est_counts = counts_from_hits(tree, queries, radii, hit_q, hit_n)
    t_count = time.perf_counter() - t0

    leaf_touch_counts = leaf_touch_counts_from_hits(hit_q, len(queries))
    node_sizes = np.zeros(tree.n_nodes(), dtype=np.int64)
    for nid, idx in tree.leaf_point_idx.items():
        node_sizes[nid] = len(idx)
    for nid, (_, mult) in tree.leaf_unique.items():
        node_sizes[nid] = int(mult.sum())
    point_checks = np.bincount(hit_q, weights=node_sizes[hit_n], minlength=len(queries)).astype(np.int64)

    n_capped_leaves = len(tree.leaf_depth_capped)
    n_capped_points = sum(len(tree.leaf_point_idx[nid]) for nid in tree.leaf_depth_capped)

    return {
        'd': d, 'C': C, 'halving': halving, 'n_nodes': tree.n_nodes(),
        'max_depth_reached': tree.max_depth_reached,
        't_build': t_build, 't_traverse': t_traverse, 't_count': t_count,
        'leaf_touch_mean': leaf_touch_counts.mean(), 'leaf_touch_p10': np.percentile(leaf_touch_counts, 10),
        'leaf_touch_p90': np.percentile(leaf_touch_counts, 90),
        'point_checks_mean': point_checks.mean(),
        'n_leaf_unique': len(tree.leaf_unique), 'n_capped_leaves': n_capped_leaves,
        'n_capped_points': n_capped_points,
        'mem_full_direction': tree_memory_bytes(tree, 'full_direction'),
        'mem_pooled_index': tree_memory_bytes(tree, 'pooled_index'),
        'est_counts': est_counts,
    }


def print_result(label: str, r: dict, true_counts: np.ndarray = None) -> None:
    print(f"  [{label}] d={r['d']} C={r['C']} halving={r['halving']}: "
          f"n_nodes={r['n_nodes']}, max_depth={r['max_depth_reached']}, "
          f"t_build={r['t_build']:.3f}s, t_traverse={r['t_traverse']:.3f}s, t_count={r['t_count']:.3f}s")
    print(f"    leaf_touch mean/p10/p90={r['leaf_touch_mean']:.2f}/{r['leaf_touch_p10']:.2f}/{r['leaf_touch_p90']:.2f}, "
          f"point_checks_mean={r['point_checks_mean']:.1f}")
    print(f"    n_leaf_unique={r['n_leaf_unique']}, n_capped_leaves={r['n_capped_leaves']}, "
          f"n_capped_points={r['n_capped_points']} ({r['n_capped_points'] / N:.4%} of N)")
    print(f"    memory: full_direction={r['mem_full_direction']} bytes, pooled_index={r['mem_pooled_index']} bytes")
    if true_counts is not None:
        err = r['est_counts'] - true_counts
        print(f"    count error mean/p10/p90={err.mean():.4f}/{np.percentile(err,10):.4f}/{np.percentile(err,90):.4f}")

#-------------------------------------------------------------------------------


def baseline_pilot(n_queries: int = 500, C: int = 20, seed: int = 42) -> None:
    print(f"=== Baseline pilot: SIFT, C={C}, n_queries={n_queries} ===")
    points, queries = load_sift_points_and_queries(n_queries, seed)
    radii = query_radii(points, queries, K)
    d = points.shape[1]
    pool = build_direction_pool(d, POOL_SIZE, np.random.default_rng(seed))

    true_counts = np.array([brute_force_count(points, queries[i], float(radii[i]))
                             for i in range(n_queries)])
    print(f"  true count percentiles: {np.percentile(true_counts, [0,25,50,75,100])}")

    for halving in (True, False):
        r = run_pilot_config(points, queries, radii, C, halving, pool, seed)
        print_result('baseline', r, true_counts)


def adversarial_pilot(n_queries: int = 200, C: int = 5, d: int = 384, seed: int = 42,
                       timeout_seconds: float = 300.0) -> None:
    print(f"\n=== Adversarial pilot: synthetic clustered, d={d}, smallest W, C={C} ===")
    rng = np.random.default_rng(seed)
    W = (1.0 / np.sqrt(d)) * 0.5
    print(f"  W={W:.6f} (0.5x the 1/sqrt(d) threshold, smallest sweep value)")

    n_cluster = int(N * 0.99)
    n_background = N - n_cluster
    pole = rng.standard_normal(d)
    pole /= np.linalg.norm(pole)
    cluster_pts = sample_cap_points_direct(pole, W, n_cluster, rng)
    background_pts = rng.standard_normal((n_background, d))
    background_pts /= np.linalg.norm(background_pts, axis=1, keepdims=True)
    points = np.vstack([cluster_pts, background_pts])

    queries = rng.standard_normal((n_queries, d))
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    radii = query_radii(points, queries, K)

    pool = build_direction_pool(d, POOL_SIZE, np.random.default_rng(seed))

    for halving in (True, False):
        label = 'H' if halving else 'N'
        print(f"  running variant {label}...")
        t0 = time.perf_counter()
        tree = build_tree(points, C=C, halving=halving, direction_pool=pool,
                           rng=np.random.default_rng(seed))
        t_build = time.perf_counter() - t0

        n_capped_leaves = len(tree.leaf_depth_capped)
        n_capped_points = sum(len(tree.leaf_point_idx[nid]) for nid in tree.leaf_depth_capped)
        print(f"    variant {label}: t_build={t_build:.2f}s, n_nodes={tree.n_nodes()}, "
              f"max_depth={tree.max_depth_reached}, n_capped_leaves={n_capped_leaves}, "
              f"n_capped_points={n_capped_points} ({n_capped_points / N:.4%} of N)")

        if t_build > timeout_seconds:
            print(f"    WARNING: exceeded {timeout_seconds}s timeout")

#-------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-adversarial', action='store_true')
    args = parser.parse_args()

    baseline_pilot()
    if not args.skip_adversarial:
        adversarial_pilot()

    print("\nDone.")


if __name__ == '__main__':
    main()
