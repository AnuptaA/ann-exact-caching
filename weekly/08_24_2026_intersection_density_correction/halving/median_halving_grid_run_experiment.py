#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Full grid: Variant H vs Variant N, C in {5,10,20,50}, across SIFT,
ESCI, synthetic uniform (dimension sweep), and synthetic clustered
(dimension x width sweep). See the plan for the full design; this
implements it directly, reusing median_halving_pilot.py's dataset
loading helpers where they apply."""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'common'))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import numpy as np
from tqdm import tqdm

from datasets.dataloaders import load_fvecs
from median_halving_tree import (
    build_direction_pool, build_tree, touching_leaves_batch, counts_from_hits,
    leaf_touch_counts_from_hits, brute_force_count, tree_memory_bytes,
)
from part2_uniform_baseline_run_experiment import _sample_uniform_sphere, _bruteforce_knn
from primitives import sample_cap_points_direct

#-------------------------------------------------------------------------------

N = 50000
K = 100
POOL_SIZE = 256
C_VALUES = [5, 10, 20, 50]
N_QUERIES = 500
SEED = 42

DIMS_SYNTHETIC = [2, 3, 5, 10, 50, 128, 384]
W_MULTIPLIERS = [0.5, 0.8, 1.0, 1.25, 2.0]
CLUSTER_FRACTION = 0.99

OUT_DIR = Path(__file__).parent.parent / 'results' / 'median_halving'

#-------------------------------------------------------------------------------


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vecs / norms


def _direct_radii(points: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """K-th NN angular radius via direct arccos(dot), matching
    count_in_radius/brute_force_count's own formula exactly (see
    median_halving_pilot.py's query_radii docstring for why this
    matters at the boundary)."""
    _, idx = _bruteforce_knn(points, queries, k)
    kth_points = points[idx[:, k - 1]]
    dots = np.clip(np.einsum('ij,ij->i', queries, kth_points), -1.0, 1.0)
    return np.arccos(dots)

#-------------------------------------------------------------------------------


def dataset_sift(seed: int) -> dict:
    repo_root = Path(__file__).parent.parent.parent.parent
    points = _normalize(np.load(repo_root / 'datasets/sift/base_vectors_50k.npy').astype(np.float64))
    all_queries = _normalize(load_fvecs(str(repo_root / 'datasets/sift/sift_query.fvecs')).astype(np.float64))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_queries), size=N_QUERIES, replace=False)
    queries = all_queries[idx]
    return {'points': points, 'queries': queries, 'tags': None}


def dataset_esci(seed: int) -> dict:
    repo_root = Path(__file__).parent.parent.parent.parent
    points = _normalize(np.load(repo_root / 'datasets/esci/base_vectors_50k.npy').astype(np.float64))
    query_data = np.load(repo_root / 'datasets/esci/queries.npz')
    all_queries = _normalize(query_data['query_vectors'].astype(np.float64))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_queries), size=N_QUERIES, replace=False)
    queries = all_queries[idx]
    return {'points': points, 'queries': queries, 'tags': None}


def dataset_uniform(d: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    points = _sample_uniform_sphere(N, d, rng)
    queries = _sample_uniform_sphere(N_QUERIES, d, rng)
    return {'points': points, 'queries': queries, 'tags': None}


def dataset_clustered(d: int, w_multiplier: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    W = (1.0 / np.sqrt(d)) * w_multiplier
    n_cluster = int(round(N * CLUSTER_FRACTION))
    n_background = N - n_cluster
    pole = rng.standard_normal(d)
    pole /= np.linalg.norm(pole)
    cluster_pts = sample_cap_points_direct(pole, W, n_cluster, rng)
    background_pts = _sample_uniform_sphere(n_background, d, rng)
    points = np.vstack([cluster_pts, background_pts])

    queries = _sample_uniform_sphere(N_QUERIES, d, rng)
    query_dists = np.arccos(np.clip(queries @ pole, -1.0, 1.0))
    tags = query_dists <= W  # True = cluster-region query
    return {'points': points, 'queries': queries, 'tags': tags, 'W': W, 'pole': pole}

#-------------------------------------------------------------------------------


def run_one_config(data: dict, radii: np.ndarray, true_counts: np.ndarray,
                    C: int, halving: bool, pool: np.ndarray, seed: int) -> dict:
    points, queries = data['points'], data['queries']
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
    n_capped_leaves = len(tree.leaf_depth_capped)
    n_capped_points = sum(len(tree.leaf_point_idx[nid]) for nid in tree.leaf_depth_capped)

    return {
        'n_nodes': tree.n_nodes(), 'n_leaves': int(tree.is_leaf.sum()),
        'max_depth_reached': tree.max_depth_reached,
        't_build': t_build, 't_traverse': t_traverse, 't_count': t_count,
        'leaf_touch_counts': leaf_touch_counts,
        'n_leaf_unique': len(tree.leaf_unique), 'n_capped_leaves': n_capped_leaves,
        'n_capped_points': n_capped_points,
        'mem_full_direction': tree_memory_bytes(tree, 'full_direction'),
        'mem_pooled_index': tree_memory_bytes(tree, 'pooled_index'),
        'est_counts': est_counts, 'true_counts': true_counts,
        'diagnostics': tree.diagnostics,
    }


def save_result(out_dir: Path, dataset_key: str, C: int, halving: bool, result: dict,
                 tags: np.ndarray = None) -> None:
    variant = 'H' if halving else 'N'
    d = out_dir / dataset_key
    d.mkdir(parents=True, exist_ok=True)
    path = d / f'C{C}_{variant}.npz'
    save_kwargs = dict(
        n_nodes=result['n_nodes'], n_leaves=result['n_leaves'],
        max_depth_reached=result['max_depth_reached'],
        t_build=result['t_build'], t_traverse=result['t_traverse'], t_count=result['t_count'],
        leaf_touch_counts=result['leaf_touch_counts'],
        n_leaf_unique=result['n_leaf_unique'], n_capped_leaves=result['n_capped_leaves'],
        n_capped_points=result['n_capped_points'],
        mem_full_direction=result['mem_full_direction'], mem_pooled_index=result['mem_pooled_index'],
        est_counts=result['est_counts'], true_counts=result['true_counts'],
    )
    if tags is not None:
        save_kwargs['tags'] = tags
    np.savez_compressed(path, **save_kwargs)
    if result['diagnostics']:
        with open(d / f'C{C}_{variant}_diagnostics.json', 'w') as f:
            json.dump(result['diagnostics'], f, indent=2)

#-------------------------------------------------------------------------------


def _write_progress(progress_path: Path, done: int, total: int, start_time: float, label: str) -> None:
    tmp_path = progress_path.with_suffix('.tmp')
    payload = {'configs_done': done, 'configs_total': total, 'current': label,
               'elapsed_seconds': time.time() - start_time, 'timestamp': datetime.now().isoformat()}
    tmp_path.write_text(json.dumps(payload))
    tmp_path.replace(progress_path)


def build_dataset_specs() -> list:
    specs = [('sift', lambda seed: dataset_sift(seed)), ('esci', lambda seed: dataset_esci(seed))]
    for d in DIMS_SYNTHETIC:
        specs.append((f'uniform_d{d}', lambda seed, d=d: dataset_uniform(d, seed)))
    for d in DIMS_SYNTHETIC:
        for wi, wm in enumerate(W_MULTIPLIERS):
            specs.append((f'clustered_d{d}_w{wi}', lambda seed, d=d, wm=wm: dataset_clustered(d, wm, seed)))
    return specs


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_specs = build_dataset_specs()
    total_configs = len(dataset_specs) * len(C_VALUES) * 2
    print(f"Dataset settings: {len(dataset_specs)}, C values: {len(C_VALUES)}, variants: 2, "
          f"total configs: {total_configs}")

    progress_path = OUT_DIR / 'progress.json'
    start_time = time.time()
    done = 0

    for dataset_key, loader in tqdm(dataset_specs, desc='datasets'):
        data = loader(SEED)
        radii = _direct_radii(data['points'], data['queries'], K)
        true_counts = np.array([brute_force_count(data['points'], data['queries'][i], float(radii[i]))
                                 for i in range(len(data['queries']))])
        d = data['points'].shape[1]
        pool = build_direction_pool(d, POOL_SIZE, np.random.default_rng(SEED))

        for C in C_VALUES:
            for halving in (True, False):
                label = f"{dataset_key} C={C} halving={halving}"
                result = run_one_config(data, radii, true_counts, C, halving, pool, SEED)
                save_result(OUT_DIR, dataset_key, C, halving, result, tags=data.get('tags'))
                done += 1
                _write_progress(progress_path, done, total_configs, start_time, label)

    print(f"\nDone. Total wall clock: {time.time() - start_time:.1f}s")


if __name__ == '__main__':
    main()
