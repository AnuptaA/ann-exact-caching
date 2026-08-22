#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Fixed-test-query sweep on real SIFT data: for each of a handful of
fixed test-query anchors, bins the real query pool by observed angular
separation and computes cap fraction / recall per bin.
"""

import sys
import math
import argparse
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from sklearn.cluster import KMeans

from primitives import exact_intersection_fraction
from datasets.dataloaders import load_sift_dataset

#-------------------------------------------------------------------------------

K_VALUES = [10, 25, 50, 100]
K_MAX = max(K_VALUES)
N_FIXED_QUERIES = 5
N_CLUSTERS = 20
FAR_SAMPLE_SIZE = 1000
CACHE_RADIUS_MULTIPLIER = 1.4
N_BACKGROUND_POINTS = 50000
N_BINS = 18
MAX_PER_BIN = 200
D = 128

OUT_DIR = Path(__file__).parent / 'results'
PART_DIR = 'debug_sift_fixed_query_sweep'
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

def _cluster_query_pool(query_pool: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    km = KMeans(n_clusters=n_clusters, n_init=3, random_state=seed).fit(query_pool)
    sizes = np.bincount(km.labels_, minlength=n_clusters)
    print(f"  cluster sizes: min={sizes.min()} max={sizes.max()} median={int(np.median(sizes))}")
    return km.labels_

#-------------------------------------------------------------------------------

def run_anchor(background: np.ndarray, anchor: np.ndarray, candidate_pool: np.ndarray, anchor_id: int,
                n_bins: int, max_per_bin: int, cache_radius_multiplier: float, seed: int) -> dict:
    """One anchor: bins candidate_pool by observed delta and computes
    fraction/recall per K within each bin.

    Returns {K: {field: array}}.
    """
    rng = np.random.default_rng(seed)

    dists_test, idx_test = _bruteforce_knn(background, anchor[None, :], K_MAX)
    delta = np.arccos(np.clip(candidate_pool @ anchor, -1.0, 1.0))

    edges = np.quantile(delta, np.linspace(0.0, 1.0, n_bins + 1))
    edges[-1] += 1e-9
    bin_idx = np.clip(np.digitize(delta, edges) - 1, 0, n_bins - 1)

    selected = []
    selected_bin = []
    for b in range(n_bins):
        members = np.where(bin_idx == b)[0]
        if len(members) == 0:
            continue
        take = members if len(members) <= max_per_bin else rng.choice(members, size=max_per_bin, replace=False)
        selected.append(take)
        selected_bin.append(np.full(len(take), b, dtype=np.int32))
    sel_idx = np.concatenate(selected)
    sel_bin = np.concatenate(selected_bin)
    candidates = candidate_pool[sel_idx]
    cand_delta = delta[sel_idx]

    dists_cache, _ = _bruteforce_knn(background, candidates, K_MAX)

    per_k = {}
    for K in K_VALUES:
        r_test = float(_chord_to_angle(dists_test[:, K - 1])[0])
        r_cache_true = _chord_to_angle(dists_cache[:, K - 1])
        r_cache = np.minimum(r_cache_true * cache_radius_multiplier, math.pi)

        fraction = np.array([exact_intersection_fraction(float(rc), r_test, float(dd), D)
                              for rc, dd in zip(r_cache, cand_delta)])

        neighbor_coords = background[idx_test[0, :K]]
        dist_to_cand = np.linalg.norm(neighbor_coords[None, :, :] - candidates[:, None, :], axis=2)
        r_cache_chord = 2.0 * np.sin(r_cache / 2.0)
        recall = (dist_to_cand <= r_cache_chord[:, None]).mean(axis=1)

        per_k[K] = {
            'anchor_id': np.full(len(sel_idx), anchor_id, dtype=np.int32),
            'bin': sel_bin,
            'delta': cand_delta,
            'r_cache': r_cache,
            'r_test': np.full(len(sel_idx), r_test),
            'fraction': fraction,
            'recall': recall,
        }
    return per_k


def run_all_anchors(background: np.ndarray, query_pool: np.ndarray, n_anchors: int, n_clusters: int,
                     far_sample_size: int, n_bins: int, max_per_bin: int, cache_radius_multiplier: float,
                     base_seed: int) -> dict:
    rng = np.random.default_rng(base_seed)
    labels = _cluster_query_pool(query_pool, n_clusters, base_seed)
    cluster_ids = rng.choice(n_clusters, size=n_anchors, replace=False)

    fields = ('anchor_id', 'cluster_id', 'bin', 'delta', 'r_cache', 'r_test', 'fraction', 'recall')
    pooled = {K: {f: [] for f in fields} for K in K_VALUES}

    for a, cluster_id in enumerate(cluster_ids):
        print(f"  anchor {a + 1}/{n_anchors} (cluster {cluster_id})...")
        members = np.where(labels == cluster_id)[0]
        anchor_idx = rng.choice(members)
        anchor = query_pool[anchor_idx]

        cluster_candidates = members[members != anchor_idx]
        outside = np.where(labels != cluster_id)[0]
        far_sample = rng.choice(outside, size=min(far_sample_size, len(outside)), replace=False)
        candidate_idx = np.concatenate([cluster_candidates, far_sample])
        candidate_pool = query_pool[candidate_idx]

        per_k = run_anchor(background, anchor, candidate_pool, a, n_bins, max_per_bin, cache_radius_multiplier,
                           seed=base_seed + a)
        for K in K_VALUES:
            per_k[K]['cluster_id'] = np.full(len(per_k[K]['bin']), cluster_id, dtype=np.int32)
            for f in fields:
                pooled[K][f].append(per_k[K][f])

    for K in K_VALUES:
        for f in fields:
            pooled[K][f] = np.concatenate(pooled[K][f])
    return pooled


def save_results(pooled: dict, run_dir: Path) -> None:
    dim_dir = run_dir / PART_DIR / f'{D}d' / 'raw'
    dim_dir.mkdir(parents=True, exist_ok=True)
    for K, arrays in pooled.items():
        path = dim_dir / f'K{K}.npz'
        np.savez_compressed(path, d=D, K=K, **arrays)
        print(f"    wrote {path}")


def bin_count_diagnostic(pooled: dict) -> None:
    print(f"\n  Diagnostic (SIFT d={D}):")
    for K in K_VALUES:
        fraction = pooled[K]['fraction']
        print(f"    K={K}: total={len(fraction)}, informative={float((fraction > 0).mean()):.4f}, "
              f"fraction range=[{fraction.min():.4f}, {fraction.max():.4f}]")

#-------------------------------------------------------------------------------

def main() -> None:
    global K_VALUES, K_MAX, OUT_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--k-values', type=int, nargs='+', default=K_VALUES, dest='k_values')
    parser.add_argument('--n-anchors', type=int, default=N_FIXED_QUERIES, dest='n_anchors')
    parser.add_argument('--n-clusters', type=int, default=N_CLUSTERS, dest='n_clusters')
    parser.add_argument('--far-sample-size', type=int, default=FAR_SAMPLE_SIZE, dest='far_sample_size')
    parser.add_argument('--cache-radius-multiplier', type=float, default=CACHE_RADIUS_MULTIPLIER,
                         dest='cache_radius_multiplier')
    parser.add_argument('--n-background-points', type=int, default=N_BACKGROUND_POINTS, dest='n_background_points')
    parser.add_argument('--n-bins', type=int, default=N_BINS, dest='n_bins')
    parser.add_argument('--max-per-bin', type=int, default=MAX_PER_BIN, dest='max_per_bin')
    parser.add_argument('--sift-dir', type=str, default='datasets/sift', dest='sift_dir')
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
    print(f"K values: {K_VALUES}, n_anchors={args.n_anchors}, n_clusters={args.n_clusters}, "
          f"far_sample_size={args.far_sample_size}, cache_radius_multiplier={args.cache_radius_multiplier}, "
          f"n_bins={args.n_bins}, max_per_bin={args.max_per_bin}\n")

    sift_dir = Path(args.sift_dir)
    if not sift_dir.is_absolute():
        sift_dir = Path(__file__).parent.parent.parent / sift_dir
    base, _, query, _ = load_sift_dataset(str(sift_dir))

    rng = np.random.default_rng(args.seed)
    bg_idx = rng.choice(len(base), size=min(args.n_background_points, len(base)), replace=False)
    background = _normalize(base[bg_idx].astype(np.float64))
    query_pool = _normalize(query.astype(np.float64))

    pooled = run_all_anchors(background, query_pool, args.n_anchors, args.n_clusters, args.far_sample_size,
                              args.n_bins, args.max_per_bin, args.cache_radius_multiplier, base_seed=args.seed)
    save_results(pooled, run_dir)
    bin_count_diagnostic(pooled)

    print(f"\nDone: {run_dir}")


if __name__ == '__main__':
    main()
