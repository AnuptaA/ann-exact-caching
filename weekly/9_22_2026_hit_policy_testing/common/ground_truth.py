#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Brute-force full sorted top-K ground truth (spec 4): for every query,
returns both the K nearest indices AND their distances, ascending."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import numpy as np

from utils.base.distance_metrics import pairwise_angular_distance

#-------------------------------------------------------------------------------

def batched_topk_full(queries: np.ndarray, base: np.ndarray, K: int, batch_size: int = 256) -> tuple:
    """Top-K nearest neighbor (indices, distances) for each query against
    base, angular distance ascending. Queries assumed distinct from base
    rows (no self-exclusion). Batches to avoid OOM on large query sets."""
    n = len(queries)
    indices = np.empty((n, K), dtype=np.int64)
    distances = np.empty((n, K), dtype=np.float64)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        dists = pairwise_angular_distance(queries[start:end], base)
        part = np.argpartition(dists, K - 1, axis=1)[:, :K]
        part_dists = np.take_along_axis(dists, part, axis=1)
        row_order = np.argsort(part_dists, axis=1)
        indices[start:end] = np.take_along_axis(part, row_order, axis=1)
        distances[start:end] = np.take_along_axis(part_dists, row_order, axis=1)
    return indices, distances

#-------------------------------------------------------------------------------

def batched_topk_full_skip_self(queries: np.ndarray, base: np.ndarray, query_base_indices: np.ndarray,
                                 K: int, batch_size: int = 256) -> tuple:
    """Same as batched_topk_full, but each query is itself a row of base
    (at query_base_indices[i]) and must be excluded from its own K nearest
    neighbors. Computes K+1 and drops the self-match (distance ~0) from
    each row."""
    indices_k1, distances_k1 = batched_topk_full(queries, base, K + 1, batch_size)
    n = len(queries)
    indices = np.empty((n, K), dtype=np.int64)
    distances = np.empty((n, K), dtype=np.float64)
    for i in range(n):
        self_mask = indices_k1[i] == query_base_indices[i]
        if self_mask.any():
            keep = ~self_mask
        else:
            # self wasn't in the K+1 nearest (shouldn't happen at distance 0, but
            # guard defensively), drop the farthest of the K+1 instead
            keep = np.ones(K + 1, dtype=bool)
            keep[-1] = False
        indices[i] = indices_k1[i][keep][:K]
        distances[i] = distances_k1[i][keep][:K]
    return indices, distances

#-------------------------------------------------------------------------------

def precompute_ground_truth(dataset: str, regime: str, split_name: str, queries: np.ndarray,
                             base: np.ndarray, K: int, out_dir: Path, k_extended: int = None) -> Path:
    """Persists (indices, distances) for one (dataset, regime, split) to a
    single .npz file. The Kth column of distances (0-indexed) is the radius
    of the Kth nearest neighbor for each query: theta_K(query) = distances[:, K-1]

    k_extended (item G): CALIB/TEST are stored to this depth (200) instead of
    just K (100)."""
    depth = k_extended if k_extended is not None else K
    out_path = out_dir / f"{dataset}_{regime}_{split_name}_ground_truth.npz"
    if out_path.exists():
        existing_depth = np.load(out_path)["indices"].shape[1]
        if existing_depth >= depth:
            return out_path
        # a prior run persisted this file at a shallower depth (e.g. K=100
        # before item G extended calib/test to 200) -- stale, regenerate
    indices, distances = batched_topk_full(queries, base, depth)
    np.savez(out_path, indices=indices.astype(np.int32), distances=distances.astype(np.float32))
    return out_path

#-------------------------------------------------------------------------------

def load_ground_truth(path: Path) -> tuple:
    data = np.load(path)
    return data["indices"], data["distances"]
