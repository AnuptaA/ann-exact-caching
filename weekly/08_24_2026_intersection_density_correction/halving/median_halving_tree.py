#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Recursive bucket-construction tree for exact density counting on the
unit hypersphere: Variant H (median-halving splits) vs Variant N
(origin-hyperplane splits), a one-line difference in threshold
computation. See median_halving_tree_plan.md-derived margin-test:
margin(q, u, t) = |arccos(q.u) - arccos(t)|, which reduces to the
origin-only arcsin formula at t=0.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

#-------------------------------------------------------------------------------

DEFAULT_MAX_DEPTH = 40
DEFAULT_POOL_SIZE = 256

# Matches density_correction_common.py's convention: the K-th NN radius
# is by definition exactly the K-th neighbor's own distance, so a plain
# "<=" boundary comparison is fragile to which floating-point code path
# (einsum vs matmul vs partition) computed that distance. A tiny
# absolute tolerance avoids spurious off-by-one counts at the boundary.
_BOUNDARY_EPS = 1e-9

#-------------------------------------------------------------------------------


def build_direction_pool(d: int, P: int, rng: np.random.Generator) -> np.ndarray:
    pts = rng.standard_normal((P, d))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

#-------------------------------------------------------------------------------


@dataclass
class Tree:
    points: np.ndarray
    direction_pool: np.ndarray
    direction_idx: np.ndarray
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    is_leaf: np.ndarray
    leaf_point_idx: dict
    leaf_unique: dict
    leaf_depth_capped: set
    diagnostics: list
    halving: bool
    C: int
    max_depth_reached: int = 0

    def n_nodes(self) -> int:
        return len(self.is_leaf)

#-------------------------------------------------------------------------------


def build_tree(points: np.ndarray, C: int, halving: bool, direction_pool: np.ndarray,
                rng: np.random.Generator, max_depth: int = DEFAULT_MAX_DEPTH) -> Tree:
    """Iterative (stack-based, not recursive) construction. Node arrays
    grow dynamically since Variant N's true node count is not bounded
    by N/C the way Variant H's is (though both are bounded by 2N-1)."""
    n_points = points.shape[0]
    P = direction_pool.shape[0]

    direction_idx: list = []
    threshold: list = []
    left: list = []
    right: list = []
    is_leaf: list = []
    leaf_point_idx: dict = {}
    leaf_unique: dict = {}
    leaf_depth_capped: set = set()
    diagnostics: list = []
    max_depth_reached = 0

    def new_node() -> int:
        nid = len(direction_idx)
        direction_idx.append(-1)
        threshold.append(0.0)
        left.append(-1)
        right.append(-1)
        is_leaf.append(False)
        return nid

    root_id = new_node()
    stack = [(root_id, np.arange(n_points), 0, frozenset())]

    while stack:
        node_id, idx, depth, used = stack.pop()
        max_depth_reached = max(max_depth_reached, depth)
        n = len(idx)

        if n <= C:
            is_leaf[node_id] = True
            leaf_point_idx[node_id] = idx
            continue

        if depth >= max_depth:
            is_leaf[node_id] = True
            leaf_point_idx[node_id] = idx
            leaf_depth_capped.add(node_id)
            diagnostics.append({'node_id': node_id, 'depth': depth, 'n_points': n,
                                 'event': 'depth_capped'})
            continue

        available = [i for i in range(P) if i not in used]
        if not available:
            is_leaf[node_id] = True
            leaf_point_idx[node_id] = idx
            leaf_depth_capped.add(node_id)
            diagnostics.append({'node_id': node_id, 'depth': depth, 'n_points': n,
                                 'event': 'direction_pool_exhausted'})
            continue

        dir_choice = int(rng.choice(available))
        direction = direction_pool[dir_choice]
        proj = points[idx] @ direction
        t = float(np.median(proj)) if halving else 0.0

        left_mask = proj <= t
        n_left = int(left_mask.sum())

        if n_left == 0 or n_left == n:
            unique_pts, counts = np.unique(points[idx], axis=0, return_counts=True)
            if len(unique_pts) < n:
                is_leaf[node_id] = True
                leaf_unique[node_id] = (unique_pts, counts)
            else:
                stack.append((node_id, idx, depth, used | {dir_choice}))
            continue

        direction_idx[node_id] = dir_choice
        threshold[node_id] = t
        left_id = new_node()
        right_id = new_node()
        left[node_id] = left_id
        right[node_id] = right_id

        new_used = used | {dir_choice}
        stack.append((left_id, idx[left_mask], depth + 1, new_used))
        stack.append((right_id, idx[~left_mask], depth + 1, new_used))

    return Tree(
        points=points, direction_pool=direction_pool,
        direction_idx=np.array(direction_idx, dtype=np.int32),
        threshold=np.array(threshold, dtype=np.float64),
        left=np.array(left, dtype=np.int64), right=np.array(right, dtype=np.int64),
        is_leaf=np.array(is_leaf, dtype=bool),
        leaf_point_idx=leaf_point_idx, leaf_unique=leaf_unique,
        leaf_depth_capped=leaf_depth_capped, diagnostics=diagnostics,
        halving=halving, C=C, max_depth_reached=max_depth_reached,
    )

#-------------------------------------------------------------------------------


def touching_leaves_batch(tree: Tree, queries: np.ndarray, radii: np.ndarray,
                           max_frontier_ratio: float = 3.0) -> tuple:
    """Flat (hit_query_idx, hit_node_id) arrays: one entry per (query,
    leaf) touch. Returning flat arrays instead of a per-query list lets
    every downstream consumer (touch counts, exact counting) stay fully
    vectorized - a per-pair Python append here was the actual bottleneck
    once the high-dimensional pruning collapse (see module docs) made
    most queries touch most leaves; at that point a per-pair loop
    degenerates into millions of iterations, the same class of
    unbatched-call cost this project has hit before. radii is one value
    per query, never a shared scalar. Frontier-batched across all
    queries at once, so the outer Python loop runs O(tree depth) times,
    not O(n_queries) or O(n_nodes)."""
    n_queries = queries.shape[0]
    hit_q_parts = []
    hit_n_parts = []

    # Precompute the full (n_queries, P) projection table once: only P
    # (<=256) distinct directions and n_queries distinct queries ever
    # appear, so every level's per-pair projection is a lookup into this
    # small table, not a fresh (frontier_size, d) direction-vector
    # gather + dot product. Gathering full d-dim vectors per pair was
    # the actual bottleneck once frontier size reached millions (the
    # high-dimensional pruning collapse means most/all leaves get
    # touched, see module docs) - a (frontier_size, d) intermediate at
    # d=128 and an 8M-row frontier is an 8GB array recomputed every
    # level, not the arccos/comparison logic itself.
    query_proj = np.clip(queries @ tree.direction_pool.T, -1.0, 1.0)  # (n_queries, P)

    frontier_q = np.arange(n_queries)
    frontier_n = np.zeros(n_queries, dtype=np.int64)
    # Peak per-level frontier for one query is bounded by that level's
    # node count, which across a binary tree never exceeds the total
    # leaf count - so n_queries * n_leaves is the real worst-case bound
    # (every query uncertain at every node), not an arbitrary multiplier.
    n_leaves = int(tree.is_leaf.sum())
    max_frontier = max(int(n_queries * n_leaves * max_frontier_ratio), 10000)

    while len(frontier_q) > 0:
        leaf_mask = tree.is_leaf[frontier_n]
        if leaf_mask.any():
            hit_q_parts.append(frontier_q[leaf_mask])
            hit_n_parts.append(frontier_n[leaf_mask])

        active_q = frontier_q[~leaf_mask]
        active_n = frontier_n[~leaf_mask]
        if len(active_q) == 0:
            break

        node_dir_idx = tree.direction_idx[active_n]
        thresholds = tree.threshold[active_n]
        r = radii[active_q]

        proj = query_proj[active_q, node_dir_idx]
        t_clipped = np.clip(thresholds, -1.0, 1.0)
        margin = np.abs(np.arccos(proj) - np.arccos(t_clipped))
        # Conservative direction: require a hair more margin than
        # strictly needed before excluding a branch, so floating-point
        # noise can never turn a truly-uncertain node falsely certain
        # (which would silently miss points) - the reverse mistake
        # (uncertain when actually certain) only costs a wasted visit.
        certain = margin >= r + _BOUNDARY_EPS
        q_left = proj <= thresholds

        certain_left = certain & q_left
        certain_right = certain & ~q_left
        uncertain = ~certain

        next_q = np.concatenate([
            active_q[certain_left], active_q[certain_right],
            active_q[uncertain], active_q[uncertain],
        ])
        next_n = np.concatenate([
            tree.left[active_n[certain_left]], tree.right[active_n[certain_right]],
            tree.left[active_n[uncertain]], tree.right[active_n[uncertain]],
        ])

        if len(next_q) > max_frontier:
            raise RuntimeError(
                f"touching_leaves_batch: frontier size {len(next_q)} exceeds safety cap "
                f"{max_frontier} (n_queries={n_queries}) - likely pathological branching, "
                f"flagging rather than continuing silently."
            )

        frontier_q, frontier_n = next_q, next_n

    hit_query_idx = np.concatenate(hit_q_parts) if hit_q_parts else np.zeros(0, dtype=np.int64)
    hit_node_id = np.concatenate(hit_n_parts) if hit_n_parts else np.zeros(0, dtype=np.int64)
    return hit_query_idx, hit_node_id


def leaf_touch_counts_from_hits(hit_query_idx: np.ndarray, n_queries: int) -> np.ndarray:
    return np.bincount(hit_query_idx, minlength=n_queries)


def _leaf_layout(tree: Tree) -> dict:
    """Lazily computed, cached-on-the-tree flattening of every regular
    leaf's point indices into one concatenated array plus per-node
    (start, length) offsets - lets counts_from_hits expand (query, leaf)
    hits into (query, point) pairs via pure array arithmetic (repeat +
    cumsum), not a Python loop over hits."""
    cached = getattr(tree, '_leaf_layout_cache', None)
    if cached is not None:
        return cached

    n_nodes = tree.n_nodes()
    node_start = np.full(n_nodes, -1, dtype=np.int64)
    node_length = np.zeros(n_nodes, dtype=np.int64)
    is_unique_node = np.zeros(n_nodes, dtype=bool)
    for nid in tree.leaf_unique:
        is_unique_node[nid] = True

    parts = []
    offset = 0
    for nid, idx in tree.leaf_point_idx.items():
        node_start[nid] = offset
        node_length[nid] = len(idx)
        parts.append(idx)
        offset += len(idx)
    flat_points = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)

    layout = {'node_start': node_start, 'node_length': node_length,
              'is_unique_node': is_unique_node, 'flat_points': flat_points}
    tree._leaf_layout_cache = layout
    return layout


def counts_from_hits(tree: Tree, queries: np.ndarray, radii: np.ndarray,
                      hit_query_idx: np.ndarray, hit_node_id: np.ndarray) -> np.ndarray:
    """Exact in-radius count per query. Grouped by query (sort + split,
    not a Python dict), then one vectorized gather + distance check per
    query - bounded to that query's own touched-point count (at most N).

    Deliberately NOT flattened across the whole query batch at once:
    an earlier version expanded every query's touched points into one
    global (total_hits, d) point matrix via fancy indexing, which is
    fine for hit *count* but not for point *coordinates* - with the
    high-dimensional pruning collapse (most/all leaves touched per
    query, see module docs), that matrix reaches n_queries * N rows,
    e.g. 25M rows * 128 cols for 500 SIFT queries at C=5 - about 25GB,
    which thrashed rather than ran. The 500-query outer loop here was
    never the actual bottleneck (500 Python iterations is cheap); the
    original bottleneck was a Python loop over millions of individual
    (query, leaf) *hits*, already eliminated in touching_leaves_batch."""
    n_queries = queries.shape[0]
    counts = np.zeros(n_queries, dtype=np.int64)
    if len(hit_query_idx) == 0:
        return counts

    layout = _leaf_layout(tree)
    unique_mask = layout['is_unique_node'][hit_node_id]

    if unique_mask.any():
        for qi, nid in zip(hit_query_idx[unique_mask].tolist(), hit_node_id[unique_mask].tolist()):
            unique_pts, mult = tree.leaf_unique[nid]
            dists = np.arccos(np.clip(unique_pts @ queries[qi], -1.0, 1.0))
            counts[qi] += int(mult[dists <= radii[qi] + _BOUNDARY_EPS].sum())

    regular_q = hit_query_idx[~unique_mask]
    regular_n = hit_node_id[~unique_mask]
    if len(regular_q) == 0:
        return counts

    order = np.argsort(regular_q, kind='stable')
    sorted_q = regular_q[order]
    sorted_n = regular_n[order]
    group_starts = np.flatnonzero(np.diff(sorted_q, prepend=sorted_q[0] - 1))
    group_query_ids = sorted_q[group_starts]
    group_ends = np.append(group_starts[1:], len(sorted_q))

    node_start, node_length, flat_points = layout['node_start'], layout['node_length'], layout['flat_points']
    for qi, start, end in zip(group_query_ids.tolist(), group_starts.tolist(), group_ends.tolist()):
        nodes_for_q = sorted_n[start:end]
        lengths = node_length[nodes_for_q]
        starts_for_q = node_start[nodes_for_q]
        total = int(lengths.sum())
        if total == 0:
            continue
        hit_output_start = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        within_hit_pos = np.arange(total) - np.repeat(hit_output_start, lengths)
        source_indices = np.repeat(starts_for_q, lengths) + within_hit_pos
        point_idx = flat_points[source_indices]

        dots = tree.points[point_idx] @ queries[qi]
        dists = np.arccos(np.clip(dots, -1.0, 1.0))
        counts[qi] += int((dists <= radii[qi] + _BOUNDARY_EPS).sum())

    return counts


def brute_force_count(points: np.ndarray, query: np.ndarray, radius: float) -> int:
    dists = np.arccos(np.clip(points @ query, -1.0, 1.0))
    return int((dists <= radius + _BOUNDARY_EPS).sum())

#-------------------------------------------------------------------------------


def tree_memory_bytes(tree: Tree, mode: str) -> int:
    """mode: 'full_direction' stores one full d-dim float64 direction per
    internal node; 'pooled_index' stores only the int32 pool index
    (already what Tree stores) plus the one-time shared pool cost."""
    n_internal = int((~tree.is_leaf).sum())
    d = tree.points.shape[1]
    structural = (
        tree.threshold.nbytes + tree.left.nbytes + tree.right.nbytes + tree.is_leaf.nbytes
    )
    leaf_storage = sum(arr.nbytes for idx in tree.leaf_point_idx.values() for arr in [idx])
    leaf_storage += sum(u.nbytes + c.nbytes for u, c in tree.leaf_unique.values())

    if mode == 'pooled_index':
        direction_storage = tree.direction_idx.nbytes + tree.direction_pool.nbytes
    elif mode == 'full_direction':
        direction_storage = n_internal * d * 8  # float64 per internal node, no shared pool
    else:
        raise ValueError(f"unknown mode {mode}")

    return structural + leaf_storage + direction_storage
