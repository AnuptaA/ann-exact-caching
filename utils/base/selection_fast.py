#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Optimized selection using numpy vectorization instead of Python sets.

Drop-in replacement for select_candidates with identical outputs.
Uses bool arrays for train/test coverage (small dimensions) and
vectorized index-array lookups for R_c budget checks.
"""

import numpy as np
from typing import List, Tuple

from .population import CandidateEntry

#-------------------------------------------------------------------------------

def select_candidates_fast(
    candidates: List[CandidateEntry],
    budget_B: int,
    strategy: str,
    n_train_queries: int,
    n_test_queries: int,
    seed: int = 42,
) -> Tuple[List[CandidateEntry], List[dict]]:
    """
    Optimized select_candidates using numpy vectorization.

    Same interface and semantics as select_candidates in selection.py.
    """
    if not candidates:
        return [], []

    n_cand = len(candidates)
    n_base = max(max(c.R_c) for c in candidates) + 1

    # Build concatenated R_c index structure for vectorized marginal computation.
    # offsets[i] = start of candidate i's indices in the flat array.
    rc_arrays = [np.array(sorted(c.R_c), dtype=np.int32) for c in candidates]
    rc_sizes = np.array([len(a) for a in rc_arrays], dtype=np.int32)
    offsets = np.zeros(n_cand + 1, dtype=np.int64)
    np.cumsum(rc_sizes, out=offsets[1:])
    rc_flat = np.concatenate(rc_arrays) if offsets[-1] > 0 else np.empty(0, dtype=np.int32)

    # train/test masks: (n_cand, n_train/n_test) bool
    train_masks = np.zeros((n_cand, n_train_queries), dtype=np.bool_)
    test_masks = np.zeros((n_cand, n_test_queries), dtype=np.bool_)

    for i, c in enumerate(candidates):
        if c.covered_train_indices:
            train_masks[i, list(c.covered_train_indices)] = True
        if c.covered_test_indices:
            test_masks[i, list(c.covered_test_indices)] = True

    selected: List[CandidateEntry] = []
    trajectory: List[dict] = []

    ctx = _Context(n_base, n_cand, n_train_queries, n_test_queries,
                   rc_flat, offsets, rc_sizes, train_masks, test_masks)

    if strategy == "random":
        _fast_random(candidates, ctx, budget_B, seed, selected, trajectory)
    elif strategy == "greedy_naive":
        _fast_greedy_naive(candidates, ctx, budget_B, selected, trajectory)
    elif strategy == "greedy_cost_static":
        _fast_greedy_cost_static(candidates, ctx, budget_B, selected, trajectory)
    elif strategy == "greedy_cost":
        _fast_greedy_cost(candidates, ctx, budget_B, selected, trajectory)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'")

    return selected, trajectory

#-------------------------------------------------------------------------------

class _Context:
    """Shared precomputed arrays for all strategies."""
    __slots__ = ('n_base', 'n_cand', 'n_train', 'n_test',
                 'rc_flat', 'offsets', 'rc_sizes',
                 'train_masks', 'test_masks')

    def __init__(self, n_base, n_cand, n_train, n_test,
                 rc_flat, offsets, rc_sizes, train_masks, test_masks):
        self.n_base = n_base
        self.n_cand = n_cand
        self.n_train = n_train
        self.n_test = n_test
        self.rc_flat = rc_flat
        self.offsets = offsets
        self.rc_sizes = rc_sizes
        self.train_masks = train_masks
        self.test_masks = test_masks

#-------------------------------------------------------------------------------

def _compute_marginals_batch(ctx: _Context, alive_idx: np.ndarray,
                             cached_bool: np.ndarray) -> np.ndarray:
    """Compute marginal cost for each alive candidate via vectorized lookup.

    Uses a single fancy-index into cached_bool for ALL alive candidates' R_c
    indices at once, then segments the result using np.add.reduceat.
    """
    n_alive = len(alive_idx)
    if n_alive == 0:
        return np.empty(0, dtype=np.int32)

    # gather flat slices for alive candidates
    starts = ctx.offsets[alive_idx]
    ends = ctx.offsets[alive_idx + 1]
    sizes = ctx.rc_sizes[alive_idx]
    total = int(sizes.sum())

    if total == 0:
        return np.zeros(n_alive, dtype=np.int32)

    # build flat index array for all alive candidates
    flat_idx = np.empty(total, dtype=np.int32)
    pos = 0
    for j in range(n_alive):
        s = int(starts[j])
        sz = int(sizes[j])
        flat_idx[pos:pos + sz] = ctx.rc_flat[s:s + sz]
        pos += sz

    # single vectorized lookup: which indices are NOT cached?
    uncached = ~cached_bool[flat_idx]  # (total,) bool

    # segment sum using reduceat
    seg_starts = np.zeros(n_alive, dtype=np.int64)
    np.cumsum(sizes[:-1], out=seg_starts[1:])
    # reduceat needs int input
    marginals = np.add.reduceat(uncached.view(np.uint8), seg_starts).astype(np.int32)
    return marginals

#-------------------------------------------------------------------------------

def _record_fast(
    selected, cached_count, covered_train, prev_train_count,
    n_train_queries, covered_test, prev_test_count, n_test_queries, trajectory,
):
    n_train = int(np.count_nonzero(covered_train))
    n_test = int(np.count_nonzero(covered_test))
    trajectory.append({
        "num_entries": len(selected),
        "vectors_cached": cached_count,
        "cig_train_coverage": n_train / n_train_queries if n_train_queries > 0 else 0.0,
        "cig_test_coverage": n_test / n_test_queries if n_test_queries > 0 else 0.0,
        "new_train_coverage": n_train - prev_train_count,
        "new_test_coverage": n_test - prev_test_count,
    })

#-------------------------------------------------------------------------------

def _select_best(ctx, alive_idx, cached_bool, cached_count, budget_B,
                 covered_train, covered_test, selected, trajectory,
                 score_fn):
    """Core greedy loop shared across strategies.

    score_fn(newly, marginals, alive_idx) -> scores array
    Returns updated (alive_idx, cached_count) or None if done.
    """
    marginals = _compute_marginals_batch(ctx, alive_idx, cached_bool)
    fits = (cached_count + marginals) <= budget_B
    alive_idx = alive_idx[fits]
    marginals = marginals[fits]
    if len(alive_idx) == 0:
        return None

    uncovered = ~covered_train
    newly = np.count_nonzero(ctx.train_masks[alive_idx] & uncovered, axis=1)

    has_benefit = newly > 0
    alive_idx = alive_idx[has_benefit]
    marginals = marginals[has_benefit]
    newly = newly[has_benefit]
    if len(alive_idx) == 0:
        return None

    return alive_idx, marginals, newly

#-------------------------------------------------------------------------------

def _fast_random(candidates, ctx, budget_B, seed, selected, trajectory):
    cached_bool = np.zeros(ctx.n_base, dtype=np.bool_)
    covered_train = np.zeros(ctx.n_train, dtype=np.bool_)
    covered_test = np.zeros(ctx.n_test, dtype=np.bool_)
    cached_count = 0

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(candidates))

    for idx in order:
        s = int(ctx.offsets[idx])
        e = int(ctx.offsets[idx + 1])
        idx_arr = ctx.rc_flat[s:e]
        marginal = int(np.count_nonzero(~cached_bool[idx_arr]))
        if cached_count + marginal > budget_B:
            continue

        prev_train = int(np.count_nonzero(covered_train))
        prev_test = int(np.count_nonzero(covered_test))

        selected.append(candidates[idx])
        cached_bool[idx_arr] = True
        cached_count += marginal
        covered_train |= ctx.train_masks[idx]
        covered_test |= ctx.test_masks[idx]

        _record_fast(selected, cached_count,
                     covered_train, prev_train, ctx.n_train,
                     covered_test, prev_test, ctx.n_test, trajectory)

#-------------------------------------------------------------------------------

def _fast_greedy_naive(candidates, ctx, budget_B, selected, trajectory):
    cached_bool = np.zeros(ctx.n_base, dtype=np.bool_)
    covered_train = np.zeros(ctx.n_train, dtype=np.bool_)
    covered_test = np.zeros(ctx.n_test, dtype=np.bool_)
    cached_count = 0
    alive = np.ones(ctx.n_cand, dtype=np.bool_)

    while True:
        alive_idx = np.where(alive)[0]
        if len(alive_idx) == 0:
            break

        result = _select_best(ctx, alive_idx, cached_bool, cached_count, budget_B,
                              covered_train, covered_test, selected, trajectory, None)
        if result is None:
            break
        alive_idx, marginals, newly = result

        new_alive = np.zeros(ctx.n_cand, dtype=np.bool_)
        new_alive[alive_idx] = True
        alive = new_alive

        best_pos = int(np.argmax(newly))
        best_i = alive_idx[best_pos]

        prev_train = int(np.count_nonzero(covered_train))
        prev_test = int(np.count_nonzero(covered_test))

        selected.append(candidates[best_i])
        s, e = int(ctx.offsets[best_i]), int(ctx.offsets[best_i + 1])
        cached_bool[ctx.rc_flat[s:e]] = True
        cached_count = int(np.count_nonzero(cached_bool))
        covered_train |= ctx.train_masks[best_i]
        covered_test |= ctx.test_masks[best_i]
        alive[best_i] = False

        _record_fast(selected, cached_count,
                     covered_train, prev_train, ctx.n_train,
                     covered_test, prev_test, ctx.n_test, trajectory)

#-------------------------------------------------------------------------------

def _fast_greedy_cost_static(candidates, ctx, budget_B, selected, trajectory):
    cached_bool = np.zeros(ctx.n_base, dtype=np.bool_)
    covered_train = np.zeros(ctx.n_train, dtype=np.bool_)
    covered_test = np.zeros(ctx.n_test, dtype=np.bool_)
    cached_count = 0
    alive = np.ones(ctx.n_cand, dtype=np.bool_)

    static_cost = ctx.rc_sizes.astype(np.float64)
    static_cost[static_cost == 0.0] = 1.0

    while True:
        alive_idx = np.where(alive)[0]
        if len(alive_idx) == 0:
            break

        result = _select_best(ctx, alive_idx, cached_bool, cached_count, budget_B,
                              covered_train, covered_test, selected, trajectory, None)
        if result is None:
            break
        alive_idx, marginals, newly = result

        new_alive = np.zeros(ctx.n_cand, dtype=np.bool_)
        new_alive[alive_idx] = True
        alive = new_alive

        scores = newly / static_cost[alive_idx]
        best_pos = int(np.argmax(scores))
        best_i = alive_idx[best_pos]

        prev_train = int(np.count_nonzero(covered_train))
        prev_test = int(np.count_nonzero(covered_test))

        selected.append(candidates[best_i])
        s, e = int(ctx.offsets[best_i]), int(ctx.offsets[best_i + 1])
        cached_bool[ctx.rc_flat[s:e]] = True
        cached_count = int(np.count_nonzero(cached_bool))
        covered_train |= ctx.train_masks[best_i]
        covered_test |= ctx.test_masks[best_i]
        alive[best_i] = False

        _record_fast(selected, cached_count,
                     covered_train, prev_train, ctx.n_train,
                     covered_test, prev_test, ctx.n_test, trajectory)

#-------------------------------------------------------------------------------

def _fast_greedy_cost(candidates, ctx, budget_B, selected, trajectory):
    cached_bool = np.zeros(ctx.n_base, dtype=np.bool_)
    covered_train = np.zeros(ctx.n_train, dtype=np.bool_)
    covered_test = np.zeros(ctx.n_test, dtype=np.bool_)
    cached_count = 0
    alive = np.ones(ctx.n_cand, dtype=np.bool_)
    deferred_indices: List[int] = []

    while True:
        alive_idx = np.where(alive)[0]
        if len(alive_idx) == 0:
            break

        marginals = _compute_marginals_batch(ctx, alive_idx, cached_bool)
        fits = (cached_count + marginals) <= budget_B
        alive_idx = alive_idx[fits]
        marginals = marginals[fits]
        if len(alive_idx) == 0:
            break

        uncovered = ~covered_train
        newly = np.count_nonzero(ctx.train_masks[alive_idx] & uncovered, axis=1)

        has_benefit = newly > 0
        alive_idx = alive_idx[has_benefit]
        marginals = marginals[has_benefit]
        newly = newly[has_benefit]
        if len(alive_idx) == 0:
            break

        # defer zero-cost candidates
        zero_cost = marginals == 0
        if np.any(zero_cost):
            deferred_indices.extend(alive_idx[zero_cost].tolist())
            keep = ~zero_cost
            alive_idx = alive_idx[keep]
            marginals = marginals[keep]
            newly = newly[keep]

        if len(alive_idx) == 0:
            break

        new_alive = np.zeros(ctx.n_cand, dtype=np.bool_)
        new_alive[alive_idx] = True
        alive = new_alive

        scores = newly / marginals.astype(np.float64)
        best_pos = int(np.argmax(scores))
        best_i = alive_idx[best_pos]

        prev_train = int(np.count_nonzero(covered_train))
        prev_test = int(np.count_nonzero(covered_test))

        selected.append(candidates[best_i])
        s, e = int(ctx.offsets[best_i]), int(ctx.offsets[best_i + 1])
        cached_bool[ctx.rc_flat[s:e]] = True
        cached_count = int(np.count_nonzero(cached_bool))
        covered_train |= ctx.train_masks[best_i]
        covered_test |= ctx.test_masks[best_i]
        alive[best_i] = False

        _record_fast(selected, cached_count,
                     covered_train, prev_train, ctx.n_train,
                     covered_test, prev_test, ctx.n_test, trajectory)

    # deferred flush
    if deferred_indices:
        uncovered = ~covered_train
        deferred_newly = np.count_nonzero(
            ctx.train_masks[deferred_indices] & uncovered, axis=1
        )
        order = np.argsort(deferred_newly)[::-1]
        for pos in order:
            if deferred_newly[pos] <= 0:
                break
            di = deferred_indices[pos]
            prev_train = int(np.count_nonzero(covered_train))
            prev_test = int(np.count_nonzero(covered_test))

            selected.append(candidates[di])
            covered_train |= ctx.train_masks[di]
            covered_test |= ctx.test_masks[di]

            _record_fast(selected, cached_count,
                         covered_train, prev_train, ctx.n_train,
                         covered_test, prev_test, ctx.n_test, trajectory)
