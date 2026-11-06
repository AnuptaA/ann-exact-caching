#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import numpy as np
from typing import List, Tuple

from .population import CandidateEntry

#-------------------------------------------------------------------------------

def _record(
    selected: list,
    n_cached: int,
    covered_train_mask: np.ndarray,
    prev_train_count: int,
    n_train_queries: int,
    covered_test_mask: np.ndarray,
    prev_test_count: int,
    n_test_queries: int,
    trajectory: list,
) -> None:
    n_train = int(covered_train_mask.sum())
    n_test = int(covered_test_mask.sum())
    trajectory.append({
        "num_entries": len(selected),
        "vectors_cached": n_cached,
        "cig_train_coverage": n_train / n_train_queries if n_train_queries > 0 else 0.0,
        "cig_test_coverage": n_test / n_test_queries if n_test_queries > 0 else 0.0,
        "new_train_coverage": n_train - prev_train_count,
        "new_test_coverage": n_test - prev_test_count,
    })

#-------------------------------------------------------------------------------

def select_candidates(
    candidates: List[CandidateEntry],
    budget_B: int,
    strategy: str,
    n_train_queries: int,
    n_test_queries: int,
    seed: int = 42,
) -> Tuple[List[CandidateEntry], List[dict]]:
    """
    Select a subset of candidates subject to the memory budget B.

    Returns (selected, trajectory) where trajectory records per-step stats.
    Selection criterion is based on train (oracle) coverage.
    Both train and test CIG coverage are tracked in the trajectory.

    Strategies: 'random', 'greedy_naive', 'greedy_cost_static', 'greedy_cost'.
    """
    selected: List[CandidateEntry] = []
    # Boolean mask over base indices, same length as every candidate's R_c.
    n_base = len(candidates[0].R_c) if candidates else 1
    cached_mask = np.zeros(n_base, dtype=bool)
    covered_train_mask = np.zeros(n_train_queries, dtype=bool)
    covered_test_mask = np.zeros(n_test_queries, dtype=bool)
    trajectory: List[dict] = []

    if strategy == "random":
        _select_random(candidates, budget_B, n_train_queries, n_test_queries,
                       seed, selected, cached_mask,
                       covered_train_mask, covered_test_mask, trajectory)
    elif strategy == "greedy_naive":
        _select_greedy_naive(candidates, budget_B, n_train_queries, n_test_queries,
                             selected, cached_mask,
                             covered_train_mask, covered_test_mask, trajectory)
    elif strategy == "greedy_cost_static":
        _select_greedy_cost_static(candidates, budget_B, n_train_queries, n_test_queries,
                                   selected, cached_mask,
                                   covered_train_mask, covered_test_mask, trajectory)
    elif strategy == "greedy_cost":
        _select_greedy_cost(candidates, budget_B, n_train_queries, n_test_queries,
                            selected, cached_mask,
                            covered_train_mask, covered_test_mask, trajectory)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'")

    return selected, trajectory

#-------------------------------------------------------------------------------
# Scalar helpers
#-------------------------------------------------------------------------------

def _fits_budget(c: CandidateEntry, cached_mask: np.ndarray, n_cached: int, budget_B: int) -> bool:
    if c.R_c is None:
        raise ValueError("_fits_budget requires R_c; use store_rc=True in propose_candidates")
    # Short-circuit: best case (zero overlap) - if it still doesn't fit, reject early.
    if n_cached + c.rc_size <= budget_B:
        return True
    new_vecs = int(np.sum(c.R_c & ~cached_mask))
    return n_cached + new_vecs <= budget_B

def _add_to_cache(c: CandidateEntry, cached_mask: np.ndarray, n_cached: int) -> int:
    new_vecs = int(np.sum(c.R_c & ~cached_mask))
    cached_mask |= c.R_c
    return n_cached + new_vecs

#-------------------------------------------------------------------------------

def _select_random(
    candidates, budget_B, n_train_queries, n_test_queries, seed,
    selected, cached_mask, covered_train_mask, covered_test_mask, trajectory,
):
    n_cached = 0
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(candidates))
    for idx in order:
        c = candidates[idx]
        if _fits_budget(c, cached_mask, n_cached, budget_B):
            prev_train = int(covered_train_mask.sum())
            prev_test = int(covered_test_mask.sum())

            selected.append(c)
            n_cached = _add_to_cache(c, cached_mask, n_cached)
            covered_train_mask |= c.covered_train_indices
            covered_test_mask |= c.covered_test_indices

            _record(selected, n_cached,
                    covered_train_mask, prev_train, n_train_queries,
                    covered_test_mask, prev_test, n_test_queries, trajectory)

#-------------------------------------------------------------------------------
# Vectorized batch helpers for the three greedy strategies below.
#-------------------------------------------------------------------------------

def _fits_mask(working: list, cached_mask: np.ndarray, n_cached: int, budget_B: int) -> np.ndarray:
    """Vectorized _fits_budget across working, touching R_c only for the
    subset that fails the cheap rc_size short-circuit."""
    rc_sizes = np.array([c.rc_size for c in working])
    fits = (n_cached + rc_sizes) <= budget_B
    need_idx = np.flatnonzero(~fits)
    if len(need_idx) > 0:
        R_c_check = np.array([working[i].R_c for i in need_idx])
        new_vecs = np.count_nonzero(R_c_check & ~cached_mask, axis=1)
        fits[need_idx] = (n_cached + new_vecs) <= budget_B
    return fits


def _newly_covered_mask(working: list, covered_train_mask: np.ndarray) -> np.ndarray:
    """Vectorized count of newly-covered train queries for each candidate in
    working, via one fresh stack + one batched reduction."""
    train_matrix = np.array([c.covered_train_indices for c in working])
    return np.count_nonzero(train_matrix & ~covered_train_mask, axis=1)


_DELTA_CHUNK_SIZE = 6000

# Popcount lookup table for the bit-packed path below: POPCOUNT_TABLE[b]
# is the number of set bits in byte value b.
_POPCOUNT_TABLE = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint16)

# Crossover for switching from the fancy-index gather to the bit-packed
# path, expressed as a fraction of the attribute array's length.
_BITPACK_IDX_FRACTION = 1 / 20


def _chunked_delta_counts(active: list, attr: str, idx: np.ndarray,
                           chunk_size: int = _DELTA_CHUNK_SIZE) -> np.ndarray:
    """Count set bits at positions idx within each active candidate's attr
    array (R_c or covered_train_indices), processing active in fixed-size
    batches. Bounds peak transient memory to (chunk_size * len(idx))
    regardless of len(active) -- early greedy_cost iterations can have both
    a large active set and a large idx (many newly-cached bits), so stacking
    the whole active set at once has a data-dependent worst case; chunking
    gives a predictable ceiling instead.

    Two exact (bit-identical, not approximate) code paths per chunk,
    chosen by idx size relative to the attribute length -- see
    _BITPACK_IDX_FRACTION's derivation above. Both compute the exact same
    integer popcount, verified via np.array_equal in
    tests/utils/test_selection.py; this is purely a representation/
    performance choice, not a behavior change.
    """
    if not active:
        return np.empty(0, dtype=np.int64)
    attr_len = len(getattr(active[0], attr))
    use_bitpack = len(idx) > attr_len * _BITPACK_IDX_FRACTION

    counts = np.empty(len(active), dtype=np.int64)
    if use_bitpack:
        mask_full = np.zeros(attr_len, dtype=bool)
        mask_full[idx] = True
        packed_mask = np.packbits(mask_full)
        for start in range(0, len(active), chunk_size):
            chunk = active[start:start + chunk_size]
            batch = np.array([getattr(c, attr) for c in chunk])
            packed_batch = np.packbits(batch, axis=1)
            counts[start:start + len(chunk)] = (
                _POPCOUNT_TABLE[packed_batch & packed_mask[None, :]].sum(axis=1)
            )
    else:
        for start in range(0, len(active), chunk_size):
            chunk = active[start:start + chunk_size]

            batch_full = np.array([getattr(c, attr) for c in chunk])
            batch = batch_full[:, idx]
            counts[start:start + len(chunk)] = np.count_nonzero(batch, axis=1)
    return counts

#-------------------------------------------------------------------------------

def _select_greedy_naive(
    candidates, budget_B, n_train_queries, n_test_queries,
    selected, cached_mask, covered_train_mask, covered_test_mask, trajectory,
):
    working = list(candidates)
    while working:
        n_cached = int(cached_mask.sum())
        fits = _fits_mask(working, cached_mask, n_cached, budget_B)
        working = [c for c, f in zip(working, fits) if f]
        if not working:
            break

        newly = _newly_covered_mask(working, covered_train_mask)
        keep = newly > 0
        working = [c for c, k in zip(working, keep) if k]
        newly = newly[keep]
        if not working:
            break

        best_idx = int(np.argmax(newly))
        best = working.pop(best_idx)
        prev_train = int(covered_train_mask.sum())
        prev_test = int(covered_test_mask.sum())

        selected.append(best)
        n_cached = _add_to_cache(best, cached_mask, n_cached)
        covered_train_mask |= best.covered_train_indices
        covered_test_mask |= best.covered_test_indices

        _record(selected, n_cached,
                covered_train_mask, prev_train, n_train_queries,
                covered_test_mask, prev_test, n_test_queries, trajectory)

#-------------------------------------------------------------------------------

def _select_greedy_cost_static(
    candidates, budget_B, n_train_queries, n_test_queries,
    selected, cached_mask, covered_train_mask, covered_test_mask, trajectory,
):
    working = list(candidates)
    while working:
        n_cached = int(cached_mask.sum())
        fits = _fits_mask(working, cached_mask, n_cached, budget_B)
        working = [c for c, f in zip(working, fits) if f]
        if not working:
            break

        newly = _newly_covered_mask(working, covered_train_mask)
        keep = newly > 0
        working = [c for c, k in zip(working, keep) if k]
        newly = newly[keep]
        if not working:
            break

        static_cost = np.array([c.rc_size for c in working], dtype=float)
        static_cost[static_cost == 0.0] = 1.0
        scores = newly / static_cost

        best_idx = int(np.argmax(scores))
        best = working.pop(best_idx)
        prev_train = int(covered_train_mask.sum())
        prev_test = int(covered_test_mask.sum())

        selected.append(best)
        n_cached = _add_to_cache(best, cached_mask, n_cached)
        covered_train_mask |= best.covered_train_indices
        covered_test_mask |= best.covered_test_indices

        _record(selected, n_cached,
                covered_train_mask, prev_train, n_train_queries,
                covered_test_mask, prev_test, n_test_queries, trajectory)

#-------------------------------------------------------------------------------

def _select_greedy_cost(
    candidates, budget_B, n_train_queries, n_test_queries,
    selected, cached_mask, covered_train_mask, covered_test_mask, trajectory,
):
    # marginal/newly are maintained incrementally across iterations rather
    # than recomputed from scratch each time.
    active = list(candidates)
    deferred: List[CandidateEntry] = []

    if active:
        marginal = np.array([c.rc_size for c in active], dtype=np.int64)
        newly = _newly_covered_mask(active, covered_train_mask)
    else:
        marginal = np.array([], dtype=np.int64)
        newly = np.array([], dtype=np.int64)

    while active:
        n_cached = int(cached_mask.sum())

        fits = (n_cached + marginal) <= budget_B
        active = [c for c, f in zip(active, fits) if f]
        marginal = marginal[fits]
        newly = newly[fits]
        if not active:
            break

        # permanently remove zero-benefit (covered_train_mask only grows, so safe)
        keep = newly > 0
        active = [c for c, k in zip(active, keep) if k]
        marginal = marginal[keep]
        newly = newly[keep]
        if not active:
            break

        # zero-cost positive-benefit candidates: collect for deferred flush
        zero_cost = marginal == 0
        deferred.extend(c for c, z in zip(active, zero_cost) if z)
        non_zero = ~zero_cost
        active = [c for c, z in zip(active, zero_cost) if not z]
        marginal = marginal[non_zero]
        newly = newly[non_zero]
        if not active:
            break

        scores = newly / marginal
        best_idx = int(np.argmax(scores))
        best = active.pop(best_idx)
        marginal = np.delete(marginal, best_idx)
        newly = np.delete(newly, best_idx)

        prev_train = int(covered_train_mask.sum())
        prev_test = int(covered_test_mask.sum())

        # bits best newly contributes
        new_base_idx = np.flatnonzero(best.R_c & ~cached_mask)
        new_train_idx = np.flatnonzero(best.covered_train_indices & ~covered_train_mask)

        selected.append(best)
        n_cached = _add_to_cache(best, cached_mask, n_cached)
        covered_train_mask |= best.covered_train_indices
        covered_test_mask |= best.covered_test_indices

        if active:
            if len(new_base_idx) > 0:
                marginal = marginal - _chunked_delta_counts(active, 'R_c', new_base_idx)
            if len(new_train_idx) > 0:
                newly = newly - _chunked_delta_counts(
                    active, 'covered_train_indices', new_train_idx
                )

        _record(selected, n_cached,
                covered_train_mask, prev_train, n_train_queries,
                covered_test_mask, prev_test, n_test_queries, trajectory)

    # deferred flush: recompute benefit against final covered_train_mask
    # budget always satisfied: R_c is a subset of cached_mask (marginal was 0)
    if deferred:
        deferred_newly = _newly_covered_mask(deferred, covered_train_mask)
        deferred_scored = sorted(
            zip(deferred_newly.tolist(), deferred),
            key=lambda x: x[0],
            reverse=True,
        )
    else:
        deferred_scored = []
    n_cached = int(cached_mask.sum())
    for benefit, c in deferred_scored:
        if benefit > 0:
            prev_train = int(covered_train_mask.sum())
            prev_test = int(covered_test_mask.sum())
            selected.append(c)
            covered_train_mask |= c.covered_train_indices
            covered_test_mask |= c.covered_test_indices
            _record(selected, n_cached,
                    covered_train_mask, prev_train, n_train_queries,
                    covered_test_mask, prev_test, n_test_queries, trajectory)
