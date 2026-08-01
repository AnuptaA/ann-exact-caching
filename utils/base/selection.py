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
# Scalar helpers -- used by _select_random, which processes one candidate at a
# time from a fixed permutation and never rescans, so it has no need for the
# batched/vectorized helpers below.
#-------------------------------------------------------------------------------

def _fits_budget(c: CandidateEntry, cached_mask: np.ndarray, n_cached: int, budget_B: int) -> bool:
    if c.R_c is None:
        raise ValueError("_fits_budget requires R_c; use store_rc=True in propose_candidates")
    # Short-circuit: best case (zero overlap) — if it still doesn't fit, reject early.
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
#
# All three repeatedly rescan a shrinking "working"/"active" list to find the
# next best candidate. At large budgets, _fits_budget's capacity check stops
# rejecting candidates (the budget rarely gets exceeded), so the active set
# stays large for many iterations -- a Python-level loop calling one numpy
# reduction per candidate (the previous implementation) pays per-candidate
# Python/numpy dispatch overhead on every single iteration, which dominates
# at scale (~41k candidates x hundreds of iterations on real data).
#
# Design notes (important for memory -- an earlier version of this file
# maintained a persistent stacked matrix shrunk via repeated boolean-mask
# indexing, which turned out to create several large transient copies per
# iteration, ballooning peak memory to ~6GB with no speed benefit over the
# original Python loop at real scale; that approach was abandoned):
#   - The object list itself (`working`/`active`) is filtered the same cheap
#     way the original code did (a plain Python list comprehension) -- this
#     never touches R_c/covered_train_indices data, so it's O(active) in
#     Python-object-reference cost only, not O(active * n_base).
#   - A 2D stack of R_c/covered_train_indices is only ever built fresh, for
#     whatever the *current* (already-filtered) working set is, used for
#     exactly one vectorized computation, and then discarded -- never
#     persisted or incrementally shrunk. This bounds peak memory to one
#     stack's worth of data at a time.
#   - For greedy_naive/greedy_cost_static, R_c is only needed for the budget
#     fits-check (scoring uses covered_train_indices and, for cost_static,
#     the scalar rc_size -- never R_c itself), so the expensive R_c check is
#     only performed for the subset that fails the cheap rc_size
#     short-circuit -- typically small at large budgets, since most
#     candidates trivially fit.
#   - greedy_cost no longer follows this "fresh stack per iteration" pattern
#     for its own marginal/newly bookkeeping -- see the incremental-delta
#     design note at the top of _select_greedy_cost below, which replaced
#     the per-iteration full recompute with delta updates (a further round
#     of optimization after the "fresh stack" approach above stopped being
#     fast enough at budget_fraction close to 1.0, where the active set
#     barely shrinks and hundreds of full O(active * n_base) recomputes
#     dominated runtime). The "fresh stack, one-time use" pattern is still
#     exactly right for greedy_naive/greedy_cost_static below, which don't
#     have this problem (their filters shrink the active set quickly).
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
# path, expressed as a fraction of the attribute array's length. Measured
# directly (utils/base/selection.py's _chunked_delta_counts profiling,
# real SIFT c2048/K=5 data): fancy-index gather cost scales with
# len(idx), bit-packing's cost is ~fixed per chunk (dominated by
# np.packbits(chunk, axis=1), independent of idx size) -- the two cross
# over empirically around idx ~= attr_len/20 (e.g. ~2500 for a 50000-long
# R_c array). Below that, gather wins (often heavily -- most calls in a
# real run have idx in the single digits to low hundreds, where packing
# would be a large regression); above it, packing wins by up to ~12x at
# the largest real idx sizes observed (~33000).
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
            # One cheap full-array grab per candidate (contiguous copy, not
            # fancy-indexed) then a single vectorized 2D gather across the
            # whole chunk -- not one fancy-index call per candidate. The
            # per-candidate version (getattr(c, attr)[idx] in the list
            # comprehension) was measured spending ~97% of wall time in
            # numpy's mapiter_trivial_get at real scale (c4096, K=0: many
            # greedy iterations x a large active set, each iteration paying
            # per-call fancy-index overhead len(active) times). Matches the
            # bitpack path's existing full-grab pattern; bit-identical
            # result, not an approximation.
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
    # than recomputed from scratch each time. cached_mask/covered_train_mask
    # only ever grow (|= only), so both quantities are monotonically
    # non-increasing, and the *change* each iteration depends only on the
    # bits the just-selected candidate's R_c/covered_train_indices newly
    # set -- not on the full n_base/n_train_queries dimension. This turns
    # the dominant per-iteration cost from O(|active| * n_base) (rebuilding
    # and reducing a fresh stacked matrix every iteration) into
    # O(|active| * k), where k is the number of newly-set bits that
    # iteration; summed over the whole run, total k is bounded by n_base
    # (for marginal) / n_train_queries (for newly), giving a large net
    # reduction in total work at large budgets, where the active set barely
    # shrinks. Numerically exact: these are integer popcounts of set
    # differences, not an approximation.
    active = list(candidates)
    deferred: List[CandidateEntry] = []

    if active:
        # cached_mask is freshly all-zero here (just initialized in
        # select_candidates, untouched before this point), so
        # count_nonzero(c.R_c & ~cached_mask) == count_nonzero(c.R_c) ==
        # c.rc_size exactly -- already computed and stored on every
        # candidate, so the initial marginal needs no matrix construction
        # at all (this used to build a full (|active|, n_base) ~2GB stack
        # plus a same-size transient AND result, ~4.1GB peak at real scale).
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

        # Bits best newly contributes -- computed before cached_mask/
        # covered_train_mask are mutated below, so this is exactly the set
        # difference that needs to be subtracted from every remaining
        # active candidate's marginal/newly.
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
