#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import numpy as np
from typing import List, Tuple

from .population import CandidateEntry

#-------------------------------------------------------------------------------

def _record(
    selected: list,
    cached_vector_set: set,
    covered_train_set: set,
    prev_train_count: int,
    n_train_queries: int,
    covered_test_set: set,
    prev_test_count: int,
    n_test_queries: int,
    trajectory: list,
) -> None:
    n_train = len(covered_train_set)
    n_test = len(covered_test_set)
    trajectory.append({
        "num_entries": len(selected),
        "vectors_cached": len(cached_vector_set),
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
    cached_vector_set: set = set()
    covered_train_set: set = set()
    covered_test_set: set = set()
    trajectory: List[dict] = []

    if strategy == "random":
        _select_random(candidates, budget_B, n_train_queries, n_test_queries,
                       seed, selected, cached_vector_set,
                       covered_train_set, covered_test_set, trajectory)
    elif strategy == "greedy_naive":
        _select_greedy_naive(candidates, budget_B, n_train_queries, n_test_queries,
                             selected, cached_vector_set,
                             covered_train_set, covered_test_set, trajectory)
    elif strategy == "greedy_cost_static":
        _select_greedy_cost_static(candidates, budget_B, n_train_queries, n_test_queries,
                                   selected, cached_vector_set,
                                   covered_train_set, covered_test_set, trajectory)
    elif strategy == "greedy_cost":
        _select_greedy_cost(candidates, budget_B, n_train_queries, n_test_queries,
                            selected, cached_vector_set,
                            covered_train_set, covered_test_set, trajectory)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'")

    return selected, trajectory

#-------------------------------------------------------------------------------

def _fits_budget(c: CandidateEntry, cached_vector_set: set, budget_B: int) -> bool:
    if c.R_c is None:
        raise ValueError("_fits_budget requires R_c; use store_rc=True in propose_candidates")
    return len(cached_vector_set | c.R_c) <= budget_B

#-------------------------------------------------------------------------------

def _select_random(
    candidates, budget_B, n_train_queries, n_test_queries, seed,
    selected, cached_vector_set, covered_train_set, covered_test_set, trajectory,
):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(candidates))
    for idx in order:
        c = candidates[idx]
        if _fits_budget(c, cached_vector_set, budget_B):
            prev_train = len(covered_train_set)
            prev_test = len(covered_test_set)

            selected.append(c)

            cached_vector_set |= c.R_c
            covered_train_set |= c.covered_train_indices
            covered_test_set |= c.covered_test_indices

            _record(selected, cached_vector_set,
                    covered_train_set, prev_train, n_train_queries,
                    covered_test_set, prev_test, n_test_queries, trajectory)

#-------------------------------------------------------------------------------

def _select_greedy_naive(
    candidates, budget_B, n_train_queries, n_test_queries,
    selected, cached_vector_set, covered_train_set, covered_test_set, trajectory,
):
    working = list(candidates)
    while working:
        working = [c for c in working if _fits_budget(c, cached_vector_set, budget_B)]
        if not working:
            break

        newly = np.array([len(c.covered_train_indices - covered_train_set) for c in working])
        keep = newly > 0
        working = [c for c, k in zip(working, keep) if k]
        newly = newly[keep]

        if not working:
            break

        best_idx = int(np.argmax(newly))
        best = working.pop(best_idx)
        prev_train = len(covered_train_set)
        prev_test = len(covered_test_set)

        selected.append(best)

        cached_vector_set |= best.R_c
        covered_train_set |= best.covered_train_indices
        covered_test_set |= best.covered_test_indices

        _record(selected, cached_vector_set,
                covered_train_set, prev_train, n_train_queries,
                covered_test_set, prev_test, n_test_queries, trajectory)

#-------------------------------------------------------------------------------

def _select_greedy_cost_static(
    candidates, budget_B, n_train_queries, n_test_queries,
    selected, cached_vector_set, covered_train_set, covered_test_set, trajectory,
):
    working = list(candidates)
    while working:
        working = [c for c in working if _fits_budget(c, cached_vector_set, budget_B)]
        if not working:
            break

        newly = np.array([len(c.covered_train_indices - covered_train_set) for c in working])
        keep = newly > 0
        working = [c for c, k in zip(working, keep) if k]
        newly = newly[keep]

        if not working:
            break

        static_cost = np.array([len(c.R_c) for c in working], dtype=float)
        static_cost[static_cost == 0.0] = 1.0  # guard for degenerate empty-ball entries
        scores = newly / static_cost

        best_idx = int(np.argmax(scores))
        best = working.pop(best_idx)
        prev_train = len(covered_train_set)
        prev_test = len(covered_test_set)

        selected.append(best)

        cached_vector_set |= best.R_c
        covered_train_set |= best.covered_train_indices
        covered_test_set |= best.covered_test_indices

        _record(selected, cached_vector_set,
                covered_train_set, prev_train, n_train_queries,
                covered_test_set, prev_test, n_test_queries, trajectory)

#-------------------------------------------------------------------------------

def _select_greedy_cost(
    candidates, budget_B, n_train_queries, n_test_queries,
    selected, cached_vector_set, covered_train_set, covered_test_set, trajectory,
):
    active = list(candidates)
    deferred: List[CandidateEntry] = []

    while active:
        active = [c for c in active if _fits_budget(c, cached_vector_set, budget_B)]
        if not active:
            break

        marginal = np.array([len(c.R_c - cached_vector_set) for c in active])
        newly = np.array([len(c.covered_train_indices - covered_train_set) for c in active])

        # permanently remove zero-benefit (covered_train_set only grows, so safe)
        keep = newly > 0
        active = [c for c, k in zip(active, keep) if k]
        marginal = marginal[keep]
        newly = newly[keep]
        if not active:
            break

        # zero-cost positive-benefit candidates: collect for deferred flush
        zero_cost = marginal == 0
        deferred.extend(c for c, z in zip(active, zero_cost) if z)
        active = [c for c, z in zip(active, zero_cost) if not z]
        marginal = marginal[~zero_cost]
        newly = newly[~zero_cost]
        if not active:
            break

        scores = newly / marginal
        best_idx = int(np.argmax(scores))
        best = active.pop(best_idx)
        prev_train = len(covered_train_set)
        prev_test = len(covered_test_set)

        selected.append(best)

        cached_vector_set |= best.R_c
        covered_train_set |= best.covered_train_indices
        covered_test_set |= best.covered_test_indices
        
        _record(selected, cached_vector_set,
                covered_train_set, prev_train, n_train_queries,
                covered_test_set, prev_test, n_test_queries, trajectory)

    # deferred flush: recompute benefit against final covered_train_set
    # budget always satisfied: R_c is a subset of cached_vector_set (marginal was 0)
    deferred_scored = sorted(
        [(len(c.covered_train_indices - covered_train_set), c) for c in deferred],
        key=lambda x: x[0],
        reverse=True,
    )
    for benefit, c in deferred_scored:
        if benefit > 0:
            prev_train = len(covered_train_set)
            prev_test = len(covered_test_set)
            selected.append(c)
            covered_train_set |= c.covered_train_indices
            covered_test_set |= c.covered_test_indices
            _record(selected, cached_vector_set,
                    covered_train_set, prev_train, n_train_queries,
                    covered_test_set, prev_test, n_test_queries, trajectory)
