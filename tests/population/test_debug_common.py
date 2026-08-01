#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Tests for debug_common.py's shared setup helper (Phase 3 Round 14, plan
Step G)."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from simulations.population.debug_common import (
    propose_and_select_full_budget, load_gt_r_t,
)
from utils.base.population import propose_candidates_ball
from utils.base.selection import select_candidates
from tests.helpers import make_clustered_candidate_benchmark, check_selection_invariants

#-------------------------------------------------------------------------------


def _make_benchmark(seed: int = 0):
    return make_clustered_candidate_benchmark(
        seed=seed, dim=16, n_clusters=5,
        n_base_per_cluster=16, n_cache_per_cluster=2, n_test_per_cluster=2,
        K=2, noise_scale=0.05,
    )


def test_propose_and_select_full_budget_matches_direct_calls():
    """propose_and_select_full_budget must return the same candidates
    propose_candidates_ball + select_candidates would, called directly."""
    bm = _make_benchmark()
    K = 2

    selected, trajectory = propose_and_select_full_budget(
        bm['base'], bm['cache'], bm['test'], bm['r_Q'], bm['test_r_t'], K,
        n_max=3, seed=42, metric='angular',
    )

    n_cache = len(bm['cache'])
    tv_dummy = [[] for _ in range(n_cache)]
    ti_dummy = [[] for _ in range(n_cache)]
    hg_dummy = np.zeros(n_cache, dtype=np.float32)
    candidates = propose_candidates_ball(
        bm['cache'], bm['r_Q'], tv_dummy, ti_dummy, hg_dummy,
        bm['cache'], bm['r_Q'], bm['test'], bm['test_r_t'],
        bm['base'], 'angular', n_max=3, K=K, store_rc=True,
    )
    expected_selected, expected_trajectory = select_candidates(
        candidates, budget_B=bm['n_base'], strategy='greedy_cost',
        n_train_queries=bm['n_cache'], n_test_queries=bm['n_test'], seed=42,
    )

    assert len(selected) == len(expected_selected)
    assert trajectory == expected_trajectory
    for a, b in zip(selected, expected_selected):
        assert a.rc_size == b.rc_size
        assert np.array_equal(a.query, b.query)

    check_selection_invariants(selected, trajectory, bm['n_base'], label="debug_common")
    print("  OK: propose_and_select_full_budget matches direct propose+select calls")


def test_load_gt_r_t_k0_returns_zeros():
    """K=0 (containment mode) must return all-zero train/test r_t, matching
    population_runner.py's k_test_override=0 branch."""
    train_r_t, test_r_t = load_gt_r_t(Path('.'), K=0, n_cache=5, n_test=3)
    assert np.all(train_r_t == 0.0)
    assert np.all(test_r_t == 0.0)
    assert len(train_r_t) == 5 and len(test_r_t) == 3
    print("  OK: load_gt_r_t(K=0) returns all-zero arrays without touching disk")


#-------------------------------------------------------------------------------

def main() -> None:
    tests = [
        test_propose_and_select_full_budget_matches_direct_calls,
        test_load_gt_r_t_k0_returns_zeros,
    ]
    for fn in tests:
        print(f"\n[{fn.__name__}]")
        fn()
    print("\nAll passed.")


if __name__ == '__main__':
    main()
