#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Confirm selection algorithms work correctly with K=0 (point-equivalent,
exercised via propose_candidates_ball(K=0); see docs/POPULATION_DESIGN.md)
and K>=1 ball candidates. K=0 is a genuinely different R_c/coverage density
pattern than K>=1 ball, worth testing selection against independently."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from utils.base.population import propose_candidates_ball
from utils.base.selection import select_candidates
from tests.helpers import make_clustered_candidate_benchmark, check_selection_invariants

#-------------------------------------------------------------------------------

def _make_benchmark(seed: int = 0):
    """Clustered benchmark so CIG coverage is non-empty (n>=2 for point-like)."""
    return make_clustered_candidate_benchmark(
        seed=seed, dim=12, n_clusters=4,
        n_base_per_cluster=12, n_cache_per_cluster=2, n_test_per_cluster=2,
        K=2, noise_scale=0.05,
    )

#-------------------------------------------------------------------------------

STRATEGIES = ['greedy_naive', 'greedy_cost_static', 'greedy_cost']


def test_selection_point_candidates_greedy():
    bm = _make_benchmark()
    cands = propose_candidates_ball(
        bm['cache'], bm['r_Q'], bm['tv'], bm['ti'], bm['hg'],
        bm['cache'], bm['r_Q'], bm['test'], bm['test_r_t'],
        bm['base'], n_max=3, K=0, metric='angular',
    )
    budget = 30
    for strategy in STRATEGIES:
        selected, trajectory = select_candidates(
            cands, budget, strategy,
            n_train_queries=bm['n_cache'],
            n_test_queries=bm['n_test'],
            seed=42,
        )
        check_selection_invariants(selected, trajectory, budget, f"point/{strategy}")


def test_selection_ball_candidates_greedy():
    bm = _make_benchmark()
    cands = propose_candidates_ball(
        bm['cache'], bm['r_Q'], bm['tv'], bm['ti'], bm['hg'],
        bm['cache'], bm['r_Q'], bm['test'], bm['test_r_t'],
        bm['base'], n_max=3, K=3, metric='angular',
    )
    budget = 30
    for strategy in STRATEGIES:
        selected, trajectory = select_candidates(
            cands, budget, strategy,
            n_train_queries=bm['n_cache'],
            n_test_queries=bm['n_test'],
            seed=42,
        )
        check_selection_invariants(selected, trajectory, budget, f"ball/{strategy}")


def test_selection_budget_not_exceeded():
    bm = _make_benchmark()
    for method in ('point', 'ball'):
        if method == 'point':
            cands = propose_candidates_ball(
                bm['cache'], bm['r_Q'], bm['tv'], bm['ti'], bm['hg'],
                bm['cache'], bm['r_Q'], bm['test'], bm['test_r_t'],
                bm['base'], n_max=3, K=0, metric='angular',
            )
        else:
            cands = propose_candidates_ball(
                bm['cache'], bm['r_Q'], bm['tv'], bm['ti'], bm['hg'],
                bm['cache'], bm['r_Q'], bm['test'], bm['test_r_t'],
                bm['base'], n_max=3, K=3, metric='angular',
            )
        for strategy in ['greedy_cost_static', 'greedy_cost']:
            for budget in [10, 20, 40]:
                selected, trajectory = select_candidates(
                    cands, budget, strategy,
                    n_train_queries=bm['n_cache'],
                    n_test_queries=bm['n_test'],
                    seed=42,
                )
                check_selection_invariants(
                    selected, trajectory, budget, f"{method}/{strategy}/budget={budget}"
                )
