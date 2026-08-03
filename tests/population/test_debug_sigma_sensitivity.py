#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Tests for debug_sigma_sensitivity.py's certified_counts computation."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from simulations.population.debug_sigma_sensitivity import certified_counts
from utils.base.population import _normalize, propose_candidates_ball
from utils.base.selection import select_candidates
from utils.base.lemmas_sigma import lemma1_no_union_sigma
from tests.helpers import make_clustered_candidate_benchmark

#-------------------------------------------------------------------------------


def _make_benchmark(seed: int = 0):
    return make_clustered_candidate_benchmark(
        seed=seed, dim=16, n_clusters=5,
        n_base_per_cluster=16, n_cache_per_cluster=3, n_test_per_cluster=3,
        K=2, noise_scale=0.05,
    )


def _select_full_budget(bm, K=2):
    n_cache = len(bm['cache'])
    tv_dummy = [[] for _ in range(n_cache)]
    ti_dummy = [[] for _ in range(n_cache)]
    hg_dummy = np.zeros(n_cache, dtype=np.float32)
    candidates = propose_candidates_ball(
        bm['cache'], bm['r_Q'], tv_dummy, ti_dummy, hg_dummy,
        bm['cache'], bm['r_Q'], bm['test'], bm['test_r_t'],
        bm['base'], 'angular', n_max=3, K=K, store_rc=True,
    )
    selected, _trajectory = select_candidates(
        candidates, budget_B=bm['n_base'], strategy='greedy_cost',
        n_train_queries=bm['n_cache'], n_test_queries=bm['n_test'], seed=42,
    )
    return selected


def test_certified_counts_matches_direct_lemma_call():
    bm = _make_benchmark()
    K = 2
    selected = _select_full_budget(bm, K=K)
    assert selected, "expected at least one selected candidate"

    base_hat = _normalize(bm['base'].copy())
    counts = certified_counts(selected, bm['test'], K, bm['base'], base_hat, 'angular')

    entries = [c.to_cache_entry() for c in selected]
    for qi in range(len(bm['test'])):
        expected_best = 0
        for entry in entries:
            _, _, metadata = lemma1_no_union_sigma(
                bm['test'][qi], [entry], K, 1.0, 'angular',
                base_hat=base_hat, base_vectors=bm['base'],
            )
            n_certified = len(metadata.get('certified_indices', []))
            if n_certified > expected_best:
                expected_best = n_certified
            if expected_best >= K:
                break
        assert counts[qi] == expected_best, (
            f"query {qi}: got {counts[qi]}, expected {expected_best}"
        )
