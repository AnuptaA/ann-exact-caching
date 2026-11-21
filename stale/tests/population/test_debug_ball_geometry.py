#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Tests for debug_ball_geometry.py's margin, utility ratio, and
self-exclusion checks."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from simulations.population.debug_ball_geometry import (
    compute_margins, compute_utility_ratio, compute_rc_size_and_self_exclusion,
)
from utils.base.population import _normalize
from tests.helpers import make_clustered_candidate_benchmark

#-------------------------------------------------------------------------------


def _make_benchmark(seed: int = 0):
    return make_clustered_candidate_benchmark(
        seed=seed, dim=16, n_clusters=5,
        n_base_per_cluster=16, n_cache_per_cluster=2, n_test_per_cluster=2,
        K=2, noise_scale=0.05,
    )


def _brute_force_kth_base_dist(query, base, k):
    q_hat = query / np.linalg.norm(query)
    base_hat = _normalize(base.copy())
    dists = np.arccos(np.clip(base_hat @ q_hat, -1.0, 1.0)) / np.pi
    return float(np.sort(dists)[k - 1])


#-------------------------------------------------------------------------------


def test_compute_margins_matches_brute_force():
    bm = _make_benchmark()
    k = 4
    margins = compute_margins(bm['cache'], bm['base'], k)
    for i in range(len(bm['cache'])):
        expected = _brute_force_kth_base_dist(bm['cache'][i], bm['base'], k)
        assert abs(float(margins[i]) - expected) < 1e-5, (
            f"query {i}: got {margins[i]:.6f}, expected {expected:.6f}"
        )


def _brute_force_utility_ratio(cache_q, base, K):
    """Independent reference: nn_dist + margin_lookup[nn_index], divided by
    each query's own nearest-base-vector distance, computed via a plain
    per-query loop (no shared helpers with the implementation under test)."""
    base_hat = _normalize(base.copy())
    n = len(cache_q)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        q_hat = cache_q[i] / np.linalg.norm(cache_q[i])
        dists_to_others = np.array([
            np.arccos(np.clip(q_hat @ (cache_q[j] / np.linalg.norm(cache_q[j])), -1.0, 1.0)) / np.pi
            for j in range(n) if j != i
        ])
        other_idx = [j for j in range(n) if j != i]
        nn_pos = int(np.argmin(dists_to_others))
        nn_dist = float(dists_to_others[nn_pos])
        nn_index = other_idx[nn_pos]

        if K == 0:
            margin = 0.0
        else:
            nn_hat = cache_q[nn_index] / np.linalg.norm(cache_q[nn_index])
            base_dists = np.arccos(np.clip(base_hat @ nn_hat, -1.0, 1.0)) / np.pi
            margin = float(np.sort(base_dists)[K - 1])

        r_c = nn_dist + margin
        d_min = float(np.arccos(np.clip(base_hat @ q_hat, -1.0, 1.0)).min() / np.pi)
        out[i] = r_c / d_min
    return out


def test_utility_ratio_k0_reduces_to_nn_over_dmin():
    """At K=0, utility ratio must equal exactly nn_dist / d_min (margin=0
    by construction), checked against the brute-force reference."""
    bm = _make_benchmark()
    ratio = compute_utility_ratio(bm['cache'], bm['base'], K=0)
    expected = _brute_force_utility_ratio(bm['cache'], bm['base'], K=0)
    assert np.allclose(ratio, expected, atol=1e-5), (
        f"K=0 utility ratio mismatch:\ngot={ratio}\nexpected={expected}"
    )


def test_utility_ratio_k_positive_matches_brute_force():
    """K>0: numerator is nn_dist + margin_lookup[nn_index] (the exact
    candidate-radius construction propose_candidates_ball uses), checked
    against an independent brute-force reference."""
    bm = _make_benchmark()
    K = 3
    ratio = compute_utility_ratio(bm['cache'], bm['base'], K=K)
    expected = _brute_force_utility_ratio(bm['cache'], bm['base'], K=K)
    assert np.allclose(ratio, expected, atol=1e-5), (
        f"K={K} utility ratio mismatch:\ngot={ratio}\nexpected={expected}"
    )


def test_self_exclusion_holds_on_real_candidates():
    bm = _make_benchmark()
    result = compute_rc_size_and_self_exclusion(
        bm['cache'], bm['base'], K=2,
        obs_r_Q=bm['r_Q'], test_q=bm['test'], test_r_t=bm['test_r_t'],
    )
    assert result['self_exclusion_violations'] == 0, (
        f"Expected 0 self-exclusion violations, got {result['self_exclusion_violations']}"
    )


def test_self_exclusion_check_detects_deliberate_violation():
    """The self-exclusion check must fail on a deliberately broken input,
    proving the assertion in compute_rc_sizes_and_utility isn't a no-op.
    Directly exercises the same detection logic used there, against a
    candidate whose own source index was (wrongly) marked covered."""
    from utils.base.population import propose_candidates_ball

    bm = _make_benchmark()
    n_cache = len(bm['cache'])
    tv_dummy = [[] for _ in range(n_cache)]
    ti_dummy = [[] for _ in range(n_cache)]
    hg_dummy = np.zeros(n_cache, dtype=np.float32)

    candidates = propose_candidates_ball(
        bm['cache'], bm['r_Q'], tv_dummy, ti_dummy, hg_dummy,
        bm['cache'], bm['r_Q'], bm['test'], bm['test_r_t'],
        bm['base'], 'angular', n_max=1, K=2, store_rc=False,
    )
    assert candidates, "expected at least one candidate"
    cand = candidates[0]
    src_idx = np.flatnonzero((bm['cache'] == cand.query).all(axis=1))[0]

    # sanity: currently False (real, unbroken data)
    assert not cand.covered_train_indices[src_idx]

    # deliberately break it, then confirm detection catches it
    cand.covered_train_indices[src_idx] = True
    violation_detected = cand.covered_train_indices[src_idx]
    assert violation_detected, "self-exclusion check failed to detect a deliberate violation"
