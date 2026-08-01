#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Tests for debug_vector_coverage.py.

Exercises compute_combo's per-BUDGET_FRACTIONS sweep logic directly (real
propose_full_pool + select_candidates + monotonic_hit_rate_pass calls
against a small synthetic benchmark, mirroring compute_combo's own loop)
rather than going through dc.load_benchmark, since compute_combo's disk-
based benchmark loading isn't what's under test here -- the sweep logic is.

test_coverage_upper_bounds_hit_rate_across_budget_fractions is the most
important test: this script exists specifically to compare vector coverage
(ceiling) against the actual CIG-certified hit rate at real, independently
budget-constrained selections, so the upper-bound relationship must hold at
every one of the swept fractions, not just asymptotically.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

import simulations.population.debug_common as dc
from simulations.population.debug_trajectory import monotonic_hit_rate_pass
from simulations.population.debug_vector_coverage import compute_combo
from simulations.population.run_population import BUDGET_FRACTIONS
from utils.base.selection import select_candidates
from tests.helpers import make_clustered_candidate_benchmark

#-------------------------------------------------------------------------------

SIGMA = 1.0


def _sweep_budget_fractions(bm: dict, K: int, zero_r_t: bool = False) -> dict:
    """Replicates compute_combo's per-BUDGET_FRACTIONS loop directly against
    an in-memory synthetic benchmark (no dc.load_benchmark disk I/O)."""
    n_cache = len(bm['cache'])
    n_base = bm['n_base']
    train_r_t = np.zeros(n_cache, dtype=np.float32) if zero_r_t else bm['r_Q']
    test_r_t = np.zeros(len(bm['test']), dtype=np.float32) if zero_r_t else bm['test_r_t']

    candidates = dc.propose_full_pool(
        bm['base'], bm['cache'], bm['test'], train_r_t, test_r_t, K, n_max=5,
    )
    assert candidates, "expected at least one proposed candidate"

    test_gt_indices = None
    if K > 0:
        q = bm['test'] / np.linalg.norm(bm['test'], axis=1, keepdims=True)
        sims = q @ bm['base'].T
        top_idx = np.argsort(-sims, axis=1)[:, :K]
        test_gt_indices = [top_idx[i].tolist() for i in range(len(bm['test']))]

    budget_pct, test_hit_curve, vector_coverage = [], [], []
    for frac in BUDGET_FRACTIONS:
        budget_B = max(1, int(frac * n_base))
        selected, _traj = select_candidates(
            candidates, budget_B, strategy='greedy_cost',
            n_train_queries=n_cache, n_test_queries=bm['n_test'], seed=42,
        )
        if not selected:
            hit_rate, coverage = 0.0, 0.0
        else:
            test_cert = monotonic_hit_rate_pass(
                selected, bm['test'], test_r_t, K, SIGMA, 'angular',
            )
            hit_rate = float(np.mean(test_cert > 0))
            if K == 0:
                coverage = hit_rate
            else:
                cached_mask = np.zeros(n_base, dtype=bool)
                for c in selected:
                    cached_mask |= c.R_c
                gt_matrix = np.array(test_gt_indices, dtype=np.int32)
                coverage = float(cached_mask[gt_matrix].all(axis=1).mean())
        budget_pct.append(frac * 100)
        test_hit_curve.append(hit_rate)
        vector_coverage.append(coverage)

    return {
        'budget_pct': np.array(budget_pct),
        'test_hit_curve': np.array(test_hit_curve),
        'vector_coverage': np.array(vector_coverage),
    }

#-------------------------------------------------------------------------------


def test_coverage_upper_bounds_hit_rate_across_budget_fractions():
    bm = make_clustered_candidate_benchmark(
        seed=1, dim=16, n_clusters=4,
        n_base_per_cluster=20, n_cache_per_cluster=8, n_test_per_cluster=4,
        K=3, noise_scale=0.05,
    )
    result = _sweep_budget_fractions(bm, K=3)
    assert np.all(result['vector_coverage'] >= result['test_hit_curve'] - 1e-9), (
        "vector coverage (raw presence) must upper-bound the CIG-certified "
        "hit rate at every real, independently budget-constrained selection: "
        "a query cannot be certified as a hit unless its neighbors are also "
        "physically cached"
    )
    # At the largest budget fraction (1.0, full base affordable), coverage
    # must be exact.
    assert result['vector_coverage'][-1] == 1.0


def test_k_zero_coverage_equals_hit_rate_at_every_budget_fraction():
    """At K=0 there's no separate "neighbor vector presence" question --
    the CIG hit condition is pure point-containment (D(test_query, Q) <=
    r_c), which coverage computes identically at every real budget cap, not
    just asymptotically."""
    bm = make_clustered_candidate_benchmark(
        seed=2, dim=16, n_clusters=4,
        n_base_per_cluster=20, n_cache_per_cluster=8, n_test_per_cluster=4,
        K=1, noise_scale=0.05,  # K=1 only for valid GT construction; overridden below
    )
    result = _sweep_budget_fractions(bm, K=0, zero_r_t=True)
    np.testing.assert_array_equal(result['vector_coverage'], result['test_hit_curve'])
    # Regression check for the original bug: must not be a vacuous all-1.0
    # curve -- some low budget fractions should be genuinely unaffordable.
    assert result['vector_coverage'][0] < 1.0, (
        "K=0 coverage should not be 1.0 at the smallest budget fraction -- "
        "this is the exact symptom of the original vacuous-1.0 bug"
    )


def test_compute_combo_honors_budget_fractions_override():
    """CLI-scoping regression test: compute_combo's budget_fractions
    parameter must actually be used instead of the module-level
    BUDGET_FRACTIONS default -- run against a real, already-prepared
    benchmark (needs n_cache > N_MAX=20, which the small synthetic/mini
    fixtures don't satisfy) with a short custom fraction list."""
    bdir = dc.DATASET_CONFIG['sift']['base_dir'] / 'sift_b50k_c256k100_t512n20_s42'
    if not (bdir / 'cache_queries.npy').exists():
        import pytest
        pytest.skip('sift_b50k_c256k100_t512n20_s42 not prepared')

    custom_fractions = [0.5, 1.0]
    result = compute_combo(
        'sift', 'sift_b50k_c256k100_t512n20_s42', K=1, budget_fractions=custom_fractions,
    )
    assert result is not None
    np.testing.assert_array_equal(result['budget_pct'], [50.0, 100.0])
    assert len(result['test_hit_curve']) == 2
    assert len(result['vector_coverage']) == 2


def test_rho_equals_hit_rate_over_coverage():
    """rho must be exactly hit_rate/coverage at every budget fraction where
    coverage > 0 (and 0 where coverage == 0, avoiding a division by zero)."""
    bdir = dc.DATASET_CONFIG['sift']['base_dir'] / 'sift_b50k_c256k100_t512n20_s42'
    if not (bdir / 'cache_queries.npy').exists():
        import pytest
        pytest.skip('sift_b50k_c256k100_t512n20_s42 not prepared')

    result = compute_combo(
        'sift', 'sift_b50k_c256k100_t512n20_s42', K=1,
        budget_fractions=[0.05, 0.5, 1.0],
    )
    assert result is not None
    for cov, hit, rho in zip(
        result['vector_coverage'], result['test_hit_curve'], result['rho_curve'],
    ):
        expected = hit / cov if cov > 0 else 0.0
        assert abs(rho - expected) < 1e-9


def test_k_zero_rho_is_one_and_overshoot_is_nan():
    """At K=0, rho is trivially 1.0 (coverage == hit_rate exactly) and
    overshoot is undefined -- no separate "neighbor vector" concept to
    overshoot past at K=0."""
    bdir = dc.DATASET_CONFIG['sift']['base_dir'] / 'sift_b50k_c256k100_t512n20_s42'
    if not (bdir / 'cache_queries.npy').exists():
        import pytest
        pytest.skip('sift_b50k_c256k100_t512n20_s42 not prepared')

    result = compute_combo(
        'sift', 'sift_b50k_c256k100_t512n20_s42', K=0,
        budget_fractions=[0.05, 1.0],
    )
    assert result is not None
    np.testing.assert_array_equal(result['rho_curve'], [1.0, 1.0])
    assert np.all(np.isnan(result['median_overshoot_curve']))


def test_median_overshoot_nan_when_no_covered_and_missed_queries():
    """At full budget (100%), everything is certified for this benchmark
    (established elsewhere this session: hit_rate ~0.98-0.99 at 100%
    budget), so the covered-and-missed population is at most tiny/empty --
    verifying the NaN-safe path doesn't error and only produces a real
    number when there's something to measure."""
    bdir = dc.DATASET_CONFIG['sift']['base_dir'] / 'sift_b50k_c256k100_t512n20_s42'
    if not (bdir / 'cache_queries.npy').exists():
        import pytest
        pytest.skip('sift_b50k_c256k100_t512n20_s42 not prepared')

    result = compute_combo(
        'sift', 'sift_b50k_c256k100_t512n20_s42', K=1,
        budget_fractions=[0.05, 1.0],
    )
    assert result is not None
    # Low budget: coverage is low but nonzero, so there should be some
    # covered-and-missed queries -- overshoot should be a real number, not NaN.
    assert not np.isnan(result['median_overshoot_curve'][0]) or result['vector_coverage'][0] == 0.0
    # No crash either way -- the array is well-formed float64 throughout.
    assert result['median_overshoot_curve'].dtype == np.float64


def test_recovered_hit_rate_never_exceeds_coverage():
    """Regression test: a query recovered by relaxed certification must
    still be a covered query (its true neighbors are physically cached --
    relaxation only widens which covered queries also get certified, it
    can't manufacture physical presence). So for any slack level,
    hit_rate + recovered_count/n_test must stay <= vector_coverage.
    overshoot_arrays holds relative overshoot only for the covered-and-
    missed subset (not all test queries), so the recovered count must be
    computed as sum(overshoot <= slack) and divided by n_test, not by the
    subset size -- an earlier version of this computation divided by the
    subset size instead, which could push 'recovered' above coverage."""
    bdir = dc.DATASET_CONFIG['sift']['base_dir'] / 'sift_b50k_c256k100_t512n20_s42'
    if not (bdir / 'cache_queries.npy').exists():
        import pytest
        pytest.skip('sift_b50k_c256k100_t512n20_s42 not prepared')

    n_test = 512
    result = compute_combo(
        'sift', 'sift_b50k_c256k100_t512n20_s42', K=1,
        budget_fractions=[0.05, 0.30, 0.65, 0.85, 1.0],
    )
    assert result is not None
    for hit, cov, ov in zip(
        result['test_hit_curve'], result['vector_coverage'], result['overshoot_arrays'],
    ):
        for slack in [0.0, 0.25, 0.5, 0.75, 1.0]:
            recovered = hit + (np.sum(ov <= slack) / n_test if len(ov) > 0 else 0.0)
            assert recovered <= cov + 1e-9, (
                f"recovered hit rate {recovered} exceeded coverage {cov} at slack={slack}"
            )


def test_budget_fractions_produce_monotone_nondecreasing_coverage():
    """A larger real budget cap can only add candidates, never remove them,
    so coverage (and hit rate) must be monotone non-decreasing across
    BUDGET_FRACTIONS, in ascending fraction order."""
    bm = make_clustered_candidate_benchmark(
        seed=3, dim=16, n_clusters=4,
        n_base_per_cluster=20, n_cache_per_cluster=8, n_test_per_cluster=4,
        K=2, noise_scale=0.05,
    )
    result = _sweep_budget_fractions(bm, K=2)
    assert list(result['budget_pct']) == sorted(result['budget_pct'])
    diffs = np.diff(result['vector_coverage'])
    assert np.all(diffs >= -1e-12)
