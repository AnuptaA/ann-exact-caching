#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Cross-check exact_intersection_fraction (debug_intersection_calibration.py)
against an independently-derived reference formula, implemented fresh here
without calling into the existing module's internals.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from scipy.special import betainc
from scipy.integrate import quad

import simulations.population.debug_common as dc
from simulations.population.debug_intersection_calibration import (
    exact_intersection_fraction, sample_cap_points,
)
from utils.base.selection import select_candidates

#-------------------------------------------------------------------------------
# Reference formula (independent implementation)
#-------------------------------------------------------------------------------


def _single_cap_fraction_reference(r: float, d: int) -> float:
    """Independent reimplementation of the single-cap closed form (Li 2011)."""
    theta = r * np.pi
    x = np.sin(theta) ** 2
    a = (d - 1) / 2.0
    ib = betainc(a, 0.5, x)
    return 0.5 * ib if theta <= np.pi / 2 else 1.0 - 0.5 * ib


def reference_intersection_fraction(D: float, r_c: float, r_t: float, d: int) -> float:
    """Independent reference implementation of intersection_fraction(c, t):

        tau(phi) = (cos(r_c) - cos(phi)*cos(Delta)) / (sin(phi)*sin(Delta))
        g(phi) = 1 if tau(phi) < -1, 0 if tau(phi) > 1, else the branched
                 small/large-cap fraction of the cross-sectional sphere
        intersection_fraction = integral_0^{r_t} sin(phi)^(d-2) g(phi) dphi
                               / integral_0^{r_t} sin(phi)^(d-2) dphi

    D, r_c, r_t are in this codebase's normalized units (angle/pi); d is
    the ambient dimension (sphere is S^{d-1}). Inputs are cast to float64
    defensively -- float32 pushes some near-boundary configs into spurious
    underflow.
    """
    D, r_c, r_t = float(D), float(r_c), float(r_t)
    D_a, rc_a, rt_a = D * np.pi, r_c * np.pi, r_t * np.pi

    if D_a >= rc_a + rt_a:
        return 0.0
    if D_a + rt_a <= rc_a:
        return 1.0
    if D_a + rc_a <= rt_a:
        return _single_cap_fraction_reference(r_c, d) / _single_cap_fraction_reference(r_t, d)
    if D_a < 1e-12:
        return 1.0 if rc_a >= rt_a else (
            _single_cap_fraction_reference(r_c, d) / _single_cap_fraction_reference(r_t, d)
        )

    def g(phi, branched: bool = True):
        tau = (np.cos(rc_a) - np.cos(phi) * np.cos(D_a)) / (np.sin(phi) * np.sin(D_a))
        if tau < -1:
            return 1.0
        if tau > 1:
            return 0.0
        r_cross = np.arccos(tau)
        x = np.sin(r_cross) ** 2
        a = (d - 2) / 2.0
        ib = betainc(a, 0.5, x)
        if not branched:
            return 0.5 * ib
        return 0.5 * ib if r_cross <= np.pi / 2 else 1.0 - 0.5 * ib

    eps = 1e-10
    num, _ = quad(lambda phi: (np.sin(phi) ** (d - 2)) * g(phi), eps, rt_a, limit=200)
    den, _ = quad(lambda phi: np.sin(phi) ** (d - 2), eps, rt_a, limit=200)
    return num / den


def reference_intersection_fraction_unbranched(D: float, r_c: float, r_t: float, d: int) -> float:
    """Same as reference_intersection_fraction but without the small/large
    cap branch correction in g(phi) -- for testing that the branch matters."""
    D_a, rc_a, rt_a = D * np.pi, r_c * np.pi, r_t * np.pi
    if D_a >= rc_a + rt_a:
        return 0.0
    if D_a + rt_a <= rc_a:
        return 1.0
    if D_a + rc_a <= rt_a:
        return _single_cap_fraction_reference(r_c, d) / _single_cap_fraction_reference(r_t, d)
    if D_a < 1e-12:
        return 1.0 if rc_a >= rt_a else (
            _single_cap_fraction_reference(r_c, d) / _single_cap_fraction_reference(r_t, d)
        )

    def g_unbranched(phi):
        tau = (np.cos(rc_a) - np.cos(phi) * np.cos(D_a)) / (np.sin(phi) * np.sin(D_a))
        if tau < -1:
            return 1.0
        if tau > 1:
            return 0.0
        r_cross = np.arccos(tau)
        x = np.sin(r_cross) ** 2
        a = (d - 2) / 2.0
        return 0.5 * betainc(a, 0.5, x)

    eps = 1e-10
    num, _ = quad(lambda phi: (np.sin(phi) ** (d - 2)) * g_unbranched(phi), eps, rt_a, limit=200)
    den, _ = quad(lambda phi: np.sin(phi) ** (d - 2), eps, rt_a, limit=200)
    return num / den

#-------------------------------------------------------------------------------
# Validation checks
#-------------------------------------------------------------------------------


def test_hemisphere_sanity():
    """d=3, r_c=r_t=Delta=pi/2: symmetric case, expected exactly 0.5."""
    D = r_c = r_t = 0.5
    d = 3
    ref = reference_intersection_fraction(D, r_c, r_t, d)
    existing = exact_intersection_fraction(D, r_c, r_t, d)
    assert abs(ref - 0.5) < 1e-9
    assert abs(existing - 0.5) < 1e-9


def test_branch_correction_matters():
    """Asymmetric config where tau(phi) < 0 for part of the range: the
    branched and unbranched reference formulas must diverge substantially,
    and the existing implementation must match the branched (correct) one."""
    D, r_c, r_t, d = 0.15, 0.3, 0.2, 16
    branched = reference_intersection_fraction(D, r_c, r_t, d)
    unbranched = reference_intersection_fraction_unbranched(D, r_c, r_t, d)
    existing = exact_intersection_fraction(D, r_c, r_t, d)
    assert abs(branched - unbranched) > 0.5
    assert abs(existing - branched) < 1e-6


def test_degenerate_concentric_poles():
    d = 20
    for r_c, r_t in [(0.1, 0.2), (0.2, 0.1)]:
        ref = reference_intersection_fraction(0.0, r_c, r_t, d)
        existing = exact_intersection_fraction(0.0, r_c, r_t, d)
        expected = 1.0 if r_c >= r_t else (
            _single_cap_fraction_reference(r_c, d) / _single_cap_fraction_reference(r_t, d)
        )
        assert abs(ref - expected) < 1e-6
        assert abs(existing - expected) < 1e-6


def test_degenerate_no_overlap():
    d = 20
    D, r_c, r_t = 0.5, 0.1, 0.1
    assert reference_intersection_fraction(D, r_c, r_t, d) == 0.0
    assert exact_intersection_fraction(D, r_c, r_t, d) == 0.0


def test_degenerate_full_containment_candidate_larger():
    """t's cap fully inside Q's cap: fraction is exactly 1.0."""
    d = 20
    D, r_c, r_t = 0.05, 0.3, 0.1
    assert reference_intersection_fraction(D, r_c, r_t, d) == 1.0
    assert exact_intersection_fraction(D, r_c, r_t, d) == 1.0


def test_degenerate_full_containment_query_larger():
    """Q's cap fully inside t's cap: fraction is Vol(cap_c)/Vol(cap_t), a
    ratio in (0,1), not exactly 0 or 1."""
    d = 20
    D, r_c, r_t = 0.05, 0.1, 0.3
    ref = reference_intersection_fraction(D, r_c, r_t, d)
    existing = exact_intersection_fraction(D, r_c, r_t, d)
    assert 0.0 < ref < 1.0
    assert abs(ref - existing) < 1e-6


@pytest.mark.parametrize("dataset,benchmark_name,K,budget_frac,d", [
    ('sift', 'sift_b50k_c1024k100_t512n20_s42', 5, 0.5, 128),
    ('esci', 'esci_c1024k100_t512n20_s42', 5, 0.5, 384),
])
def test_matches_existing_on_real_pairs(dataset, benchmark_name, K, budget_frac, d):
    """Reference and existing implementations must agree on real
    (candidate, test query) pairs sampled from a prepared benchmark."""
    bdir = dc.DATASET_CONFIG[dataset]['base_dir'] / benchmark_name
    if not (bdir / 'cache_queries.npy').exists():
        pytest.skip(f'{benchmark_name} not prepared')

    base, cache_q, test_q, bdir = dc.load_benchmark(dataset, benchmark_name)
    n_cache, n_test = len(cache_q), len(test_q)
    train_r_t, test_r_t = dc.load_gt_r_t(bdir, K, n_cache, n_test)
    n_base = len(base)
    candidates = dc.propose_full_pool(base, cache_q, test_q, train_r_t, test_r_t, K)
    budget_B = max(1, int(budget_frac * n_base))
    selected, _ = select_candidates(
        candidates, budget_B, strategy=dc.STRATEGY,
        n_train_queries=n_cache, n_test_queries=n_test, seed=dc.SEED,
    )
    assert selected

    test_q_hat = test_q / np.linalg.norm(test_q, axis=1, keepdims=True)
    cand_centers = np.array([c.query for c in selected])
    cand_centers_hat = cand_centers / np.linalg.norm(cand_centers, axis=1, keepdims=True)
    cand_rc = np.array([c.r_c for c in selected])

    rng = np.random.default_rng(0)
    n_pairs = 300
    cand_idx = rng.integers(0, len(selected), n_pairs)
    test_idx = rng.integers(0, n_test, n_pairs)

    diffs = []
    for ci, ti in zip(cand_idx, test_idx):
        dot = np.clip(cand_centers_hat[ci] @ test_q_hat[ti], -1.0, 1.0)
        D = float(np.arccos(dot) / np.pi)
        r_c = float(cand_rc[ci])
        r_t = float(test_r_t[ti])
        ref = reference_intersection_fraction(D, r_c, r_t, d)
        existing = exact_intersection_fraction(D, r_c, r_t, d)
        diffs.append(abs(ref - existing))

    diffs = np.array(diffs)
    assert np.percentile(diffs, 99) < 1e-4, diffs
    assert diffs.max() < 1e-3, diffs


def test_matches_monte_carlo_subsample():
    """Reference, existing, and an independent Monte Carlo sampler must all
    agree within the MC estimator's own standard error."""
    d = 128
    rng = np.random.default_rng(1)
    configs = [
        (0.05, 0.1, 0.08, d), (0.02, 0.05, 0.05, d), (0.1, 0.15, 0.12, d),
        (0.08, 0.2, 0.15, d), (0.03, 0.06, 0.04, d),
    ]
    n_samples = 50000
    for D, r_c, r_t, d in configs:
        ref = reference_intersection_fraction(D, r_c, r_t, d)
        existing = exact_intersection_fraction(D, r_c, r_t, d)

        pole_t = np.zeros(d, dtype=np.float32); pole_t[0] = 1.0
        pole_q = np.zeros(d, dtype=np.float32)
        pole_q[0] = np.cos(D * np.pi); pole_q[1] = np.sin(D * np.pi)
        pts = sample_cap_points(pole_t[None, :], np.array([r_t]), n_samples, rng)[0]
        dist_to_q = np.arccos(np.clip(pts @ pole_q, -1.0, 1.0)) / np.pi
        mc = float(np.mean(dist_to_q <= r_c))
        se = np.sqrt(max(mc * (1 - mc), 1e-9) / n_samples)

        assert abs(ref - existing) < 1e-4, (D, r_c, r_t, d, ref, existing)
        assert abs(ref - mc) < 4 * se + 0.005, (D, r_c, r_t, d, ref, mc, se)
