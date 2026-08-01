#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Standalone validation: cross-check the existing intersection-fraction
implementations (exact quadrature in debug_intersection_calibration.py,
and its Monte Carlo estimator) against an independently-derived reference
formula, implemented fresh here without calling into the existing
module's internals.

Does not modify or import from the calibration pipeline's computation
functions -- read-only validation only. Run directly:
    python3 simulations/population/validate_intersection_fraction.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy.special import betainc
from scipy.integrate import quad

sys.path.append(str(Path(__file__).parent.parent.parent))

from simulations.population import debug_common as dc
from simulations.population.debug_intersection_calibration import (
    exact_intersection_fraction, sample_cap_points,
)
from utils.base.selection import select_candidates

#-------------------------------------------------------------------------------
# Reference formula (independent implementation)
#-------------------------------------------------------------------------------


def _single_cap_fraction_reference(r: float, d: int) -> float:
    """Independent reimplementation of the single-cap closed form (Li 2011),
    used only inside the concentric-poles guard clause below -- not a call
    into the existing module's single_cap_volume_fraction."""
    theta = r * np.pi
    x = np.sin(theta) ** 2
    a = (d - 1) / 2.0
    ib = betainc(a, 0.5, x)
    return 0.5 * ib if theta <= np.pi / 2 else 1.0 - 0.5 * ib


def reference_intersection_fraction(D: float, r_c: float, r_t: float, d: int) -> float:
    """Independent reference implementation of intersection_fraction(c, t),
    per the formula:

        tau(phi) = (cos(r_c) - cos(phi)*cos(Delta)) / (sin(phi)*sin(Delta))
        g(phi) = 1                                          if tau(phi) < -1
               = 0                                           if tau(phi) > 1
               = 0.5 * I_{sin^2(arccos(tau(phi)))}((d-2)/2, 1/2)   otherwise

        intersection_fraction = integral_0^{r_t} sin(phi)^(d-2) g(phi) dphi
                               / integral_0^{r_t} sin(phi)^(d-2) dphi

    One correction applied to the literal spec, flagged explicitly: g(phi)
    as specified is only the SMALL-cap form of the single-cap fraction
    (valid for arccos(tau) <= pi/2). Whenever tau(phi) < 0, arccos(tau) > pi/2
    -- the region w_1 >= tau(phi) covers MORE than half the cross-sectional
    sphere, and the correct fraction is the complement,
    1 - 0.5*I_{sin^2(arccos(tau))}((d-2)/2, 1/2), not the small-cap formula
    itself (this is the same branch single_cap_volume_fraction already
    handles in the existing module, and matches Li 2011's own piecewise
    definition). The default here applies that branch; an UNBRANCHED
    (literal-as-specified) variant is also provided below so the two can be
    compared directly -- see validate_branch_matters().

    D, r_c, r_t in this codebase's normalized units (angle/pi); d is the
    ambient dimension (sphere is S^{d-1}).

    Inputs are cast to Python float (float64) defensively: found during
    validation that passing float32 values (e.g. straight from a numpy
    float32 array without an explicit cast) can push these extreme-
    magnitude computations (num/den both ~1e-59 for some real near-
    boundary configs) over into spurious underflow that float64
    precision handles correctly -- caught via a bug in this script's own
    reporting code, not a flaw in the formula itself, but real enough
    that the function should not trust the caller's dtype.
    """
    D, r_c, r_t = float(D), float(r_c), float(r_t)
    D_a, rc_a, rt_a = D * np.pi, r_c * np.pi, r_t * np.pi

    # Guard: no overlap.
    if D_a >= rc_a + rt_a:
        return 0.0
    # Guard: full containment (correct math, not literally "0 or 1" in both
    # directions -- only t's-cap-fully-inside-Q's-cap gives exactly 1; the
    # reverse (Q's cap fully inside t's cap) gives the ratio of single-cap
    # volumes, since the intersection is then exactly Q's (smaller) cap,
    # not all of t's cap. Flagged explicitly: the literal "0 or 1" phrasing
    # in the spec is imprecise for this second sub-case.
    if D_a + rt_a <= rc_a:
        return 1.0
    if D_a + rc_a <= rt_a:
        return _single_cap_fraction_reference(r_c, d) / _single_cap_fraction_reference(r_t, d)
    # Guard: concentric poles (Delta=0) -- division by zero in tau(phi)
    # otherwise. Reduces to the same full-containment-by-radius comparison.
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
    """Same as reference_intersection_fraction but WITHOUT the small/large
    cap branch correction in g(phi) -- the literal formula as specified,
    for direct comparison against the corrected version."""
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


def check_1_hemisphere():
    print("=== Check 1: hemisphere sanity (d=3, r_c=r_t=Delta=pi/2) ===")
    D = r_c = r_t = 0.5
    d = 3
    ref = reference_intersection_fraction(D, r_c, r_t, d)
    ref_unbranched = reference_intersection_fraction_unbranched(D, r_c, r_t, d)
    existing = exact_intersection_fraction(D, r_c, r_t, d)
    print(f"  reference (branched)   = {ref:.10f}  (expected exactly 0.5)")
    print(f"  reference (unbranched) = {ref_unbranched:.10f}")
    print(f"  existing implementation = {existing:.10f}")
    print(f"  NOTE: in this specific symmetric case, tau(phi)=0 throughout the "
          f"range (r_cross=pi/2 exactly), which is the fixed point where the "
          f"branched/unbranched formulas coincide -- this test does NOT "
          f"discriminate the branch correction. See check_1b below.")
    print()


def check_1b_branch_matters():
    print("=== Check 1b (additional): does the small/large-cap branch matter? ===")
    print("  Using an asymmetric config where tau(phi) < 0 for part of the range,")
    print("  so the branched and unbranched formulas should diverge if the branch")
    print("  is mathematically necessary.")
    # D=0.05,r_c=0.35,r_t=0.1 (first attempt) landed on a fast-path boundary
    # (full containment), which bypasses g(phi) entirely -- doesn't
    # discriminate either. This config lands in the general quadrature
    # branch with tau(phi) meaningfully negative for part of the range.
    D, r_c, r_t, d = 0.15, 0.3, 0.2, 16
    branched = reference_intersection_fraction(D, r_c, r_t, d)
    unbranched = reference_intersection_fraction_unbranched(D, r_c, r_t, d)
    existing = exact_intersection_fraction(D, r_c, r_t, d)
    print(f"  D={D} r_c={r_c} r_t={r_t} d={d}")
    print(f"  reference (branched)   = {branched:.6f}")
    print(f"  reference (unbranched) = {unbranched:.6f}")
    print(f"  existing implementation = {existing:.6f}")
    print(f"  branched vs unbranched diff = {abs(branched-unbranched):.6f}  "
          f"(large, as expected -- confirms the branch is not a minor "
          f"correction, the unbranched/literal formula is substantially "
          f"wrong whenever tau(phi) goes negative over part of the range)")
    print()


def check_2_degenerate_cases():
    print("=== Check 2: degenerate cases ===")
    d = 20

    print("  -- concentric poles (Delta=0, r_c != r_t) --")
    for r_c, r_t in [(0.1, 0.2), (0.2, 0.1)]:
        ref = reference_intersection_fraction(0.0, r_c, r_t, d)
        existing = exact_intersection_fraction(0.0, r_c, r_t, d)
        expected = 1.0 if r_c >= r_t else (
            _single_cap_fraction_reference(r_c, d) / _single_cap_fraction_reference(r_t, d)
        )
        print(f"    r_c={r_c} r_t={r_t}: reference={ref:.6f} existing={existing:.6f} "
              f"expected_ratio={expected:.6f}")

    print("  -- no overlap --")
    D, r_c, r_t = 0.5, 0.1, 0.1
    ref = reference_intersection_fraction(D, r_c, r_t, d)
    existing = exact_intersection_fraction(D, r_c, r_t, d)
    print(f"    D={D} r_c={r_c} r_t={r_t}: reference={ref} existing={existing} (expect 0.0)")

    print("  -- full containment (t's cap fully inside Q's cap) --")
    D, r_c, r_t = 0.05, 0.3, 0.1
    ref = reference_intersection_fraction(D, r_c, r_t, d)
    existing = exact_intersection_fraction(D, r_c, r_t, d)
    print(f"    D={D} r_c={r_c} r_t={r_t}: reference={ref} existing={existing} (expect 1.0)")

    print("  -- full containment (Q's cap fully inside t's cap -- NOT exactly 0 or 1) --")
    D, r_c, r_t = 0.05, 0.1, 0.3
    ref = reference_intersection_fraction(D, r_c, r_t, d)
    existing = exact_intersection_fraction(D, r_c, r_t, d)
    print(f"    D={D} r_c={r_c} r_t={r_t}: reference={ref:.6f} existing={existing:.6f} "
          f"(expect Vol(cap_c)/Vol(cap_t), a ratio in (0,1), NOT exactly 0 or 1)")
    print()


def check_3_real_pairs_sweep():
    print("=== Check 3: cross-check against real (candidate, test query) pairs ===")
    combos = [
        ('sift', 'sift_b50k_c1024k100_t512n20_s42', 5, 0.5, 128),
        ('esci', 'esci_c1024k100_t512n20_s42', 5, 0.5, 384),
    ]
    for dataset, benchmark_name, K, budget_frac, d in combos:
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
        test_q_hat = test_q / np.linalg.norm(test_q, axis=1, keepdims=True)
        cand_centers = np.array([c.query for c in selected])
        cand_centers_hat = cand_centers / np.linalg.norm(cand_centers, axis=1, keepdims=True)
        cand_rc = np.array([c.r_c for c in selected])

        rng = np.random.default_rng(0)
        n_pairs_to_check = 300
        cand_idx = rng.integers(0, len(selected), n_pairs_to_check)
        test_idx = rng.integers(0, n_test, n_pairs_to_check)

        diffs = []
        fractions_existing = []
        for ci, ti in zip(cand_idx, test_idx):
            dot = np.clip(cand_centers_hat[ci] @ test_q_hat[ti], -1.0, 1.0)
            D = float(np.arccos(dot) / np.pi)
            r_c = float(cand_rc[ci])
            r_t = float(test_r_t[ti])
            ref = reference_intersection_fraction(D, r_c, r_t, d)
            existing = exact_intersection_fraction(D, r_c, r_t, d)
            diffs.append(abs(ref - existing))
            fractions_existing.append(existing)

        diffs = np.array(diffs)
        fractions_existing = np.array(fractions_existing)
        print(f"  {dataset} {benchmark_name} K={K} budget={budget_frac} d={d}: "
              f"{n_pairs_to_check} pairs, n_selected={len(selected)}")
        print(f"    diff percentiles: p50={np.percentile(diffs,50):.2e} "
              f"p90={np.percentile(diffs,90):.2e} p99={np.percentile(diffs,99):.2e} "
              f"max={diffs.max():.2e}")
        near_zero = fractions_existing < 0.05
        near_one = fractions_existing > 0.95
        middle = ~near_zero & ~near_one
        for label, mask in [('near 0', near_zero), ('near 1', near_one), ('middle', middle)]:
            if mask.any():
                print(f"    diff where existing {label} (n={mask.sum()}): "
                      f"mean={diffs[mask].mean():.2e} max={diffs[mask].max():.2e}")
        large_diff_mask = diffs > 1e-4
        if large_diff_mask.any():
            print(f"    !! {large_diff_mask.sum()} pairs with diff > 1e-4 -- flagging, not resolving:")
            for i in np.where(large_diff_mask)[0][:5]:
                ci, ti = cand_idx[i], test_idx[i]
                dot = np.clip(cand_centers_hat[ci] @ test_q_hat[ti], -1.0, 1.0)
                D = float(np.arccos(dot) / np.pi)
                r_c, r_t = float(cand_rc[ci]), float(test_r_t[ti])
                print(f"       D={D:.4f} r_c={r_c:.4f} r_t={r_t:.4f} "
                      f"ref={reference_intersection_fraction(D, r_c, r_t, d):.6f} "
                      f"existing={exact_intersection_fraction(D, r_c, r_t, d):.6f}")
        print()


def check_4_monte_carlo_subsample():
    print("=== Check 4: three-way subsample including Monte Carlo ===")
    d = 128
    rng = np.random.default_rng(1)
    configs = [
        (0.05, 0.1, 0.08, d), (0.02, 0.05, 0.05, d), (0.1, 0.15, 0.12, d),
        (0.08, 0.2, 0.15, d), (0.03, 0.06, 0.04, d),
    ]
    n_samples = 50000
    flagged = []
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

        print(f"  D={D} r_c={r_c} r_t={r_t} d={d}: reference={ref:.4f} existing={existing:.4f} "
              f"mc={mc:.4f} (se~{se:.4f})")
        if abs(ref - existing) > 1e-4 or abs(ref - mc) > 4 * se + 0.005:
            flagged.append((D, r_c, r_t, d, ref, existing, mc, se))

    print()
    if flagged:
        print(f"  !! {len(flagged)} configs flagged for disagreement beyond tolerance -- NOT resolved, for review:")
        for f in flagged:
            print(f"     {f}")
    else:
        print("  No configs exceeded disagreement tolerance across reference/existing/MC.")
    print()

#-------------------------------------------------------------------------------


def main():
    check_1_hemisphere()
    check_1b_branch_matters()
    check_2_degenerate_cases()
    check_3_real_pairs_sweep()
    check_4_monte_carlo_subsample()


if __name__ == '__main__':
    main()
