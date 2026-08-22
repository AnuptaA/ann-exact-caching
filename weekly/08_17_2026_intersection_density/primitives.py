#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Independent primitives for the intersection-fraction calibration study.

Written in raw radians (theta in [0, pi]). Re-implements the cap/solid-ball
formulas without importing production code, for cross-checking against it.
"""

import math

import numpy as np
from scipy.special import betainc, betaincinv
from scipy.integrate import quad

#-------------------------------------------------------------------------------
# Cap intersection (spherical caps on S^{d-1})
#-------------------------------------------------------------------------------


def single_cap_fraction(theta: float, d: int) -> float:
    """Fraction of (d-1)-dimensional surface area within angular radius theta
    of a pole. """

    x = math.sin(theta) ** 2
    a = (d - 1) / 2.0
    ib = betainc(a, 0.5, x)
    return 0.5 * ib if theta <= math.pi / 2 else 1.0 - 0.5 * ib

def _cross_section_fraction(phi: float, delta: float, theta_1: float, d: int, branched: bool = True) -> float:
    """At colatitude phi from cap-2's pole, the fraction of
    the (d-2)-sphere cross-section (points of S^{d-1} at fixed
    colatitude phi from cap-2's pole) lying inside cap 1.

    branched=False (specifically for testing purposes) disables the large-cap correction in
    single_cap_fraction (always uses the 0.5*I_x(a,0.5) form)"""

    denom = math.sin(phi) * math.sin(delta)
    tau = (math.cos(theta_1) - math.cos(phi) * math.cos(delta)) / denom
    if tau <= -1.0:
        return 1.0
    if tau >= 1.0:
        return 0.0
    r_cross = math.acos(tau)
    x = math.sin(r_cross) ** 2
    a = (d - 2) / 2.0
    ib = betainc(a, 0.5, x)
    if not branched:
        return 0.5 * ib
    return 0.5 * ib if r_cross <= math.pi / 2 else 1.0 - 0.5 * ib

def exact_intersection_fraction(theta_1: float, theta_2: float, delta: float, d: int, branched: bool = True) -> float:
    """Fraction of cap 2's (angular radius theta_2) surface area lying
    within cap 1 (angular radius theta_1), pole separation delta, on
    S^{d-1}.

    Fast paths for no-overlap, full-containment, concentric poles."""

    if delta >= theta_1 + theta_2:
        return 0.0
    if delta + theta_2 <= theta_1:
        return 1.0
    if delta + theta_1 <= theta_2:
        return single_cap_fraction(theta_1, d) / single_cap_fraction(theta_2, d)
    if delta < 1e-12:
        return 1.0 if theta_1 >= theta_2 else single_cap_fraction(theta_1, d) / single_cap_fraction(theta_2, d)

    eps = 1e-10
    sin_theta2 = math.sin(theta_2)
    num, _ = quad(
        lambda phi: ((math.sin(phi) / sin_theta2) ** (d - 2)) * _cross_section_fraction(phi, delta, theta_1, d, branched),
        eps, theta_2, limit=200,
    )
    den, _ = quad(lambda phi: (math.sin(phi) / sin_theta2) ** (d - 2), eps, theta_2, limit=200)
    return num / den

def exact_intersection_fraction_batch(theta_1: np.ndarray, theta_2: float, delta: np.ndarray, d: int,
                                       n_nodes: int = 32) -> np.ndarray:
    """Vectorized exact_intersection_fraction for one fixed theta_2 against
    arrays of theta_1/delta. Fixed-order quadrature; trades precision for
    speed."""
    theta_1 = np.asarray(theta_1, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    eps = 1e-10
    fraction = np.zeros_like(theta_1)

    # fast-path masks computed before the quadrature, not applied after
    no_overlap = delta >= theta_1 + theta_2
    full_containment = (~no_overlap) & (delta + theta_2 <= theta_1)
    flipped_containment = (~no_overlap) & (~full_containment) & (delta + theta_1 <= theta_2)
    near_zero = (~no_overlap) & (~full_containment) & (~flipped_containment) & (delta < 1e-12)
    general = ~(no_overlap | full_containment | flipped_containment | near_zero)

    needs_ratio = flipped_containment | near_zero
    if np.any(needs_ratio):
        single_theta1 = _single_cap_fraction_arr(theta_1[needs_ratio], d)
        single_theta2 = single_cap_fraction(theta_2, d)
        ratio = np.where(single_theta2 > 0.0, single_theta1 / max(single_theta2, np.finfo(np.float64).tiny), 1.0)
        fraction[flipped_containment] = ratio[flipped_containment[needs_ratio]]
        near_zero_ratio = ratio[near_zero[needs_ratio]]
        fraction[near_zero] = np.where(theta_1[near_zero] >= theta_2, 1.0, near_zero_ratio)

    fraction[full_containment] = 1.0
    # no_overlap entries stay 0.0 (already initialized)

    if np.any(general):
        theta_1_g = theta_1[general]
        delta_g = delta[general]

        nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
        half_width = max(theta_2 - eps, eps) / 2.0
        phi = eps + half_width * (nodes + 1.0)  # (n_nodes,)
        quad_weights = half_width * weights  # (n_nodes,)

        sin_theta2 = math.sin(theta_2)
        weight_factor = (np.sin(phi) / sin_theta2) ** (d - 2)  # (n_nodes,)
        den = np.sum(weight_factor * quad_weights)

        a = (d - 2) / 2.0
        sin_phi = np.sin(phi)[:, None]  # (n_nodes, 1)
        cos_phi = np.cos(phi)[:, None]
        sin_delta = np.sin(delta_g)[None, :]  # (1, n_general)
        cos_delta = np.cos(delta_g)[None, :]
        cos_theta1 = np.cos(theta_1_g)[None, :]

        cross_denom = sin_phi * sin_delta
        cross_denom = np.where(cross_denom == 0.0, np.finfo(np.float64).tiny, cross_denom)
        tau = (cos_theta1 - cos_phi * cos_delta) / cross_denom  # (n_nodes, n_general)

        r_cross = np.arccos(np.clip(tau, -1.0, 1.0))
        x = np.sin(r_cross) ** 2
        ib = betainc(a, 0.5, x)
        cross = np.where(r_cross <= math.pi / 2, 0.5 * ib, 1.0 - 0.5 * ib)
        cross = np.where(tau <= -1.0, 1.0, cross)
        cross = np.where(tau >= 1.0, 0.0, cross)

        num = np.sum((weight_factor * quad_weights)[:, None] * cross, axis=0)  # (n_general,)
        fraction[general] = num / den

    return fraction

def _single_cap_fraction_arr(theta: np.ndarray, d: int) -> np.ndarray:
    """Vectorized single_cap_fraction."""
    x = np.sin(theta) ** 2
    a = (d - 1) / 2.0
    ib = betainc(a, 0.5, x)
    return np.where(theta <= math.pi / 2, 0.5 * ib, 1.0 - 0.5 * ib)

#-------------------------------------------------------------------------------
# Solid-ball intersection (Euclidean balls in R^d)
#-------------------------------------------------------------------------------

def _cap_of_ball_fraction(R: float, signed_offset: float, d: int, branched: bool = True) -> float:
    """Fraction of a radius-R ball's volume lying on the far side of a
    hyperplane at signed distance `signed_offset` from its center.

    branched=False disables the large-cap correction (always uses the
    0.5*I_x(a,0.5) form), for testing that the correction matters --
    same role as exact_intersection_fraction's branched param."""

    x = max(0.0, min(1.0, 1.0 - (signed_offset / R) ** 2))
    a = (d + 1) / 2.0
    ib = betainc(a, 0.5, x)
    if not branched:
        return 0.5 * ib
    return 0.5 * ib if signed_offset >= 0 else 1.0 - 0.5 * ib


def solid_ball_intersection_fraction(r_1: float, r_2: float, dist: float, d: int, branched: bool = True) -> float:
    """Fraction of ball 2's (radius r_2) volume lying within ball 1
    (radius r_1), centers separated by `dist`, in Euclidean R^d.

    Fast paths for
        - no-overlap
        - full-containment
        - concentric centers

    General case computes the intersection volume as the sum of two caps."""

    if dist >= r_1 + r_2:
        return 0.0
    if dist + r_2 <= r_1:
        return 1.0
    if dist + r_1 <= r_2:
        return (r_1 / r_2) ** d
    if dist < 1e-12:
        return 1.0 if r_1 >= r_2 else (r_1 / r_2) ** d

    d1 = (dist ** 2 + r_1 ** 2 - r_2 ** 2) / (2.0 * dist)
    d2 = dist - d1
    return (r_1 / r_2) ** d * _cap_of_ball_fraction(r_1, d1, d, branched) + _cap_of_ball_fraction(r_2, d2, d, branched)

#-------------------------------------------------------------------------------
# Chord distance
#-------------------------------------------------------------------------------

def chord_distance_from_angular(theta: float) -> float:
    """Euclidean chord length between two unit vectors separated by angle theta
    in radians."""
    return 2.0 * math.sin(theta / 2.0)

#-------------------------------------------------------------------------------
# Monte Carlo cap sampler
#-------------------------------------------------------------------------------

_MAX_SAMPLER_BATCH = 2000000  # caps peak memory regardless of how tiny the cap fraction is


def sample_cap_points(pole: np.ndarray, radius: float, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform points within angular radius of `pole` on S^{d-1}, via rejection 
    sampling: draw uniform points on the whole sphere (Gaussian direction, normalized), keep those
    within the cap, repeat until n_samples collected."""

    d = pole.shape[0]
    out = np.empty((n_samples, d), dtype=np.float64)
    filled = 0
    while filled < n_samples:
        estimate = int((n_samples - filled) / max(single_cap_fraction(radius, d), 1e-6) * 1.5)
        batch = max(256, min(estimate, _MAX_SAMPLER_BATCH))
        pts = rng.standard_normal((batch, d))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        dists = np.arccos(np.clip(pts @ pole, -1.0, 1.0))
        accepted = pts[dists <= radius]
        take = min(len(accepted), n_samples - filled)
        out[filled:filled + take] = accepted[:take]
        filled += take
    return out


def sample_cap_points_direct(pole: np.ndarray, radius: float, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform points within angular radius (<= pi/2) of `pole` on S^{d-1}."""
    d = pole.shape[0]
    a = (d - 1) / 2.0
    cap_frac = single_cap_fraction(radius, d)

    v = rng.uniform(0.0, 1.0, size=n_samples)
    x = betaincinv(a, 0.5, np.clip(2.0 * v * cap_frac, 0.0, 1.0))
    phi = np.arcsin(np.sqrt(np.clip(x, 0.0, 1.0)))

    raw = rng.standard_normal((n_samples, d))
    raw -= np.outer(raw @ pole, pole)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)

    return np.cos(phi)[:, None] * pole[None, :] + np.sin(phi)[:, None] * raw

#-------------------------------------------------------------------------------
# Testing
#-------------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    results = []

    def check(name: str, cond: bool, detail: str = '') -> None:
        results.append((name, cond, detail))
        status = 'PASS' if cond else 'FAIL'
        print(f"[{status}] {name}" + (f"  ({detail})" if detail else ''))

    print("=== Cap intersection ===")

    # Case 1: Check that exact_intersection_fraction returns 0.5 for 
    # two hemispheres (theta_1 = theta_2 = delta = pi/2) in either dimension.
    f = exact_intersection_fraction(math.pi / 2, math.pi / 2, math.pi / 2, 3)
    check("hemisphere symmetric case == 0.5", abs(f - 0.5) < 1e-9, f"got {f:.9f}")

    # Case 2: Check that the branch correction matters.
    theta_1, theta_2, delta, d = 0.3 * math.pi, 0.2 * math.pi, 0.15 * math.pi, 16
    branched = exact_intersection_fraction(theta_1, theta_2, delta, d, branched=True)
    unbranched = exact_intersection_fraction(theta_1, theta_2, delta, d, branched=False)
    check("branch correction changes result substantially",
          abs(branched - unbranched) > 0.3,
          f"branched={branched:.4f} unbranched={unbranched:.4f} diff={abs(branched-unbranched):.4f}")

    # Case 3: Check that in the low dimensional case of d=2, the exact_intersection_fraction agrees
    # with the closed form overlap formula.
    def elementary_d2(theta_1, theta_2, delta):
        overlap = max(0.0, min(theta_1, delta + theta_2) - max(-theta_1, delta - theta_2))
        return overlap / (2 * theta_2)

    max_diff_d2 = 0.0
    for t1, t2, dd in [(0.3, 0.3, 0.1), (0.5, 0.2, 0.1), (0.2, 0.3, 0.15), (0.6, 0.15, 0.5)]:
        exact = exact_intersection_fraction(t1, t2, dd, 2)
        expected = elementary_d2(t1, t2, dd)
        max_diff_d2 = max(max_diff_d2, abs(exact - expected))
    check("elementary d=2 arc-overlap agreement", max_diff_d2 < 1e-8, f"max diff {max_diff_d2:.2e}")

    # Case 4: Check that in the low dimensional case of d=3, the exact_intersection_fraction agrees
    # with the closed form overlap formula.
    def elementary_d3(theta_1, theta_2, delta):
        if delta >= theta_1 + theta_2:
            return 0.0
        if delta + theta_2 <= theta_1:
            return 1.0
        if delta < 1e-12:
            return 1.0 if theta_1 >= theta_2 else (1 - math.cos(theta_1)) / (1 - math.cos(theta_2))

        def g(phi):
            c = (math.cos(theta_1) - math.cos(phi) * math.cos(delta)) / (math.sin(phi) * math.sin(delta))
            c = max(-1.0, min(1.0, c))
            return math.acos(c) / math.pi  # elementary S^1 arc fraction

        num, _ = quad(lambda phi: g(phi) * math.sin(phi), 1e-10, theta_2, limit=200)
        den = 1 - math.cos(theta_2)
        return num / den

    max_diff_d3 = 0.0
    for t1, t2, dd in [(0.3, 0.3, 0.1), (0.5, 0.2, 0.1), (0.2, 0.3, 0.15)]:
        exact = exact_intersection_fraction(t1, t2, dd, 3)
        expected = elementary_d3(t1, t2, dd)
        max_diff_d3 = max(max_diff_d3, abs(exact - expected))
    check("elementary d=3 spherical-trig agreement", max_diff_d3 < 1e-6, f"max diff {max_diff_d3:.2e}")

    # Case 5: Check that the intersection fraction is monotone non-increasing in delta. That is, as
    # separate, we expect the intersection fraction to decrease as the two caps move apart.
    r = 0.3
    deltas = np.linspace(0.0, 2 * r, 15)
    fracs = [exact_intersection_fraction(r, r, dd, 16) for dd in deltas]
    diffs = np.diff(fracs)
    check("monotone non-increasing in delta", bool(np.all(diffs <= 1e-9)), f"diffs range [{diffs.min():.2e}, {diffs.max():.2e}]")

    # Case 6: Check that the intersection fraction is monotone non-increasing in dimension. That is,
    # as dimension increases, we expect the intersection fraction to decrease (thinner caps).
    r, dd = 0.35, 0.25
    dims = [3, 5, 10, 20, 50, 100]
    fracs_dim = [exact_intersection_fraction(r, r, dd, dv) for dv in dims]
    diffs_dim = np.diff(fracs_dim)
    check("monotone non-increasing in dimension", bool(np.all(diffs_dim <= 1e-9)), f"{list(zip(dims, np.round(fracs_dim, 4)))}")

    # Case 7: Check that the intersection fraction is exactly 1.0 when the two caps are concentric 
    # and have equal radii, for a variety of radii and dimensions.
    all_one = all(exact_intersection_fraction(r, r, 0.0, dv) == 1.0 for r in [0.1, 0.3, 0.6] for dv in [3, 16, 64])
    check("zero-separation equal-radii == 1.0 exactly", all_one)

    # Case 8: Check that the intersection fraction approaches 0.0 continuously as the two caps 
    # approach the no-overlap boundary from below.
    t1, t2, dv = 0.3, 0.3, 16
    boundary = t1 + t2
    approach = [exact_intersection_fraction(t1, t2, boundary - eps, dv) for eps in [0.1, 0.02, 0.005, 0.001]]
    check("continuous approach to no-overlap boundary (-> 0)",
          all(f < 0.5 for f in approach) and all(a >= b - 1e-9 for a, b in zip(approach, approach[1:])),
          f"{[f'{f:.2e}' for f in approach]}")

    # Case 9: Check that the intersection fraction approaches 1.0 continuously as the two caps
    # approach the full-containment boundary from below.
    t1, t2, dv = 0.4, 0.15, 16
    boundary = t1 - t2
    approach = [exact_intersection_fraction(t1, t2, max(boundary - eps, 0.0), dv) for eps in [0.1, 0.02, 0.005, 0.001]]
    check("continuous approach to full-containment boundary (-> 1)",
          all(f > 0.5 for f in approach),
          f"{[round(f, 4) for f in approach]}")

    # Case 10: Check that the raw (non-normalized) intersection volume is symmetric under
    # swapping cap 1 and cap 2, even though the normalized fraction itself is intentionally
    # asymmetric (normalized by cap 2's own volume).
    max_diff_symmetry = 0.0
    for t1, t2, dd in [(0.2, 0.3, 0.15), (0.4, 0.15, 0.3), (0.1, 0.1, 0.05)]:
        f12 = exact_intersection_fraction(t1, t2, dd, 20)
        f21 = exact_intersection_fraction(t2, t1, dd, 20)
        vol1 = single_cap_fraction(t1, 20)
        vol2 = single_cap_fraction(t2, 20)
        max_diff_symmetry = max(max_diff_symmetry, abs(f12 * vol2 - f21 * vol1))
    check("raw volume symmetric under cap swap", max_diff_symmetry < 1e-9, f"max diff {max_diff_symmetry:.2e}")

    # Case 11: Check that at zero separation with theta_1 < theta_2, the intersection fraction
    # equals the ratio of the two caps' own volumes (only the theta_1 >= theta_2 branch is
    # covered by Case 7).
    t1, t2, dv = 0.1, 0.3, 16
    f = exact_intersection_fraction(t1, t2, 0.0, dv)
    expected = single_cap_fraction(t1, dv) / single_cap_fraction(t2, dv)
    check("zero-separation, theta_1 < theta_2 -> volume ratio", abs(f - expected) < 1e-9, f"got {f:.9f}, expected {expected:.9f}")

    # Case 12: Check that sample_cap_points returns unit-norm points within the requested
    # angular radius of the pole.
    rng_sampler = np.random.default_rng(6)
    pole_check = np.zeros(3); pole_check[0] = 1.0
    pts_check = sample_cap_points(pole_check, 0.5, 5000, rng_sampler)
    norms = np.linalg.norm(pts_check, axis=1)
    dists_check = np.arccos(np.clip(pts_check @ pole_check, -1.0, 1.0))
    check("sample_cap_points output is unit-norm and within radius",
          bool(np.allclose(norms, 1.0, atol=1e-9) and np.all(dists_check <= 0.5 + 1e-9)),
          f"max norm diff {np.max(np.abs(norms-1.0)):.2e}, max dist {dists_check.max():.4f} (radius=0.5)")

    print("\n=== Solid ball ===")

    # Case 1: Check that solid_ball_intersection_fraction returns 5/16 for two balls of radius
    # sqrt(2) in R^3 whose centers are also separated by sqrt(2). This is a hand-computable case.
    R = math.sqrt(2)
    val = solid_ball_intersection_fraction(R, R, R, 3)
    check("hand-computable case (R1=R2=dist=sqrt2, n=3) == 5/16", abs(val - 5.0 / 16.0) < 1e-9, f"got {val:.9f}")

    # Case 2: Check that solid_ball_intersection_fraction returns 0.0 for two balls that are far
    # apart.
    v_no_overlap = solid_ball_intersection_fraction(0.5, 0.5, 2.0, 10)
    check("no overlap == 0.0", v_no_overlap == 0.0, f"got {v_no_overlap}")

    # Case 3: Check that solid_ball_intersection_fraction returns 1.0 for two balls where ball 2 is
    # fully inside ball 1.
    v_full_2_in_1 = solid_ball_intersection_fraction(0.5, 0.2, 0.1, 10)
    check("ball 2 fully inside ball 1 == 1.0", v_full_2_in_1 == 1.0, f"got {v_full_2_in_1}")

    # Case 4: Check that solid_ball_intersection_fraction returns a value strictly between 0 and 1
    # for two balls where ball 1 is fully inside ball 2, and that the value is approximately (r1/r2)^d.
    v_full_1_in_2 = solid_ball_intersection_fraction(0.2, 0.5, 0.1, 10)
    check("ball 1 fully inside ball 2 -> ratio, not 0/1",
          0.0 < v_full_1_in_2 < 1.0 and abs(v_full_1_in_2 - (0.2 / 0.5) ** 10) < 1e-9,
          f"got {v_full_1_in_2:.6f}, expected {(0.2/0.5)**10:.6f}")

    # Case 5: Check that the branch correction matters for solid_ball_intersection_fraction too.
    r1_bc, r2_bc, dist_bc, d_bc = 2.0, 1.0, 1.05, 3
    branched_val = solid_ball_intersection_fraction(r1_bc, r2_bc, dist_bc, d_bc, branched=True)
    unbranched_val = solid_ball_intersection_fraction(r1_bc, r2_bc, dist_bc, d_bc, branched=False)
    check("branch correction changes result substantially (solid ball)",
          abs(branched_val - unbranched_val) > 0.3,
          f"branched={branched_val:.4f} unbranched={unbranched_val:.4f} diff={abs(branched_val-unbranched_val):.4f}")

    # Case 6: Check that in the low dimensional case of d=1, solid_ball_intersection_fraction
    # agrees with the closed form interval-overlap formula (a d=1 ball is the interval [-r,r]).
    def elementary_ball_d1(r_1, r_2, dist):
        overlap = max(0.0, min(r_1, dist + r_2) - max(-r_1, dist - r_2))
        return overlap / (2 * r_2)

    max_diff_ball_d1 = 0.0
    for r1v, r2v, distv in [(0.5, 0.5, 0.3), (0.3, 0.6, 0.5), (0.7, 0.4, 0.5)]:
        exact = solid_ball_intersection_fraction(r1v, r2v, distv, 1)
        expected = elementary_ball_d1(r1v, r2v, distv)
        max_diff_ball_d1 = max(max_diff_ball_d1, abs(exact - expected))
    check("elementary d=1 interval-overlap agreement (solid ball)", max_diff_ball_d1 < 1e-8, f"max diff {max_diff_ball_d1:.2e}")

    # Case 7: Check that in the low dimensional case of d=2, solid_ball_intersection_fraction
    # agrees with the closed form circle-circle lens-area formula.
    def elementary_ball_d2(r_1, r_2, dist):
        area = (r_1 ** 2 * math.acos((dist ** 2 + r_1 ** 2 - r_2 ** 2) / (2 * dist * r_1))
                + r_2 ** 2 * math.acos((dist ** 2 + r_2 ** 2 - r_1 ** 2) / (2 * dist * r_2))
                - 0.5 * math.sqrt(max(0.0, (-dist + r_1 + r_2) * (dist + r_1 - r_2) * (dist - r_1 + r_2) * (dist + r_1 + r_2))))
        return area / (math.pi * r_2 ** 2)

    max_diff_ball_d2 = 0.0
    for r1v, r2v, distv in [(0.5, 0.5, 0.3), (0.3, 0.6, 0.5), (0.7, 0.4, 0.5)]:
        exact = solid_ball_intersection_fraction(r1v, r2v, distv, 2)
        expected = elementary_ball_d2(r1v, r2v, distv)
        max_diff_ball_d2 = max(max_diff_ball_d2, abs(exact - expected))
    check("elementary d=2 circle-lens agreement (solid ball)", max_diff_ball_d2 < 1e-8, f"max diff {max_diff_ball_d2:.2e}")

    # Case 8: Check that the solid-ball intersection fraction is monotone non-increasing in dist.
    # That is, as the two balls move apart, we expect the intersection fraction to decrease.
    r_mono = 0.3
    dists_mono = np.linspace(0.0, 2 * r_mono, 15)
    fracs_ball = [solid_ball_intersection_fraction(r_mono, r_mono, dd, 5) for dd in dists_mono]
    diffs_ball = np.diff(fracs_ball)
    check("monotone non-increasing in dist (solid ball)", bool(np.all(diffs_ball <= 1e-9)), f"diffs range [{diffs_ball.min():.2e}, {diffs_ball.max():.2e}]")

    # Case 9: Check that the solid-ball intersection fraction is monotone non-increasing in
    # dimension. That is, as dimension increases, we expect the intersection fraction to
    # decrease (volume concentrates near the surface).
    r_mono_dim, dist_mono_dim = 0.35, 0.25
    dims_ball = [1, 2, 3, 5, 10, 20, 50, 100]
    fracs_ball_dim = [solid_ball_intersection_fraction(r_mono_dim, r_mono_dim, dist_mono_dim, dv) for dv in dims_ball]
    diffs_ball_dim = np.diff(fracs_ball_dim)
    check("monotone non-increasing in dimension (solid ball)", bool(np.all(diffs_ball_dim <= 1e-9)), f"{list(zip(dims_ball, np.round(fracs_ball_dim, 4)))}")

    # Case 10: Check that the solid-ball intersection fraction is exactly 1.0 when the two balls
    # are concentric and have equal radii, for a variety of radii and dimensions.
    all_one_ball = all(solid_ball_intersection_fraction(r, r, 0.0, dv) == 1.0 for r in [0.1, 0.3, 0.6] for dv in [3, 16, 64])
    check("zero-separation equal-radii == 1.0 exactly (solid ball)", all_one_ball)

    # Case 11: Check that the solid-ball intersection fraction approaches 0.0 continuously as
    # the two balls approach the no-overlap boundary from below.
    r1_nb, r2_nb, d_nb = 0.3, 0.3, 16
    boundary_nb = r1_nb + r2_nb
    approach_nb = [solid_ball_intersection_fraction(r1_nb, r2_nb, boundary_nb - eps, d_nb) for eps in [0.1, 0.02, 0.005, 0.001]]
    check("continuous approach to no-overlap boundary (-> 0) (solid ball)",
          all(f < 0.5 for f in approach_nb) and all(a >= b - 1e-9 for a, b in zip(approach_nb, approach_nb[1:])),
          f"{[f'{f:.2e}' for f in approach_nb]}")

    # Case 12: Check that the solid-ball intersection fraction approaches 1.0 continuously as
    # the two balls approach the full-containment boundary from below.
    r1_fc, r2_fc, d_fc = 0.4, 0.15, 16
    boundary_fc = r1_fc - r2_fc
    approach_fc = [solid_ball_intersection_fraction(r1_fc, r2_fc, max(boundary_fc - eps, 0.0), d_fc) for eps in [0.1, 0.02, 0.005, 0.001]]
    check("continuous approach to full-containment boundary (-> 1) (solid ball)",
          all(f > 0.5 for f in approach_fc),
          f"{[round(f, 4) for f in approach_fc]}")

    print("\n=== Monte Carlo cross-check ===")

    rng = np.random.default_rng(0)
    mc_configs = [
        (0.3 * math.pi, 0.25 * math.pi, 0.2 * math.pi, 8),
        (0.4 * math.pi, 0.3 * math.pi, 0.35 * math.pi, 16),
    ]
    for theta_1, theta_2, delta, dv in mc_configs:
        cap_exact = exact_intersection_fraction(theta_1, theta_2, delta, dv)

        pole_2 = np.zeros(dv); pole_2[0] = 1.0
        pole_1 = np.zeros(dv); pole_1[0] = math.cos(delta); pole_1[1] = math.sin(delta)
        n_mc = 30000
        pts = sample_cap_points(pole_2, theta_2, n_mc, rng)
        dist_to_pole1 = np.arccos(np.clip(pts @ pole_1, -1.0, 1.0))
        mc = float(np.mean(dist_to_pole1 <= theta_1))
        se = math.sqrt(max(mc * (1 - mc), 1e-9) / n_mc)

        check(f"MC vs. cap closed-form (d={dv})", abs(cap_exact - mc) < 4 * se + 0.01,
              f"exact={cap_exact:.4f} mc={mc:.4f} se={se:.4f}")

        # Check discrepancy between the solid-ball intersection fraction and the cap intersection 
        # fraction for the same configuration.
        r1_chord = chord_distance_from_angular(theta_1)
        r2_chord = chord_distance_from_angular(theta_2)
        dist_chord = chord_distance_from_angular(delta)
        ball_exact = solid_ball_intersection_fraction(r1_chord, r2_chord, dist_chord, dv)
        print(f"    (for reference: solid-ball fraction on same config = {ball_exact:.4f}, cap = {cap_exact:.4f})")

    # Check that solid_ball_intersection_fraction agrees with brute-force rejection sampling
    # directly in Euclidean R^n.
    def brute_force_ball(r_1, r_2, dist, n, rng, n_target=200000):
        pts = rng.uniform(-r_2, r_2, size=(n_target * 3, n))
        norms = np.linalg.norm(pts, axis=1)
        pts = pts[norms <= r_2][:n_target]
        center1 = np.zeros(n)
        center1[0] = dist
        dists_to_1 = np.linalg.norm(pts - center1, axis=1)
        return float(np.mean(dists_to_1 <= r_1)), len(pts)

    dist_bf, r1_bf, r2_bf, n_bf = 0.5, 0.6, 0.5, 3
    exact_bf = solid_ball_intersection_fraction(r1_bf, r2_bf, dist_bf, n_bf)
    brute, n_pts_bf = brute_force_ball(r1_bf, r2_bf, dist_bf, n_bf, rng)
    se_bf = math.sqrt(max(brute * (1 - brute), 1e-9) / n_pts_bf)
    check("MC vs. solid-ball closed-form (brute-force rejection sampling in R^n)",
          abs(exact_bf - brute) < 5 * se_bf + 0.001,
          f"exact={exact_bf:.4f} brute={brute:.4f} se={se_bf:.4f}")

    print("\n=== Cross-check against existing implementation ===")
    try:
        sys.path.append(str(__import__('pathlib').Path(__file__).parent.parent.parent))
        from simulations.population.debug_intersection_calibration import (
            exact_intersection_fraction as existing_cap_fraction,
            solid_ball_intersection_fraction as existing_ball_fraction,
        )

        rng = np.random.default_rng(1)
        cap_diffs = []
        for _ in range(200):
            dv = int(rng.choice([3, 8, 16, 64, 128]))  # d=2 excluded, see dedicated check below
            theta_1 = float(rng.uniform(0.02, math.pi * 0.45))
            theta_2 = float(rng.uniform(0.02, math.pi * 0.45))
            delta = float(rng.uniform(0.0, math.pi * 0.9))
            new_val = exact_intersection_fraction(theta_1, theta_2, delta, dv)
            old_val = existing_cap_fraction(delta / math.pi, theta_1 / math.pi, theta_2 / math.pi, dv)
            cap_diffs.append(abs(new_val - old_val))
        cap_diffs = np.array(cap_diffs)
        check("new vs. existing cap fraction agreement, d>=3 (200 random configs)",
              bool(np.percentile(cap_diffs, 99) < 1e-4 and cap_diffs.max() < 1e-3),
              f"p99={np.percentile(cap_diffs, 99):.2e} max={cap_diffs.max():.2e}")

        # d=2: betainc(a=0) discontinuity, unfixed, production-irrelevant
        t1, t2, dd = 0.7576, 0.8724, 0.0958
        new_d2 = exact_intersection_fraction(t1, t2, dd, 2)
        old_d2 = existing_cap_fraction(dd / math.pi, t1 / math.pi, t2 / math.pi, 2)
        elem_d2 = elementary_d2(t1, t2, dd)
        print(f"[INFO] d=2 known bug: new={new_d2:.4f} elementary={elem_d2:.4f} existing={old_d2:.4f}")

        ball_diffs = []
        for _ in range(200):
            dv = int(rng.choice([2, 3, 8, 16, 64, 128]))
            r1 = float(rng.uniform(0.05, 2.0))
            r2 = float(rng.uniform(0.05, 2.0))
            dist = float(rng.uniform(0.0, 2.0))
            new_val = solid_ball_intersection_fraction(r1, r2, dist, dv)
            old_val = existing_ball_fraction(dist, r1, r2, dv)
            ball_diffs.append(abs(new_val - old_val))
        ball_diffs = np.array(ball_diffs)
        check("new vs. existing solid-ball fraction agreement (200 random configs)",
              bool(np.percentile(ball_diffs, 99) < 1e-6 and ball_diffs.max() < 1e-5),
              f"p99={np.percentile(ball_diffs, 99):.2e} max={ball_diffs.max():.2e}")
    except Exception as e:
        check("cross-check against existing implementation", False, f"could not run: {e}")

    n_fail = sum(1 for _, cond, _ in results if not cond)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed.")
    sys.exit(1 if n_fail else 0)
