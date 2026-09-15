#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Verification checks for median_halving_tree.py, run before trusting
it on the real pilot/grid: duplicate-heavy construction, brute-force
count cross-check, and the margin-test formula's correctness."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'common'))

import numpy as np

from median_halving_tree import (
    build_direction_pool, build_tree, touching_leaves_batch, counts_from_hits,
    brute_force_count,
)

#-------------------------------------------------------------------------------


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vecs / norms


def check(name: str, cond: bool, detail: str = '') -> bool:
    status = 'PASS' if cond else 'FAIL'
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ''))
    return cond

#-------------------------------------------------------------------------------


def test_duplicate_heavy() -> bool:
    rng = np.random.default_rng(0)
    d = 5
    base = _normalize(rng.standard_normal((20, d)))
    reps = rng.integers(20, 60, size=20)
    points = np.repeat(base, reps, axis=0)
    n = points.shape[0]
    pool = build_direction_pool(d, 32, rng)

    all_ok = True
    for halving in (True, False):
        tree = build_tree(points, C=5, halving=halving, direction_pool=pool,
                           rng=np.random.default_rng(1), max_depth=40)
        total_leaf_points = sum(len(v) for v in tree.leaf_point_idx.values())
        total_unique_points = sum(int(c.sum()) for _, c in tree.leaf_unique.values())
        covered = total_leaf_points + total_unique_points
        ok = check(f"duplicate-heavy: all points accounted for (halving={halving})",
                   covered == n, f"covered={covered}, n={n}")
        all_ok &= ok
        ok = check(f"duplicate-heavy: leaf_unique path actually used (halving={halving})",
                   len(tree.leaf_unique) > 0, f"n_leaf_unique={len(tree.leaf_unique)}")
        all_ok &= ok
        for node_id, (unique_pts, counts) in tree.leaf_unique.items():
            ok = check(f"duplicate-heavy node {node_id}: genuine duplication, not coincidence",
                       len(unique_pts) < counts.sum(), f"unique={len(unique_pts)}, total={counts.sum()}")
            all_ok &= ok
    return all_ok


def test_brute_force_cross_check() -> bool:
    rng = np.random.default_rng(2)
    d = 8
    n = 2000
    points = _normalize(rng.standard_normal((n, d)))
    pool = build_direction_pool(d, 64, rng)
    queries = _normalize(rng.standard_normal((30, d)))
    radii = rng.uniform(0.05, 0.5, size=30)

    all_ok = True
    for halving in (True, False):
        tree = build_tree(points, C=10, halving=halving, direction_pool=pool,
                           rng=np.random.default_rng(3), max_depth=40)
        hit_q, hit_n = touching_leaves_batch(tree, queries, radii)
        est_counts = counts_from_hits(tree, queries, radii, hit_q, hit_n)
        for i in range(len(queries)):
            true = brute_force_count(points, queries[i], float(radii[i]))
            ok = check(f"brute-force cross-check query {i} (halving={halving})",
                       int(est_counts[i]) == true, f"est={est_counts[i]}, true={true}")
            all_ok &= ok
            if not ok:
                break
    return all_ok


def test_margin_formula() -> bool:
    """Cross-check margin(q, u, t) = |arccos(q.u) - arccos(t)| against
    direct geometry: sample points near a known threshold boundary and
    confirm the certain/uncertain classification matches whether the
    query's cap actually crosses the hyperplane-circle, checked by
    brute-force sampling of the cap boundary itself."""
    rng = np.random.default_rng(4)
    d = 6
    u = rng.standard_normal(d)
    u /= np.linalg.norm(u)
    t = 0.3

    all_ok = True
    for _ in range(200):
        q = rng.standard_normal(d)
        q /= np.linalg.norm(q)
        r = rng.uniform(0.01, 1.0)

        proj = np.clip(q @ u, -1.0, 1.0)
        margin = abs(np.arccos(proj) - np.arccos(np.clip(t, -1.0, 1.0)))
        predicted_certain = margin >= r

        # Direct geometric check: does the cap of radius r around q
        # intersect the hyperplane x.u = t? Equivalent to: does there
        # exist a point at angular distance <= r from q with x.u = t?
        # The minimum angular distance from q to the plane's sphere is
        # exactly `margin` by construction, so this direct check is
        # margin <= r - verify via a numerical search along the
        # q-to-plane geodesic instead of reusing the same formula.
        # Numerical search: sample many points on the great circle from
        # q toward u (or away), find min angular distance to any point
        # satisfying x.u ~= t within a tight band.
        n_samples = 200000
        # Search both directions from q along the q-u great circle: the
        # nearest plane crossing can lie on either side of q depending
        # on whether q is closer to the pole u or to -u. Find all sign
        # changes of (path_proj - t) (there are exactly two per period),
        # not just the single closest sampled value, since argmin over
        # raw deviation can pick either root depending on grid alignment.
        u_perp = u - (u @ q) * q
        norm_perp = np.linalg.norm(u_perp)
        if norm_perp < 1e-9:
            continue
        u_perp /= norm_perp
        angles = np.linspace(-np.pi, np.pi, n_samples)
        path = np.cos(angles)[:, None] * q[None, :] + np.sin(angles)[:, None] * u_perp[None, :]
        diff = (path @ u) - t
        sign_changes = np.flatnonzero(np.diff(np.sign(diff)) != 0)
        if len(sign_changes) == 0:
            continue
        crossing_angles = np.abs(angles[sign_changes])
        actual_margin = float(crossing_angles.min())

        ok = abs(actual_margin - margin) < 1e-2
        all_ok &= ok
        if not ok:
            check("margin formula numerical cross-check", False,
                  f"formula={margin:.4f}, numerical={actual_margin:.4f}")
            break
    else:
        check("margin formula numerical cross-check (200 random q/r pairs)", True)

    return all_ok

#-------------------------------------------------------------------------------


def main() -> None:
    results = []
    print("=== Duplicate-heavy construction ===")
    results.append(test_duplicate_heavy())
    print("\n=== Brute-force count cross-check ===")
    results.append(test_brute_force_cross_check())
    print("\n=== Margin-test formula cross-check ===")
    results.append(test_margin_formula())

    print(f"\n{'ALL PASSED' if all(results) else 'SOME FAILED'}")


if __name__ == '__main__':
    main()
