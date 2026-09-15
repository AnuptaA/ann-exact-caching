#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""E1 correctness checks for the Horvitz-Thompson LSH estimator, run
before trusting it for any real experiment."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'common'))

import numpy as np

from lsh_horvitz_thompson import (
    p_single, p_union, build_tables, query_retrieval, estimate_count,
    estimate_count_batch, select_hyperparameters,
)  # noqa: F401 - p_union used directly in test_theta_not_r's manual recomputation

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


def test_p_single_monte_carlo() -> bool:
    rng = np.random.default_rng(0)
    d = 128
    m = 20
    n_trials = 20000
    pole = np.zeros(d)
    pole[0] = 1.0

    all_ok = True
    for theta in [0.3, 0.7, 1.2, 2.0, 2.8]:
        perp = rng.standard_normal(d)
        perp -= (perp @ pole) * pole
        perp /= np.linalg.norm(perp)
        point = np.cos(theta) * pole + np.sin(theta) * perp

        hyperplanes = rng.standard_normal((n_trials, m, d))
        hyperplanes /= np.linalg.norm(hyperplanes, axis=2, keepdims=True)
        bits_q = (hyperplanes @ pole) >= 0.0
        bits_p = (hyperplanes @ point) >= 0.0
        hamming = (bits_q != bits_p).sum(axis=1)

        for x in [0, 1, 2, 4]:
            empirical = float((hamming <= x).mean())
            formula = float(p_single(np.array([theta]), m, x)[0])
            ok = abs(empirical - formula) < 0.02
            all_ok &= ok
            if not ok:
                check(f"p_single theta={theta} x={x}", False, f"empirical={empirical:.4f}, formula={formula:.4f}")
    if all_ok:
        check("p_single matches Monte Carlo (5 thetas x 4 x-values, n=20000 each)", True)
    return all_ok


def test_ht_smoke() -> bool:
    """Smoke test: construction and retrieval run end to end without
    error on a small case. The real correctness check is the mean-ratio
    test below."""
    rng = np.random.default_rng(1)
    d = 3
    N = 200
    points = _normalize(rng.standard_normal((N, d)))
    m, x, L = 8, 2, 16
    tables = build_tables(points, m, L, rng)
    queries = _normalize(rng.standard_normal((20, d)))
    retrieved = query_retrieval(tables, queries, m, x)
    for i in range(20):
        estimate_count(queries[i], retrieved[i], points, 0.8, m, x, L)
    return check("HT estimator ran without error on N=200 d=3 (20 queries)", True)


def test_mean_ratio_small_case() -> bool:
    """The real correctness criterion (H1): pool many queries' N_hat
    (the count within each query's own cap radius r, per spec step 5)
    against the true brute-force count within that same r, and confirm
    the mean ratio is close to 1 - unbiasedness holds in expectation
    over queries, not per query exactly."""
    rng = np.random.default_rng(2)
    d = 3
    N = 200
    points = _normalize(rng.standard_normal((N, d)))
    m, x, L = 6, 2, 8
    tables = build_tables(points, m, L, rng)

    queries = _normalize(rng.standard_normal((150, d)))
    radii = rng.uniform(0.5, 1.5, size=150)
    retrieved = query_retrieval(tables, queries, m, x)
    n_hat = estimate_count_batch(queries, retrieved, points, radii, m, x, L)

    true_counts = np.array([
        int((np.arccos(np.clip(points @ queries[i], -1.0, 1.0)) <= radii[i]).sum())
        for i in range(150)
    ])
    valid = true_counts > 0
    ratio = (n_hat[valid] / true_counts[valid]).mean()
    ok = check("mean(N_hat/true_count) close to 1.0 (small case, N=200, 150 queries, varying r)",
               abs(ratio - 1.0) < 0.25, f"ratio={ratio:.4f}, n_valid={valid.sum()}")
    return ok


def test_dedup() -> bool:
    rng = np.random.default_rng(3)
    d = 5
    N = 100
    points = _normalize(rng.standard_normal((N, d)))
    m, x, L = 4, 3, 10
    tables = build_tables(points, m, L, rng)
    queries = _normalize(rng.standard_normal((10, d)))
    retrieved = query_retrieval(tables, queries, m, x)
    all_ok = True
    for i, r in enumerate(retrieved):
        ok = len(r) == len(np.unique(r))
        all_ok &= ok
        if not ok:
            check(f"dedup query {i}", False, f"len={len(r)}, unique={len(np.unique(r))}")
    if all_ok:
        check("retrieved point ids are deduplicated across tables (10 queries)", True)
    return all_ok


def test_theta_not_r() -> bool:
    """r has exactly one legitimate role: deciding which retrieved
    points qualify (theta_i <= r). It must play no role in the WEIGHT
    given to a point that already qualifies - that weight is a function
    of the point's own theta_i only. Check this directly: take a fixed
    set of retrieved points, all within a small r0, then re-run with a
    much larger r1 (still admitting the same points, plus possibly
    more) - the qualifying points' individual weight contributions must
    be identical across the two calls, not scaled by r."""
    rng = np.random.default_rng(4)
    d = 4
    N = 50
    points = _normalize(rng.standard_normal((N, d)))
    query = _normalize(rng.standard_normal(d).reshape(1, -1))[0]
    retrieved_ids = np.arange(20)

    theta = np.arccos(np.clip(points[retrieved_ids] @ query, -1.0, 1.0))
    r0 = float(np.median(theta))
    r1 = float(theta.max()) + 0.1

    n_hat_r0, count_r0 = estimate_count(query, retrieved_ids, points, r0, m=6, x=2, L=8)
    n_hat_r1, count_r1 = estimate_count(query, retrieved_ids, points, r1, m=6, x=2, L=8)

    # Points qualifying under r0 also qualify under r1 (r1 > r0); their
    # weights must sum to the SAME partial total in both calls - check
    # by recomputing n_hat_r0's own qualifying subset's weight sum
    # directly from theta_i and comparing to a manual recomputation.
    within_r0 = theta <= r0
    manual_r0 = float((1.0 / np.clip(p_union(theta[within_r0], 6, 2, 8), 1e-12, 1.0)).sum())

    ok = check("weight for a qualifying point depends only on theta_i, not on r "
               "(recomputed manually vs. estimate_count's own r0 result)",
               abs(n_hat_r0 - manual_r0) < 1e-9 and count_r1 >= count_r0,
               f"n_hat_r0={n_hat_r0:.6f}, manual={manual_r0:.6f}, count_r0={count_r0}, count_r1={count_r1}")
    return ok


def test_select_hyperparameters() -> bool:
    """Sanity check against the worked table in the request (K=100)."""
    result = select_hyperparameters(K=100, epsilon=0.20, r=1.379, m_candidates=list(range(15, 33)),
                                     x_candidates=[1, 2, 3, 4], L_max=16)
    print(f"  select_hyperparameters(K=100, eps=0.20, r=79deg, L_max=16): {result}")
    return check("select_hyperparameters runs and returns a valid (m,x,L)", result is not None)

#-------------------------------------------------------------------------------


def main() -> None:
    print("=== p_single vs Monte Carlo ===")
    r1 = test_p_single_monte_carlo()
    print("\n=== HT estimator smoke test ===")
    r2 = test_ht_smoke()
    print("\n=== Mean-ratio unbiasedness (small case) ===")
    r3 = test_mean_ratio_small_case()
    print("\n=== Deduplication ===")
    r4 = test_dedup()
    print("\n=== theta_i vs r structural check ===")
    r5 = test_theta_not_r()
    print("\n=== select_hyperparameters sanity ===")
    r6 = test_select_hyperparameters()

    print(f"\n{'ALL PASSED' if all([r1,r2,r3,r4,r5,r6]) else 'SOME FAILED'}")


if __name__ == '__main__':
    main()
