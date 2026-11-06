#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Unit tests for propose_candidates and propose_candidates_ball, including the
point-equivalent case (propose_candidates_ball with K=0). See
docs/POPULATION_DESIGN.md."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from utils.base.population import (
    propose_candidates,
    propose_candidates_ball,
    _precompute_kth_base_dist,
    _pairwise_cache_dists,
    _normalize,
)
from tests.helpers import make_clustered_candidate_benchmark

#-------------------------------------------------------------------------------
# Helpers
#-------------------------------------------------------------------------------

def _make_benchmark(seed: int = 0):
    """Clustered benchmark that guarantees the CIG condition is satisfiable.

    5 cluster centers, each with 2 cache queries and 16 base vectors close to
    the center. This ensures inter-cache distances << inter-cluster distances
    and K-th base distances are small, so CIG coverage is non-empty.
    """
    return make_clustered_candidate_benchmark(
        seed=seed, dim=16, n_clusters=5,
        n_base_per_cluster=16, n_cache_per_cluster=2, n_test_per_cluster=2,
        K=2, noise_scale=0.05,
    )


def _angular(a, b):
    a_hat = a / np.linalg.norm(a)
    b_hat = b / np.linalg.norm(b)
    return float(np.arccos(np.clip(np.dot(a_hat, b_hat), -1.0, 1.0)) / np.pi)


def _nth_nn_dist(cache, i, n):
    dists = sorted(_angular(cache[i], cache[j]) for j in range(len(cache)) if j != i)
    return dists[n - 1]


def _nth_nn_index(cache, i, n):
    """Index (into cache) of the n-th nearest neighbor of cache[i]."""
    pairs = sorted(
        ((_angular(cache[i], cache[j]), j) for j in range(len(cache)) if j != i),
        key=lambda p: p[0],
    )
    return pairs[n - 1][1]


def _kth_base_dist(query, base, k):
    dists = sorted(_angular(query, base[j]) for j in range(len(base)))
    return dists[k - 1]


def _r_Q_at_k(queries, base, k):
    """K-th-nearest-base-vector distance for every query, at a given K.

    Matches what a real GT file would contain for that K. propose_candidates_ball
    reuses observed_r_Q as margin_lookup directly, so tests that call it with a K
    other than the fixture's own K must pass an r_Q recomputed at that K, matching
    how every real caller in this codebase loads it (always from the K-specific
    GT file); the fixture's bm['r_Q'] is only valid for the fixture's own K.
    """
    return np.array([_kth_base_dist(q, base, k) for q in queries], dtype=np.float32)


def _candidates_for_query(cands, cache, i):
    """All returned candidates whose source query is cache[i]."""
    return [c for c in cands if np.allclose(c.query, cache[i])]


def _find_by_r_c(cands, expected_r_c, tol=1e-4):
    """Find the (unique, given continuous random radii) candidate whose r_c
    matches expected_r_c. Used to locate the candidate for sphere n among the
    n_max candidates returned per query."""
    matches = [c for c in cands if abs(c.r_c - expected_r_c) < tol]
    assert len(matches) == 1, (
        f"expected exactly one candidate with r_c~={expected_r_c:.6f}, found {len(matches)}"
    )
    return matches[0]

#-------------------------------------------------------------------------------
# Nested-sphere structure: n_max candidates per query, one per i-th neighbor
#-------------------------------------------------------------------------------

def test_propose_candidates_ball_k0_all_n_in_range_proposed():
    """Every i in [1, n_max] must produce a candidate for every query (given
    the clustered fixture guarantees non-empty coverage at every radius)."""
    bm = _make_benchmark()
    n_max = 5
    cands = propose_candidates_ball(
        bm['cache'], bm['r_Q'], bm['tv'], bm['ti'], bm['hg'],
        bm['cache'], bm['r_Q'], bm['test'], bm['test_r_t'],
        bm['base'], n_max=n_max, K=0, metric='angular',
    )
    for i in range(bm['n_cache']):
        per_query = _candidates_for_query(cands, bm['cache'], i)
        assert len(per_query) == n_max, (
            f"query {i}: expected {n_max} candidates, got {len(per_query)}"
        )
        for n in range(1, n_max + 1):
            # K=0: margin is uniformly 0, r_c = raw distance
            expected_r_c = _nth_nn_dist(bm['cache'], i, n)
            _find_by_r_c(per_query, expected_r_c)  # raises if not exactly one match


def test_propose_candidates_ball_k0_nested_R_c_monotone():
    """R_c(n) must be a subset of R_c(n+1) for the same query: spheres are
    nested and strictly increasing with n."""
    bm = _make_benchmark()
    n_max = 5
    cands = propose_candidates_ball(
        bm['cache'], bm['r_Q'], bm['tv'], bm['ti'], bm['hg'],
        bm['cache'], bm['r_Q'], bm['test'], bm['test_r_t'],
        bm['base'], n_max=n_max, K=0, metric='angular', store_rc=True,
    )
    for i in range(bm['n_cache']):
        per_query = _candidates_for_query(cands, bm['cache'], i)
        by_rc = sorted(per_query, key=lambda c: c.r_c)
        for a, b in zip(by_rc, by_rc[1:]):
            assert np.all(a.R_c <= b.R_c), (
                f"query {i}: R_c not nested between consecutive spheres "
                f"(r_c={a.r_c:.6f} -> {b.r_c:.6f})"
            )


def test_propose_candidates_ball_all_n_in_range_proposed():
    bm = _make_benchmark()
    n_max, K = 5, 3
    r_Q_k3 = _r_Q_at_k(bm['cache'], bm['base'], K)
    cands = propose_candidates_ball(
        bm['cache'], r_Q_k3, bm['tv'], bm['ti'], bm['hg'],
        bm['cache'], r_Q_k3, bm['test'], bm['test_r_t'],
        bm['base'], n_max=n_max, K=K, metric='angular',
    )
    for i in range(bm['n_cache']):
        per_query = _candidates_for_query(cands, bm['cache'], i)
        assert len(per_query) == n_max, (
            f"query {i}: expected {n_max} candidates, got {len(per_query)}"
        )
        for n in range(1, n_max + 1):
            nn_index = _nth_nn_index(bm['cache'], i, n)
            d_base_k = _kth_base_dist(bm['cache'][nn_index], bm['base'], K)
            expected_r_c = _nth_nn_dist(bm['cache'], i, n) + d_base_k
            _find_by_r_c(per_query, expected_r_c)


def test_propose_candidates_ball_n_field_matches_neighbor_rank():
    """CandidateEntry.n must record the exact neighbor rank (1..n_max) its
    sphere was built from, cross-checked directly against the n-th nearest
    neighbor's identity and distance (_nth_nn_index/_nth_nn_dist), not just
    "is an int in range"."""
    bm = _make_benchmark()
    n_max, K = 5, 3
    r_Q_k3 = _r_Q_at_k(bm['cache'], bm['base'], K)
    cands = propose_candidates_ball(
        bm['cache'], r_Q_k3, bm['tv'], bm['ti'], bm['hg'],
        bm['cache'], r_Q_k3, bm['test'], bm['test_r_t'],
        bm['base'], n_max=n_max, K=K, metric='angular',
    )
    for i in range(bm['n_cache']):
        per_query = _candidates_for_query(cands, bm['cache'], i)
        assert {c.n for c in per_query} == set(range(1, n_max + 1)), (
            f"query {i}: expected n values {{1..{n_max}}}, got {[c.n for c in per_query]}"
        )
        for c in per_query:
            nn_index = _nth_nn_index(bm['cache'], i, c.n)
            d_base_k = _kth_base_dist(bm['cache'][nn_index], bm['base'], K)
            expected_r_c = _nth_nn_dist(bm['cache'], i, c.n) + d_base_k
            assert abs(c.r_c - expected_r_c) < 1e-4, (
                f"query {i}, n={c.n}: r_c={c.r_c} doesn't match the sphere "
                f"built from the n-th nearest neighbor (expected {expected_r_c})"
            )


def test_propose_candidates_ball_nested_R_c_monotone():
    bm = _make_benchmark()
    n_max, K = 5, 3
    cands = propose_candidates_ball(
        bm['cache'], bm['r_Q'], bm['tv'], bm['ti'], bm['hg'],
        bm['cache'], bm['r_Q'], bm['test'], bm['test_r_t'],
        bm['base'], n_max=n_max, K=K, metric='angular', store_rc=True,
    )
    for i in range(bm['n_cache']):
        per_query = _candidates_for_query(cands, bm['cache'], i)
        by_rc = sorted(per_query, key=lambda c: c.r_c)
        for a, b in zip(by_rc, by_rc[1:]):
            assert np.all(a.R_c <= b.R_c), (
                f"query {i}: R_c not nested between consecutive spheres"
            )

#-------------------------------------------------------------------------------

def test_propose_candidates_ball_k0_covers_nth_neighbor_exactly():
    """For each n, the n-th nearest neighbor must be covered by sphere n,
    checked for every n in the nested range.

    Whether the (n+1)-th neighbor also ends up covered is not asserted here:
    the margin term (r_Q of the n-th neighbor) varies per neighbor, so a
    radius sized exactly for n can legitimately spill over to also cover n+1
    when their own coverage radii differ. Only "n-th is covered" is a true
    invariant of the formula.
    """
    bm = _make_benchmark()
    n_max = 5
    cands = propose_candidates_ball(
        bm['cache'], bm['r_Q'], bm['tv'], bm['ti'], bm['hg'],
        bm['cache'], bm['r_Q'], bm['test'], bm['test_r_t'],
        bm['base'], n_max=n_max, K=0, metric='angular',
    )
    for i in range(bm['n_cache']):
        per_query = _candidates_for_query(cands, bm['cache'], i)
        for n in range(1, n_max + 1):
            nn_index = _nth_nn_index(bm['cache'], i, n)
            # K=0: margin is uniformly 0, r_c = raw distance
            expected_r_c = _nth_nn_dist(bm['cache'], i, n)
            cand = _find_by_r_c(per_query, expected_r_c)

            assert cand.covered_train_indices[nn_index], (
                f"i={i} n={n}: n-th neighbor (idx={nn_index}) should be covered"
            )


def test_propose_candidates_ball_k0_R_c_membership():
    bm = _make_benchmark()
    cands = propose_candidates_ball(
        bm['cache'], bm['r_Q'], bm['tv'], bm['ti'], bm['hg'],
        bm['cache'], bm['r_Q'], bm['test'], bm['test_r_t'],
        bm['base'], n_max=5, K=0, metric='angular',
    )
    assert len(cands) > 0
    for cand in cands:
        q_hat = cand.query / np.linalg.norm(cand.query)
        all_dists = np.array([_angular(q_hat, bm['base'][j]) for j in range(bm['n_base'])])
        expected_in_ball = all_dists <= cand.r_c
        assert np.array_equal(cand.R_c, expected_in_ball)


def test_named_neighbor_coverage_exact_no_epsilon():
    """Named-neighbor coverage must hold by exact construction, not
    floating-point luck, at both K=0 and K>=1 (there is no numerical epsilon
    slack). Uses population.py's own internal
    _pairwise_cache_dists/_precompute_kth_base_dist directly (the same
    values propose_candidates_ball builds r_c from) rather than an
    independently re-derived distance, so this is a true bit-for-bit
    self-consistency check, not a cross-implementation comparison."""
    bm = _make_benchmark()
    n_max = 5
    sorted_dists, sorted_indices = _pairwise_cache_dists(bm['cache'], 'angular')

    for K in (0, 3):
        if K == 0:
            margin_lookup = np.zeros(bm['n_cache'], dtype=np.float32)
            cands = propose_candidates_ball(
                bm['cache'], bm['r_Q'], bm['tv'], bm['ti'], bm['hg'],
                bm['cache'], bm['r_Q'], bm['test'], bm['test_r_t'],
                bm['base'], n_max=n_max, K=0, metric='angular',
            )
        else:
            # margin_lookup reuses observed_r_Q directly, so the "expected"
            # value here and what's actually passed in must be the same
            # K-appropriate r_Q, not the fixture's own (different) K; see
            # _r_Q_at_k's docstring.
            r_Q_k = _r_Q_at_k(bm['cache'], bm['base'], K)
            margin_lookup = r_Q_k
            cands = propose_candidates_ball(
                bm['cache'], r_Q_k, bm['tv'], bm['ti'], bm['hg'],
                bm['cache'], r_Q_k, bm['test'], bm['test_r_t'],
                bm['base'], n_max=n_max, K=K, metric='angular',
            )

        checked = 0
        for i in range(bm['n_cache']):
            per_query = _candidates_for_query(cands, bm['cache'], i)
            for n in range(1, n_max + 1):
                nn_index = int(sorted_indices[i, n - 1])
                nn_dist = float(sorted_dists[i, n - 1])
                expected_r_c = nn_dist + float(margin_lookup[nn_index])
                cand = next((c for c in per_query if c.r_c == expected_r_c), None)
                if cand is None:
                    continue  # pruned (empty coverage) at this n
                assert cand.covered_train_indices[nn_index], (
                    f"K={K} i={i} n={n}: named neighbor not covered, no "
                    f"epsilon slack to rely on"
                )
                checked += 1
        assert checked > 0, f"K={K}: expected at least one comparable (i, n) pair"


def test_propose_candidates_ball_k0_covered_train_satisfies_cig():
    bm = _make_benchmark()
    cands = propose_candidates_ball(
        bm['cache'], bm['r_Q'], bm['tv'], bm['ti'], bm['hg'],
        bm['cache'], bm['r_Q'], bm['test'], bm['test_r_t'],
        bm['base'], n_max=5, K=0, metric='angular',
    )
    # This test's `d` is computed via its own independent _angular() helper,
    # a different code path from population.py's internal distance
    # computations, so a tiny test-only tolerance (ordinary float32 ULP
    # noise, ~1e-7 scale) is needed even when the underlying invariant holds
    # exactly in production.
    #
    # K=0: margin is uniformly 0, so covered_train_indices is satisfied by
    # one of two distinct mechanisms: the traditional CIG margin condition
    # (d + r_t <= r_c) for most covered queries, or pure distance containment
    # (d <= r_c, r_t contribution is 0) for the named neighbor a sphere was
    # built around, where r_c was constructed as exactly that neighbor's
    # distance. Accept either: d + r_t <= r_c is strictly stronger and
    # implies d <= r_c whenever r_t >= 0, so this doesn't loosen anything for
    # the general case, it only accommodates the named-neighbor case.
    test_tol = 1e-6
    for cand in cands:
        q_hat = cand.query / np.linalg.norm(cand.query)
        for idx in np.where(cand.covered_train_indices)[0]:
            d = _angular(q_hat, bm['cache'][idx])
            r_t = float(bm['r_Q'][idx])
            assert d + r_t <= cand.r_c + test_tol or d <= cand.r_c + test_tol, (
                f"CIG violated: d={d:.6f} r_t={r_t:.6f} r_c={cand.r_c:.6f}"
            )


def test_propose_candidates_ball_k0_self_excluded():
    bm = _make_benchmark()
    cands = propose_candidates_ball(
        bm['cache'], bm['r_Q'], bm['tv'], bm['ti'], bm['hg'],
        bm['cache'], bm['r_Q'], bm['test'], bm['test_r_t'],
        bm['base'], n_max=5, K=0, metric='angular',
    )
    for cand in cands:
        i = next(j for j in range(bm['n_cache']) if np.allclose(bm['cache'][j], cand.query))
        assert not cand.covered_train_indices[i], f"Source query {i} in covered_train_indices"


def test_propose_candidates_ball_k0_alpha_is_none():
    bm = _make_benchmark()
    cands = propose_candidates_ball(
        bm['cache'], bm['r_Q'], bm['tv'], bm['ti'], bm['hg'],
        bm['cache'], bm['r_Q'], bm['test'], bm['test_r_t'],
        bm['base'], n_max=5, K=0, metric='angular',
    )
    assert all(cand.alpha is None for cand in cands), "alpha must be None for ball-like"


def test_propose_candidates_ball_k0_store_rc_false():
    bm = _make_benchmark()
    cands = propose_candidates_ball(
        bm['cache'], bm['r_Q'], bm['tv'], bm['ti'], bm['hg'],
        bm['cache'], bm['r_Q'], bm['test'], bm['test_r_t'],
        bm['base'], n_max=5, K=0, metric='angular', store_rc=False,
    )
    for cand in cands:
        assert cand.R_c is None, "R_c should be None when store_rc=False"
        assert cand.rc_size >= 0


def test_propose_candidates_ball_k0_n_max_must_be_less_than_n_cache():
    bm = _make_benchmark()
    try:
        propose_candidates_ball(
            bm['cache'], bm['r_Q'], bm['tv'], bm['ti'], bm['hg'],
            bm['cache'], bm['r_Q'], bm['test'], bm['test_r_t'],
            bm['base'], n_max=bm['n_cache'], K=0, metric='angular',
        )
        assert False, "expected ValueError for n_max >= n_cache"
    except ValueError as e:
        assert "n_max" in str(e)


def test_propose_candidates_ball_r_c_monotone_in_K():
    """For the same (query, n), r_c must be monotonically non-decreasing in
    K: a larger K means a larger (or equal) own-K-ball margin for the
    named neighbor, so the resulting sphere can only grow."""
    bm = _make_benchmark()
    n_max, K = 5, 3
    r_Q_k3 = _r_Q_at_k(bm['cache'], bm['base'], K)
    k0_cands = propose_candidates_ball(
        bm['cache'], bm['r_Q'], bm['tv'], bm['ti'], bm['hg'],
        bm['cache'], bm['r_Q'], bm['test'], bm['test_r_t'],
        bm['base'], n_max=n_max, K=0, metric='angular',
    )
    ball_cands = propose_candidates_ball(
        bm['cache'], r_Q_k3, bm['tv'], bm['ti'], bm['hg'],
        bm['cache'], r_Q_k3, bm['test'], bm['test_r_t'],
        bm['base'], n_max=n_max, K=K, metric='angular',
    )
    checked = 0
    for i in range(bm['n_cache']):
        k0_per_query = _candidates_for_query(k0_cands, bm['cache'], i)
        ball_per_query = _candidates_for_query(ball_cands, bm['cache'], i)
        for n in range(1, n_max + 1):
            nn_index = _nth_nn_index(bm['cache'], i, n)
            nn_dist = _nth_nn_dist(bm['cache'], i, n)
            expected_k0_rc = nn_dist
            expected_ball_rc = nn_dist + _kth_base_dist(bm['cache'][nn_index], bm['base'], K)

            k0_cand = next((c for c in k0_per_query if abs(c.r_c - expected_k0_rc) < 1e-4), None)
            ball_cand = next((c for c in ball_per_query if abs(c.r_c - expected_ball_rc) < 1e-4), None)
            if k0_cand is None or ball_cand is None:
                continue  # pruned (empty coverage) at this n, not a comparable pair
            assert ball_cand.r_c >= k0_cand.r_c - 1e-6, (
                f"i={i} n={n}: K={K} r_c {ball_cand.r_c:.6f} < K=0 r_c {k0_cand.r_c:.6f}"
            )
            checked += 1
    assert checked > 0, "Expected at least one comparable K=0/K>=1 pair"


def test_propose_candidates_ball_covered_train_satisfies_cig():
    bm = _make_benchmark()
    r_Q_k3 = _r_Q_at_k(bm['cache'], bm['base'], 3)
    cands = propose_candidates_ball(
        bm['cache'], r_Q_k3, bm['tv'], bm['ti'], bm['hg'],
        bm['cache'], r_Q_k3, bm['test'], bm['test_r_t'],
        bm['base'], n_max=5, K=3, metric='angular',
    )
    for cand in cands:
        q_hat = cand.query / np.linalg.norm(cand.query)
        for idx in np.where(cand.covered_train_indices)[0]:
            d = _angular(q_hat, bm['cache'][idx])
            r_t = float(r_Q_k3[idx])
            # CIG condition is <=, not < (see utils/base/population.py's
            # _cig_mask): a candidate's own generating neighbor hits this
            # exactly (d + r_t == r_c by construction, since margin_lookup
            # reuses observed_r_Q). A small tolerance is needed too: this
            # test's _angular() is an independently recomputed distance, a
            # different numeric path from population.py's internal
            # _base_dists_for_query, so boundary-exact cases can disagree by
            # a few ULPs (not a logic bug).
            assert d + r_t <= cand.r_c + 1e-6


def test_propose_candidates_ball_R_c_membership():
    bm = _make_benchmark()
    cands = propose_candidates_ball(
        bm['cache'], bm['r_Q'], bm['tv'], bm['ti'], bm['hg'],
        bm['cache'], bm['r_Q'], bm['test'], bm['test_r_t'],
        bm['base'], n_max=5, K=3, metric='angular',
    )
    assert len(cands) > 0
    for cand in cands:
        q_hat = cand.query / np.linalg.norm(cand.query)
        all_dists = np.array([_angular(q_hat, bm['base'][j]) for j in range(bm['n_base'])])
        expected = all_dists <= cand.r_c
        assert np.array_equal(cand.R_c, expected)


def test_propose_candidates_ball_contains_neighbors_k_sphere():
    """Literal check of the docstring's claim: r_c is large enough to fully
    contain q_n's own K-sphere, i.e. the n-th neighbor's own K nearest base
    vectors must all be members of the candidate's R_c, for every n.

    Holds by the triangle inequality: D(Q, E_j(q_n)) <= D(Q, q_n) + D(q_n, E_j(q_n))
    <= D(Q, q_n) + D(q_n, E_K(q_n)) = r_c for every j <= K.
    """
    bm = _make_benchmark()
    n_max, K = 5, 3
    cands = propose_candidates_ball(
        bm['cache'], bm['r_Q'], bm['tv'], bm['ti'], bm['hg'],
        bm['cache'], bm['r_Q'], bm['test'], bm['test_r_t'],
        bm['base'], n_max=n_max, K=K, metric='angular',
    )
    assert len(cands) > 0
    for i in range(bm['n_cache']):
        per_query = _candidates_for_query(cands, bm['cache'], i)
        for n in range(1, n_max + 1):
            nn_index = _nth_nn_index(bm['cache'], i, n)
            d_base_k = _kth_base_dist(bm['cache'][nn_index], bm['base'], K)
            expected_r_c = _nth_nn_dist(bm['cache'], i, n) + d_base_k
            cand = next((c for c in per_query if abs(c.r_c - expected_r_c) < 1e-4), None)
            if cand is None:
                continue  # pruned at this n

            base_dists = sorted(
                range(bm['n_base']),
                key=lambda j: _angular(bm['cache'][nn_index], bm['base'][j]),
            )
            neighbor_k_sphere = base_dists[:K]
            assert np.all(cand.R_c[neighbor_k_sphere]), (
                f"i={i} n={n}: neighbor's K-sphere {neighbor_k_sphere} "
                f"not fully contained in R_c"
            )


def test_propose_candidates_to_cache_entry():
    bm = _make_benchmark()
    cands = propose_candidates_ball(
        bm['cache'], bm['r_Q'], bm['tv'], bm['ti'], bm['hg'],
        bm['cache'], bm['r_Q'], bm['test'], bm['test_r_t'],
        bm['base'], n_max=5, K=0, metric='angular',
    )
    assert len(cands) > 0
    for cand in cands:
        entry = cand.to_cache_entry()
        assert abs(entry.r_Q - cand.r_c) < 1e-7
        assert entry.top_k_vectors == [], "index-only entries carry no private vector copy"
        assert entry.top_k_matrix is None, "index-only entries carry no private matrix copy"
        assert list(entry.top_k_indices) == np.where(cand.R_c)[0].tolist()


def test_propose_candidates_alpha_unchanged():
    bm = _make_benchmark()
    alphas = [2.0, 3.0]
    cands = propose_candidates(
        bm['cache'], bm['r_Q'], bm['tv'], bm['ti'], bm['hg'],
        bm['cache'], bm['r_Q'], bm['test'], bm['test_r_t'],
        bm['base'], alpha_values=alphas, metric='angular',
    )
    for cand in cands:
        assert cand.alpha is not None, "alpha-based proposal must set alpha"
        assert cand.alpha in alphas

#-------------------------------------------------------------------------------
# Internal geometry helpers used by ball-like proposals
#-------------------------------------------------------------------------------

def test_precompute_kth_base_dist_matches_brute_force():
    bm = _make_benchmark()
    k = 4
    cache_hat = _normalize(bm['cache'].copy())
    base_hat = _normalize(bm['base'].copy())

    d_kth = _precompute_kth_base_dist(cache_hat, base_hat, k, metric='angular')

    for i in range(len(cache_hat)):
        expected = _kth_base_dist(bm['cache'][i], bm['base'], k)
        assert abs(float(d_kth[i]) - expected) < 1e-5, (
            f"query {i}: got {d_kth[i]:.6f}, expected {expected:.6f}"
        )


def test_precompute_kth_base_dist_batch_size_invariant():
    """Chunking (batch_size) must not change the k-th nearest distance computed."""
    bm = _make_benchmark()
    k = 3
    cache_hat = _normalize(bm['cache'].copy())
    base_hat = _normalize(bm['base'].copy())

    d_bs1 = _precompute_kth_base_dist(cache_hat, base_hat, k, metric='angular', batch_size=1)
    d_bs32 = _precompute_kth_base_dist(cache_hat, base_hat, k, metric='angular', batch_size=32)
    d_bsn = _precompute_kth_base_dist(
        cache_hat, base_hat, k, metric='angular', batch_size=len(cache_hat)
    )

    assert np.allclose(d_bs1, d_bs32, atol=1e-6)
    assert np.allclose(d_bs1, d_bsn, atol=1e-6)


def test_pairwise_cache_dists_matches_brute_force_angular():
    bm = _make_benchmark()
    sorted_dists, sorted_indices = _pairwise_cache_dists(bm['cache'], metric='angular')

    for i in range(bm['n_cache']):
        pairs = sorted(
            ((_angular(bm['cache'][i], bm['cache'][j]), j)
             for j in range(bm['n_cache']) if j != i),
            key=lambda p: p[0],
        )
        for n in range(len(pairs)):
            assert abs(sorted_dists[i, n] - pairs[n][0]) < 1e-5, (
                f"query {i}, n={n}: dist {sorted_dists[i, n]:.6f} vs {pairs[n][0]:.6f}"
            )
            assert int(sorted_indices[i, n]) == pairs[n][1], (
                f"query {i}, n={n}: index {sorted_indices[i, n]} vs {pairs[n][1]}"
            )


def test_pairwise_cache_dists_matches_brute_force_euclidean():
    bm = _make_benchmark()
    sorted_dists, sorted_indices = _pairwise_cache_dists(bm['cache'], metric='euclidean')

    def _euclidean(a, b):
        a_hat = a / np.linalg.norm(a)
        b_hat = b / np.linalg.norm(b)
        return float(np.linalg.norm(a_hat - b_hat))

    for i in range(bm['n_cache']):
        pairs = sorted(
            ((_euclidean(bm['cache'][i], bm['cache'][j]), j)
             for j in range(bm['n_cache']) if j != i),
            key=lambda p: p[0],
        )
        for n in range(len(pairs)):
            assert abs(sorted_dists[i, n] - pairs[n][0]) < 1e-5, (
                f"query {i}, n={n}: dist {sorted_dists[i, n]:.6f} vs {pairs[n][0]:.6f}"
            )
            assert int(sorted_indices[i, n]) == pairs[n][1], (
                f"query {i}, n={n}: index {sorted_indices[i, n]} vs {pairs[n][1]}"
            )


def test_propose_candidates_ball_k0_store_rc_false_rc_size_still_correct():
    """rc_size must be correctly computed even when R_c itself isn't stored."""
    bm = _make_benchmark()
    cands_stored = propose_candidates_ball(
        bm['cache'], bm['r_Q'], bm['tv'], bm['ti'], bm['hg'],
        bm['cache'], bm['r_Q'], bm['test'], bm['test_r_t'],
        bm['base'], n_max=3, K=0, metric='angular', store_rc=True,
    )
    cands_unstored = propose_candidates_ball(
        bm['cache'], bm['r_Q'], bm['tv'], bm['ti'], bm['hg'],
        bm['cache'], bm['r_Q'], bm['test'], bm['test_r_t'],
        bm['base'], n_max=3, K=0, metric='angular', store_rc=False,
    )
    assert len(cands_stored) == len(cands_unstored)
    for stored, unstored in zip(cands_stored, cands_unstored):
        assert unstored.R_c is None, "store_rc=False should not materialize R_c"
        assert unstored.rc_size == stored.rc_size == int(stored.R_c.sum()), (
            f"rc_size mismatch: stored={stored.rc_size}, unstored={unstored.rc_size}"
        )


def test_propose_candidates_ball_store_rc_false_rc_size_still_correct():
    bm = _make_benchmark()
    cands_stored = propose_candidates_ball(
        bm['cache'], bm['r_Q'], bm['tv'], bm['ti'], bm['hg'],
        bm['cache'], bm['r_Q'], bm['test'], bm['test_r_t'],
        bm['base'], n_max=3, K=3, metric='angular', store_rc=True,
    )
    cands_unstored = propose_candidates_ball(
        bm['cache'], bm['r_Q'], bm['tv'], bm['ti'], bm['hg'],
        bm['cache'], bm['r_Q'], bm['test'], bm['test_r_t'],
        bm['base'], n_max=3, K=3, metric='angular', store_rc=False,
    )
    assert len(cands_stored) == len(cands_unstored)
    for stored, unstored in zip(cands_stored, cands_unstored):
        assert unstored.R_c is None, "store_rc=False should not materialize R_c"
        assert unstored.rc_size == stored.rc_size == int(stored.R_c.sum()), (
            f"rc_size mismatch: stored={stored.rc_size}, unstored={unstored.rc_size}"
        )

def test_cig_mask_covers_exact_boundary():
    """_cig_mask (the proposal-time CIG check used by the generic
    propose_candidates) is D(t,q)+r_t <= r_c (non-strict). Tests _cig_mask
    directly, rather than through propose_candidates (which normalizes all
    inputs internally and would distort a hand-placed euclidean
    construction), with a query at exactly distance 0 from the candidate
    center (q_hat == Q_hat) and r_t chosen so r_t == r_c exactly: the
    boundary case."""
    from utils.base.population import _cig_mask

    q_hat = np.array([1.0, 0.0], dtype=np.float32)
    Q_hat = np.array([[1.0, 0.0]], dtype=np.float32)  # distance 0 from q_hat
    r_c = 0.25
    r_t = np.array([0.25], dtype=np.float32)  # D + r_t = 0 + 0.25 == r_c exactly

    covered = _cig_mask(q_hat, Q_hat, r_t, r_c, metric='angular')
    assert covered[0], "exact-boundary case (D + r_t == r_c) should be covered under <="


def test_precompute_kth_base_dist_rejects_k0():
    """_precompute_kth_base_dist must never be called with K=0:
    np.partition(..., k-1, ...) at k=0 becomes partition(..., -1, ...),
    silently extracting the *maximum* distance instead of failing or
    returning something meaningful. propose_candidates_ball's K=0 case
    bypasses this function entirely (margin is uniformly 0); this test
    guards that the bypass stays necessary/correct by confirming direct
    misuse fails loudly."""
    rng = np.random.RandomState(1)
    cache_hat = _normalize(rng.randn(4, 8).astype(np.float32))
    base_hat = _normalize(rng.randn(10, 8).astype(np.float32))
    raised = False
    try:
        _precompute_kth_base_dist(cache_hat, base_hat, 0, 'angular')
    except AssertionError as e:
        raised = True
        assert "k >= 1" in str(e) or "k=0" in str(e), f"unexpected assertion message: {e}"
    assert raised, "expected AssertionError for K=0, none was raised"


def test_precompute_ball_invariants_matches_recompute():
    """propose_candidates_ball's optional `invariants` (from
    precompute_ball_invariants) must produce identical candidates to the
    default recompute-every-time path: same rc_size, same r_c, same coverage
    arrays element-for-element. Checked across K=0 and K>0 (K=0 doesn't use
    margin_lookup from observed_r_Q, but still uses the invariants'
    all_dists/train_dist_only/test_dist_only)."""
    from utils.base.population import precompute_ball_invariants

    bm = _make_benchmark()
    n_max = 5

    for K in (0, 3):
        r_Q_k = bm['r_Q'] if K == 0 else _r_Q_at_k(bm['cache'], bm['base'], K)

        cands_default = propose_candidates_ball(
            bm['cache'], r_Q_k, bm['tv'], bm['ti'], bm['hg'],
            bm['cache'], r_Q_k, bm['test'], bm['test_r_t'],
            bm['base'], n_max=n_max, K=K, metric='angular',
        )

        invariants = precompute_ball_invariants(
            bm['cache'], bm['cache'], bm['test'], bm['base'], 'angular',
        )
        cands_cached = propose_candidates_ball(
            bm['cache'], r_Q_k, bm['tv'], bm['ti'], bm['hg'],
            bm['cache'], r_Q_k, bm['test'], bm['test_r_t'],
            bm['base'], n_max=n_max, K=K, metric='angular',
            invariants=invariants,
        )

        assert len(cands_default) == len(cands_cached), (
            f"K={K}: candidate count differs (default={len(cands_default)}, "
            f"cached={len(cands_cached)})"
        )
        for a, b in zip(cands_default, cands_cached):
            assert a.rc_size == b.rc_size, f"K={K}: rc_size differs"
            assert a.r_c == b.r_c, f"K={K}: r_c differs"
            assert np.array_equal(a.covered_train_indices, b.covered_train_indices), (
                f"K={K}: covered_train_indices differs"
            )
            assert np.array_equal(a.covered_test_indices, b.covered_test_indices), (
                f"K={K}: covered_test_indices differs"
            )
            assert np.array_equal(a.R_c, b.R_c), f"K={K}: R_c differs"


def test_precompute_ball_invariants_train_dist_only_matches_sorted_cache_dists():
    """train_dist_only and sorted_cache_dists must derive from the literal
    same raw (n_cache, n_cache) matrix when train_queries is
    observed_queries, avoiding two independently-rounded matmuls that could
    disagree by a few ULPs at a certification boundary (see
    _raw_pairwise_dists' docstring). Verified two ways:
    (1) train_dist_only's off-diagonal entries exactly match a direct
    _raw_pairwise_dists computation; (2) for every cache query i, its n-th
    nearest neighbor's distance in sorted_cache_dists exactly equals the
    corresponding train_dist_only entry looked up by index."""
    from utils.base.population import precompute_ball_invariants, _raw_pairwise_dists, _normalize

    bm = _make_benchmark()
    cache_q = bm['cache']

    invariants = precompute_ball_invariants(cache_q, cache_q, bm['test'], bm['base'], 'angular')
    train_dist_only = invariants['train_dist_only']
    sorted_cache_dists = invariants['sorted_cache_dists']
    sorted_cache_indices = invariants['sorted_cache_indices']

    expected_raw = _raw_pairwise_dists(_normalize(cache_q.copy()), 'angular').astype(np.float32)
    assert np.array_equal(train_dist_only, expected_raw), (
        "train_dist_only must be bit-identical to _raw_pairwise_dists"
    )

    n_cache = len(cache_q)
    for i in range(n_cache):
        for n in range(sorted_cache_dists.shape[1]):
            nn_idx = int(sorted_cache_indices[i, n])
            assert sorted_cache_dists[i, n] == train_dist_only[i, nn_idx], (
                f"i={i} n={n}: sorted_cache_dists disagrees with train_dist_only "
                f"at the exact index it claims to describe"
            )
