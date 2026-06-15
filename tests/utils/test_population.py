#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from utils.base import MainMemory, angular_distance, euclidean_distance
from utils.base.population import CandidateEntry, propose_candidates

#-------------------------------------------------------------------------------

def _make_dataset(n_base=30, n_obs=5, n_test=10, d=8, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n_base, d)).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    obs = rng.standard_normal((n_obs, d)).astype(np.float32)
    obs /= np.linalg.norm(obs, axis=1, keepdims=True)
    test = rng.standard_normal((n_test, d)).astype(np.float32)
    test /= np.linalg.norm(test, axis=1, keepdims=True)
    return base, obs, test

def _gt_for_queries(queries, base, K=5):
    """Brute-force top-K for each query. Returns r_Q, top_k_vecs, top_k_idx, half_gaps."""
    mm = MainMemory(vectors=list(base))
    r_Q_list, tv_list, ti_list, hg_list = [], [], [], []
    for q in queries:
        vecs, dists, gap, indices = mm.top_k_search(q, k=K, metric='angular',
                                                     return_indices=True)
        r_Q_list.append(float(dists[-1]))
        tv_list.append(vecs)
        ti_list.append(indices)
        hg_list.append(gap / 2.0)
    return (np.array(r_Q_list, dtype=np.float32), tv_list, ti_list,
            np.array(hg_list, dtype=np.float32))

#-------------------------------------------------------------------------------

def test_candidate_entry_to_cache_entry():
    """to_cache_entry() uses r_c (not r_Q) as the CacheEntry radius."""
    print("Test: CandidateEntry.to_cache_entry()")
    base, obs, test = _make_dataset()
    r_Q, tv, ti, hg = _gt_for_queries(obs[:1], base)

    cand = CandidateEntry(
        query=obs[0],
        r_Q=float(r_Q[0]),
        alpha=1.2,
        r_c=1.2 * float(r_Q[0]),
        R_c={0, 1, 2},
        rc_size=3,
        covered_train_indices={0, 1},
        covered_test_indices={3, 7},
        top_k_vectors=tv[0],
        top_k_indices=ti[0],
        half_gap=float(hg[0]),
    )
    entry = cand.to_cache_entry()

    assert abs(entry.r_Q - cand.r_c) < 1e-7, "r_Q should equal r_c"
    assert abs(entry.half_gap - cand.half_gap) < 1e-7, "half_gap should be preserved"
    assert entry.top_k_indices == cand.top_k_indices, "top_k_indices should be preserved"
    assert len(entry.top_k_vectors) == len(cand.top_k_vectors)
    print("  Passed.")

#-------------------------------------------------------------------------------

def test_propose_candidates_coverage_correctness():
    """covered_test_indices matches the manual CIG coverage check."""
    print("Test: coverage correctness")
    for metric in ('angular', 'euclidean'):
        base, obs, test = _make_dataset(n_base=30, n_obs=3, n_test=8)
        r_Q, tv, ti, hg = _gt_for_queries(obs, base, K=3)
        test_r_Q, _, _, _ = _gt_for_queries(test, base, K=3)
        test_r_t = test_r_Q

        alpha = 1.3
        candidates = propose_candidates(
            obs, r_Q, tv, ti, hg,
            obs, r_Q,
            test, test_r_t,
            base, alpha_values=[alpha], metric=metric,
        )

        dist_fn = angular_distance if metric == 'angular' else euclidean_distance

        for cand in candidates:
            r_c = cand.r_c
            expected = set()
            for j, tq in enumerate(test):
                d = dist_fn(tq, cand.query)
                if d + test_r_t[j] < r_c:
                    expected.add(j)
            assert cand.covered_test_indices == expected, \
                f"Coverage mismatch for metric={metric}: got {cand.covered_test_indices}, expected {expected}"

    print("  Passed.")

#-------------------------------------------------------------------------------

def test_propose_candidates_alpha_scaling():
    """Larger alpha produces equal or more R_c entries for the same query."""
    print("Test: alpha scaling")
    # Use test queries as train (no source exclusion) to ensure both alphas produce candidates.
    base, obs, test = _make_dataset(n_base=50)
    r_Q, tv, ti, hg = _gt_for_queries(obs, base, K=3)
    test_r_t = np.zeros(len(test), dtype=np.float32)

    candidates = propose_candidates(
        obs, r_Q, tv, ti, hg,
        test, test_r_t,
        test, test_r_t,
        base, alpha_values=[1.0, 2.0], metric='angular',
    )

    # group by query
    by_query: dict = {}
    for c in candidates:
        key = tuple(c.query.tolist())
        by_query.setdefault(key, {})[c.alpha] = c

    found = False
    for key, by_alpha in by_query.items():
        if 1.0 in by_alpha and 2.0 in by_alpha:
            assert len(by_alpha[2.0].R_c) >= len(by_alpha[1.0].R_c), \
                "Larger alpha must produce >= |R_c|"
            found = True

    assert found, "Need at least one query with both alphas to test scaling"
    print("  Passed.")

#-------------------------------------------------------------------------------

def test_propose_source_query_excluded():
    """Source query index must not appear in any candidate's covered_train_indices."""
    print("Test: source query excluded from train coverage")
    base, obs, _ = _make_dataset(n_base=30, n_obs=5, n_test=10, seed=0)
    r_Q, tv, ti, hg = _gt_for_queries(obs, base, K=3)
    # With obs as both train and test, source i is discarded from train coverage.
    candidates = propose_candidates(
        obs, r_Q, tv, ti, hg,
        obs, r_Q,
        obs, r_Q,
        base, alpha_values=[5.0], metric='angular',
    )
    assert len(candidates) > 0, "Expected at least one candidate; increase alpha or check test data"
    for c in candidates:
        src_idx = next(idx for idx, q in enumerate(obs) if np.allclose(q, c.query))
        assert src_idx not in c.covered_train_indices, \
            f"Source index {src_idx} found in covered_train_indices for candidate {src_idx}"
    print(f"  {len(candidates)} candidates checked. Passed.")

#-------------------------------------------------------------------------------

def test_propose_empty_covered_pruned():
    """Candidates whose train covered set is empty after source exclusion are pruned."""
    print("Test: empty-covered candidates pruned")
    base, obs, _ = _make_dataset(n_base=30, n_obs=3, n_test=10)
    r_Q, tv, ti, hg = _gt_for_queries(obs[:1], base, K=3)
    # 1 train query = the source itself; after discard(0), train_covered is empty → pruned
    candidates = propose_candidates(
        obs[:1], r_Q[:1], tv[:1], ti[:1], hg[:1],
        obs[:1], r_Q[:1],
        obs[:1], r_Q[:1],
        base, alpha_values=[1.2], metric='angular',
    )
    assert candidates == [], f"Expected empty list; got {len(candidates)} candidates"
    print("  Passed.")

#-------------------------------------------------------------------------------

def test_propose_candidates_deterministic():
    """Same inputs always produce the same output."""
    print("Test: deterministic output")
    base, obs, test = _make_dataset()
    r_Q, tv, ti, hg = _gt_for_queries(obs, base, K=3)
    test_r_t = np.zeros(len(test), dtype=np.float32)

    kwargs = dict(alpha_values=[1.0, 1.2], metric='angular')
    c1 = propose_candidates(obs, r_Q, tv, ti, hg, obs, r_Q, test, test_r_t, base, **kwargs)
    c2 = propose_candidates(obs, r_Q, tv, ti, hg, obs, r_Q, test, test_r_t, base, **kwargs)

    assert len(c1) == len(c2), "Different lengths on repeat call"
    for a, b in zip(c1, c2):
        assert a.R_c == b.R_c, "R_c differs on repeat call"
        assert a.covered_train_indices == b.covered_train_indices
        assert a.covered_test_indices == b.covered_test_indices
    print("  Passed.")

#-------------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Testing population.py")
    print("=" * 70)

    test_candidate_entry_to_cache_entry()
    test_propose_candidates_coverage_correctness()
    test_propose_candidates_alpha_scaling()
    test_propose_source_query_excluded()
    test_propose_empty_covered_pruned()
    test_propose_candidates_deterministic()

    print("\n" + "=" * 70)
    print("All tests passed.")
    print("=" * 70)

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
