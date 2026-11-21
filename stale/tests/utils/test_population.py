#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from utils.base import MainMemory, angular_distance, euclidean_distance
from utils.base.population import CandidateEntry

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

def _mask(indices: set, n_domain: int) -> np.ndarray:
    """Boolean coverage array of length n_domain from a set of indices,
    matching the covered_train_indices/covered_test_indices representation."""
    m = np.zeros(n_domain, dtype=bool)
    for i in indices:
        m[i] = True
    return m

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
    """to_cache_entry() uses r_c (not r_Q) as the radius and builds the stored
    index set from R_c (the complete within-r_c index set), not top_k_indices.

    Entries are index-only (no private top_k_vectors/top_k_matrix copy);
    distance computation and result-vector lookup go through a shared
    base_hat/base_vectors matrix at hit-check time instead."""
    base, obs, test = _make_dataset()
    r_Q, tv, ti, hg = _gt_for_queries(obs[:1], base)

    R_c_indices = [0, 1, 2]
    R_c = _mask(set(R_c_indices), 30)
    cand = CandidateEntry(
        query=obs[0],
        r_Q=float(r_Q[0]),
        alpha=1.2,
        r_c=1.2 * float(r_Q[0]),
        R_c=R_c,
        rc_size=int(R_c.sum()),
        covered_train_indices=_mask({0, 1}, 10),
        covered_test_indices=_mask({3, 7}, 10),
        top_k_vectors=tv[0],
        top_k_indices=ti[0],
        half_gap=float(hg[0]),
    )
    entry = cand.to_cache_entry()

    assert abs(entry.r_Q - cand.r_c) < 1e-7, "r_Q should equal r_c"
    assert abs(entry.half_gap - cand.half_gap) < 1e-7, "half_gap should be preserved"
    assert list(entry.top_k_indices) == R_c_indices, "top_k_indices should come from R_c"
    assert entry.top_k_vectors == [], "index-only entries carry no private vector copy"
    assert entry.top_k_matrix is None, "index-only entries carry no private matrix copy"


def test_candidate_entry_to_cache_entry_requires_rc():
    """to_cache_entry() raises if R_c was not computed (store_rc=False)."""
    base, obs, test = _make_dataset()
    r_Q, tv, ti, hg = _gt_for_queries(obs[:1], base)

    cand = CandidateEntry(
        query=obs[0],
        r_Q=float(r_Q[0]),
        alpha=1.2,
        r_c=1.2 * float(r_Q[0]),
        R_c=None,
        rc_size=0,
        covered_train_indices=_mask({0, 1}, 10),
        covered_test_indices=_mask({3, 7}, 10),
        top_k_vectors=tv[0],
        top_k_indices=ti[0],
        half_gap=float(hg[0]),
    )
    try:
        cand.to_cache_entry()
        assert False, "expected ValueError when R_c is None"
    except ValueError:
        pass

def test_to_cache_entry_cig_soundness():
    """When R_c is the complete within-r_c set, certified indices satisfy CIG.

    Covers the case where r_c covers far more base vectors than K_test: since
    to_cache_entry stores the full R_c (not just K_test vectors), all certified
    indices are closer to the test query than any non-stored base vector, so
    recall >= sigma is guaranteed on every hit.

    Uses lemma1_no_union_sigma against an index-only CacheEntry, threading the
    shared base_hat/base_vectors matrices the way CacheSimulator does.
    """
    from utils.base.lemmas_sigma import lemma1_no_union_sigma
    from utils.base.distance_metrics import angular_distance

    rng = np.random.RandomState(12345)
    dim = 16
    n_base = 100
    K_TEST = 5

    base = rng.randn(n_base, dim).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    base_hat = base  # already unit-normalized above
    Q = rng.randn(dim).astype(np.float32)
    Q /= np.linalg.norm(Q)

    dists_to_Q = np.array([angular_distance(Q, base[j]) for j in range(n_base)])
    # r_c large enough to cover the entire base (reproduces the extreme real-data case)
    r_c = float(np.max(dists_to_Q)) + 0.01
    R_c_mask = dists_to_Q <= r_c
    assert int(R_c_mask.sum()) == n_base

    old_top_k_indices = list(np.argsort(dists_to_Q)[:K_TEST])
    cand = CandidateEntry(
        query=Q, r_Q=float(dists_to_Q[old_top_k_indices[-1]]),
        alpha=None, r_c=r_c, R_c=R_c_mask, rc_size=int(R_c_mask.sum()),
        covered_train_indices=np.zeros(0, dtype=bool), covered_test_indices=np.zeros(0, dtype=bool),
        top_k_vectors=[base[i] for i in old_top_k_indices],
        top_k_indices=old_top_k_indices, half_gap=0.0,
    )

    from utils.base import KVCache
    kv = KVCache(metric='angular')
    entry = cand.to_cache_entry()
    kv.add_entry(entry.query, entry.top_k_vectors, entry.r_Q, entry.half_gap, entry.top_k_indices)
    assert len(entry.top_k_indices) == n_base

    cache_entries = kv.get_all_entries()
    test_q = Q + rng.randn(dim).astype(np.float32) * 0.1
    test_q /= np.linalg.norm(test_q)
    dists_to_test_q = np.array([angular_distance(test_q, base[j]) for j in range(n_base)])
    true_topk = set(np.argsort(dists_to_test_q)[:K_TEST].tolist())

    for sigma in [0.70, 0.90, 1.00]:
        _, is_hit, meta = lemma1_no_union_sigma(
            test_q, cache_entries, K_TEST, sigma, 'angular',
            base_hat=base_hat, base_vectors=base,
        )
        if not is_hit:
            continue
        certified = set(meta['certified_indices'])
        recall = len(certified & true_topk) / K_TEST
        assert recall >= sigma - 1e-9, (
            f"sigma={sigma}: recall={recall:.4f} < sigma, CIG soundness violated"
        )


#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
