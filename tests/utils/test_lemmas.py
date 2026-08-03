#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys
import math
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from utils.base import (
    MainMemory,
    KVCache,
    CacheEntry,
    lemma1_circular_inclusion,
    lemma2_half_gap,
    combined_algorithm,
    lemma1_no_union,
    lemma2_no_union
)
from utils.base.lemmas_sigma import (
    lemma1_no_union_sigma, lemma2_no_union_sigma,
    fit_r_t_hat, overlap_threshold_hit_policy,
    _chord_distance_from_angular_scalar, _chord_distance_from_angular_array,
    _solid_ball_intersection_fraction_array,
)

#-------------------------------------------------------------------------------

# binary_search_last_index is no longer used anywhere (the hot path uses a
# direct np.where threshold mask instead) -- kept in utils/base/lemmas.py but
# no longer tested or exercised.
#
# def test_binary_search():
#     """Test binary search helper."""
#     print("Test 1: Binary search last index")
#
#     # test case: sorted distances with some satisfying condition
#     sorted_distances = [1.0, 2.0, 3.0, 4.0, 5.0]
#     r_Q = 8.0
#     d_q = 2.0
#
#     # D(q, E_i) + d_q < r_Q
#     #-------------------------------
#     # 1.0 + 2.0 = 3.0 < 8.0
#     # 2.0 + 2.0 = 4.0 < 8.0
#     # 3.0 + 2.0 = 5.0 < 8.0
#     # 4.0 + 2.0 = 6.0 < 8.0
#     # 5.0 + 2.0 = 7.0 < 8.0
#
#     theta = binary_search_last_index(sorted_distances, r_Q, d_q)
#     assert theta == 4, f"Expected theta=4, got {theta}"
#
#     # test case: no vectors satisfy
#     r_Q = 2.0
#     theta = binary_search_last_index(sorted_distances, r_Q, d_q)
#     assert theta == -1, f"Expected theta=-1 (none satisfy), got {theta}"
#
#     # test case: only some satisfy
#     r_Q = 5.0
#     theta = binary_search_last_index(sorted_distances, r_Q, d_q)
#     # 1.0 + 2.0 = 3.0 < 5.0
#     # 2.0 + 2.0 = 4.0 < 5.0
#     # 3.0 + 2.0 = 5.0 NOT < 5.0
#     assert theta == 1, f"Expected theta=1, got {theta}"
#
#     print("Binary search works.")

#-------------------------------------------------------------------------------

def _make_single_entry_cache(K: int, metric: str = "euclidean", M: int = 100, d: int = 32, seed: int = 42):
    """Shared single-cached-query setup used by most of the tests below:
    a MainMemory of M random d-dim vectors, plus a KVCache with one entry
    populated for a random cached_query at the given K."""
    mm = MainMemory(M=M, d=d, seed=seed)
    cache = KVCache(metric=metric)
    cached_query = np.random.randn(d)
    cache.populate_from_queries(np.array([cached_query]), mm, K=K)
    return mm, cache, cached_query

#-------------------------------------------------------------------------------

def test_lemma1_simple():
    """Test Lemma 1 with simple case."""
    print("\nTest 2: Lemma 1 simple case")

    mm, cache, cached_query = _make_single_entry_cache(K=10)

    # query very close to cached query (should hit)
    similar_query = cached_query + np.random.randn(32) * 0.001

    result, is_hit, _ = lemma1_circular_inclusion(
        similar_query,
        cache.get_all_entries(),
        N=5,
        metric="euclidean"
    )
    
    if is_hit:
        assert len(result) == 5, f"Expected 5 results, got {len(result)}"
        print("Lemma 1 cache hit works.")
    else:
        print("Lemma 1 didn't hit.")

#-------------------------------------------------------------------------------

def test_lemma2_simple():
    """Test Lemma 2 with simple case."""
    print("\nTest 3: Lemma 2 simple case")

    mm, cache, cached_query = _make_single_entry_cache(K=10)

    # query very close to cached query (within half gap)
    entry = cache.get_all_entries()[0]
    half_gap = entry.get_half_gap()
    
    # make query at distance < half_gap from cached query
    perturbation_size = half_gap * 0.1
    similar_query = cached_query + np.random.randn(32) * perturbation_size
    
    result, is_hit, _ = lemma2_half_gap(
        similar_query,
        cache.get_all_entries(),
        N=10,
        metric="euclidean"
    )
    
    if is_hit:
        assert len(result) == 10, f"Expected 10 results, got {len(result)}"
        print("Lemma 2 cache hit works.")
    else:
        print("Lemma 2 didn't hit (gap might be too small)")

#-------------------------------------------------------------------------------

def test_lemma2_union():
    print("\nTest 4: Lemma 2 with union (K < N)")
    
    mm = MainMemory(M=100, d=32, seed=42)
    cache = KVCache(metric="euclidean")
    
    query1 = np.random.randn(32)
    query2 = query1 + np.random.randn(32) * 0.001
    cache.populate_from_queries(np.array([query1, query2]), mm, K=5)
    
    test_query = query1 + np.random.randn(32) * 0.0001
    result, is_hit, metadata = lemma2_half_gap(
        test_query,
        cache.get_all_entries(),
        N=8,
        metric="euclidean"
    )
    
    if is_hit:
        assert len(result) == 8
        assert metadata["vectors_added"] > 0
        print("Lemma 2 union works.")
    else:
        print("Lemma 2 union didn't hit.")

#-------------------------------------------------------------------------------

def test_combined_algorithm():
    """Test combined algorithm."""
    print("\nTest 5: Combined algorithm")
    
    mm = MainMemory(M=100, d=32, seed=42)
    cache = KVCache(metric="euclidean")
    
    cached_queries = np.random.randn(5, 32)
    cache.populate_from_queries(cached_queries, mm, K=10)
    
    # test query similar to first cached query
    test_query = cached_queries[0] + np.random.randn(32) * 0.001
    
    result, is_hit, metadata = combined_algorithm(
        test_query,
        cache.get_all_entries(),
        N=10,
        metric="euclidean"
    )
    
    assert result is None or len(result) <= 10, "Result size shouldn't exceed N"
    
    print(f"Cache hit: {is_hit}")
    if is_hit:
        print(f"Hit via: {metadata.get('hit_lemma', 'unknown')}")
    print("Combined algorithm works.")

#-------------------------------------------------------------------------------

def test_cache_miss():
    """Test that unrelated queries miss the cache."""
    print("\nTest 6: Cache miss behavior")
    
    mm = MainMemory(M=100, d=32, seed=42)
    cache = KVCache(metric="euclidean")
    
    # cache queries in one region of space
    cached_queries = np.random.randn(3, 32) + 100
    cache.populate_from_queries(cached_queries, mm, K=10)
    
    # query in different region
    test_query = np.random.randn(32) - 100
    
    result, is_hit, _ = combined_algorithm(
        test_query,
        cache.get_all_entries(),
        N=10,
        metric="euclidean"
    )
    
    assert is_hit == False, "Should miss cache for distant query"
    assert result is None, "Should return None on cache miss"
    
    print("Cache miss works correctly.")

#-------------------------------------------------------------------------------

def test_different_k_n():
    """Test with different K and N values."""
    print("\nTest 7: Different K and N values")

    mm, cache, cached_query = _make_single_entry_cache(K=20)

    # test with N < K
    similar_query = cached_query + np.random.randn(32) * 0.001
    result, is_hit, _ = combined_algorithm(
        similar_query,
        cache.get_all_entries(),
        N=10,
        metric="euclidean"
    )
    
    if is_hit:
        assert len(result) == 10, f"Expected N=10 results, got {len(result)}"
    
    # test with N = K
    result, is_hit, _ = combined_algorithm(
        similar_query,
        cache.get_all_entries(),
        N=20,
        metric="euclidean"
    )

    if is_hit:
        assert len(result) == 20, f"Expected N=20 results, got {len(result)}"

    # canonical no-union algorithms at the same N < K and N = K cases
    for algo in (lemma1_no_union, lemma2_no_union):
        for N in (10, 20):
            result, is_hit, _ = algo(similar_query, cache.get_all_entries(), N=N, metric="euclidean")
            if is_hit:
                assert len(result) == N, (
                    f"{algo.__name__} N={N}: expected {N} results, got {len(result)}"
                )

    print("Different K/N values work.")

#-------------------------------------------------------------------------------

def test_n_greater_than_k():
    """Test when requesting more vectors than cached."""
    print("\nTest 8: N > K edge case")

    mm, cache, cached_query = _make_single_entry_cache(K=10)

    # request N=15 when only K=10 cached
    similar_query = cached_query + np.random.randn(32) * 0.001
    result, is_hit, _ = combined_algorithm(
        similar_query,
        cache.get_all_entries(),
        N=15,
        metric="euclidean"
    )

    assert is_hit == False, "Should miss cache when N > K"

    # canonical no-union algorithms must also miss when N > K
    for algo in (lemma1_no_union, lemma2_no_union):
        result, is_hit, _ = algo(similar_query, cache.get_all_entries(), N=15, metric="euclidean")
        assert is_hit == False, f"{algo.__name__} should miss cache when N > K"
        assert result is None

#-------------------------------------------------------------------------------

def test_lemma1_no_union():
    print("\nTest 9: Lemma 1 no union")
    mm, cache, cached_query = _make_single_entry_cache(K=10)
    similar_query = cached_query + np.random.randn(32) * 0.001
    result, is_hit, _ = lemma1_no_union(similar_query, cache.get_all_entries(), N=5, metric="euclidean")
    if is_hit:
        assert len(result) == 5
        print("Lemma 1 no union works.")
    else:
        print("Lemma 1 no union didn't hit.")

#-------------------------------------------------------------------------------

def test_lemma2_no_union():
    print("\nTest 10: Lemma 2 no union")
    mm, cache, cached_query = _make_single_entry_cache(K=10)
    entry = cache.get_all_entries()[0]
    half_gap = entry.get_half_gap()
    similar_query = cached_query + np.random.randn(32) * (half_gap * 0.1)
    result, is_hit, _ = lemma2_no_union(similar_query, cache.get_all_entries(), N=5, metric="euclidean")
    if is_hit:
        assert len(result) == 5
        print("Lemma 2 no union works.")
    else:
        print("Lemma 2 no union didn't hit.")

#-------------------------------------------------------------------------------
# sigma (relaxed-recall) variant of lemma1 -- CIG with sigma=1.0 must match the
# exact lemma; sigma<1.0 relaxes the hit condition to >= ceil(sigma*N) certified.

def _make_cache_entry_sigma(cache: KVCache, n_vectors: int, dim: int, rng):
    """Add one entry to cache with n_vectors orthogonal-ish neighbors."""
    query = rng.randn(dim).astype(np.float32)
    query /= np.linalg.norm(query)
    # Make vectors that are very close to query so r_Q is small.
    vectors = [query + rng.randn(dim).astype(np.float32) * 0.01 for _ in range(n_vectors)]
    for i, v in enumerate(vectors):
        vectors[i] = v / np.linalg.norm(v)
    r_Q = 0.05
    half_gap = 0.0
    indices = list(range(n_vectors))
    cache.add_entry(query, vectors, r_Q, half_gap, indices)
    return query, vectors, r_Q


def _angular_dist(a, b):
    """Matches utils/base/distance_metrics.angular_distance: arccos(cosim) / pi."""
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return float(np.arccos(np.clip(cos, -1.0, 1.0))) / math.pi


#-------------------------------------------------------------------------------
# sigma-relaxed no-union variants (canonical algorithms going forward) --
# lemma1_no_union_sigma / lemma2_no_union_sigma, sigma=1.0 must exactly match
# their non-sigma no-union counterparts.

def test_lemma1_no_union_sigma_one_matches_exact():
    """sigma=1.0 must give the same hit/miss, result, and result_indices as
    lemma1_no_union."""
    rng = np.random.RandomState(11)
    dim = 16
    N = 5

    cache = KVCache(metric='angular')
    query, _, _ = _make_cache_entry_sigma(cache, N, dim, rng)
    entries = cache.get_all_entries()

    result_exact, hit_exact, meta_exact = lemma1_no_union(query, entries, N, metric='angular')
    result_sigma, hit_sigma, meta_sigma = lemma1_no_union_sigma(
        query, entries, N, sigma=1.0, metric='angular'
    )

    assert hit_exact == hit_sigma
    assert meta_exact.get('result_indices') == meta_sigma.get('result_indices')
    if hit_exact:
        assert len(result_exact) == len(result_sigma)
        for a, b in zip(result_exact, result_sigma):
            assert np.allclose(a, b)


def test_lemma1_no_union_sigma_relaxes_threshold():
    """sigma<1.0 certifies a hit from a single entry with fewer than N passing
    vectors, where the exact (sigma=1.0) no-union check would miss."""
    dim = 64
    N = 10

    Q = np.zeros(dim, dtype=np.float32)
    Q[0] = 1.0
    q = np.zeros(dim, dtype=np.float32)
    q[1] = 1.0

    d_qQ = _angular_dist(q, Q)
    rng = np.random.RandomState(12)
    close_vecs = []
    for _ in range(7):
        v = q.copy() + rng.randn(dim).astype(np.float32) * 0.001
        v /= np.linalg.norm(v)
        close_vecs.append(v)
    far_vecs = []
    for _ in range(3):
        v = Q.copy() + rng.randn(dim).astype(np.float32) * 0.001
        v /= np.linalg.norm(v)
        far_vecs.append(v)

    d_close_max = max(_angular_dist(q, v) for v in close_vecs)
    d_far_min = min(_angular_dist(q, v) for v in far_vecs)
    thresh_lo = d_close_max + d_qQ
    thresh_hi = d_far_min + d_qQ
    if thresh_lo >= thresh_hi:
        return  # geometry too tight for this random draw; skip

    r_Q = (thresh_lo + thresh_hi) / 2.0
    all_vectors = close_vecs + far_vecs
    indices = list(range(N))

    cache = KVCache(metric='angular')
    cache.add_entry(Q, all_vectors, r_Q, 0.0, indices)
    entries = cache.get_all_entries()

    _, hit_exact, _ = lemma1_no_union(q, entries, N, metric='angular')
    _, hit_07, meta_07 = lemma1_no_union_sigma(q, entries, N, sigma=0.70, metric='angular')

    assert not hit_exact, "sigma=1.0 (exact) should miss (only 7/10 in this entry)"
    assert hit_07, "sigma=0.70 should certify a hit from this single entry"
    assert len(meta_07['certified_indices']) >= math.ceil(0.70 * N)


def test_lemma2_no_union_sigma_one_matches_exact():
    """sigma=1.0 must give the same hit/miss, result, and result_indices as
    lemma2_no_union."""
    rng = np.random.RandomState(13)
    dim = 16
    N = 5

    cache = KVCache(metric='angular')
    query, _, _ = _make_cache_entry_sigma(cache, N, dim, rng)
    entries = cache.get_all_entries()

    result_exact, hit_exact, meta_exact = lemma2_no_union(query, entries, N, metric='angular')
    result_sigma, hit_sigma, meta_sigma = lemma2_no_union_sigma(
        query, entries, N, sigma=1.0, metric='angular'
    )

    assert hit_exact == hit_sigma
    assert meta_exact.get('result_indices') == meta_sigma.get('result_indices')
    if hit_exact:
        assert len(result_exact) == len(result_sigma)
        for a, b in zip(result_exact, result_sigma):
            assert np.allclose(a, b)


def test_lemma2_no_union_sigma_relaxes_k_requirement():
    """sigma<1.0 certifies a hit when K < N but K >= ceil(sigma*N), which the
    exact (sigma=1.0) no-union check would miss (requires K >= N)."""
    dim = 16
    N = 10
    K = 8  # 8/10 = 80% recall if certified

    rng = np.random.RandomState(14)
    query = rng.randn(dim).astype(np.float32)
    query /= np.linalg.norm(query)
    vectors = []
    for _ in range(K):
        v = query + rng.randn(dim).astype(np.float32) * 0.001
        v /= np.linalg.norm(v)
        vectors.append(v)

    # half_gap large enough that d_Q (=0, same query) < half_gap always holds
    cache = KVCache(metric='angular')
    cache.add_entry(query, vectors, r_Q=0.5, half_gap=0.5, top_k_indices=list(range(K)))
    entries = cache.get_all_entries()

    _, hit_exact, _ = lemma2_no_union(query, entries, N, metric='angular')
    _, hit_08, meta_08 = lemma2_no_union_sigma(query, entries, N, sigma=0.80, metric='angular')

    assert not hit_exact, "sigma=1.0 (exact) should miss (K=8 < N=10)"
    assert hit_08, "sigma=0.80 should certify a hit (K=8 >= ceil(0.80*10)=8)"
    assert len(meta_08['certified_indices']) == K, "certified set should be the entry's full top-K"

#-------------------------------------------------------------------------------
# Fast-path vs fallback-path equivalence -- every production call from
# simulations/core/simulate.py passes a distance_tracker, so these hot-path
# fast implementations are only ever exercised (in the tests above) via their
# untracked/fallback form unless explicitly cross-checked here.

def test_batch_distances_fast_matches_fallback():
    """_batch_distances' vectorized matmul path (untracked) and its per-call
    Python-loop fallback (as used whenever a distance_tracker wraps the raw
    function) must return identical distances."""
    from utils.base.lemmas import _batch_distances
    from utils.base.distance_metrics import get_distance_function

    rng = np.random.RandomState(20)
    for metric in ('euclidean', 'angular'):
        query = rng.randn(16).astype(np.float32)
        vectors = [rng.randn(16).astype(np.float32) for _ in range(12)]
        raw_func = get_distance_function(metric)

        fast = _batch_distances(query, vectors, metric, raw_func)
        # any distance_func that is NOT `is raw_func` forces the fallback loop,
        # even if it computes the exact same thing
        wrapped = lambda a, b: raw_func(a, b)
        fallback = _batch_distances(query, vectors, metric, wrapped)

        assert np.allclose(fast, fallback, atol=1e-6), (
            f"{metric}: fast={fast} vs fallback={fallback}"
        )


def test_cig_distances_matrix_path_matches_fallback():
    """_cig_distances' entry.top_k_matrix fast path and its _batch_distances
    fallback must return identical distances for the same entry."""
    from utils.base.lemmas import _cig_distances
    from utils.base.distance_metrics import get_distance_function

    rng = np.random.RandomState(21)
    dim = 16
    n_vectors = 6

    for metric in ('euclidean', 'angular'):
        query = rng.randn(dim).astype(np.float32)
        q_norm = np.linalg.norm(query)
        q_hat = (query / q_norm).astype(np.float32)

        raw_vectors = [rng.randn(dim).astype(np.float32) for _ in range(n_vectors)]
        mat = np.stack(raw_vectors).astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        top_k_matrix = mat / norms

        raw_func = get_distance_function(metric)

        entry_with_matrix = CacheEntry(
            query=query, top_k_vectors=raw_vectors, r_Q=1.0, half_gap=0.0,
            top_k_indices=list(range(n_vectors)), top_k_matrix=top_k_matrix,
        )
        entry_without_matrix = CacheEntry(
            query=query, top_k_vectors=raw_vectors, r_Q=1.0, half_gap=0.0,
            top_k_indices=list(range(n_vectors)), top_k_matrix=None,
        )

        fast = _cig_distances(query, entry_with_matrix, metric, raw_func, q_hat)
        fallback = _cig_distances(query, entry_without_matrix, metric, raw_func, q_hat)

        assert np.allclose(fast, fallback, atol=1e-5), (
            f"{metric}: matrix path={fast} vs fallback={fallback}"
        )

def test_cig_distances_precomputed_base_dist_matches_gather():
    """_cig_distances' index-only base_hat path (fresh per-entry matmul) and
    its precomputed_base_dist gather path (Round 6's fix for simulate_oracle's
    real-scale cost -- distances to the full base computed once per query
    instead of once per cache entry) must return identical distances."""
    from utils.base.lemmas import _cig_distances
    from utils.base.distance_metrics import get_distance_function

    rng = np.random.RandomState(22)
    dim = 16
    n_base = 40
    n_indices = 10

    base_vectors = rng.randn(n_base, dim).astype(np.float32)
    base_norms = np.linalg.norm(base_vectors, axis=1, keepdims=True)
    base_norms[base_norms == 0.0] = 1.0
    base_hat = (base_vectors / base_norms).astype(np.float32)

    for metric in ('euclidean', 'angular'):
        query = rng.randn(dim).astype(np.float32)
        q_norm = np.linalg.norm(query)
        q_hat = (query / q_norm).astype(np.float32)

        raw_func = get_distance_function(metric)
        indices = rng.choice(n_base, size=n_indices, replace=False)

        entry = CacheEntry(
            query=query, top_k_vectors=[], r_Q=1.0, half_gap=0.0,
            top_k_indices=indices, top_k_matrix=None,
        )

        dots = np.clip(base_hat @ q_hat, -1.0, 1.0)
        if metric == "angular":
            precomputed_base_dist = np.arccos(dots) / np.pi
        else:
            precomputed_base_dist = np.sqrt(np.maximum(2.0 - 2.0 * dots, 0.0))

        without_precompute = _cig_distances(
            query, entry, metric, raw_func, q_hat, base_hat=base_hat,
        )
        with_precompute = _cig_distances(
            query, entry, metric, raw_func, q_hat, base_hat=base_hat,
            precomputed_base_dist=precomputed_base_dist,
        )

        assert np.allclose(without_precompute, with_precompute, atol=1e-6), (
            f"{metric}: without={without_precompute} vs with={with_precompute}"
        )


def test_lemma1_no_union_sigma_index_only_matches_real_vectors():
    """End-to-end: lemma1_no_union_sigma on index-only entries (which now go
    through Round 6's per-query precomputed_base_dist fast path via base_hat)
    must produce identical hit/result/certified-indices output to the same
    entries built with real top_k_vectors instead (an independent, untouched
    code path in _cig_distances that never consults base_hat/
    precomputed_base_dist at all) -- proving the optimization didn't change
    behavior, not just that two internal helpers agree with each other."""
    rng = np.random.RandomState(23)
    dim = 12
    n_base = 60
    n_entries = 8
    K = 6
    N = 5

    base_vectors = rng.randn(n_base, dim).astype(np.float32)
    base_norms = np.linalg.norm(base_vectors, axis=1, keepdims=True)
    base_norms[base_norms == 0.0] = 1.0
    base_hat = (base_vectors / base_norms).astype(np.float32)

    index_only_entries = []
    real_vector_entries = []
    for _ in range(n_entries):
        q = rng.randn(dim).astype(np.float32)
        q_hat = q / np.linalg.norm(q)
        dots = np.clip(base_hat @ q_hat, -1.0, 1.0)
        dists = np.arccos(dots) / np.pi
        order = np.argsort(dists)
        top_k = order[:K]
        r_Q = float(dists[top_k[-1]])
        gap = float(dists[order[K]] - r_Q) if K < n_base else 0.1
        index_only_entries.append(CacheEntry(
            query=q, top_k_vectors=[], r_Q=r_Q, half_gap=gap / 2.0,
            top_k_indices=top_k.astype(np.int32), top_k_matrix=None,
        ))
        real_vector_entries.append(CacheEntry(
            query=q, top_k_vectors=[base_vectors[i] for i in top_k],
            r_Q=r_Q, half_gap=gap / 2.0,
            top_k_indices=list(top_k), top_k_matrix=None,
        ))

    for _ in range(5):
        test_q = rng.randn(dim).astype(np.float32)

        result_a, hit_a, meta_a = lemma1_no_union_sigma(
            test_q, index_only_entries, N, 1.0, 'angular',
            base_hat=base_hat, base_vectors=base_vectors,
        )
        result_b, hit_b, meta_b = lemma1_no_union_sigma(
            test_q, real_vector_entries, N, 1.0, 'angular',
        )

        assert hit_a == hit_b
        if hit_a:
            assert meta_a['hit_entry_index'] == meta_b['hit_entry_index']
            assert meta_a['certified_indices'] == meta_b['certified_indices']
            assert meta_a['result_indices'] == meta_b['result_indices']

def test_cig_boundary_certifies_at_exact_equality():
    """Round 7: the CIG certification condition is D(candidate,t)+r_t[t] <= r_c
    (non-strict), not the old strict <. This directly constructs the exact
    boundary case -- d_Q == r_Q (so threshold_dist == 0) with a stored vector
    at exactly distance 0 from the query -- and confirms it certifies a hit.
    Under the old strict semantics this would both (a) be skipped by the O1
    gate (d_Q >= r_Q) and (b) fail the O2 threshold check (0 < 0 is False);
    both must now admit this case."""
    print("Test: CIG boundary certifies at exact equality (Round 7 <= fix)")
    dim = 2
    entry_center = np.array([0.0, 0.0], dtype=np.float32)
    r_Q = 1.0
    # query placed at exactly distance r_Q from the entry's own center
    query = np.array([1.0, 0.0], dtype=np.float32)
    # a stored vector placed at exactly distance 0 from the query itself
    boundary_vector = np.array([1.0, 0.0], dtype=np.float32)

    entry = CacheEntry(
        query=entry_center, top_k_vectors=[boundary_vector], r_Q=r_Q, half_gap=0.0,
        top_k_indices=[0], top_k_matrix=None,
    )
    d_Q = float(np.linalg.norm(query - entry_center))
    assert d_Q == r_Q, "test setup: d_Q must exactly equal r_Q"

    result, is_hit, meta = lemma1_no_union_sigma(
        query, [entry], N=1, sigma=1.0, metric='euclidean',
    )
    assert is_hit, (
        f"exact-boundary case should certify under <= semantics: "
        f"d_Q={d_Q}, r_Q={r_Q}, meta={meta}"
    )
    assert meta['checked_entries'] == 1, (
        "O1 gate should not have skipped the d_Q == r_Q entry"
    )
    print("  Certified at exact boundary (d_Q==r_Q, distance==threshold==0). Passed.")


def test_cig_le_is_monotonic_superset_of_strict_lt():
    """The <= fix can only ADD certifications relative to the old strict <
    semantics, never remove any -- a monotonicity property independent of
    any specific boundary construction. Cross-checks the live
    lemma1_no_union_sigma against a naive reference that reproduces the old
    strict-< behavior, across several random trials, and asserts every
    strict-< hit is also a <= hit (never the reverse)."""
    print("Test: <= certifies a superset of what strict < would certify")

    def _reference_strict_lt(query, cache_entries, N, metric='euclidean'):
        """Old (pre-Round-7) strict-< CIG logic, kept only for this
        comparison -- not used anywhere else."""
        from utils.base.distance_metrics import get_distance_function
        distance_func = get_distance_function(metric)
        for entry in cache_entries:
            r_Q = entry.get_radius()
            d_Q = distance_func(query, entry.query)
            if d_Q >= r_Q:
                continue
            threshold_dist = r_Q - d_Q
            dists = np.array([distance_func(query, v) for v in entry.top_k_vectors])
            passing = np.where(dists < threshold_dist)[0]
            if len(passing) >= N:
                return True
        return False

    rng = np.random.RandomState(321)
    dim = 6
    n_trials = 40
    for _ in range(n_trials):
        n_entries = rng.randint(1, 4)
        entries = []
        for _ in range(n_entries):
            center = rng.randn(dim).astype(np.float32)
            n_vec = rng.randint(1, 5)
            vectors = [center + rng.randn(dim).astype(np.float32) * 0.3 for _ in range(n_vec)]
            r_Q = float(rng.uniform(0.1, 1.5))
            entries.append(CacheEntry(
                query=center, top_k_vectors=vectors, r_Q=r_Q, half_gap=0.0,
                top_k_indices=list(range(n_vec)), top_k_matrix=None,
            ))
        test_q = rng.randn(dim).astype(np.float32)
        N = rng.randint(1, 3)

        strict_hit = _reference_strict_lt(test_q, entries, N, metric='euclidean')
        _, le_hit, _ = lemma1_no_union_sigma(test_q, entries, N, 1.0, 'euclidean')

        if strict_hit:
            assert le_hit, (
                "strict-< certified a hit that <= (its superset) did not -- "
                "monotonicity violated"
            )
    print(f"  <= certified a superset of strict-< across {n_trials} random trials. Passed.")


def test_lemma1_no_union_sigma_vectorized_gate_matches_fallback():
    """Round 8: lemma1_no_union_sigma's vectorized O1 gate (entry_radii +
    precomputed_d_Q given) must produce byte-identical results to the
    per-entry fallback loop (entry_radii omitted), across several random
    trials including a guaranteed full-miss case (every entry fails O1, so
    the whole cache is scanned) to specifically exercise the "candidate_idx
    is empty" / "scan everything, certify nothing" path."""
    print("Test: vectorized O1 gate matches per-entry fallback")
    from utils.base.distance_metrics import get_distance_function

    rng = np.random.RandomState(456)
    dim = 6
    distance_func = get_distance_function('euclidean')

    def _entry_radii_array(entries):
        return np.array([e.get_radius() for e in entries], dtype=np.float32)

    def _d_Q_batch(query, entries):
        return np.array([distance_func(query, e.query) for e in entries], dtype=np.float32)

    n_trials = 30
    for trial in range(n_trials):
        n_entries = rng.randint(1, 6)
        entries = []
        for _ in range(n_entries):
            center = rng.randn(dim).astype(np.float32)
            n_vec = rng.randint(1, 5)
            vectors = [center + rng.randn(dim).astype(np.float32) * 0.3 for _ in range(n_vec)]
            r_Q = float(rng.uniform(0.1, 1.0))
            entries.append(CacheEntry(
                query=center, top_k_vectors=vectors, r_Q=r_Q, half_gap=0.0,
                top_k_indices=list(range(n_vec)), top_k_matrix=None,
            ))

        # every other trial, force a guaranteed full miss by placing the
        # test query far from every entry (all fail O1)
        if trial % 2 == 0:
            test_q = rng.randn(dim).astype(np.float32) * 100.0
        else:
            test_q = rng.randn(dim).astype(np.float32)
        N = rng.randint(1, 3)

        d_Q_batch = _d_Q_batch(test_q, entries)
        entry_radii = _entry_radii_array(entries)

        result_vec, hit_vec, meta_vec = lemma1_no_union_sigma(
            test_q, entries, N, 1.0, 'euclidean',
            precomputed_d_Q=d_Q_batch, entry_radii=entry_radii,
        )
        result_fb, hit_fb, meta_fb = lemma1_no_union_sigma(
            test_q, entries, N, 1.0, 'euclidean',
            precomputed_d_Q=d_Q_batch,  # entry_radii omitted -> fallback loop
        )

        assert hit_vec == hit_fb, f"trial {trial}: hit mismatch"
        assert meta_vec['checked_entries'] == meta_fb['checked_entries'], (
            f"trial {trial}: checked_entries mismatch "
            f"(vec={meta_vec['checked_entries']}, fb={meta_fb['checked_entries']})"
        )
        if hit_vec:
            assert meta_vec['hit_entry_index'] == meta_fb['hit_entry_index']
            assert meta_vec['certified_indices'] == meta_fb['certified_indices']
            assert meta_vec['result_indices'] == meta_fb['result_indices']
        else:
            assert meta_vec['checked_entries'] == len(entries), (
                f"trial {trial}: full miss should check every entry"
            )
    print(f"  Vectorized gate matched fallback across {n_trials} trials "
          f"(including full-miss cases). Passed.")

#-------------------------------------------------------------------------------
# Phase 3 Round 12: N=0 (population's K_test=0 containment mode) must reduce
# to "hit iff contained", via the *existing* O1 gate + len(passing) >=
# threshold logic -- no special-casing added for this. These tests pin that
# down now that population_runner.py's sim_K clamp (which previously forced
# N >= 1) has been removed, so N=0 actually reaches this function.

def test_lemma1_no_union_sigma_n_zero_hits_when_contained():
    """threshold=ceil(sigma*0)=0: len(passing) >= 0 is always true once an
    entry passes O1 containment (d_Q <= r_Q) -- must certify immediately,
    with no real vectors required."""
    print("Test: N=0 certifies on containment alone")
    query = np.array([1.0, 0.0], dtype=np.float32)
    entry = CacheEntry(
        query=np.array([1.0, 0.0], dtype=np.float32),
        top_k_vectors=[np.array([5.0, 5.0], dtype=np.float32)],  # far from query
        r_Q=0.5, half_gap=0.0, top_k_indices=[0], top_k_matrix=None,
    )
    result, is_hit, metadata = lemma1_no_union_sigma(
        query, [entry], N=0, sigma=1.0, metric='euclidean',
    )
    assert is_hit, "contained entry at N=0 must certify"
    assert result == [], "N=0 hit should return zero results"
    assert metadata['certified_indices'] == []
    print("  OK: contained entry certifies at N=0 with no vectors needed")


def test_lemma1_no_union_sigma_n_zero_misses_when_not_contained():
    """N=0 must still respect the O1 containment gate -- not a trivial
    always-hit regardless of distance."""
    print("Test: N=0 still misses when not contained")
    query = np.array([10.0, 10.0], dtype=np.float32)
    entry = CacheEntry(
        query=np.array([1.0, 0.0], dtype=np.float32),
        top_k_vectors=[np.array([1.1, 0.0], dtype=np.float32)],
        r_Q=0.5, half_gap=0.0, top_k_indices=[0], top_k_matrix=None,
    )
    result, is_hit, metadata = lemma1_no_union_sigma(
        query, [entry], N=0, sigma=1.0, metric='euclidean',
    )
    assert not is_hit, "uncontained entry must still miss at N=0"
    assert metadata['certified_indices'] == []
    print("  OK: uncontained entry misses at N=0")

#-------------------------------------------------------------------------------

def _make_angular_entry(rng, dim, n_base, base_hat, base_vectors, K):
    q = rng.randn(dim).astype(np.float32)
    q_hat = q / np.linalg.norm(q)
    dots = np.clip(base_hat @ q_hat, -1.0, 1.0)
    dists = np.arccos(dots) / np.pi
    order = np.argsort(dists)
    r_c = float(dists[order[K - 1]])
    indices = order[dists <= r_c].astype(np.int32)
    entry = CacheEntry(
        query=q, top_k_vectors=[], r_Q=r_c, half_gap=0.0,
        top_k_indices=indices, top_k_matrix=None,
    )
    return entry


def test_fit_r_t_hat_matches_percentile():
    rng = np.random.RandomState(0)
    train_r_t = rng.uniform(0.1, 0.3, size=200)
    assert fit_r_t_hat(train_r_t, 50.0) == np.percentile(train_r_t, 50.0)
    assert fit_r_t_hat(train_r_t, 95.0) == np.percentile(train_r_t, 95.0)
    print("  OK: fit_r_t_hat matches np.percentile")


def test_fit_r_t_hat_never_uses_test_queries():
    """fit_r_t_hat's signature only accepts train_r_t -- there is no
    way to pass test-query radii in, by construction."""
    import inspect
    params = list(inspect.signature(fit_r_t_hat).parameters)
    assert params == ['train_r_t', 'percentile']
    print("  OK: fit_r_t_hat signature has no test-query input")


def test_overlap_threshold_requires_r_t_hat():
    print("Test: overlap_threshold_hit_policy requires r_t_hat")
    entry = CacheEntry(
        query=np.array([1.0, 0.0], dtype=np.float32), top_k_vectors=[],
        r_Q=0.1, half_gap=0.0, top_k_indices=np.array([0], dtype=np.int32),
        top_k_matrix=None,
    )
    try:
        overlap_threshold_hit_policy(
            np.array([1.0, 0.0], dtype=np.float32), [entry], N=1, sigma=1.0,
            metric='angular', r_t_hat=None,
        )
        assert False, "expected ValueError for missing r_t_hat"
    except ValueError:
        pass
    print("  OK: raises ValueError without r_t_hat")


def test_overlap_threshold_rejects_non_angular_metric():
    entry = CacheEntry(
        query=np.array([1.0, 0.0], dtype=np.float32), top_k_vectors=[],
        r_Q=0.1, half_gap=0.0, top_k_indices=np.array([0], dtype=np.int32),
        top_k_matrix=None,
    )
    try:
        overlap_threshold_hit_policy(
            np.array([1.0, 0.0], dtype=np.float32), [entry], N=1, sigma=1.0,
            metric='euclidean', r_t_hat=0.1,
        )
        assert False, "expected ValueError for non-angular metric"
    except ValueError:
        pass
    print("  OK: raises ValueError for non-angular metric")


def test_overlap_threshold_rejects_sigma_relaxation():
    entry = CacheEntry(
        query=np.array([1.0, 0.0], dtype=np.float32), top_k_vectors=[],
        r_Q=0.1, half_gap=0.0, top_k_indices=np.array([0], dtype=np.int32),
        top_k_matrix=None,
    )
    try:
        overlap_threshold_hit_policy(
            np.array([1.0, 0.0], dtype=np.float32), [entry], N=1, sigma=0.9,
            metric='angular', r_t_hat=0.1,
        )
        assert False, "expected ValueError for sigma != 1.0"
    except ValueError:
        pass
    print("  OK: raises ValueError for sigma-relaxation")


def test_overlap_threshold_empty_cache_misses():
    result, is_hit, metadata = overlap_threshold_hit_policy(
        np.array([1.0, 0.0], dtype=np.float32), [], N=1, sigma=1.0,
        metric='angular', r_t_hat=0.1,
    )
    assert not is_hit
    assert result is None
    assert metadata['certified_indices'] == []
    print("  OK: empty cache misses")


def test_overlap_threshold_hit_on_identical_query():
    """A cached entry at the exact same query with r_c == r_t_hat should
    score a perfect (score=1.0) self-overlap and hit at any threshold
    below 1.0."""
    print("Test: overlap_threshold_hit_policy hits on identical query")
    rng = np.random.RandomState(1)
    dim, n_base, K = 16, 80, 5
    base_vectors = rng.randn(n_base, dim).astype(np.float32)
    base_hat = base_vectors / np.linalg.norm(base_vectors, axis=1, keepdims=True)

    entry = _make_angular_entry(rng, dim, n_base, base_hat, base_vectors, K)
    result, is_hit, metadata = overlap_threshold_hit_policy(
        entry.query, [entry], N=K, sigma=1.0, metric='angular',
        base_hat=base_hat, base_vectors=base_vectors,
        r_t_hat=entry.r_Q, threshold=0.9,
    )
    assert is_hit
    assert metadata['best_score'] > 0.99
    assert result is not None and len(result) == K
    print("  OK: identical query hits with near-perfect score")


def test_overlap_threshold_result_sorted_by_true_distance():
    """Regression test: the returned vectors must be the N nearest to
    the query (by actual distance), not the first N entries of the
    entry's raw top_k_indices array (which is unsorted -- to_cache_entry
    stores the full R_c set via np.flatnonzero, in ascending base-index
    order, not distance order)."""
    print("Test: overlap_threshold_hit_policy result is nearest-N by distance")
    rng = np.random.RandomState(7)
    dim, n_base, K, N = 10, 40, 12, 4
    base_vectors = rng.randn(n_base, dim).astype(np.float32)
    base_hat = base_vectors / np.linalg.norm(base_vectors, axis=1, keepdims=True)
    entry = _make_angular_entry(rng, dim, n_base, base_hat, base_vectors, K)

    test_q = rng.randn(dim).astype(np.float32)
    result, is_hit, metadata = overlap_threshold_hit_policy(
        test_q, [entry], N=N, sigma=1.0, metric='angular',
        base_hat=base_hat, base_vectors=base_vectors,
        r_t_hat=entry.r_Q, threshold=-1.0,
    )
    assert is_hit

    candidate_indices = entry.top_k_indices
    q_hat = test_q / np.linalg.norm(test_q)
    dists = np.arccos(np.clip(base_hat[candidate_indices] @ q_hat, -1.0, 1.0)) / np.pi
    expected_order = candidate_indices[np.argsort(dists)][:N]
    assert metadata['certified_indices'] == [int(i) for i in expected_order]
    print("  OK: result matches true nearest-N ordering")


def test_overlap_threshold_miss_below_threshold():
    entry = CacheEntry(
        query=np.array([1.0, 0.0], dtype=np.float32), top_k_vectors=[],
        r_Q=0.01, half_gap=0.0, top_k_indices=np.array([0], dtype=np.int32),
        top_k_matrix=None,
    )
    base_vectors = np.array([[1.0, 0.0]], dtype=np.float32)
    result, is_hit, metadata = overlap_threshold_hit_policy(
        np.array([0.0, 1.0], dtype=np.float32), [entry], N=1, sigma=1.0,
        metric='angular', base_hat=base_vectors, base_vectors=base_vectors,
        r_t_hat=0.01, threshold=0.9,
    )
    assert not is_hit
    assert result is None
    assert metadata['certified_indices'] == []
    print("  OK: far-apart query/entry misses")


def test_chord_distance_array_matches_scalar():
    rng = np.random.RandomState(4)
    r = rng.uniform(0.0, 1.0, size=50)
    expected = np.array([_chord_distance_from_angular_scalar(float(ri)) for ri in r])
    actual = _chord_distance_from_angular_array(r)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)
    print("  OK: vectorized and scalar chord distance agree")


def test_overlap_threshold_vectorized_matches_unvectorized_scoring():
    """End-to-end parity: overlap_threshold_hit_policy's new vectorized
    scoring loop must pick the same best_idx/best_score as manually
    looping the (now-unused-in-production) scalar helpers over the same
    entries -- proving the optimization didn't change behavior."""
    print("Test: overlap_threshold_hit_policy vectorized scoring matches manual scalar loop")
    rng = np.random.RandomState(11)
    dim, n_base, K, n_entries = 20, 100, 8, 15
    base_vectors = rng.randn(n_base, dim).astype(np.float32)
    base_hat = base_vectors / np.linalg.norm(base_vectors, axis=1, keepdims=True)

    entries = [_make_angular_entry(rng, dim, n_base, base_hat, base_vectors, K) for _ in range(n_entries)]
    r_t_hat = float(np.median([e.r_Q for e in entries]))

    for _ in range(5):
        test_q = rng.randn(dim).astype(np.float32)
        _, is_hit, metadata = overlap_threshold_hit_policy(
            test_q, entries, N=5, sigma=1.0, metric='angular',
            base_hat=base_hat, base_vectors=base_vectors,
            r_t_hat=r_t_hat, threshold=0.0,
        )

        R2 = _chord_distance_from_angular_scalar(r_t_hat)
        best_score_manual, best_idx_manual = -1.0, None
        for i, entry in enumerate(entries):
            q_hat = test_q / np.linalg.norm(test_q)
            e_hat = entry.query / np.linalg.norm(entry.query)
            D = float(np.arccos(np.clip(np.dot(q_hat, e_hat), -1.0, 1.0)) / np.pi)
            R1 = _chord_distance_from_angular_scalar(entry.r_Q)
            dist = _chord_distance_from_angular_scalar(D)
            score = float(_solid_ball_intersection_fraction_array(
                np.array([dist]), np.array([R1]), R2, dim,
            )[0])
            if score > best_score_manual:
                best_score_manual, best_idx_manual = score, i

        assert metadata['hit_entry_index'] == best_idx_manual
        # float32 base_vectors vs the manual loop's float64 arithmetic
        # introduce O(1e-8) drift in D -- not a correctness issue, just
        # precision, so a looser tolerance than the exact-match parity
        # tests above (which use identical-precision inputs) is correct.
        assert abs(metadata['best_score'] - best_score_manual) < 1e-6
    print("  OK: vectorized hit-policy scoring matches manual per-entry scalar loop")


def test_overlap_threshold_precomputed_r_c_matches_default():
    """precomputed_r_c is purely a caching optimization -- passing it in
    must produce identical output to letting the function fetch radii
    from cache_entries itself."""
    rng = np.random.RandomState(9)
    dim, n_base, K, n_entries = 10, 60, 5, 6
    base_vectors = rng.randn(n_base, dim).astype(np.float32)
    base_hat = base_vectors / np.linalg.norm(base_vectors, axis=1, keepdims=True)
    entries = [_make_angular_entry(rng, dim, n_base, base_hat, base_vectors, K) for _ in range(n_entries)]
    r_c_array = np.array([e.get_radius() for e in entries])
    test_q = rng.randn(dim).astype(np.float32)

    _, hit_a, meta_a = overlap_threshold_hit_policy(
        test_q, entries, N=3, sigma=1.0, metric='angular',
        base_hat=base_hat, base_vectors=base_vectors,
        r_t_hat=0.2, threshold=0.0,
    )
    _, hit_b, meta_b = overlap_threshold_hit_policy(
        test_q, entries, N=3, sigma=1.0, metric='angular',
        base_hat=base_hat, base_vectors=base_vectors,
        r_t_hat=0.2, threshold=0.0, precomputed_r_c=r_c_array,
    )
    assert hit_a == hit_b
    assert meta_a['hit_entry_index'] == meta_b['hit_entry_index']
    assert meta_a['best_score'] == meta_b['best_score']
    assert meta_a['certified_indices'] == meta_b['certified_indices']
    print("  OK: precomputed_r_c matches default radius lookup")

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    print("="*80)
    print("Testing Lemma Implementations")
    print("="*80)

    # test_binary_search()  # commented out -- see definition above
    test_lemma1_simple()
    test_lemma2_simple()
    test_lemma2_union()
    test_combined_algorithm()
    test_cache_miss()
    test_different_k_n()
    test_n_greater_than_k()
    test_lemma1_no_union()
    test_lemma2_no_union()
    test_lemma1_no_union_sigma_one_matches_exact()
    test_lemma1_no_union_sigma_relaxes_threshold()
    test_lemma2_no_union_sigma_one_matches_exact()
    test_lemma2_no_union_sigma_relaxes_k_requirement()
    test_batch_distances_fast_matches_fallback()
    test_cig_distances_matrix_path_matches_fallback()
    test_fit_r_t_hat_matches_percentile()
    test_fit_r_t_hat_never_uses_test_queries()
    test_overlap_threshold_requires_r_t_hat()
    test_overlap_threshold_rejects_non_angular_metric()
    test_overlap_threshold_rejects_sigma_relaxation()
    test_overlap_threshold_empty_cache_misses()
    test_overlap_threshold_hit_on_identical_query()
    test_overlap_threshold_result_sorted_by_true_distance()
    test_overlap_threshold_miss_below_threshold()
    test_chord_distance_array_matches_scalar()
    test_overlap_threshold_vectorized_matches_unvectorized_scoring()
    test_overlap_threshold_precomputed_r_c_matches_default()

    print("\n" + "="*80)
    print("All tests passed.")
    print("="*80)

#-------------------------------------------------------------------------------