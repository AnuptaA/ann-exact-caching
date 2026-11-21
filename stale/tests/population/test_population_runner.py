#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Tests for simulations.population.population_runner's core simulation loop, including
the sigma (relaxed-recall) parameter sweep."""

import json
import tempfile
from pathlib import Path

import numpy as np
from utils.base import KVCache
from utils.base.distance_metrics import angular_distance
from utils.base.population import CandidateEntry
from simulations.population.population_runner import _run_simulation, _write_selection_log

#-------------------------------------------------------------------------------


def _make_cache_with_hits(dim: int, n_cache: int, n_vectors: int, rng):
    """Build a KVCache whose entries contain known neighbors.

    Returns (cache, base_vectors, test_queries, test_gt_indices, K).

    Test queries equal cache queries, so D(q, Q) = 0. Cached vectors are
    placed very close to each query so D(q, v_i) << r_Q and every entry
    certifies all n_vectors neighbors.
    """
    cache = KVCache(metric='angular')
    queries = []
    all_vecs = []
    gt_indices = []

    for i in range(n_cache):
        q = rng.randn(dim).astype(np.float32)
        q /= np.linalg.norm(q)
        queries.append(q)

        # Vectors very close to q (small noise => small angular distance).
        idxs = list(range(i * n_vectors, (i + 1) * n_vectors))
        vecs = []
        for _ in range(n_vectors):
            v = q + rng.randn(dim).astype(np.float32) * 0.01
            v /= np.linalg.norm(v)
            vecs.append(v)
        all_vecs.extend(vecs)
        gt_indices.append(idxs)

        # r_Q = 0.1 (in [0,1] normalized angular): D(q, v_i) << 0.1 so CIG certifies all.
        cache.add_entry(q, vecs, r_Q=0.1, half_gap=0.0, top_k_indices=idxs)

    base_vectors = np.stack(all_vecs)
    test_queries = np.stack(queries)
    return cache, base_vectors, test_queries, gt_indices, n_vectors


def _run_one(cache, base, test_q, gt_indices, K, sigma, metric='angular', **kw):
    """Wrapper: call _run_simulation for algorithm='lemma1_no_union' at a single
    sigma, return a dict matching the old _run_simulation_sigma per-sigma result
    shape."""
    results = _run_simulation(
        kv_cache=cache,
        base_vectors=base,
        test_queries=test_q,
        K=K,
        algorithms=['lemma1_no_union'],
        benchmark_name=kw.get('benchmark_name', 'bench'),
        strategy=kw.get('strategy', 'greedy_cost'),
        budget_B=kw.get('budget_B', 10),
        eval_mode=kw.get('eval_mode', 'test'),
        metric=metric,
        sigma=sigma,
        test_gt_indices=gt_indices,
    )
    r = results[0]
    return {
        'algorithm': r.algorithm,
        'sigma': r.sigma,
        'hit_rate': r.hit_rate,
        'mean_recall': r.mean_recall,
        'cache_hits': r.cache_hits,
        'total_queries': r.total_queries,
    }

#-------------------------------------------------------------------------------


def test_run_simulation_sigma_all_hits():
    """All queries hit the cache, so hit_rate should be 1.0 for low sigma."""
    rng = np.random.RandomState(10)
    dim = 16
    n_cache = 4
    n_vectors = 5

    cache, base, test_q, gt_indices, K = _make_cache_with_hits(dim, n_cache, n_vectors, rng)

    result = _run_one(
        cache, base, test_q, gt_indices, K, sigma=1.0,
        benchmark_name='test_bench', eval_mode='test',
    )

    assert result['algorithm'] == 'lemma1_no_union', f"algorithm={result['algorithm']}"
    assert result['total_queries'] == n_cache
    assert result['hit_rate'] == 1.0, f"expected hit_rate=1.0, got {result['hit_rate']}"
    assert 0.0 <= result['mean_recall'] <= 1.0


def test_run_simulation_sigma_no_hits():
    """Empty cache, so hit_rate should be 0.0."""
    rng = np.random.RandomState(20)
    dim = 8
    n_test = 3
    K = 5

    cache = KVCache(metric='angular')  # no entries
    test_q = rng.randn(n_test, dim).astype(np.float32)
    for i in range(n_test):
        test_q[i] /= np.linalg.norm(test_q[i])
    base = rng.randn(50, dim).astype(np.float32)
    gt_indices = [list(range(i, i + K)) for i in range(n_test)]

    result = _run_one(
        cache, base, test_q, gt_indices, K, sigma=0.70,
        benchmark_name='empty_bench', budget_B=0, eval_mode='test',
    )

    assert result['hit_rate'] == 0.0, f"expected 0.0, got {result['hit_rate']}"
    assert result['mean_recall'] == 0.0, f"expected 0.0 recall, got {result['mean_recall']}"


def test_result_fields_present():
    """result dict must contain all required fields."""
    rng = np.random.RandomState(30)
    dim = 8
    n_cache = 2
    cache, base, test_q, gt_indices, K = _make_cache_with_hits(dim, n_cache, 3, rng)

    result = _run_one(cache, base, test_q, gt_indices, K, sigma=0.80, eval_mode='test')

    for field in ('algorithm', 'sigma', 'hit_rate', 'mean_recall', 'cache_hits', 'total_queries'):
        assert field in result, f"missing field: {field}"
    assert result['sigma'] == 0.80
    assert isinstance(result['cache_hits'], int)
    assert 0.0 <= result['hit_rate'] <= 1.0
    assert 0.0 <= result['mean_recall'] <= 1.0


def test_recall_bounded():
    """mean_recall must always be in [0, 1]."""
    rng = np.random.RandomState(40)
    dim = 8
    n_cache = 3
    cache, base, test_q, gt_indices, K = _make_cache_with_hits(dim, n_cache, 4, rng)

    for sigma in [0.70, 0.80, 0.90, 1.00]:
        result = _run_one(cache, base, test_q, gt_indices, K, sigma=sigma, eval_mode='test')
        assert 0.0 <= result['mean_recall'] <= 1.0, (
            f"sigma={sigma}: mean_recall={result['mean_recall']} out of [0,1]"
        )


def test_recall_at_least_sigma_for_hits():
    """For every hit, certified recall against true GT must be >= sigma.

    Builds a cache where top_k_indices are exactly the true K-NN, so every
    hit's certified set is a subset of the true GT. mean_recall must then be
    >= sigma since the threshold requires ceil(sigma*K) certified out of K.
    """
    rng = np.random.RandomState(50)
    dim = 16
    n_cache = 6

    for K in (5, 10):
        cache, base, test_q, gt_indices, _ = _make_cache_with_hits(dim, n_cache, K, rng)
        for sigma in (0.70, 0.80, 0.90):
            result = _run_one(cache, base, test_q, gt_indices, K, sigma=sigma, eval_mode='test')
            assert result['hit_rate'] == 1.0, f"K={K} sigma={sigma}: expected all hits"
            assert result['mean_recall'] >= sigma - 1e-9, (
                f"K={K} sigma={sigma}: mean_recall={result['mean_recall']} < sigma"
            )


def test_run_simulation_k_zero_no_zero_division():
    """K=0 (containment mode) reaches _run_simulation without a sim_K clamp;
    recall_sum += .../K is guarded against division by zero. A hit at K=0
    scores recall=1.0, since containment alone is the full task and there
    is no K-sized target to average against."""
    query = np.array([1.0, 0.0], dtype=np.float32)
    cache = KVCache(metric='euclidean')
    # Entry contains the test query itself, well within r_Q, so it is contained.
    cache.add_entry(query, [np.array([5.0, 5.0], dtype=np.float32)],
                     r_Q=0.5, half_gap=0.0, top_k_indices=[0])
    base = np.array([[5.0, 5.0]], dtype=np.float32)
    test_q = np.array([query])
    gt_indices = [[]]

    result = _run_one(
        cache, base, test_q, gt_indices, K=0, sigma=1.0,
        metric='euclidean', eval_mode='test',
    )
    assert result['hit_rate'] == 1.0, "contained query at K=0 must certify"
    assert result['mean_recall'] == 1.0, (
        "containment alone at K=0 must score recall=1.0 for a hit, not 0.0"
    )


def test_run_simulation_k_zero_miss_stays_zero_recall():
    """K=0, no hit: mean_recall must stay 0.0 (no hits to average), not be
    pulled up by the new K=0 recall=1.0-on-hit convention."""
    query = np.array([1.0, 0.0], dtype=np.float32)
    cache = KVCache(metric='euclidean')
    # Entry query is far from the test query, so the O1 gate fails, no hit.
    cache.add_entry(np.array([-1.0, 0.0], dtype=np.float32),
                     [np.array([5.0, 5.0], dtype=np.float32)],
                     r_Q=0.1, half_gap=0.0, top_k_indices=[0])
    base = np.array([[5.0, 5.0]], dtype=np.float32)
    test_q = np.array([query])
    gt_indices = [[]]

    result = _run_one(
        cache, base, test_q, gt_indices, K=0, sigma=1.0,
        metric='euclidean', eval_mode='test',
    )
    assert result['hit_rate'] == 0.0, "far query at K=0 must not certify"
    assert result['mean_recall'] == 0.0, "no hits means mean_recall stays 0.0"


def test_index_space_consistency():
    """certified_indices and test_gt_indices share the same index space.

    Cached top_k_indices use the same local base indices as the GT for the
    matching test query, so a hit's certified set must intersect the true GT.
    """
    rng = np.random.RandomState(60)
    dim = 16
    n_cache = 4
    K = 5

    cache, base, test_q, gt_indices, _ = _make_cache_with_hits(dim, n_cache, K, rng)
    cache_entries = cache.get_all_entries()

    from utils.base.lemmas_sigma import lemma1_no_union_sigma
    for i, query in enumerate(test_q):
        _, is_hit, meta = lemma1_no_union_sigma(query, cache_entries, K, sigma=1.0, metric='angular')
        assert is_hit, f"query {i}: expected hit"
        certified = set(meta.get('certified_indices', []))
        true_gt = set(gt_indices[i])
        intersection = certified & true_gt
        assert len(intersection) > 0, (
            f"query {i}: certified {certified} does not intersect true GT {true_gt}, "
            "possible index space mismatch"
        )
        assert certified <= true_gt, (
            f"query {i}: certified {certified} not subset of true GT {true_gt}"
        )


def test_mean_recall_excludes_misses():
    """mean_recall must average over hit queries only, not all queries.

    Builds a cache with 2 entries that hit and adds 2 disjoint queries that
    miss entirely (empty entries list passed via separate no-match queries).
    Confirms mean_recall == sum(recall_per_hit) / n_hits, not / n_total.
    """
    rng = np.random.RandomState(70)
    dim = 16
    n_hit_cache = 2
    K = 4

    cache, base, hit_q, hit_gt, _ = _make_cache_with_hits(dim, n_hit_cache, K, rng)

    # Two miss queries: far from any cached query/vector, tiny r_Q already in cache
    # so they will not be certified.
    miss_q = []
    for _ in range(2):
        v = rng.randn(dim).astype(np.float32) * 5
        v /= np.linalg.norm(v)
        miss_q.append(v)
    miss_q = np.stack(miss_q)
    miss_gt = [list(range(1000, 1000 + K)) for _ in range(2)]

    test_q = np.concatenate([hit_q, miss_q], axis=0)
    gt_indices = hit_gt + miss_gt

    result = _run_one(
        cache, base, test_q, gt_indices, K, sigma=1.0,
        benchmark_name='mixed', budget_B=10, eval_mode='test',
    )

    assert result['total_queries'] == n_hit_cache + 2
    assert result['cache_hits'] == n_hit_cache, (
        f"expected {n_hit_cache} hits, got {result['cache_hits']}"
    )
    # All hit queries are exact cache queries -> recall=1.0 each -> mean_recall=1.0
    assert abs(result['mean_recall'] - 1.0) < 1e-9, (
        f"mean_recall={result['mean_recall']} should be 1.0 (over hits only), "
        "not dragged down by misses"
    )


def test_simulation_loop_large_entries():
    """Smoke test: run full simulation loop with large entries.

    Uses n_vectors=1000 per entry to catch memory or computation regressions.
    The runner uses entry-centric batch matmul from base_vectors regardless of
    whether top_k_matrix is stored in the entry.
    """
    rng = np.random.RandomState(80)
    dim = 16
    n_cache = 5
    n_vectors = 1000
    n_test = 20
    K = 5
    sigma_values = [0.70, 0.80, 1.00]

    cache = KVCache(metric='angular')
    queries = []
    all_vecs_flat = []

    for i in range(n_cache):
        q = rng.randn(dim).astype(np.float32)
        q /= np.linalg.norm(q)
        queries.append(q)

        idxs = list(range(i * n_vectors, (i + 1) * n_vectors))
        mat = (q + rng.randn(n_vectors, dim).astype(np.float32) * 0.01)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        mat /= norms
        top_k_matrix = mat.astype(np.float32)
        vecs = [mat[j] for j in range(n_vectors)]
        all_vecs_flat.extend(vecs)

        cache.add_entry(q, vecs, r_Q=0.1, half_gap=0.0,
                        top_k_indices=idxs, top_k_matrix=top_k_matrix)

    base = np.stack(all_vecs_flat)
    # Use cache queries as test queries to guarantee hits
    test_q = np.stack(queries[:n_test] if n_cache >= n_test else queries)
    actual_n_test = len(test_q)
    gt_indices = [list(range(i * n_vectors, i * n_vectors + K)) for i in range(actual_n_test)]

    results = {
        sigma: _run_one(cache, base, test_q, gt_indices, K, sigma, eval_mode='smoke')
        for sigma in sigma_values
    }

    assert set(results.keys()) == set(sigma_values), f"missing sigma keys: {results.keys()}"
    for sigma in sigma_values:
        r = results[sigma]
        assert 0.0 <= r['hit_rate'] <= 1.0
        assert 0.0 <= r['mean_recall'] <= 1.0

#-------------------------------------------------------------------------------


def _reference_write_selection_log(out_path, selected, trajectory):
    """Naive reference for _write_selection_log's min_dist_to_prev field:
    one angular_distance() Python call per (step, prior-step) pair, kept
    only in this test for cross-checking the vectorized production
    implementation."""
    log = []
    cached_before = np.zeros(len(selected[0].R_c), dtype=bool) if selected else None
    prev_queries = []
    for step, (cand, traj_entry) in enumerate(zip(selected, trajectory)):
        new_vecs = int(np.sum(cand.R_c & ~cached_before))
        cached_before |= cand.R_c
        rc_size = cand.rc_size
        overlap_pct = round((rc_size - new_vecs) / max(rc_size, 1), 4)
        min_dist = round(float(min(
            angular_distance(cand.query, pq) for pq in prev_queries
        )), 6) if prev_queries else 0.0
        prev_queries.append(cand.query)
        log.append({
            "step": step + 1,
            "alpha": round(float(cand.alpha), 4) if cand.alpha is not None else None,
            "r_Q": round(float(cand.r_Q), 6),
            "r_c": round(float(cand.r_c), 6),
            "vectors_in_R_c": rc_size,
            "new_vectors_added": new_vecs,
            "overlap_pct": overlap_pct,
            "min_dist_to_prev": min_dist,
            "queries_covered_total": int(cand.covered_train_indices.sum()),
            "new_queries_covered": traj_entry.get("new_train_coverage", 0),
            "cig_coverage": round(traj_entry["cig_train_coverage"], 6),
        })
    with open(out_path, 'w') as f:
        json.dump(log, f, indent=2)


def test_write_selection_log_matches_naive_reference():
    """The vectorized _write_selection_log (batched matmul for
    min_dist_to_prev) must match the naive per-pair-call reference exactly,
    including candidates with a zero-norm query (angular_distance's
    fallback-to-1.0 edge case)."""
    rng = np.random.RandomState(99)
    n_base = 50
    n_train = 10
    dim = 8
    n_candidates = 40

    selected = []
    trajectory = []
    covered = np.zeros(n_train, dtype=bool)
    for i in range(n_candidates):
        q = rng.randn(dim).astype(np.float32)
        if i == 5:
            q = np.zeros(dim, dtype=np.float32)  # exercise the zero-norm edge case
        R_c = rng.random(n_base) < 0.3
        cov = rng.random(n_train) < 0.2
        covered |= cov
        selected.append(CandidateEntry(
            query=q, r_Q=0.1, alpha=1.1, r_c=0.15,
            R_c=R_c, rc_size=int(R_c.sum()),
            covered_train_indices=cov, covered_test_indices=cov,
            top_k_vectors=[], top_k_indices=[], half_gap=0.0,
        ))
        trajectory.append({
            "num_entries": i + 1, "vectors_cached": int(R_c.sum()),
            "cig_train_coverage": float(covered.sum()) / n_train,
            "cig_test_coverage": 0.0,
            "new_train_coverage": int(cov.sum()), "new_test_coverage": 0,
        })

    with tempfile.TemporaryDirectory() as d:
        got_path = Path(d) / "got.json"
        ref_path = Path(d) / "ref.json"
        _write_selection_log(got_path, selected, trajectory)
        _reference_write_selection_log(ref_path, selected, trajectory)

        got = json.loads(got_path.read_text())
        ref = json.loads(ref_path.read_text())

    assert got == ref, "vectorized _write_selection_log diverges from naive reference"

#-------------------------------------------------------------------------------
