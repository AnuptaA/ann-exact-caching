#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Smoke test for run_overlap_threshold_sweep.py's batch runner: runs one
tiny real combo and checks the output shape, plus the correctness
property the free-post-processing threshold design depends on (hit_rate
must be monotonically non-increasing as threshold increases within a
fixed (benchmark, K, budget, percentile) group -- a bug in scoring
best_score once and reusing it across the threshold grid would silently
produce wrong numbers for every row without failing loudly)."""

import numpy as np
import pytest

from simulations.population import debug_common as dc
from simulations.hit_policy.run_overlap_threshold_sweep import (
    run_combo, CSV_COLUMNS, _score_and_recall_per_query, _aggregate_at_thresholds, THRESHOLDS,
)
from utils.base import KVCache
from utils.base.lemmas_sigma import overlap_threshold_hit_policy

#-------------------------------------------------------------------------------

BENCHMARK = 'sift_b50k_c256k100_t512n20_s42'


def _require_benchmark():
    bdir = dc.DATASET_CONFIG['sift']['base_dir'] / BENCHMARK
    if not (bdir / 'cache_queries.npy').exists():
        pytest.skip(f'{BENCHMARK} not prepared')


def test_run_combo_output_shape():
    _require_benchmark()
    rows = run_combo('sift', BENCHMARK, K=5, budget_fraction=0.5)

    assert len(rows) > 0
    for row in rows:
        assert set(row.keys()) == set(CSV_COLUMNS)
    assert {r['percentile'] for r in rows} == {25.0, 50.0, 95.0, 99.0}


def test_run_combo_hit_rate_monotonic_in_threshold():
    _require_benchmark()
    rows = run_combo('sift', BENCHMARK, K=5, budget_fraction=0.5)

    for percentile in {r['percentile'] for r in rows}:
        group = sorted(
            (r for r in rows if r['percentile'] == percentile),
            key=lambda r: r['threshold'],
        )
        hit_rates = [r['hit_rate'] for r in group]
        assert all(a >= b - 1e-12 for a, b in zip(hit_rates, hit_rates[1:])), (
            f"hit_rate not monotonically non-increasing in threshold for percentile={percentile}: {hit_rates}"
        )


def test_run_combo_recall_bounded_and_only_for_hits():
    _require_benchmark()
    rows = run_combo('sift', BENCHMARK, K=5, budget_fraction=0.5)
    for row in rows:
        assert 0.0 <= row['mean_recall'] <= 1.0
        if row['hit_rate'] == 0.0:
            assert row['mean_recall'] == 0.0


def test_score_and_recall_handles_empty_cache():
    """Regression test: an empty kv_cache (e.g. a tiny budget on a small
    benchmark selects zero candidates -- hit in the real esci_c256 run)
    left overlap_threshold_hit_policy's metadata['best_score'] as None,
    which crashed _aggregate_at_thresholds's `scores > threshold` numpy
    comparison instead of correctly reporting a miss at every
    threshold."""
    rng = np.random.RandomState(0)
    base = rng.randn(20, 8).astype(np.float32)
    base_hat = base / np.linalg.norm(base, axis=1, keepdims=True)
    test_q = rng.randn(5, 8).astype(np.float32)
    test_gt_indices = [[0, 1, 2] for _ in range(5)]
    empty_cache = KVCache(metric='angular')

    records = _score_and_recall_per_query(
        empty_cache, base, base_hat, test_q, K=3, r_t_hat=0.2, test_gt_indices=test_gt_indices,
    )
    assert len(records) == 5
    assert all(score == -1.0 and recall == 0.0 for score, recall in records)

    rows = _aggregate_at_thresholds(records, [0.5, 0.9])
    assert all(r['hit_rate'] == 0.0 and r['mean_recall'] == 0.0 for r in rows)


def test_score_and_recall_matches_always_hit_baseline():
    """_score_and_recall_per_query calls the policy with
    threshold=min(THRESHOLDS) instead of -1.0 (always-hit) as a
    performance optimization (skips the expensive result-building
    branch for queries that would be a miss at every threshold in the
    grid anyway). This must be mathematically exact -- the aggregated
    hit_rate/mean_recall for every threshold in THRESHOLDS must be
    identical to what the always-hit baseline would have produced,
    since _aggregate_at_thresholds never reads a query's recall unless
    best_score > threshold >= min(THRESHOLDS)."""
    rng = np.random.RandomState(3)
    dim, n_base, K, n_entries, n_test = 12, 80, 4, 10, 25
    base_vectors = rng.randn(n_base, dim).astype(np.float32)
    base_hat = base_vectors / np.linalg.norm(base_vectors, axis=1, keepdims=True)

    kv_cache = KVCache(metric='angular')
    for _ in range(n_entries):
        q = rng.randn(dim).astype(np.float32)
        q_hat = q / np.linalg.norm(q)
        dists = np.arccos(np.clip(base_hat @ q_hat, -1.0, 1.0)) / np.pi
        order = np.argsort(dists)
        r_c = float(dists[order[K - 1]])
        indices = order[dists <= r_c].astype(np.int32)
        kv_cache.add_entry(q, [], r_c, 0.0, indices, None)

    test_q = rng.randn(n_test, dim).astype(np.float32)
    test_gt_indices = [rng.choice(n_base, size=K, replace=False).tolist() for _ in range(n_test)]
    r_t_hat = 0.15

    new_records = _score_and_recall_per_query(kv_cache, base_vectors, base_hat, test_q, K, r_t_hat, test_gt_indices)

    cache_entries = kv_cache.get_all_entries()
    r_c_array = kv_cache.get_entry_radii()
    baseline_records = []
    for i, query in enumerate(test_q):
        d_Q = kv_cache.get_d_Q_batch(query)
        _, _, metadata = overlap_threshold_hit_policy(
            query, cache_entries, N=K, sigma=1.0, metric='angular',
            precomputed_d_Q=d_Q, precomputed_r_c=r_c_array,
            base_hat=base_hat, base_vectors=base_vectors,
            r_t_hat=r_t_hat, threshold=-1.0,
        )
        true_gt = set(test_gt_indices[i])
        certified = set(metadata['certified_indices'])
        recall = len(certified & true_gt) / K
        baseline_records.append((metadata['best_score'], recall))

    new_rows = _aggregate_at_thresholds(new_records, THRESHOLDS)
    baseline_rows = _aggregate_at_thresholds(baseline_records, THRESHOLDS)
    for new_row, baseline_row in zip(new_rows, baseline_rows):
        assert new_row['threshold'] == baseline_row['threshold']
        assert abs(new_row['hit_rate'] - baseline_row['hit_rate']) < 1e-12
        assert abs(new_row['mean_recall'] - baseline_row['mean_recall']) < 1e-12
