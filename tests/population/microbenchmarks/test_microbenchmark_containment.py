#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import csv
import importlib

import numpy as np
import pytest

from tests.helpers import (
    create_mini_benchmark_with_population_data,
    get_temp_dir,
    cleanup_temp_dir,
)
from simulations.population.microbenchmarks.microbenchmark_containment import (
    K_CACHE_VALUES,
    K_TEST_VALUES,
    N_VALUES,
)

#-------------------------------------------------------------------------------

# 10 valid (k_cache, k_test) pairs where k_test < k_cache
EXPECTED_ROWS_PER_BENCHMARK = len(N_VALUES) * sum(
    sum(1 for kt in K_TEST_VALUES if kt < kc) for kc in K_CACHE_VALUES
)

EXPECTED_CSV_FIELDS = {
    'benchmark', 'n', 'k_cache', 'k_test',
    'hit_rate', 'mean_r_q', 'mean_n_base_inside',
}


def _mc(dataset: str):
    return importlib.import_module('simulations.population.microbenchmarks.microbenchmark_containment')

#-------------------------------------------------------------------------------


@pytest.mark.parametrize('dataset', ['sift', 'esci'])
def test_compute_ball_costs_shape_and_indices(dataset):
    mc = _mc(dataset)
    n_cache = 25
    cache_q = np.random.randn(n_cache, 16).astype(np.float32)

    sorted_dists, sorted_indices = mc._compute_ball_costs(cache_q)

    assert sorted_dists.shape == (n_cache, n_cache), f"dists shape: {sorted_dists.shape}"
    assert sorted_indices.shape == (n_cache, n_cache), f"indices shape: {sorted_indices.shape}"
    assert np.all(np.isinf(sorted_dists[:, -1])), "Last column of sorted_dists should be inf (self excluded)"

    for i in range(n_cache):
        finite = sorted_dists[i, sorted_dists[i] < np.inf]
        assert np.all(np.diff(finite) >= 0.0), f"Row {i} not sorted ascending"

    assert np.all(sorted_dists[sorted_dists < np.inf] > 0.0), "All finite distances should be > 0"

    cache_norm = cache_q / np.linalg.norm(cache_q, axis=1, keepdims=True)
    raw_dists = np.arccos(np.clip(cache_norm @ cache_norm.T, -1.0, 1.0)) / np.pi
    np.fill_diagonal(raw_dists, np.inf)
    for i in range(n_cache):
        for j in range(n_cache - 1):
            idx = sorted_indices[i, j]
            assert abs(sorted_dists[i, j] - raw_dists[i, idx]) < 1e-5, (
                f"Index mismatch at row {i}, position {j}: "
                f"sorted_dists={sorted_dists[i, j]:.6f}, raw={raw_dists[i, idx]:.6f}"
            )


@pytest.mark.parametrize('dataset', ['sift', 'esci'])
def test_compute_rows_invariants(dataset):
    mc = _mc(dataset)
    n_cache = 25
    n_test = 15
    n_base = 50
    cache_q = np.random.randn(n_cache, 16).astype(np.float32)
    test_q = np.random.randn(n_test, 16).astype(np.float32)
    base_q = np.random.randn(n_base, 16).astype(np.float32)

    cache_norm = cache_q / np.linalg.norm(cache_q, axis=1, keepdims=True)
    test_norm = test_q / np.linalg.norm(test_q, axis=1, keepdims=True)
    base_norm = base_q / np.linalg.norm(base_q, axis=1, keepdims=True)

    test_dists = np.arccos(np.clip(test_norm @ cache_norm.T, -1.0, 1.0)) / np.pi
    cb_dists = np.arccos(np.clip(cache_norm @ base_norm.T, -1.0, 1.0)) / np.pi

    ball_costs, nn_indices = mc._compute_ball_costs(cache_q)

    r_t_by_k = {
        0: np.zeros(n_test, dtype=np.float32),
        1: np.sort(np.abs(np.random.randn(n_test).astype(np.float32))) * 0.05,
        5: np.sort(np.abs(np.random.randn(n_test).astype(np.float32))) * 0.15,
        10: np.sort(np.abs(np.random.randn(n_test).astype(np.float32))) * 0.25,
    }

    rows = mc._compute_rows(test_dists, ball_costs, nn_indices, cb_dists, r_t_by_k, 'test')

    assert len(rows) == EXPECTED_ROWS_PER_BENCHMARK, (
        f"Expected {EXPECTED_ROWS_PER_BENCHMARK} rows, got {len(rows)}"
    )

    for row in rows:
        assert 0.0 <= row['hit_rate'] <= 1.0, f"hit_rate out of range: {row}"
        assert row['mean_r_q'] >= 0.0
        assert row['mean_n_base_inside'] >= 0.0

    def get_hit(n, k_cache, k_test):
        matches = [r for r in rows if r['n'] == n and r['k_cache'] == k_cache and r['k_test'] == k_test]
        return matches[0]['hit_rate'] if matches else None

    def get_mean_r_q(n, k_cache, k_test):
        matches = [r for r in rows if r['n'] == n and r['k_cache'] == k_cache and r['k_test'] == k_test]
        return matches[0]['mean_r_q'] if matches else None

    # monotone in k_test (fixed n, k_cache): hit_rate decreasing
    for n in N_VALUES:
        for k_cache in K_CACHE_VALUES:
            valid_kts = sorted(kt for kt in K_TEST_VALUES if kt < k_cache)
            for i in range(len(valid_kts) - 1):
                kt_lo, kt_hi = valid_kts[i], valid_kts[i + 1]
                hr_lo = get_hit(n, k_cache, kt_lo)
                hr_hi = get_hit(n, k_cache, kt_hi)
                if hr_lo is not None and hr_hi is not None:
                    assert hr_lo >= hr_hi - 1e-9, (
                        f"hit_rate not monotone in k_test: n={n} k_cache={k_cache} "
                        f"k_test={kt_lo}:{hr_lo:.4f} > k_test={kt_hi}:{hr_hi:.4f}"
                    )

    # monotone in k_cache (fixed n, k_test): hit_rate and mean_r_q increasing
    for n in N_VALUES:
        for k_test in K_TEST_VALUES:
            valid_kcs = sorted(kc for kc in K_CACHE_VALUES if k_test < kc)
            for i in range(len(valid_kcs) - 1):
                kc_lo, kc_hi = valid_kcs[i], valid_kcs[i + 1]
                hr_lo = get_hit(n, kc_lo, k_test)
                hr_hi = get_hit(n, kc_hi, k_test)
                rq_lo = get_mean_r_q(n, kc_lo, k_test)
                rq_hi = get_mean_r_q(n, kc_hi, k_test)
                if hr_lo is not None and hr_hi is not None:
                    assert hr_lo <= hr_hi + 1e-9, (
                        f"hit_rate not monotone in k_cache: n={n} k_test={k_test} "
                        f"k_cache={kc_lo}:{hr_lo:.4f} > k_cache={kc_hi}:{hr_hi:.4f}"
                    )
                if rq_lo is not None and rq_hi is not None:
                    assert rq_lo <= rq_hi + 1e-9, (
                        f"mean_r_q not monotone in k_cache: n={n} k_test={k_test} "
                        f"k_cache={kc_lo}:{rq_lo:.4f} > k_cache={kc_hi}:{rq_hi:.4f}"
                    )


@pytest.mark.parametrize('dataset', ['sift', 'esci'])
def test_save_csv_columns(dataset):
    mc = _mc(dataset)
    tmp = get_temp_dir()
    try:
        rows = [
            {'benchmark': 'b', 'n': n, 'k_cache': 5, 'k_test': 1,
             'hit_rate': 0.5, 'mean_r_q': 0.3, 'mean_n_base_inside': 10.0}
            for n in N_VALUES
        ]
        out = tmp / 'test.csv'
        mc._save_csv(rows, out)

        assert out.exists()
        with open(out) as f:
            reader = csv.DictReader(f)
            read_rows = list(reader)
        assert set(reader.fieldnames) == EXPECTED_CSV_FIELDS
        assert len(read_rows) == len(N_VALUES)
    finally:
        cleanup_temp_dir(tmp)


@pytest.mark.parametrize('dataset', ['sift', 'esci'])
def test_run_benchmark_skips_missing(dataset):
    mc = _mc(dataset)
    tmp = get_temp_dir()
    try:
        bm_dir = tmp / 'no_gt'
        bm_dir.mkdir()
        rows, bc, ni = mc.run_benchmark('no_gt', mc.DATASET_CONFIG[dataset]['load_benchmark_fn'], tmp)
        assert rows == [], f"Expected [], got {rows}"
        assert bc is None
        assert ni is None
    finally:
        cleanup_temp_dir(tmp)


@pytest.mark.parametrize('dataset', ['sift', 'esci'])
def test_run_benchmark_returns_rows(dataset):
    mc = _mc(dataset)
    tmp = get_temp_dir()
    try:
        bm_dir = tmp / f'mini_{dataset}'
        create_mini_benchmark_with_population_data(dataset, bm_dir)

        rows, ball_costs, nn_indices = mc.run_benchmark(
            f'mini_{dataset}', mc.DATASET_CONFIG[dataset]['load_benchmark_fn'], tmp)

        assert len(rows) == EXPECTED_ROWS_PER_BENCHMARK, (
            f"Expected {EXPECTED_ROWS_PER_BENCHMARK} rows, got {len(rows)}"
        )

        n_cache = 20
        assert ball_costs is not None and ball_costs.shape == (n_cache, n_cache)
        assert nn_indices is not None and nn_indices.shape == (n_cache, n_cache)

        for row in rows:
            assert set(row.keys()) == EXPECTED_CSV_FIELDS
            assert 0.0 <= row['hit_rate'] <= 1.0
            assert row['mean_r_q'] >= 0.0
            assert row['mean_n_base_inside'] >= 0.0
    finally:
        cleanup_temp_dir(tmp)


@pytest.mark.parametrize('dataset', ['sift', 'esci'])
def test_aggregate_plots_creates_files(dataset):
    mc = _mc(dataset)
    tmp = get_temp_dir()
    try:
        for name in ['mini_a', 'mini_b']:
            bm_dir = tmp / name
            create_mini_benchmark_with_population_data(dataset, bm_dir)

        all_data = {}
        for name in ['mini_a', 'mini_b']:
            rows, _, _ = mc.run_benchmark(name, mc.DATASET_CONFIG[dataset]['load_benchmark_fn'], tmp)
            if rows:
                all_data[name] = rows

        assert len(all_data) == 2, "Both benchmarks should produce results"

        out = tmp / 'agg'
        mc._aggregate_plots(all_data, out, mc.DATASET_CONFIG[dataset]['label'])

        for fname in ['hit_rate_vs_n.png', 'hit_rate_vs_k_cache.png',
                      'budget_vs_k_cache.png', 'containment.csv']:
            p = out / fname
            assert p.exists(), f"Missing: {fname}"
            assert p.stat().st_size > 0, f"Empty: {fname}"

        with open(out / 'containment.csv') as f:
            reader = csv.DictReader(f)
            csv_rows = list(reader)
        assert len(csv_rows) == 2 * EXPECTED_ROWS_PER_BENCHMARK
    finally:
        cleanup_temp_dir(tmp)
