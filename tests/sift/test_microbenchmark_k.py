#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys
import csv
import warnings
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from tests.sift.helpers import (
    create_mini_sift_benchmark_with_population_data,
    get_temp_dir,
    cleanup_temp_dir,
)
from simulations.sift.microbenchmark_k import (
    ALPHA_VALUES,
    _load_gt_for_k,
    _collect_candidates_for_k,
    _save_csv,
    _cross_k_plots,
    _aggregate_plots,
)

#-------------------------------------------------------------------------------

# The mini SIFT benchmark creates GT files for K=[1, 5, 10].
MINI_K_VALUES = [1, 5, 10]
EXPECTED_FIELDS = {'k', 'alpha', 'r_Q', 'r_c', 'R_c_size', 'n_covered', 'coverage_rate'}

#-------------------------------------------------------------------------------

def test_load_gt_for_k_shapes():
    print("Test: _load_gt_for_k returns arrays with correct shapes")
    tmp = get_temp_dir()
    try:
        bm_dir = tmp / 'mini_sift'
        create_mini_sift_benchmark_with_population_data(bm_dir)
        base = np.load(bm_dir / 'base_vectors.npy')
        n_cache = 20

        for k in [1, 5]:
            tv, ti, hg, rQ = _load_gt_for_k(bm_dir, k, n_cache, base)
            assert len(tv) == n_cache
            assert len(ti) == n_cache
            assert len(hg) == n_cache
            assert len(rQ) == n_cache
            assert all(len(v) == k for v in tv), f"top_k_vectors wrong length for k={k}"
            assert all(len(idx) == k for idx in ti), f"top_k_indices wrong length for k={k}"
    finally:
        cleanup_temp_dir(tmp)
    print("  Passed.")

def test_collect_candidates_for_k_fields():
    print("Test: _collect_candidates_for_k returns rows with correct fields")
    tmp = get_temp_dir()
    try:
        bm_dir = tmp / 'mini_sift'
        create_mini_sift_benchmark_with_population_data(bm_dir)

        rows_by_k = _collect_candidates_for_k('mini_sift', benchmark_base_dir=tmp)
        assert len(rows_by_k) > 0, "Expected results for at least one k value"

        for target_k, rows in rows_by_k.items():
            assert target_k in MINI_K_VALUES, f"Unexpected k={target_k}"
            assert len(rows) > 0, f"Expected candidates for k={target_k}"
            for row in rows:
                assert set(row.keys()) == EXPECTED_FIELDS, f"Wrong fields: {set(row.keys())}"
                assert row['k'] == target_k
                assert row['alpha'] in ALPHA_VALUES
                assert 0.0 <= row['coverage_rate'] <= 1.0
    finally:
        cleanup_temp_dir(tmp)
    print("  Passed.")

def test_collect_candidates_for_k_missing_gt():
    print("Test: _collect_candidates_for_k returns {} when GT file missing")
    tmp = get_temp_dir()
    try:
        import json
        bm_dir = tmp / 'mini_sift'
        bm_dir.mkdir()
        np.save(bm_dir / 'base_vectors.npy', np.zeros((10, 4), dtype=np.float32))
        np.save(bm_dir / 'cache_queries.npy', np.zeros((5, 4), dtype=np.float32))
        np.save(bm_dir / 'test_queries.npy', np.zeros((3, 4), dtype=np.float32))
        json.dump({'cache_K': 5}, open(bm_dir / 'config.json', 'w'))

        rows_by_k = _collect_candidates_for_k('mini_sift', benchmark_base_dir=tmp)
        assert rows_by_k == {}, f"Expected {{}}, got keys {list(rows_by_k.keys())}"
    finally:
        cleanup_temp_dir(tmp)
    print("  Passed.")

def test_save_csv_writes_k_column():
    print("Test: _save_csv includes k column")
    tmp = get_temp_dir()
    try:
        rows_by_k = {
            1: [{'k': 1, 'alpha': 1.5, 'r_Q': 0.1, 'r_c': 0.15,
                 'R_c_size': 10, 'n_covered': 3, 'coverage_rate': 0.3}],
            5: [{'k': 5, 'alpha': 1.5, 'r_Q': 0.2, 'r_c': 0.3,
                 'R_c_size': 20, 'n_covered': 5, 'coverage_rate': 0.5}],
        }
        out = tmp / 'candidates.csv'
        _save_csv(rows_by_k, out)

        assert out.exists()
        with open(out) as f:
            reader = csv.DictReader(f)
            read_rows = list(reader)
        assert 'k' in reader.fieldnames
        assert len(read_rows) == 2
        assert {int(r['k']) for r in read_rows} == {1, 5}
    finally:
        cleanup_temp_dir(tmp)
    print("  Passed.")

def test_cross_k_plots_creates_files():
    print("Test: _cross_k_plots creates cross-K summary PNG files")
    tmp = get_temp_dir()
    try:
        bm_dir = tmp / 'mini_sift'
        create_mini_sift_benchmark_with_population_data(bm_dir)
        rows_by_k = _collect_candidates_for_k('mini_sift', benchmark_base_dir=tmp)
        assert rows_by_k, "Need candidates to test plotting"

        out = tmp / 'cross_k'
        _cross_k_plots(rows_by_k, 'mini_sift', out)

        for fname in ['k_vs_rc_size.png', 'k_vs_coverage.png']:
            p = out / fname
            assert p.exists(), f"Missing: {fname}"
            assert p.stat().st_size > 0, f"Empty: {fname}"
    finally:
        cleanup_temp_dir(tmp)
    print("  Passed.")

def test_aggregate_plots_creates_files():
    print("Test: _aggregate_plots creates aggregate PNG files")
    tmp = get_temp_dir()
    try:
        bm_dir = tmp / 'mini_sift'
        create_mini_sift_benchmark_with_population_data(bm_dir)
        rows_by_k = _collect_candidates_for_k('mini_sift', benchmark_base_dir=tmp)
        assert rows_by_k, "Need candidates to test plotting"

        out = tmp / 'aggregate'
        _aggregate_plots({'mini_sift': rows_by_k}, out)

        for fname in ['rc_size.png', 'coverage.png']:
            p = out / fname
            assert p.exists(), f"Missing: {fname}"
            assert p.stat().st_size > 0, f"Empty: {fname}"
    finally:
        cleanup_temp_dir(tmp)
    print("  Passed.")

#-------------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Testing simulations/sift/microbenchmark_k.py")
    print("=" * 70)

    test_load_gt_for_k_shapes()
    test_collect_candidates_for_k_fields()
    test_collect_candidates_for_k_missing_gt()
    test_save_csv_writes_k_column()
    test_cross_k_plots_creates_files()
    test_aggregate_plots_creates_files()

    print("\n" + "=" * 70)
    print("All tests passed.")
    print("=" * 70)

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
