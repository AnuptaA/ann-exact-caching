#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys
import csv
import warnings
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from tests.sift.helpers import (
    create_mini_sift_benchmark_with_population_data,
    get_temp_dir,
    cleanup_temp_dir,
)
from simulations.sift.microbenchmark_candidates import (
    ALPHA_VALUES,
    _cache_k_from_name,
    _output_label,
    _collect_candidates,
    _save_csv,
    _per_benchmark_plots,
    _aggregate_plots,
)

#-------------------------------------------------------------------------------

EXPECTED_FIELDS = {'alpha', 'r_Q', 'r_c', 'R_c_size', 'n_covered', 'coverage_rate'}

#-------------------------------------------------------------------------------

def test_output_label():
    print("Test: _output_label normalises embedded k-value")
    assert _output_label('sift_b50k_c256k100_t512n20_s42', 100) == 'sift_b50k_c256k100_t512n20_s42'
    assert _cache_k_from_name('sift_b50k_c256k100_t512n20_s42') == 100
    assert _cache_k_from_name('sift_b50k_c1024k100_t512n20_s42') == 100
    print("  Passed.")

def test_collect_candidates_returns_correct_fields():
    print("Test: _collect_candidates returns correct fields with valid constraints")
    tmp = get_temp_dir()
    try:
        bm_dir = tmp / 'mini_sift'
        create_mini_sift_benchmark_with_population_data(bm_dir)

        rows, n_queries = _collect_candidates('mini_sift', benchmark_base_dir=tmp)
        assert len(rows) > 0, "Expected at least one surviving candidate with alpha up to 3.0"
        assert n_queries > 0

        for row in rows:
            assert set(row.keys()) == EXPECTED_FIELDS, f"Wrong fields: {set(row.keys())}"
            assert row['alpha'] in ALPHA_VALUES, f"Unexpected alpha: {row['alpha']}"
            assert 0.0 <= row['coverage_rate'] <= 1.0, f"coverage_rate out of range: {row['coverage_rate']}"
    finally:
        cleanup_temp_dir(tmp)
    print("  Passed.")

def test_collect_candidates_missing_cache_r_Q():
    print("Test: _collect_candidates returns ([], 0) when cache_r_Q.npy is missing")
    tmp = get_temp_dir()
    try:
        bm_dir = tmp / 'mini_sift'
        bm_dir.mkdir()
        # create benchmark WITHOUT cache_r_Q.npy
        import json, numpy as np
        np.save(bm_dir / 'base_vectors.npy', np.zeros((10, 4), dtype=np.float32))
        json.dump({'cache_K': 5}, open(bm_dir / 'config.json', 'w'))

        rows, n_queries = _collect_candidates('mini_sift', benchmark_base_dir=tmp)
        assert rows == [], f"Expected [], got {len(rows)} rows"
        assert n_queries == 0
    finally:
        cleanup_temp_dir(tmp)
    print("  Passed.")

def test_save_csv_writes_correct_structure():
    print("Test: _save_csv writes correct header and row count")
    tmp = get_temp_dir()
    try:
        sample_rows = [
            {'alpha': 1.5, 'r_Q': 0.1, 'r_c': 0.15, 'R_c_size': 10, 'n_covered': 3,
             'coverage_rate': 0.3},
            {'alpha': 2.0, 'r_Q': 0.1, 'r_c': 0.2, 'R_c_size': 20, 'n_covered': 5,
             'coverage_rate': 0.5},
        ]
        out = tmp / 'out' / 'candidates.csv'
        _save_csv(sample_rows, out)

        assert out.exists(), "CSV file was not created"
        with open(out) as f:
            reader = csv.DictReader(f)
            read_rows = list(reader)
        assert set(reader.fieldnames) == EXPECTED_FIELDS
        assert len(read_rows) == 2
        assert float(read_rows[0]['alpha']) == 1.5
        assert int(read_rows[1]['R_c_size']) == 20
    finally:
        cleanup_temp_dir(tmp)
    print("  Passed.")

def test_save_csv_empty_rows_no_file():
    print("Test: _save_csv with empty rows does not create file")
    tmp = get_temp_dir()
    try:
        out = tmp / 'candidates.csv'
        _save_csv([], out)
        assert not out.exists(), "File should not be created for empty rows"
    finally:
        cleanup_temp_dir(tmp)
    print("  Passed.")

def test_per_benchmark_plots_creates_files():
    print("Test: _per_benchmark_plots creates PNG files")
    tmp = get_temp_dir()
    try:
        bm_dir = tmp / 'mini_sift'
        create_mini_sift_benchmark_with_population_data(bm_dir)
        rows, _ = _collect_candidates('mini_sift', benchmark_base_dir=tmp)
        assert len(rows) > 0, "Need candidates to test plotting"

        out = tmp / 'plots'
        _per_benchmark_plots(rows, 'mini_sift', out)

        for fname in ['rc_size_vs_alpha.png', 'coverage_vs_alpha.png',
                      'rc_vs_coverage_scatter.png']:
            p = out / fname
            assert p.exists(), f"Missing: {fname}"
            assert p.stat().st_size > 0, f"Empty file: {fname}"
    finally:
        cleanup_temp_dir(tmp)
    print("  Passed.")

def test_aggregate_plots_creates_files():
    print("Test: _aggregate_plots creates PNG files")
    tmp = get_temp_dir()
    try:
        bm_dir = tmp / 'mini_sift'
        create_mini_sift_benchmark_with_population_data(bm_dir)
        rows, _ = _collect_candidates('mini_sift', benchmark_base_dir=tmp)
        assert len(rows) > 0, "Need candidates to test plotting"

        out = tmp / 'aggregate'
        _aggregate_plots({'mini_sift': rows}, out)

        for fname in ['rc_size_vs_alpha.png', 'coverage_vs_alpha.png',
                      'rc_vs_coverage_scatter.png']:
            p = out / fname
            assert p.exists(), f"Missing: {fname}"
            assert p.stat().st_size > 0, f"Empty file: {fname}"
    finally:
        cleanup_temp_dir(tmp)
    print("  Passed.")

def test_aggregate_plots_no_empty_slice_warning():
    print("Test: _aggregate_plots emits no RuntimeWarning about empty slices")
    tmp = get_temp_dir()
    try:
        bm_dir = tmp / 'mini_sift'
        create_mini_sift_benchmark_with_population_data(bm_dir)
        rows, _ = _collect_candidates('mini_sift', benchmark_base_dir=tmp)
        assert len(rows) > 0, "Need candidates to test warning behavior"

        out = tmp / 'aggregate_warn'
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _aggregate_plots({'mini_sift': rows}, out)

        runtime_warns = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert len(runtime_warns) == 0, \
            f"Unexpected RuntimeWarning(s): {[str(w.message) for w in runtime_warns]}"
    finally:
        cleanup_temp_dir(tmp)
    print("  Passed.")

#-------------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Testing simulations/sift/microbenchmark_candidates.py")
    print("=" * 70)

    test_output_label()
    test_collect_candidates_returns_correct_fields()
    test_collect_candidates_missing_cache_r_Q()
    test_save_csv_writes_correct_structure()
    test_save_csv_empty_rows_no_file()
    test_per_benchmark_plots_creates_files()
    test_aggregate_plots_creates_files()
    test_aggregate_plots_no_empty_slice_warning()

    print("\n" + "=" * 70)
    print("All tests passed.")
    print("=" * 70)

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
