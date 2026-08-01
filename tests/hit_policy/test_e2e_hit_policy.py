#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""End-to-end test for the SIFT/ESCI simulation workflow using synthetic mini data.

Steps tested:
  1. Create mini benchmark (synthetic vectors)
  2. Run run_benchmark() for the canonical no-union lemma algorithms
  3. Verify output files: summary.json, per_query.csv, per-algo JSONs, logs
  4. Run analyze() and verify CSVs + summary JSON
"""

import json
import importlib

import pytest

from tests.helpers import create_mini_benchmark, get_temp_dir, cleanup_temp_dir

#-------------------------------------------------------------------------------

# lemma1/lemma2/combined (union-based) are deprecated -- see simulations/core/simulate.py.
ALGORITHMS = ['lemma1_no_union', 'lemma2_no_union']


def _run_module(dataset: str):
    return importlib.import_module('simulations.hit_policy.run_hit_policy')


def _analyze_module(dataset: str):
    return importlib.import_module('simulations.hit_policy.analyze_hit_policy')

#-------------------------------------------------------------------------------


@pytest.mark.parametrize('dataset', ['sift', 'esci'])
def test_e2e_workflow(dataset):
    """End-to-end test of complete SIFT/ESCI workflow."""
    run_mod = _run_module(dataset)
    analyze_mod = _analyze_module(dataset)

    tmpdir = get_temp_dir(f"e2e_{dataset}_")
    try:
        benchmark_name = "test_benchmark"
        benchmark_dir = tmpdir / benchmark_name
        raw_dir = tmpdir / "raw"
        processed_dir = tmpdir / "processed"

        config = create_mini_benchmark(dataset, benchmark_dir)
        print(f"  Cache queries: {config['num_cache_queries']} (K={config['cache_K']})")
        print(f"  Test queries:  {config['num_test_queries']} (N={config['test_N']})")
        print(f"  Dimension:     {config['dimension']}")

        run_mod.run_benchmark(
            dataset=dataset,
            benchmark_name=benchmark_name,
            algorithms=ALGORITHMS,
            output_dir=str(raw_dir),
            debug=False,
            benchmark_base_dir=str(tmpdir),
        )

        dataset_out = raw_dir / benchmark_name
        assert dataset_out.exists(), f"Missing dataset output dir: {dataset_out}"

        summary_file = dataset_out / "summary.json"
        per_query_file = dataset_out / "per_query.csv"
        logs_dir = dataset_out / "logs"

        assert summary_file.exists(), "Missing summary.json"
        assert per_query_file.exists(), "Missing per_query.csv"
        assert logs_dir.exists(), "Missing logs directory"

        with open(summary_file) as f:
            summary = json.load(f)

        assert len(summary['results']) == len(ALGORITHMS), \
            f"Expected {len(ALGORITHMS)} results, got {len(summary['results'])}"

        lines = per_query_file.read_text().splitlines()
        assert len(lines) > 1, "per_query.csv is empty"

        # both run_{dataset}.py pipelines are documented as angular-metric-only
        for r in summary['results']:
            assert 'hit_rate' in r, f"Missing hit_rate in result: {r['algorithm']}"
            assert r['metric'] == 'angular', f"Expected angular metric, got {r['metric']}"

        analyze_mod.analyze(dataset, str(raw_dir), str(processed_dir))

        assert processed_dir.exists(), "Missing processed directory"
        assert (processed_dir / "summary.csv").exists(), "Missing summary.csv"
        assert (processed_dir / "per_query_all.csv").exists(), "Missing per_query_all.csv"
        assert (processed_dir / "summaries.json").exists(), "Missing summaries.json"

        summary_lines = (processed_dir / "summary.csv").read_text().splitlines()
        assert len(summary_lines) > 1, "summary.csv is empty"
    finally:
        cleanup_temp_dir(tmpdir)


@pytest.mark.parametrize('dataset', ['sift', 'esci'])
def test_e2e_debug_mode(dataset):
    """Test that debug=True limits to 5 queries per benchmark."""
    run_mod = _run_module(dataset)

    tmpdir = get_temp_dir(f"e2e_{dataset}_debug_")
    try:
        benchmark_name = "test_benchmark"
        benchmark_dir = tmpdir / benchmark_name
        raw_dir = tmpdir / "raw"

        create_mini_benchmark(dataset, benchmark_dir)

        results = run_mod.run_benchmark(
            dataset=dataset,
            benchmark_name=benchmark_name,
            algorithms=['lemma1_no_union', 'lemma2_no_union'],
            output_dir=str(raw_dir),
            benchmark_base_dir=str(tmpdir),
            debug=True,
        )

        dataset_out = raw_dir / benchmark_name
        summary_file = dataset_out / "summary.json"
        assert summary_file.exists(), "Missing summary.json in debug mode"

        with open(summary_file) as f:
            summary = json.load(f)

        # mini benchmark has 10 test queries; debug mode limits to 5
        for r in summary['results']:
            assert r['total_queries'] == 5, \
                f"Debug mode should limit to 5 queries, got {r['total_queries']}"

        per_query_file = dataset_out / "per_query.csv"
        assert per_query_file.exists(), "Missing per_query.csv"
        lines = per_query_file.read_text().splitlines()
        # header + up to 5 data rows per algorithm (2 algorithms here)
        assert len(lines) <= 1 + 5 * 2, f"Debug mode should limit queries, got {len(lines)-1}"
    finally:
        cleanup_temp_dir(tmpdir)
