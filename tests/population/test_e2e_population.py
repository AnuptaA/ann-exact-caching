#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import functools
import json

import numpy as np
import pytest

from datasets.dataloaders import load_sift_benchmark, load_esci_benchmark
from simulations.population.population_runner import run_population_benchmark
from simulations.core.simulate import CacheSimulator
from utils.base import KVCache
from utils.base.population import propose_candidates
from utils.base.selection import select_candidates
from tests.helpers import (
    create_mini_benchmark_with_population_data,
    get_temp_dir,
    cleanup_temp_dir,
)

#-------------------------------------------------------------------------------

STRATEGIES = ['random', 'greedy_naive', 'greedy_cost_static', 'greedy_cost']

LOAD_BENCHMARK_FN = {'sift': load_sift_benchmark, 'esci': load_esci_benchmark}

#-------------------------------------------------------------------------------


@pytest.mark.parametrize('dataset', ['sift', 'esci'])
def test_e2e_population_workflow(dataset):
    """
    End-to-end test of the full population pipeline on a mini benchmark.

    Steps:
    1. Create mini benchmark with population data
    2. Propose candidates
    3. For each strategy: select, verify invariants, build cache, run simulation
    """
    tmpdir = get_temp_dir(f"{dataset}_e2e_population_")
    try:
        benchmark_dir = tmpdir / "test_benchmark"
        config = create_mini_benchmark_with_population_data(dataset, benchmark_dir)
        K = config['cache_K']

        base = np.load(benchmark_dir / 'base_vectors.npy')
        cache_q = np.load(benchmark_dir / 'cache_queries.npy')
        r_Q_all = np.load(benchmark_dir / 'cache_r_Q.npy')
        test_r_t = np.load(benchmark_dir / 'test_r_t.npy')
        test_q = np.load(benchmark_dir / 'test_queries.npy')

        # build top_k_vectors and related arrays from cache GT
        cache_gt = np.load(benchmark_dir / f'cache_gt_population_K{K}.npz')
        tv_all, ti_all, hg_all = [], [], []
        for i in range(len(cache_q)):
            indices = cache_gt[f'indices_{i}'].tolist()
            tv_all.append([base[idx] for idx in indices])
            ti_all.append(indices)
            hg_all.append(float(cache_gt[f'half_gap_{i}']))
        hg_arr = np.array(hg_all, dtype=np.float32)

        candidates = propose_candidates(
            cache_q, r_Q_all, tv_all, ti_all, hg_arr,
            cache_q, r_Q_all,
            test_q, test_r_t,
            base, alpha_values=[5.0], metric='angular',
        )
        assert len(candidates) > 0, "Expected candidates with alpha=5.0"

        for strategy in STRATEGIES:
            selected, trajectory = select_candidates(
                candidates, budget_B=100,
                strategy=strategy,
                n_train_queries=len(cache_q),
                n_test_queries=len(test_q),
                seed=42,
            )

            # invariants
            cached_set = np.empty(0, dtype=np.int32)
            for c in selected:
                cached_set = np.union1d(cached_set, c.R_c)
            assert len(cached_set) <= 100, f"cached_set={len(cached_set)} > 100"
            assert len(trajectory) == len(selected)
            for i in range(1, len(trajectory)):
                assert trajectory[i]['cig_train_coverage'] >= trajectory[i-1]['cig_train_coverage'] - 1e-15
                assert trajectory[i]['vectors_cached'] >= trajectory[i-1]['vectors_cached']

            # build KVCache
            kv_cache = KVCache(metric='angular')
            for cand in selected:
                entry = cand.to_cache_entry()
                kv_cache.add_entry(
                    entry.query, entry.top_k_vectors,
                    entry.r_Q, entry.half_gap, entry.top_k_indices,
                )
            assert kv_cache.size() == len(selected)

            # inject into CacheSimulator and run lemma1
            simulator = CacheSimulator(
                base_vectors=base,
                cache_queries=cache_q[:1],
                test_queries=cache_q,
                K=K, N=K, metric='angular',
            )
            simulator.cache = kv_cache

            results = []
            for i, query in enumerate(cache_q):
                r = simulator.run_query(i, query, 'lemma1_no_union', ground_truth_indices=[])
                results.append(r)
                assert r.error is None, f"Unexpected error at query {i}: {r.error}"

            hit_rate = sum(r.cache_hit for r in results) / len(results)
            print(f"  [{dataset}/{strategy}] {len(selected)} entries | sim hit rate: {hit_rate:.1%}")
    finally:
        cleanup_temp_dir(tmpdir)


@pytest.mark.parametrize('dataset', ['sift', 'esci'])
def test_e2e_population_output_files(dataset):
    """Test that run_population_benchmark output files have correct structure."""
    tmpdir = get_temp_dir(f"{dataset}_e2e_popfiles_")
    try:
        benchmark_dir = tmpdir / "test_benchmark"
        config = create_mini_benchmark_with_population_data(dataset, benchmark_dir)
        K = config['cache_K']

        base = np.load(benchmark_dir / 'base_vectors.npy')
        cache_q = np.load(benchmark_dir / 'cache_queries.npy')
        r_Q_all = np.load(benchmark_dir / 'cache_r_Q.npy')
        test_r_t = np.load(benchmark_dir / 'test_r_t.npy')
        test_q = np.load(benchmark_dir / 'test_queries.npy')

        cache_gt = np.load(benchmark_dir / f'cache_gt_population_K{K}.npz')
        tv_all, ti_all, hg_all = [], [], []
        for i in range(len(cache_q)):
            indices = cache_gt[f'indices_{i}'].tolist()
            tv_all.append([base[idx] for idx in indices])
            ti_all.append(indices)
            hg_all.append(float(cache_gt[f'half_gap_{i}']))
        hg_arr = np.array(hg_all, dtype=np.float32)

        candidates = propose_candidates(
            cache_q, r_Q_all, tv_all, ti_all, hg_arr,
            cache_q, r_Q_all,
            test_q, test_r_t,
            base, alpha_values=[5.0], metric='angular',
        )

        output_dir = tmpdir / "results"

        for strategy in STRATEGIES:
            selected, trajectory = select_candidates(
                candidates, 50, strategy,
                len(cache_q), len(test_q), seed=42,
            )

            traj_file = output_dir / "test_benchmark" / f"{strategy}_trajectory.json"
            traj_file.parent.mkdir(parents=True, exist_ok=True)
            with open(traj_file, 'w') as f:
                json.dump(trajectory, f)

            assert traj_file.exists(), f"Missing trajectory file for {strategy}"

            with open(traj_file) as f:
                traj_data = json.load(f)
            assert isinstance(traj_data, list), "Trajectory must be a list"
            for entry in traj_data:
                assert 'num_entries' in entry
                assert 'vectors_cached' in entry
                assert 'cig_train_coverage' in entry
                assert 'cig_test_coverage' in entry

        run_ts = "test_run"
        propose_fn = functools.partial(propose_candidates, alpha_values=[5.0])
        run_population_benchmark(
            benchmark_name="test_benchmark",
            load_benchmark_fn=LOAD_BENCHMARK_FN[dataset],
            metric='angular',
            output_dir=output_dir,
            benchmark_base_dir=tmpdir,
            propose_fn=propose_fn,
            budget_values=[100],
            strategies=['random'],
            algorithms=['lemma1_no_union'],
            seed=42,
            run_ts=run_ts,
        )

        strat_dir = output_dir / run_ts / "test_benchmark" / "B100" / "oracle" / "random"
        assert strat_dir.exists(), "Oracle strategy output directory missing"

        non_oracle_dir = output_dir / run_ts / "test_benchmark" / "B100" / "non_oracle" / "random"
        assert non_oracle_dir.exists(), "Non-oracle strategy output directory missing"

        per_query = strat_dir / "per_query.csv"
        assert per_query.exists(), "per_query.csv missing"
        lines = per_query.read_text().splitlines()
        header = lines[0].split(',')
        expected_cols = ['dataset', 'algorithm', 'metric', 'query_id',
                         'cache_hit', 'hit_source', 'distance_calcs', 'time_us']
        assert header == expected_cols, f"Wrong columns: {header}"
        assert len(lines) > 1, "per_query.csv has no data rows"

        summary_file = strat_dir / "summary.json"
        assert summary_file.exists(), "summary.json missing"
        with open(summary_file) as f:
            summary = json.load(f)
        assert 'dataset' in summary, "summary.json missing 'dataset'"
        assert 'config' in summary, "summary.json missing 'config'"
        assert 'results' in summary, "summary.json missing 'results'"
    finally:
        cleanup_temp_dir(tmpdir)
