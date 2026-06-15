#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import json
import numpy as np
from simulations.simulate import CacheSimulator
from utils.base import KVCache
from utils.base.population import propose_candidates
from utils.base.selection import select_candidates
from tests.esci.helpers_population import (
    create_mini_esci_benchmark_with_population_data,
    get_temp_dir,
    cleanup_temp_dir,
    POPULATION_K,
)

#-------------------------------------------------------------------------------

STRATEGIES = ['random', 'greedy_naive', 'greedy_cost_static', 'greedy_cost']

#-------------------------------------------------------------------------------

def test_e2e_population_workflow():
    """End-to-end test of the ESCI population pipeline on a mini benchmark."""
    print("\n" + "=" * 80)
    print("Testing ESCI E2E population workflow")
    print("=" * 80)

    tmpdir = get_temp_dir("esci_e2e_population_")
    try:
        benchmark_dir = tmpdir / "test_benchmark"
        config = create_mini_esci_benchmark_with_population_data(benchmark_dir)
        K = POPULATION_K

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

        print("\n[1/3] Proposing candidates...")
        candidates = propose_candidates(
            cache_q, r_Q_all, tv_all, ti_all, hg_arr,
            cache_q, r_Q_all,
            test_q, test_r_t,
            base, alpha_values=[5.0], metric='angular',
        )
        print(f"  {len(candidates)} candidates")
        assert len(candidates) > 0, "Expected candidates with alpha=5.0"

        print("\n[2/3] Selecting and simulating...")
        for strategy in STRATEGIES:
            print(f"\n  Strategy: {strategy}")

            selected, trajectory = select_candidates(
                candidates, budget_B=100,
                strategy=strategy,
                n_train_queries=len(cache_q),
                n_test_queries=len(test_q),
                seed=42,
            )

            cached_set = set()
            for c in selected:
                cached_set |= c.R_c
            assert len(cached_set) <= 100, f"cached_set={len(cached_set)} > 100"
            assert len(trajectory) == len(selected)
            for i in range(1, len(trajectory)):
                assert trajectory[i]['cig_train_coverage'] >= trajectory[i-1]['cig_train_coverage'] - 1e-15
                assert trajectory[i]['vectors_cached'] >= trajectory[i-1]['vectors_cached']

            kv_cache = KVCache(metric='angular')
            for cand in selected:
                entry = cand.to_cache_entry()
                kv_cache.add_entry(
                    entry.query, entry.top_k_vectors,
                    entry.r_Q, entry.half_gap, entry.top_k_indices,
                )
            assert kv_cache.size() == len(selected)

            simulator = CacheSimulator(
                base_vectors=base,
                cache_queries=cache_q[:1],
                test_queries=cache_q,
                K=K, N=K, metric='angular',
            )
            simulator.cache = kv_cache

            results = []
            for i, query in enumerate(cache_q):
                r = simulator.run_query(i, query, 'lemma1', ground_truth_indices=[])
                results.append(r)
                assert r.error is None or not r.cache_hit, \
                    f"Cache hit with error at query {i}"

            hit_rate = sum(r.cache_hit for r in results) / len(results)
            print(f"    {len(selected)} entries | sim hit rate: {hit_rate:.1%}")

        print("\n[3/3] All strategies verified.")
        print("\n" + "=" * 80)
        print("Passed.")
        print("=" * 80)

    finally:
        cleanup_temp_dir(tmpdir)

#-------------------------------------------------------------------------------

def test_e2e_population_output_files():
    """Test that run_population_esci output files have correct structure."""
    print("\n" + "=" * 80)
    print("Testing ESCI E2E population output files")
    print("=" * 80)

    tmpdir = get_temp_dir("esci_e2e_popfiles_")
    try:
        benchmark_dir = tmpdir / "test_benchmark"
        config = create_mini_esci_benchmark_with_population_data(benchmark_dir)
        K = POPULATION_K

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
            assert traj_file.exists()
            with open(traj_file) as f:
                traj_data = json.load(f)
            assert isinstance(traj_data, list)
            for entry in traj_data:
                assert 'num_entries' in entry
                assert 'vectors_cached' in entry
                assert 'cig_train_coverage' in entry
                assert 'cig_test_coverage' in entry

        print("  Trajectory files: OK")

        from simulations.esci.run_population_esci import run_population_benchmark

        run_ts = "test_run"
        run_population_benchmark(
            benchmark_name="test_benchmark",
            output_dir=output_dir,
            benchmark_base_dir=tmpdir,
            alpha_values=[5.0],
            budget_values=[100],
            strategies=['random'],
            algorithms=['lemma1'],
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
        assert 'dataset' in summary
        assert 'config' in summary
        assert 'results' in summary

        print("  Output files: OK")
        print("\n" + "=" * 80)
        print("Passed.")
        print("=" * 80)

    finally:
        cleanup_temp_dir(tmpdir)

#-------------------------------------------------------------------------------

def main():
    test_e2e_population_workflow()
    test_e2e_population_output_files()

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
