#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys, shutil, json, tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from datasets.synthetic.generator import DatasetConfig, SyntheticDatasetGenerator
from simulations.synthetic.run_synthetic import run_synthetic_simulations
from simulations.synthetic.analyze_synthetic import analyze

#-------------------------------------------------------------------------------

# lemma1/lemma2/combined (union-based) are deprecated, see simulations/core/simulate.py.
ALGOS = ["lemma1_no_union", "lemma2_no_union", "brute"]
METRICS = ["euclidean", "angular"]

#-------------------------------------------------------------------------------

# tiny common params to run fast
TINY = dict(num_base_vectors=200, dimension=16, seed=42)

MINI_CONFIGS = [
    # set1: baseline
    DatasetConfig(name="set1_baseline_seed42",
                  num_cache_queries=30, num_test_queries=15,
                  K=5, N=5, perturbation_level=2,
                  test_query_strategy="similar", **TINY),
    # set2: three perturbation levels
    DatasetConfig(name="set2_pert_small_seed42",
                  num_cache_queries=30, num_test_queries=15,
                  K=5, N=5, perturbation_level=2,
                  test_query_strategy="similar", **TINY),
    DatasetConfig(name="set2_pert_medium_seed42",
                  num_cache_queries=30, num_test_queries=15,
                  K=5, N=5, perturbation_level=1,
                  test_query_strategy="similar", **TINY),
    DatasetConfig(name="set2_pert_large_seed42",
                  num_cache_queries=30, num_test_queries=15,
                  K=5, N=5, perturbation_level=0,
                  test_query_strategy="similar", **TINY),
    # set3: two (K, N) combos
    DatasetConfig(name="set3_kn_K5_N5_seed42",
                  num_cache_queries=30, num_test_queries=15,
                  K=5, N=5, perturbation_level=2,
                  test_query_strategy="similar", **TINY),
    DatasetConfig(name="set3_kn_K5_N10_seed42",
                  num_cache_queries=30, num_test_queries=15,
                  K=5, N=10, perturbation_level=2,
                  test_query_strategy="similar", **TINY),
    # set4: variable N
    DatasetConfig(name="set4_varn_low_seed42",
                  num_cache_queries=30, num_test_queries=15,
                  K=15, N=0, N_range=(5, 10),
                  perturbation_level=2,
                  test_query_strategy="similar", **TINY),
    DatasetConfig(name="set4_varn_high_seed42",
                  num_cache_queries=30, num_test_queries=15,
                  K=15, N=0, N_range=(3, 12),
                  perturbation_level=2,
                  test_query_strategy="similar", **TINY),
    # set5: union effectiveness (two cluster counts)
    DatasetConfig(name="set5_union_X5_seed42",
                  num_cache_queries=30, num_test_queries=15,
                  K=5, N=5, perturbation_level=2,
                  test_query_strategy="union_clustered",
                  num_clusters=5, **TINY),
    DatasetConfig(name="set5_union_X10_seed42",
                  num_cache_queries=30, num_test_queries=15,
                  K=5, N=5, perturbation_level=2,
                  test_query_strategy="union_clustered",
                  num_clusters=10, **TINY),
    # set6: two cache sizes
    DatasetConfig(name="set6_cache_20_seed42",
                  num_cache_queries=20, num_test_queries=15,
                  K=5, N=5, perturbation_level=2,
                  test_query_strategy="similar", **TINY),
    DatasetConfig(name="set6_cache_30_seed42",
                  num_cache_queries=30, num_test_queries=15,
                  K=5, N=5, perturbation_level=2,
                  test_query_strategy="similar", **TINY),
]

#-------------------------------------------------------------------------------

def test_e2e_mini():
    tmpdir = Path(tempfile.mkdtemp(prefix="thesis_minitest_"))
    datasets_dir = tmpdir / "datasets"
    raw_dir = tmpdir / "raw"
    processed_dir = tmpdir / "processed"

    try:
        # step 1: generate
        print("=" * 70)
        print(f"[1/3] Generating {len(MINI_CONFIGS)} mini datasets in {datasets_dir}")
        print("=" * 70)
        for cfg in MINI_CONFIGS:
            g = SyntheticDatasetGenerator(cfg)
            g.generate()
            g.save(str(datasets_dir))
            print(f"{cfg.name}")

        generated = [d for d in datasets_dir.iterdir() if d.is_dir()]
        assert len(generated) == len(MINI_CONFIGS), \
            f"Expected {len(MINI_CONFIGS)} dataset dirs, got {len(generated)}"

        # step 2: simulate
        print()
        print("=" * 70)
        print(f"[2/3] Running simulations in {raw_dir}")
        print("=" * 70)
        for ds_path in sorted(generated):
            run_synthetic_simulations(
                str(ds_path),
                algorithms=ALGOS,
                metrics=METRICS,
                output_dir=str(raw_dir),
                use_cache=True,
                num_workers=1,
            )

        # verify outputs
        raw_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
        assert len(raw_dirs) == len(MINI_CONFIGS), \
            f"Expected {len(MINI_CONFIGS)} raw result dirs, got {len(raw_dirs)}"

        print("\nVerifying raw outputs...")
        for rd in raw_dirs:
            summary_file = rd / "summary.json"
            per_query_file = rd / "per_query.csv"
            assert summary_file.exists(), f"Missing summary.json in {rd}"
            assert per_query_file.exists(), f"Missing per_query.csv in {rd}"

            with open(summary_file) as f:
                s = json.load(f)
            n_results = len(s['results'])
            expected = len(ALGOS) * len(METRICS)
            assert n_results == expected, \
                f"{rd.name}: expected {expected} results, got {n_results}"

            # check per_query.csv has rows
            lines = per_query_file.read_text().splitlines()
            assert len(lines) > 1, f"per_query.csv in {rd.name} is empty"
            header = lines[0]
            assert 'cache_hit' in header and 'distance_calcs' in header, \
                f"per_query.csv header looks wrong: {header}"

            print(f"{rd.name}: {n_results} results, {len(lines)-1} per-query rows")

        # step 3: analyze
        print()
        print("=" * 70)
        print(f"[3/3] Running analysis in {processed_dir}")
        print("=" * 70)

        out_base = analyze(str(raw_dir), out_name="minitest")

        print("\nVerifying analysis outputs...")
        expected_subdirs = ["global", "set1", "set2", "set3", "set4", "set6"]
        for sub in expected_subdirs:
            sub_path = out_base / sub
            assert sub_path.exists(), f"Missing subdir: {sub}"
            pngs = list(sub_path.glob("*.png"))
            assert len(pngs) > 0, f"No PNG files in {sub}"
            print(f"{sub}/  in {len(pngs)} PNG(s): {[p.name for p in pngs]}")

        print()
        print("=" * 70)
        print("All checks passed.")
        print("=" * 70)

    finally:
        print(f"\nCleaning up temp dir: {tmpdir}")
        shutil.rmtree(tmpdir, ignore_errors=True)
        # also remove the analysis output written to project processed dir
        minitest_out = Path("simulations/synthetic/processed/minitest")
        if minitest_out.exists():
            shutil.rmtree(minitest_out)
            print(f"Cleaned up {minitest_out}")

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    test_e2e_mini()
