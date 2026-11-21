#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Analysis-pipeline test for simulations/synthetic/analyze_synthetic.py.

analyze_partial() below is also the manual CLI entry point (run via
`python tests/synthetic/test_analysis.py --raw-dir ...`) for inspecting
partial/complete real simulation runs. test_analyze_partial_produces_plots
exercises the same function against a small hermetic dataset generated on
the fly, so it doesn't depend on any real prior run existing on disk.
"""

import sys, re, argparse, shutil, tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from datasets.synthetic.generator import DatasetConfig, SyntheticDatasetGenerator
from simulations.synthetic.run_synthetic import run_synthetic_simulations
from simulations.synthetic.analyze_synthetic import (
    load_summaries,
    build_summary_df,
    load_per_query_df,
    plot_global_accuracy_heatmap,
    plot_global_hit_rate_heatmap,
    plot_per_set_hit_rate_heatmaps,
    plot_avg_time,
    plot_set1,
    plot_set2,
    plot_set3,
    plot_set4,
    plot_set6,
)

#-------------------------------------------------------------------------------

def find_latest_raw_dir() -> Path:
    base = Path('simulations/synthetic/raw')
    stamped = sorted(
        d for d in base.iterdir()
        if d.is_dir() and re.fullmatch(r'\d{8}_\d{6}', d.name)
    )
    if stamped:
        return stamped[-1]

    return base

#-------------------------------------------------------------------------------

def analyze_partial(raw_dir: Path, verbose: bool = False):
    """
    Run the full analysis pipeline on whatever summaries exist in raw_dir.
    Partial datasets (simulation still running) are fine.
    """
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path('simulations/synthetic/processed') / f'test_{ts}'

    dirs = {s: outdir / s for s in ['global','set1','set2','set3','set4','set6']}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # load
    summaries = load_summaries(str(raw_dir))
    if not summaries:
        print("No summary.json files found yet in raw_dir, cannot analyze")
        return

    df = build_summary_df(summaries)
    pq = load_per_query_df(str(raw_dir), summaries)

    sets_present = sorted(df['set_prefix'].unique())
    print(f"\nDatasets loaded: {len(summaries)}")
    print(f"Sets present: {sets_present}")
    print(f"Per-query rows: {len(pq):,}" if not pq.empty else "Per-query rows: 0 (per_query.csv not yet written)")

    if verbose:
        print("\nAlgorithms found:", sorted(df['algorithm'].unique()))
        print("Metrics found:", sorted(df['metric'].unique()))
        print()

    # global
    print("\n[Global]")
    plot_global_accuracy_heatmap(df, dirs['global'])
    plot_global_hit_rate_heatmap(df, dirs['global'])
    plot_per_set_hit_rate_heatmaps(df, dirs['global'])
    plot_avg_time(df, dirs['global'])

    # per set
    print("\n[Set 1]")
    plot_set1(df, dirs['set1'])

    print("\n[Set 2]")
    plot_set2(df, pq, dirs['set2'])

    print("\n[Set 3]")
    plot_set3(df, dirs['set3'])

    print("\n[Set 4]")
    plot_set4(df, pq, dirs['set4'])

    print("\n[Set 6]")
    plot_set6(df, dirs['set6'])

    # summary
    print()
    print("=" * 70)
    total_pngs = len(list(outdir.rglob("*.png")))
    print(f"Done. {total_pngs} PNGs written to: {outdir}.")
    print("=" * 70)
    return outdir


#-------------------------------------------------------------------------------

def test_analyze_partial_produces_plots():
    """analyze_partial() must produce per-set + global PNGs from a small,
    hermetically-generated raw_dir (no dependency on a real prior run)."""
    tmpdir = Path(tempfile.mkdtemp(prefix="thesis_analysis_test_"))
    raw_dir = tmpdir / "raw"
    outdir = None
    try:
        cfg = DatasetConfig(
            name="set1_baseline_seed42",
            num_cache_queries=20, num_test_queries=10,
            K=5, N=5, perturbation_level=2,
            test_query_strategy="similar",
            num_base_vectors=100, dimension=16, seed=42,
        )
        ds_dir = tmpdir / "datasets" / cfg.name
        g = SyntheticDatasetGenerator(cfg)
        g.generate()
        g.save(str(tmpdir / "datasets"))

        run_synthetic_simulations(
            str(ds_dir),
            algorithms=["lemma1_no_union", "lemma2_no_union", "brute"],
            metrics=["euclidean", "angular"],
            output_dir=str(raw_dir),
            use_cache=True,
            num_workers=1,
        )

        outdir = analyze_partial(raw_dir)
        assert outdir is not None, "analyze_partial returned None (no summaries found)"
        pngs = list(outdir.rglob("*.png"))
        assert len(pngs) > 0, "analyze_partial produced no PNGs"
        assert (outdir / "global").exists(), "Missing global/ output dir"
        assert (outdir / "set1").exists(), "Missing set1/ output dir"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        # analyze_partial() always writes under the real project processed/
        # dir (it's also a manual CLI tool, not just a test helper), so
        # clean up the test-generated output here to avoid accumulation.
        if outdir is not None and outdir.exists():
            shutil.rmtree(outdir, ignore_errors=True)

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run analysis on partial or complete raw simulation results."
    )
    parser.add_argument(
        '--raw-dir', default=None,
        help="Path to raw dir (default: latest timestamped run)"
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help="Print extra info about algorithms/metrics found"
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir) if args.raw_dir else find_latest_raw_dir()
    analyze_partial(raw_dir, verbose=args.verbose)
