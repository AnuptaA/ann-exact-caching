#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Rebuilds the plots from saved tables.

Plots 1 and 2 for every stage; plots 3 to 5 for stages 3 and 4.
Run from the thesis root:  python3 .../rt_replot.py [--results-dir DIR] [--plots-dir DIR]"""

import argparse
import sys
from pathlib import Path

import rt_config
import rt_plots
import rt_tables

def replot(results_dir: Path, plots_dir: Path, dataset: str, stage: int) -> list:
    d = rt_tables.load_stage(results_dir, dataset, stage)
    plots_dir.mkdir(parents=True, exist_ok=True)
    out = [rt_plots.plot_scatter(d["rows"], d["cov"], dataset, stage, plots_dir),
           rt_plots.plot_mean_error_vs_nu(d["agg"], dataset, stage, plots_dir)]
    if stage in rt_config.DUMP_STAGES:
        out += [rt_plots.plot_bias_profile(d["rows"], d["cov"], dataset, stage, plots_dir),
                rt_plots.plot_neighbor_error(d["dump"], d["cov"], dataset, stage, plots_dir),
                rt_plots.plot_error_vs_geometry(d["rows"], d["cov"], dataset, stage, plots_dir)]
    return out

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=str(rt_config.RESULTS_DIR))
    ap.add_argument("--plots-dir", default=str(rt_config.PLOTS_DIR))
    ap.add_argument("--stages", default="0,1,2,3,4")
    ap.add_argument("--datasets", default="sift,esci")
    args = ap.parse_args()
    for stage in [int(x) for x in args.stages.split(",")]:
        for ds in args.datasets.split(","):
            if not (Path(args.results_dir) / f"{ds}_stage{stage}.ok").exists():
                print(f"skip {ds} stage {stage}: not complete")
                continue
            for p in replot(Path(args.results_dir), Path(args.plots_dir), ds, stage):
                print(p)
    return 0

if __name__ == "__main__":
    sys.exit(main())
