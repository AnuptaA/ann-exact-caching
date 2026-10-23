#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Assembles the report PDFs from saved results.

Run from the thesis root:
    python3 .../rt_report.py --proof                      # SIFT stage 3 proof PDF
    python3 .../rt_report.py --datasets both              # full reports, stages 0 to 4
New files are written next to the existing PDFs; existing files are never overwritten."""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rt_config
import rt_report_data as rd
import rt_report_slides as sl
from simulations.pdf.report import PdfReportBuilder

PDF_DIR = rt_config.PDF_DIR
DPI = 120

def add(report: PdfReportBuilder, fig, kind: str, title: str, manifest: list) -> None:
    report._pdf.savefig(fig, dpi=DPI)
    plt.close(fig)
    manifest.append((len(manifest) + 1, kind, title))

def front_slides(report: PdfReportBuilder, dataset: str, manifest: list) -> None:
    add(report, sl.title_slide(dataset), "title", f"Multi-table LSH radius estimation: {sl.DATASET_NAME[dataset]}", manifest)
    add(report, sl.glossary_slide(), "glossary", "Glossary", manifest)
    add(report, sl.config_slide(), "configurations", "Configurations", manifest)

def stage_slides(report: PdfReportBuilder, data: dict, manifest: list) -> None:
    stage = data["stage"]
    decimals = 4 if stage <= 2 else 2
    db, q = sl.STAGE_TEXT[stage]
    add(report, sl.stage_header_slide(data), "stage header", f"Stage {stage}: {db}, {q}", manifest)
    for nu in (1, 8, 32):
        for fig in sl.slide_table_main(data, nu, decimals):
            add(report, fig, "T1", f"Main table, nu = {nu}", manifest)
    for fig in sl.slide_table_sample_size(data, decimals):
        add(report, fig, "T2", "Sample size table, nu = 8", manifest)
    for fig in sl.slide_table_train_points(data, decimals):
        add(report, fig, "T3", "Train points table, nu = 8", manifest)
    add(report, sl.fig_scatter_theta(data), "P1", "Scatter: r_hat against theta, all configurations", manifest)
    for cname in rt_config.CONFIG_NAMES:
        add(report, sl.fig_scatter_theta(data, cname), "P2", f"Scatter: r_hat against theta, {cname}", manifest)
    add(report, sl.fig_mean_error_vs_nu(data), "P3", "Mean error against nu", manifest)
    if stage >= 3:
        add(report, sl.fig_error_vs_distinct(data), "P4", "Error against distinct labeled points", manifest)
        add(report, sl.fig_bias_profile_a(data), "P5A", "Error against theta, nu = 8", manifest)
        add(report, sl.fig_bias_profile_b(data), "P5B", "Error profile against theta, all configurations", manifest)
        add(report, sl.fig_error_vs_cutoff(data), "P6", "Error against cutoff Hamming distance, nu = 8", manifest)

def build(dataset: str, stages: list, out_path: Path, results_dir: Path) -> list:
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists; refusing to overwrite")
    manifest = []
    with PdfReportBuilder(out_path) as report:
        front_slides(report, dataset, manifest)
        for stage in stages:
            data = rd.load_stage_data(results_dir, dataset, stage)
            stage_slides(report, data, manifest)
            print(f"{dataset} stage {stage}: {len(manifest)} pages so far", file=sys.stderr)
    return manifest

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proof", action="store_true", help="SIFT stage 3 proof PDF only")
    ap.add_argument("--datasets", default="both", choices=["sift", "esci", "both"])
    ap.add_argument("--stages", default="0,1,2,3,4")
    ap.add_argument("--results-dir", default=str(rt_config.RESULTS_DIR))
    ap.add_argument("--out-dir", default=str(PDF_DIR))
    a = ap.parse_args()
    out_dir, results_dir = Path(a.out_dir), Path(a.results_dir)
    if a.proof:
        jobs = [("sift", [3], out_dir / "radius_eval_tables_report_proof_sift_stage3.pdf")]
    else:
        datasets = rt_config.DATASETS if a.datasets == "both" else [a.datasets]
        stages = [int(s) for s in a.stages.split(",")]
        jobs = [(d, stages, out_dir / f"radius_eval_tables_report_{d}.pdf") for d in datasets]
    for dataset, stages, path in jobs:
        manifest = build(dataset, stages, path, results_dir)
        print(f"Saved: {path} ({len(manifest)} pages)")
        for n, kind, title in manifest:
            print(f"  {n:3d}  {kind:15s} {title}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
