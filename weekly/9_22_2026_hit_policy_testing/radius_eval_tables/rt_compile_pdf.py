#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Compiles saved tables and plots into one PDF per dataset. Layout: a cover, then for
each complete stage one text page (run checks and headline table) followed by that
stage's plots."""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rt_config
import rt_tables
from simulations.pdf.report import PAGE_SIZE, PdfReportBuilder

PDF_DIR = rt_config.PDF_DIR
STAGE_NAMES = {0: "uniform database, uniform queries", 1: "uniform database, real queries",
               2: "uniform database, locality queries", 3: "real database, real queries",
               4: "real database, locality queries"}

def text_pages(report: PdfReportBuilder, title: str, lines: list, per_page: int = 52) -> None:
    for start in range(0, len(lines), per_page):
        fig = plt.figure(figsize=PAGE_SIZE)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.text(0.5, 0.97, title, ha="center", va="top", fontsize=15, fontweight="bold", transform=ax.transAxes)
        ax.text(0.04, 0.93, "\n".join(lines[start:start + per_page]), ha="left", va="top", fontsize=6.5,
                family="monospace", transform=ax.transAxes)
        report._pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

def stage_lines(results_dir: Path, dataset: str, stage: int) -> list:
    import json
    s = json.loads((results_dir / f"{dataset}_stage{stage}_summary.json").read_text())
    lines = [f"EXACT max |r_hat - theta| = {s['exact_max_abs_error']} (must be 0)",
             f"per-query rows (S = 10,000, sample+train): {s['per_query_rows']}; aggregate rows: {s['aggregate_rows']}",
             f"seeds {s['seeds']}, n_train {s['n_train']}, n_test {s['n_test']}, N {s['N']}",
             f"test theta mean {s['test_theta_mean']:.4f}, relative sd {100 * s['test_theta_relative_sd']:.2f}%",
             f"wall {s['wall_seconds']:.0f}s, peak RSS {s['peak_rss_mb']:.0f} MB, theta cache hit {s['theta_cache_hit']}",
             f"mean error = mean of (r_hat - theta)/theta in percent; sd = sd over queries; +/- is sd over seeds", ""]
    tables = rt_tables.build_tables(results_dir, dataset, stage)
    for nu in (1, 8, 32):
        lines.append(rt_tables.headline_text(tables, dataset, stage, nus=(nu,)).replace("**", ""))
        lines.append("")
    return lines

def compile_dataset(dataset: str, stages: list, results_dir: Path, plots_dir: Path, pdf_dir: Path) -> Path:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    done = [s for s in stages if (results_dir / f"{dataset}_stage{s}.ok").exists()]
    out = pdf_dir / f"radius_eval_tables_{dataset}.pdf"
    with PdfReportBuilder(out) as report:
        report.cover_page(f"Multi-table LSH radius estimation: {dataset.upper()}",
                          subtitle=["L tables per estimate, train queries inserted as labeled points, no learning",
                                    "stages: " + ", ".join(f"{s} ({STAGE_NAMES[s]})" for s in done)])
        for s in done:
            text_pages(report, f"{dataset} stage {s} ({STAGE_NAMES[s]})", stage_lines(results_dir, dataset, s))
            for img in sorted(plots_dir.glob(f"{dataset}_stage{s}_plot*.png")):
                report.image_page(img)
    print(f"Saved: {out}")
    return out

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", default="0,1,2,3,4")
    ap.add_argument("--datasets", default="sift,esci")
    ap.add_argument("--results-dir", default=str(rt_config.RESULTS_DIR))
    ap.add_argument("--plots-dir", default=str(rt_config.PLOTS_DIR))
    ap.add_argument("--pdf-dir", default=str(PDF_DIR))
    a = ap.parse_args()
    for ds in a.datasets.split(","):
        compile_dataset(ds, [int(x) for x in a.stages.split(",")], Path(a.results_dir), Path(a.plots_dir),
                        Path(a.pdf_dir))
    return 0

if __name__ == "__main__":
    sys.exit(main())
