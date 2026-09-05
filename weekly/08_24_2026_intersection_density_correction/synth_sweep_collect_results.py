#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Compiles the synthetic angular-sweep IF-vs-recall plots (SIFT, ESCI)
into a standalone PDF."""

import sys
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent.parent.parent))

from simulations.pdf.report import PdfReportBuilder, latest_run_dir, divider, header, render_sections

#-------------------------------------------------------------------------------

PLOTS_BASE = Path(__file__).parent / 'plots'
OUTPUT_BASE = Path(__file__).parent / 'pdf'
DATASET_DIMS = {'sift': 128, 'esci': 384}

PLOT_FILES = [
    '{dataset}_{d}d_if_coverage.png',
    '{dataset}_{d}d_scatter.png',
    '{dataset}_{d}d_binned.png',
]

#-------------------------------------------------------------------------------


def collect_results() -> None:
    run_dir = latest_run_dir(PLOTS_BASE)
    if run_dir is None:
        print(f"No timestamped plot output found under {PLOTS_BASE}")
        return

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_pdf = OUTPUT_BASE / ts / 'synth_sweep_true_radii_if_vs_recall.pdf'

    print(f"Plot run: {run_dir.name}")
    print(f"Output: {out_pdf}\n")

    sections = []
    for dataset, d in DATASET_DIMS.items():
        children = []
        for pattern in PLOT_FILES:
            path = run_dir / pattern.format(dataset=dataset, d=d)
            children.append(header(None, lambda p=path: [p] if p.exists() else []))
        sections.append(divider(dataset.upper(), children))

    with PdfReportBuilder(out_pdf) as report:
        report.cover_page('Synthetic Angular-Sweep IF-vs-Recall (True Radii, Validation Stage)',
                           subtitle=f'Plot run: {run_dir.name}')
        render_sections(report, sections, run_name=run_dir.name)

    size_mb = out_pdf.stat().st_size / 1_000_000
    print(f"\nDone: {out_pdf}  ({size_mb:.1f} MB)")

#-------------------------------------------------------------------------------

if __name__ == '__main__':
    collect_results()
