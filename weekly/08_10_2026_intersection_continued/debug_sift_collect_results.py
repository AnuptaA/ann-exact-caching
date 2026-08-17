#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Compile the SIFT validation's output PNGs into a standalone PDF.
"""

import sys
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent.parent.parent))

from simulations.pdf.report import PdfReportBuilder, latest_run_dir, divider, header, render_sections

#-------------------------------------------------------------------------------

PLOTS_BASE = Path(__file__).parent / 'plots'
OUTPUT_BASE = Path(__file__).parent / 'pdf'
PART_DIR = 'debug_sift'
D = 128

PLOT_FILES = [
    '{d}d_binned_calibration.png',
    '{d}d_calibration_scatter.png',
]

#-------------------------------------------------------------------------------


def collect_results() -> None:
    run_dir = latest_run_dir(PLOTS_BASE)
    if run_dir is None:
        print(f"No timestamped plot output found under {PLOTS_BASE}")
        return

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_pdf = OUTPUT_BASE / ts / 'debug_sift_validation.pdf'

    print(f"Plot run: {run_dir.name}")
    print(f"Output: {out_pdf}\n")

    part_dir = run_dir / PART_DIR
    children = []
    for pattern in PLOT_FILES:
        path = part_dir / pattern.format(d=D)
        children.append(header(None, lambda p=path: [p] if p.exists() else []))
    section = divider('SIFT Validation', children)

    with PdfReportBuilder(out_pdf) as report:
        report.cover_page('SIFT Validation of Intersection-Fraction Calibration',
                           subtitle=f'Plot run: {run_dir.name}')
        render_sections(report, [section], run_name=run_dir.name)

    size_mb = out_pdf.stat().st_size / 1_000_000
    print(f"\nDone: {out_pdf}  ({size_mb:.1f} MB)")

#-------------------------------------------------------------------------------

if __name__ == '__main__':
    collect_results()
