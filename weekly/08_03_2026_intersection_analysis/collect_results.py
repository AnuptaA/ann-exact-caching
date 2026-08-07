#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Compile the intersection-fraction calibration study's output PNGs 
into a single PDF.
"""

import sys
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent.parent.parent))

from simulations.pdf.report import PdfReportBuilder, latest_run_dir, divider, header, render_sections

#-------------------------------------------------------------------------------

PLOTS_BASE = Path(__file__).parent / 'plots'
OUTPUT_BASE = Path(__file__).parent / 'pdf'

DIMS = [(2, '2D'), (3, '3D')]

PLOT_FILES = [
    '{d}d_binned_calibration.png',
    '{d}d_binned_calibration_per_background.png',
    '{d}d_calibration_scatter.png',
    '{d}d_point_scatter_cap.png',
    '{d}d_point_scatter_solid.png',
    '{d}d_method_comparison_scatter.png',
    '{d}d_discrepancy_vs_angle_normalized.png',
    '{d}d_discrepancy_vs_angle_raw.png',
    '{d}d_peak_discrepancy_vs_r.png',
    '{d}d_aggregate_summary.png',
    '{d}d_aggregate_summary_raw.png',
    '{d}d_radius_distributions.png',
    '{d}d_selection_agreement.png',
]

#-------------------------------------------------------------------------------


def _dimension_section(d: int, dim_title: str, run_dir: Path) -> dict:
    children = []
    for pattern in PLOT_FILES:
        filename = pattern.format(d=d)
        path = run_dir / filename
        children.append(header(None, lambda p=path: [p] if p.exists() else []))
    return divider(dim_title, children)


def collect_results() -> None:
    run_dir = latest_run_dir(PLOTS_BASE)
    if run_dir is None:
        print(f"No timestamped plot output found under {PLOTS_BASE}")
        return

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_pdf = OUTPUT_BASE / ts / 'intersection_fraction_2d3d_results.pdf'

    print(f"Plot run: {run_dir.name}")
    print(f"Output: {out_pdf}\n")

    with PdfReportBuilder(out_pdf) as report:
        report.cover_page('2D/3D Synthetic Intersection-Fraction Calibration Study',
                           subtitle=f'Plot run: {run_dir.name}')
        for d, dim_title in DIMS:
            print(f"[{dim_title}]")
            section = _dimension_section(d, dim_title, run_dir)
            render_sections(report, [section], run_name=run_dir.name)

    size_mb = out_pdf.stat().st_size / 1_000_000
    print(f"\nDone: {out_pdf}  ({size_mb:.1f} MB)")

#-------------------------------------------------------------------------------

if __name__ == '__main__':
    collect_results()
