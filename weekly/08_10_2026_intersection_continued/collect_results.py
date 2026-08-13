#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Compile the intersection-fraction calibration study's output PNGs
into a single PDF, split by part then by dimension.
"""

import sys
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent.parent.parent))

from simulations.pdf.report import PdfReportBuilder, latest_run_dir, divider, header, render_sections

#-------------------------------------------------------------------------------

PLOTS_BASE = Path(__file__).parent / 'plots'
OUTPUT_BASE = Path(__file__).parent / 'pdf'

FULL_DIMS = [(d, f'{d}D') for d in (2, 3, 8, 16, 32, 64, 128, 256, 384)]

POOL_PLOT_FILES = [
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

CONTROLLED_PLOT_FILES = [
    '{d}d_binned_calibration.png',
    '{d}d_calibration_scatter.png',
    '{d}d_method_comparison_scatter.png',
    '{d}d_aggregate_summary.png',
]

DIAGNOSTIC_PLOT_FILES = [
    '{d}d_recall_vs_coverage.png',
    '{d}d_binned_calibration.png',
    '{d}d_method_comparison_scatter.png',
]

ENLARGED_POOL_PLOT_FILES = [
    '{d}d_recall_vs_coverage.png',
    '{d}d_binned_calibration.png',
    '{d}d_method_comparison_scatter.png',
]

PARTS = [
    ('part1', 'Part 1: Higher-Dimension Extension (Pool + Argmax)', POOL_PLOT_FILES, FULL_DIMS),
    ('part2', 'Part 2: Controlled Construction (Fixed Cache Pole, Delta Sweep)', CONTROLLED_PLOT_FILES, FULL_DIMS),
    ('part3', 'Part 3: r_cache Diagnostic (Fixed Cache Pole, Fixed Delta, Coverage-Fraction Sweep)',
     DIAGNOSTIC_PLOT_FILES, FULL_DIMS),
    ('part4', 'Part 4: Enlarged-Radius Pool + Argmax (Coverage-Fraction Cache Radius)',
     ENLARGED_POOL_PLOT_FILES, FULL_DIMS),
]

#-------------------------------------------------------------------------------


def _plot_children(d: int, part_dir: Path, patterns: list) -> list:
    children = []
    for pattern in patterns:
        filename = pattern.format(d=d)
        path = part_dir / filename
        children.append(header(None, lambda p=path: [p] if p.exists() else []))
    return children


def _part_section(part_dir_name: str, part_title: str, plot_files: list, dims: list, run_dir: Path) -> dict:
    part_dir = run_dir / part_dir_name
    children = [
        divider(dim_title, _plot_children(d, part_dir, plot_files))
        for d, dim_title in dims
    ]
    return divider(part_title, children)


def collect_results() -> None:
    run_dir = latest_run_dir(PLOTS_BASE)
    if run_dir is None:
        print(f"No timestamped plot output found under {PLOTS_BASE}")
        return

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_pdf = OUTPUT_BASE / ts / 'intersection_fraction_multidim_results.pdf'

    print(f"Plot run: {run_dir.name}")
    print(f"Output: {out_pdf}\n")

    with PdfReportBuilder(out_pdf) as report:
        report.cover_page('Multi-Dimensional Synthetic Intersection-Fraction Calibration Study',
                           subtitle=f'Plot run: {run_dir.name}')
        for part_dir_name, part_title, plot_files, dims in PARTS:
            print(f"[{part_title}]")
            section = _part_section(part_dir_name, part_title, plot_files, dims, run_dir)
            render_sections(report, [section], run_name=run_dir.name)

    size_mb = out_pdf.stat().st_size / 1_000_000
    print(f"\nDone: {out_pdf}  ({size_mb:.1f} MB)")

#-------------------------------------------------------------------------------

if __name__ == '__main__':
    collect_results()
