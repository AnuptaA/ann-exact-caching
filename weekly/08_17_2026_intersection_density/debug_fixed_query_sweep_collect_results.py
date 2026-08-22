#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Compiles the fixed-query sweep plots (SIFT + ESCI) into one PDF."""

import sys
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent.parent.parent))

from simulations.pdf.report import PdfReportBuilder, latest_run_dir

#-------------------------------------------------------------------------------

PLOTS_BASE = Path(__file__).parent / 'plots'
OUTPUT_BASE = Path(__file__).parent / 'pdf'

DATASETS = [
    ('SIFT', 128, 'debug_sift_fixed_query_sweep'),
    ('ESCI', 384, 'debug_esci_fixed_query_sweep'),
]

PLOT_FILES = [
    ('{d}d_anchor_overlay.png', 'All anchors overlaid, per K.'),
    ('{d}d_log_scale_scatter.png', 'Fraction vs. recall, log-scale x.'),
    ('{d}d_per_anchor.png', 'Binned calibration, per anchor.'),
]

#-------------------------------------------------------------------------------


def collect_results() -> None:
    run_dir = latest_run_dir(PLOTS_BASE)
    if run_dir is None:
        print(f"No timestamped plot output found under {PLOTS_BASE}")
        return

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_pdf = OUTPUT_BASE / ts / 'fixed_query_sweep.pdf'

    print(f"Plot run: {run_dir.name}")
    print(f"Output: {out_pdf}\n")

    with PdfReportBuilder(out_pdf) as report:
        report.cover_page('Fixed-Query Sweep: Real-Data Trend Confirmation',
                           subtitle=[f'Plot run: {run_dir.name}'])

        for dataset_name, d, part_dir in DATASETS:
            print(f"[{dataset_name}]")
            report.divider_page(dataset_name, run_dir.name)
            for pattern, caption in PLOT_FILES:
                path = run_dir / part_dir / pattern.format(d=d)
                if not path.exists():
                    print(f"  skipped (missing): {path.name}")
                    continue
                report.image_page(path, caption=caption)
                print(f"  added: {path.name}")

    size_mb = out_pdf.stat().st_size / 1_000_000
    print(f"\nDone: {out_pdf}  ({size_mb:.1f} MB)")

#-------------------------------------------------------------------------------

if __name__ == '__main__':
    collect_results()
