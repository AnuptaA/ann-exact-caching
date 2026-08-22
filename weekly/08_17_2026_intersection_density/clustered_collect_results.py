#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Compiles the clustered-synthetic experiment 1/2 plots into one PDF."""

import sys
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent.parent.parent))

from simulations.pdf.report import PdfReportBuilder, latest_run_dir

#-------------------------------------------------------------------------------

PLOTS_BASE = Path(__file__).parent / 'plots'
OUTPUT_BASE = Path(__file__).parent / 'pdf'

PLOT_FILES = [
    ('exp1_clustered_vs_uniform.png', 'Experiment 1: pool+argmax, naive random pairing.'),
    ('exp2_clustered_vs_uniform.png', 'Experiment 2: fixed-cache, controlled angular sweep.'),
]

#-------------------------------------------------------------------------------


def collect_results() -> None:
    run_dir = latest_run_dir(PLOTS_BASE)
    if run_dir is None:
        print(f"No timestamped plot output found under {PLOTS_BASE}")
        return

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_pdf = OUTPUT_BASE / ts / 'clustered_reruns.pdf'

    print(f"Plot run: {run_dir.name}")
    print(f"Output: {out_pdf}\n")

    with PdfReportBuilder(out_pdf) as report:
        report.cover_page('Clustered Synthetic Reruns', subtitle=[f'Plot run: {run_dir.name}'])

        part_dir = run_dir / 'clustered_reruns'
        for pattern, caption in PLOT_FILES:
            path = part_dir / pattern
            if not path.exists():
                print(f"  skipped (missing): {pattern}")
                continue
            report.image_page(path, caption=caption)
            print(f"  added: {pattern}")

    size_mb = out_pdf.stat().st_size / 1_000_000
    print(f"\nDone: {out_pdf}  ({size_mb:.1f} MB)")

#-------------------------------------------------------------------------------

if __name__ == '__main__':
    collect_results()
