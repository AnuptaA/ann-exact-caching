#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Compile the full 2x2 (scheme x corpus) comparison plots (SIFT, ESCI)
into a standalone PDF.
"""

import sys
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent.parent.parent))

from simulations.pdf.report import PdfReportBuilder, latest_run_dir, divider, header, render_sections

#-------------------------------------------------------------------------------

PLOTS_BASE = Path(__file__).parent / 'plots'
OUTPUT_BASE = Path(__file__).parent / 'pdf'
DATASET_DIMS = {'sift': 128, 'esci': 384}
K_VALUES = [25, 50, 100]

PLOT_FILES = [
    '{dataset}_{d}d_K{K}_2x2_scatter.png',
    '{dataset}_{d}d_K{K}_2x2_binned.png',
]

#-------------------------------------------------------------------------------


def collect_results() -> None:
    run_dir = latest_run_dir(PLOTS_BASE)
    if run_dir is None:
        print(f"No timestamped plot output found under {PLOTS_BASE}")
        return

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_pdf = OUTPUT_BASE / ts / 'scheme_corpus_comparison.pdf'

    print(f"Plot run: {run_dir.name}")
    print(f"Output: {out_pdf}\n")

    sections = []
    for dataset, d in DATASET_DIMS.items():
        dataset_children = []
        for K in K_VALUES:
            children = []
            for pattern in PLOT_FILES:
                path = run_dir / pattern.format(dataset=dataset, d=d, K=K)
                children.append(header(None, lambda p=path: [p] if p.exists() else []))
            dataset_children.append(divider(f'K={K}', children))
        sections.append(divider(dataset.upper(), dataset_children))

    with PdfReportBuilder(out_pdf) as report:
        report.cover_page('Full 2x2 Comparison: Scheme (A/B) x Corpus (Real/Uniform)',
                           subtitle=f'Plot run: {run_dir.name}')
        render_sections(report, sections, run_name=run_dir.name)

    size_mb = out_pdf.stat().st_size / 1_000_000
    print(f"\nDone: {out_pdf}  ({size_mb:.1f} MB)")

#-------------------------------------------------------------------------------

if __name__ == '__main__':
    collect_results()
