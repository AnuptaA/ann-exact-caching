#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Compile all population benchmark outputs into a single timestamped PDF.
"""

import sys
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent.parent.parent))

from simulations.pdf.report import PdfReportBuilder, latest_run_dir, divider, header, render_sections

#-------------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent.parent
OUTPUT_BASE = ROOT / 'simulations' / 'results' / 'population'

DATASETS = [
    ('sift', 'SIFT Population Benchmarks'),
    ('esci', 'ESCI Population Benchmarks'),
]

MICROBENCHMARK_SECTIONS = [
    ('containment-microbenchmarks', 'Containment Microbenchmark'),
]

#-------------------------------------------------------------------------------


def _dataset_section(dataset: str, dataset_title: str) -> dict:
    pop_base = ROOT / 'simulations' / dataset / 'processed' / 'population'
    children = []
    for section_dir, section_title in MICROBENCHMARK_SECTIONS:
        latest = latest_run_dir(pop_base / section_dir)
        if latest is None:
            print(f"  {section_dir}: skipped (no processed dir)")
            continue
        children.append(header(
            section_title,
            sorted(latest.glob('*.png')),
            caption_fn=lambda img: str(img.relative_to(ROOT)),
            run_name=latest.name,
        ))
    return divider(dataset_title, children)


def collect_population() -> None:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_pdf = OUTPUT_BASE / ts / 'population_results.pdf'

    print(f"Output: {out_pdf}\n")

    with PdfReportBuilder(out_pdf) as report:
        report.cover_page('Population Benchmark Results')

        for dataset, dataset_title in DATASETS:
            print(f"[{dataset.upper()}]")
            section = _dataset_section(dataset, dataset_title)
            render_sections(report, [section], run_name=ts)

    size_mb = out_pdf.stat().st_size / 1_000_000
    print(f"\nDone: {out_pdf}  ({size_mb:.1f} MB)")

#-------------------------------------------------------------------------------

if __name__ == '__main__':
    collect_population()
