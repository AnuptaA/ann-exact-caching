#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Compile the overlap-threshold hit-policy sweep's output PNGs into a
single PDF.
"""

import re
import sys
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent.parent.parent))

from simulations.pdf.report import PdfReportBuilder, latest_run_dir, divider, header, render_sections

#-------------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent.parent
SWEEP_BASE = ROOT / 'simulations' / 'hit_policy' / 'reports' / 'overlap_threshold'
OUTPUT_BASE = ROOT / 'simulations' / 'results' / 'overlap_threshold'

DATASETS = [
    ('sift', 'SIFT'),
    ('esci', 'ESCI'),
]

#-------------------------------------------------------------------------------


def _numbered_images(dataset_dir: Path, prefix: str, pattern: str) -> list:
    """{prefix}{pattern}.png files, sorted numerically by the captured value."""
    def num(p: Path) -> float:
        m = re.search(pattern, p.name)
        return float(m.group(1)) if m else 0.0
    return sorted(dataset_dir.glob(f'{prefix}*.png'), key=num)


def _dataset_section(dataset: str, dataset_title: str, run_dir: Path) -> dict:
    dataset_dir = run_dir / dataset
    return divider(dataset_title, [
        header('Hit Rate vs. Budget Fraction (benchmark x K grid, one per threshold)',
               lambda: _numbered_images(dataset_dir, 'hit_rate_vs_budget_threshold', r'threshold(\d+\.\d+)'),
               caption_fn=lambda img: img.name),
        header('Mean Recall vs. Budget Fraction (benchmark x K grid, one per threshold)',
               lambda: _numbered_images(dataset_dir, 'recall_vs_budget_threshold', r'threshold(\d+\.\d+)'),
               caption_fn=lambda img: img.name),
        header('Recall vs. Hit-Rate Tradeoff (benchmark x K grid, one per budget)',
               lambda: _numbered_images(dataset_dir, 'recall_vs_hit_rate_tradeoff_budget', r'budget(\d+)pct'),
               caption_fn=lambda img: img.name),
    ])


def collect_overlap_threshold() -> None:
    run_dir = latest_run_dir(SWEEP_BASE)
    if run_dir is None:
        print(f"No timestamped sweep output found under {SWEEP_BASE}")
        return

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_pdf = OUTPUT_BASE / ts / 'overlap_threshold_results.pdf'

    print(f"Sweep run: {run_dir.name}")
    print(f"Output: {out_pdf}\n")

    with PdfReportBuilder(out_pdf) as report:
        report.cover_page('Overlap-Threshold Hit Policy Results',
                           subtitle=f'Sweep run: {run_dir.name}')
        for dataset, dataset_title in DATASETS:
            print(f"[{dataset.upper()}]")
            section = _dataset_section(dataset, dataset_title, run_dir)
            render_sections(report, [section], run_name=run_dir.name)

    size_mb = out_pdf.stat().st_size / 1_000_000
    print(f"\nDone: {out_pdf}  ({size_mb:.1f} MB)")

#-------------------------------------------------------------------------------

if __name__ == '__main__':
    collect_overlap_threshold()
