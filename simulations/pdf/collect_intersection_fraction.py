#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Compile the intersection-fraction calibration sweep's output PNGs into a
single PDF, sectioned by dataset.
"""

import re
import sys
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent.parent.parent))

from simulations.pdf.report import PdfReportBuilder, latest_run_dir, divider, header, render_sections

#-------------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent.parent
SWEEP_BASE = ROOT / 'simulations' / 'debug' / 'population_intersection_calibration'
OUTPUT_BASE = ROOT / 'simulations' / 'results' / 'intersection_fraction'

DATASETS = [
    ('sift', 'SIFT'),
    ('esci', 'ESCI'),
]

#-------------------------------------------------------------------------------


def _budget_images(dataset_dir: Path, prefix: str) -> list:
    """{prefix}_budget{N}pct.png files, sorted numerically by budget (not
    lexicographically -- "budget10pct" < "budget100pct" < "budget15pct"
    alphabetically, which is the wrong order)."""
    def budget_num(p: Path) -> float:
        m = re.search(r'budget(\d+(?:\.\d+)?)pct', p.name)
        return float(m.group(1)) if m else 0.0
    return sorted(dataset_dir.glob(f'{prefix}_budget*pct.png'), key=budget_num)


def _existing(paths: list) -> list:
    return [p for p in paths if p.exists()]


def _dataset_section(dataset: str, dataset_title: str, run_dir: Path) -> dict:
    dataset_dir = run_dir / dataset
    return divider(dataset_title, [
        header('Dual Reliability (Angular vs. Solid-Ball, Pooled Budgets)',
               lambda: _existing([dataset_dir / 'dual_calibration_vs_intersection_fraction.png'])),
        header('Dual Reliability by Budget',
               lambda: _budget_images(dataset_dir, 'dual_calibration_vs_intersection_fraction'),
               caption_fn=lambda img: img.name),
        header('Recall Reliability (Continuous Top-K Recall, K>1, Pooled Budgets)',
               lambda: _existing([dataset_dir / 'recall_vs_intersection_fraction.png'])),
        header('Recall Reliability by Budget',
               lambda: _budget_images(dataset_dir, 'recall_vs_intersection_fraction'),
               caption_fn=lambda img: img.name),
        header('Best-Candidate Scatter by Budget',
               lambda: _budget_images(dataset_dir, 'dual_best_candidate_scatter'),
               caption_fn=lambda img: img.name),
        header('Confusion Matrix (Pooled Budgets)',
               lambda: _existing([dataset_dir / 'confusion_matrix_pooled.png'])),
        header('Confusion Matrix by Budget',
               lambda: _budget_images(dataset_dir, 'confusion_matrix'),
               caption_fn=lambda img: img.name),
    ])


def collect_intersection_fraction() -> None:
    run_dir = latest_run_dir(SWEEP_BASE)
    if run_dir is None:
        print(f"No timestamped sweep output found under {SWEEP_BASE}")
        return

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_pdf = OUTPUT_BASE / ts / 'intersection_fraction_results.pdf'

    print(f"Sweep run: {run_dir.name}")
    print(f"Output: {out_pdf}\n")

    with PdfReportBuilder(out_pdf) as report:
        report.cover_page('Intersection-Fraction Calibration Results',
                           subtitle=f'Sweep run: {run_dir.name}')

        for dataset, dataset_title in DATASETS:
            print(f"[{dataset.upper()}]")
            section = _dataset_section(dataset, dataset_title, run_dir)
            render_sections(report, [section], run_name=run_dir.name)

    size_mb = out_pdf.stat().st_size / 1_000_000
    print(f"\nDone: {out_pdf}  ({size_mb:.1f} MB)")

#-------------------------------------------------------------------------------

if __name__ == '__main__':
    collect_intersection_fraction()
