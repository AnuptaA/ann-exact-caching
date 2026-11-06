#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Collect the latest analysis outputs from all workloads into a timestamped
results/ folder and produce a combined PDF.
"""

import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent.parent.parent))

from simulations.pdf.report import PdfReportBuilder, latest_run_dir, divider, header, render_sections

#-------------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent.parent

WORKLOADS = [
    ('synthetic', 'Synthetic Simulation Analysis'),
    ('sift', 'SIFT Simulation Analysis'),
    ('esci', 'ESCI Simulation Analysis'),
]

# section labels per workload
# the script globs all PNGs in each subdir in sorted order
SECTIONS = {
    'synthetic': [
        ('global', 'Synthetic - Global / Cross-Set'),
        ('set1', 'Synthetic - Set 1: Baseline'),
        ('set2', 'Synthetic - Set 2: Perturbation Levels'),
        ('set3', 'Synthetic - Set 3: K/N Ratios'),
        ('set4', 'Synthetic - Set 4: Variable N'),
        ('set5', 'Synthetic - Set 5: Union Effectiveness'),
        ('set6', 'Synthetic - Set 6: Cache Size Variation'),
    ],
    'sift': [
        ('global', 'SIFT - Global'),
        ('cache_scaling', 'SIFT - Set 2: Cache Size Scaling'),
    ],
    'esci': [
        ('global', 'ESCI - Global'),
        ('set1', 'ESCI - Set 1: Baseline'),
        ('set2', 'ESCI - Set 2: Cache Size Scaling'),
        ('set3', 'ESCI - Set 3: K/N Ratios'),
    ],
}

#-------------------------------------------------------------------------------

def _copy_workload(src: Path, dst: Path):
    """Copy all subdir/PNG trees from src into dst."""
    for subdir in sorted(src.iterdir()):
        if not subdir.is_dir():
            continue
        pngs = sorted(subdir.glob('*.png'))
        if not pngs:
            continue
        out_sub = dst / subdir.name
        out_sub.mkdir(parents=True, exist_ok=True)
        for p in pngs:
            shutil.copy2(p, out_sub / p.name)
    print(f"  Copied {src.name} -> {dst}")

#-------------------------------------------------------------------------------

def _workload_section(workload: str, workload_title: str, src: Path) -> dict:
    known_sections = SECTIONS.get(workload, [])
    seen_subdirs = {s for s, _ in known_sections}

    # emit known sections in order, then any extra subdirs alphabetically
    all_subdirs = known_sections + [
        (d.name, d.name.replace('_', ' ').title())
        for d in sorted(src.iterdir())
        if d.is_dir() and d.name not in seen_subdirs
    ]

    return divider(workload_title, [
        header(section_title, sorted((src / subdir).glob('*.png')) if (src / subdir).exists() else [])
        for subdir, section_title in all_subdirs
    ])

#-------------------------------------------------------------------------------

def collect(results_base: Path, out_name: str):
    """Collect the latest processed outputs from all workloads and build a combined PDF."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = results_base / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCollecting results into: {out_dir}")

    processed = {}
    for workload, _ in WORKLOADS:
        p = latest_run_dir(ROOT / 'simulations' / workload / 'processed')
        processed[workload] = p
        if p is not None:
            print(f"  {workload}: {p.name}")
        else:
            print(f"  {workload}: SKIPPED (no processed dir)")

    # copy PNGs
    print("\nCopying plots...")
    for workload, _ in WORKLOADS:
        src = processed[workload]
        if src is None:
            continue
        _copy_workload(src, out_dir / workload)

    # build combined PDF
    out_pdf = out_dir / out_name
    print(f"\nBuilding combined PDF: {out_pdf}")

    with PdfReportBuilder(out_pdf) as report:
        report.cover_page('Combined Simulation Analysis')
        for workload, workload_title in WORKLOADS:
            src = processed[workload]
            if src is None:
                print(f"  Skipping {workload} (no processed dir)")
                continue
            section = _workload_section(workload, workload_title, src)
            render_sections(report, [section], run_name=src.name)

    print(f"\nDone: {out_pdf}  ({out_pdf.stat().st_size / 1_000_000:.1f} MB)")
    print(f"Results folder: {out_dir}")

#-------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect all workload results into a timestamped folder')
    parser.add_argument('--results-dir', default=None,
                        help='Parent results directory (default: <repo_root>/results)')
    parser.add_argument('--out', default='analysis_combined.pdf',
                        help='Output PDF filename inside the timestamped folder')
    args = parser.parse_args()

    results_base = Path(args.results_dir) if args.results_dir else ROOT / 'results'
    collect(results_base, args.out)
