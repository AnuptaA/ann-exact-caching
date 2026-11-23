#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Collect the latest analysis outputs from all workloads into a timestamped
results/ folder and produce a combined PDF.
"""

import re
import shutil
import argparse
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

#-------------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent

WORKLOADS = [
    ('synthetic', 'Synthetic Simulation Analysis'),
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
    'esci': [
        ('global', 'ESCI - Global'),
        ('set1', 'ESCI - Set 1: Baseline'),
        ('set2', 'ESCI - Set 2: Cache Size Scaling'),
        ('set3', 'ESCI - Set 3: K/N Ratios'),
    ],
}

#-------------------------------------------------------------------------------

def _latest_processed_dir(workload: str) -> Path:
    """Return the most recent timestamped processed directory for a workload."""
    base = ROOT / 'simulations' / workload / 'processed'
    if not base.exists():
        raise FileNotFoundError(f"Processed dir not found: {base}")
    stamped = sorted(
        d for d in base.iterdir()
        if d.is_dir() and re.fullmatch(r'\d{8}_\d{6}', d.name)
    )
    if not stamped:
        raise FileNotFoundError(f"No timestamped run directories found in {base}")
    return stamped[-1]


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


def add_divider_page(pdf: PdfPages, workload_title: str, run_name: str):
    """Render a workload divider page with dark background."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('#0f3460')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('#0f3460')
    ax.axis('off')
    ax.text(0.5, 0.58, workload_title, transform=ax.transAxes,
            ha='center', va='center', fontsize=34, color='white', fontweight='bold')
    ax.text(0.5, 0.44, f'Run: {run_name}', transform=ax.transAxes,
            ha='center', va='center', fontsize=14, color='#b0c4de')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def add_header_page(pdf: PdfPages, title: str, run_name: str):
    """Render a section header page."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('#1a1a2e')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('#1a1a2e')
    ax.axis('off')
    ax.text(0.5, 0.6, title, transform=ax.transAxes,
            ha='center', va='center', fontsize=28, color='white', fontweight='bold',
            wrap=True)
    ax.text(0.5, 0.38, f'Run: {run_name}', transform=ax.transAxes,
            ha='center', va='center', fontsize=13, color='#aaaacc')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def add_image_page(pdf: PdfPages, img_path: Path):
    """Render a single image into the PDF."""
    img = mpimg.imread(str(img_path))
    h, w = img.shape[:2]
    page_w, page_h = 11.0, 8.5
    scale = min(page_w / (w / 100), page_h / (h / 100))
    fig_w = (w / 100) * scale
    fig_h = (h / 100) * scale

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(img_path.name, fontsize=7, color='#555555', pad=3)
    plt.subplots_adjust(left=0, right=1, top=0.97, bottom=0)
    pdf.savefig(fig, bbox_inches='tight', dpi=150)
    plt.close(fig)


#-------------------------------------------------------------------------------

def collect(results_base: Path, out_name: str):
    """Collect the latest processed outputs from all workloads and build a combined PDF."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = results_base / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCollecting results into: {out_dir}")

    processed = {}
    for workload, _ in WORKLOADS:
        try:
            p = _latest_processed_dir(workload)
            processed[workload] = p
            print(f"  {workload}: {p.name}")
        except FileNotFoundError as e:
            print(f"  {workload}: SKIPPED ({e})")
            processed[workload] = None

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

    with PdfPages(str(out_pdf)) as pdf:
        # cover page
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor('#0d1b2a')
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor('#0d1b2a')
        ax.axis('off')
        ax.text(0.5, 0.62, 'Combined Simulation Analysis', transform=ax.transAxes,
                ha='center', va='center', fontsize=32, color='white', fontweight='bold')
        ax.text(0.5, 0.48, datetime.now().strftime('%B %d, %Y'), transform=ax.transAxes,
                ha='center', va='center', fontsize=16, color='#b0c4de')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        for workload, workload_title in WORKLOADS:
            src = processed[workload]
            if src is None:
                print(f"  Skipping {workload} (no processed dir)")
                continue

            run_name = src.name
            add_divider_page(pdf, workload_title, run_name)

            sections = SECTIONS.get(workload, [])
            seen_subdirs = {s for s, _ in sections}

            # emit known sections in order, then any extra subdirs alphabetically
            all_subdirs = sections + [
                (d.name, d.name.replace('_', ' ').title())
                for d in sorted(src.iterdir())
                if d.is_dir() and d.name not in seen_subdirs
            ]

            for subdir, section_title in all_subdirs:
                section_dir = src / subdir
                if not section_dir.exists():
                    continue
                files = sorted(section_dir.glob('*.png'))
                if not files:
                    continue
                print(f"    [{workload}/{subdir}] {len(files)} image(s)")
                add_header_page(pdf, section_title, run_name)
                for img_path in files:
                    add_image_page(pdf, img_path)

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
