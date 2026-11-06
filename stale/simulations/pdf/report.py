#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Shared PDF-report building blocks used by every compile/collect script in
this repo.
"""

import re
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

#-------------------------------------------------------------------------------

PAGE_SIZE = (11, 8.5)  # letter landscape, inches


class PdfReportBuilder:
    """Wraps a PdfPages, exposing the page kinds shared across all reports.

    Usage:
        with PdfReportBuilder(out_path) as report:
            report.cover_page('Title', subtitle='Run: ...')
            report.divider_page('Section', run_name='...')
            report.header_page('Subsection', run_name='...')
            report.image_page(png_path)
    """

    def __init__(self, out_path: Path):
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._pdf = PdfPages(str(self.out_path))

    def __enter__(self) -> "PdfReportBuilder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._pdf.close()

    def cover_page(self, title: str, subtitle: str = None, date: bool = True,
                    facecolor: str = '#0d1b2a') -> None:
        """subtitle may be a single string or a list of lines, each on its own row."""
        fig = plt.figure(figsize=PAGE_SIZE)
        fig.patch.set_facecolor(facecolor)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor(facecolor)
        ax.axis('off')
        ax.text(0.5, 0.62, title, transform=ax.transAxes,
                ha='center', va='center', fontsize=32, color='white', fontweight='bold')
        y = 0.48
        subtitle_lines = [subtitle] if isinstance(subtitle, str) else (subtitle or [])
        for line in subtitle_lines:
            ax.text(0.5, y, line, transform=ax.transAxes,
                    ha='center', va='center', fontsize=16, color='#b0c4de')
            y -= 0.10
        if date:
            ax.text(0.5, y, datetime.now().strftime('%B %d, %Y'), transform=ax.transAxes,
                    ha='center', va='center', fontsize=14, color='#b0c4de')
        self._pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def divider_page(self, title: str, run_name: str) -> None:
        fig = plt.figure(figsize=PAGE_SIZE)
        fig.patch.set_facecolor('#0f3460')
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor('#0f3460')
        ax.axis('off')
        ax.text(0.5, 0.58, title, transform=ax.transAxes,
                ha='center', va='center', fontsize=34, color='white', fontweight='bold')
        ax.text(0.5, 0.44, f'Run: {run_name}', transform=ax.transAxes,
                ha='center', va='center', fontsize=14, color='#b0c4de')
        self._pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def header_page(self, title: str, run_name: str) -> None:
        fig = plt.figure(figsize=PAGE_SIZE)
        fig.patch.set_facecolor('#1a1a2e')
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor('#1a1a2e')
        ax.axis('off')
        ax.text(0.5, 0.6, title, transform=ax.transAxes,
                ha='center', va='center', fontsize=28, color='white', fontweight='bold',
                wrap=True)
        ax.text(0.5, 0.38, f'Run: {run_name}', transform=ax.transAxes,
                ha='center', va='center', fontsize=13, color='#aaaacc')
        self._pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def image_page(self, img_path: Path, caption: str = None) -> None:
        img_path = Path(img_path)
        img = mpimg.imread(str(img_path))
        h, w = img.shape[:2]
        page_w, page_h = PAGE_SIZE
        scale = min(page_w / (w / 100), page_h / (h / 100))
        fig, ax = plt.subplots(figsize=((w / 100) * scale, (h / 100) * scale))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(caption if caption is not None else img_path.name,
                      fontsize=7, color='#555555', pad=3)
        plt.subplots_adjust(left=0, right=1, top=0.97, bottom=0)
        self._pdf.savefig(fig, bbox_inches='tight', dpi=150)
        plt.close(fig)

#-------------------------------------------------------------------------------


def latest_run_dir(base: Path, pattern: str = r'\d{8}_\d{6}') -> Path:
    """Most recent subdirectory of base whose name matches pattern, or None."""
    base = Path(base)
    if not base.exists():
        return None
    stamped = sorted(
        d for d in base.iterdir()
        if d.is_dir() and re.fullmatch(pattern, d.name)
    )
    return stamped[-1] if stamped else None


def images_in(section_dir: Path, ordered_names: list = None) -> list:
    """PNGs in section_dir: explicit ordered_names first (if present and existing),
    followed by any remaining *.png files in sorted order."""
    section_dir = Path(section_dir)
    if not section_dir.exists():
        return []
    if not ordered_names:
        return sorted(section_dir.glob('*.png'))
    files = [section_dir / name for name in ordered_names if (section_dir / name).exists()]
    extra = sorted(p for p in section_dir.glob('*.png') if p.name not in ordered_names)
    return files + extra

#-------------------------------------------------------------------------------

def divider(title: str, children: list) -> dict:
    return {'kind': 'divider', 'title': title, 'children': children}


def header(title: str, images, caption_fn=None, also_divider: bool = False, run_name: str = None) -> dict:
    """run_name: per-node override of the run name shown on the header (and
    also_divider) page; falls back to the run_name passed to render_sections."""
    return {
        'kind': 'header', 'title': title, 'images': images,
        'caption_fn': caption_fn, 'also_divider': also_divider, 'run_name': run_name,
    }

def render_sections(report: PdfReportBuilder, sections: list, run_name: str, indent: str = '  ') -> None:
    """Recursively render a section tree built from divider()/header() nodes."""
    for node in sections:
        if node['kind'] == 'divider':
            print(f"{indent}[{node['title']}]")
            report.divider_page(node['title'], run_name)
            render_sections(report, node['children'], run_name, indent + '  ')
        else:
            images = node['images']
            imgs = images() if callable(images) else images
            label = node['title'] or '(untitled)'
            if not imgs:
                print(f"{indent}{label}: skipped (no PNGs)")
                continue
            print(f"{indent}[{label}] {len(imgs)} image(s)")
            node_run_name = node['run_name'] if node['run_name'] is not None else run_name
            if node['also_divider']:
                report.divider_page(node['title'], node_run_name)
            # A falsy title (None/'') skips the header page entirely
            if node['title']:
                report.header_page(node['title'], node_run_name)
            caption_fn = node['caption_fn']
            for img in imgs:
                report.image_page(img, caption=caption_fn(img) if caption_fn else None)

#-------------------------------------------------------------------------------
