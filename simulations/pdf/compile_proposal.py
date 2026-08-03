#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Compile population proposal analysis outputs into a single PDF."""

import re
import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from simulations.pdf.report import PdfReportBuilder, latest_run_dir, divider, header, render_sections

#-------------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent.parent
SIM_DIR = ROOT / 'simulations'

BALL_GEOMETRY_DIR = SIM_DIR / 'debug' / 'population_geometry'
TRAJECTORY_DIR = SIM_DIR / 'debug' / 'population_trajectory'
DISTANCE_CONCENTRATION_DIR = SIM_DIR / 'debug' / 'population_distance_concentration'

DEBUG_FILES = [
    'margin_vs_k.png', 'rc_size_vs_k.png', 'rt_vs_k.png',
    'utility_ratio_vs_k.png', 'utility_ratio_density_vs_k.png',
]

#-------------------------------------------------------------------------------


def _nat_sort_key(p: Path) -> int:
    m = re.search(r'n(\d+)', p.stem)
    return int(m.group(1)) if m else 0


def _method_images(method_dir: Path) -> list:
    """Return PNGs in display order: hitrate, coverage, candidates,
    recall-vs-sigma, aggregate."""
    glob_patterns = [
        'memory_vs_hitrate*.png',
        'memory_vs_coverage*.png',
        'memory_vs_candidates_cached*.png',
        'recall_vs_sigma*.png',
    ]
    imgs = []
    for pattern in glob_patterns:
        imgs += sorted(method_dir.glob(pattern), key=_nat_sort_key)
    agg_dir = method_dir / 'aggregate'
    if agg_dir.exists():
        imgs += sorted(agg_dir.glob('*.png'))
    return imgs


def _debug_images(dataset: str) -> list:
    imgs = [BALL_GEOMETRY_DIR / dataset / f for f in DEBUG_FILES]
    imgs.append(TRAJECTORY_DIR / dataset / 'trajectory.png')
    imgs.append(TRAJECTORY_DIR / dataset / 'benefit_vs_iteration.png')
    imgs.append(TRAJECTORY_DIR / dataset / 'budget_pct_vs_metrics.png')
    imgs.append(TRAJECTORY_DIR / dataset / 'selected_radius_vs_k.png')
    imgs.append(TRAJECTORY_DIR / dataset / 'selected_n_density_vs_k.png')
    return [p for p in imgs if p.exists()]


def _dataset_section(processed_dir: Path, dataset_label: str) -> dict:
    dataset_key = dataset_label.lower()
    return divider(f'{dataset_label} Dataset', [
        header(
            None,
            lambda md=(processed_dir / 'ball'), ds=dataset_key: (
                _method_images(md) + _debug_images(ds)
            ),
            caption_fn=lambda img: (
                str(img.relative_to(processed_dir))
                if str(img).startswith(str(processed_dir)) else img.name
            ),
        )
    ])

def _cross_dataset_section() -> dict:
    return divider('Cross-Dataset Comparison', [
        header(
            None,
            lambda: [
                p for p in [DISTANCE_CONCENTRATION_DIR / 'distance_concentration.png']
                if p.exists()
            ],
            caption_fn=lambda img: img.name,
        )
    ])

#-------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description='Compile proposal analysis PNGs into a PDF.')
    parser.add_argument('--esci-dir', type=Path, default=None,
                        help='Processed ESCI run dir')
    parser.add_argument('--sift-dir', type=Path, default=None,
                        help='Processed SIFT run dir')
    parser.add_argument('--out', type=Path, default=None,
                        help='Output PDF path')
    args = parser.parse_args()

    esci_dir = args.esci_dir or latest_run_dir(
        SIM_DIR / 'esci' / 'processed' / 'population_proposal', pattern=r'.*'
    )
    sift_dir = args.sift_dir or latest_run_dir(
        SIM_DIR / 'sift' / 'processed' / 'population_proposal', pattern=r'.*'
    )

    if esci_dir is None or sift_dir is None:
        raise SystemExit('Could not find processed dirs. Pass --esci-dir and --sift-dir.')

    run_ts = esci_dir.name
    out_pdf = args.out or SIM_DIR / 'population_proposal_sigma' / f'population_proposal_{run_ts}.pdf'

    print(f'ESCI dir : {esci_dir}')
    print(f'SIFT dir : {sift_dir}')
    print(f'Output   : {out_pdf}\n')

    with PdfReportBuilder(out_pdf) as report:
        report.cover_page('Population Proposal Analysis')
        for processed_dir, label in [(esci_dir, 'ESCI'), (sift_dir, 'SIFT')]:
            print(f'[{label}]')
            render_sections(report, [_dataset_section(processed_dir, label)], processed_dir.name)
        print('[Cross-Dataset Comparison]')
        render_sections(report, [_cross_dataset_section()], run_ts)

    size_mb = out_pdf.stat().st_size / 1_000_000
    print(f'\nDone: {out_pdf}  ({size_mb:.1f} MB)')


if __name__ == '__main__':
    main()
