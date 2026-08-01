#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Compile all analysis PNGs from a processed run into a single PDF, for a
single-dataset workload (sift, esci, synthetic). Flat section list: one
header page per subdir, no nesting."""

import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from simulations.pdf.report import PdfReportBuilder, latest_run_dir, images_in, header, render_sections

#-------------------------------------------------------------------------------

DATASET_CONFIG = {
    'esci': dict(
        title='ESCI Simulation Analysis',
        processed_base=Path('simulations/esci/processed'),
        sections=[
            ('global', 'Global', [
                'hit_rate_heatmap_global.png',
                'lemma_breakdown_global.png',
                'avg_time_by_algorithm.png',
            ]),
            ('set1', 'Set 1 -- Baseline (K=99, N=20)', [
                'baseline_hit_rate.png',
                'baseline_distance_calcs.png',
            ]),
            ('set2', 'Set 2 -- Cache Size Scaling (K=99, N=20)', [
                'hit_rate_vs_cache_size.png',
                'distance_calcs_vs_cache_size.png',
                'set2_coverage_and_hit_rate.png',
                'set2_coverage_estimated_vs_actual.png',
                'hit_rate_heatmap_angular.png',
            ]),
            ('set3', 'Set 3 -- K/N Ratios', [
                'kn_heatmap_combined_angular.png',
                'kn_ratio_matrix_angular.png',
            ]),
        ],
    ),
    'sift': dict(
        title='SIFT Simulation Analysis',
        processed_base=Path('simulations/sift/processed'),
        sections=[
            ('global', 'Global', [
                'hit_rate_heatmap_global.png',
                'lemma_breakdown_global.png',
                'avg_time_by_algorithm.png',
            ]),
            ('cache_scaling', 'Set 2 -- Cache Size Scaling (K=100, N=20)', [
                'hit_rate_vs_cache_size.png',
                'distance_calcs_vs_cache_size.png',
                'set2_coverage_and_hit_rate.png',
                'set2_coverage_estimated_vs_actual.png',
                'hit_rate_heatmap_angular.png',
            ]),
        ],
    ),
    'synthetic': dict(
        title='Synthetic Simulation Analysis',
        processed_base=Path('simulations/synthetic/processed'),
        sections=[
            ('global', 'Global / Cross-Set', [
                'accuracy_heatmap_global.png',
                'hit_rate_heatmap_global.png',
                'hit_rate_heatmap_per_set.png',
                'lemma_breakdown_global.png',
                'time_relative_to_brute.png',
            ]),
            ('set1', 'Set 1 -- Baseline (K=100, N=20)', None),
            ('set2', 'Set 2 -- Perturbation Levels (K=100, N=20)', [
                'set2_hit_rate_heatmap_all.png',
                'set2_lemma_breakdown.png',
                'set2_angle_hist_consolidated.png',
                'set2_angle_hist_zoomed.png',
            ]),
            ('set3', 'Set 3 -- K/N Ratios', [
                'set3_kn_heatmap_combined_euclidean.png',
                'set3_kn_heatmap_combined_angular.png',
                'set3_kn_ratio_matrix_euclidean.png',
                'set3_kn_ratio_matrix_angular.png',
            ]),
            ('set4', 'Set 4 -- Variable N (K=100)', [
                'set4_hit_rate_by_var_level.png',
                'set4_dist_calcs_by_var_level.png',
                'set4_hit_rate_vs_n_value.png',
            ]),
            ('set5', 'Set 5 -- Union Effectiveness (K=100, N=20)', None),
            ('set6', 'Set 6 -- Cache Size Variation (K=100, N=20)', None),
        ],
    ),
}

#-------------------------------------------------------------------------------

def compile(dataset: str, processed_dir: Path, out_path: Path) -> None:
    """Compile all PNGs in processed_dir into a PDF at out_path."""
    cfg = DATASET_CONFIG[dataset]
    run_name = processed_dir.name

    print(f"Compiling PDF from: {processed_dir}")
    print(f"Output: {out_path}")

    sections = [
        header(
            section_title,
            lambda sd=(processed_dir / subdir), of=ordered_files: images_in(sd, of),
        )
        for subdir, section_title, ordered_files in cfg['sections']
    ]

    with PdfReportBuilder(out_path) as report:
        report.cover_page(cfg['title'], subtitle=f'Run: {run_name}')
        render_sections(report, sections, run_name)

    print(f"\nDone: {out_path}  ({out_path.stat().st_size / 1_000_000:.1f} MB)")

#-------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Compile dataset analysis PNGs into a PDF')
    parser.add_argument('--dataset', choices=list(DATASET_CONFIG), required=True)
    parser.add_argument('--processed-dir', default=None,
                        help='Path to processed run dir (default: latest timestamped)')
    parser.add_argument('--out', default=None,
                        help='Output PDF path (default: <processed_dir>/analysis.pdf)')
    args = parser.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    if args.processed_dir:
        processed_dir = Path(args.processed_dir)
    else:
        processed_dir = latest_run_dir(cfg['processed_base'])
        if processed_dir is None:
            raise FileNotFoundError(
                f"No timestamped run directories found in {cfg['processed_base']}"
            )
        print(f"Auto-selected run: {processed_dir}")

    out_path = Path(args.out) if args.out else processed_dir / 'analysis.pdf'

    compile(args.dataset, processed_dir, out_path)

#-------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
