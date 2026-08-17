#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from simulations.pdf.report import PdfReportBuilder
from debug_esci_run_experiment import merge_checkpoints
from debug_report_extra_plots import (
    plot_sift_power_check_overlay,
    plot_sift_vs_esci_overlay,
    plot_sift_whitening_overlay,
    plot_unbinned_pairwise_scatter,
    plot_gap_vs_density,
    mc_formula_check_summary_page,
    summary_table_page,
    _load_pairwise,
)

#-------------------------------------------------------------------------------

BASE = Path(__file__).parent
OUT_BASE = BASE / 'pdf'

SIFT_VALIDATION_DIR = BASE / 'plots/20260813_103643/debug_sift'
ESCI_VALIDATION_DIR = BASE / 'plots/20260813_132413/debug_esci'
SYNTHETIC_CONTROL_DIR = BASE / 'plots/20260813_151128/debug_synthetic_control'
SIFT_POWER_CHECK_DIR = BASE / 'plots/20260814_101913/debug_sift'
PAIRWISE_VS_SELECTION_DIR = BASE / 'plots/20260813_224448/debug_pairwise_vs_selection'

SIFT_VALIDATION_RUN = BASE / 'results/20260813_103643'
ESCI_VALIDATION_RUN = BASE / 'results/20260813_111554'
SIFT_POWER_CHECK_RUN = BASE / 'results_sift_power_check/20260813_151256'
SIFT_PAIRWISE_RUN = BASE / 'results_sift_pairwise_recall/20260813_170113'
ESCI_PAIRWISE_RUN = BASE / 'results_esci_pairwise_recall/run1'
ESCI_PAIRWISE_SEEDS = 3
SIFT_WHITENED_RUN = BASE / 'results_sift_whitened/20260814_120345'

K_VALUES = [10, 25, 50, 100]

#-------------------------------------------------------------------------------


def main() -> None:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = BASE / 'plots' / ts / 'debug_full_report'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_BASE / ts / 'full_diagnostic_report.pdf'

    power_check_overlay = plot_sift_power_check_overlay(SIFT_VALIDATION_RUN, SIFT_POWER_CHECK_RUN, out_dir)
    cross_dataset_overlay = plot_sift_vs_esci_overlay(SIFT_VALIDATION_RUN, ESCI_VALIDATION_RUN, out_dir)
    whitening_overlay = plot_sift_whitening_overlay(SIFT_VALIDATION_RUN, SIFT_WHITENED_RUN, out_dir)

    sift_pairwise = {K: _load_pairwise(SIFT_PAIRWISE_RUN, 'debug_sift', 128, K) for K in K_VALUES}
    esci_pooled = merge_checkpoints(ESCI_PAIRWISE_RUN, n_seeds=ESCI_PAIRWISE_SEEDS)
    esci_pairwise = {K: {'fraction': esci_pooled[K]['pairwise']['fraction'],
                          'recall': esci_pooled[K]['pairwise']['recall'],
                          'r_test': esci_pooled[K]['pairwise']['r_test']}
                      for K in K_VALUES}
    unbinned_scatter = plot_unbinned_pairwise_scatter(
        [('SIFT', 'tab:green', sift_pairwise), ('ESCI (partial, 3/5 seeds)', 'tab:purple', esci_pairwise)], out_dir)
    gap_vs_density = plot_gap_vs_density(
        [('SIFT', 'tab:green', sift_pairwise), ('ESCI (partial, 3/5 seeds)', 'tab:purple', esci_pairwise)], out_dir)

    mc_check_page = mc_formula_check_summary_page(out_dir / 'mc_formula_check.png')

    summary_table = summary_table_page(
        [
            ('SIFT', SIFT_VALIDATION_RUN, 'debug_sift', 128),
            ('ESCI', ESCI_VALIDATION_RUN, 'debug_esci', 384),
        ],
        out_dir / 'calibration_gap_summary.png',
    )

    with PdfReportBuilder(out_pdf) as report:
        report.cover_page(
            'SIFT/ESCI Cap-Intersection Calibration Diagnostics',
            subtitle=['Validation, hypothesis tests, and robustness checks'],
        )

        report.divider_page('1. SIFT/ESCI Validation', 'debug_sift, debug_esci (5 seeds, 1000/1000)')
        report.header_page('SIFT (d=128): Binned Calibration', SIFT_VALIDATION_DIR.name)
        report.image_page(SIFT_VALIDATION_DIR / '128d_binned_calibration.png',
                           caption='Predicted fraction vs. actual recall, binned.')
        report.image_page(SIFT_VALIDATION_DIR / '128d_calibration_scatter.png',
                           caption='Unbinned scatter, same data.')
        report.header_page('ESCI (d=384): Binned Calibration', ESCI_VALIDATION_DIR.name)
        report.image_page(ESCI_VALIDATION_DIR / '384d_binned_calibration.png',
                           caption='Predicted fraction vs. actual recall, binned.')
        report.image_page(ESCI_VALIDATION_DIR / '384d_calibration_scatter.png',
                           caption='Unbinned scatter, same data.')

        report.divider_page('2. Cross-Dataset Comparison', 'SIFT vs. ESCI, same K values')
        report.header_page('SIFT vs. ESCI Overlay', 'derived from both validation runs')
        report.image_page(cross_dataset_overlay, caption='Both datasets on the same axes.')

        report.divider_page('3. Synthetic Control', 'part1 at matched candidate/test-query scale')
        report.header_page('Synthetic Uniform-Sphere Control', SYNTHETIC_CONTROL_DIR.name)
        report.image_page(SYNTHETIC_CONTROL_DIR / '128d_sift_vs_synthetic.png',
                           caption='Synthetic vs. SIFT, matched scale, d=128.')
        report.image_page(SYNTHETIC_CONTROL_DIR / '384d_esci_vs_synthetic.png',
                           caption='Synthetic vs. ESCI, matched scale, d=384.')

        report.divider_page('4. Statistical Power Check', 'debug_sift, 1000 vs. 3000 test queries')
        report.header_page('SIFT: 1000 vs. 3000 Test Queries', 'side by side')
        report.image_page(SIFT_POWER_CHECK_DIR / '128d_binned_calibration.png',
                           caption='3000 test queries.')
        report.header_page('SIFT: Power-Check Overlay', 'both scales, same axes')
        report.image_page(power_check_overlay, caption='1000 vs. 3000 test queries, overlaid.')

        report.divider_page('5. Pairwise vs. Selection', 'debug_pairwise_vs_selection (SIFT; ESCI partial)')
        report.header_page('SIFT: Argmax-Selection vs. Unconditioned Pairwise', PAIRWISE_VS_SELECTION_DIR.name)
        report.image_page(PAIRWISE_VS_SELECTION_DIR / '128d_selection_vs_pairwise.png',
                           caption='Selection vs. pairwise calibration, overlaid.')
        report.header_page('Unbinned Pairwise Calibration', 'hexbin density, SIFT + ESCI (partial)')
        report.image_page(unbinned_scatter, caption='Pairwise fraction vs. recall, unbinned.')

        report.divider_page('6. Non-Uniformity Hypothesis', 'gap-vs-density, PCA-whitening intervention')
        report.header_page('Calibration Gap vs. Local Density', 'quantile-binned by r_test')
        report.image_page(gap_vs_density, caption='Mean gap by r_test quantile.')
        report.header_page('PCA-Whitening Intervention', SIFT_WHITENED_RUN.name)
        report.image_page(whitening_overlay, caption='Original vs. ZCA-whitened calibration.')

        report.divider_page('7. Formula Correctness Check', 'Monte Carlo vs. closed-form, real SIFT triples')
        report.header_page('Monte Carlo Cross-Check at d=128', 'real (r_cache, r_test, delta) triples')
        report.image_page(mc_check_page, caption='Closed-form vs. Monte Carlo, per triple.')

        report.divider_page('8. Summary', 'calibration gap by dataset and K')
        report.header_page('Calibration Gap Summary Table', 'lowest populated fraction bin')
        report.image_page(summary_table, caption='Fraction, recall, and gap by dataset and K.')

    size_mb = out_pdf.stat().st_size / 1_000_000
    print(f"Done: {out_pdf} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
