#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Log-x histogram of IF among touching (query, candidate) pairs, setup
(b), SIFT K=100. Complements sift_touch_coverage_test.py's boxplot."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'common'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------

RESULTS_PATH = Path(__file__).parent.parent / 'results' / 'touch_coverage' / 'sift' / '128d' / 'K100.npz'
OUT_PATH = Path(__file__).parent.parent / 'plots' / 'touch_coverage' / 'sift' / 'sift_128d_K100_if_histogram.png'

#-------------------------------------------------------------------------------


def main() -> None:
    data = np.load(RESULTS_PATH)
    if_value = data['if_value']
    n = len(if_value)
    n_zero = int((if_value == 0).sum())
    positive = if_value[if_value > 0]

    print(f"Loaded {n} touching pairs from {RESULTS_PATH}")
    print(f"  exactly IF=0: {n_zero} ({n_zero / n:.4%})")
    print(f"  IF>0 range: [{positive.min():.3e}, {positive.max():.3e}]")

    log_bins = np.logspace(np.floor(np.log10(positive.min())), 0, 60)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(positive, bins=log_bins, color='tab:blue', alpha=0.8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('IF (log scale)')
    ax.set_ylabel('count of touching pairs (log scale)')
    ax.set_title(f'SIFT K=100: IF distribution among touching pairs\n'
                 f'n={n} ({n_zero} at exactly IF=0, not shown on log axis)')
    plt.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"\nWrote {OUT_PATH}")


if __name__ == '__main__':
    main()
