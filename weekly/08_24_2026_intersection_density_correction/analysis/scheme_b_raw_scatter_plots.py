#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Raw predicted-vs-actual scatter for per_cache_query_recall_hat and
per_test_query_recall_hat, every sampled (query, candidate) pair - no
argmax selection - complementing the winner-only Scheme B plots. Data
from sift_scheme_b_raw_scatter_run_experiment.py."""

import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------

K_VALUES = [25, 50, 100]
CORPORA = ('real', 'uniform')
CORPUS_LABELS = {'real': 'real corpus', 'uniform': 'uniform corpus'}
PREDICTORS = [('pred_pcq', 'per_cache_query_recall_hat', 'tab:green'), ('pred_ptq', 'per_test_query_recall_hat', 'tab:blue')]

OUT_DIR = Path(__file__).parent.parent / 'plots' / 'scheme_b_raw_scatter'

#-------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=str, required=True)
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for corpus in CORPORA:
        fig, axes = plt.subplots(2, len(K_VALUES), figsize=(3.2 * len(K_VALUES), 6.4), squeeze=False)
        for col, K in enumerate(K_VALUES):
            d = np.load(data_dir / f'{corpus}_K{K}.npz')
            actual = d['actual']
            for row, (pred_f, label, color) in enumerate(PREDICTORS):
                pred = np.clip(d[pred_f], 0.0, 1.0)
                finite = np.isfinite(pred)
                ax = axes[row][col]
                ax.scatter(pred[finite], actual[finite], s=1, alpha=0.03, color=color, linewidths=0)
                ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
                ax.set_xlim(-0.02, 1.02)
                ax.set_ylim(-0.05, 1.05)
                ax.set_title(f'{label}, K={K}', fontsize=8)
                if row == 1:
                    ax.set_xlabel('predicted recall')
                if col == 0:
                    ax.set_ylabel('actual recall')
                ax.grid(alpha=0.3)
        fig.suptitle(f'Scheme B raw (query, candidate) pairs - no argmax, SIFT (128D), {CORPUS_LABELS[corpus]} '
                     f'(n={len(actual):,} pairs/panel)')
        plt.tight_layout()
        out_path = OUT_DIR / f'raw_scatter_{corpus}.png'
        fig.savefig(out_path, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f"wrote {out_path}")


if __name__ == '__main__':
    main()
