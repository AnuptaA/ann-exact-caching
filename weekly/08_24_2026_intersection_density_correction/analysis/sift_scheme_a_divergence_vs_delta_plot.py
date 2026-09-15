#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Divergence (true_recall - IF) vs. delta, fine resolution, SIFT real
anchors, K in {5,25,50,100}."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'common'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent.parent / 'results' / 'density_correction'
OUT_PATH = Path(__file__).parent.parent / 'plots' / 'density_correction' / 'sift_scheme_a_divergence_vs_delta_fine.png'
K_VALUES = [5, 25, 50, 100]

#-------------------------------------------------------------------------------


def main() -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for K in K_VALUES:
        path = RESULTS_DIR / f'sift_scheme_a_real_K{K}.npz'
        data = np.load(path)
        delta, if_value, true_recall = data['delta'], data['if_value'], data['true_recall']
        divergence = true_recall - if_value

        order = np.argsort(delta)
        delta_sorted, div_sorted = delta[order], divergence[order]
        n_bins = 60
        edges = np.linspace(delta_sorted.min(), delta_sorted.max(), n_bins + 1)
        idx = np.clip(np.digitize(delta_sorted, edges[1:-1]), 0, n_bins - 1)
        centers = (edges[:-1] + edges[1:]) / 2.0
        means = np.full(n_bins, np.nan)
        for b in range(n_bins):
            mask = idx == b
            if mask.sum() > 0:
                means[b] = div_sorted[mask].mean()
        valid = ~np.isnan(means)

        ax.plot(centers[valid], means[valid], marker='o', markersize=3, label=f'K={K}')
        print(f"K={K}: n={len(delta)}, delta range=[{delta.min():.4f}, {delta.max():.4f}], "
              f"peak divergence={np.nanmax(means):.4f} at delta={centers[np.nanargmax(means)]:.4f}")

    ax.axhline(0.0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('delta')
    ax.set_ylabel('divergence (true_recall - IF)')
    ax.set_title('SIFT real anchors: divergence vs delta, fine resolution')
    ax.legend()
    plt.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"\nWrote {OUT_PATH}")


if __name__ == '__main__':
    main()
