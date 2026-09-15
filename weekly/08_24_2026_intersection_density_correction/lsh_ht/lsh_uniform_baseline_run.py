#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Runs the uniform-sampling baseline across the same datasets/queries
used in E3, sweeping sample fraction, for direct comparison against the
LSH-HT frontier."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'common'))

import numpy as np

from lsh_ht_grid_run_experiment import OUT_DIR, N, K, E3_DATASETS, _direct_radii
from lsh_uniform_baseline import uniform_sample_estimate_batch

#-------------------------------------------------------------------------------

FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90, 0.99]

#-------------------------------------------------------------------------------


def main() -> None:
    out_dir = OUT_DIR / 'uniform_baseline'
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset_key, loader in E3_DATASETS.items():
        data = loader()
        radii = _direct_radii(data['points'], data['queries'], K)
        true_counts = np.array([
            int((np.arccos(np.clip(data['points'] @ data['queries'][i], -1.0, 1.0)) <= radii[i] + 1e-9).sum())
            for i in range(len(data['queries']))
        ])
        valid = true_counts > 0
        n_points = data['points'].shape[0]

        results = []
        for frac in FRACTIONS:
            n = max(1, int(round(frac * n_points)))
            rng = np.random.default_rng(hash((dataset_key, frac)) % (2**32))
            n_hat = uniform_sample_estimate_batch(data['points'], data['queries'], radii, n, rng)
            rel_err = np.abs(n_hat[valid] - true_counts[valid]) / true_counts[valid]
            results.append({'frac': frac, 'n': n, 'mean_rel_err': float(rel_err.mean()),
                             'mean_ratio': float(np.mean(n_hat[valid] / true_counts[valid]))})
            print(f"  {dataset_key} frac={frac:.2f} (n={n}): mean_rel_err={rel_err.mean():.4f}, "
                  f"mean_ratio={np.mean(n_hat[valid] / true_counts[valid]):.4f}")

        np.savez_compressed(out_dir / f'{dataset_key}.npz',
                             fracs=np.array([r['frac'] for r in results]),
                             mean_rel_err=np.array([r['mean_rel_err'] for r in results]),
                             mean_ratio=np.array([r['mean_ratio'] for r in results]))

    print("\nDone.")


if __name__ == '__main__':
    main()
