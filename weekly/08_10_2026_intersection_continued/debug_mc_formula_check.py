#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Monte Carlo cross-check of exact_intersection_fraction at d=128,
using real (r_cache, r_test, delta) triples from the SIFT pairwise data.
"""

import sys
import math
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np

from primitives import exact_intersection_fraction, sample_cap_points_direct

#-------------------------------------------------------------------------------

D = 128
N_TRIALS = 25
N_MC = 100_000
SIFT_PAIRWISE_RUN = Path(__file__).parent / 'results_sift_pairwise_recall' / '20260813_170113'

#-------------------------------------------------------------------------------


def load_real_triples(run_dir: Path, K: int, n_trials: int, rng: np.random.Generator) -> list:
    """n_trials real (r_cache, r_test, delta) triples, stratified by rank
    within the nonzero-fraction subset."""
    path = run_dir / 'debug_sift' / f'{D}d' / 'raw' / f'K{K}_pairwise.npz'
    arr = np.load(path)
    fraction = arr['fraction']

    informative = np.where(fraction > 1e-6)[0]
    order = informative[np.argsort(fraction[informative])]
    stratum_edges = np.linspace(0, len(order) - 1, n_trials).astype(int)
    idx = order[stratum_edges]

    return [(float(arr['r_cache'][i]), float(arr['r_test'][i]), float(arr['delta'][i]), float(fraction[i]))
            for i in idx]


def mc_check(theta_1: float, theta_2: float, delta: float, d: int, n_mc: int, rng: np.random.Generator) -> tuple:
    pole_2 = np.zeros(d); pole_2[0] = 1.0
    pole_1 = np.zeros(d); pole_1[0] = math.cos(delta); pole_1[1] = math.sin(delta)

    pts = sample_cap_points_direct(pole_2, theta_2, n_mc, rng)
    dist_to_pole1 = np.arccos(np.clip(pts @ pole_1, -1.0, 1.0))
    mc = float(np.mean(dist_to_pole1 <= theta_1))
    se = math.sqrt(max(mc * (1 - mc), 1e-12) / n_mc)
    return mc, se


def main() -> None:
    rng = np.random.default_rng(7)
    triples = load_real_triples(SIFT_PAIRWISE_RUN, K=25, n_trials=N_TRIALS, rng=rng)

    print(f"MC cross-check of exact_intersection_fraction at d={D}, {N_TRIALS} real triples, "
          f"{N_MC} MC samples each\n")

    n_fail = 0
    max_diff = 0.0
    for i, (r_cache, r_test, delta, stored_fraction) in enumerate(triples):
        exact = exact_intersection_fraction(r_cache, r_test, delta, D)
        mc, se = mc_check(r_cache, r_test, delta, D, N_MC, rng)
        diff = abs(exact - mc)
        max_diff = max(max_diff, diff)
        tol = 4 * se + 0.002
        ok = diff < tol
        n_fail += not ok
        status = 'PASS' if ok else 'FAIL'
        print(f"  [{status}] r_cache={r_cache:.4f} r_test={r_test:.4f} delta={delta:.4f}  "
              f"exact={exact:.5f} mc={mc:.5f} se={se:.5f} diff={diff:.5f}")

    print(f"\n{N_TRIALS - n_fail}/{N_TRIALS} passed. Max diff: {max_diff:.5f}")
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
