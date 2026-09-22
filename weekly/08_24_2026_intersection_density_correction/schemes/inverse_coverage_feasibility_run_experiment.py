#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Feasibility check for inverse-coverage counting (see
../POSTMORTEM.md and common/inverse_coverage.py) - the successor to the
deprecated full-corpus-dependent recall estimators. Never touches
corpus-scale structure at query time: everything needed comes from the
incoming query and a cache entry's own stored (K+Delta) neighbors.

E1: synthetic, exact-arithmetic correctness gate (small N, no LSH/no
approximation of any kind) - confirms the port before touching real data.
E2: real SIFT/ESCI, real greedy-population cache entries, real query set
conventions - the main feasibility read, pilot-scale.
E3: Delta sweep on real data, reusing E2's harness.
E4: real per-candidate wall-clock cost, using the actual
single_cap_fraction/quadrature code path.

Reuses (does not reimplement): common/primitives.py::single_cap_fraction
(via common/inverse_coverage.py), common/part2_uniform_baseline_run_experiment.py::_bruteforce_knn,
datasets/dataloaders.py (load_sift_benchmark/load_esci_benchmark),
utils/base/population.py::propose_candidates_ball, utils/base/selection.py::select_candidates,
simulations/population/population_runner.py (_load_cache_gt_arrays/_load_gt_indices).
None of utils/base/, datasets/, or simulations/ is modified - the K+Delta
reserve (which doesn't exist in the population pipeline) is computed here,
as new code, via an extra _bruteforce_knn call per selected candidate."""

import sys
import time
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'common'))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import numpy as np

from inverse_coverage import coverage, accumulate_to_k, recall_hat_from_r_hat
from part2_uniform_baseline_run_experiment import _bruteforce_knn, _chord_to_angle
from datasets.dataloaders import load_sift_benchmark, load_esci_benchmark
from utils.base.population import propose_candidates_ball
from utils.base.selection import select_candidates
from simulations.population.population_runner import _load_cache_gt_arrays, _load_gt_indices

#-------------------------------------------------------------------------------

METRIC = 'angular'
N_MAX = 20
SEED = 42
BUDGET_FRACTION = 0.99  # near-maximal population size (never exactly 1.0, per this project's
                         # convention) - checked empirically: at budget=0.3 (28 candidates) the
                         # nearest cached candidate is typically ~1.85x farther than its own
                         # reserve radius, an unfairly sparse regime; at 0.99 that ratio drops to
                         # ~0.95, a fair test of the mechanism itself rather than of population sparsity

DATASETS = {
    'sift': {'benchmark_dir': 'datasets/sift/sift_b50k_c2048k100_t512n20_s42', 'D': 128, 'K': 100},
    'esci': {'benchmark_dir': 'datasets/esci/esci_c1024k100_t512n20_s42', 'D': 384, 'K': 100},
}

OUT_DIR = Path(__file__).parent.parent / 'results' / 'inverse_coverage_feasibility'

#-------------------------------------------------------------------------------


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vecs / norms

#-------------------------------------------------------------------------------
# E1: synthetic, exact-arithmetic correctness gate
#-------------------------------------------------------------------------------


def run_e1(seed: int = 0, n_trials: int = 300) -> None:
    print("=== E1: synthetic exact-arithmetic correctness gate ===")
    rng = np.random.default_rng(seed)
    N, D, K, DELTA_RESERVE = 20000, 64, 20, 100
    K_RESERVE = K + DELTA_RESERVE

    points = rng.standard_normal((N, D))
    points = _normalize(points)

    r_hats, true_r_stars, recall_hats, true_recalls = [], [], [], []
    for _ in range(n_trials):
        q_c = _normalize(rng.standard_normal((1, D)))[0]
        dists_c, idx_c = _bruteforce_knn(points, q_c[None, :], K_RESERVE)
        rho_c = float(_chord_to_angle(dists_c[0, -1]))
        reserve_idx = idx_c[0]  # K+Delta nearest neighbors of q_c, exact

        delta_dir = rng.standard_normal(D)
        delta_dir -= (delta_dir @ q_c) * q_c
        delta_dir /= np.linalg.norm(delta_dir)
        delta_mag = rng.uniform(0.05, min(rho_c * 1.5, np.pi * 0.4))
        q = _normalize((np.cos(delta_mag) * q_c + np.sin(delta_mag) * delta_dir)[None, :])[0]

        reserve_points = points[reserve_idx]
        t = np.arccos(np.clip(reserve_points @ q, -1.0, 1.0))
        order = np.argsort(t)
        t_sorted = t[order]
        cov = coverage(t_sorted, delta_mag, rho_c, D)
        r_hat, min_cov, n_used = accumulate_to_k(t_sorted, cov, K)

        dists_q, idx_q = _bruteforce_knn(points, q[None, :], K)
        true_r_star = float(_chord_to_angle(dists_q[0, -1]))

        top_k_dists_from_q = np.arccos(np.clip(points[reserve_idx[:K]] @ q, -1.0, 1.0))
        rh = recall_hat_from_r_hat(r_hat, top_k_dists_from_q)
        true_topk = set(idx_q[0].tolist())
        true_recall = len(true_topk & set(reserve_idx[:K].tolist())) / K

        r_hats.append(r_hat)
        true_r_stars.append(true_r_star)
        recall_hats.append(rh)
        true_recalls.append(true_recall)

    r_hats, true_r_stars = np.array(r_hats), np.array(true_r_stars)
    recall_hats, true_recalls = np.array(recall_hats), np.array(true_recalls)
    finite = np.isfinite(r_hats)
    print(f"  trials: {n_trials}, finite r_hat: {finite.sum()}")
    print(f"  mean r_hat: {r_hats[finite].mean():.4f}, mean true r_q*: {true_r_stars[finite].mean():.4f}, "
          f"mean bias: {(true_r_stars[finite] - r_hats[finite]).mean():+.4f}")
    print(f"  mean recall_hat: {recall_hats.mean():.4f}, mean true recall: {true_recalls.mean():.4f}, "
          f"mean bias: {(true_recalls - recall_hats).mean():+.4f}")

#-------------------------------------------------------------------------------
# E2/E3/E4: real data
#-------------------------------------------------------------------------------


def load_population_inputs(dataset: str, repo_root: Path) -> dict:
    cfg = DATASETS[dataset]
    benchmark_dir = repo_root / cfg['benchmark_dir']
    loader = load_sift_benchmark if dataset == 'sift' else load_esci_benchmark
    base, cache_q, test_q, config = loader(str(benchmark_dir))
    K = cfg['K']
    n_cache, n_test = len(cache_q), len(test_q)

    obs_r_Q = np.load(benchmark_dir / 'cache_r_Q.npy')
    test_r_t = np.load(benchmark_dir / 'test_r_t.npy')
    obs_tv, obs_ti, obs_hg = _load_cache_gt_arrays(benchmark_dir, K, n_cache, base)

    return dict(base=base, cache_q=cache_q, test_q=test_q, config=config,
                obs_r_Q=obs_r_Q, test_r_t=test_r_t, obs_tv=obs_tv, obs_ti=obs_ti, obs_hg=obs_hg)


def propose_and_select(inputs: dict, D: int, K: int, budget_fraction: float, seed: int) -> list:
    candidates = propose_candidates_ball(
        inputs['cache_q'], inputs['obs_r_Q'], inputs['obs_tv'], inputs['obs_ti'], inputs['obs_hg'],
        inputs['cache_q'], inputs['obs_r_Q'], inputs['test_q'], inputs['test_r_t'],
        inputs['base'], METRIC, n_max=N_MAX, K=K,
    )
    n_base = inputs['base'].shape[0]
    budget_B = max(1, int(budget_fraction * n_base))
    selected, _ = select_candidates(candidates, budget_B, 'greedy_cost',
                                     n_train_queries=len(inputs['cache_q']),
                                     n_test_queries=len(inputs['test_q']), seed=seed)
    return selected


def compute_reserves(base_hat: np.ndarray, selected: list, K: int, delta_reserve: int) -> list:
    """New code (not a utils/base/ change): for each selected candidate,
    an extra brute-force K+Delta nearest-neighbor lookup around its own
    pole - the reserve the population pipeline doesn't otherwise store."""
    reserves = []
    for cand in selected:
        pole = _normalize(cand.query[None, :])[0]
        dists, idx = _bruteforce_knn(base_hat, pole[None, :], K + delta_reserve)
        reserves.append({'pole': pole, 'rho_c': float(_chord_to_angle(dists[0, -1])), 'idx': idx[0],
                          'top_k_idx': idx[0, :K]})
    return reserves


def run_e2_e3(dataset: str, delta_reserve: int, n_samples: int, seed: int, timing: dict = None,
              budget_fraction: float = BUDGET_FRACTION) -> dict:
    cfg = DATASETS[dataset]
    D, K = cfg['D'], cfg['K']
    repo_root = Path(__file__).parent.parent.parent.parent

    inputs = load_population_inputs(dataset, repo_root)
    base_hat = _normalize(inputs['base'])
    test_q_hat = _normalize(inputs['test_q'])

    selected = propose_and_select(inputs, D, K, budget_fraction, seed)
    print(f"  [{dataset}] selected {len(selected)} candidates at budget_fraction={budget_fraction}")

    reserves = compute_reserves(base_hat, selected, K, delta_reserve)

    rng = np.random.default_rng(seed + 1)
    n_test = len(test_q_hat)
    n_cand = len(selected)

    # Pair each test query with its NEAREST candidate(s) by delta - a realistic
    # cache-hit scenario (a hit-policy only ever considers plausible/nearby
    # candidates), not a uniform-random cross-product, which mostly draws
    # irrelevant far-apart pairs and was confirmed empirically to land almost
    # entirely in a near-zero-recall, near-zero-coverage regime.
    cand_poles = np.array([r['pole'] for r in reserves])  # (n_cand, D)
    delta_matrix = np.arccos(np.clip(test_q_hat @ cand_poles.T, -1.0, 1.0))  # (n_test, n_cand)
    N_NEAREST = min(3, n_cand)
    nearest_cand_idx = np.argsort(delta_matrix, axis=1)[:, :N_NEAREST]  # (n_test, N_NEAREST)

    # Population-density diagnostic (every test query's single nearest candidate,
    # not just the sampled pairs) - lets the budget-sensitivity check use the
    # full query set cheaply, without waiting on the (slower) per-pair loop below.
    nearest_only = nearest_cand_idx[:, 0]
    all_nearest_delta = delta_matrix[np.arange(n_test), nearest_only]
    all_rho_c = np.array([reserves[c]['rho_c'] for c in nearest_only])

    pairs = [(qi, int(nearest_cand_idx[qi, j])) for qi in range(n_test) for j in range(N_NEAREST)]
    rng.shuffle(pairs)

    r_hats, min_covs, recall_hats, true_recalls, baseline_recalls = [], [], [], [], []
    deltas, rho_cs, n_useds = [], [], []
    quad_times = []
    n_collected = 0
    for qi, ci in pairs:
        if n_collected >= n_samples:
            break
        q = test_q_hat[qi]
        r = reserves[ci]
        cand = selected[ci]

        delta = float(delta_matrix[qi, ci])
        if delta < 1e-6:
            continue  # degenerate - test query coincides with the candidate pole

        t = np.arccos(np.clip(base_hat[r['idx']] @ q, -1.0, 1.0))
        order = np.argsort(t)
        t_sorted = t[order]

        t0 = time.perf_counter()
        cov = coverage(t_sorted, delta, r['rho_c'], D)
        r_hat, min_cov, n_used = accumulate_to_k(t_sorted, cov, K)
        quad_times.append(time.perf_counter() - t0)

        top_k_dists_from_q = np.arccos(np.clip(base_hat[r['top_k_idx']] @ q, -1.0, 1.0))
        rh = recall_hat_from_r_hat(r_hat, top_k_dists_from_q)

        dists_true, idx_true = _bruteforce_knn(base_hat, q[None, :], K)
        true_topk = set(idx_true[0].tolist())
        response = set(r['top_k_idx'].tolist())
        true_recall = len(true_topk & response) / K

        r_c_rad = float(cand.r_c) * np.pi  # r_c is stored as a fraction of pi, per score_selected's
                                            # convention in population/sift_population_if_recall_run_experiment.py
        baseline_recall = float(np.mean(top_k_dists_from_q <= r_c_rad))

        r_hats.append(r_hat)
        min_covs.append(min_cov)
        recall_hats.append(rh)
        true_recalls.append(true_recall)
        baseline_recalls.append(baseline_recall)
        deltas.append(delta)
        rho_cs.append(r['rho_c'])
        n_useds.append(n_used)
        n_collected += 1

    print(f"  [{dataset}] collected {n_collected}/{n_samples} pairs (from {len(pairs)} nearest-candidate pairs)")
    result = {
        'r_hat': np.array(r_hats), 'min_coverage': np.array(min_covs),
        'recall_hat': np.array(recall_hats), 'true_recall': np.array(true_recalls),
        'baseline_recall': np.array(baseline_recalls), 'delta_reserve': delta_reserve,
        'delta': np.array(deltas), 'rho_c': np.array(rho_cs), 'n_used': np.array(n_useds),
        'n_candidates': n_cand, 'budget_fraction': budget_fraction,
        'all_nearest_delta': all_nearest_delta, 'all_rho_c': all_rho_c,
    }
    if timing is not None:
        timing[delta_reserve] = np.array(quad_times)
    return result

#-------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=['e1', 'e2', 'e3', 'e4', 'all'], default='all')
    parser.add_argument('--dataset', choices=['sift', 'esci', 'both'], default='both')
    parser.add_argument('--n-samples', type=int, default=800)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = ['sift', 'esci'] if args.dataset == 'both' else [args.dataset]

    if args.experiment in ('e1', 'all'):
        run_e1(seed=args.seed)
        print()

    if args.experiment in ('e2', 'all'):
        print("=== E2: real data, main feasibility read (Delta=200) ===")
        for ds in datasets:
            result = run_e2_e3(ds, delta_reserve=200, n_samples=args.n_samples, seed=args.seed)
            np.savez_compressed(OUT_DIR / f'e2_{ds}.npz', **result)
            print(f"  wrote {OUT_DIR / f'e2_{ds}.npz'}")
        print()

    if args.experiment in ('e3', 'all'):
        print("=== E3: Delta sweep ===")
        for ds in datasets:
            for delta_reserve in (0, 100, 200, 400):
                result = run_e2_e3(ds, delta_reserve=delta_reserve, n_samples=max(200, args.n_samples // 2),
                                    seed=args.seed)
                np.savez_compressed(OUT_DIR / f'e3_{ds}_delta{delta_reserve}.npz', **result)
                print(f"  [{ds}] Delta={delta_reserve}: mean recall_hat bias = "
                      f"{(result['true_recall'] - result['recall_hat']).mean():+.4f}")
        print()

    if args.experiment in ('e4', 'all'):
        print("=== E4: cost, real code path ===")
        timing = {}
        for ds in datasets:
            run_e2_e3(ds, delta_reserve=200, n_samples=max(200, args.n_samples // 2),
                       seed=args.seed, timing=timing)
            t = timing[200]
            print(f"  [{ds}] coverage+accumulate_to_k: mean {t.mean()*1e6:.1f} us/candidate "
                  f"({len(t)} samples)")

    print("\nDone.")


if __name__ == '__main__':
    main()
