#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""First-pass test of the inverse-coverage hit policy on the real greedy
population mechanism (see ../mechanism_spec_v2.md). Unlike the earlier
K+Delta feasibility script (inverse_coverage_feasibility_run_experiment.py),
this uses each selected candidate's real, emergent r_c/R_c exactly as the
population pipeline produced them - no synthetic reserve construction.

r_c is a triangle-inequality containment guarantee over another train
query's true K-sphere (utils/base/population.py::_propose_nested_spheres),
and R_c is the full set of corpus points within r_c, generally much larger
than K. coverage()/accumulate_to_k()/recall_hat_from_r_hat() from
common/inverse_coverage.py are construction-agnostic and reused unchanged.

No confidence gate or recall_hat threshold is applied this pass - every
(query, candidate) pair's (recall_hat, true_recall, min_coverage) is
recorded for a later calibration pass.

E1: synthetic exact-arithmetic correctness gate (unchanged from the earlier
script - self-contained, doesn't touch the population pipeline).
pilot: real wall-clock cost of the per-candidate evaluation, to set the
main run's sample size.
main: full evaluation of every selected candidate against sampled real
test queries."""

import sys
import time
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'common'))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import numpy as np

from inverse_coverage import coverage, accumulate_to_k, recall_hat_from_r_hat
from primitives import exact_intersection_fraction
from part2_uniform_baseline_run_experiment import _bruteforce_knn, _chord_to_angle
from datasets.dataloaders import load_sift_benchmark, load_esci_benchmark
from utils.base.population import propose_candidates_ball
from utils.base.selection import select_candidates
from simulations.population.population_runner import _load_cache_gt_arrays, _load_gt_indices

#-------------------------------------------------------------------------------

METRIC = 'angular'
N_MAX = 20
SEED = 42
BUDGET_FRACTION = 0.99  # never 1.0, per this project's convention

DATASETS = {
    'sift': {'benchmark_dir': 'datasets/sift/sift_b50k_c2048k100_t512n20_s42', 'D': 128, 'K': 100},
    'esci': {'benchmark_dir': 'datasets/esci/esci_c1024k100_t512n20_s42', 'D': 384, 'K': 100},
}

OUT_DIR = Path(__file__).parent.parent / 'results' / 'inverse_coverage_feasibility'
# 0.99 already run and saved separately (hitpolicy_{dataset}.npz) - not repeated here.
# Never 1.0, per project convention.
BUDGET_SWEEP_LEVELS = (0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

#-------------------------------------------------------------------------------


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vecs / norms

#-------------------------------------------------------------------------------
# E1: synthetic, exact-arithmetic correctness gate (unchanged)
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
        reserve_idx = idx_c[0]

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
# Real population pipeline (unmodified reuse)
#-------------------------------------------------------------------------------


def load_population_inputs(dataset: str, repo_root: Path) -> dict:
    cfg = DATASETS[dataset]
    benchmark_dir = repo_root / cfg['benchmark_dir']
    loader = load_sift_benchmark if dataset == 'sift' else load_esci_benchmark
    base, cache_q, test_q, config = loader(str(benchmark_dir))
    K = cfg['K']
    n_cache = len(cache_q)

    obs_r_Q = np.load(benchmark_dir / 'cache_r_Q.npy')
    test_r_t = np.load(benchmark_dir / 'test_r_t.npy')
    obs_tv, obs_ti, obs_hg = _load_cache_gt_arrays(benchmark_dir, K, n_cache, base)

    return dict(base=base, cache_q=cache_q, test_q=test_q, config=config, benchmark_dir=benchmark_dir,
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

#-------------------------------------------------------------------------------
# Real entries: r_c/R_c exactly as selected, no reserve construction
#-------------------------------------------------------------------------------


def extract_entries(selected: list) -> list:
    """Each selected candidate's real pole, rho_c (radians), and full stored
    point set - no brute-force lookup, no reserve size assumption."""
    entries = []
    for cand in selected:
        pole = _normalize(cand.query[None, :])[0]
        stored_idx = np.flatnonzero(cand.R_c)
        entries.append({
            'pole': pole,
            'rho_c': float(cand.r_c) * np.pi,
            'stored_idx': stored_idx,
            'rc_size': len(stored_idx),
        })
    return entries


def evaluate_query_against_entry(q: np.ndarray, entry: dict, base_hat: np.ndarray, K: int):
    """Returns (recall_hat, min_coverage, true_response_idx, recall_hat_if)
    for one (query, candidate) pair, or None if the entry can't even serve K
    points (should not occur by construction - flagged rather than silently
    handled).

    recall_hat_if is vanilla intersection-fraction: exact_intersection_fraction
    of the candidate's own ball (rho_c) against a proxy for the test query's
    true K-sphere (theta_2), avoiding oracle ground truth by reusing the
    radius-proxy convention from the earlier (deprecated) IF work - the
    distance from the test query to the K-th closest vector already in the
    candidate's own stored set, i.e. t_sorted[K-1] - the same quantity
    already computed for inverse-coverage counting, at no extra cost to get."""
    stored_idx = entry['stored_idx']
    if len(stored_idx) < K:
        return None

    pole, rho_c = entry['pole'], entry['rho_c']
    delta = float(np.arccos(np.clip(pole @ q, -1.0, 1.0)))

    t = np.arccos(np.clip(base_hat[stored_idx] @ q, -1.0, 1.0))
    order = np.argsort(t)
    t_sorted = t[order]
    response_idx = stored_idx[order[:K]]

    if delta < 1e-9:
        cov = coverage(t_sorted, 1e-9, rho_c, base_hat.shape[1])
    else:
        cov = coverage(t_sorted, delta, rho_c, base_hat.shape[1])
    r_hat, min_cov, n_used = accumulate_to_k(t_sorted, cov, K)
    recall_hat = recall_hat_from_r_hat(r_hat, t_sorted[:K])

    r_t_proxy = float(t_sorted[K - 1])
    recall_hat_if = exact_intersection_fraction(rho_c, r_t_proxy, delta, base_hat.shape[1])

    return recall_hat, min_cov, response_idx, recall_hat_if

#-------------------------------------------------------------------------------
# Cost pilot
#-------------------------------------------------------------------------------


def run_pilot(dataset: str, n_pilot_queries: int = 5, seed: int = SEED,
              budget_fraction: float = BUDGET_FRACTION) -> None:
    print(f"=== pilot: {dataset} (budget_fraction={budget_fraction}) ===")
    cfg = DATASETS[dataset]
    D, K = cfg['D'], cfg['K']
    repo_root = Path(__file__).parent.parent.parent.parent

    inputs = load_population_inputs(dataset, repo_root)
    base_hat = _normalize(inputs['base'])
    test_q_hat = _normalize(inputs['test_q'])

    selected = propose_and_select(inputs, D, K, budget_fraction, seed)
    entries = extract_entries(selected)
    if len(entries) == 0:
        print("  n_candidates=0 - budget too small to fit even the cheapest candidate, skipping")
        return
    rc_sizes = np.array([e['rc_size'] for e in entries])
    print(f"  n_candidates={len(entries)}, rc_size: mean={rc_sizes.mean():.1f} "
          f"median={np.median(rc_sizes):.1f} min={rc_sizes.min()} max={rc_sizes.max()}")

    rng = np.random.default_rng(seed + 1)
    pilot_qi = rng.choice(len(test_q_hat), size=min(n_pilot_queries, len(test_q_hat)), replace=False)

    t0 = time.perf_counter()
    n_pairs = 0
    for qi in pilot_qi:
        q = test_q_hat[qi]
        for entry in entries:
            result = evaluate_query_against_entry(q, entry, base_hat, K)
            n_pairs += 1
    elapsed = time.perf_counter() - t0

    per_pair_ms = (elapsed / n_pairs) * 1e3
    n_test = len(test_q_hat)
    est_full_run_s = per_pair_ms * 1e-3 * len(entries) * n_test
    print(f"  {n_pairs} pairs in {elapsed:.2f}s -> {per_pair_ms:.3f} ms/pair")
    print(f"  estimated cost for all {n_test} test queries x {len(entries)} candidates: {est_full_run_s:.1f}s")


def run_budget_sweep_pilot(dataset: str, budget_levels=BUDGET_SWEEP_LEVELS,
                            n_pilot_queries: int = 5, seed: int = SEED) -> None:
    for budget_fraction in budget_levels:
        run_pilot(dataset, n_pilot_queries=n_pilot_queries, seed=seed, budget_fraction=budget_fraction)

#-------------------------------------------------------------------------------
# Main experiment
#-------------------------------------------------------------------------------


def run_main(dataset: str, n_test_samples: int, seed: int = SEED,
             budget_fraction: float = BUDGET_FRACTION, sampled_qi: np.ndarray = None) -> dict:
    """sampled_qi, if given, overrides n_test_samples/seed-based sampling - used by the
    budget sweep to evaluate every budget level against the exact same test queries."""
    cfg = DATASETS[dataset]
    D, K = cfg['D'], cfg['K']
    repo_root = Path(__file__).parent.parent.parent.parent

    inputs = load_population_inputs(dataset, repo_root)
    base_hat = _normalize(inputs['base'])
    test_q_hat = _normalize(inputs['test_q'])
    n_test = len(test_q_hat)

    selected = propose_and_select(inputs, D, K, budget_fraction, seed)
    entries = extract_entries(selected)
    print(f"  [{dataset}] selected {len(entries)} candidates at budget_fraction={budget_fraction}")

    test_gt_idx = _load_gt_indices(inputs['benchmark_dir'] / f'test_gt_population_K{K}.npz', n_test)

    if sampled_qi is None:
        rng = np.random.default_rng(seed + 1)
        n_sample = min(n_test_samples, n_test)
        sampled_qi = rng.choice(n_test, size=n_sample, replace=False)

    pair_recall_hat, pair_true_recall, pair_min_cov, pair_recall_hat_if = [], [], [], []
    pair_query_idx, pair_cand_idx = [], []
    selected_true_recall, oracle_best_true_recall = [], []
    selected_true_recall_if = []

    for qi in sampled_qi:
        q = test_q_hat[qi]
        true_topk = set(test_gt_idx[qi])

        q_recall_hats, q_true_recalls, q_recall_hats_if = [], [], []
        for ci, entry in enumerate(entries):
            result = evaluate_query_against_entry(q, entry, base_hat, K)
            if result is None:
                continue
            recall_hat, min_cov, response_idx, recall_hat_if = result
            true_recall = len(true_topk & set(response_idx.tolist())) / K

            pair_recall_hat.append(recall_hat)
            pair_true_recall.append(true_recall)
            pair_min_cov.append(min_cov)
            pair_recall_hat_if.append(recall_hat_if)
            pair_query_idx.append(qi)
            pair_cand_idx.append(ci)
            q_recall_hats.append(recall_hat)
            q_true_recalls.append(true_recall)
            q_recall_hats_if.append(recall_hat_if)

        if not q_recall_hats:
            continue
        q_recall_hats = np.array(q_recall_hats)
        q_true_recalls = np.array(q_true_recalls)
        q_recall_hats_if = np.array(q_recall_hats_if)
        argmax_idx = int(np.argmax(q_recall_hats))
        selected_true_recall.append(q_true_recalls[argmax_idx])
        oracle_best_true_recall.append(q_true_recalls.max())
        argmax_idx_if = int(np.argmax(q_recall_hats_if))
        selected_true_recall_if.append(q_true_recalls[argmax_idx_if])

    print(f"  [{dataset}] {len(sampled_qi)} queries x {len(entries)} candidates -> "
          f"{len(pair_recall_hat)} pairs recorded")

    result = {
        'recall_hat': np.array(pair_recall_hat), 'true_recall': np.array(pair_true_recall),
        'recall_hat_if': np.array(pair_recall_hat_if),
        'selected_true_recall_if': np.array(selected_true_recall_if),
        'min_coverage': np.array(pair_min_cov), 'query_idx': np.array(pair_query_idx),
        'cand_idx': np.array(pair_cand_idx), 'selected_true_recall': np.array(selected_true_recall),
        'oracle_best_true_recall': np.array(oracle_best_true_recall), 'n_candidates': len(entries),
        'budget_fraction': budget_fraction,
    }
    return result

#-------------------------------------------------------------------------------
# Budget sweep
#-------------------------------------------------------------------------------


def run_budget_sweep(dataset: str, n_test_samples: int, budget_levels=BUDGET_SWEEP_LEVELS,
                      seed: int = SEED) -> dict:
    """Evaluates every budget level against the same fixed test-query sample,
    so differences across levels reflect the candidate-set change, not
    query-sampling noise. 0.99 is not included - already run and saved
    separately as hitpolicy_{dataset}.npz."""
    cfg = DATASETS[dataset]
    repo_root = Path(__file__).parent.parent.parent.parent
    inputs = load_population_inputs(dataset, repo_root)
    n_test = len(inputs['test_q'])

    rng = np.random.default_rng(seed + 1)
    n_sample = min(n_test_samples, n_test)
    sampled_qi = rng.choice(n_test, size=n_sample, replace=False)

    all_recall_hat, all_true_recall, all_min_cov, all_recall_hat_if = [], [], [], []
    all_query_idx, all_cand_idx, all_budget = [], [], []
    all_sel_true_recall, all_oracle_true_recall, all_sel_budget = [], [], []
    all_sel_true_recall_if = []
    n_candidates_by_budget = {}

    for budget_fraction in budget_levels:
        result = run_main(dataset, n_test_samples, seed=seed, budget_fraction=budget_fraction,
                           sampled_qi=sampled_qi)
        n_pairs = len(result['recall_hat'])
        all_recall_hat.append(result['recall_hat'])
        all_true_recall.append(result['true_recall'])
        all_min_cov.append(result['min_coverage'])
        all_recall_hat_if.append(result['recall_hat_if'])
        all_query_idx.append(result['query_idx'])
        all_cand_idx.append(result['cand_idx'])
        all_budget.append(np.full(n_pairs, budget_fraction))
        all_sel_true_recall.append(result['selected_true_recall'])
        all_oracle_true_recall.append(result['oracle_best_true_recall'])
        all_sel_true_recall_if.append(result['selected_true_recall_if'])
        all_sel_budget.append(np.full(len(result['selected_true_recall']), budget_fraction))
        n_candidates_by_budget[budget_fraction] = result['n_candidates']

    return {
        'recall_hat': np.concatenate(all_recall_hat), 'true_recall': np.concatenate(all_true_recall),
        'min_coverage': np.concatenate(all_min_cov), 'recall_hat_if': np.concatenate(all_recall_hat_if),
        'query_idx': np.concatenate(all_query_idx),
        'cand_idx': np.concatenate(all_cand_idx), 'pair_budget_fraction': np.concatenate(all_budget),
        'selected_true_recall': np.concatenate(all_sel_true_recall),
        'oracle_best_true_recall': np.concatenate(all_oracle_true_recall),
        'selected_true_recall_if': np.concatenate(all_sel_true_recall_if),
        'query_budget_fraction': np.concatenate(all_sel_budget),
        'budget_levels': np.array(budget_levels),
        'n_candidates_by_budget': np.array([n_candidates_by_budget[b] for b in budget_levels]),
    }

#-------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',
                         choices=['e1', 'pilot', 'main', 'budget_sweep_pilot', 'budget_sweep', 'all'],
                         default='all')
    parser.add_argument('--dataset', choices=['sift', 'esci', 'both'], default='both')
    parser.add_argument('--n-samples', type=int, default=400)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = ['sift', 'esci'] if args.dataset == 'both' else [args.dataset]

    if args.experiment == 'budget_sweep_pilot':
        for ds in datasets:
            run_budget_sweep_pilot(ds, seed=args.seed)
        return

    if args.experiment == 'budget_sweep':
        print("=== budget sweep ===")
        for ds in datasets:
            result = run_budget_sweep(ds, n_test_samples=args.n_samples, seed=args.seed)
            np.savez_compressed(OUT_DIR / f'hitpolicy_budget_sweep_{ds}.npz', **result)
            print(f"  wrote {OUT_DIR / f'hitpolicy_budget_sweep_{ds}.npz'}")
        return

    if args.experiment in ('e1', 'all'):
        run_e1(seed=args.seed)
        print()

    if args.experiment in ('pilot', 'all'):
        for ds in datasets:
            run_pilot(ds, seed=args.seed)
        print()

    if args.experiment in ('main', 'all'):
        print("=== main: real population, no confidence gate ===")
        for ds in datasets:
            result = run_main(ds, n_test_samples=args.n_samples, seed=args.seed)
            np.savez_compressed(OUT_DIR / f'hitpolicy_{ds}.npz', **result)
            print(f"  wrote {OUT_DIR / f'hitpolicy_{ds}.npz'}")
        print()

    print("Done.")


if __name__ == '__main__':
    main()
