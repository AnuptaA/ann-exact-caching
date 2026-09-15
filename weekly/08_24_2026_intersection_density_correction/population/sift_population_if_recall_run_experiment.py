#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Realistic workload: IF vs. recall on population-selected SIFT cache sets.

Runs the production population pipeline (ball-like proposal + CIG-based
greedy selection, unmodified) at a sweep of budgets, then scores every
real test query's argmax-IF cache hit against the actual response set
that hit would return. Pooled across all budgets; budget is only a
mechanism for generating differently-sized realistic cache populations,
not an axis to stratify by.
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'common'))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import numpy as np
from tqdm import tqdm

from primitives import exact_intersection_fraction_batch
from part2_uniform_baseline_run_experiment import _bruteforce_knn, _chord_to_angle
from datasets.dataloaders import load_sift_benchmark
from utils.base.population import propose_candidates_ball
from utils.base.selection import select_candidates
from simulations.population.population_runner import _load_cache_gt_arrays, _load_gt_indices

#-------------------------------------------------------------------------------

BENCHMARK_DIR = Path('datasets/sift/sift_b50k_c2048k100_t512n20_s42')
METRIC = 'angular'
N_MAX = 20
K = 100
SEED = 42
D = 128

BUDGET_FRACTIONS = [0.01, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]

OUT_DIR = Path(__file__).parent.parent / 'results' / 'sift_population_if_recall'

#-------------------------------------------------------------------------------


def load_population_inputs(repo_root: Path) -> dict:
    benchmark_dir = repo_root / BENCHMARK_DIR
    base, cache_q, test_q, config = load_sift_benchmark(str(benchmark_dir))
    n_cache = len(cache_q)
    n_test = len(test_q)

    obs_r_Q = np.load(benchmark_dir / 'cache_r_Q.npy')
    test_r_t = np.load(benchmark_dir / 'test_r_t.npy')
    obs_tv, obs_ti, obs_hg = _load_cache_gt_arrays(benchmark_dir, K, n_cache, base)
    test_gt_indices = _load_gt_indices(benchmark_dir / f'test_gt_population_K{K}.npz', n_test)

    return dict(base=base, cache_q=cache_q, test_q=test_q, config=config,
                obs_r_Q=obs_r_Q, test_r_t=test_r_t, obs_tv=obs_tv, obs_ti=obs_ti,
                obs_hg=obs_hg, test_gt_indices=test_gt_indices)


def propose_once(inputs: dict) -> list:
    return propose_candidates_ball(
        inputs['cache_q'], inputs['obs_r_Q'], inputs['obs_tv'], inputs['obs_ti'], inputs['obs_hg'],
        inputs['cache_q'], inputs['obs_r_Q'], inputs['test_q'], inputs['test_r_t'],
        inputs['base'], METRIC, n_max=N_MAX, K=K,
    )


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vecs / norms


def precompute_test_query_ground_truth(base: np.ndarray, test_q: np.ndarray, k: int) -> tuple:
    """Bare K-NN radius (radians) and true top-k indices per test query,
    independent of budget. _bruteforce_knn assumes unit-norm inputs."""
    dists, idx = _bruteforce_knn(_normalize(base), _normalize(test_q), k)
    r_bare = _chord_to_angle(dists[:, k - 1])
    return r_bare, idx[:, :k]


def score_selected(selected: list, test_q: np.ndarray, r_bare: np.ndarray, true_topk_idx: np.ndarray,
                    budget_fraction: float, d: int, k: int) -> dict:
    n_test = test_q.shape[0]
    cand_queries = np.array([c.query for c in selected])
    cand_norms = np.linalg.norm(cand_queries, axis=1, keepdims=True)
    cand_norms[cand_norms == 0.0] = 1.0
    cand_hat = cand_queries / cand_norms
    r_cache = np.array([c.r_c for c in selected], dtype=np.float64) * np.pi
    r_bare_source = np.array([c.r_Q for c in selected], dtype=np.float64)
    r_bare_source[r_bare_source == 0.0] = 1.0
    inflation_per_candidate = (np.array([c.r_c for c in selected], dtype=np.float64) / r_bare_source)

    test_norms = np.linalg.norm(test_q, axis=1, keepdims=True)
    test_norms[test_norms == 0.0] = 1.0
    test_hat = test_q / test_norms

    out_if = np.empty(n_test)
    out_recall = np.empty(n_test)
    out_inflation = np.empty(n_test)

    for i in range(n_test):
        delta = np.arccos(np.clip(cand_hat @ test_hat[i], -1.0, 1.0))
        ifs = exact_intersection_fraction_batch(r_cache, float(r_bare[i]), delta, d)
        best = int(ifs.argmax())
        response = np.flatnonzero(selected[best].R_c)
        true_topk = set(true_topk_idx[i].tolist())
        out_if[i] = ifs[best]
        out_recall[i] = len(true_topk & set(response.tolist())) / k
        out_inflation[i] = inflation_per_candidate[best]

    return {'fraction': out_if, 'recall': out_recall, 'inflation_factor': out_inflation,
            'budget_fraction': np.full(n_test, budget_fraction)}

#-------------------------------------------------------------------------------


def _write_progress(progress_path: Path, done: int, total: int, start_time: float) -> None:
    tmp_path = progress_path.with_suffix('.tmp')
    payload = {
        'budgets_done': done, 'budgets_total': total,
        'elapsed_seconds': time.time() - start_time,
        'timestamp': datetime.now().isoformat(),
    }
    tmp_path.write_text(json.dumps(payload))
    tmp_path.replace(progress_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget-fractions', type=float, nargs='+', default=BUDGET_FRACTIONS,
                         dest='budget_fractions')
    parser.add_argument('--out-dir', type=str, default=None, dest='out_dir')
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).parent.parent.parent.parent
    print("Loading population inputs...")
    t0 = time.perf_counter()
    inputs = load_population_inputs(repo_root)
    n_base = inputs['base'].shape[0]
    print(f"  n_base={n_base}, n_cache={len(inputs['cache_q'])}, n_test={len(inputs['test_q'])}")

    print("Proposing candidates (n_max={}, K={})...".format(N_MAX, K))
    t_propose_start = time.perf_counter()
    candidates = propose_once(inputs)
    t_propose = time.perf_counter() - t_propose_start
    print(f"  proposed {len(candidates)} candidates in {t_propose:.2f}s")

    print("Precomputing test-query ground truth (bare K-NN radius, true top-K)...")
    r_bare, true_topk_idx = precompute_test_query_ground_truth(inputs['base'], inputs['test_q'], K)

    progress_path = out_dir / 'progress.json'
    start_time = time.time()
    pooled = {key: [] for key in ('fraction', 'recall', 'inflation_factor', 'budget_fraction')}
    timing = {'propose_seconds': t_propose, 'per_budget': []}

    for i, fraction in enumerate(tqdm(args.budget_fractions, desc='budgets', mininterval=1.0)):
        budget_B = max(1, int(fraction * n_base))

        t_select_start = time.perf_counter()
        selected, trajectory = select_candidates(
            candidates, budget_B, 'greedy_cost',
            n_train_queries=len(inputs['cache_q']), n_test_queries=len(inputs['test_q']), seed=SEED,
        )
        t_select = time.perf_counter() - t_select_start

        t_score_start = time.perf_counter()
        if selected:
            result = score_selected(selected, inputs['test_q'], r_bare, true_topk_idx, fraction, D, K)
            for key in pooled:
                pooled[key].append(result[key])
        t_score = time.perf_counter() - t_score_start

        timing['per_budget'].append({
            'budget_fraction': fraction, 'budget_B': budget_B, 'n_selected': len(selected),
            'select_seconds': t_select, 'score_seconds': t_score,
        })
        print(f"  budget={fraction:.2f} (B={budget_B}): {len(selected)} selected, "
              f"select={t_select:.2f}s, score={t_score:.2f}s")

        _write_progress(progress_path, i + 1, len(args.budget_fractions), start_time)

    for key in pooled:
        pooled[key] = np.concatenate(pooled[key]) if pooled[key] else np.array([])

    out_path = out_dir / 'pooled.npz'
    np.savez_compressed(out_path, **pooled)
    with open(out_dir / 'timing.json', 'w') as f:
        json.dump(timing, f, indent=2)

    print(f"\nWrote {out_path}")
    print(f"Total wall clock: {time.perf_counter() - t0:.2f}s")
    print("Done.")


if __name__ == '__main__':
    main()
