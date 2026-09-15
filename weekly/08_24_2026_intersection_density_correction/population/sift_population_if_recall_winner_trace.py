#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Re-derives winner-candidate metadata (source_index, n, nn_dist_of_source)
for the population IF/recall sweep, for characterizing the high-inflation
bin. Deterministic replay of the same pipeline (same seed/budgets); the
fraction/recall/inflation columns should match the existing pooled.npz
exactly.
"""

import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'common'))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import numpy as np
from tqdm import tqdm

from primitives import exact_intersection_fraction_batch
from utils.base.population import _pairwise_cache_dists
from utils.base.selection import select_candidates
from sift_population_if_recall_run_experiment import (
    BUDGET_FRACTIONS, D, K, METRIC, SEED,
    load_population_inputs, propose_once, precompute_test_query_ground_truth,
)

#-------------------------------------------------------------------------------

OUT_PATH = Path(__file__).parent.parent / 'results' / 'sift_population_if_recall' / 'winner_trace.npz'

#-------------------------------------------------------------------------------


def score_with_winner_metadata(selected: list, test_q: np.ndarray, r_bare: np.ndarray,
                                true_topk_idx: np.ndarray, sorted_cache_dists: np.ndarray,
                                budget_fraction: float, d: int, k: int) -> dict:
    n_test = test_q.shape[0]
    cand_queries = np.array([c.query for c in selected])
    cand_norms = np.linalg.norm(cand_queries, axis=1, keepdims=True)
    cand_norms[cand_norms == 0.0] = 1.0
    cand_hat = cand_queries / cand_norms
    r_cache = np.array([c.r_c for c in selected], dtype=np.float64) * np.pi
    r_bare_source = np.array([c.r_Q for c in selected], dtype=np.float64)
    r_bare_source[r_bare_source == 0.0] = 1.0
    inflation_per_candidate = np.array([c.r_c for c in selected], dtype=np.float64) / r_bare_source
    source_index_arr = np.array([c.source_index for c in selected], dtype=np.int64)
    n_arr = np.array([c.n for c in selected], dtype=np.int64)
    nn_dist_arr = np.array([sorted_cache_dists[c.source_index, c.n - 1] for c in selected], dtype=np.float64)

    test_norms = np.linalg.norm(test_q, axis=1, keepdims=True)
    test_norms[test_norms == 0.0] = 1.0
    test_hat = test_q / test_norms

    out = {key: np.empty(n_test) for key in ('fraction', 'recall', 'inflation_factor', 'source_index', 'n', 'nn_dist_of_source')}

    for i in range(n_test):
        delta = np.arccos(np.clip(cand_hat @ test_hat[i], -1.0, 1.0))
        ifs = exact_intersection_fraction_batch(r_cache, float(r_bare[i]), delta, d)
        best = int(ifs.argmax())
        response = np.flatnonzero(selected[best].R_c)
        true_topk = set(true_topk_idx[i].tolist())
        out['fraction'][i] = ifs[best]
        out['recall'][i] = len(true_topk & set(response.tolist())) / k
        out['inflation_factor'][i] = inflation_per_candidate[best]
        out['source_index'][i] = source_index_arr[best]
        out['n'][i] = n_arr[best]
        out['nn_dist_of_source'][i] = nn_dist_arr[best]

    out['budget_fraction'] = np.full(n_test, budget_fraction)
    return out


def main() -> None:
    repo_root = Path(__file__).parent.parent.parent.parent
    print("Loading population inputs...")
    inputs = load_population_inputs(repo_root)
    n_base = inputs['base'].shape[0]

    print("Proposing candidates...")
    candidates = propose_once(inputs)

    print("Precomputing test-query ground truth...")
    r_bare, true_topk_idx = precompute_test_query_ground_truth(inputs['base'], inputs['test_q'], K)

    print("Computing pairwise cache-query distances (for nn_dist_of_source)...")
    sorted_cache_dists, _ = _pairwise_cache_dists(inputs['cache_q'], METRIC)

    pooled = {key: [] for key in ('fraction', 'recall', 'inflation_factor', 'source_index', 'n',
                                    'nn_dist_of_source', 'budget_fraction')}
    t0 = time.perf_counter()
    for fraction in tqdm(BUDGET_FRACTIONS, desc='budgets'):
        budget_B = max(1, int(fraction * n_base))
        selected, _ = select_candidates(
            candidates, budget_B, 'greedy_cost',
            n_train_queries=len(inputs['cache_q']), n_test_queries=len(inputs['test_q']), seed=SEED,
        )
        if not selected:
            continue
        result = score_with_winner_metadata(selected, inputs['test_q'], r_bare, true_topk_idx,
                                             sorted_cache_dists, fraction, D, K)
        for key in pooled:
            pooled[key].append(result[key])

    for key in pooled:
        pooled[key] = np.concatenate(pooled[key]) if pooled[key] else np.array([])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_PATH, **pooled)
    print(f"\nWrote {OUT_PATH}")
    print(f"Total wall clock: {time.perf_counter() - t0:.2f}s")
    print("Done.")


if __name__ == '__main__':
    main()
