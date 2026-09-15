#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Selection-based comparison: IF vs recall_hat = K/N_hat (N_hat via the
validated multi-table Horvitz-Thompson LSH estimator), on the same
population-pipeline candidate pools and argmax-selection methodology
already used for the IF-only validation."""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'common'))
sys.path.append(str(Path(__file__).parent.parent / 'lsh_ht'))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import numpy as np

from primitives import exact_intersection_fraction_batch
from utils.base.selection import select_candidates
from sift_population_if_recall_run_experiment import (
    BUDGET_FRACTIONS, D, K, load_population_inputs, propose_once,
    precompute_test_query_ground_truth,
)
from lsh_horvitz_thompson import build_tables, query_retrieval, estimate_count, select_hyperparameters
from lsh_ht_grid_run_experiment import M_VALUES, X_VALUES, L_VALUES

#-------------------------------------------------------------------------------

SEED = 42
OUT_DIR = Path(__file__).parent.parent / 'results' / 'recall_hat_vs_if'

#-------------------------------------------------------------------------------


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vecs / norms


def compute_candidate_r(selected: list, points: np.ndarray, test_q_hat: np.ndarray, k: int) -> np.ndarray:
    """r[c, i] = distance from test query i to the K-th nearest point
    within candidate c's own response set R_c. NaN where rc_size < K
    (undefined - can't take a K-th nearest among fewer than K members),
    per the plan's explicit handling of this asymmetry with IF."""
    n_cand = len(selected)
    n_test = test_q_hat.shape[0]
    r = np.full((n_cand, n_test), np.nan)
    for c_idx, cand in enumerate(selected):
        resp_idx = np.flatnonzero(cand.R_c)
        if len(resp_idx) < k:
            continue
        members = points[resp_idx]
        dots = np.clip(test_q_hat @ members.T, -1.0, 1.0)
        dists = np.arccos(dots)  # (n_test, rc_size)
        kth = np.partition(dists, k - 1, axis=1)[:, k - 1]
        r[c_idx] = kth
    return r


def compute_oracle_recall_matrix(selected: list, true_topk_idx: np.ndarray, k: int) -> np.ndarray:
    """recall[c, i] = actual recall of candidate c's response set
    against test query i's true top-K, vectorized over queries per
    candidate."""
    n_cand = len(selected)
    n_test = true_topk_idx.shape[0]
    recall = np.empty((n_cand, n_test))
    for c_idx, cand in enumerate(selected):
        recall[c_idx] = cand.R_c[true_topk_idx].sum(axis=1) / k
    return recall


def compute_if_matrix(selected: list, test_q_hat: np.ndarray, r_bare: np.ndarray, d: int) -> np.ndarray:
    """if_matrix[i, c] via the same per-query call pattern as the
    original score_selected, unchanged - IF computation is not
    reimplemented, only wrapped to return the full matrix instead of
    just the argmax winner."""
    n_test = test_q_hat.shape[0]
    n_cand = len(selected)
    cand_hat = _normalize(np.array([c.query for c in selected]))
    r_cache = np.array([c.r_c for c in selected], dtype=np.float64) * np.pi

    if_matrix = np.empty((n_test, n_cand))
    for i in range(n_test):
        delta = np.arccos(np.clip(cand_hat @ test_q_hat[i], -1.0, 1.0))
        if_matrix[i] = exact_intersection_fraction_batch(r_cache, float(r_bare[i]), delta, d)
    return if_matrix


def compute_recall_hat_matrix(selected: list, test_q: np.ndarray, points: np.ndarray,
                               r_matrix: np.ndarray, tables: list, m: int, x: int, L: int) -> np.ndarray:
    """recall_hat[i, c] = K / N_hat(query i, candidate c's own r).
    Retrieval happens once per query (query_retrieval is radius-
    independent - it only depends on the query's code and the tables);
    only the downstream filter+weight in estimate_count varies per
    candidate, reusing that one retrieved set."""
    n_test = test_q.shape[0]
    n_cand = len(selected)
    retrieved = query_retrieval(tables, test_q, m, x)  # retrieved[i] = point ids for query i

    recall_hat = np.full((n_test, n_cand), np.nan)
    for i in range(n_test):
        ids_i = retrieved[i]
        for c_idx in range(n_cand):
            r_ic = r_matrix[c_idx, i]
            if np.isnan(r_ic):
                continue
            n_hat, _ = estimate_count(test_q[i], ids_i, points, r_ic, m, x, L)
            recall_hat[i, c_idx] = K / n_hat if n_hat > 0 else np.inf
    return recall_hat

#-------------------------------------------------------------------------------


def calibrate_hyperparameters(points: np.ndarray, r_samples: np.ndarray, q_samples: np.ndarray,
                               true_counts: np.ndarray, error_target: float, rng: np.random.Generator) -> dict:
    """Grid search over the existing candidate values (not E3's saved
    numbers - this r distribution is different, measured directly),
    picking the cheapest (m*L) config meeting error_target on this
    real sample."""
    best = None
    for m in M_VALUES:
        for x in X_VALUES:
            if x >= m:
                continue
            for L in L_VALUES:
                tables = build_tables(points, m, L, rng)
                retrieved = query_retrieval(tables, q_samples, m, x)
                n_hat = np.array([estimate_count(q_samples[i], retrieved[i], points, float(r_samples[i]), m, x, L)[0]
                                   for i in range(len(q_samples))])
                valid = true_counts > 0
                rel_err = np.mean(np.abs(n_hat[valid] - true_counts[valid]) / true_counts[valid])
                cost = m * L
                if rel_err <= error_target and (best is None or cost < best['cost']):
                    best = {'m': m, 'x': x, 'L': L, 'rel_err': float(rel_err), 'cost': cost}
    return best


def brute_force_counts(points: np.ndarray, queries: np.ndarray, radii: np.ndarray) -> np.ndarray:
    return np.array([
        int((np.arccos(np.clip(points @ queries[i], -1.0, 1.0)) <= radii[i] + 1e-9).sum())
        for i in range(len(queries))
    ])

#-------------------------------------------------------------------------------


def run_budget(budget_fraction: int, candidates: list, inputs: dict, r_bare: np.ndarray,
                true_topk_idx: np.ndarray, tables: list, m: int, x: int, L: int, n_base: int,
                seed: int) -> dict:
    budget_B = max(1, int(budget_fraction * n_base))
    selected, _ = select_candidates(candidates, budget_B, 'greedy_cost',
                                     n_train_queries=len(inputs['cache_q']),
                                     n_test_queries=len(inputs['test_q']), seed=seed)
    if not selected:
        return None

    test_q_hat = _normalize(inputs['test_q'])
    points = inputs['points_hat']  # already normalized once in main()

    r_matrix = compute_candidate_r(selected, points, test_q_hat, K)
    oracle_recall_matrix = compute_oracle_recall_matrix(selected, true_topk_idx, K)
    if_matrix = compute_if_matrix(selected, test_q_hat, r_bare, D)
    recall_hat_matrix = compute_recall_hat_matrix(selected, test_q_hat, points, r_matrix, tables, m, x, L)

    n_test = test_q_hat.shape[0]
    argmax_if = if_matrix.argmax(axis=1)
    # recall_hat argmax excludes undefined (nan) entries - treat as -inf
    rh_for_argmax = np.where(np.isnan(recall_hat_matrix), -np.inf, recall_hat_matrix)
    argmax_rh = rh_for_argmax.argmax(axis=1)
    has_valid_rh = np.any(~np.isnan(recall_hat_matrix), axis=1)

    recall_if_selected = oracle_recall_matrix[argmax_if, np.arange(n_test)]
    recall_rh_selected = np.where(has_valid_rh, oracle_recall_matrix[argmax_rh, np.arange(n_test)], np.nan)
    oracle_best = oracle_recall_matrix.max(axis=0)

    recall_hat_clipped = np.clip(recall_hat_matrix, 0.0, 1.0)
    rh_clipped_for_argmax = np.where(np.isnan(recall_hat_clipped), -np.inf, recall_hat_clipped)
    argmax_rh_clipped = rh_clipped_for_argmax.argmax(axis=1)
    recall_rh_clipped_selected = np.where(has_valid_rh, oracle_recall_matrix[argmax_rh_clipped, np.arange(n_test)], np.nan)
    clip_changed_selection = (argmax_rh_clipped != argmax_rh) & has_valid_rh

    max_if_per_query = if_matrix.max(axis=1)

    return {
        'n_candidates': len(selected), 'budget_fraction': budget_fraction, 'budget_B': budget_B,
        'recall_if_selected': recall_if_selected, 'recall_rh_selected': recall_rh_selected,
        'recall_rh_clipped_selected': recall_rh_clipped_selected,
        'clip_changed_selection': clip_changed_selection,
        'oracle_best': oracle_best, 'max_if_per_query': max_if_per_query,
        'r_matrix_valid_frac': float(np.mean(~np.isnan(r_matrix))),
        'recall_hat_raw_selected': recall_hat_matrix[argmax_rh, np.arange(n_test)],
    }

#-------------------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).parent.parent.parent.parent

    print("Loading population inputs...")
    inputs = load_population_inputs(repo_root)
    n_base = inputs['base'].shape[0]
    # inputs['base']/['test_q'] are raw (SIFT vectors have norm ~508, not
    # unit) - load_population_inputs doesn't normalize them (each existing
    # consumer, e.g. precompute_test_query_ground_truth, normalizes
    # internally). compute_candidate_r/build_tables/estimate_count all
    # assume unit vectors, so normalize once here rather than per call.
    inputs['points_hat'] = _normalize(inputs['base'])
    print(f"  n_base={n_base}, n_cache={len(inputs['cache_q'])}, n_test={len(inputs['test_q'])}")

    print("Proposing candidates...")
    candidates = propose_once(inputs)
    print(f"  proposed {len(candidates)} candidates")

    print("Precomputing test-query ground truth...")
    r_bare, true_topk_idx = precompute_test_query_ground_truth(inputs['base'], inputs['test_q'], K)

    rng = np.random.default_rng(SEED)

    # --- calibration: realized r distribution from a few representative budgets ---
    print("\nCalibrating: measuring realized r distribution...")
    calib_r_samples = []
    for bf in [0.10, 0.50, 0.99]:
        budget_B = max(1, int(bf * n_base))
        selected, _ = select_candidates(candidates, budget_B, 'greedy_cost',
                                         n_train_queries=len(inputs['cache_q']),
                                         n_test_queries=len(inputs['test_q']), seed=SEED)
        if not selected:
            continue
        test_q_hat = _normalize(inputs['test_q'])
        r_matrix = compute_candidate_r(selected, inputs['points_hat'], test_q_hat, K)
        calib_r_samples.append(r_matrix[~np.isnan(r_matrix)])
    calib_r_samples = np.concatenate(calib_r_samples)
    print(f"  realized r distribution: n={len(calib_r_samples)}, "
          f"percentiles={np.percentile(calib_r_samples, [0,25,50,75,100])}")

    # sample (query, r) pairs for calibration against brute-force ground truth
    n_calib = min(200, len(calib_r_samples))
    calib_idx = rng.choice(len(calib_r_samples), size=n_calib, replace=False)
    calib_r = calib_r_samples[calib_idx]
    calib_q_idx = rng.choice(len(inputs['test_q']), size=n_calib, replace=True)
    calib_q = _normalize(inputs['test_q'])[calib_q_idx]
    calib_true_counts = brute_force_counts(inputs['points_hat'], calib_q, calib_r)

    print("Calibrating hyperparameters for <=15% error target...")
    t0 = time.perf_counter()
    config_15 = calibrate_hyperparameters(inputs['points_hat'], calib_r, calib_q, calib_true_counts, 0.15, rng)
    print(f"  config_15={config_15} ({time.perf_counter()-t0:.1f}s)")

    print("Calibrating hyperparameters for <=25% error target...")
    t0 = time.perf_counter()
    config_25 = calibrate_hyperparameters(inputs['points_hat'], calib_r, calib_q, calib_true_counts, 0.25, rng)
    print(f"  config_25={config_25} ({time.perf_counter()-t0:.1f}s)")

    with open(OUT_DIR / 'calibration.json', 'w') as f:
        json.dump({'r_percentiles': np.percentile(calib_r_samples, [0,25,50,75,100]).tolist(),
                   'config_15': config_15, 'config_25': config_25}, f, indent=2)

    if config_15 is None:
        print("No configuration met the 15% error target - stopping, see plan for fallback.")
        return

    # --- pilot: largest-pool budget alone, full query set, primary config ---
    print("\nPilot: budget=0.99 (largest pool), primary config...")
    tables = build_tables(inputs['points_hat'], config_15['m'], config_15['L'], rng)
    t0 = time.perf_counter()
    result = run_budget(0.99, candidates, inputs, r_bare, true_topk_idx, tables,
                         config_15['m'], config_15['x'], config_15['L'], n_base, SEED)
    t_pilot = time.perf_counter() - t0
    print(f"  pilot: n_candidates={result['n_candidates']}, wall_clock={t_pilot:.1f}s")
    est_total = t_pilot * len(BUDGET_FRACTIONS)  # conservative: treats every budget as worst-case
    print(f"  extrapolated worst-case full sweep (11 budgets): {est_total:.1f}s")
    if est_total > 3600 * 3:
        print("  WARNING: extrapolated time exceeds ~3h - stopping for review before full run.")
        return

    progress_path = OUT_DIR / 'progress.json'
    total_steps = len(BUDGET_FRACTIONS) * (2 if config_25 is not None else 1)
    step = 0
    start_time = time.time()

    def _write_progress(label: str) -> None:
        tmp = progress_path.with_suffix('.tmp')
        tmp.write_text(json.dumps({'done': step, 'total': total_steps, 'current': label,
                                    'elapsed_seconds': time.time() - start_time,
                                    'timestamp': datetime.now().isoformat()}))
        tmp.replace(progress_path)

    # --- full run, primary config ---
    print("\nRunning full 11-budget sweep, primary config (<=15% target)...")
    results_primary = []
    for bf in BUDGET_FRACTIONS:
        r = run_budget(bf, candidates, inputs, r_bare, true_topk_idx, tables,
                        config_15['m'], config_15['x'], config_15['L'], n_base, SEED)
        step += 1
        _write_progress(f'primary budget={bf}')
        if r is not None:
            results_primary.append(r)
            print(f"  budget={bf}: n_candidates={r['n_candidates']}, "
                  f"mean_recall_if={np.nanmean(r['recall_if_selected']):.4f}, "
                  f"mean_recall_rh={np.nanmean(r['recall_rh_selected']):.4f}")

    print("\nRunning full 11-budget sweep, secondary config (<=25% target, H4)...")
    results_secondary = []
    if config_25 is not None:
        tables_25 = build_tables(inputs['points_hat'], config_25['m'], config_25['L'], rng)
        for bf in BUDGET_FRACTIONS:
            r = run_budget(bf, candidates, inputs, r_bare, true_topk_idx, tables_25,
                            config_25['m'], config_25['x'], config_25['L'], n_base, SEED)
            step += 1
            _write_progress(f'secondary budget={bf}')
            if r is not None:
                results_secondary.append(r)

    def _save(results, label):
        for r in results:
            path = OUT_DIR / f"{label}_budget{r['budget_fraction']}.npz"
            np.savez_compressed(path, **{k: v for k, v in r.items() if isinstance(v, np.ndarray)},
                                 n_candidates=r['n_candidates'], budget_fraction=r['budget_fraction'],
                                 r_matrix_valid_frac=r['r_matrix_valid_frac'])

    _save(results_primary, 'primary')
    _save(results_secondary, 'secondary')

    print("\nDone.")


if __name__ == '__main__':
    main()
