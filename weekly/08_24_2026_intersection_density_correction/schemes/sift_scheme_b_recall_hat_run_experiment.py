#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Scheme B (sift_synth_sweep_run_experiment.py) rerun with three
selection predictors computed together in one pass (same candidate pool/
queries/corpus for all three, by construction):

- IF, query-side radius now r_hat_q* (estimated) instead of the
  historical brute-force ground-truth radius - see plan notes on the
  confound this retires.
- per_cache_query_recall_hat = K / N_hat(r_ic), r_ic from the candidate's
  own farthest response member's distance to the query - the original
  estimator, now known to carry a structural order-statistic bias.
- per_test_query_recall_hat: r_hat_q* is the query's own estimated true
  K-NN radius (shared across every candidate for that query, independent
  of any candidate), and the predictor is directly the fraction of each
  candidate's K response members within r_hat_q*.

Selection is an argmax over the FULL pooled candidate population
(thousands of candidates, not a small per-anchor swarm), independently
per predictor - see evaluate_test_queries_recall_hat for the vectorized
gather pattern this requires; a naive per-candidate Python loop measured
~150 hours at this scale, this approach ~1 hour."""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'common'))
sys.path.append(str(Path(__file__).parent.parent / 'lsh_ht'))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import numpy as np
from tqdm import tqdm

from sift_synth_sweep_run_experiment import (
    D, K_VALUES, K_MAX, N_ANCHORS, N_DIRECTIONS, BACKGROUND_PATH, UNIFORM_CORPUS_SEED, LOG_IF_TARGETS,
    _normalize, _random_tangent, _move_along_geodesic, _anchor_delta_grid,
    build_candidate_pool, precompute_candidates, _write_progress,
)
from part2_uniform_baseline_run_experiment import _sample_uniform_sphere, _bruteforce_knn
from lsh_horvitz_thompson import build_tables, query_retrieval, estimate_count, p_union
from lsh_ht_grid_run_experiment import M_VALUES, X_VALUES, L_VALUES
from primitives import exact_intersection_fraction_batch

#-------------------------------------------------------------------------------

ERROR_TARGET = 0.15
CALIB_N_ANCHORS = 3
CALIB_N_DELTAS_PER_ANCHOR = 3
CALIB_N_DIRECTIONS = 10

OUT_DIR = Path(__file__).parent.parent / 'results'
PART_DIR = 'sift_synth_sweep_recall_hat'
PART_DIR_UNIFORM = 'sift_synth_sweep_on_uniform_recall_hat'
_RUN_TIMESTAMP = None


def _run_dir() -> Path:
    global _RUN_TIMESTAMP
    if _RUN_TIMESTAMP is None:
        _RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
    return OUT_DIR / _RUN_TIMESTAMP

#-------------------------------------------------------------------------------


def evaluate_test_queries_recall_hat(queries: np.ndarray, background: np.ndarray, candidates: np.ndarray,
                                      cand_data: dict, tables: list, hparams: tuple, home_idx: int) -> dict:
    """One anchor/delta's batch of test queries, argmax over the FULL
    candidate population independently for IF, per_cache_query_recall_hat,
    and per_test_query_recall_hat. Mirrors the original evaluate_test_queries's
    structure/return shape, extended to three predictors sharing one
    retrieval and one corpus-distance vector per query - no per-candidate
    step ever touches D-dimensional coordinates again after those two."""
    m, x, L = hparams
    dists_test, idx_test = _bruteforce_knn(background, queries, K_MAX)
    theta_corpus_all = np.arccos(np.clip(background @ queries.T, -1.0, 1.0))  # (N, n_queries)
    delta_qc_all = np.arccos(np.clip(candidates @ queries.T, -1.0, 1.0))  # (n_cand, n_queries)
    retrieved_all = query_retrieval(tables, queries, m, x)  # list, one retrieved-id array per query

    fields = ('if_fair', 'recall_if', 'is_home_if',
              'per_cache_query_recall_hat', 'recall_pcq', 'is_home_pcq',
              'per_test_query_recall_hat', 'recall_ptq', 'is_home_ptq')
    out = {K: {f: np.empty(len(queries)) if 'is_home' not in f else np.empty(len(queries), dtype=bool)
               for f in fields} for K in K_VALUES}

    # The retrieved set's sorted (theta, cumulative HT weight) construction is
    # independent of K - only the final radius/count lookup differs per K.
    # Building it once per query here (not once per (K, query) pair, which
    # measured ~4x slower in a full-scale pilot) is the whole point of
    # sharing retrieval across K in the first place.
    per_query_cdf = []
    for i in range(len(queries)):
        retrieved_ids = retrieved_all[i]
        if len(retrieved_ids) == 0:
            per_query_cdf.append((np.empty(0), np.empty(0)))
            continue
        theta_retrieved = np.arccos(np.clip(background[retrieved_ids] @ queries[i], -1.0, 1.0))
        weights = 1.0 / np.clip(p_union(theta_retrieved, m, x, L), 1e-12, 1.0)
        order = np.argsort(theta_retrieved)
        theta_sorted = theta_retrieved[order]
        cum_weights = np.cumsum(weights[order])
        per_query_cdf.append((theta_sorted, cum_weights))

    for K in K_VALUES:
        resp_idx = cand_data[K]['idx']  # (n_cand, K) - fixed-width, exactly K members per candidate
        r_cache = cand_data[K]['r_cache']  # (n_cand,)
        for i in range(len(queries)):
            theta_corpus = theta_corpus_all[:, i]
            member_dists = theta_corpus[resp_idx]  # (n_cand, K) - shared by both recall_hat variants below
            true_topk = set(idx_test[i, :K].tolist())
            theta_sorted, cum_weights = per_query_cdf[i]

            def score(best: int) -> float:
                return len(true_topk & set(resp_idx[best].tolist())) / K

            # per_cache_query_recall_hat = K / N_hat(r_ic), r_ic from the candidate's own farthest member
            r_ic_all = member_dists.max(axis=1)
            if len(theta_sorted) == 0:
                n_hat_all = np.zeros(len(r_ic_all))
            else:
                cum_weights_padded = np.concatenate([[0.0], cum_weights])
                n_hat_all = cum_weights_padded[np.searchsorted(theta_sorted, r_ic_all, side='right')]
            pcq_all = np.where(n_hat_all > 0, K / np.clip(n_hat_all, 1e-9, None), np.inf)
            best_pcq = int(pcq_all.argmax())

            # r_hat_q*: the query's own estimated true K-NN radius, independent of any candidate -
            # shared across every candidate for this query, one inversion of the same CDF.
            if len(theta_sorted) == 0:
                r_hat_q_star = float('inf')
            else:
                idx = np.searchsorted(cum_weights, K, side='left')
                r_hat_q_star = float(theta_sorted[idx]) if idx < len(theta_sorted) else float('inf')

            # IF, query-side radius now r_hat_q* instead of the historical brute-force ground truth
            if_all = exact_intersection_fraction_batch(r_cache, r_hat_q_star, delta_qc_all[:, i], D)
            best_if = int(if_all.argmax())

            # per_test_query_recall_hat: fraction of each candidate's K members within r_hat_q*
            ptq_all = (member_dists <= r_hat_q_star).mean(axis=1)
            best_ptq = int(ptq_all.argmax())

            dst = out[K]
            dst['if_fair'][i] = if_all[best_if]
            dst['recall_if'][i] = score(best_if)
            dst['is_home_if'][i] = (best_if == home_idx)
            dst['per_cache_query_recall_hat'][i] = pcq_all[best_pcq]
            dst['recall_pcq'][i] = score(best_pcq)
            dst['is_home_pcq'][i] = (best_pcq == home_idx)
            dst['per_test_query_recall_hat'][i] = ptq_all[best_ptq]
            dst['recall_ptq'][i] = score(best_ptq)
            dst['is_home_ptq'][i] = (best_ptq == home_idx)
    return out

#-------------------------------------------------------------------------------


def collect_calibration_samples(background: np.ndarray, cand_data: dict, anchors: np.ndarray,
                                 rng: np.random.Generator) -> dict:
    """Realistic (query, r_ic, true_count) triples pooled across K, using
    actual production candidates/geometry (not synthetic) for a handful
    of anchors - kept small and decoupled from the main run's
    N_ANCHORS/N_DIRECTIONS so calibration cost stays bounded regardless
    of full-run scale. True counts via a sorted search over theta_corpus
    rather than an (n_candidates, N) broadcast comparison - the latter
    would materialize hundreds of MB per query for no reason, the same
    class of mistake the r_ic gather itself is designed to avoid."""
    CANDS_PER_QUERY = 5  # a few candidates' r_ic per query is enough signal; keeps the
                          # calibration grid search's own per-sample Python loop cheap
    q_list, r_list, true_count_list = [], [], []
    for anchor in anchors[:CALIB_N_ANCHORS]:
        deltas = rng.uniform(1e-3, 1.5, size=CALIB_N_DELTAS_PER_ANCHOR)
        for delta in deltas:
            directions = np.array([_random_tangent(anchor, rng, D) for _ in range(CALIB_N_DIRECTIONS)])
            queries = _move_along_geodesic(anchor, np.full(CALIB_N_DIRECTIONS, delta), directions)
            theta_corpus_all = np.arccos(np.clip(background @ queries.T, -1.0, 1.0))
            for K in K_VALUES:
                resp_idx = cand_data[K]['idx']
                cand_sample = rng.choice(len(resp_idx), size=CANDS_PER_QUERY, replace=False)
                for i in range(len(queries)):
                    theta_corpus = theta_corpus_all[:, i]
                    theta_sorted = np.sort(theta_corpus)
                    r_ic_sample = theta_corpus[resp_idx[cand_sample]].max(axis=1)  # (CANDS_PER_QUERY,)
                    true_count_sample = np.searchsorted(theta_sorted, r_ic_sample, side='right')
                    q_list.append(np.tile(queries[i], (CANDS_PER_QUERY, 1)))
                    r_list.append(r_ic_sample)
                    true_count_list.append(true_count_sample)

    q_arr = np.concatenate(q_list, axis=0)
    r_arr = np.concatenate(r_list)
    true_count_arr = np.concatenate(true_count_list)
    return {'q': q_arr, 'r': r_arr, 'true_count': true_count_arr}


def calibrate_hyperparameters(points: np.ndarray, q_samples: np.ndarray, r_samples: np.ndarray,
                               true_counts: np.ndarray, error_target: float, rng: np.random.Generator) -> dict:
    """First (m,x,L) in ascending-cost order meeting error_target - see
    sift_scheme_a_recall_hat_run_experiment.py's identical rationale for
    ascending-cost early exit over an exhaustive grid scan."""
    valid = true_counts > 0
    configs = sorted(
        ((m, x, L) for m in M_VALUES for x in X_VALUES for L in L_VALUES if x < m),
        key=lambda mxl: mxl[0] * mxl[2],
    )
    for m, x, L in configs:
        tables = build_tables(points, m, L, rng)
        retrieved = query_retrieval(tables, q_samples, m, x)
        n_hat = np.array([estimate_count(q_samples[i], retrieved[i], points, float(r_samples[i]), m, x, L)[0]
                           for i in range(len(q_samples))])
        rel_err = np.mean(np.abs(n_hat[valid] - true_counts[valid]) / true_counts[valid])
        if rel_err <= error_target:
            return {'m': m, 'x': x, 'L': L, 'rel_err': float(rel_err), 'cost': m * L}
    return None

#-------------------------------------------------------------------------------


def save_results(pooled: dict, run_dir: Path, part_dir: str) -> None:
    dim_dir = run_dir / part_dir / f'{D}d' / 'raw'
    dim_dir.mkdir(parents=True, exist_ok=True)
    for K, arrays in pooled.items():
        path = dim_dir / f'K{K}.npz'
        np.savez_compressed(path, d=D, K=K, **arrays)
        print(f"    wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-anchors', type=int, default=N_ANCHORS, dest='n_anchors')
    parser.add_argument('--n-directions', type=int, default=N_DIRECTIONS, dest='n_directions')
    parser.add_argument('--background-path', type=str, default=BACKGROUND_PATH, dest='background_path')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--error-target', type=float, default=ERROR_TARGET, dest='error_target')
    parser.add_argument('--run-dir', type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else _run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")
    print(f"K values: {K_VALUES}, n_anchors={args.n_anchors}, n_directions={args.n_directions}\n")

    repo_root = Path(__file__).parent.parent.parent.parent
    bg_path = Path(args.background_path)
    if not bg_path.is_absolute():
        bg_path = repo_root / bg_path
    background = _normalize(np.load(bg_path).astype(np.float64))

    rng = np.random.default_rng(args.seed)
    anchor_idx = rng.choice(len(background), size=args.n_anchors, replace=False)
    anchors = background[anchor_idx]

    print("Building candidate pool...")
    candidates, home_indices = build_candidate_pool(background, anchors, rng, D)
    print(f"  candidate pool size: {len(candidates)} ({len(candidates) / args.n_anchors:.2f}/anchor avg)")

    print("Precomputing candidate response sets (real corpus)...")
    cand_data_real = precompute_candidates(background, candidates)

    uniform_rng = np.random.default_rng(UNIFORM_CORPUS_SEED)
    uniform_corpus = _normalize(uniform_rng.standard_normal((len(background), D)))
    print("Precomputing candidate response sets (uniform corpus)...")
    cand_data_uniform = precompute_candidates(uniform_corpus, candidates)

    print("\nCalibrating hyperparameters (real corpus)...")
    calib_rng = np.random.default_rng(args.seed + 100)
    calib_real = collect_calibration_samples(background, cand_data_real, anchors, calib_rng)
    print(f"  real: r_ic percentiles={np.percentile(calib_real['r'], [0, 25, 50, 75, 100])}, "
          f"n_samples={len(calib_real['r'])}")
    t0 = time.time()
    config_real = calibrate_hyperparameters(background, calib_real['q'], calib_real['r'],
                                             calib_real['true_count'], args.error_target, calib_rng)
    print(f"  real config: {config_real} ({time.time() - t0:.1f}s)")

    print("\nCalibrating hyperparameters (uniform corpus)...")
    calib_uniform = collect_calibration_samples(uniform_corpus, cand_data_uniform, anchors, calib_rng)
    print(f"  uniform: r_ic percentiles={np.percentile(calib_uniform['r'], [0, 25, 50, 75, 100])}, "
          f"n_samples={len(calib_uniform['r'])}")
    t0 = time.time()
    config_uniform = calibrate_hyperparameters(uniform_corpus, calib_uniform['q'], calib_uniform['r'],
                                                calib_uniform['true_count'], args.error_target, calib_rng)
    print(f"  uniform config: {config_uniform} ({time.time() - t0:.1f}s)")

    calib_out = {'error_target': args.error_target, 'real': config_real, 'uniform': config_uniform}
    (run_dir / 'calibration.json').write_text(json.dumps(calib_out, indent=2))

    table_rng = np.random.default_rng(args.seed + 1)
    print("\nBuilding LSH tables...")
    tables_real = build_tables(background, config_real['m'], config_real['L'], table_rng)
    tables_uniform = build_tables(uniform_corpus, config_uniform['m'], config_uniform['L'], table_rng)
    hparams_real = (config_real['m'], config_real['x'], config_real['L'])
    hparams_uniform = (config_uniform['m'], config_uniform['x'], config_uniform['L'])

    fields = ('anchor_id', 'delta', 'if_fair', 'recall_if', 'is_home_if',
              'per_cache_query_recall_hat', 'recall_pcq', 'is_home_pcq',
              'per_test_query_recall_hat', 'recall_ptq', 'is_home_ptq')
    pooled_real = {K: {f: [] for f in fields} for K in K_VALUES}
    pooled_uniform = {K: {f: [] for f in fields} for K in K_VALUES}

    progress_path = run_dir / 'progress.json'
    start_time = time.time()
    total_deltas = 0
    for a_id in tqdm(range(args.n_anchors), desc='anchors', mininterval=1.0):
        anchor = anchors[a_id]
        home_idx = int(home_indices[a_id])
        r_cache_by_K = {K: float(cand_data_real[K]['r_cache'][home_idx]) for K in K_VALUES}
        deltas = _anchor_delta_grid(r_cache_by_K, D, LOG_IF_TARGETS)
        total_deltas += len(deltas)

        for delta in deltas:
            directions = np.array([_random_tangent(anchor, rng, D) for _ in range(args.n_directions)])
            queries = _move_along_geodesic(anchor, np.full(args.n_directions, delta), directions)

            results_real = evaluate_test_queries_recall_hat(queries, background, candidates, cand_data_real,
                                                              tables_real, hparams_real, home_idx)
            results_uniform = evaluate_test_queries_recall_hat(queries, uniform_corpus, candidates, cand_data_uniform,
                                                                 tables_uniform, hparams_uniform, home_idx)
            for K in K_VALUES:
                for results, pooled in ((results_real, pooled_real), (results_uniform, pooled_uniform)):
                    pooled[K]['anchor_id'].append(np.full(args.n_directions, a_id, dtype=np.int32))
                    pooled[K]['delta'].append(np.full(args.n_directions, delta, dtype=np.float64))
                    for f in fields[2:]:
                        pooled[K][f].append(results[K][f])

        _write_progress(progress_path, a_id, args.n_anchors, start_time)
        if a_id == 0:
            elapsed = time.time() - start_time
            print(f"  anchor 0 took {elapsed:.1f}s, projected total ~{elapsed * args.n_anchors / 60:.1f} min")

    for pooled in (pooled_real, pooled_uniform):
        for K in K_VALUES:
            for f in fields:
                pooled[K][f] = np.concatenate(pooled[K][f])

    save_results(pooled_real, run_dir, PART_DIR)
    save_results(pooled_uniform, run_dir, PART_DIR_UNIFORM)

    print(f"\nDone: {run_dir}")


if __name__ == '__main__':
    main()
