#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Raw (non-argmax) predicted-vs-actual data for per_cache_query_recall_hat
and per_test_query_recall_hat: every (query, candidate) pair in a sample,
not just the argmax winner per query. Complements the main Scheme B run
(sift_scheme_b_recall_hat_run_experiment.py), which only records the
winner and therefore can't show the raw predictor-vs-actual trend the
same way Scheme A's pointwise (no-selection) design already does.

Reuses the exact same candidate pool (same seed) and the already-saved
calibration from the full three-way run, so this is directly comparable
- just a different slice of the same setup (a query sample, all
candidates per query, instead of all queries, argmax candidate only)."""

import sys
import json
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'common'))
sys.path.append(str(Path(__file__).parent.parent / 'lsh_ht'))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import numpy as np
from tqdm import tqdm

from sift_synth_sweep_run_experiment import (
    D, K_VALUES, K_MAX, N_ANCHORS, BACKGROUND_PATH, UNIFORM_CORPUS_SEED, LOG_IF_TARGETS,
    _normalize, _random_tangent, _move_along_geodesic, _anchor_delta_grid,
    build_candidate_pool, precompute_candidates,
)
from part2_uniform_baseline_run_experiment import _bruteforce_knn
from lsh_horvitz_thompson import build_tables, query_retrieval, p_union

#-------------------------------------------------------------------------------

N_SAMPLE_ANCHORS = 8
N_DIRECTIONS = 10
N_CANDS_PER_QUERY = 300  # random subsample of the 8000-candidate pool, per query

OUT_DIR = Path(__file__).parent.parent / 'results' / 'scheme_b_raw_scatter'

#-------------------------------------------------------------------------------


def collect_raw(background: np.ndarray, candidates: np.ndarray, cand_data: dict, tables: list,
                 hparams: tuple, anchors: np.ndarray, home_indices: np.ndarray, rng: np.random.Generator) -> dict:
    m, x, L = hparams
    n_cand = len(candidates)

    out = {K: {'pred_pcq': [], 'pred_ptq': [], 'actual': []} for K in K_VALUES}

    for a_id in tqdm(range(N_SAMPLE_ANCHORS), desc='sample anchors'):
        anchor = anchors[a_id]
        home_idx = int(home_indices[a_id])
        r_cache_by_K = {K: float(cand_data[K]['r_cache'][home_idx]) for K in K_VALUES}
        deltas = _anchor_delta_grid(r_cache_by_K, D, LOG_IF_TARGETS)

        for delta in deltas:
            directions = np.array([_random_tangent(anchor, rng, D) for _ in range(N_DIRECTIONS)])
            queries = _move_along_geodesic(anchor, np.full(N_DIRECTIONS, delta), directions)

            dists_test, idx_test = _bruteforce_knn(background, queries, K_MAX)
            theta_corpus_all = np.arccos(np.clip(background @ queries.T, -1.0, 1.0))
            retrieved_all = query_retrieval(tables, queries, m, x)

            cand_sample = rng.choice(n_cand, size=N_CANDS_PER_QUERY, replace=False)

            for i in range(N_DIRECTIONS):
                theta_corpus = theta_corpus_all[:, i]
                retrieved_ids = retrieved_all[i]
                if len(retrieved_ids) == 0:
                    continue
                theta_retrieved = np.arccos(np.clip(background[retrieved_ids] @ queries[i], -1.0, 1.0))
                weights = 1.0 / np.clip(p_union(theta_retrieved, m, x, L), 1e-12, 1.0)
                order = np.argsort(theta_retrieved)
                theta_sorted = theta_retrieved[order]
                cum_weights = np.cumsum(weights[order])
                cum_weights_padded = np.concatenate([[0.0], cum_weights])

                for K in K_VALUES:
                    resp_idx = cand_data[K]['idx'][cand_sample]  # (N_CANDS_PER_QUERY, K)
                    member_dists = theta_corpus[resp_idx]

                    r_ic_all = member_dists.max(axis=1)
                    n_hat_all = cum_weights_padded[np.searchsorted(theta_sorted, r_ic_all, side='right')]
                    pcq_all = np.where(n_hat_all > 0, K / np.clip(n_hat_all, 1e-9, None), np.inf)

                    idx = np.searchsorted(cum_weights, K, side='left')
                    r_hat_q_star = float(theta_sorted[idx]) if idx < len(theta_sorted) else float('inf')
                    ptq_all = (member_dists <= r_hat_q_star).mean(axis=1)

                    true_topk = set(idx_test[i, :K].tolist())
                    actual_all = np.array([
                        len(true_topk & set(resp_idx[c].tolist())) / K for c in range(N_CANDS_PER_QUERY)
                    ])

                    out[K]['pred_pcq'].append(pcq_all)
                    out[K]['pred_ptq'].append(ptq_all)
                    out[K]['actual'].append(actual_all)

    for K in K_VALUES:
        for f in ('pred_pcq', 'pred_ptq', 'actual'):
            out[K][f] = np.concatenate(out[K][f])
    return out

#-------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--calibration-path', type=str,
                         default=str(Path(__file__).parent.parent / 'results' / 'scheme_b_three_way' / 'calibration.json'))
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    calib = json.loads(Path(args.calibration_path).read_text())

    repo_root = Path(__file__).parent.parent.parent.parent
    background = _normalize(np.load(repo_root / BACKGROUND_PATH).astype(np.float64))

    rng = np.random.default_rng(args.seed)
    anchor_idx = rng.choice(len(background), size=N_ANCHORS, replace=False)
    anchors = background[anchor_idx]

    print("Building candidate pool (same seed as the main three-way run)...")
    candidates, home_indices = build_candidate_pool(background, anchors, rng, D)
    print(f"  candidate pool size: {len(candidates)}")

    print("Precomputing candidate response sets (real corpus)...")
    cand_data_real = precompute_candidates(background, candidates)
    uniform_rng = np.random.default_rng(UNIFORM_CORPUS_SEED)
    uniform_corpus = _normalize(uniform_rng.standard_normal((len(background), D)))
    print("Precomputing candidate response sets (uniform corpus)...")
    cand_data_uniform = precompute_candidates(uniform_corpus, candidates)

    table_rng = np.random.default_rng(args.seed + 1)
    tables_real = build_tables(background, calib['real']['m'], calib['real']['L'], table_rng)
    tables_uniform = build_tables(uniform_corpus, calib['uniform']['m'], calib['uniform']['L'], table_rng)
    hparams_real = (calib['real']['m'], calib['real']['x'], calib['real']['L'])
    hparams_uniform = (calib['uniform']['m'], calib['uniform']['x'], calib['uniform']['L'])

    sample_rng = np.random.default_rng(args.seed + 2)
    print("\nCollecting raw (query, candidate) pairs (real corpus)...")
    raw_real = collect_raw(background, candidates, cand_data_real, tables_real, hparams_real,
                            anchors, home_indices, sample_rng)
    print("Collecting raw (query, candidate) pairs (uniform corpus)...")
    raw_uniform = collect_raw(uniform_corpus, candidates, cand_data_uniform, tables_uniform, hparams_uniform,
                               anchors, home_indices, sample_rng)

    for K in K_VALUES:
        np.savez_compressed(OUT_DIR / f'real_K{K}.npz', **raw_real[K])
        np.savez_compressed(OUT_DIR / f'uniform_K{K}.npz', **raw_uniform[K])
        print(f"  K={K}: {len(raw_real[K]['actual'])} real pairs, {len(raw_uniform[K]['actual'])} uniform pairs")

    print(f"\nDone: {OUT_DIR}")


if __name__ == '__main__':
    main()
