#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Scheme A (sift_part2_real_data_run_experiment.py) rerun with three
recall predictors computed together in one pass, guaranteeing they share
the same candidate pool/queries/corpus by construction:

- IF, with its query-side radius now r_hat_q* (estimated, see below) -
  historically this used the brute-force ground-truth K-NN radius, a
  confound diagnosed and retired this session (see plan notes).
- per_cache_query_recall_hat = K / N_hat(r_ic), r_ic from the candidate's
  own farthest response member's distance to the query - the original
  estimator, now known to carry a structural order-statistic bias.
- per_test_query_recall_hat, the fix: r_hat_q* is the query's own
  estimated true K-NN radius (via inverting the HT-LSH count estimator),
  and the predictor is directly the fraction of the candidate's K
  response members within r_hat_q*.

Regenerates the identical anchors/test poles (same seeds) as the
already-saved historical IF-vs-recall data, so per_cache_query_recall_hat
rows correspond to that data one-to-one; the recall sanity check confirms
this."""

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

from part2_uniform_baseline_run_experiment import (
    _sample_uniform_sphere, _bruteforce_knn, _construct_test_poles, _chord_to_angle,
    K_VALUES, K_MAX, N_BACKGROUNDS, N_BACKGROUND_POINTS, N_THETA, M_PER_THETA, GRID_MARGIN,
)
from sift_part2_real_data_run_experiment import (
    D, BACKGROUND_PATH, UNIFORM_CORPUS_SEED, _normalize, _real_anchor, _householder_e1_to,
)
from density_correction_common import covered_mask
from lsh_horvitz_thompson import build_tables, query_retrieval, estimate_count, estimate_knn_radius
from lsh_ht_grid_run_experiment import M_VALUES, X_VALUES, L_VALUES
from primitives import exact_intersection_fraction_batch_theta2

#-------------------------------------------------------------------------------

ERROR_TARGET = 0.15
CALIB_BACKGROUND_ID = 0
# Calibration cost scales with sample count independent of the main run's
# n_theta/m_per_theta - kept small and decoupled so a full-scale main run
# doesn't blow up calibration time.
CALIB_N_THETA = 6
CALIB_M_PER_THETA = 10

OUT_DIR = Path(__file__).parent.parent / 'results'
PART_DIR = 'sift_part2_real_data_recall_hat'
PART_DIR_UNIFORM = 'sift_part2_real_anchor_on_uniform_recall_hat'
_RUN_TIMESTAMP = None


def _run_dir() -> Path:
    global _RUN_TIMESTAMP
    if _RUN_TIMESTAMP is None:
        _RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
    return OUT_DIR / _RUN_TIMESTAMP

#-------------------------------------------------------------------------------


def _delta_grid_and_anchor(background_id: int, real_background: np.ndarray, n_points: int,
                            n_theta: int, seed: int) -> tuple:
    """Reproduces the exact rng stream sift_part2_real_data_run_experiment.py
    uses: consumes `rng` for the synthetic delta-grid sizing, then leaves it
    positioned for the per-delta _construct_test_poles calls below."""
    rng = np.random.default_rng(seed)
    synthetic_background = _sample_uniform_sphere(n_points, D, rng)
    e1 = np.zeros(D)
    e1[0] = 1.0
    dists_cache, _ = _bruteforce_knn(synthetic_background, e1[None, :], K_MAX)
    r_cache_synthetic = float(_chord_to_angle(dists_cache[0, K_MAX - 1]))
    grid_max = min(GRID_MARGIN * 2.0 * r_cache_synthetic, np.pi)
    delta_grid = np.linspace(0.0, grid_max, n_theta)
    real_anchor = _real_anchor(real_background, background_id, D)
    return rng, delta_grid, real_anchor


def _anchor_response_sets(corpus: np.ndarray, anchor: np.ndarray) -> tuple:
    """Anchor's own true top-K_MAX neighbor indices/radius, prefix-sliced
    per K (R_c = top-K neighbors, |R_c| = K exactly for every candidate in
    this scheme)."""
    dists, idx = _bruteforce_knn(corpus, anchor[None, :], K_MAX)
    r_cache_by_K = {K: float(_chord_to_angle(dists[0, K - 1])) for K in K_VALUES}
    idx_by_K = {K: idx[0, :K] for K in K_VALUES}
    return r_cache_by_K, idx_by_K


def score_background_recall_hat(background_id: int, real_background: np.ndarray, uniform_corpus: np.ndarray,
                                 tables_real: list, tables_uniform: list, hparams_real: tuple, hparams_uniform: tuple,
                                 n_points: int, n_theta: int, m_per_theta: int, base_seed: int) -> tuple:
    """One background's full delta grid, scored against both corpora.
    Returns (out_real, out_uniform), each {K: {field: array}}. Matches
    sift_part2_real_data_run_experiment.py's run_dimension_real, which
    seeds each background as base_seed + background_id."""
    rng, delta_grid, real_anchor = _delta_grid_and_anchor(background_id, real_background, n_points, n_theta,
                                                            base_seed + background_id)
    H = _householder_e1_to(real_anchor)

    r_cache_real, idx_real = _anchor_response_sets(real_background, real_anchor)
    r_cache_uniform, idx_uniform = _anchor_response_sets(uniform_corpus, real_anchor)

    fields = ('background_id', 'delta', 'r_cache', 'r_ic', 'per_cache_query_n_hat', 'per_cache_query_recall_hat',
               'r_hat_q_star', 'if_fair', 'per_test_query_recall_hat', 'recall')
    out_real = {K: {f: [] for f in fields} for K in K_VALUES}
    out_uniform = {K: {f: [] for f in fields} for K in K_VALUES}

    for delta in delta_grid:
        delta = float(delta)
        test_poles = _construct_test_poles(delta, m_per_theta, D, rng)  # same call/rng state as the baseline
        relocated = _normalize(test_poles @ H)

        for corpus, idx_by_K, r_cache_by_K, tables, hparams, out in (
            (real_background, idx_real, r_cache_real, tables_real, hparams_real, out_real),
            (uniform_corpus, idx_uniform, r_cache_uniform, tables_uniform, hparams_uniform, out_uniform),
        ):
            m, x, L = hparams
            dists_test, idx_test = _bruteforce_knn(corpus, relocated, K_MAX)
            retrieved = query_retrieval(tables, relocated, m, x)  # one retrieval per query, shared below

            for K in K_VALUES:
                r_cache = r_cache_by_K[K]
                resp_coords = corpus[idx_by_K[K]]  # (K, d)
                member_dists = np.arccos(np.clip(relocated @ resp_coords.T, -1.0, 1.0))  # (m_per_theta, K)
                r_ic = member_dists.max(axis=1)  # (m_per_theta,)

                true_topk_coords = corpus[idx_test[:, :K]]
                covered = covered_mask(true_topk_coords, real_anchor, r_cache)
                recall = covered.mean(axis=1)

                r_hat_q_star = np.array([
                    estimate_knn_radius(relocated[i], retrieved[i], corpus, K, m, x, L)
                    for i in range(m_per_theta)
                ])

                if_fair = exact_intersection_fraction_batch_theta2(r_cache, r_hat_q_star, delta, D)
                per_test_query_recall_hat = (member_dists <= r_hat_q_star[:, None]).mean(axis=1)

                per_cache_query_n_hat = np.empty(m_per_theta)
                per_cache_query_recall_hat = np.empty(m_per_theta)
                for i in range(m_per_theta):
                    n_hat_i, _ = estimate_count(relocated[i], retrieved[i], corpus, float(r_ic[i]), m, x, L)
                    per_cache_query_n_hat[i] = n_hat_i
                    per_cache_query_recall_hat[i] = K / n_hat_i if n_hat_i > 0 else np.inf

                dst = out[K]
                dst['background_id'].append(np.full(m_per_theta, background_id, dtype=np.int32))
                dst['delta'].append(np.full(m_per_theta, delta, dtype=np.float64))
                dst['r_cache'].append(np.full(m_per_theta, r_cache, dtype=np.float64))
                dst['r_ic'].append(r_ic)
                dst['per_cache_query_n_hat'].append(per_cache_query_n_hat)
                dst['per_cache_query_recall_hat'].append(per_cache_query_recall_hat)
                dst['r_hat_q_star'].append(r_hat_q_star)
                dst['if_fair'].append(if_fair)
                dst['per_test_query_recall_hat'].append(per_test_query_recall_hat)
                dst['recall'].append(recall)

    for out in (out_real, out_uniform):
        for K in K_VALUES:
            for f in fields:
                out[K][f] = np.concatenate(out[K][f])
    return out_real, out_uniform

#-------------------------------------------------------------------------------


def collect_calibration_samples(real_background: np.ndarray, uniform_corpus: np.ndarray,
                                 n_points: int, seed: int) -> dict:
    """Pools (query, r_ic, true_count) triples across K for one background,
    per corpus, to calibrate LSH hyperparameters against this scheme's
    actual r_ic distribution rather than reusing another run's numbers.
    Uses its own small delta/pole grid, independent of the main run's
    n_theta/m_per_theta, to keep calibration cost bounded."""
    rng, delta_grid, real_anchor = _delta_grid_and_anchor(CALIB_BACKGROUND_ID, real_background, n_points,
                                                            CALIB_N_THETA, seed)
    H = _householder_e1_to(real_anchor)
    r_cache_real, idx_real = _anchor_response_sets(real_background, real_anchor)
    r_cache_uniform, idx_uniform = _anchor_response_sets(uniform_corpus, real_anchor)

    samples = {'real': {'q': [], 'r': [], 'true_count': []}, 'uniform': {'q': [], 'r': [], 'true_count': []}}
    for delta in delta_grid:
        delta = float(delta)
        test_poles = _construct_test_poles(delta, CALIB_M_PER_THETA, D, rng)
        relocated = _normalize(test_poles @ H)

        for corpus, idx_by_K, label in ((real_background, idx_real, 'real'), (uniform_corpus, idx_uniform, 'uniform')):
            for K in K_VALUES:
                resp_coords = corpus[idx_by_K[K]]
                r_ic = np.arccos(np.clip(relocated @ resp_coords.T, -1.0, 1.0)).max(axis=1)
                true_count = np.array([
                    int((np.arccos(np.clip(corpus @ relocated[i], -1.0, 1.0)) <= r_ic[i] + 1e-9).sum())
                    for i in range(CALIB_M_PER_THETA)
                ])
                samples[label]['q'].append(relocated)
                samples[label]['r'].append(r_ic)
                samples[label]['true_count'].append(true_count)

    for label in samples:
        samples[label]['q'] = np.concatenate(samples[label]['q'], axis=0)
        samples[label]['r'] = np.concatenate(samples[label]['r'])
        samples[label]['true_count'] = np.concatenate(samples[label]['true_count'])
    return samples


def calibrate_hyperparameters(points: np.ndarray, q_samples: np.ndarray, r_samples: np.ndarray,
                               true_counts: np.ndarray, error_target: float, rng: np.random.Generator) -> dict:
    """First (m,x,L) in ascending-cost order meeting error_target on a real
    sample from this scheme's own r_ic distribution. Building L tables of
    N=50,000 points is the dominant cost, so an exhaustive scan over the
    full grid (up to L=64) is far more expensive than needed - ascending-
    cost order with early exit finds a cheap-enough config without paying
    for every combination."""
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


def sanity_check_recall(pooled: dict, run_dir_old: Path, part_dir_old: str) -> None:
    """Compares regenerated recall against the previously saved recall,
    which must match closely if the rng stream was reproduced correctly."""
    print(f"\n  Sanity check vs {run_dir_old / part_dir_old}:")
    for K in K_VALUES:
        old_path = run_dir_old / part_dir_old / f'{D}d' / 'raw' / f'K{K}.npz'
        if not old_path.exists():
            print(f"    K={K}: old file not found at {old_path}, skipping")
            continue
        old = np.load(old_path)
        new_recall = pooled[K]['recall']
        old_recall = old['recall']
        if len(new_recall) != len(old_recall):
            print(f"    K={K}: row count mismatch (new={len(new_recall)}, old={len(old_recall)}) - cannot compare")
            continue
        max_diff = float(np.max(np.abs(new_recall - old_recall)))
        print(f"    K={K}: max|recall_new - recall_old|={max_diff:.6f}")


def _write_progress(progress_path: Path, done: int, total: int, start_time: float) -> None:
    tmp_path = progress_path.with_suffix('.tmp')
    payload = {'done': done, 'total': total, 'elapsed_seconds': time.time() - start_time,
               'timestamp': datetime.now().isoformat()}
    tmp_path.write_text(json.dumps(payload))
    tmp_path.replace(progress_path)

#-------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-backgrounds', type=int, default=N_BACKGROUNDS, dest='n_backgrounds')
    parser.add_argument('--n-background-points', type=int, default=N_BACKGROUND_POINTS, dest='n_background_points')
    parser.add_argument('--n-theta', type=int, default=N_THETA, dest='n_theta')
    parser.add_argument('--m-per-theta', type=int, default=M_PER_THETA, dest='m_per_theta')
    parser.add_argument('--background-path', type=str, default=BACKGROUND_PATH, dest='background_path')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--error-target', type=float, default=ERROR_TARGET, dest='error_target')
    parser.add_argument('--run-dir', type=str, default=None)
    parser.add_argument('--old-run-dir', type=str, default='results/20260829_193502', dest='old_run_dir',
                         help='previously saved IF-vs-recall run, for the recall sanity check')
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else _run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")
    print(f"K values: {K_VALUES}, n_backgrounds={args.n_backgrounds}, "
          f"n_background_points={args.n_background_points}, n_theta={args.n_theta}, "
          f"m_per_theta={args.m_per_theta}\n")

    repo_root = Path(__file__).parent.parent.parent.parent
    bg_path = Path(args.background_path)
    if not bg_path.is_absolute():
        bg_path = repo_root / bg_path
    real_background = _normalize(np.load(bg_path).astype(np.float64))

    uniform_corpus_rng = np.random.default_rng(UNIFORM_CORPUS_SEED)
    uniform_corpus = _sample_uniform_sphere(len(real_background), D, uniform_corpus_rng)

    print("Calibrating: pooling r_ic samples across K on background 0...")
    calib = collect_calibration_samples(real_background, uniform_corpus, args.n_background_points, args.seed)
    calibration = {}
    calib_rng = np.random.default_rng(args.seed)
    for label, points in (('real', real_background), ('uniform', uniform_corpus)):
        s = calib[label]
        print(f"  {label}: r_ic percentiles={np.percentile(s['r'], [0, 25, 50, 75, 100])}")
        t0 = time.time()
        best = calibrate_hyperparameters(points, s['q'], s['r'], s['true_count'], args.error_target, calib_rng)
        print(f"  {label} config: {best} ({time.time() - t0:.1f}s)")
        calibration[label] = best

    calib_out = {'error_target': args.error_target,
                 'real': calibration['real'], 'uniform': calibration['uniform'],
                 'r_ic_percentiles_real': np.percentile(calib['real']['r'], [0, 25, 50, 75, 100]).tolist(),
                 'r_ic_percentiles_uniform': np.percentile(calib['uniform']['r'], [0, 25, 50, 75, 100]).tolist()}
    (run_dir / 'calibration.json').write_text(json.dumps(calib_out, indent=2))

    m_real, x_real, L_real = calibration['real']['m'], calibration['real']['x'], calibration['real']['L']
    m_uni, x_uni, L_uni = calibration['uniform']['m'], calibration['uniform']['x'], calibration['uniform']['L']
    table_rng = np.random.default_rng(args.seed + 1)
    tables_real = build_tables(real_background, m_real, L_real, table_rng)
    tables_uniform = build_tables(uniform_corpus, m_uni, L_uni, table_rng)

    fields = ('background_id', 'delta', 'r_cache', 'r_ic', 'per_cache_query_n_hat', 'per_cache_query_recall_hat',
               'r_hat_q_star', 'if_fair', 'per_test_query_recall_hat', 'recall')
    pooled_real = {K: {f: [] for f in fields} for K in K_VALUES}
    pooled_uniform = {K: {f: [] for f in fields} for K in K_VALUES}

    progress_path = run_dir / 'progress.json'
    start_time = time.time()
    for bg_id in tqdm(range(args.n_backgrounds), desc='backgrounds'):
        t0 = time.time()
        out_real, out_uniform = score_background_recall_hat(
            bg_id, real_background, uniform_corpus, tables_real, tables_uniform,
            (m_real, x_real, L_real), (m_uni, x_uni, L_uni),
            args.n_background_points, args.n_theta, args.m_per_theta, args.seed)
        if bg_id == 0:
            elapsed = time.time() - t0
            print(f"  background 0 took {elapsed:.1f}s, projected total ~{elapsed * args.n_backgrounds / 60:.1f} min")
        for K in K_VALUES:
            for f in fields:
                pooled_real[K][f].append(out_real[K][f])
                pooled_uniform[K][f].append(out_uniform[K][f])
        _write_progress(progress_path, bg_id + 1, args.n_backgrounds, start_time)

    for pooled in (pooled_real, pooled_uniform):
        for K in K_VALUES:
            for f in fields:
                pooled[K][f] = np.concatenate(pooled[K][f])

    save_results(pooled_real, run_dir, PART_DIR)
    save_results(pooled_uniform, run_dir, PART_DIR_UNIFORM)

    old_run_dir = Path(args.old_run_dir)
    if not old_run_dir.is_absolute():
        old_run_dir = Path(__file__).parent.parent / old_run_dir
    sanity_check_recall(pooled_real, old_run_dir, 'sift_part2_real_data')
    sanity_check_recall(pooled_uniform, old_run_dir, 'sift_part2_real_anchor_on_uniform')

    print(f"\nDone: {run_dir}")


if __name__ == '__main__':
    main()
