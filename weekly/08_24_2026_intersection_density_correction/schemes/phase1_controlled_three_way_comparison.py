#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Phase 1 (controlled synthetic queries) of the three-way recall
predictor comparison: IF vs. per_cache_query_recall_hat vs.
per_test_query_recall_hat, under full control of the geometry and with
exact (brute-force, no LSH) arithmetic throughout - isolates whether each
formula is structurally correct before any real-query or LSH-estimation
variability is introduced (Phase 2, in the run-experiment scripts).

Two corpora, matching the plan's split:
(a) fully synthetic uniform-on-hypersphere - density heterogeneity
    provably absent.
(b) real SIFT vectors as the corpus, but with synthetic/controlled query
    placement (not the real SIFT query set) - isolates whether the
    mechanism holds against real corpus geometry, separate from real
    query-distribution effects.

r_hat_q* is computed exactly here (brute-force true K-NN radius of the
query) rather than via LSH, matching this session's original bias
diagnosis methodology - Phase 2 is where LSH estimation noise enters."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'common'))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import numpy as np

from primitives import exact_intersection_fraction
from sift_part2_real_data_run_experiment import _normalize, _real_anchor

#-------------------------------------------------------------------------------

D = 128
K = 100
N_TRIALS_PER_DELTA = 200
DELTAS = [1e-6, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.4]
BACKGROUND_PATH = 'datasets/sift/base_vectors_50k.npy'

#-------------------------------------------------------------------------------


def _random_tangent(anchor: np.ndarray, rng: np.random.Generator, d: int) -> np.ndarray:
    v = rng.standard_normal(d)
    v -= (v @ anchor) * anchor
    v /= np.linalg.norm(v)
    return v


def _move(anchor: np.ndarray, delta: float, direction: np.ndarray) -> np.ndarray:
    moved = np.cos(delta) * anchor + np.sin(delta) * direction
    return moved / np.linalg.norm(moved)


def run_one_corpus(points: np.ndarray, anchor: np.ndarray, corpus_label: str, seed: int) -> None:
    """anchor is the fixed candidate position c (matching Scheme A/B's own
    convention: the candidate sits at/near a real anchor, and test queries
    are relocated away from it at controlled delta) - not the reverse,
    since real corpus density isn't translation-invariant and Phase 1
    should isolate the same geometry Phase 2 actually uses. All three
    predictors computed exactly (brute-force), no LSH."""
    rng = np.random.default_rng(seed)
    c = anchor
    dists_c = np.arccos(np.clip(points @ c, -1.0, 1.0))
    idx_c_sorted = np.argsort(dists_c)
    r_c = idx_c_sorted[:K]  # candidate's own exact K-NN (R_c), fixed across all deltas/trials
    r_cache = float(dists_c[idx_c_sorted[K - 1]])  # candidate's own true K-NN radius

    print(f"\n=== {corpus_label} (r_cache = {r_cache:.4f}) ===")
    header = f"{'delta':>8} | {'IF':>8} {'pcq':>8} {'ptq':>8} | {'actual':>8} || {'IF-act':>8} {'pcq-act':>8} {'ptq-act':>8}"
    print(header)
    print('-' * len(header))

    for delta in DELTAS:
        actuals, if_preds, pcq_preds, ptq_preds = [], [], [], []
        for _ in range(N_TRIALS_PER_DELTA):
            q = _move(c, delta, _random_tangent(c, rng, D))
            dists_q = np.arccos(np.clip(points @ q, -1.0, 1.0))
            idx_q_sorted = np.argsort(dists_q)
            topk_q = set(idx_q_sorted[:K].tolist())
            r_q_star = float(dists_q[idx_q_sorted[K - 1]])  # exact true K-NN radius of q
            theta_sorted_q = np.sort(dists_q)  # for exact N(q, r) via searchsorted

            member_dists_to_q = np.arccos(np.clip(points[r_c] @ q, -1.0, 1.0))
            r_ic = float(member_dists_to_q.max())
            n_exact = int(np.searchsorted(theta_sorted_q, r_ic, side='right'))

            actual_recall = len(topk_q & set(r_c.tolist())) / K
            pcq_pred = K / n_exact if n_exact > 0 else float('inf')
            ptq_pred = float((member_dists_to_q <= r_q_star).mean())
            if_pred = exact_intersection_fraction(r_cache, r_q_star, delta, D)

            actuals.append(actual_recall)
            pcq_preds.append(np.clip(pcq_pred, 0, 1))
            ptq_preds.append(ptq_pred)
            if_preds.append(if_pred)

        mean_actual = np.mean(actuals)
        mean_if, mean_pcq, mean_ptq = np.mean(if_preds), np.mean(pcq_preds), np.mean(ptq_preds)
        print(f"{delta:>8.4f} | {mean_if:>8.4f} {mean_pcq:>8.4f} {mean_ptq:>8.4f} | {mean_actual:>8.4f} || "
              f"{mean_actual - mean_if:>8.4f} {mean_actual - mean_pcq:>8.4f} {mean_actual - mean_ptq:>8.4f}")


def run_uniform_sphere(seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    N = 50000
    points = rng.standard_normal((N, D))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    anchor = np.zeros(D)
    anchor[0] = 1.0
    run_one_corpus(points, anchor, 'synthetic uniform-on-hypersphere', seed + 1)


def run_real_sift(seed: int = 0) -> None:
    repo_root = Path(__file__).parent.parent.parent.parent
    bg_path = repo_root / BACKGROUND_PATH
    background = _normalize(np.load(bg_path).astype(np.float64))
    anchor = _real_anchor(background, background_id=0, d=D)
    run_one_corpus(background, anchor, 'real SIFT corpus, controlled query placement', seed + 1)


def main() -> None:
    run_uniform_sphere()
    run_real_sift()


if __name__ == '__main__':
    main()
