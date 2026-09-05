#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Synthetic angular-separation sweep on real ESCI data: tests how
recall tracks IF as synthetic candidates/queries move away from a
real-corpus anchor."""

import sys
import argparse
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from scipy.optimize import brentq

from primitives import exact_intersection_fraction, exact_intersection_fraction_batch
from part2_uniform_baseline_run_experiment import _chord_to_angle, _bruteforce_knn

#-------------------------------------------------------------------------------

K_VALUES = [25, 50, 100]
K_MAX = max(K_VALUES)
D = 384
N_ANCHORS = 200
N_DIRECTIONS = 10
N_IF_BINS = 20
IF_BIN_EDGES = np.linspace(0.0, 1.0, N_IF_BINS + 1)
IF_BIN_CENTERS = (IF_BIN_EDGES[:-1] + IF_BIN_EDGES[1:]) / 2.0

HOME_DELTA = 1e-6
CANDIDATE_TARGET_IFS = np.array([0.8, 0.5, 0.25, 0.1, 0.02])

BACKGROUND_PATH = 'datasets/esci/base_vectors_50k.npy'
UNIFORM_CORPUS_SEED = 800000  # independent of the generation-critical rng stream

OUT_DIR = Path(__file__).parent / 'results'
PART_DIR = 'esci_synth_sweep'
PART_DIR_UNIFORM = 'esci_synth_sweep_on_uniform'
_RUN_TIMESTAMP = None


def _run_dir() -> Path:
    global _RUN_TIMESTAMP
    if _RUN_TIMESTAMP is None:
        _RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
    return OUT_DIR / _RUN_TIMESTAMP

#-------------------------------------------------------------------------------

def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vecs / norms

def _random_tangent(anchor: np.ndarray, rng: np.random.Generator, d: int) -> np.ndarray:
    v = rng.standard_normal(d)
    v -= (v @ anchor) * anchor
    v /= np.linalg.norm(v)
    return v

def _move_along_geodesic(anchor: np.ndarray, deltas: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """deltas: (n,), directions: (n, d), anchor: (d,). Returns (n, d), normalized."""
    moved = np.cos(deltas)[:, None] * anchor[None, :] + np.sin(deltas)[:, None] * directions
    return _normalize(moved)


def _delta_targets_for_radius(r: float, d: int, targets: np.ndarray) -> np.ndarray:
    """Root-finds delta values hitting each target IF under the
    equal-radii approximation, bracketed in [eps, 2r]."""
    delta_hi = 2.0 * r
    deltas = []
    for target in targets:
        f = lambda delta: exact_intersection_fraction(r, r, delta, d) - target
        try:
            deltas.append(brentq(f, 1e-6, delta_hi))
        except ValueError:
            continue
    return np.array(deltas)


def _anchor_delta_grid(r_cache_by_K: dict, d: int, targets: np.ndarray) -> np.ndarray:
    """Union of per-K IF-targeted delta grids for one anchor, deduplicated."""
    all_deltas = [_delta_targets_for_radius(r, d, targets) for r in r_cache_by_K.values()]
    combined = np.concatenate(all_deltas)
    return np.unique(np.round(combined, 6))

#-------------------------------------------------------------------------------

def build_candidate_pool(background: np.ndarray, anchors: np.ndarray, rng: np.random.Generator,
                          d: int) -> tuple:
    """Per anchor: one tiny-delta 'home' candidate plus IF-targeted
    'sibling' candidates. Returns (candidates, home_indices)."""
    home_directions = np.array([_random_tangent(a, rng, d) for a in anchors])
    # Elementwise (not _move_along_geodesic): every row has its own anchor.
    homes = _normalize(np.cos(HOME_DELTA) * anchors + np.sin(HOME_DELTA) * home_directions)

    dists, _ = _bruteforce_knn(background, homes, K_MAX)
    r_scale_by_K = {K: _chord_to_angle(dists[:, K - 1]) for K in K_VALUES}

    chunks = []
    home_indices = np.empty(len(anchors), dtype=np.int64)
    offset = 0
    for a_id, anchor in enumerate(anchors):
        home_indices[a_id] = offset
        r_cache_by_K = {K: float(r_scale_by_K[K][a_id]) for K in K_VALUES}
        sib_deltas = _anchor_delta_grid(r_cache_by_K, d, CANDIDATE_TARGET_IFS)
        sib_directions = np.array([_random_tangent(anchor, rng, d) for _ in range(len(sib_deltas))])
        siblings = _move_along_geodesic(anchor, sib_deltas, sib_directions) if len(sib_deltas) else \
            np.empty((0, d))
        chunk = np.vstack([homes[a_id:a_id + 1], siblings])
        chunks.append(chunk)
        offset += chunk.shape[0]

    candidates = np.concatenate(chunks, axis=0)
    return candidates, home_indices


def precompute_candidates(background: np.ndarray, candidates: np.ndarray) -> dict:
    """Per-K response-set indices and true K-NN radius, computed once
    per candidate."""
    dists, idx = _bruteforce_knn(background, candidates, K_MAX)
    out = {}
    for K in K_VALUES:
        out[K] = {'idx': idx[:, :K], 'r_cache': _chord_to_angle(dists[:, K - 1])}
    return out


def evaluate_test_queries(queries: np.ndarray, background: np.ndarray, candidates: np.ndarray,
                           cand_data: dict, home_idx: int) -> dict:
    """One anchor/delta's batch of test queries (varying only by random
    tangent direction). Returns {K: {'fraction', 'recall', 'is_home_winner'}}."""
    dists_test, idx_test = _bruteforce_knn(background, queries, K_MAX)
    delta_qc = np.arccos(np.clip(candidates @ queries.T, -1.0, 1.0))  # (n_cand, n_queries)

    out = {K: {'fraction': np.empty(len(queries)), 'recall': np.empty(len(queries)),
               'is_home_winner': np.empty(len(queries), dtype=bool)} for K in K_VALUES}
    for K in K_VALUES:
        r_test_all = _chord_to_angle(dists_test[:, K - 1])
        r_cache = cand_data[K]['r_cache']  # (n_cand,)
        resp_idx = cand_data[K]['idx']  # (n_cand, K)
        for i in range(len(queries)):
            r_test = float(r_test_all[i])
            IFs = exact_intersection_fraction_batch(r_cache, r_test, delta_qc[:, i], D)
            best = int(IFs.argmax())
            true_topk = set(idx_test[i, :K].tolist())
            best_response = set(resp_idx[best].tolist())
            out[K]['fraction'][i] = IFs[best]
            out[K]['recall'][i] = len(true_topk & best_response) / K
            out[K]['is_home_winner'][i] = (best == home_idx)
    return out


def run_all(background: np.ndarray, n_anchors: int, n_directions: int, seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    anchor_idx = rng.choice(len(background), size=n_anchors, replace=False)
    anchors = background[anchor_idx]

    print("  building candidate swarm...")
    candidates, home_indices = build_candidate_pool(background, anchors, rng, D)
    print(f"  candidate pool size: {len(candidates)} ({len(candidates) / n_anchors:.2f}/anchor avg)")

    print("  precomputing candidate response sets and radii (real corpus)...")
    cand_data_real = precompute_candidates(background, candidates)

    uniform_rng = np.random.default_rng(UNIFORM_CORPUS_SEED)
    uniform_corpus = _normalize(uniform_rng.standard_normal((len(background), D)))
    print("  precomputing candidate response sets and radii (size-matched uniform corpus)...")
    cand_data_uniform = precompute_candidates(uniform_corpus, candidates)

    fields = ('anchor_id', 'delta', 'fraction', 'recall', 'is_home_winner')
    pooled_real = {K: {f: [] for f in fields} for K in K_VALUES}
    pooled_uniform = {K: {f: [] for f in fields} for K in K_VALUES}

    total_deltas = 0
    for a_id in range(n_anchors):
        anchor = anchors[a_id]
        home_idx = int(home_indices[a_id])
        # Query placement stays calibrated to the real corpus's r_cache.
        r_cache_by_K = {K: float(cand_data_real[K]['r_cache'][home_idx]) for K in K_VALUES}
        deltas = _anchor_delta_grid(r_cache_by_K, D, IF_BIN_CENTERS)
        total_deltas += len(deltas)
        if a_id == 0:
            print(f"  anchor 0: {len(deltas)} IF-targeted test-query delta values (union over K)")

        for delta in deltas:
            directions = np.array([_random_tangent(anchor, rng, D) for _ in range(n_directions)])
            queries = _move_along_geodesic(anchor, np.full(n_directions, delta), directions)

            results_real = evaluate_test_queries(queries, background, candidates, cand_data_real, home_idx=home_idx)
            results_uniform = evaluate_test_queries(queries, uniform_corpus, candidates, cand_data_uniform,
                                                      home_idx=home_idx)
            for K in K_VALUES:
                for results, pooled in ((results_real, pooled_real), (results_uniform, pooled_uniform)):
                    pooled[K]['anchor_id'].append(np.full(n_directions, a_id, dtype=np.int32))
                    pooled[K]['delta'].append(np.full(n_directions, delta, dtype=np.float64))
                    pooled[K]['fraction'].append(results[K]['fraction'])
                    pooled[K]['recall'].append(results[K]['recall'])
                    pooled[K]['is_home_winner'].append(results[K]['is_home_winner'])

        if (a_id + 1) % 20 == 0 or a_id == n_anchors - 1:
            print(f"  anchor {a_id + 1}/{n_anchors}... (avg {total_deltas / (a_id + 1):.1f} test deltas/anchor so far)")

    for K in K_VALUES:
        for f in fields:
            pooled_real[K][f] = np.concatenate(pooled_real[K][f])
            pooled_uniform[K][f] = np.concatenate(pooled_uniform[K][f])
    return pooled_real, pooled_uniform


def save_results(pooled: dict, run_dir: Path, part_dir: str) -> None:
    dim_dir = run_dir / part_dir / f'{D}d' / 'raw'
    dim_dir.mkdir(parents=True, exist_ok=True)
    for K, arrays in pooled.items():
        path = dim_dir / f'K{K}.npz'
        np.savez_compressed(path, d=D, K=K, **arrays)
        print(f"    wrote {path}")


def coverage_diagnostic(pooled: dict, corpus_label: str) -> None:
    """Pre-downsampling IF coverage and is_home_winner rate by bin,
    reported before any recall/relationship analysis. Balancing
    happens at analysis time, not here."""
    print(f"\n  Coverage diagnostic (ESCI d={D}, {corpus_label}), pre-downsampling, "
          f"{N_IF_BINS} bins of width {1.0 / N_IF_BINS:.2f}:")
    for K in K_VALUES:
        fraction = pooled[K]['fraction']
        is_home = pooled[K]['is_home_winner']
        counts, _ = np.histogram(fraction, bins=IF_BIN_EDGES)
        bin_idx = np.clip(np.digitize(fraction, IF_BIN_EDGES[1:-1]), 0, N_IF_BINS - 1)
        print(f"    K={K}: total={len(fraction)}, min bin count={counts.min()}, max bin count={counts.max()}, "
              f"overall is_home_winner rate={is_home.mean():.4f}")
        for b in range(N_IF_BINS):
            mask = bin_idx == b
            if mask.sum() == 0:
                continue
            print(f"      [{IF_BIN_EDGES[b]:.2f},{IF_BIN_EDGES[b+1]:.2f}): n={counts[b]}, "
                  f"is_home_winner rate={is_home[mask].mean():.4f}")


def recall_diagnostic(pooled: dict, corpus_label: str) -> None:
    """Recall-vs-IF summary, reported only after coverage is confirmed
    acceptable."""
    print(f"\n  Recall diagnostic (ESCI d={D}, {corpus_label}):")
    for K in K_VALUES:
        recall = pooled[K]['recall']
        print(f"    K={K}: recall range=[{recall.min():.4f}, {recall.max():.4f}], recall mean={recall.mean():.4f}")

#-------------------------------------------------------------------------------

def main() -> None:
    global K_VALUES, K_MAX, OUT_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--k-values', type=int, nargs='+', default=K_VALUES, dest='k_values')
    parser.add_argument('--n-anchors', type=int, default=N_ANCHORS, dest='n_anchors')
    parser.add_argument('--n-directions', type=int, default=N_DIRECTIONS, dest='n_directions')
    parser.add_argument('--background-path', type=str, default=BACKGROUND_PATH, dest='background_path')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--run-dir', type=str, default=None)
    parser.add_argument('--coverage-only', action='store_true',
                         help='report only the IF coverage diagnostic, skip the recall diagnostic')
    args = parser.parse_args()

    K_VALUES = sorted(args.k_values)
    K_MAX = max(K_VALUES)
    if args.output_dir:
        OUT_DIR = Path(args.output_dir)

    run_dir = Path(args.run_dir) if args.run_dir else _run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")
    print(f"K values: {K_VALUES}, n_anchors={args.n_anchors}, n_directions={args.n_directions}\n")

    repo_root = Path(__file__).parent.parent.parent
    bg_path = Path(args.background_path)
    if not bg_path.is_absolute():
        bg_path = repo_root / bg_path
    background = _normalize(np.load(bg_path).astype(np.float64))

    pooled_real, pooled_uniform = run_all(background, args.n_anchors, args.n_directions, args.seed)

    save_results(pooled_real, run_dir, PART_DIR)
    coverage_diagnostic(pooled_real, 'real corpus')
    if not args.coverage_only:
        recall_diagnostic(pooled_real, 'real corpus')

    save_results(pooled_uniform, run_dir, PART_DIR_UNIFORM)
    coverage_diagnostic(pooled_uniform, 'uniform corpus, same anchor/queries')
    if not args.coverage_only:
        recall_diagnostic(pooled_uniform, 'uniform corpus, same anchor/queries')

    print(f"\nDone: {run_dir}")


if __name__ == '__main__':
    main()
