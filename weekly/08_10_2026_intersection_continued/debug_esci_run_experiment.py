#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""ESCI validation of the pool+argmax intersection-fraction calibration
experiment, cap intersection only.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from primitives import exact_intersection_fraction
from datasets.dataloaders import load_esci_data

#-------------------------------------------------------------------------------

K_VALUES = [10, 25, 50, 100]
K_MAX = max(K_VALUES)
N_SEEDS = 5
N_CANDIDATES = 1000
N_TEST_QUERIES = 1000
D = 384
MAX_PAIRWISE_SAMPLE = 20000

# fixed 50,000-vector background sample shared by every esci_c*k100_* benchmark directory
BASE_VECTORS_PATH = 'datasets/esci/esci_c1024k100_t512n20_s42/base_vectors.npy'

OUT_DIR = Path(__file__).parent / 'results'
PART_DIR = 'debug_esci'
_RUN_TIMESTAMP = None


def _run_dir() -> Path:
    """One timestamp per process run."""
    global _RUN_TIMESTAMP
    if _RUN_TIMESTAMP is None:
        _RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
    return OUT_DIR / _RUN_TIMESTAMP

#-------------------------------------------------------------------------------

def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vecs / norms

def _chord_to_angle(chord: np.ndarray) -> np.ndarray:
    """Inverse of chord_distance_from_angular: theta = 2*arcsin(chord/2)."""
    return 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))

def _bruteforce_knn(background: np.ndarray, queries: np.ndarray, k: int, chunk_size: int = 1000) -> tuple:
    """Chunked over queries to bound memory."""
    n_queries = queries.shape[0]
    all_dists = np.empty((n_queries, k), dtype=np.float64)
    all_idx = np.empty((n_queries, k), dtype=np.int64)
    for start in range(0, n_queries, chunk_size):
        end = min(start + chunk_size, n_queries)
        sims = queries[start:end] @ background.T
        top_idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        top_sims = np.take_along_axis(sims, top_idx, axis=1)
        order = np.argsort(-top_sims, axis=1)
        sorted_idx = np.take_along_axis(top_idx, order, axis=1)
        sorted_sims = np.take_along_axis(top_sims, order, axis=1)
        all_dists[start:end] = np.sqrt(np.clip(2.0 - 2.0 * sorted_sims, 0.0, None))
        all_idx[start:end] = sorted_idx
    return all_dists, all_idx

def _query_split(n_queries: int, n_cache: int, n_test: int, seed: int) -> tuple:
    """Non-overlapping (cache_indices, test_indices) permutation of the
    query pool."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_queries)
    return perm[:n_cache], perm[n_cache:n_cache + n_test]


def run_seed(background: np.ndarray, query_pool: np.ndarray, seed_id: int, n_candidates: int,
             n_test_queries: int, seed: int) -> dict:
    """One seed: splits cache_q/test_q from query_pool and computes every K
    stratum's selection and pairwise data.

    Returns {K: {'selection': {...}, 'pairwise': {...}}} for K in K_VALUES.
    """
    cache_idx, test_idx = _query_split(len(query_pool), n_candidates, n_test_queries, seed)
    cache_q = query_pool[cache_idx]
    test_q = query_pool[test_idx]

    dists_cache, _ = _bruteforce_knn(background, cache_q, K_MAX)
    dists_test, idx_test = _bruteforce_knn(background, test_q, K_MAX)

    delta_matrix = np.arccos(np.clip(test_q @ cache_q.T, -1.0, 1.0))

    per_k = {}
    for K in K_VALUES:
        r_cache = _chord_to_angle(dists_cache[:, K - 1])
        r_test = _chord_to_angle(dists_test[:, K - 1])

        cap_matrix = np.array([
            [exact_intersection_fraction(float(r_cache[j]), float(r_test[i]), float(delta_matrix[i, j]), D)
             for j in range(n_candidates)]
            for i in range(n_test_queries)
        ])

        w_idx = cap_matrix.argmax(axis=1)
        winner_r_cache = r_cache[w_idx]
        winner_delta = delta_matrix[np.arange(n_test_queries), w_idx]
        winner_fraction = cap_matrix[np.arange(n_test_queries), w_idx]

        neighbor_coords = background[idx_test[:, :K]]
        winner_cache_q = cache_q[w_idx]
        dist_to_winner = np.linalg.norm(neighbor_coords - winner_cache_q[:, None, :], axis=2)
        winner_r_cache_chord = dists_cache[w_idx, K - 1]
        covered = dist_to_winner <= winner_r_cache_chord[:, None]
        recall = covered.mean(axis=1)

        selection = {
            'seed_id': np.full(n_test_queries, seed_id, dtype=np.int32),
            'r_test': r_test,
            'winner_idx': w_idx.astype(np.int32),
            'r_cache': winner_r_cache,
            'delta': winner_delta,
            'fraction': winner_fraction,
            'recall': recall,
        }

        n_total = n_test_queries * n_candidates
        n_sample = min(n_total, MAX_PAIRWISE_SAMPLE)
        rng = np.random.default_rng(seed)
        flat_i, flat_j = np.divmod(rng.choice(n_total, size=n_sample, replace=False), n_candidates)

        pairwise_neighbor_coords = background[idx_test[flat_i][:, :K]]
        pairwise_cache_q = cache_q[flat_j]
        pairwise_dist = np.linalg.norm(pairwise_neighbor_coords - pairwise_cache_q[:, None, :], axis=2)
        pairwise_r_cache_chord = dists_cache[flat_j, K - 1]
        pairwise_recall = (pairwise_dist <= pairwise_r_cache_chord[:, None]).mean(axis=1)

        pairwise = {
            'seed_id': np.full(n_sample, seed_id, dtype=np.int32),
            'r_cache': r_cache[flat_j],
            'r_test': r_test[flat_i],
            'delta': delta_matrix[flat_i, flat_j],
            'fraction': cap_matrix[flat_i, flat_j],
            'recall': pairwise_recall,
        }

        per_k[K] = {'selection': selection, 'pairwise': pairwise}

    return per_k


def _checkpoint_dir(run_dir: Path) -> Path:
    return run_dir / PART_DIR / f'{D}d' / 'raw' / 'checkpoints'


def _seed_checkpoint_paths(run_dir: Path, seed_id: int) -> dict:
    ckpt_dir = _checkpoint_dir(run_dir)
    return {K: {part: ckpt_dir / f'K{K}_{part}_seed{seed_id}.npz' for part in ('selection', 'pairwise')}
            for K in K_VALUES}


def _seed_is_checkpointed(run_dir: Path, seed_id: int) -> bool:
    paths = _seed_checkpoint_paths(run_dir, seed_id)
    return all(p.exists() for parts in paths.values() for p in parts.values())


def _save_seed_checkpoint(run_dir: Path, seed_id: int, per_k: dict) -> None:
    """Persists one seed's results immediately."""
    ckpt_dir = _checkpoint_dir(run_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    paths = _seed_checkpoint_paths(run_dir, seed_id)
    for K in K_VALUES:
        for part in ('selection', 'pairwise'):
            np.savez_compressed(paths[K][part], d=D, K=K, **per_k[K][part])


def run_all_seeds(background: np.ndarray, query_pool: np.ndarray, n_seeds: int, n_candidates: int,
                   n_test_queries: int, base_seed: int, run_dir: Path) -> None:
    """Runs and checkpoints every seed not already checkpointed."""
    for seed_id in range(n_seeds):
        if _seed_is_checkpointed(run_dir, seed_id):
            print(f"  seed {seed_id + 1}/{n_seeds}: already checkpointed, skipping")
            continue
        print(f"  seed {seed_id + 1}/{n_seeds}...")
        per_k = run_seed(background, query_pool, seed_id, n_candidates, n_test_queries, seed=base_seed + seed_id)
        _save_seed_checkpoint(run_dir, seed_id, per_k)
        print(f"    checkpointed seed {seed_id + 1}/{n_seeds}")


def merge_checkpoints(run_dir: Path, n_seeds: int) -> dict:
    """Combines every seed's checkpoint into one pooled array per field,
    for the bin-count diagnostic and the final combined .npz files."""
    selection_fields = ('seed_id', 'r_test', 'winner_idx', 'r_cache', 'delta', 'fraction', 'recall')
    pairwise_fields = ('seed_id', 'r_cache', 'r_test', 'delta', 'fraction', 'recall')

    pooled = {K: {'selection': {f: [] for f in selection_fields}, 'pairwise': {f: [] for f in pairwise_fields}}
              for K in K_VALUES}

    for seed_id in range(n_seeds):
        paths = _seed_checkpoint_paths(run_dir, seed_id)
        for K in K_VALUES:
            sel = np.load(paths[K]['selection'])
            for field in selection_fields:
                pooled[K]['selection'][field].append(sel[field])
            pw = np.load(paths[K]['pairwise'])
            for field in pairwise_fields:
                pooled[K]['pairwise'][field].append(pw[field])

    for K in K_VALUES:
        for part in ('selection', 'pairwise'):
            for field in pooled[K][part]:
                pooled[K][part][field] = np.concatenate(pooled[K][part][field])
    return pooled


def save_results(pooled: dict, run_dir: Path) -> None:
    """One combined selection .npz and one pairwise .npz per K persisted."""
    dim_dir = run_dir / PART_DIR / f'{D}d' / 'raw'
    dim_dir.mkdir(parents=True, exist_ok=True)
    for K, parts in pooled.items():
        for part_name, arrays in parts.items():
            path = dim_dir / f'K{K}_{part_name}.npz'
            np.savez_compressed(path, d=D, K=K, **arrays)
            print(f"    wrote {path}")


def bin_count_diagnostic(pooled: dict, min_count: int = 30) -> None:
    """Prints per-K bin population counts (5% predicted-fraction
    increments) and flags sparse or empty bins, plus the informative
    fraction."""
    bin_edges = np.arange(0.0, 1.05, 0.05)
    print(f"\n  Bin-count diagnostic (ESCI d={D}, 5% predicted-fraction bins, selected candidate):")
    for K in K_VALUES:
        fractions = pooled[K]['selection']['fraction']
        counts, _ = np.histogram(fractions, bins=bin_edges)
        sparse = [(bin_edges[i], bin_edges[i + 1], c) for i, c in enumerate(counts) if 0 < c < min_count]
        empty = int(np.sum(counts == 0))
        print(f"    K={K}: total={len(fractions)}, min bin count={counts.min()}, "
              f"empty bins={empty}, sparse bins (<{min_count})={len(sparse)}")
        if sparse:
            ranges = ', '.join(f'[{lo:.2f},{hi:.2f}):{c}' for lo, hi, c in sparse[:6])
            more = '' if len(sparse) <= 6 else f' (+{len(sparse)-6} more)'
            print(f"      FLAG sparse bins: {ranges}{more}")

    print(f"\n  Informative fraction (ESCI d={D})")
    for K in K_VALUES:
        sel = pooled[K]['selection']
        informative_frac = (sel['fraction'] > 0).mean()
        print(f"    K={K}: informative={informative_frac:.4f}")


def main() -> None:
    global K_VALUES, K_MAX, OUT_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--k-values', type=int, nargs='+', default=K_VALUES, dest='k_values')
    parser.add_argument('--n-seeds', type=int, default=N_SEEDS, dest='n_seeds')
    parser.add_argument('--n-candidates', type=int, default=N_CANDIDATES, dest='n_candidates')
    parser.add_argument('--n-test-queries', type=int, default=N_TEST_QUERIES, dest='n_test_queries')
    parser.add_argument('--esci-dir', type=str, default='datasets/esci', dest='esci_dir')
    parser.add_argument('--base-vectors-path', type=str, default=BASE_VECTORS_PATH, dest='base_vectors_path')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--run-dir', type=str, default=None,
                         help='write directly into this exact directory (no new timestamp minted)')
    args = parser.parse_args()

    K_VALUES = sorted(args.k_values)
    K_MAX = max(K_VALUES)
    if args.output_dir:
        OUT_DIR = Path(args.output_dir)

    run_dir = Path(args.run_dir) if args.run_dir else _run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")
    print(f"K values: {K_VALUES}, n_seeds={args.n_seeds}, "
          f"n_candidates={args.n_candidates}, n_test_queries={args.n_test_queries}\n")

    repo_root = Path(__file__).parent.parent.parent

    esci_dir = Path(args.esci_dir)
    if not esci_dir.is_absolute():
        esci_dir = repo_root / esci_dir
    _, query, _, _ = load_esci_data(str(esci_dir), load_embeddings=False)

    base_path = Path(args.base_vectors_path)
    if not base_path.is_absolute():
        base_path = repo_root / base_path
    base = np.load(base_path)

    background = _normalize(base.astype(np.float64))
    query_pool = _normalize(query.astype(np.float64))

    run_all_seeds(background, query_pool, args.n_seeds, args.n_candidates, args.n_test_queries,
                  base_seed=args.seed, run_dir=run_dir)
    pooled = merge_checkpoints(run_dir, args.n_seeds)
    save_results(pooled, run_dir)
    bin_count_diagnostic(pooled)

    print(f"\nDone: {run_dir}")


if __name__ == '__main__':
    main()
