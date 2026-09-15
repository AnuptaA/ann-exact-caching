#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Setup (a) on real SIFT data: cache candidates placed uniformly at
random (decoupled from data density), test queries drawn directly from
the real query pool. Tests whether recall tracks IF closer to y=x than
the anchor-perturbation design (Scheme A/B) when candidate placement is
no longer tied to a real-corpus anchor."""

import sys
import json
import time
import resource
import argparse
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'common'))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import numpy as np
from tqdm import tqdm

from datasets.dataloaders import load_fvecs
from primitives import exact_intersection_fraction_batch
from part2_uniform_baseline_run_experiment import _chord_to_angle, _bruteforce_knn, _sample_uniform_sphere
from sift_synth_sweep_run_experiment import _normalize, precompute_candidates, K_VALUES, K_MAX, D, BACKGROUND_PATH
from sift_touch_coverage_test import _angular_distance_matrix

#-------------------------------------------------------------------------------

QUERY_PATH = 'datasets/sift/sift_query.fvecs'
N_CANDIDATES = 2000
N_TEST_QUERIES = 8000
SEED = 42
CHUNK_SIZE = 200

OUT_DIR = Path(__file__).parent.parent / 'results'
PART_DIR = 'sift_setup_a'
_RUN_TIMESTAMP = None

# Percentile summary used to characterize the realized IF distribution
# before any bin edges are chosen for analysis.
FRACTION_PERCENTILES = [0, 1, 5, 25, 50, 75, 95, 99, 100]

#-------------------------------------------------------------------------------


def _run_dir() -> Path:
    global _RUN_TIMESTAMP
    if _RUN_TIMESTAMP is None:
        _RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
    return OUT_DIR / _RUN_TIMESTAMP


def _write_progress(progress_path: Path, q_done: int, n_queries: int, start_time: float) -> None:
    """Overwrites a small JSON progress file, readable at any instant
    regardless of stdout buffering."""
    tmp_path = progress_path.with_suffix('.tmp')
    payload = {
        'queries_done': q_done + 1,
        'queries_total': n_queries,
        'elapsed_seconds': time.time() - start_time,
        'timestamp': datetime.now().isoformat(),
    }
    tmp_path.write_text(json.dumps(payload))
    tmp_path.replace(progress_path)


def evaluate_setup_a(queries: np.ndarray, background: np.ndarray, candidates: np.ndarray,
                      cand_data: dict, k_values: list, k_max: int, d: int,
                      progress_path: Path = None, chunk_size: int = CHUNK_SIZE) -> dict:
    """Argmax-selected best candidate per test query, per K, chunked
    over queries to bound the (n_cand, chunk) angular-distance matrix.
    Returns {K: {'fraction': array, 'recall': array}}."""
    n_queries = queries.shape[0]
    dists_test, idx_test = _bruteforce_knn(background, queries, k_max)

    r_cache_by_K = {K: cand_data[K]['r_cache'] for K in k_values}
    resp_idx_by_K = {K: cand_data[K]['idx'] for K in k_values}
    r_test_by_K = {K: _chord_to_angle(dists_test[:, K - 1]) for K in k_values}

    out = {K: {'fraction': np.empty(n_queries), 'recall': np.empty(n_queries)} for K in k_values}
    start_time = time.time()
    n_chunks = (n_queries + chunk_size - 1) // chunk_size

    for c in tqdm(range(n_chunks), desc='query chunks', mininterval=1.0):
        start = c * chunk_size
        end = min(start + chunk_size, n_queries)
        chunk = queries[start:end]
        delta_qc = _angular_distance_matrix(candidates, chunk)  # (n_cand, chunk_n)

        for K in k_values:
            r_cache = r_cache_by_K[K]
            resp_idx = resp_idx_by_K[K]
            for j in range(chunk.shape[0]):
                q_global = start + j
                r_test = float(r_test_by_K[K][q_global])
                IFs = exact_intersection_fraction_batch(r_cache, r_test, delta_qc[:, j], d)
                best = int(IFs.argmax())
                true_topk = set(idx_test[q_global, :K].tolist())
                best_response = set(resp_idx[best].tolist())
                out[K]['fraction'][q_global] = IFs[best]
                out[K]['recall'][q_global] = len(true_topk & best_response) / K

        if progress_path is not None:
            _write_progress(progress_path, end - 1, n_queries, start_time)

    return out


def report_and_save(results: dict, run_dir: Path, d: int, part_dir: str = PART_DIR,
                     dataset_label: str = 'sift') -> None:
    """Prints the realized IF-distribution percentiles (needed before
    any bin edges are chosen for analysis) and saves raw arrays."""
    print(f"\n=== Setup (a), {dataset_label} (d={d}) ===")
    for K in K_VALUES:
        fraction = results[K]['fraction']
        recall = results[K]['recall']
        pct = np.percentile(fraction, FRACTION_PERCENTILES)
        pct_str = ', '.join(f'p{p}={v:.3e}' for p, v in zip(FRACTION_PERCENTILES, pct))
        print(f"  K={K}: n={len(fraction)}, fraction percentiles: {pct_str}")
        print(f"    recall range=[{recall.min():.4f}, {recall.max():.4f}], mean={recall.mean():.4f}")

    dim_dir = run_dir / part_dir / f'{d}d' / 'raw'
    dim_dir.mkdir(parents=True, exist_ok=True)
    for K in K_VALUES:
        path = dim_dir / f'K{K}.npz'
        np.savez_compressed(path, d=d, K=K, fraction=results[K]['fraction'], recall=results[K]['recall'])
        print(f"    wrote {path}")

#-------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-candidates', type=int, default=N_CANDIDATES, dest='n_candidates')
    parser.add_argument('--n-test-queries', type=int, default=N_TEST_QUERIES, dest='n_test_queries')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--run-dir', type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else _run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")
    print(f"n_candidates={args.n_candidates}, n_test_queries={args.n_test_queries}, seed={args.seed}\n")

    repo_root = Path(__file__).parent.parent.parent.parent
    bg_path = repo_root / BACKGROUND_PATH
    background = _normalize(np.load(bg_path).astype(np.float64))

    query_path = repo_root / QUERY_PATH
    all_queries = _normalize(load_fvecs(str(query_path)).astype(np.float64))
    print(f"Loaded {len(all_queries)} real SIFT queries from {query_path}")

    if args.n_test_queries > len(all_queries):
        raise ValueError(f"n_test_queries ({args.n_test_queries}) exceeds available queries ({len(all_queries)})")
    rng = np.random.default_rng(args.seed)
    test_idx = rng.choice(len(all_queries), size=args.n_test_queries, replace=False)
    queries = all_queries[test_idx]

    candidates = _sample_uniform_sphere(args.n_candidates, D, rng)
    print(f"  candidates (uniform-random): {len(candidates)}, test queries (real pool): {len(queries)}")

    cand_data = precompute_candidates(background, candidates)

    progress_path = run_dir / 'progress.json'
    wall_start = time.time()
    print(f"Evaluating {len(queries)} test queries against the candidate pool...")
    results = evaluate_setup_a(queries, background, candidates, cand_data, K_VALUES, K_MAX, D,
                                progress_path=progress_path)
    wall_elapsed = time.time() - wall_start

    report_and_save(results, run_dir, D)

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"\nWall clock: {wall_elapsed:.2f}s ({wall_elapsed / 3600:.4f}h)")
    print(f"Peak RSS: {peak_rss} (platform units: KB on Linux, bytes on macOS)")
    print(f"Per-query cost: {wall_elapsed / len(queries) * 1000:.4f}ms")
    print("\nDone.")


if __name__ == '__main__':
    main()
