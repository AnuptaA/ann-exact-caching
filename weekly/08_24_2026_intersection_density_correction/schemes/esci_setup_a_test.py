#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Setup (a) on real ESCI data: cache candidates placed uniformly at
random (decoupled from data density), test queries drawn directly from
the real query pool. Tests whether recall tracks IF closer to y=x than
the anchor-perturbation design (Scheme A/B) when candidate placement is
no longer tied to a real-corpus anchor."""

import sys
import time
import resource
import argparse
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'common'))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import numpy as np

from part2_uniform_baseline_run_experiment import _sample_uniform_sphere
from esci_synth_sweep_run_experiment import _normalize, precompute_candidates, K_VALUES, K_MAX, D, BACKGROUND_PATH
from sift_setup_a_test import evaluate_setup_a, report_and_save

#-------------------------------------------------------------------------------

QUERY_PATH = 'datasets/esci/queries.npz'
N_CANDIDATES = 2000
N_TEST_QUERIES = 8000
SEED = 42

OUT_DIR = Path(__file__).parent.parent / 'results'
PART_DIR = 'esci_setup_a'
_RUN_TIMESTAMP = None

#-------------------------------------------------------------------------------


def _run_dir() -> Path:
    global _RUN_TIMESTAMP
    if _RUN_TIMESTAMP is None:
        _RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
    return OUT_DIR / _RUN_TIMESTAMP

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
    query_data = np.load(query_path)
    all_queries = _normalize(query_data['query_vectors'].astype(np.float64))
    print(f"Loaded {len(all_queries)} real ESCI query embeddings from {query_path}")

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

    report_and_save(results, run_dir, D, PART_DIR, 'esci')

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"\nWall clock: {wall_elapsed:.2f}s ({wall_elapsed / 3600:.4f}h)")
    print(f"Peak RSS: {peak_rss} (platform units: KB on Linux, bytes on macOS)")
    print(f"Per-query cost: {wall_elapsed / len(queries) * 1000:.4f}ms")
    print("\nDone.")


if __name__ == '__main__':
    main()
