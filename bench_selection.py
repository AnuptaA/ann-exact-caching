#!/usr/bin/env python3
"""Benchmark selection performance at SIFT-like scale."""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.base.population import CandidateEntry
from utils.base.selection import select_candidates
from utils.base.selection_fast import select_candidates_fast

# Parameters matching real SIFT population runs
N_BASE = 50_000
N_CACHE = 1024
N_TEST = 512
DIM = 128
K = 100
BUDGET_B = 10_000
SEED = 42
N_CANDIDATES = 6000

STRATEGIES = ['greedy_naive', 'greedy_cost_static', 'greedy_cost']


def make_candidates(rng):
    """Synthesize candidates with realistic R_c sizes and sparse coverage."""
    candidates = []
    for i in range(N_CANDIDATES):
        rc_size = int(np.clip(rng.lognormal(6.5, 0.8), 50, N_BASE))
        rc_indices = set(rng.choice(N_BASE, size=rc_size, replace=False).tolist())
        n_train = int(np.clip(rng.lognormal(1.2, 0.7), 1, 30))
        covered_train = set(rng.choice(N_CACHE, size=n_train, replace=False).tolist())
        n_test = max(1, int(n_train * N_TEST / N_CACHE))
        covered_test = set(rng.choice(N_TEST, size=n_test, replace=False).tolist())

        candidates.append(CandidateEntry(
            query=rng.standard_normal(DIM).astype(np.float32),
            r_Q=rng.uniform(0.01, 0.1),
            alpha=rng.choice([1.05, 1.5, 2.0, 2.5, 3.0]),
            r_c=rng.uniform(0.02, 0.3),
            R_c=rc_indices,
            rc_size=rc_size,
            covered_train_indices=covered_train,
            covered_test_indices=covered_test,
            top_k_vectors=[rng.standard_normal(DIM).astype(np.float32) for _ in range(K)],
            top_k_indices=list(range(K)),
            half_gap=rng.uniform(0.001, 0.01),
        ))
    return candidates


def main():
    rng = np.random.default_rng(SEED)

    print(f"Constructing {N_CANDIDATES} candidates with realistic R_c sizes...")
    candidates = make_candidates(rng)

    rc_sizes = [c.rc_size for c in candidates]
    train_sizes = [len(c.covered_train_indices) for c in candidates]
    print(f"  R_c sizes: min={min(rc_sizes)}, median={int(np.median(rc_sizes))}, "
          f"max={max(rc_sizes)}")
    print(f"  covered_train sizes: min={min(train_sizes)}, "
          f"median={int(np.median(train_sizes))}, max={max(train_sizes)}")

    old_times = {}
    fast_times = {}
    old_results = {}
    fast_results = {}

    # --- OLD implementation ---
    print(f"\n{'='*60}")
    print("ORIGINAL (Python sets) implementation:")
    print(f"{'='*60}")
    for strategy in STRATEGIES:
        print(f"\n  {strategy}...", end=" ", flush=True)
        t0 = time.time()
        sel, traj = select_candidates(
            candidates, BUDGET_B, strategy,
            n_train_queries=N_CACHE, n_test_queries=N_TEST, seed=SEED,
        )
        elapsed = time.time() - t0
        old_times[strategy] = elapsed
        old_results[strategy] = (sel, traj)
        cov = traj[-1]['cig_train_coverage'] if traj else 0
        print(f"{elapsed:.1f}s  ({len(sel)} selected, {cov:.1%} coverage)")

    # --- FAST implementation ---
    print(f"\n{'='*60}")
    print("FAST (numpy vectorized) implementation:")
    print(f"{'='*60}")
    for strategy in STRATEGIES:
        print(f"\n  {strategy}...", end=" ", flush=True)
        t0 = time.time()
        sel, traj = select_candidates_fast(
            candidates, BUDGET_B, strategy,
            n_train_queries=N_CACHE, n_test_queries=N_TEST, seed=SEED,
        )
        elapsed = time.time() - t0
        fast_times[strategy] = elapsed
        fast_results[strategy] = (sel, traj)
        cov = traj[-1]['cig_train_coverage'] if traj else 0
        print(f"{elapsed:.1f}s  ({len(sel)} selected, {cov:.1%} coverage)")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("SPEEDUP SUMMARY:")
    print(f"{'='*60}")
    print(f"  {'Strategy':<25} {'Old':>8} {'Fast':>8} {'Speedup':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    for strategy in STRATEGIES:
        old_t = old_times[strategy]
        fast_t = fast_times[strategy]
        speedup = old_t / max(fast_t, 0.001)
        print(f"  {strategy:<25} {old_t:>7.1f}s {fast_t:>7.1f}s {speedup:>7.1f}x")

    # --- Correctness ---
    print(f"\nCORRECTNESS CHECK:")
    for strategy in STRATEGIES:
        sel_old, traj_old = old_results[strategy]
        sel_fast, traj_fast = fast_results[strategy]
        assert len(sel_old) == len(sel_fast), \
            f"{strategy}: count mismatch {len(sel_old)} vs {len(sel_fast)}"
        for i, (t1, t2) in enumerate(zip(traj_old, traj_fast)):
            for k in t1:
                assert abs(t1[k] - t2[k]) < 1e-9, \
                    f"{strategy} step {i} key {k}: {t1[k]} vs {t2[k]}"
    print("  All strategies match perfectly!")


if __name__ == "__main__":
    main()
