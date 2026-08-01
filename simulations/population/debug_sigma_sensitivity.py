#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Sigma-sensitivity diagnostic (Phase 3 Round 14, plan Step D).

Distinct from debug_ball_geometry.py (Step B): B asks how big candidate
spheres are (a proposal-time geometry question). This asks, for queries that
get certified, how many base vectors actually land in the certifying entry's
result set -- a simulate-time certification-count question, unrelated to
radii or coverage. Explains whether sigma has room to matter given how
certification counts are distributed relative to the ceil(sigma*K)
thresholds.

Entirely separate from population_runner.py -- shares setup with
debug_trajectory.py via debug_common.py.
"""

import sys
import math
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.base.population import _normalize
from simulations.population import debug_common as dc

#-------------------------------------------------------------------------------

K_VALUES = [1, 5, 10]
SIGMA_THRESHOLDS = [0.7, 0.8, 0.9, 1.0]
OUT_DIR = Path('simulations/debug/population_sigma_sensitivity')

#-------------------------------------------------------------------------------


def certified_counts(
    selected: list, queries: np.ndarray, K: int,
    base: np.ndarray, base_hat: np.ndarray, metric: str,
) -> np.ndarray:
    """Best (max) certified-vector count across all reachable entries, for
    every query -- the quantity that would ultimately be compared against
    ceil(sigma*K) for any sigma value, since certification count doesn't
    depend on sigma, only the threshold comparison at simulate time does.

    Replicates lemma1_no_union_sigma's O1/O2 gate math (see
    monotonic_hit_rate_pass in debug_trajectory.py for the same pattern and
    its floating-point-boundary caveat for self-referential candidates),
    but precomputes the query-to-base distance matrix ONCE for all queries
    up front (O(n_queries * n_base)) instead of once per (query, entry)
    pair -- calling lemma1_no_union_sigma directly per entry would redo
    that O(n_base) work on every single entry it reaches, which is what
    made an earlier version of this function pay full-base-distance cost
    proportional to n_queries * n_entries_checked instead of n_queries.
    """
    n_queries = len(queries)
    query_hat = _normalize(queries.copy().astype(np.float64))
    base_hat = base_hat.astype(np.float64)
    query_base_dist = dc.pairwise_dist(query_hat, base_hat, metric)

    counts = np.zeros(n_queries, dtype=np.int64)
    entries = [c.to_cache_entry() for c in selected]

    remaining = np.arange(n_queries)
    for entry in entries:
        if len(remaining) == 0:
            break
        entry_query = entry.query.astype(np.float64)
        entry_q_hat = entry_query / np.linalg.norm(entry_query)
        r_Q = entry.get_radius()

        d_Q = dc.pairwise_dist(query_hat[remaining], entry_q_hat[None, :], metric)[:, 0]
        gate = d_Q <= r_Q
        if not np.any(gate):
            continue

        gated_idx = remaining[gate]
        threshold_dist = r_Q - d_Q[gate]
        sub = query_base_dist[np.ix_(gated_idx, entry.top_k_indices)]
        passing_counts = (sub <= threshold_dist[:, None]).sum(axis=1)

        counts[gated_idx] = np.maximum(counts[gated_idx], passing_counts)
        done = gated_idx[counts[gated_idx] >= K]
        if len(done):
            remaining = remaining[~np.isin(remaining, done)]

    return counts


#-------------------------------------------------------------------------------


def run_combo(dataset: str, benchmark_name: str, K: int) -> dict:
    base, cache_q, test_q, bdir = dc.load_benchmark(dataset, benchmark_name)
    n_cache, n_test = len(cache_q), len(test_q)
    train_r_t, test_r_t = dc.load_gt_r_t(bdir, K, n_cache, n_test)

    selected, _trajectory = dc.propose_and_select_full_budget(
        base, cache_q, test_q, train_r_t, test_r_t, K,
    )
    if not selected:
        print(f"    {dataset}/{benchmark_name} K={K}: no candidates selected, skipping")
        return None

    base_hat = _normalize(base.copy())
    counts = certified_counts(selected, test_q, K, base, base_hat, dc.METRIC)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(counts, bins=range(0, K + 2), align='left', color='tab:blue', alpha=0.7)
    for sigma in SIGMA_THRESHOLDS:
        threshold = math.ceil(sigma * K)
        ax.axvline(threshold, linestyle='--', label=f'sigma={sigma} (>={threshold})')
    ax.set_xlabel('certified count (best entry)')
    ax.set_ylabel('# test queries')
    ax.set_title(f'{dataset}/{benchmark_name} K={K}: certified-count distribution')
    ax.legend(fontsize=7)
    plt.tight_layout()

    outdir = OUT_DIR / dataset / benchmark_name
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f'certified_counts_K{K}.png'
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"    wrote {out_path}")

    return {
        'n_queries': int(n_test),
        'mean_certified': float(counts.mean()) if n_test else 0.0,
        'median_certified': float(np.median(counts)) if n_test else 0.0,
        'counts_at_threshold': {
            f'sigma={s}': int((counts >= math.ceil(s * K)).sum()) for s in SIGMA_THRESHOLDS
        },
    }


def main() -> None:
    summary = {}
    for dataset, cfg in dc.DATASET_CONFIG.items():
        print(f"\n=== {dataset.upper()} ===")
        summary[dataset] = {}
        for benchmark_name in cfg['benchmarks']:
            bdir = cfg['base_dir'] / benchmark_name
            if not (bdir / 'cache_queries.npy').exists():
                print(f"  {benchmark_name}: not prepared, skipping")
                continue
            print(f"  {benchmark_name}")
            summary[dataset][benchmark_name] = {}
            for K in K_VALUES:
                result = run_combo(dataset, benchmark_name, K)
                if result is not None:
                    summary[dataset][benchmark_name][f'K={K}'] = result

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. Summary: {OUT_DIR / 'summary.json'}")


if __name__ == '__main__':
    main()
