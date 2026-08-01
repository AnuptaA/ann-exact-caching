#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Query-query vs. query-data distance ratio: direct empirical test of the
hypothesized key certifiability factor (CIG inequality: D(t,Q) + r_t <= r_c
-- a candidate centered at query Q can only certify test query t cheaply if
t is much closer to Q than Q's own neighbors are).

Standalone, read-only analysis script (no select_candidates sweeps, no
benchmark preparation) -- pure pairwise-distance computation on already-
prepared benchmark data. For each cache query Q: query-query distance is
the median distance to its 5 nearest other cache queries; query-data
distance is Q's own r_t (K-th nearest base point, already stored on disk).
ratio = query-query distance / r_t. ratio << 1 means nearby queries are
much closer to each other than to their own true neighbors -- the regime
where one cached candidate's ball can plausibly cover a nearby query
without much overshoot.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent.parent))

from simulations.population import debug_common as dc
from simulations.population.debug_common import parse_benchmark_sizes

#-------------------------------------------------------------------------------

K_VALUES = [1, 5, 10]
N_NEAREST = 5
OUT_DIR = Path('simulations/debug/population_distance_ratio')

#-------------------------------------------------------------------------------


def compute_ratio(dataset: str, benchmark_name: str, K: int) -> np.ndarray:
    """Per-cache-query ratio = (median distance to N_NEAREST nearest other
    cache queries) / r_t."""
    base, cache_q, test_q, bdir = dc.load_benchmark(dataset, benchmark_name)
    n_cache = len(cache_q)
    train_r_t, _test_r_t = dc.load_gt_r_t(bdir, K, n_cache, len(test_q))

    cache_q_hat = cache_q / np.linalg.norm(cache_q, axis=1, keepdims=True)
    D = dc.pairwise_dist(cache_q_hat, cache_q_hat, dc.METRIC)
    np.fill_diagonal(D, np.inf)

    k = min(N_NEAREST, n_cache - 1)
    nearest = np.partition(D, k - 1, axis=1)[:, :k]
    query_query_dist = np.median(nearest, axis=1)

    valid = train_r_t > 0
    ratio = np.full(n_cache, np.nan, dtype=np.float64)
    ratio[valid] = query_query_dist[valid] / train_r_t[valid]
    return ratio


def plot_dataset_ratios(dataset: str, data_by_bm_k: dict, k_values: list) -> None:
    benchmarks = sorted(data_by_bm_k.keys(), key=lambda bm: parse_benchmark_sizes(bm)[0])
    n_rows, n_cols = len(benchmarks), len(k_values)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.8 * n_rows), squeeze=False)

    for r, bm in enumerate(benchmarks):
        train_n, _ = parse_benchmark_sizes(bm)
        for c, K in enumerate(k_values):
            ax = axes[r][c]
            ratio = data_by_bm_k[bm].get(K)
            if ratio is None:
                ax.axis('off')
                continue
            ratio = ratio[np.isfinite(ratio)]
            ax.boxplot(ratio, showfliers=False, tick_labels=[f'K={K}'])
            ax.axhline(1.0, linestyle=':', color='gray', linewidth=0.8, alpha=0.5)

            if r == 0:
                ax.set_title(f'$K={K}$', fontsize=10)
            if c == 0:
                ax.set_ylabel(f'{train_n} Train Queries\nquery-query / query-data dist')

    fig.suptitle(
        f'{dataset.upper()}: Query-Query vs. Query-Data Distance Ratio', fontsize=13,
    )
    plt.tight_layout()

    outdir = OUT_DIR / dataset
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / 'ratio_distribution.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")


def main() -> None:
    summary_rows = []
    all_by_dataset = {}

    for dataset in dc.DATASET_CONFIG:
        print(f"\n=== {dataset.upper()} ===")
        cfg = dc.DATASET_CONFIG[dataset]
        data_by_bm_k = {}
        dataset_ratios = []

        for benchmark_name in cfg['benchmarks']:
            bdir = cfg['base_dir'] / benchmark_name
            if not (bdir / 'cache_queries.npy').exists():
                print(f"  {benchmark_name}: not prepared, skipping")
                continue
            data_by_bm_k[benchmark_name] = {}
            for K in K_VALUES:
                ratio = compute_ratio(dataset, benchmark_name, K)
                data_by_bm_k[benchmark_name][K] = ratio
                dataset_ratios.append(ratio)

                valid = ratio[np.isfinite(ratio)]
                median_ratio = float(np.median(valid))
                frac_below_1 = float(np.mean(valid < 1.0))
                summary_rows.append((dataset, benchmark_name, K, median_ratio, frac_below_1))
                print(f"  {benchmark_name} K={K}: median ratio={median_ratio:.3f}, "
                      f"frac<1={frac_below_1:.3f}")

        plot_dataset_ratios(dataset, data_by_bm_k, K_VALUES)
        all_by_dataset[dataset] = np.concatenate(dataset_ratios) if dataset_ratios else np.array([])

    print("\n=== Summary ===")
    print(f"{'dataset':8} {'benchmark':32} {'K':>3} {'median ratio':>14} {'frac<1':>8}")
    for dataset, bm, K, median_ratio, frac_below_1 in summary_rows:
        print(f"{dataset:8} {bm:32} {K:>3} {median_ratio:>14.3f} {frac_below_1:>8.3f}")

    print("\n=== SIFT vs. ESCI (all benchmarks/K pooled) ===")
    for dataset, ratios in all_by_dataset.items():
        valid = ratios[np.isfinite(ratios)]
        if len(valid) == 0:
            continue
        print(f"  {dataset}: median={np.median(valid):.3f}, "
              f"frac<1={np.mean(valid < 1.0):.3f}, n={len(valid)}")


if __name__ == '__main__':
    main()
