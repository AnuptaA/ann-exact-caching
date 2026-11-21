#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Ball-proposal geometry diagnostics.

Does K reasonably change candidate-sphere geometry (curse-of-dimensionality
check) + utility-ratio distribution (K=0 vs K>0) + sanity checks

"""

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.append(str(Path(__file__).parent.parent.parent))

from datasets.dataloaders import load_esci_benchmark, load_sift_benchmark
from utils.base.population import (
    _normalize, _precompute_kth_base_dist, _pairwise_cache_dists, propose_candidates_ball,
)
from simulations.population.debug_common import (
    parse_benchmark_sizes, precompute_ball_invariants_for_benchmark,
    pairwise_dist, _boxplot_grid, _density_grid,
)

#-------------------------------------------------------------------------------

METRIC = 'angular'
K_VALUES = [1, 5, 10, 20]
UTILITY_K_VALUES = [0, 1, 5, 10]
N_MAX_FOR_RC = 1  # single (nearest-neighbor) sphere per query, maybe should change

DATASET_CONFIG = {
    'sift': dict(
        load_fn=load_sift_benchmark,
        base_dir=Path('datasets/sift'),
        benchmarks=[
            'sift_b50k_c256k100_t512n20_s42',
            'sift_b50k_c512k100_t512n20_s42',
            'sift_b50k_c1024k100_t512n20_s42',
            'sift_b50k_c2048k100_t512n20_s42',
        ],
    ),
    'esci': dict(
        load_fn=load_esci_benchmark,
        base_dir=Path('datasets/esci'),
        benchmarks=[
            'esci_c256k100_t512n20_s42',
            'esci_c512k100_t512n20_s42',
            'esci_c1024k100_t512n20_s42',
            'esci_c2048k100_t512n20_s42',
        ],
    ),
}

OUT_DIR = Path('simulations/debug/population_geometry')

#-------------------------------------------------------------------------------


def _load_gt_r_Q(path: Path, n: int) -> np.ndarray:
    """K-th nearest neighbor distance for each of n queries, from a
    *_gt_population_K{k}.npz file's r_Q_{i} fields."""
    data = np.load(path)
    return np.array([float(data[f'r_Q_{i}']) for i in range(n)], dtype=np.float64)


def compute_margins(cache_q: np.ndarray, base: np.ndarray, K: int) -> np.ndarray:
    """margin_lookup: K-th nearest base distance for each cache query."""
    cache_hat = _normalize(cache_q.copy())
    base_hat = _normalize(base.copy())
    return _precompute_kth_base_dist(cache_hat, base_hat, K, METRIC).astype(np.float64)


def compute_utility_ratio(
    cache_q: np.ndarray, base: np.ndarray, K: int,
    margin_lookup: np.ndarray = None, invariants: dict = None,
) -> np.ndarray:
    """Ratio of the radius needed to reach a query's nearest cache-neighbor's
    K-sphere, to the distance from the query to its own nearest base vector.
    """
    if invariants is not None:
        sorted_dists = invariants['sorted_cache_dists']
        sorted_indices = invariants['sorted_cache_indices']
        all_dists = invariants['all_dists']
    else:
        sorted_dists, sorted_indices = _pairwise_cache_dists(cache_q, METRIC)
        cache_hat = _normalize(cache_q.copy())
        base_hat = _normalize(base.copy())
        all_dists = pairwise_dist(cache_hat, base_hat, METRIC)

    nn_dist = sorted_dists[:, 0].astype(np.float64)
    nn_index = sorted_indices[:, 0]

    if K == 0:
        margin = np.zeros(len(cache_q), dtype=np.float64)
    else:
        if margin_lookup is None:
            cache_hat = _normalize(cache_q.copy())
            base_hat = _normalize(base.copy())
            margin_lookup = _precompute_kth_base_dist(cache_hat, base_hat, K, METRIC)
        margin = margin_lookup[nn_index].astype(np.float64)

    r_c = nn_dist + margin
    d_min = np.asarray(all_dists).min(axis=1).astype(np.float64)

    return r_c / d_min


def compute_rc_size_and_self_exclusion(
    cache_q: np.ndarray, base: np.ndarray, K: int,
    obs_r_Q: np.ndarray, test_q: np.ndarray, test_r_t: np.ndarray,
    invariants: dict = None,
) -> dict:
    """One propose_candidates_ball(K, n_max=1) call: nearest-neighbor sphere
    per cache query. Returns rc sizes and a self-exclusion violation count.
    """
    n_cache = len(cache_q)
    tv_dummy = [[] for _ in range(n_cache)]
    ti_dummy = [[] for _ in range(n_cache)]
    hg_dummy = np.zeros(n_cache, dtype=np.float32)

    candidates = propose_candidates_ball(
        cache_q, obs_r_Q, tv_dummy, ti_dummy, hg_dummy,
        cache_q, obs_r_Q, test_q, test_r_t,
        base, METRIC, n_max=N_MAX_FOR_RC, K=K, store_rc=False,
        invariants=invariants,
    )

    rc_sizes = np.array([c.rc_size for c in candidates], dtype=np.int64)

    self_exclusion_violations = 0
    for c in candidates:
        src_idx = np.flatnonzero((cache_q == c.query).all(axis=1))
        if len(src_idx) == 0:
            continue
        if c.covered_train_indices[src_idx[0]]:
            self_exclusion_violations += 1

    return {
        'rc_sizes': rc_sizes,
        'self_exclusion_violations': self_exclusion_violations,
        'n_candidates': len(candidates),
    }


def _grouped_boxplot_grid(
    panels: list, suptitle: str, ylabel: str, out_path: Path, log_y: bool = False,
) -> None:
    """panels: list of (train_by_k, test_by_k, title) tuples, one per
    subplot (one per cache size). Each subplot shows train/test as a
    paired boxplot per K value (offset positions, distinct fill color)."""
    fig, axes = plt.subplots(1, len(panels), figsize=(max(6, len(panels) * 5.5), 5), squeeze=False)
    axes = axes[0]
    for ax, (train_by_k, test_by_k, title) in zip(axes, panels):
        k_labels = list(train_by_k.keys())
        n = len(k_labels)
        pos_train = np.arange(n) * 2.0 - 0.35
        pos_test = np.arange(n) * 2.0 + 0.35
        train_vals = [train_by_k[k] for k in k_labels]
        test_vals = [test_by_k.get(k, []) for k in k_labels]

        bt = ax.boxplot(train_vals, positions=pos_train, widths=0.6,
                         patch_artist=True, showfliers=False)
        bs = ax.boxplot(test_vals, positions=pos_test, widths=0.6,
                         patch_artist=True, showfliers=False)
        for patch in bt['boxes']:
            patch.set_facecolor('tab:blue')
        for patch in bs['boxes']:
            patch.set_facecolor('tab:orange')

        ax.set_xticks(np.arange(n) * 2.0)
        ax.set_xticklabels(k_labels, rotation=30)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        if log_y:
            ax.set_yscale('log')
        if ax is axes[0]:
            ax.legend(
                handles=[Patch(facecolor='tab:blue', label='train'),
                         Patch(facecolor='tab:orange', label='test')],
                fontsize=8,
            )

    fig.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")


#-------------------------------------------------------------------------------


def run_dataset(dataset: str, cfg: dict) -> dict:
    print(f"\n=== {dataset.upper()} ===")
    summary = {'benchmarks': {}}

    margin_panels, rc_panels, utility_panels, rt_panels = [], [], [], []
    test_n_seen = None

    for benchmark_name in cfg['benchmarks']:
        bdir = cfg['base_dir'] / benchmark_name
        if not (bdir / 'cache_queries.npy').exists():
            print(f"  {benchmark_name}: not prepared, skipping")
            continue
        print(f"  {benchmark_name}")
        train_n, test_n = parse_benchmark_sizes(benchmark_name)
        test_n_seen = test_n_seen or test_n

        base, cache_q, test_q, _config = cfg['load_fn'](str(bdir))
        n_cache = len(cache_q)
        n_test = len(test_q)

        # K-independent proposal invariants computed once per benchmark and
        # reused across the K-loop below
        invariants = precompute_ball_invariants_for_benchmark(base, cache_q, test_q, metric=METRIC)

        margin_by_k = {}
        rc_by_k = {}
        rt_train_by_k = {}
        rt_test_by_k = {}
        utility_by_k = {}
        self_excl = {}

        for K in K_VALUES:
            cache_gt_path = bdir / f'cache_gt_population_K{K}.npz'
            test_gt_path = bdir / f'test_gt_population_K{K}.npz'
            train_r_t = (
                _load_gt_r_Q(cache_gt_path, n_cache) if cache_gt_path.exists() else None
            )
            test_r_t = (
                _load_gt_r_Q(test_gt_path, n_test) if test_gt_path.exists() else None
            )
            if train_r_t is not None:
                rt_train_by_k[f'$K={K}$'] = train_r_t
            if test_r_t is not None:
                rt_test_by_k[f'$K={K}$'] = test_r_t

            # Reuse the GT-loaded train_r_t as margin_lookup. Falls back to
            # compute_margins only if the per-K GT file is missing for this benchmark.
            margin_lookup = (
                train_r_t.astype(np.float64) if train_r_t is not None
                else compute_margins(cache_q, base, K)
            )
            margin_by_k[f'$K={K}$'] = margin_lookup
            utility_by_k[f'$K={K}$'] = compute_utility_ratio(
                cache_q, base, K, margin_lookup=margin_lookup, invariants=invariants,
            )

            if train_r_t is None or test_r_t is None:
                continue
            result = compute_rc_size_and_self_exclusion(
                cache_q, base, K, train_r_t.astype(np.float32), test_q,
                test_r_t.astype(np.float32), invariants=invariants,
            )
            rc_by_k[f'$K={K}$'] = result['rc_sizes']
            self_excl[f'K={K}'] = result['self_exclusion_violations']

        # K=0: pure containment, train_r_t/test_r_t both zero.
        zero_train_r_t = np.zeros(n_cache, dtype=np.float32)
        zero_test_r_t = np.zeros(n_test, dtype=np.float32)
        result0 = compute_rc_size_and_self_exclusion(
            cache_q, base, 0, zero_train_r_t, test_q, zero_test_r_t, invariants=invariants,
        )
        rc_by_k['$K=0$'] = result0['rc_sizes']
        self_excl['K=0'] = result0['self_exclusion_violations']
        utility_by_k['$K=0$'] = compute_utility_ratio(cache_q, base, 0, invariants=invariants)

        rc_ordered = {k: rc_by_k[k] for k in ['$K=0$'] + [f'$K={K}$' for K in K_VALUES] if k in rc_by_k}
        utility_ordered = {
            k: utility_by_k[k] for k in ['$K=0$'] + [f'$K={K}$' for K in UTILITY_K_VALUES if K != 0]
            if k in utility_by_k
        }

        label = f'{train_n} Train Queries'
        margin_panels.append((margin_by_k, label, 'distance to Kth nearest base vector', True))
        rc_panels.append((rc_ordered, label, r'$R_c$ size (# base vectors)', True))
        utility_panels.append((utility_ordered, label, 'utility ratio', False))
        rt_panels.append((rt_train_by_k, rt_test_by_k, label))

        print(f"    self-exclusion violations by K: {self_excl}")

        summary['benchmarks'][benchmark_name] = {
            'n_cache': n_cache,
            'n_test': n_test,
            'self_exclusion_violations': self_excl,
            'margin_median_by_k': {k: float(np.median(v)) for k, v in margin_by_k.items()},
            'rc_size_median_by_k': {k: float(np.median(v)) for k, v in rc_by_k.items()},
            'utility_ratio_median_by_k': {k: float(np.median(v)) for k, v in utility_by_k.items()},
        }

    if margin_panels:
        test_note = f' ({test_n_seen} test queries)' if test_n_seen else ''
        _boxplot_grid(
            margin_panels,
            f'{dataset.upper()}: Distance to Kth Nearest Base Vector vs. $K${test_note}',
            OUT_DIR / dataset / 'margin_vs_k.png',
        )
        _boxplot_grid(
            rc_panels,
            f'{dataset.upper()}: $R_c$ Size vs. $K${test_note}',
            OUT_DIR / dataset / 'rc_size_vs_k.png',
        )
        _boxplot_grid(
            utility_panels,
            f'{dataset.upper()}: Utility Ratio vs. $K${test_note}',
            OUT_DIR / dataset / 'utility_ratio_vs_k.png',
        )
        _density_grid(
            [(data_by_label, title, ylabel) for data_by_label, title, ylabel, _ in utility_panels],
            f'{dataset.upper()}: Utility Ratio Density vs. $K${test_note}',
            OUT_DIR / dataset / 'utility_ratio_density_vs_k.png',
        )
        _grouped_boxplot_grid(
            rt_panels,
            f'{dataset.upper()}: Distance to Kth Nearest Base Vector vs. $K${test_note}',
            'distance to Kth nearest base vector',
            OUT_DIR / dataset / 'rt_vs_k.png',
            log_y=True,
        )

    return summary


def main() -> None:
    summary = {}
    for dataset, cfg in DATASET_CONFIG.items():
        summary[dataset] = run_dataset(dataset, cfg)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. Summary: {OUT_DIR / 'summary.json'}")

    total_violations = sum(
        v for ds in summary.values() for bm in ds['benchmarks'].values()
        for v in bm['self_exclusion_violations'].values()
    )
    if total_violations:
        print(f"WARNING: {total_violations} self-exclusion violations found across all benchmarks/K")
    else:
        print("Self-exclusion check: OK (0 violations)")


if __name__ == '__main__':
    main()
