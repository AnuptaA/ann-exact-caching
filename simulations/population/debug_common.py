#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Shared setup for the Phase 3 Round 14 trajectory/sigma-sensitivity debug
scripts (debug_trajectory.py, debug_sigma_sensitivity.py). Entirely separate
from population_runner.py/selection.py -- calls propose_candidates_ball and
select_candidates directly against already-prepared benchmark data, at
budget=100% (full base size, i.e. no memory constraint), matching K=K_cache=
K_test convention used throughout the real pipeline (run_population.py).
"""

import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datasets.dataloaders import load_esci_benchmark, load_sift_benchmark
from utils.base.population import propose_candidates_ball, precompute_ball_invariants
from utils.base.selection import select_candidates

#-------------------------------------------------------------------------------

METRIC = 'angular'
SEED = 42
N_MAX = 20
STRATEGY = 'greedy_cost'

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

#-------------------------------------------------------------------------------


def load_gt_r_t(bdir: Path, K: int, n_cache: int, n_test: int) -> tuple:
    """(train_r_t, test_r_t) for a given K, matching population_runner.py's
    k_test_override=0 branch: K=0 means pure containment, both zeroed."""
    if K == 0:
        return (
            np.zeros(n_cache, dtype=np.float32),
            np.zeros(n_test, dtype=np.float32),
        )
    cache_gt = np.load(bdir / f'cache_gt_population_K{K}.npz')
    test_gt = np.load(bdir / f'test_gt_population_K{K}.npz')
    train_r_t = np.array(
        [float(cache_gt[f'r_Q_{i}']) for i in range(n_cache)], dtype=np.float32
    )
    test_r_t = np.array(
        [float(test_gt[f'r_Q_{i}']) for i in range(n_test)], dtype=np.float32
    )
    return train_r_t, test_r_t


def load_gt_indices(bdir: Path, K: int, n: int) -> list:
    """Test queries' true top-K neighbor index lists for a given K, matching
    population_runner.py's _load_gt_indices. K=0 has no GT file (containment
    mode has no neighbor requirement); callers needing K=0 semantics should
    branch before calling this."""
    data = np.load(bdir / f'test_gt_population_K{K}.npz')
    return [data[f'indices_{i}'].tolist() for i in range(n)]


def propose_full_pool(
    base: np.ndarray, cache_q: np.ndarray, test_q: np.ndarray,
    train_r_t: np.ndarray, test_r_t: np.ndarray, K: int,
    n_max: int = N_MAX, metric: str = METRIC, invariants: dict = None,
) -> list:
    """Just the propose step -- budget-independent, so callers that need to
    sweep select_candidates over multiple real budget caps against the same
    candidate pool (e.g. debug_vector_coverage.py) can call this once and
    reuse the result, instead of re-proposing per budget value.

    top_k_vectors/top_k_indices/half_gap are unused by the debug scripts
    (they build cache entries via CandidateEntry.to_cache_entry(), which
    only needs R_c/r_c/query), so dummy placeholders are passed to
    propose_candidates_ball for the observed_top_k_* args.

    invariants: optional precomputed K-independent quantities (see
    precompute_ball_invariants_for_benchmark below), reused across a
    K-sweep for the same benchmark. Default None recomputes everything
    per call, unchanged from before.
    """
    n_cache = len(cache_q)
    tv_dummy = [[] for _ in range(n_cache)]
    ti_dummy = [[] for _ in range(n_cache)]
    hg_dummy = np.zeros(n_cache, dtype=np.float32)

    return propose_candidates_ball(
        cache_q, train_r_t, tv_dummy, ti_dummy, hg_dummy,
        cache_q, train_r_t, test_q, test_r_t,
        base, metric, n_max=n_max, K=K, store_rc=True,
        invariants=invariants,
    )


def propose_and_select_full_budget(
    base: np.ndarray, cache_q: np.ndarray, test_q: np.ndarray,
    train_r_t: np.ndarray, test_r_t: np.ndarray, K: int,
    n_max: int = N_MAX, seed: int = SEED, metric: str = METRIC,
    invariants: dict = None,
) -> tuple:
    """Propose ball candidates and greedily select at budget=100% (=n_base,
    i.e. effectively unconstrained). Returns (selected, trajectory)."""
    candidates = propose_full_pool(
        base, cache_q, test_q, train_r_t, test_r_t, K,
        n_max=n_max, metric=metric, invariants=invariants,
    )

    n_base = len(base)
    selected, trajectory = select_candidates(
        candidates, budget_B=n_base, strategy=STRATEGY,
        n_train_queries=len(cache_q), n_test_queries=len(test_q), seed=seed,
    )
    return selected, trajectory


def precompute_ball_invariants_for_benchmark(
    base: np.ndarray, cache_q: np.ndarray, test_q: np.ndarray, metric: str = METRIC,
) -> dict:
    """Thin wrapper: cache_q doubles as both observed_queries and
    train_queries (matching every real call site's convention -- see
    utils/base/population.py's precompute_ball_invariants docstring for
    why this reuse is valid), so this is the exact invariants payload
    every K value in a K-sweep for one benchmark can share."""
    return precompute_ball_invariants(cache_q, cache_q, test_q, base, metric)


def pairwise_dist(a_hat: np.ndarray, b_hat: np.ndarray, metric: str) -> np.ndarray:
    """Distance matrix between two sets of pre-normalized vectors, matching
    utils/base/lemmas.py's _cig_distances formula exactly (angular: arccos
    of the dot product; euclidean-on-normalized: sqrt(2 - 2*dot))."""
    dots = np.clip(a_hat @ b_hat.T, -1.0, 1.0)
    if metric == "angular":
        return np.arccos(dots) / np.pi
    return np.sqrt(np.maximum(2.0 - 2.0 * dots, 0.0))


def parse_benchmark_sizes(benchmark_name: str) -> tuple:
    """(train_n, test_n) from a benchmark name, e.g.
    'sift_b50k_c256k100_t512n20_s42' -> (256, 512)."""
    m = re.search(r'_c(\d+)k\d+_t(\d+)n\d+_s\d+', benchmark_name)
    return int(m.group(1)), int(m.group(2))


def load_benchmark(dataset: str, benchmark_name: str) -> tuple:
    """(base, cache_q, test_q, bdir) for a prepared benchmark."""
    cfg = DATASET_CONFIG[dataset]
    bdir = cfg['base_dir'] / benchmark_name
    base, cache_q, test_q, _config = cfg['load_fn'](str(bdir))
    return base, cache_q, test_q, bdir

#-------------------------------------------------------------------------------
# Shared plotting helpers (moved from debug_ball_geometry.py so
# debug_trajectory.py can reuse them for selected-candidate plots without
# debug_ball_geometry.py's own lightweight n_max=1/no-selection candidate
# generation having to duplicate a full n_max=20+selection pass just to
# get access to this plotting code).
#-------------------------------------------------------------------------------


def _boxplot_panel(ax, data_by_label: dict, title: str, ylabel: str, log_y: bool = False) -> None:
    labels = list(data_by_label.keys())
    values = [data_by_label[k] for k in labels]
    ax.boxplot(values, tick_labels=labels, showfliers=False)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale('log')
    ax.tick_params(axis='x', rotation=30)


def _boxplot_grid(panels: list, suptitle: str, out_path: Path) -> None:
    """panels: list of (data_by_label, title, ylabel, log_y) tuples, one
    per subplot (one per cache size), laid out in a single row."""
    fig, axes = plt.subplots(1, len(panels), figsize=(max(6, len(panels) * 5.5), 5), squeeze=False)
    axes = axes[0]
    for ax, (data_by_label, title, ylabel, log_y) in zip(axes, panels):
        _boxplot_panel(ax, data_by_label, title, ylabel, log_y)
    fig.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _density_panel(ax, data_by_label: dict, title: str, xlabel: str) -> None:
    """Overlapping KDE curves, one per label (K value), on the same axes --
    complements _boxplot_panel's per-K summary with the full distribution
    shape (multimodality, skew, tail weight aren't visible in a boxplot)."""
    from scipy.stats import gaussian_kde

    for label, values in data_by_label.items():
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if len(values) < 2 or np.ptp(values) < 1e-12:
            continue
        kde = gaussian_kde(values)
        x_grid = np.linspace(values.min(), values.max(), 200)
        ax.plot(x_grid, kde(x_grid), label=label, linewidth=1.5)
        ax.fill_between(x_grid, kde(x_grid), alpha=0.15)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('density')
    ax.legend(fontsize=7)


def _density_grid(panels: list, suptitle: str, out_path: Path) -> None:
    """panels: list of (data_by_label, title, xlabel) tuples, one per
    subplot (one per cache size), laid out in a single row -- same grid
    convention as _boxplot_grid."""
    fig, axes = plt.subplots(1, len(panels), figsize=(max(6, len(panels) * 5.5), 5), squeeze=False)
    axes = axes[0]
    for ax, (data_by_label, title, xlabel) in zip(axes, panels):
        _density_panel(ax, data_by_label, title, xlabel)
    fig.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")
