#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Distance-concentration comparison: SIFT vs. ESCI (Phase 3 Round 16).

Standalone, cross-dataset diagnostic (unlike debug_ball_geometry.py/
debug_trajectory.py, which loop per-dataset): a single figure comparing
both datasets directly, one panel per train size.

Motivated by a real finding this session: ESCI's utility ratio (r_c/d_min,
"how much extra radius do you pay beyond the bare-minimum nearest-base-
vector distance") is comparable to or slightly *better* than SIFT's at
every K -- so ESCI's true near neighbors are not, relatively speaking,
any harder to reach than SIFT's. Yet ESCI's candidate spheres end up
covering nearly the entire base once K>=1 (R_c size ~40-50k out of 50k),
while SIFT's stay far smaller and more varied. The reconciliation: this
isn't about *local* clustering quality (utility ratio measures that, and
it's fine) -- it's about *global* concentration of the background
query-to-base distance distribution. ESCI's distances are far more
tightly clustered around their mean (coefficient of variation ~0.08) than
SIFT's (~0.21), so a radius calibrated with the same modest multiplicative
slack over the nearest neighbor immediately sweeps up most of the base in
ESCI, while the same relative slack still stays well short of the bulk of
SIFT's much more spread-out base.

No pipeline rerun, no changes to population_runner.py/selection.py --
reads already-prepared benchmark data via debug_common.py, same as every
other debug_*.py script.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent.parent))

from scipy.stats import gaussian_kde

from utils.base.population import _normalize
from simulations.population import debug_common as dc

#-------------------------------------------------------------------------------

METRIC = 'angular'
N_QUERY_SAMPLE = 50
SEED = 42
OUT_DIR = Path('simulations/debug/population_distance_concentration')

DATASET_COLORS = {'sift': 'tab:blue', 'esci': 'tab:orange'}
DATASET_LABELS = {'sift': 'SIFT', 'esci': 'ESCI'}

#-------------------------------------------------------------------------------


def _dist_summary(flat: np.ndarray) -> dict:
    return dict(
        mean=float(flat.mean()), std=float(flat.std()),
        cv=float(flat.std() / flat.mean()),
    )


def compute_distance_stats(base: np.ndarray, cache_q: np.ndarray) -> dict:
    """Query-to-base and query-to-query angular distance distributions for
    the same sample of cache queries -- the latter checks whether the
    concentration effect is specific to the query/base relationship or a
    general property of the embedding space (also present among queries
    themselves)."""
    rng = np.random.RandomState(SEED)
    n_sample = min(N_QUERY_SAMPLE, len(cache_q))
    idx = rng.choice(len(cache_q), n_sample, replace=False)
    sample_q = cache_q[idx]

    base_hat = _normalize(base.copy())
    q_hat = _normalize(sample_q.copy())

    qb_dots = np.clip(q_hat @ base_hat.T, -1.0, 1.0)
    qb_flat = (np.arccos(qb_dots) / np.pi).flatten()  # (n_sample, n_base)

    qq_dots = np.clip(q_hat @ q_hat.T, -1.0, 1.0)
    qq_dists = np.arccos(qq_dots) / np.pi
    qq_flat = qq_dists[np.triu_indices(len(q_hat), k=1)]  # exclude self, dedupe symmetric pairs

    return {
        'qb_background': qb_flat, 'qb_stats': _dist_summary(qb_flat),
        'qq_background': qq_flat, 'qq_stats': _dist_summary(qq_flat),
    }


def _plot_panel(ax, results_by_dataset: dict, title: str) -> None:
    x_min = min(r['qb_background'].min() for r in results_by_dataset.values())
    x_max = max(r['qb_background'].max() for r in results_by_dataset.values())
    x_grid = np.linspace(x_min, x_max, 300)

    for dataset, result in results_by_dataset.items():
        color = DATASET_COLORS[dataset]
        label = DATASET_LABELS[dataset]

        qb_s = result['qb_stats']
        qb_kde = gaussian_kde(result['qb_background'])
        ax.plot(
            x_grid, qb_kde(x_grid), color=color, linewidth=1.5, linestyle='-',
            label=f"{label} query-to-base (mean={qb_s['mean']:.3f}, std={qb_s['std']:.3f})",
        )
        ax.fill_between(x_grid, qb_kde(x_grid), alpha=0.12, color=color)

        qq_s = result['qq_stats']
        qq_kde = gaussian_kde(result['qq_background'])
        ax.plot(
            x_grid, qq_kde(x_grid), color=color, linewidth=1.5, linestyle='--',
            label=f"{label} query-to-query (mean={qq_s['mean']:.3f}, std={qq_s['std']:.3f})",
        )

    ax.set_title(title, fontsize=10)
    ax.set_xlabel('angular distance')
    ax.legend(fontsize=6.5, loc='upper left')


def main() -> None:
    train_sizes = [256, 512, 1024, 2048]
    sift_benchmarks = dict(zip(train_sizes, dc.DATASET_CONFIG['sift']['benchmarks']))
    esci_benchmarks = dict(zip(train_sizes, dc.DATASET_CONFIG['esci']['benchmarks']))

    panels_data = []
    for size in train_sizes:
        print(f"  train_size={size}")
        results_by_dataset = {}
        for dataset, benchmarks in (('sift', sift_benchmarks), ('esci', esci_benchmarks)):
            bm_name = benchmarks[size]
            bdir = dc.DATASET_CONFIG[dataset]['base_dir'] / bm_name
            if not (bdir / 'cache_queries.npy').exists():
                print(f"    {dataset}/{bm_name}: not prepared, skipping")
                continue
            base, cache_q, test_q, _bdir = dc.load_benchmark(dataset, bm_name)
            result = compute_distance_stats(base, cache_q)
            results_by_dataset[dataset] = result
            qb_s, qq_s = result['qb_stats'], result['qq_stats']
            print(
                f"    {dataset}: query-to-base mean={qb_s['mean']:.3f} std={qb_s['std']:.3f} cv={qb_s['cv']:.3f}"
                f"  |  query-to-query mean={qq_s['mean']:.3f} std={qq_s['std']:.3f} cv={qq_s['cv']:.3f}"
            )
        if results_by_dataset:
            panels_data.append((size, results_by_dataset))

    if not panels_data:
        print("No prepared benchmarks found, skipping.")
        return

    n_cols = len(panels_data)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.0 * n_cols, 4.5), squeeze=False)
    axes = axes[0]
    for ax, (size, results_by_dataset) in zip(axes, panels_data):
        _plot_panel(ax, results_by_dataset, f'{size} Train Queries')
    axes[0].set_ylabel('density')

    fig.suptitle('Distance Concentration: Query-to-Base vs. Query-to-Query, SIFT vs. ESCI', fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.93))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / 'distance_concentration.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_path}")


if __name__ == '__main__':
    main()
