#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Part 3: plots for the 2D/3D synthetic intersection-fraction
calibration study.

Each K's data has two parts: 'selection' (per test query, per method,
the winning candidate's own fraction/recall) and 'pairwise' (a raw subsample
of the full test_q x cache_q cross product before any argmax, used for 
diagnostics.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from primitives import exact_intersection_fraction, solid_ball_intersection_fraction, chord_distance_from_angular

#-------------------------------------------------------------------------------

RESULTS_BASE = Path(__file__).parent / 'results'
OUT_BASE = Path(__file__).parent / 'plots'
DIMS = [2, 3]
K_VALUES = [1, 5, 10, 25, 50, 100]
METHOD_COLORS = {'cap': 'tab:green', 'solid': 'tab:orange'}
METHOD_LABELS = {'cap': 'cap intersection', 'solid': 'solid-ball intersection'}
N_BINS = 20

#-------------------------------------------------------------------------------

def latest_run_dir(base: Path) -> Path:
    runs = sorted(p for p in base.glob('*') if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"no run directories under {base}")
    return runs[-1]

def load_dimension(run_dir: Path, d: int, k_values: list) -> dict:
    out = {}
    for K in k_values:
        out[K] = {}
        for part in ('selection', 'pairwise'):
            path = run_dir / f'{d}d' / 'raw' / f'K{K}_{part}.npz'
            arr = np.load(path)
            out[K][part] = {key: arr[key] for key in arr.files}
    return out


def _bin_stats(x: np.ndarray, y: np.ndarray, n_bins: int = N_BINS):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    means = np.full(n_bins, np.nan)
    sems = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    for b in range(n_bins):
        mask = idx == b
        counts[b] = mask.sum()
        if counts[b] > 0:
            means[b] = y[mask].mean()
            sems[b] = y[mask].std(ddof=1) / np.sqrt(counts[b]) if counts[b] > 1 else 0.0
    return centers, means, sems, counts


def _grid(n_cols: int, n_rows: int = 1, figsize_scale=(3.6, 3.4)):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_scale[0] * n_cols, figsize_scale[1] * n_rows),
                              squeeze=False)
    return fig, axes

#-------------------------------------------------------------------------------
# 1. Binned calibration plot (+ per-background overlay variant), on the
# selected argmax candidate per method
#-------------------------------------------------------------------------------

def plot_binned_calibration(d: int, data: dict, out_dir: Path) -> None:
    fig, axes = _grid(len(K_VALUES))
    for c, K in enumerate(K_VALUES):
        ax = axes[0][c]
        sel = data[K]['selection']
        for method in ('cap', 'solid'):
            centers, means, sems, counts = _bin_stats(sel[f'{method}_fraction'], sel[f'{method}_recall'])
            valid = counts > 0
            ax.errorbar(centers[valid], means[valid], yerr=sems[valid], marker='o', markersize=3,
                        linestyle='-', color=METHOD_COLORS[method], label=METHOD_LABELS[method], capsize=2)
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1, label='y=x')
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f'$K={K}$', fontsize=10)
        ax.set_xlabel('intersection fraction (best candidate)')
        if c == 0:
            ax.set_ylabel('mean actual recall')
            ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle(f'{d}D: Binned Calibration (best-candidate intersection fraction vs. actual recall, pooled across backgrounds)')
    plt.tight_layout()
    out_path = out_dir / f'{d}d_binned_calibration.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")


def plot_binned_calibration_per_background(d: int, data: dict, out_dir: Path) -> None:
    fig, axes = _grid(len(K_VALUES))
    for c, K in enumerate(K_VALUES):
        ax = axes[0][c]
        sel = data[K]['selection']
        bg_ids = np.unique(sel['background_id'])
        colors = cm.viridis(np.linspace(0, 1, len(bg_ids)))
        for bg_id, color in zip(bg_ids, colors):
            mask = sel['background_id'] == bg_id
            centers, means, sems, counts = _bin_stats(sel['cap_fraction'][mask], sel['cap_recall'][mask])
            valid = counts > 0
            ax.plot(centers[valid], means[valid], marker='o', markersize=2, linewidth=1,
                    color=color, label=f'bg {bg_id}')
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f'$K={K}$', fontsize=10)
        ax.set_xlabel('intersection fraction (best candidate)')
        if c == 0:
            ax.set_ylabel('mean actual recall')
            ax.legend(fontsize=6)
        ax.grid(alpha=0.3)
    fig.suptitle(f'{d}D: Binned Calibration by Background (cap method, background-to-background variance check)')
    plt.tight_layout()
    out_path = out_dir / f'{d}d_binned_calibration_per_background.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")

def plot_calibration_scatter(d: int, data: dict, out_dir: Path) -> None:
    """Unbinned companion to plot_binned_calibration: same K-column
    grid, both methods overlaid with raw (jittered) points."""
    fig, axes = _grid(len(K_VALUES))
    rng = np.random.default_rng(0)
    for c, K in enumerate(K_VALUES):
        ax = axes[0][c]
        sel = data[K]['selection']
        for method in ('cap', 'solid'):
            fraction = sel[f'{method}_fraction']
            jitter = rng.uniform(-0.015, 0.015, size=len(fraction))
            ax.scatter(fraction, sel[f'{method}_recall'] + jitter, s=4, alpha=0.25,
                      color=METHOD_COLORS[method], label=METHOD_LABELS[method], linewidths=0)
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1, label='y=x')
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f'$K={K}$', fontsize=10)
        ax.set_xlabel('intersection fraction (best candidate)')
        if c == 0:
            ax.set_ylabel('actual recall (jittered)')
            ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle(f'{d}D: Calibration Scatter, Unbinned (best-candidate intersection fraction vs. actual recall)')
    plt.tight_layout()
    out_path = out_dir / f'{d}d_calibration_scatter.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")

#-------------------------------------------------------------------------------
# 2. Point-level scatter (predicted vs. actual, colored by r_cache), on
# the selected argmax candidate per method
#-------------------------------------------------------------------------------

def plot_point_scatter(d: int, data: dict, out_dir: Path) -> None:
    for method in ('cap', 'solid'):
        fig, axes = _grid(len(K_VALUES))
        for c, K in enumerate(K_VALUES):
            ax = axes[0][c]
            sel = data[K]['selection']
            rng = np.random.default_rng(0)
            jitter = rng.uniform(-0.015, 0.015, size=len(sel[f'{method}_recall']))
            sc = ax.scatter(sel[f'{method}_fraction'], sel[f'{method}_recall'] + jitter,
                            c=sel[f'{method}_r_cache'], cmap='plasma', s=4, alpha=0.3, linewidths=0)
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f'$K={K}$', fontsize=10)
            ax.set_xlabel('intersection fraction (best candidate)')
            if c == 0:
                ax.set_ylabel('actual recall (jittered)')
            if c == len(K_VALUES) - 1:
                plt.colorbar(sc, ax=ax, label='r_cache (rad)')
            ax.grid(alpha=0.3)
        fig.suptitle(f'{d}D: Point-Level Scatter, {METHOD_LABELS[method]} (best candidate, colored by r_cache)')
        plt.tight_layout()
        out_path = out_dir / f'{d}d_point_scatter_{method}.png'
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"    wrote {out_path}")

#-------------------------------------------------------------------------------
# 3. Solid-ball vs. cap fraction scatter (raw pairwise before selection)
#-------------------------------------------------------------------------------

def plot_method_comparison_scatter(d: int, data: dict, out_dir: Path) -> None:
    fig, axes = _grid(len(K_VALUES))
    for c, K in enumerate(K_VALUES):
        ax = axes[0][c]
        pw = data[K]['pairwise']
        sc = ax.scatter(pw['cap_fraction'], pw['solid_fraction'], c=pw['r_cache'], cmap='plasma',
                        s=4, alpha=0.3, linewidths=0)
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1, label='y=x')
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f'$K={K}$', fontsize=10)
        ax.set_xlabel('cap fraction')
        if c == 0:
            ax.set_ylabel('solid-ball fraction')
        if c == len(K_VALUES) - 1:
            plt.colorbar(sc, ax=ax, label='r_cache (rad)')
        ax.grid(alpha=0.3)
    fig.suptitle(f'{d}D: Solid-Ball vs. Cap Fraction (raw pairwise, before selection, colored by r_cache)')
    plt.tight_layout()
    out_path = out_dir / f'{d}d_method_comparison_scatter.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")

#-------------------------------------------------------------------------------
# 4. Discrepancy vs. angle
#-------------------------------------------------------------------------------

PERCENTILES = [5, 25, 50, 75, 95, 99]


def _widest_range_r_cache(data: dict) -> np.ndarray:
    """r_cache (pairwise sample) for whichever K has the widest
    max-min range."""
    best_K, best_range = None, -1.0
    for K, parts in data.items():
        r = parts['pairwise']['r_cache']
        r_range = float(r.max() - r.min())
        if r_range > best_range:
            best_K, best_range = K, r_range
    return data[best_K]['pairwise']['r_cache']


def _discrepancy_curve(r: float, d: int, ratios: np.ndarray) -> tuple:
    """cap_fraction - solid_fraction at equal radii r_cache=r_test=r,
    swept over delta = ratio*r for ratio in ratios. Pure function of
    (r, d), no sampled data involved."""
    deltas = ratios * r
    diffs = np.array([
        exact_intersection_fraction(r, r, delta, d)
        - solid_ball_intersection_fraction(
            chord_distance_from_angular(r), chord_distance_from_angular(r), chord_distance_from_angular(delta), d,
        )
        for delta in deltas
    ])
    return deltas, diffs


def plot_discrepancy_vs_angle(d: int, data: dict, out_dir: Path) -> None:
    """cap_fraction - solid_fraction as a function of separation, at
    equal radii r_cache=r_test=r, swept at r = each of PERCENTILES of
    the widest-range K's r_cache distribution.

    Two x-axis variants: normalized by r (reveals whether the curves
    collapse onto one shape, the small-angle-regime question) and raw
    delta (shows where the discrepancy peaks in absolute angular
    terms, which the normalized view obscures since different r have
    very different absolute scales)."""
    r_cache = _widest_range_r_cache(data)
    r_values = np.percentile(r_cache, PERCENTILES)
    colors = cm.viridis(np.linspace(0, 1, len(r_values)))
    ratios = np.linspace(0.01, 1.99, 80)
    variants = [
        ('normalized', 'delta / r (equal radii)', True),
        ('raw', 'delta (rad, equal radii)', False),
    ]
    for variant_name, xlabel, normalized in variants:
        fig, ax = plt.subplots(figsize=(6, 5))
        for pct, r, color in zip(PERCENTILES, r_values, colors):
            deltas, diffs = _discrepancy_curve(float(r), d, ratios)
            x = ratios if normalized else deltas
            ax.plot(x, diffs, color=color, label=f'p{pct} (r={r:.4f} rad)')
        ax.axhline(0.0, color='gray', linewidth=1, linestyle='--')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Difference in Methods (Cap - Solid)')
        ax.set_title(f'{d}D: Discrepancy vs. Angular Separation ({variant_name} x-axis, deterministic sweep, equal radii)')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        out_path = out_dir / f'{d}d_discrepancy_vs_angle_{variant_name}.png'
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"    wrote {out_path}")


def plot_peak_discrepancy_vs_r(d: int, data: dict, out_dir: Path) -> None:
    """Peak (over delta) cap-solid discrepancy as a function of r,
    swept over a fixed log-spaced grid, independent of any sampled
    data (unlike plot_discrepancy_vs_angle's percentile-based r
    values). Directly shows that the peak discrepancy is flat across
    the small-angle regime and only starts rising once r gets large."""
    fine_r = np.geomspace(0.002, 1.2, 30)
    ratios = np.linspace(0.01, 1.99, 200)
    peaks = [float(np.max(_discrepancy_curve(float(r), d, ratios)[1])) for r in fine_r]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fine_r, peaks, marker='o', markersize=3, color='tab:red')
    ax.set_xscale('log')
    ax.set_xlabel('r (rad, equal radii, log scale)')
    ax.set_ylabel('peak Difference in Methods (Cap - Solid) over delta')
    ax.set_title(f'{d}D: Peak Discrepancy vs. r (deterministic sweep, equal radii)')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / f'{d}d_peak_discrepancy_vs_r.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")

#-------------------------------------------------------------------------------
# 5. Aggregate summary: mean/median/p90/p95 of r_cache, r_test, delta
#-------------------------------------------------------------------------------

def plot_aggregate_summary(d: int, data: dict, out_dir: Path) -> None:
    fig, axes = _grid(3, figsize_scale=(4.2, 4.0))
    quantities = ['r_cache', 'r_test', 'delta']
    stats_fns = [('mean', np.mean), ('median', np.median),
                 ('p90', lambda x: np.percentile(x, 90)), ('p95', lambda x: np.percentile(x, 95))]
    x = np.arange(len(K_VALUES))
    for c, qty in enumerate(quantities):
        ax = axes[0][c]
        for stat_name, fn in stats_fns:
            values = [fn(data[K]['pairwise'][qty]) for K in K_VALUES]
            ax.plot(x, values, marker='o', markersize=4, label=stat_name)
        ax.set_xticks(x)
        ax.set_xticklabels([str(K) for K in K_VALUES])
        ax.set_xlabel('K')
        ax.set_ylabel(f'{qty} (rad)')
        ax.set_title(qty)
        ax.grid(alpha=0.3)
        if c == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f'{d}D: Aggregate Summary of r_cache, r_test, delta by K (raw pairwise pool)')
    plt.tight_layout()
    out_path = out_dir / f'{d}d_aggregate_summary.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")


def plot_aggregate_summary_raw(d: int, data: dict, out_dir: Path) -> None:
    """Non-aggregated companion to plot_aggregate_summary: the actual
    per-pair values (jittered strip per K)."""
    fig, axes = _grid(3, figsize_scale=(4.2, 4.0))
    quantities = ['r_cache', 'r_test', 'delta']
    rng = np.random.default_rng(0)
    for c, qty in enumerate(quantities):
        ax = axes[0][c]
        for i, K in enumerate(K_VALUES):
            values = data[K]['pairwise'][qty]
            jitter = rng.uniform(-0.3, 0.3, size=len(values))
            ax.scatter(np.full(len(values), i) + jitter, values, s=2, alpha=0.15, color='tab:blue', linewidths=0)
        ax.set_xticks(range(len(K_VALUES)))
        ax.set_xticklabels([str(K) for K in K_VALUES])
        ax.set_xlabel('K')
        ax.set_ylabel(f'{qty} (rad)')
        ax.set_title(qty)
        ax.grid(alpha=0.3)
    fig.suptitle(f'{d}D: Raw (Non-Aggregated) r_cache, r_test, delta by K (raw pairwise pool)')
    plt.tight_layout()
    out_path = out_dir / f'{d}d_aggregate_summary_raw.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")

#-------------------------------------------------------------------------------
# 6. Radius density distributions per K, on the selected (cap-winning)
# candidate, plus the shared test-query radius pool
#-------------------------------------------------------------------------------

def plot_radius_distributions(d: int, data: dict, out_dir: Path) -> None:
    fig, axes = _grid(2, figsize_scale=(5.5, 4.5))
    colors = cm.viridis(np.linspace(0, 1, len(K_VALUES)))
    panels = [('cap_r_cache', 'r_cache (cap best candidate)'), ('r_test', 'r_test')]
    for c, (field, label) in enumerate(panels):
        ax = axes[0][c]
        for K, color in zip(K_VALUES, colors):
            values = data[K]['selection'][field]
            ax.hist(values, bins=60, density=True, histtype='step', color=color, label=f'K={K}')
        ax.set_xlabel(f'{label} (rad)')
        ax.set_ylabel('density')
        ax.set_title(label)
        if c == 0:
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f'{d}D: Radius Distributions by K (best-candidate pool)')
    plt.tight_layout()
    out_path = out_dir / f'{d}d_radius_distributions.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")

#-------------------------------------------------------------------------------
# 7. Selection agreement: how often cap and solid-ball pick the same candidate
#-------------------------------------------------------------------------------

def plot_selection_agreement(d: int, data: dict, out_dir: Path) -> None:
    """Bar chart, K as a discrete category.

    Shows the fraction of test queries that are informative (at least one candidate had a
    nonzero fraction for either method) and, among only those, thereal agreement rate."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(K_VALUES))
    width = 0.35
    informative_frac, real_agreement = [], []
    for K in K_VALUES:
        sel = data[K]['selection']
        cap_frac, solid_frac = sel['cap_fraction'], sel['solid_fraction']
        cap_idx, solid_idx = sel['cap_winner_idx'], sel['solid_winner_idx']
        informative = (cap_frac > 0) | (solid_frac > 0)
        informative_frac.append(float(informative.mean()))
        real_agreement.append(float(np.mean(cap_idx[informative] == solid_idx[informative])) if informative.any() else float('nan'))
    ax.bar(x - width / 2, informative_frac, width, label='fraction of informative test queries', color='tab:blue')
    ax.bar(x + width / 2, real_agreement, width, label='agreement rate among informative queries', color='tab:purple')
    ax.set_xticks(x)
    ax.set_xticklabels([str(K) for K in K_VALUES])
    ax.set_xlabel('K')
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f'{d}D: Cap vs. Solid-Ball Selection Agreement by K')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    out_path = out_dir / f'{d}d_selection_agreement.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")

#-------------------------------------------------------------------------------

def main() -> None:
    global K_VALUES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=str, default=None, help='defaults to latest run under results/')
    parser.add_argument('--dims', type=int, nargs='+', default=DIMS)
    parser.add_argument('--k-values', type=int, nargs='+', default=K_VALUES, dest='k_values')
    args = parser.parse_args()

    K_VALUES = sorted(args.k_values)

    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir(RESULTS_BASE)
    print(f"Loading from: {run_dir}")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = OUT_BASE / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing plots to: {out_dir}\n")

    for d in args.dims:
        print(f"=== d={d} ===")
        data = load_dimension(run_dir, d, args.k_values)
        plot_binned_calibration(d, data, out_dir)
        plot_binned_calibration_per_background(d, data, out_dir)
        plot_calibration_scatter(d, data, out_dir)
        plot_point_scatter(d, data, out_dir)
        plot_method_comparison_scatter(d, data, out_dir)
        plot_discrepancy_vs_angle(d, data, out_dir)
        plot_peak_discrepancy_vs_r(d, data, out_dir)
        plot_aggregate_summary(d, data, out_dir)
        plot_aggregate_summary_raw(d, data, out_dir)
        plot_radius_distributions(d, data, out_dir)
        plot_selection_agreement(d, data, out_dir)
        print()

    print(f"Done: {out_dir}")

if __name__ == '__main__':
    main()
