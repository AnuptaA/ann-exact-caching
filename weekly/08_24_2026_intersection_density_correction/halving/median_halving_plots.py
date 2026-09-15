#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Plots for the median-halving tree grid, built from the already-saved
results/median_halving/*.npz files - no new experiment runs."""

import sys
import math
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'common'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from median_halving_analysis import load_result, all_dataset_keys
from median_halving_grid_run_experiment import N, C_VALUES, DIMS_SYNTHETIC, W_MULTIPLIERS

#-------------------------------------------------------------------------------

PLOTS_DIR = Path(__file__).parent.parent / 'plots' / 'median_halving'

#-------------------------------------------------------------------------------


def plot_pruning_collapse() -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    for C in C_VALUES:
        dims, fracs = [], []
        for d in DIMS_SYNTHETIC:
            r = load_result(f'uniform_d{d}', C, True)
            if r is None:
                continue
            frac = r['leaf_touch_counts'].mean() / int(r['n_leaves'])
            dims.append(d)
            fracs.append(frac)
        ax.plot(dims, fracs, 'o-', label=f'C={C}')

    for label, key, marker in [('SIFT (d=128)', 'sift', '*'), ('ESCI (d=384)', 'esci', 'X')]:
        for C in C_VALUES:
            r = load_result(key, C, True)
            if r is None:
                continue
            frac = r['leaf_touch_counts'].mean() / int(r['n_leaves'])
            d = 128 if key == 'sift' else 384
            ax.scatter([d], [frac], marker=marker, s=80, color='black', zorder=5)

    ax.set_xscale('log')
    ax.set_xlabel('dimension (log scale)')
    ax.set_ylabel('fraction of leaves touched per query (mean)')
    ax.set_title('Query-time pruning collapse: uniform data, Variant H\n'
                  '(black */X markers: SIFT/ESCI real data at their native C values)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / 'pruning_collapse.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_depth_h_vs_n() -> Path:
    panels = ['sift', 'esci', 'uniform_d2', 'uniform_d10', 'uniform_d128']
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.5), squeeze=False)
    for col, key in enumerate(panels):
        ax = axes[0][col]
        depths_h, depths_n, theory = [], [], []
        for C in C_VALUES:
            rh = load_result(key, C, True)
            rn = load_result(key, C, False)
            depths_h.append(int(rh['max_depth_reached']) if rh is not None else np.nan)
            depths_n.append(int(rn['max_depth_reached']) if rn is not None else np.nan)
            theory.append(math.ceil(math.log2(N / C)))
        x = np.arange(len(C_VALUES))
        width = 0.35
        ax.bar(x - width / 2, depths_h, width, label='Variant H', color='tab:blue')
        ax.bar(x + width / 2, depths_n, width, label='Variant N', color='tab:orange')
        ax.plot(x, theory, 'k--', marker='x', label='ceil(log2(N/C))')
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in C_VALUES])
        ax.set_xlabel('C')
        ax.set_title(key)
        if col == 0:
            ax.set_ylabel('max depth reached')
            ax.legend(fontsize=8)
    fig.suptitle('Construction depth: Variant H vs N, vs theoretical balanced-split depth')
    plt.tight_layout()
    out = PLOTS_DIR / 'depth_h_vs_n.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_capped_fraction_clustered() -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    for wi, wm in enumerate(W_MULTIPLIERS):
        dims, fracs = [], []
        for d in DIMS_SYNTHETIC:
            r = load_result(f'clustered_d{d}_w{wi}', 5, False)
            if r is None:
                continue
            dims.append(d)
            fracs.append(int(r['n_capped_points']) / N)
        ax.plot(dims, fracs, 'o-', label=f'W = {wm} x 1/sqrt(d)')
    ax.set_xscale('log')
    ax.set_xlabel('dimension (log scale)')
    ax.set_ylabel('fraction of points left in depth-capped (unresolved) leaves')
    ax.set_title('Variant N construction failure vs dimension, synthetic clustered data (C=5)\n'
                  'Non-monotonic: peaks at low/mid dimension, vanishes by d=384')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / 'capped_fraction_clustered.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_memory() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    x = np.arange(len(C_VALUES))
    width = 0.2
    for i, key in enumerate(['sift', 'esci']):
        full_vals = [load_result(key, C, True)['mem_full_direction'].item() for C in C_VALUES]
        pooled_vals = [load_result(key, C, True)['mem_pooled_index'].item() for C in C_VALUES]
        offset = (i - 0.5) * 2 * width
        ax.bar(x + offset - width / 2, full_vals, width, label=f'{key} full_direction',
               color='tab:red' if key == 'sift' else 'tab:orange')
        ax.bar(x + offset + width / 2, pooled_vals, width, label=f'{key} pooled_index',
               color='tab:blue' if key == 'sift' else 'tab:cyan')
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in C_VALUES])
    ax.set_xlabel('C')
    ax.set_ylabel('bytes (measured, N=50,000)')
    ax.set_yscale('log')
    ax.set_title('Measured memory at N=50,000')
    ax.legend(fontsize=7)

    ax = axes[1]
    scale = 100_000_000 / N
    for i, key in enumerate(['sift', 'esci']):
        d = 128 if key == 'sift' else 384
        pool_fixed = 256 * d * 8
        full_vals = [load_result(key, C, True)['mem_full_direction'].item() * scale / 1e9 for C in C_VALUES]
        pooled_vals = [((load_result(key, C, True)['mem_pooled_index'].item() - pool_fixed) * scale
                         + pool_fixed) / 1e9 for C in C_VALUES]
        offset = (i - 0.5) * 2 * width
        ax.bar(x + offset - width / 2, full_vals, width, label=f'{key} full_direction',
               color='tab:red' if key == 'sift' else 'tab:orange')
        ax.bar(x + offset + width / 2, pooled_vals, width, label=f'{key} pooled_index',
               color='tab:blue' if key == 'sift' else 'tab:cyan')
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in C_VALUES])
    ax.set_xlabel('C')
    ax.set_ylabel('GB (extrapolated, N=100,000,000)')
    ax.set_yscale('log')
    ax.set_title('Extrapolated memory at N=100M (labeled estimate)')
    ax.legend(fontsize=7)

    fig.suptitle('H5: direction storage, full per-node vector vs shared pool + index')
    plt.tight_layout()
    out = PLOTS_DIR / 'memory_comparison.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_degenerate_leaf_accounting() -> Path:
    total_unique_h = total_capped_h = total_unique_n = total_capped_n = 0
    for dataset_key in all_dataset_keys():
        for C in C_VALUES:
            rh = load_result(dataset_key, C, True)
            rn = load_result(dataset_key, C, False)
            if rh is not None:
                total_unique_h += int(rh['n_leaf_unique'])
                total_capped_h += int(rh['n_capped_leaves'])
            if rn is not None:
                total_unique_n += int(rn['n_leaf_unique'])
                total_capped_n += int(rn['n_capped_leaves'])

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(2)
    width = 0.35
    ax.bar(x - width / 2, [total_unique_h, total_capped_h], width, label='Variant H', color='tab:blue')
    ax.bar(x + width / 2, [total_unique_n, total_capped_n], width, label='Variant N', color='tab:orange')
    ax.set_xticks(x)
    ax.set_xticklabels(['duplicate-degenerate\n(exact)', 'depth-capped\n(unresolved)'])
    ax.set_ylabel('total leaves across entire grid (log scale)')
    ax.set_yscale('symlog')
    ax.set_title('Degenerate-leaf accounting, summed across all 352 configs')
    ax.legend()
    for i, v in enumerate([total_unique_h, total_capped_h]):
        ax.text(i - width / 2, v, str(v), ha='center', va='bottom', fontsize=8)
    for i, v in enumerate([total_unique_n, total_capped_n]):
        ax.text(i + width / 2, v, str(v), ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    out = PLOTS_DIR / 'degenerate_leaf_accounting.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out


def check_variant_n_accuracy() -> dict:
    """Not previously reported: Variant N's own count accuracy."""
    all_err = []
    per_config = []
    for dataset_key in all_dataset_keys():
        for C in C_VALUES:
            r = load_result(dataset_key, C, False)
            if r is None:
                continue
            err = r['est_counts'].astype(np.int64) - r['true_counts'].astype(np.int64)
            all_err.append(err)
            if np.abs(err).max() > 0:
                per_config.append((dataset_key, C, err))
    all_err_concat = np.concatenate(all_err)
    return {'all_err': all_err_concat, 'nonzero_configs': per_config}


def plot_variant_n_accuracy(nonzero_configs: list) -> Path:
    picks = sorted(nonzero_configs, key=lambda x: -np.abs(x[2]).mean())[:4]
    if not picks:
        return None
    fig, axes = plt.subplots(1, len(picks), figsize=(4.5 * len(picks), 4.5), squeeze=False)
    for col, (dataset_key, C, err) in enumerate(picks):
        ax = axes[0][col]
        ax.hist(err, bins=30, color='tab:orange')
        ax.set_title(f'{dataset_key}, C={C}\nmean err={err.mean():.2f}, min={err.min()}', fontsize=9)
        ax.set_xlabel('estimated - true count')
        if col == 0:
            ax.set_ylabel('query count')
    fig.suptitle('Variant N count error where depth-capping occurred (undercounting)')
    plt.tight_layout()
    out = PLOTS_DIR / 'variant_n_accuracy.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out

#-------------------------------------------------------------------------------


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Spot checks:")
    r = load_result('sift', 5, True)
    print(f"  sift C=5 H: leaf_touch_frac={r['leaf_touch_counts'].mean()/int(r['n_leaves']):.4f} "
          f"(expect ~1.0)")
    r = load_result('esci', 5, True)
    print(f"  esci C=5 H: leaf_touch_frac={r['leaf_touch_counts'].mean()/int(r['n_leaves']):.4f} "
          f"(expect ~1.0)")

    print(f"\nWrote {plot_pruning_collapse()}")
    print(f"Wrote {plot_depth_h_vs_n()}")
    print(f"Wrote {plot_capped_fraction_clustered()}")
    print(f"Wrote {plot_memory()}")
    print(f"Wrote {plot_degenerate_leaf_accounting()}")

    acc = check_variant_n_accuracy()
    print(f"\nVariant N accuracy across whole grid: n={len(acc['all_err'])}, "
          f"mean={acc['all_err'].mean():.4f}, min={acc['all_err'].min()}, max={acc['all_err'].max()}")
    print(f"Configs with nonzero Variant N error: {len(acc['nonzero_configs'])}")
    out = plot_variant_n_accuracy(acc['nonzero_configs'])
    if out:
        print(f"Wrote {out}")

    print("\nDone.")


if __name__ == '__main__':
    main()
