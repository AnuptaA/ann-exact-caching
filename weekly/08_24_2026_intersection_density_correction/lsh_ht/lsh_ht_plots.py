#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""The 7 specified plots for the Horvitz-Thompson LSH grid."""

import sys
import json
import math
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'common'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lsh_ht_analysis import load_e3, all_e3_configs
from lsh_ht_grid_run_experiment import OUT_DIR, N, K, E3_DATASETS, _direct_radii
from lsh_horvitz_thompson import select_hyperparameters, p_single

#-------------------------------------------------------------------------------

PLOTS_DIR = Path(__file__).parent.parent / 'plots' / 'lsh_ht'
COLORS = {'sift': 'tab:blue', 'esci': 'tab:orange', 'isotropic_d128': 'tab:green',
          'isotropic_d384': 'tab:red', 'anisotropic_d128_r5': 'tab:purple',
          'anisotropic_d128_r15': 'tab:brown', 'anisotropic_d128_r40': 'tab:pink'}

#-------------------------------------------------------------------------------


Y_CAP = 1.5  # relative error beyond 150% is "very bad" either way; capping keeps
             # the actionable 0-150% range legible instead of compressed near 0


def plot1_frontier() -> Path:
    fig, ax = plt.subplots(figsize=(9, 7))
    n_clipped = 0
    for dataset_key in E3_DATASETS:
        fracs, errs, Ls = [], [], []
        for m, x, L in all_e3_configs(dataset_key):
            r = load_e3(dataset_key, m, x, L)
            valid = r['true_counts'] > 0
            if not valid.any():
                continue
            err = np.nanmean(r['rel_err'][valid])
            if err == 0 or np.isnan(err):
                continue
            fracs.append(r['retrieved_counts'].mean() / N)
            errs.append(min(err, Y_CAP))
            n_clipped += err > Y_CAP
            Ls.append(L)
        sizes = 10 + 3 * np.log2(np.array(Ls))
        ax.scatter(fracs, errs, s=sizes, alpha=0.5, color=COLORS[dataset_key], label=dataset_key)

    # Uniform-sampling baseline: nearly identical across datasets (it
    # only depends on K, N, and sample fraction, not on any geometric
    # structure), so one averaged line represents all of them.
    uniform_dir = OUT_DIR / 'uniform_baseline'
    if uniform_dir.exists():
        all_fracs, all_errs = None, []
        for dataset_key in E3_DATASETS:
            path = uniform_dir / f'{dataset_key}.npz'
            if not path.exists():
                continue
            data = np.load(path)
            all_fracs = data['fracs']
            all_errs.append(data['mean_rel_err'])
        if all_errs:
            mean_err = np.mean(all_errs, axis=0)
            ax.plot(all_fracs, np.minimum(mean_err, Y_CAP), 'k--', marker='s', linewidth=2,
                     label='uniform sampling (baseline, avg over datasets)')

    ax.set_xscale('log')
    ax.set_ylim(0, Y_CAP)
    ax.set_xlabel('retrieved fraction (log scale - spans many orders of magnitude by design)')
    ax.set_ylabel(f'mean relative error (linear, capped at {Y_CAP:.1f} = 150%)')
    ax.set_title(f'Cost/accuracy frontier (point size ~ log2(L)); '
                 f'{n_clipped} point(s) above {Y_CAP:.1f} shown at the cap')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / 'plot1_frontier.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out


def plot2_error_vs_L() -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), squeeze=False)
    dataset_list = list(E3_DATASETS.keys())
    for i, dataset_key in enumerate(dataset_list):
        ax = axes[i // 4][i % 4]
        L_to_errs = {}
        L_to_frac = {}
        for m, x, L in all_e3_configs(dataset_key):
            r = load_e3(dataset_key, m, x, L)
            valid = r['true_counts'] > 0
            if not valid.any():
                continue
            err = np.nanmean(r['rel_err'][valid])
            L_to_errs.setdefault(L, []).append(err)
            L_to_frac.setdefault(L, []).append(r['retrieved_counts'].mean() / N)
        Ls = sorted(L_to_errs.keys())
        observed = [min(np.mean(L_to_errs[L]), Y_CAP) for L in Ls]
        mean_frac = [np.mean(L_to_frac[L]) for L in Ls]
        predicted = [min(math.sqrt(max((1 - f) / (f * K), 0)), Y_CAP) if f > 0 else np.nan for f in mean_frac]
        ax.plot(Ls, observed, 'o-', label='observed')
        ax.plot(Ls, predicted, 'k--', label='predicted (variance formula)')
        ax.set_xscale('log')
        ax.set_ylim(0, Y_CAP)
        ax.set_title(dataset_key, fontsize=9)
        ax.set_xlabel('L (log scale)')
        if i % 4 == 0:
            ax.set_ylabel(f'mean relative error (linear, capped at {Y_CAP:.1f})')
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(len(dataset_list), 8):
        axes[j // 4][j % 4].axis('off')
    fig.suptitle('H3: error vs L, observed vs variance-formula prediction (linear y, capped at 150%)')
    plt.tight_layout()
    out = PLOTS_DIR / 'plot2_error_vs_L.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out


def plot3_fixed_table_bias() -> Path:
    from lsh_ht_grid_run_experiment import L_VALUES
    stds, fracs, Ls = [], [], []
    for L in L_VALUES:
        path = OUT_DIR / 'E2' / f'L{L}.npz'
        if not path.exists():
            continue
        data = np.load(path)
        stds.append(float(data['mean_ratios'].std()))
        fracs.append(float(data['mean_retrieved_fracs'].mean()) if 'mean_retrieved_fracs' in data else np.nan)
        Ls.append(L)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))

    ax = axes[0]
    ax.plot(Ls, stds, 'o-', color='tab:blue', label='observed sigma')
    ref = stds[0] / np.sqrt(np.array(Ls) / Ls[0])
    ax.plot(Ls, ref, 'k--', label='1/sqrt(L) reference')
    ax.set_xscale('log')
    ax.set_xlabel('L (log scale)')
    ax.set_ylabel('sigma of per-table-set mean ratio (linear)')
    ax.set_title('Linear y: shows the actual magnitude at each L')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(Ls, stds, 'o-', color='tab:blue', label='observed sigma')
    ax.plot(Ls, ref, 'k--', label='1/sqrt(L) reference')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('L (log scale)')
    ax.set_ylabel('sigma (log)')
    ax.set_title('Log y: shows the decay is much steeper than 1/sqrt(L)')
    ax.legend()
    ax.grid(alpha=0.3, which='both')

    ax = axes[2]
    ax.plot(Ls, fracs, 'o-', color='tab:red')
    ax.set_xscale('log')
    ax.set_ylim(0, 1.0)
    ax.set_xlabel('L (log scale)')
    ax.set_ylabel('mean retrieved fraction')
    ax.set_title('Retrieval fraction vs L: sigma->0 tracks retrieving ~half the database, not a free lunch')
    ax.grid(alpha=0.3)

    fig.suptitle('H2: fixed-table bias vs L (SIFT, m=20, x=2, 30 draws each)')
    plt.tight_layout()
    out = PLOTS_DIR / 'plot3_fixed_table_bias.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out


def plot4_scaling() -> Path:
    n_values = [4000, 8000, 16000, 32000, 50000]
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, color in [('fixedL', 'tab:blue'), ('targeterr', 'tab:orange')]:
        counts = []
        ns = []
        for n_pts in n_values:
            path = OUT_DIR / 'E4' / f'{label}_N{n_pts}.npz'
            if not path.exists():
                continue
            data = np.load(path)
            counts.append(float(data['retrieved_counts'].mean()))
            ns.append(n_pts)
        if len(counts) < 2:
            continue
        log_n, log_c = np.log(ns), np.log(counts)
        alpha, intercept = np.polyfit(log_n, log_c, 1)
        ax.plot(ns, counts, 'o-', color=color, label=f'{label} (alpha={alpha:.3f})')
        fit_line = np.exp(intercept) * np.array(ns, dtype=float) ** alpha
        ax.plot(ns, fit_line, '--', color=color, alpha=0.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('N (log)')
    ax.set_ylabel('mean retrieved count (log)')
    ax.set_title('H4: retrieval scaling, isotropic d=128')
    ax.legend()
    ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    out = PLOTS_DIR / 'plot4_scaling.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out


def plot5_calibration() -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), squeeze=False)
    dataset_list = list(E3_DATASETS.keys())
    for i, dataset_key in enumerate(dataset_list):
        ax = axes[i // 4][i % 4]
        configs = all_e3_configs(dataset_key)
        mid = configs[len(configs) // 2] if configs else None
        if mid is None:
            continue
        r = load_e3(dataset_key, *mid)
        valid = r['true_counts'] > 0
        true_c, n_hat = r['true_counts'][valid], r['n_hat'][valid]
        ax.scatter(true_c, n_hat, s=8, alpha=0.4, color=COLORS[dataset_key])
        lo, hi = 0, max(true_c.max(), n_hat.max()) * 1.1
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1)
        ax.set_title(f'{dataset_key}\n(m={mid[0]},x={mid[1]},L={mid[2]})', fontsize=9)
        ax.set_xlabel('true count')
        if i % 4 == 0:
            ax.set_ylabel('N_hat')
    for j in range(len(dataset_list), 8):
        axes[j // 4][j % 4].axis('off')
    fig.suptitle('Calibration: N_hat vs true count (representative mid-grid config per dataset)')
    plt.tight_layout()
    out = PLOTS_DIR / 'plot5_calibration.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out


def plot6_cap_radius_histogram() -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), squeeze=False)
    dataset_list = list(E3_DATASETS.keys())
    for i, dataset_key in enumerate(dataset_list):
        ax = axes[i // 4][i % 4]
        data = E3_DATASETS[dataset_key]()
        radii = _direct_radii(data['points'], data['queries'], K)
        ax.hist(radii, bins=30, color=COLORS[dataset_key], alpha=0.7)
        ax.set_title(dataset_key, fontsize=9)
        ax.set_xlabel('K=100 cap radius (rad)')

        ax2 = ax.twinx()
        theta_range = np.linspace(0.01, np.pi - 0.01, 200)
        ax2.plot(theta_range, p_single(theta_range, 20, 2), color='black', linewidth=1)
        ax2.set_ylabel('p_single(theta, m=20, x=2)', fontsize=7)
    for j in range(len(dataset_list), 8):
        axes[j // 4][j % 4].axis('off')
    fig.suptitle('Cap radius distribution per dataset, with p_single(m=20,x=2) overlaid')
    plt.tight_layout()
    out = PLOTS_DIR / 'plot6_cap_radius_histogram.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out


def plot7_memory() -> Path:
    mem = json.load(open(OUT_DIR / 'E6' / 'memory.json'))
    fig, ax = plt.subplots(figsize=(9, 6))
    labels = [f"{r['dataset']}\nm={r['m']},L={r['L']}" for r in mem]
    x = np.arange(len(mem))
    width = 0.35
    full = [r['full_direction_bytes'] for r in mem]
    pooled = [r['pooled_direction_bytes'] for r in mem]
    ax.bar(x - width / 2, full, width, label='full direction storage', color='tab:red')
    ax.bar(x + width / 2, pooled, width, label='pooled (P=256) + index', color='tab:blue')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
    ax.set_ylabel('bytes (log scale)')
    ax.set_yscale('log')
    ax.set_title('Direction storage: full per-table vs shared pool + index')
    ax.legend()
    plt.tight_layout()
    out = PLOTS_DIR / 'plot7_memory.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out

#-------------------------------------------------------------------------------


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Wrote {plot1_frontier()}")
    print(f"Wrote {plot2_error_vs_L()}")
    print(f"Wrote {plot3_fixed_table_bias()}")
    print(f"Wrote {plot4_scaling()}")
    print(f"Wrote {plot5_calibration()}")
    print(f"Wrote {plot6_cap_radius_histogram()}")
    print(f"Wrote {plot7_memory()}")
    print("\nDone.")


if __name__ == '__main__':
    main()
