#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Extra comparison plots for the compiled diagnostic report."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------

K_VALUES = [10, 25, 50, 100]
N_BINS = 20

#-------------------------------------------------------------------------------

def _load_selection(run_dir: Path, part_dir: str, d: int, K: int) -> dict:
    path = Path(run_dir) / part_dir / f'{d}d' / 'raw' / f'K{K}_selection.npz'
    arr = np.load(path)
    return {'fraction': arr['fraction'], 'recall': arr['recall']}


def _load_pairwise(run_dir: Path, part_dir: str, d: int, K: int) -> dict:
    path = Path(run_dir) / part_dir / f'{d}d' / 'raw' / f'K{K}_pairwise.npz'
    arr = np.load(path)
    return {'fraction': arr['fraction'], 'recall': arr['recall'], 'r_test': arr['r_test']}


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


def _plot_overlay(series: list, title: str, out_path: Path) -> None:
    """series: list of (label, color, marker, {K: {'fraction':..., 'recall':...}})."""
    fig, axes = plt.subplots(1, len(K_VALUES), figsize=(3.6 * len(K_VALUES), 3.4), squeeze=False)
    for c, K in enumerate(K_VALUES):
        ax = axes[0][c]
        for label, color, marker, data in series:
            centers, means, sems, counts = _bin_stats(data[K]['fraction'], data[K]['recall'])
            valid = counts > 0
            ax.errorbar(centers[valid], means[valid], yerr=sems[valid], marker=marker, markersize=3,
                        linestyle='-', color=color, label=label, capsize=2)
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1, label='y=x')
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f'$K={K}$', fontsize=10)
        ax.set_xlabel('intersection fraction (best candidate)')
        if c == 0:
            ax.set_ylabel('mean actual recall')
            ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")

#-------------------------------------------------------------------------------

def plot_sift_power_check_overlay(original_run_dir: Path, power_check_run_dir: Path, out_dir: Path) -> Path:
    original = {K: _load_selection(original_run_dir, 'debug_sift', 128, K) for K in K_VALUES}
    power_check = {K: _load_selection(power_check_run_dir, 'debug_sift', 128, K) for K in K_VALUES}
    out_path = out_dir / '128d_sift_power_check_overlay.png'
    series = [
        ('1000 test queries (original)', 'tab:green', 'o', original),
        ('3000 test queries (3x)', 'tab:orange', 's', power_check),
    ]
    _plot_overlay(series, 'SIFT (d=128): Statistical-Power Check (1000 vs. 3000 test queries)', out_path)
    return out_path


def plot_sift_whitening_overlay(original_run_dir: Path, whitened_run_dir: Path, out_dir: Path) -> Path:
    original = {K: _load_selection(original_run_dir, 'debug_sift', 128, K) for K in K_VALUES}
    whitened = {K: _load_selection(whitened_run_dir, 'debug_sift_whitened', 128, K) for K in K_VALUES}
    out_path = out_dir / '128d_sift_whitening_overlay.png'
    series = [
        ('original (1000/1000, 5 seeds)', 'tab:green', 'o', original),
        ('ZCA-whitened (300/300, 2 seeds)', 'tab:red', 'd', whitened),
    ]
    _plot_overlay(series, 'SIFT (d=128): Original vs. ZCA-Whitened Calibration', out_path)
    return out_path


def plot_sift_vs_esci_overlay(sift_run_dir: Path, esci_run_dir: Path, out_dir: Path) -> Path:
    sift = {K: _load_selection(sift_run_dir, 'debug_sift', 128, K) for K in K_VALUES}
    esci = {K: _load_selection(esci_run_dir, 'debug_esci', 384, K) for K in K_VALUES}
    out_path = out_dir / 'sift_vs_esci_cross_dataset_overlay.png'
    series = [
        ('SIFT (d=128)', 'tab:green', 'o', sift),
        ('ESCI (d=384)', 'tab:purple', '^', esci),
    ]
    _plot_overlay(series, 'Cross-Dataset Comparison: SIFT vs. ESCI Binned Calibration', out_path)
    return out_path

def plot_unbinned_pairwise_scatter(datasets: list, out_dir: Path) -> Path:
    """datasets: list of (label, color, {K: {'fraction':..., 'recall':...}}),
    unconditioned pairwise samples, no binning. Hexbin density."""
    fig, axes = plt.subplots(len(datasets), len(K_VALUES),
                              figsize=(3.6 * len(K_VALUES), 3.2 * len(datasets)), squeeze=False)
    for r, (label, _color, data) in enumerate(datasets):
        for c, K in enumerate(K_VALUES):
            ax = axes[r][c]
            fraction = data[K]['fraction']
            recall = data[K]['recall']
            hb = ax.hexbin(fraction, recall, gridsize=40, extent=(0, 1, 0, 1),
                           bins='log', cmap='viridis', mincnt=1)
            ax.plot([0, 1], [0, 1], linestyle='--', color='red', linewidth=1.2)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.05, 1.05)
            if r == 0:
                ax.set_title(f'$K={K}$', fontsize=10)
            if c == 0:
                ax.set_ylabel(f'{label}\nactual recall')
            if r == len(datasets) - 1:
                ax.set_xlabel('intersection fraction')
            if c == len(K_VALUES) - 1:
                plt.colorbar(hb, ax=ax, label='log10(count)', fraction=0.046, pad=0.04)
    fig.suptitle('Unconditioned Pairwise Calibration, Unbinned (hexbin density, no argmax selection)')
    plt.tight_layout()
    out_path = out_dir / 'unbinned_pairwise_scatter.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")
    return out_path


def plot_gap_vs_density(datasets: list, out_dir: Path, n_quantile_bins: int = 10) -> Path:
    """datasets: list of (label, color, {K: {'fraction':..., 'recall':..., 'r_test':...}}).
    Mean gap = recall - fraction, binned by r_test quantile."""
    fig, axes = plt.subplots(1, len(K_VALUES), figsize=(3.6 * len(K_VALUES), 3.4), squeeze=False)
    for c, K in enumerate(K_VALUES):
        ax = axes[0][c]
        for label, color, data in datasets:
            r_test = data[K]['r_test']
            gap = data[K]['recall'] - data[K]['fraction']
            edges = np.quantile(r_test, np.linspace(0, 1, n_quantile_bins + 1))
            edges[-1] += 1e-9
            idx = np.clip(np.digitize(r_test, edges) - 1, 0, n_quantile_bins - 1)
            centers = np.full(n_quantile_bins, np.nan)
            means = np.full(n_quantile_bins, np.nan)
            sems = np.full(n_quantile_bins, np.nan)
            for b in range(n_quantile_bins):
                mask = idx == b
                if mask.sum() > 0:
                    centers[b] = r_test[mask].mean()
                    means[b] = gap[mask].mean()
                    sems[b] = gap[mask].std(ddof=1) / np.sqrt(mask.sum()) if mask.sum() > 1 else 0.0
            valid = ~np.isnan(means)
            ax.errorbar(centers[valid], means[valid], yerr=sems[valid], marker='o', markersize=3,
                        linestyle='-', color=color, label=label, capsize=2)
        ax.axhline(0.0, linestyle='--', color='gray', linewidth=1)
        ax.set_title(f'$K={K}$', fontsize=10)
        ax.set_xlabel('r_test (smaller = denser region)')
        if c == 0:
            ax.set_ylabel('mean gap (recall - fraction)')
            ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle('Calibration Gap vs. Local Density (test query\'s own K-NN radius, quantile-binned)')
    plt.tight_layout()
    out_path = out_dir / 'gap_vs_density.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")
    return out_path

#-------------------------------------------------------------------------------

def summary_table_page(configs: list, out_path: Path) -> Path:
    """configs: list of (dataset_label, run_dir, part_dir, d). Renders a
    text table of mean fraction / mean recall / gap in the lowest
    populated fraction bin, per dataset x K."""
    rows = []
    for label, run_dir, part_dir, d in configs:
        for K in K_VALUES:
            sel = _load_selection(run_dir, part_dir, d, K)
            centers, means, sems, counts = _bin_stats(sel['fraction'], sel['recall'])
            valid = np.where(counts > 0)[0]
            if len(valid) == 0:
                continue
            lowest = valid[0]
            rows.append((label, K, centers[lowest], means[lowest], means[lowest] - centers[lowest], counts[lowest]))

    fig, ax = plt.subplots(figsize=(11, 0.6 + 0.4 * (len(rows) + 1)))
    ax.axis('off')
    ax.set_title('Calibration Gap Summary (lowest populated fraction bin, per dataset x K)',
                  fontsize=13, fontweight='bold', pad=12)

    col_labels = ['Dataset', 'K', 'Predicted fraction', 'Actual recall', 'Gap (recall - fraction)', 'Bin count']
    cell_text = [[r[0], str(r[1]), f'{r[2]:.3f}', f'{r[3]:.3f}', f'{r[4]:+.3f}', str(r[5])] for r in rows]

    table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(col_labels))))

    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")
    return out_path


def mc_formula_check_summary_page(out_path: Path) -> Path:
    """Renders a summary table of the Monte Carlo cross-check of
    exact_intersection_fraction at d=128 against real SIFT triples."""
    from debug_mc_formula_check import D, N_TRIALS, N_MC, SIFT_PAIRWISE_RUN, load_real_triples, mc_check
    from primitives import exact_intersection_fraction

    rng = np.random.default_rng(7)
    triples = load_real_triples(SIFT_PAIRWISE_RUN, K=25, n_trials=N_TRIALS, rng=rng)

    rows = []
    n_pass = 0
    for r_cache, r_test, delta, _ in triples:
        exact = exact_intersection_fraction(r_cache, r_test, delta, D)
        mc, se = mc_check(r_cache, r_test, delta, D, N_MC, rng)
        diff = abs(exact - mc)
        ok = diff < 4 * se + 0.002
        n_pass += ok
        rows.append((exact, mc, se, diff, 'PASS' if ok else 'FAIL'))

    max_diff = max(r[3] for r in rows)

    fig, ax = plt.subplots(figsize=(9, 0.5 + 0.35 * (min(len(rows), 12) + 3)))
    ax.axis('off')
    ax.set_title(f'Monte Carlo Formula Check (d={D}, real SIFT triples)\n'
                  f'{n_pass}/{len(rows)} passed, max diff = {max_diff:.5f}',
                  fontsize=13, fontweight='bold', pad=12)

    shown = rows[::max(1, len(rows) // 12)][:12]
    col_labels = ['Closed-form', 'Monte Carlo', 'MC std. error', 'Diff', 'Result']
    cell_text = [[f'{r[0]:.5f}', f'{r[1]:.5f}', f'{r[2]:.5f}', f'{r[3]:.5f}', r[4]] for r in shown]

    table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(col_labels))))

    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")
    return out_path
