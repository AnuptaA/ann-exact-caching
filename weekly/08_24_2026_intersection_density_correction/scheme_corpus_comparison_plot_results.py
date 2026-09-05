#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Full 2x2 comparison plots: query-generation scheme (A, B) crossed
with corpus (real, uniform), for the IF-vs-recall relationship."""

import sys
import argparse
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------

RESULTS_BASE = Path(__file__).parent / 'results'
OUT_BASE = Path(__file__).parent / 'plots'
N_IF_BINS = 20
IF_BIN_EDGES = np.linspace(0.0, 1.0, N_IF_BINS + 1)
DATASET_DIMS = {'sift': 128, 'esci': 384}
SHARED_K_VALUES = [25, 50, 100]  # present in both schemes' saved output

SCHEME_CONFIGS = {
    'A': {
        'real_part_dir': '{dataset}_part2_real_data',
        'uniform_part_dir': '{dataset}_part2_real_anchor_on_uniform',
        'fraction_field': 'cap_fraction',
        'label': 'Scheme A',
    },
    'B': {
        'real_part_dir': '{dataset}_synth_sweep',
        'uniform_part_dir': '{dataset}_synth_sweep_on_uniform',
        'fraction_field': 'fraction',
        'label': 'Scheme B',
    },
}
CORPUS_LABELS = {'real': 'real corpus', 'uniform': 'uniform corpus (same anchor/queries)'}

#-------------------------------------------------------------------------------

def load_cell(run_dir: Path, part_dir: str, d: int, k_values: list, fraction_field: str) -> dict:
    out = {}
    for K in k_values:
        path = run_dir / part_dir / f'{d}d' / 'raw' / f'K{K}.npz'
        arr = np.load(path)
        out[K] = {'fraction': arr[fraction_field], 'recall': arr['recall']}
    return out


def balance_by_if_bin(fraction: np.ndarray, recall: np.ndarray, rng: np.random.Generator) -> tuple:
    finite = np.isfinite(fraction) & np.isfinite(recall)
    fraction, recall = fraction[finite], recall[finite]
    bin_idx = np.clip(np.digitize(fraction, IF_BIN_EDGES[1:-1]), 0, N_IF_BINS - 1)
    raw_counts = np.array([np.sum(bin_idx == b) for b in range(N_IF_BINS)])
    nonzero = raw_counts[raw_counts > 0]
    n_per_bin = int(nonzero.min()) if len(nonzero) else 0

    selected = []
    for b in range(N_IF_BINS):
        idx_in_bin = np.flatnonzero(bin_idx == b)
        if len(idx_in_bin) == 0 or n_per_bin == 0:
            continue
        chosen = rng.choice(idx_in_bin, size=n_per_bin, replace=False)
        selected.append(chosen)
    if not selected:
        return fraction[:0], recall[:0], 0
    selected = np.concatenate(selected)
    return fraction[selected], recall[selected], n_per_bin


def _ols_fit(x: np.ndarray, y: np.ndarray) -> tuple:
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return float(slope), float(intercept), float(r_squared)


def _concordance_correlation_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    mean_x, mean_y = x.mean(), y.mean()
    var_x, var_y = x.var(), y.var()
    covariance = np.mean((x - mean_x) * (y - mean_y))
    return float(2 * covariance / (var_x + var_y + (mean_x - mean_y) ** 2))


def _fixed_bin_stats(x: np.ndarray, y: np.ndarray):
    centers = (IF_BIN_EDGES[:-1] + IF_BIN_EDGES[1:]) / 2.0
    means = np.full(N_IF_BINS, np.nan)
    sems = np.full(N_IF_BINS, np.nan)
    counts = np.zeros(N_IF_BINS, dtype=int)
    idx = np.clip(np.digitize(x, IF_BIN_EDGES[1:-1]), 0, N_IF_BINS - 1)
    for b in range(N_IF_BINS):
        mask = idx == b
        counts[b] = mask.sum()
        if counts[b] > 0:
            means[b] = y[mask].mean()
            sems[b] = y[mask].std(ddof=1) / np.sqrt(counts[b]) if counts[b] > 1 else 0.0
    return centers, means, sems, counts

#-------------------------------------------------------------------------------

def analyze(data: dict, rng: np.random.Generator) -> dict:
    """Balanced-bin fit and raw (full-n) fit, per K."""
    results = {}
    for K, arrays in data.items():
        fraction, recall = arrays['fraction'], arrays['recall']
        finite = np.isfinite(fraction) & np.isfinite(recall)
        fraction, recall = fraction[finite], recall[finite]

        raw_slope, raw_intercept, raw_r2 = _ols_fit(fraction, recall)
        raw_ccc = _concordance_correlation_coefficient(fraction, recall)

        bal_fraction, bal_recall, n_per_bin = balance_by_if_bin(fraction, recall, rng)
        if len(bal_fraction) >= 2:
            bal_slope, bal_intercept, bal_r2 = _ols_fit(bal_fraction, bal_recall)
            bal_ccc = _concordance_correlation_coefficient(bal_fraction, bal_recall)
        else:
            bal_slope = bal_intercept = bal_r2 = bal_ccc = float('nan')

        results[K] = {
            'raw_n': len(fraction), 'raw_r2': raw_r2, 'raw_slope': raw_slope,
            'raw_intercept': raw_intercept, 'raw_ccc': raw_ccc,
            'bal_n_per_bin': n_per_bin, 'bal_r2': bal_r2, 'bal_slope': bal_slope,
            'bal_intercept': bal_intercept, 'bal_ccc': bal_ccc,
            'bal_fraction': bal_fraction, 'bal_recall': bal_recall,
        }
    return results

#-------------------------------------------------------------------------------

def plot_2x2_scatter(dataset: str, d: int, all_results: dict, out_dir: Path) -> None:
    """One figure per K: 2x2 grid (scheme x corpus) of balanced scatter."""
    for K in SHARED_K_VALUES:
        fig, axes = plt.subplots(2, 2, figsize=(9, 8), squeeze=False)
        for row, scheme in enumerate(('A', 'B')):
            for col, corpus in enumerate(('real', 'uniform')):
                ax = axes[row][col]
                r = all_results[scheme][corpus][K]
                ax.scatter(r['bal_fraction'], r['bal_recall'], s=4, alpha=0.15,
                          color='tab:blue', linewidths=0)
                ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
                ax.set_xlim(-0.02, 1.02)
                ax.set_ylim(-0.05, 1.05)
                ax.set_title(f"{SCHEME_CONFIGS[scheme]['label']}\n{CORPUS_LABELS[corpus]}\n"
                             f"balanced R2={r['bal_r2']:.3f}, CCC={r['bal_ccc']:.3f} (n/bin={r['bal_n_per_bin']})",
                             fontsize=8)
                ax.set_xlabel('IF')
                if col == 0:
                    ax.set_ylabel('recall')
                ax.grid(alpha=0.3)
        fig.suptitle(f'{dataset.upper()} ({d}D), K={K}: 2x2 Scheme x Corpus Comparison (balanced)')
        plt.tight_layout()
        out_path = out_dir / f'{dataset}_{d}d_K{K}_2x2_scatter.png'
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"    wrote {out_path}")


def plot_2x2_binned(dataset: str, d: int, all_results: dict, out_dir: Path) -> None:
    for K in SHARED_K_VALUES:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), squeeze=False)
        axes = axes[0]
        colors = {'A': 'tab:orange', 'B': 'tab:blue'}
        for col, corpus in enumerate(('real', 'uniform')):
            ax = axes[col]
            for scheme in ('A', 'B'):
                r = all_results[scheme][corpus][K]
                centers, means, sems, counts = _fixed_bin_stats(r['bal_fraction'], r['bal_recall'])
                valid = counts > 0
                ax.errorbar(centers[valid], means[valid], yerr=sems[valid], marker='o', markersize=3,
                           color=colors[scheme], capsize=2,
                           label=f"{SCHEME_CONFIGS[scheme]['label']} (R2={r['bal_r2']:.2f}, CCC={r['bal_ccc']:.2f})")
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(CORPUS_LABELS[corpus], fontsize=9)
            ax.set_xlabel('IF')
            if col == 0:
                ax.set_ylabel('mean recall')
            ax.legend(fontsize=6)
            ax.grid(alpha=0.3)
        fig.suptitle(f'{dataset.upper()} ({d}D), K={K}: Binned Average, Scheme Comparison per Corpus')
        plt.tight_layout()
        out_path = out_dir / f'{dataset}_{d}d_K{K}_2x2_binned.png'
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"    wrote {out_path}")

#-------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=str, choices=['sift', 'esci'], required=True)
    parser.add_argument('--scheme-a-run-dir', type=str, required=True)
    parser.add_argument('--scheme-b-run-dir', type=str, required=True)
    parser.add_argument('--out-dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    d = DATASET_DIMS[args.dataset]
    run_dirs = {'A': Path(args.scheme_a_run_dir), 'B': Path(args.scheme_b_run_dir)}

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = OUT_BASE / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    all_results = {}
    for scheme, cfg in SCHEME_CONFIGS.items():
        all_results[scheme] = {}
        for corpus in ('real', 'uniform'):
            part_dir = cfg[f'{corpus}_part_dir'].format(dataset=args.dataset)
            data = load_cell(run_dirs[scheme], part_dir, d, SHARED_K_VALUES, cfg['fraction_field'])
            all_results[scheme][corpus] = analyze(data, rng)

    print(f"\n{args.dataset.upper()} ({d}D) full 2x2 comparison (raw = full-n fit, bal = balanced-by-IF-bin fit):")
    header = f"  {'scheme':>8}  {'corpus':>8}  {'K':>4}  {'raw_n':>8}  {'raw_R2':>7}  {'raw_CCC':>7}  " \
             f"{'bal_N/bin':>10}  {'bal_R2':>7}  {'bal_CCC':>7}"
    print(header)
    for scheme in ('A', 'B'):
        for corpus in ('real', 'uniform'):
            for K in SHARED_K_VALUES:
                r = all_results[scheme][corpus][K]
                print(f"  {scheme:>8}  {corpus:>8}  {K:>4}  {r['raw_n']:>8}  {r['raw_r2']:>7.4f}  "
                      f"{r['raw_ccc']:>7.4f}  {r['bal_n_per_bin']:>10}  {r['bal_r2']:>7.4f}  {r['bal_ccc']:>7.4f}")

    plot_2x2_scatter(args.dataset, d, all_results, out_dir)
    plot_2x2_binned(args.dataset, d, all_results, out_dir)

    print(f"\nDone: {out_dir}")


if __name__ == '__main__':
    main()
