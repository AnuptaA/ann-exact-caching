#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Analysis phase of the overnight log-spaced Scheme B sweep: loads
SIFT/ESCI results, produces recall-vs-IF plots on fine log-scale bins
(matched to argmax-winner IF's actual range, not the touch-coverage
test's all-pairs bins), and reports divergence-pattern, low-IF-region,
and timing findings."""

import re
import math
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'common'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scheme_corpus_comparison_plot_results import _ols_fit, _concordance_correlation_coefficient

#-------------------------------------------------------------------------------

HERE = Path(__file__).parent.parent
SUMMARY_LOG = HERE / 'overnight_summary.log'
PLOTS_DIR = HERE / 'plots' / 'overnight_log_targets'

DATASETS = {
    'sift': {'d': 128, 'run_dir': HERE / 'results' / 'overnight_sift',
             'real_part': 'sift_synth_sweep', 'uniform_part': 'sift_synth_sweep_on_uniform'},
    'esci': {'d': 384, 'run_dir': HERE / 'results' / 'overnight_esci',
             'real_part': 'esci_synth_sweep', 'uniform_part': 'esci_synth_sweep_on_uniform'},
}
K_VALUES = [25, 50, 100]

# Fine bins matched to argmax-winner IF's actual range (validated
# directly: winners concentrate from ~1e-5 up to 1, unlike the
# touch-coverage test's all-pairs data which spans down to 1e-20).
FINE_IF_BIN_EDGES = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 5e-1, 0.9, 1.0]
FINE_IF_BIN_LABELS = ['0', '(0,1e-5]', '(1e-5,1e-4]', '(1e-4,1e-3]', '(1e-3,1e-2]',
                       '(1e-2,1e-1]', '(1e-1,5e-1]', '(5e-1,0.9]', '(0.9,1]']
FINE_N_BINS = len(FINE_IF_BIN_LABELS)

#-------------------------------------------------------------------------------


def fine_bin_index(values: np.ndarray) -> np.ndarray:
    idx = np.zeros(len(values), dtype=np.int64)
    for b in range(1, len(FINE_IF_BIN_EDGES)):
        idx[values > FINE_IF_BIN_EDGES[b - 1]] = b
    return idx


def fine_bin_centers() -> np.ndarray:
    centers = np.zeros(FINE_N_BINS)
    centers[1] = FINE_IF_BIN_EDGES[1] / 2.0
    for b in range(2, FINE_N_BINS):
        centers[b] = math.sqrt(FINE_IF_BIN_EDGES[b - 1] * FINE_IF_BIN_EDGES[b])
    return centers


def fine_fixed_bin_stats(x: np.ndarray, y: np.ndarray) -> tuple:
    centers = fine_bin_centers()
    means = np.full(FINE_N_BINS, np.nan)
    sems = np.full(FINE_N_BINS, np.nan)
    counts = np.zeros(FINE_N_BINS, dtype=int)
    idx = fine_bin_index(x)
    for b in range(FINE_N_BINS):
        mask = idx == b
        counts[b] = mask.sum()
        if counts[b] > 0:
            means[b] = y[mask].mean()
            sems[b] = y[mask].std(ddof=1) / np.sqrt(counts[b]) if counts[b] > 1 else 0.0
    return centers, means, sems, counts


def fine_balance_by_if_bin(fraction: np.ndarray, recall: np.ndarray, rng: np.random.Generator) -> tuple:
    """Same equal-per-bin balancing as balance_by_if_bin, against the
    fine bins instead of the module's coarse ones."""
    bin_idx = fine_bin_index(fraction)
    raw_counts = np.array([np.sum(bin_idx == b) for b in range(FINE_N_BINS)])
    nonzero = raw_counts[raw_counts > 0]
    n_per_bin = int(nonzero.min()) if len(nonzero) else 0

    selected = []
    for b in range(FINE_N_BINS):
        idx_in_bin = np.flatnonzero(bin_idx == b)
        if len(idx_in_bin) == 0 or n_per_bin == 0:
            continue
        chosen = rng.choice(idx_in_bin, size=n_per_bin, replace=False)
        selected.append(chosen)
    if not selected:
        return fraction[:0], recall[:0], 0
    selected = np.concatenate(selected)
    return fraction[selected], recall[selected], n_per_bin


# Linear 0.05-wide bins for the primary histogram-style plot, separate
# from the fine log-scale bins above (which remain in use for the
# R2/CCC statistics, since those need balanced sampling across a range
# spanning many decades). The plot itself uses raw, unbalanced data, so
# 0.05-wide bins directly reflect the realized distribution.
LINEAR_BIN_EDGES = np.round(np.arange(0.0, 1.0001, 0.05), 2)
LINEAR_N_BINS = len(LINEAR_BIN_EDGES) - 1
LINEAR_BIN_LABELS = [f'[{LINEAR_BIN_EDGES[b]:.2f},{LINEAR_BIN_EDGES[b+1]:.2f})' for b in range(LINEAR_N_BINS)]


def linear_bin_index(values: np.ndarray) -> np.ndarray:
    idx = np.digitize(values, LINEAR_BIN_EDGES[1:-1], right=False)
    return np.clip(idx, 0, LINEAR_N_BINS - 1)


def linear_bin_centers() -> np.ndarray:
    return (LINEAR_BIN_EDGES[:-1] + LINEAR_BIN_EDGES[1:]) / 2.0


def linear_fixed_bin_stats(x: np.ndarray, y: np.ndarray) -> tuple:
    means = np.full(LINEAR_N_BINS, np.nan)
    counts = np.zeros(LINEAR_N_BINS, dtype=int)
    idx = linear_bin_index(x)
    for b in range(LINEAR_N_BINS):
        mask = idx == b
        counts[b] = mask.sum()
        if counts[b] > 0:
            means[b] = y[mask].mean()
    return means, counts


def load_cell(run_dir: Path, part_dir: str, d: int, K: int) -> dict:
    path = run_dir / part_dir / f'{d}d' / 'raw' / f'K{K}.npz'
    if not path.exists():
        return None
    arr = np.load(path)
    return {'fraction': arr['fraction'], 'recall': arr['recall']}


def analyze_cell(data: dict, rng: np.random.Generator) -> dict:
    fraction, recall = data['fraction'], data['recall']
    finite = np.isfinite(fraction) & np.isfinite(recall)
    fraction, recall = fraction[finite], recall[finite]

    raw_bin_idx = fine_bin_index(fraction)
    raw_bin_counts = {label: int((raw_bin_idx == b).sum()) for b, label in enumerate(FINE_IF_BIN_LABELS)}

    bal_fraction, bal_recall, n_per_bin = fine_balance_by_if_bin(fraction, recall, rng)
    bal_bin_idx = fine_bin_index(bal_fraction) if len(bal_fraction) else np.array([], dtype=np.int64)
    bal_bin_counts = {label: int((bal_bin_idx == b).sum()) for b, label in enumerate(FINE_IF_BIN_LABELS)}

    if len(bal_fraction) >= 2:
        slope, intercept, r2 = _ols_fit(bal_fraction, bal_recall)
        ccc = _concordance_correlation_coefficient(bal_fraction, bal_recall)
    else:
        slope = intercept = r2 = ccc = float('nan')

    bin_report = []
    for label, n in raw_bin_counts.items():
        if n == 0:
            continue
        mask = raw_bin_idx == FINE_IF_BIN_LABELS.index(label)
        bin_report.append((label, n, float(recall[mask].mean())))

    return {
        'n': len(fraction), 'r2': r2, 'ccc': ccc, 'slope': slope, 'intercept': intercept,
        'n_per_bin': n_per_bin, 'bal_fraction': bal_fraction, 'bal_recall': bal_recall,
        'fraction': fraction, 'recall': recall,
        'bin_report': bin_report, 'raw_bin_counts': raw_bin_counts, 'bal_bin_counts': bal_bin_counts,
    }


def plot_dataset(dataset: str, d: int, results: dict) -> Path:
    """Histogram-style plot: 20 equal-width (0.05) linear bins across the
    full [0,1] IF range, one bar per bin showing mean recall, computed on
    the raw (unbalanced) data. y=x is overlaid as a line through each
    bin's own center value."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    centers = linear_bin_centers()
    fig, axes = plt.subplots(2, len(K_VALUES), figsize=(4.5 * len(K_VALUES), 8), squeeze=False)
    for row, corpus in enumerate(('real', 'uniform')):
        for col, K in enumerate(K_VALUES):
            ax = axes[row][col]
            cell = results[corpus].get(K)
            if cell is None:
                ax.set_title(f'{corpus}, K={K} (missing)')
                continue
            means, counts = linear_fixed_bin_stats(cell['fraction'], cell['recall'])
            valid = counts > 0

            ax.bar(centers[valid], means[valid], width=0.045, color='tab:blue',
                   alpha=0.7, label='mean recall')
            ax.plot(centers[valid], centers[valid], 'k--', marker='x', linewidth=1, label='y=x')

            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel('IF')
            if col == 0:
                ax.set_ylabel('mean recall')
            ax.grid(alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
            ax.set_title(f'{corpus}, K={K}\nR2={cell["r2"]:.2f}, CCC={cell["ccc"]:.2f} (n={cell["n"]})',
                         fontsize=9)
    fig.suptitle(f'{dataset.upper()} ({d}D): recall vs IF, 0.05-wide linear bins')
    plt.tight_layout()
    out_path = PLOTS_DIR / f'{dataset}_{d}d_log_targets_binned.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_dataset_raw_scatter(dataset: str, d: int, results: dict) -> Path:
    """Every (IF, recall) row, no binning or downsampling. Plain linear
    x-axis: with a linear axis, a literal two-point y=x line is
    geometrically correct (unlike the symlog case, where it needed
    dense sampling to trace the diagonal correctly)."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, len(K_VALUES), figsize=(4.5 * len(K_VALUES), 8), squeeze=False)
    for row, corpus in enumerate(('real', 'uniform')):
        for col, K in enumerate(K_VALUES):
            ax = axes[row][col]
            cell = results[corpus].get(K)
            if cell is None:
                ax.set_title(f'{corpus}, K={K} (missing)')
                continue
            ax.scatter(cell['fraction'], cell['recall'], s=3, alpha=0.08, color='tab:blue', linewidths=0)
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel('IF')
            if col == 0:
                ax.set_ylabel('recall')
            ax.grid(alpha=0.3)
            ax.set_title(f'{corpus}, K={K} (n={cell["n"]}, raw, no downsampling)', fontsize=9)
    fig.suptitle(f'{dataset.upper()} ({d}D): recall vs IF, raw scatter, all rows')
    plt.tight_layout()
    out_path = PLOTS_DIR / f'{dataset}_{d}d_log_targets_raw_scatter.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out_path


def parse_summary_log() -> dict:
    if not SUMMARY_LOG.exists():
        return {}
    text = SUMMARY_LOG.read_text()
    steps = {}
    for name in ('SIFT', 'ESCI', 'Uniform baseline'):
        start_m = re.search(rf'{re.escape(name)} start: (.+)', text)
        end_m = re.search(rf'{re.escape(name)} completed: (.+)', text)
        fail_m = re.search(rf'{re.escape(name)} FAILED.*: (.+)', text)
        steps[name] = {'start': start_m.group(1) if start_m else None,
                       'end': end_m.group(1) if end_m else None,
                       'failed_at': fail_m.group(1) if fail_m else None}
    return steps

#-------------------------------------------------------------------------------


def main() -> None:
    rng = np.random.default_rng(123)
    print("=== Overnight sweep analysis (fine bins) ===\n")

    all_results = {}
    for dataset, cfg in DATASETS.items():
        print(f"--- {dataset.upper()} ---")
        results = {'real': {}, 'uniform': {}}
        for corpus, part_dir in (('real', cfg['real_part']), ('uniform', cfg['uniform_part'])):
            for K in K_VALUES:
                cell = load_cell(cfg['run_dir'], part_dir, cfg['d'], K)
                if cell is None:
                    print(f"  MISSING: {corpus} K={K}")
                    continue
                results[corpus][K] = analyze_cell(cell, rng)
                r = results[corpus][K]
                print(f"  {corpus} K={K}: n={r['n']}, R2={r['r2']:.3f}, CCC={r['ccc']:.3f}, "
                      f"slope={r['slope']:.3f}, intercept={r['intercept']:.3f}, n_per_bin={r['n_per_bin']}")
                for label, n, mean_recall in r['bin_report']:
                    print(f"      IF={label:>16}: n={n}, mean_recall={mean_recall:.4f}")
        all_results[dataset] = results
        out_path = plot_dataset(dataset, cfg['d'], results)
        print(f"  wrote {out_path}")
        raw_path = plot_dataset_raw_scatter(dataset, cfg['d'], results)
        print(f"  wrote {raw_path}")
        print()

    print("=== Divergence pattern check ===")
    for dataset, results in all_results.items():
        for K in K_VALUES:
            real_cell = results['real'].get(K)
            uniform_cell = results['uniform'].get(K)
            if real_cell is None or uniform_cell is None:
                continue
            print(f"  {dataset} K={K}: real CCC={real_cell['ccc']:.3f} (expect low, above y=x), "
                  f"uniform CCC={uniform_cell['ccc']:.3f} (expect high, tracks y=x)")

    print("\n=== Low-IF region check (real corpus) ===")
    print("  Touch-coverage earlier finding: recall ~0.0002-0.005 at IF<=1e-20, "
          "climbing meaningfully only past ~1e-5.")
    for dataset, results in all_results.items():
        for K in K_VALUES:
            real_cell = results['real'].get(K)
            if real_cell is None:
                continue
            print(f"  {dataset} K={K}:")
            for label, n, mean_recall in real_cell['bin_report']:
                print(f"      IF={label:>16}: mean_recall={mean_recall:.4f}")

    print("\n=== ESCI K=50 sparse-panel check (required) ===")
    esci_k50_real = all_results.get('esci', {}).get('real', {}).get(50)
    if esci_k50_real is not None:
        print(f"  Raw (pre-balancing) per-bin counts, ESCI K=50 real corpus (n={esci_k50_real['n']}):")
        for label, n in esci_k50_real['raw_bin_counts'].items():
            if n > 0:
                print(f"      {label:>16}: n={n}")
        print(f"  Balanced (post-balance_by_if_bin) per-bin counts (n_per_bin={esci_k50_real['n_per_bin']}):")
        for label, n in esci_k50_real['bal_bin_counts'].items():
            if n > 0:
                print(f"      {label:>16}: n={n}")
        n_nonzero_raw = sum(1 for n in esci_k50_real['raw_bin_counts'].values() if n > 0)
        n_nonzero_bal = sum(1 for n in esci_k50_real['bal_bin_counts'].values() if n > 0)
        print(f"  Verdict: {n_nonzero_raw} raw nonzero bins, {n_nonzero_bal} balanced nonzero bins, "
              f"total balanced n={sum(esci_k50_real['bal_bin_counts'].values())}.")
        if n_nonzero_bal <= 2 or sum(esci_k50_real['bal_bin_counts'].values()) < 20:
            print("  STILL DEGENERATE at K=50: fine bins did not resolve this panel into a "
                  "genuinely multi-point curve.")
        else:
            print("  RESOLVED at K=50: genuinely multi-point curve under fine bins.")
    else:
        print("  ESCI K=50 real corpus data not found.")

    print("\n=== Timing ===")
    steps = parse_summary_log()
    expectations = {'SIFT': '~3h54m (old target scheme)', 'ESCI': '~2h48m (old target scheme)',
                     'Uniform baseline': 'not previously timed this session'}
    for name, info in steps.items():
        status = 'FAILED' if info['failed_at'] else ('completed' if info['end'] else 'not found in log')
        print(f"  {name}: start={info['start']}, end={info['end'] or info['failed_at']}, "
              f"status={status}, expected={expectations[name]}")

    print("\nDone.")


if __name__ == '__main__':
    main()
