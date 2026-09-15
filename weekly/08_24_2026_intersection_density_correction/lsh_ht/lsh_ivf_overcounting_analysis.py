#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Analysis of the LSH-vs-IVF overcounting-factor microbenchmark: summary
tables, hypothesis 1-7 verdicts, and the four required plots."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'common'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lsh_ivf_overcounting_run_experiment import D_VALUES, M_VALUES, N_CLUSTERS_VALUES, KAPPA_RANGE, CAP_RADIUS

#-------------------------------------------------------------------------------

HERE = Path(__file__).parent.parent
RESULTS_DIR = HERE / 'results' / 'lsh_ivf_overcounting'
PLOTS_DIR = HERE / 'plots' / 'lsh_ivf_overcounting'

#-------------------------------------------------------------------------------


def load_combo(d: int, m: int, n_clusters: int) -> dict:
    path = RESULTS_DIR / f'd{d}_m{m}_c{n_clusters}.npz'
    if not path.exists():
        return None
    arr = np.load(path)
    return {k: arr[k] for k in arr.files}


def summarize_combo(data: dict) -> dict:
    """Overcounting-factor median/IQR (true_captured > 0) and
    false-signal rate/magnitude (true_captured == 0, estimate > 0),
    per method."""
    tc = data['true_captured']
    out = {}
    for method in ('lsh', 'ivf'):
        est = data[f'estimate_{method}']
        pos_mask = tc > 0
        overcounting = est[pos_mask] / tc[pos_mask]
        fs_mask = (tc == 0) & (est > 0)
        out[method] = {
            'median': float(np.median(overcounting)) if overcounting.size else float('nan'),
            'q25': float(np.percentile(overcounting, 25)) if overcounting.size else float('nan'),
            'q75': float(np.percentile(overcounting, 75)) if overcounting.size else float('nan'),
            'n_positive': int(pos_mask.sum()),
            'false_signal_rate': float(fs_mask.mean()),
            'false_signal_avg_estimate': float(est[fs_mask].mean()) if fs_mask.any() else 0.0,
            'overcounting_raw': overcounting,
        }
    return out


def build_summary_table() -> dict:
    table = {}
    for d in D_VALUES:
        for m in M_VALUES:
            for n_clusters in N_CLUSTERS_VALUES:
                data = load_combo(d, m, n_clusters)
                if data is None:
                    continue
                table[(d, m, n_clusters)] = summarize_combo(data)
    return table

#-------------------------------------------------------------------------------


def print_summary_table(table: dict) -> None:
    print("=== Summary: median [IQR] overcounting factor, false-signal rate ===")
    for d in D_VALUES:
        for n_clusters in N_CLUSTERS_VALUES:
            print(f"\n  d={d}, n_clusters={n_clusters}")
            for m in M_VALUES:
                key = (d, m, n_clusters)
                if key not in table:
                    continue
                lsh, ivf = table[key]['lsh'], table[key]['ivf']
                print(f"    m={m}: LSH median={lsh['median']:.3f} [{lsh['q25']:.3f},{lsh['q75']:.3f}], "
                      f"fs_rate={lsh['false_signal_rate']:.3f}, fs_avg_est={lsh['false_signal_avg_estimate']:.1f} | "
                      f"IVF median={ivf['median']:.3f} [{ivf['q25']:.3f},{ivf['q75']:.3f}], "
                      f"fs_rate={ivf['false_signal_rate']:.3f}, fs_avg_est={ivf['false_signal_avg_estimate']:.1f}")


def check_hypotheses(table: dict) -> None:
    print("\n=== Hypothesis checks ===")

    # H1: LSH overcounting > 1 at low-to-moderate m.
    low_m = [m for m in M_VALUES if m <= 4]
    lsh_low_m_medians = [table[(d, m, c)]['lsh']['median'] for d in D_VALUES for m in low_m
                          for c in N_CLUSTERS_VALUES if (d, m, c) in table]
    frac_above_1 = np.mean([v > 1.0 for v in lsh_low_m_medians])
    print(f"H1 (LSH overcounting > 1 at low m): median LSH values at m<=4 exceed 1.0 in "
          f"{frac_above_1:.1%} of (d, n_clusters) combos; mean={np.mean(lsh_low_m_medians):.2f} "
          f"-> {'CONFIRMED' if frac_above_1 > 0.9 else 'AMBIGUOUS'}")

    # H2: LSH overcounting decreases with m, gradually.
    print("H2 (LSH declines gradually with m, possible knee):")
    for d in D_VALUES:
        for c in N_CLUSTERS_VALUES:
            vals = [table[(d, m, c)]['lsh']['median'] for m in M_VALUES if (d, m, c) in table]
            if len(vals) < 2:
                continue
            diffs = np.diff(vals)
            monotonic = np.all(diffs <= 1e-9)
            rel_drops = -diffs / np.array(vals[:-1])
            knee_m = M_VALUES[int(np.argmax(rel_drops)) + 1] if len(rel_drops) else None
            print(f"    d={d}, n_clusters={c}: values={['%.2f' % v for v in vals]}, "
                  f"monotonic_decrease={monotonic}, largest relative drop at m={knee_m}")

    # H3: IVF < LSH at every matched resolution.
    comparisons = [(key, table[key]['ivf']['median'] < table[key]['lsh']['median']) for key in table]
    ivf_wins = sum(1 for _, w in comparisons if w)
    print(f"H3 (IVF < LSH at every matched resolution): IVF beats LSH in {ivf_wins}/{len(comparisons)} combos "
          f"-> {'CONFIRMED' if ivf_wins == len(comparisons) else 'MOSTLY CONFIRMED, exceptions exist' if ivf_wins / len(comparisons) > 0.8 else 'REFUTED'}")
    exceptions = [key for key, w in comparisons if not w]
    if exceptions:
        print(f"    exceptions (LSH beat or tied IVF): {exceptions}")

    # H4: IVF declines faster with resolution than LSH.
    print("H4 (IVF's relative decline m=2 -> m=8 is larger than LSH's):")
    for d in D_VALUES:
        for c in N_CLUSTERS_VALUES:
            if (d, 2, c) not in table or (d, 8, c) not in table:
                continue
            lsh_decline = 1.0 - table[(d, 8, c)]['lsh']['median'] / table[(d, 2, c)]['lsh']['median']
            ivf_decline = 1.0 - table[(d, 8, c)]['ivf']['median'] / table[(d, 2, c)]['ivf']['median']
            faster = ivf_decline > lsh_decline
            print(f"    d={d}, n_clusters={c}: LSH decline={lsh_decline:.1%}, IVF decline={ivf_decline:.1%} "
                  f"-> {'IVF faster' if faster else 'LSH faster or equal'}")

    # H5: does IVF's advantage hold across d?
    print("H5 (IVF advantage across dimension d):")
    for d in D_VALUES:
        ratios = [table[(d, m, c)]['lsh']['median'] / table[(d, m, c)]['ivf']['median']
                  for m in M_VALUES for c in N_CLUSTERS_VALUES if (d, m, c) in table
                  and table[(d, m, c)]['ivf']['median'] > 0]
        print(f"    d={d}: mean LSH/IVF advantage ratio = {np.mean(ratios):.2f} (n={len(ratios)})")

    # H6: n_clusters=8 worsens overcounting for both methods; gap trend vs n_clusters.
    print("H6 (more clusters worsens overcounting; LSH-IVF gap trend vs n_clusters):")
    for d in D_VALUES:
        for m in M_VALUES:
            if (d, m, 1) not in table or (d, m, 8) not in table:
                continue
            lsh_worse = table[(d, m, 8)]['lsh']['median'] > table[(d, m, 1)]['lsh']['median']
            ivf_worse = table[(d, m, 8)]['ivf']['median'] > table[(d, m, 1)]['ivf']['median']
            ratios = [table[(d, m, c)]['lsh']['median'] / table[(d, m, c)]['ivf']['median']
                      for c in N_CLUSTERS_VALUES if (d, m, c) in table and table[(d, m, c)]['ivf']['median'] > 0]
            print(f"    d={d}, m={m}: LSH worse at c=8 vs c=1: {lsh_worse}, IVF worse: {ivf_worse}, "
                  f"gap ratio by n_clusters {dict(zip(N_CLUSTERS_VALUES, ['%.2f' % r for r in ratios]))}")

    # H7: false-signal rate/magnitude worse for LSH than IVF.
    fs_comparisons = [(key, table[key]['lsh']['false_signal_rate'] >= table[key]['ivf']['false_signal_rate'])
                       for key in table]
    lsh_worse_fs = sum(1 for _, w in fs_comparisons if w)
    print(f"H7 (LSH false-signal rate >= IVF's): true in {lsh_worse_fs}/{len(fs_comparisons)} combos "
          f"-> {'CONFIRMED' if lsh_worse_fs == len(fs_comparisons) else 'MOSTLY CONFIRMED' if lsh_worse_fs / len(fs_comparisons) > 0.8 else 'REFUTED'}")

#-------------------------------------------------------------------------------


def plot1_overcounting_grid(table: dict) -> Path:
    fig, axes = plt.subplots(len(N_CLUSTERS_VALUES), len(D_VALUES),
                              figsize=(4.5 * len(D_VALUES), 4.0 * len(N_CLUSTERS_VALUES)), squeeze=False)
    for row, n_clusters in enumerate(N_CLUSTERS_VALUES):
        for col, d in enumerate(D_VALUES):
            ax = axes[row][col]
            for method, color in (('lsh', 'tab:blue'), ('ivf', 'tab:orange')):
                medians = [table[(d, m, n_clusters)][method]['median'] for m in M_VALUES if (d, m, n_clusters) in table]
                q25 = [table[(d, m, n_clusters)][method]['q25'] for m in M_VALUES if (d, m, n_clusters) in table]
                q75 = [table[(d, m, n_clusters)][method]['q75'] for m in M_VALUES if (d, m, n_clusters) in table]
                ms = [m for m in M_VALUES if (d, m, n_clusters) in table]
                ax.plot(ms, medians, 'o-', color=color, label=method.upper())
                ax.fill_between(ms, q25, q75, color=color, alpha=0.2)
            ax.axhline(1.0, color='black', linestyle='--', linewidth=1)
            ax.set_yscale('log')
            ax.set_title(f'd={d}, n_clusters={n_clusters}')
            ax.set_xlabel('m')
            ax.set_ylabel('overcounting factor')
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
    fig.suptitle('Overcounting factor vs resolution (median, IQR band)')
    plt.tight_layout()
    out_path = PLOTS_DIR / 'plot1_overcounting_grid.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot2_false_signal_grid(table: dict) -> Path:
    fig, axes = plt.subplots(len(N_CLUSTERS_VALUES), len(D_VALUES),
                              figsize=(4.5 * len(D_VALUES), 4.0 * len(N_CLUSTERS_VALUES)), squeeze=False)
    for row, n_clusters in enumerate(N_CLUSTERS_VALUES):
        for col, d in enumerate(D_VALUES):
            ax = axes[row][col]
            for method, color in (('lsh', 'tab:blue'), ('ivf', 'tab:orange')):
                rates = [table[(d, m, n_clusters)][method]['false_signal_rate'] for m in M_VALUES if (d, m, n_clusters) in table]
                ms = [m for m in M_VALUES if (d, m, n_clusters) in table]
                ax.plot(ms, rates, 'o-', color=color, label=method.upper())
            ax.set_title(f'd={d}, n_clusters={n_clusters}')
            ax.set_xlabel('m')
            ax.set_ylabel('false-signal rate')
            ax.set_ylim(-0.02, 1.02)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
    fig.suptitle('False-signal rate (true_captured=0 but estimate>0) vs resolution')
    plt.tight_layout()
    out_path = PLOTS_DIR / 'plot2_false_signal_grid.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot3_advantage_ratio_grid(table: dict) -> Path:
    fig, axes = plt.subplots(len(N_CLUSTERS_VALUES), len(D_VALUES),
                              figsize=(4.5 * len(D_VALUES), 4.0 * len(N_CLUSTERS_VALUES)), squeeze=False)
    for row, n_clusters in enumerate(N_CLUSTERS_VALUES):
        for col, d in enumerate(D_VALUES):
            ax = axes[row][col]
            ms = [m for m in M_VALUES if (d, m, n_clusters) in table]
            ratios = [table[(d, m, n_clusters)]['lsh']['median'] / table[(d, m, n_clusters)]['ivf']['median']
                      for m in ms]
            ax.plot(ms, ratios, 'o-', color='tab:green')
            ax.axhline(1.0, color='black', linestyle='--', linewidth=1)
            ax.set_title(f'd={d}, n_clusters={n_clusters}')
            ax.set_xlabel('m')
            ax.set_ylabel('LSH/IVF median ratio')
    fig.suptitle('IVF advantage ratio (median LSH overcounting / median IVF overcounting)')
    plt.tight_layout()
    out_path = PLOTS_DIR / 'plot3_advantage_ratio_grid.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot4_distribution_sanity_check(table: dict) -> Path:
    d, n_clusters = 2, 3
    ms = [2, 4, 8]
    fig, ax = plt.subplots(figsize=(8, 5))
    positions, data, colors = [], [], []
    for i, m in enumerate(ms):
        for j, method in enumerate(('lsh', 'ivf')):
            positions.append(i * 3 + j)
            data.append(table[(d, m, n_clusters)][method]['overcounting_raw'])
            colors.append('tab:blue' if method == 'lsh' else 'tab:orange')
    parts = ax.violinplot(data, positions=positions, showmedians=True)
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    ax.set_xticks([i * 3 + 0.5 for i in range(len(ms))])
    ax.set_xticklabels([f'm={m}' for m in ms])
    ax.set_yscale('log')
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel('overcounting factor')
    ax.set_title(f'd={d}, n_clusters={n_clusters}: full overcounting distribution, LSH (blue) vs IVF (orange)')
    plt.tight_layout()
    out_path = PLOTS_DIR / 'plot4_distribution_sanity_check.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out_path

#-------------------------------------------------------------------------------


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Kappa ranges (loose, tight) per dimension:", KAPPA_RANGE)
    print("Cap radius per dimension:", CAP_RADIUS)

    table = build_summary_table()
    print(f"\nLoaded {len(table)} combos.\n")

    print_summary_table(table)
    check_hypotheses(table)

    paths = [
        plot1_overcounting_grid(table),
        plot2_false_signal_grid(table),
        plot3_advantage_ratio_grid(table),
        plot4_distribution_sanity_check(table),
    ]
    for p in paths:
        print(f"Wrote {p}")

    print("\nDone.")


if __name__ == '__main__':
    main()
