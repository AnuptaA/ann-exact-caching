#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Report for the full estimator comparison (M1 inverse-coverage, M2 vanilla
IF, M3 Bayesian LSH bucket-count, M4 hybrid) crossed with per-candidate (R1)
and shared-per-query (R2) radius regimes. Reads
estimator_comparison_{dataset}.npz.

Primary metric per H3's correction: all-pairs calibration (MAE, bias,
bin-level calibration error), not selected-point. Winner's-curse
decomposition (all-pairs vs selected-point bias) reported alongside, since
that's what separates a real estimator advantage from a selection artifact.
Everything stratified by true_recall band per H6. Degenerate-budget results
reported separately, never pooled."""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent.parent / 'results' / 'estimator_comparison'
OUT_DIR = Path(__file__).parent.parent / 'plots' / 'estimator_comparison'
BANDS = [(0.0, 0.3), (0.3, 0.8), (0.8, 1.01)]
BAND_LABELS = ['[0,0.3)', '[0.3,0.8)', '[0.8,1.0]']
MIN_BAND_N = 30

CELLS = ['M1_R1', 'M1_R2', 'M2_R1', 'M2_R2', 'M3_R2', 'M4_R1', 'M4_R2']
CELL_COLORS = {
    'M1_R1': 'tab:blue', 'M1_R2': 'tab:cyan', 'M2_R1': 'tab:red', 'M2_R2': 'tab:orange',
    'M3_R2': 'tab:green', 'M4_R1': 'tab:gray', 'M4_R2': 'tab:gray',
}
# Human-readable names for anything user-facing (titles/legends). Internal CELLS keys stay as-is.
CELL_LABELS = {
    'M1_R1': 'Inverse-Coverage (per-candidate radius)',
    'M1_R2': 'Inverse-Coverage (shared radius)',
    'M2_R1': 'Vanilla IF (per-candidate radius)',
    'M2_R2': 'Vanilla IF (shared radius)',
    'M3_R2': 'Bayesian LSH (shared radius)',
    'M4_R1': 'Hybrid (per-candidate radius) [unresolved bug]',
    'M4_R2': 'Hybrid (shared radius) [unresolved bug]',
}
BROKEN_CELLS = {'M4_R1', 'M4_R2'}  # rendered muted/dashed - known-broken reference, not comparable

# Corpus-fraction ranges each budget group covers, from this session's already-measured
# rc_size/n_base numbers - used only to make the "typical vs extreme" subplot titles concrete
# instead of unexplained jargon like "degenerate".
BUDGET_GROUP_INFO = {
    'sift': {'typical': 'budgets 0.1-0.4, entries cover 3-14% of corpus',
             'extreme': 'budget 0.99, entries cover ~55% of corpus'},
    'esci': {'typical': 'budgets 0.1-0.7, entries cover 3-27% of corpus',
             'extreme': 'budget 0.99, entries cover ~48% of corpus'},
}


def _style(cell: str) -> dict:
    if cell in BROKEN_CELLS:
        return dict(color='tab:gray', linestyle='--', alpha=0.5, zorder=1)
    return dict(color=CELL_COLORS[cell], linestyle='-', alpha=0.9, zorder=2)

#-------------------------------------------------------------------------------


def load(dataset: str) -> dict:
    return np.load(RESULTS_DIR / f'estimator_comparison_{dataset}.npz')


def _split_degenerate(d: dict) -> tuple:
    """Returns (nondeg_mask, deg_mask) over pair-level rows."""
    deg = float(d['degenerate_budget'])
    is_deg = np.isclose(d['pair_budget'], deg)
    return ~is_deg, is_deg

#-------------------------------------------------------------------------------


def report_all_pairs_calibration(dataset: str, d: dict, mask: np.ndarray, label: str) -> None:
    print(f"\n=== {dataset} ({label}): all-pairs calibration, primary metric (H3) ===")
    print(f"{'cell':>8} {'n':>7} {'MAE':>8} {'bias':>8} {'RMSE':>8}")
    for cell in CELLS:
        rh = d[f'pair_recall_hat_{cell}'][mask]
        tr = d['pair_true_recall'][mask]
        mae = np.mean(np.abs(rh - tr))
        bias = np.mean(rh - tr)
        rmse = np.sqrt(np.mean((rh - tr) ** 2))
        print(f"{cell:>8} {mask.sum():>7} {mae:>8.4f} {bias:>+8.4f} {rmse:>8.4f}")


def report_winners_curse(dataset: str, d: dict, label: str) -> None:
    """All-pairs bias vs selected-point bias, per cell - the decomposition
    that separates a real calibration advantage from a winner's-curse
    selection artifact (see this branch's prior round)."""
    print(f"\n=== {dataset} ({label}): winner's-curse decomposition (H2) ===")
    deg = float(d['degenerate_budget'])
    is_deg = np.isclose(d['pair_budget'], deg)
    pair_mask = is_deg if label == 'degenerate' else ~is_deg
    q_is_deg = np.isclose(d['query_budget'], deg)
    q_mask = q_is_deg if label == 'degenerate' else ~q_is_deg

    print(f"{'cell':>8} {'all-pairs bias':>15} {'selected bias':>15} {'shift':>10}")
    for cell in CELLS:
        rh = d[f'pair_recall_hat_{cell}'][pair_mask]
        tr = d['pair_true_recall'][pair_mask]
        bias_all = np.mean(rh - tr)

        sel_rh = d[f'query_selected_recall_hat_{cell}'][q_mask]
        sel_tr = d[f'query_selected_true_recall_{cell}'][q_mask]
        bias_sel = np.mean(sel_rh - sel_tr)
        print(f"{cell:>8} {bias_all:>+15.4f} {bias_sel:>+15.4f} {bias_sel - bias_all:>+10.4f}")


def report_stratified(dataset: str, d: dict, mask: np.ndarray, label: str) -> None:
    print(f"\n=== {dataset} ({label}): stratified correlation, selected candidate (H6) ===")
    q_deg = np.isclose(d['query_budget'], float(d['degenerate_budget']))
    q_mask = q_deg if label == 'degenerate' else ~q_deg
    for cell in CELLS:
        tr = d[f'query_selected_true_recall_{cell}'][q_mask]
        rh = d[f'query_selected_recall_hat_{cell}'][q_mask]
        parts = []
        for (lo, hi), lbl in zip(BANDS, BAND_LABELS):
            band = (tr >= lo) & (tr < hi)
            n = int(band.sum())
            if n < 2:
                parts.append(f"{lbl} n={n} corr=n/a")
            elif n < MIN_BAND_N:
                corr = np.corrcoef(rh[band], tr[band])[0, 1]
                parts.append(f"{lbl} n={n} corr={corr:+.3f} (UNDERPOWERED)")
            else:
                corr = np.corrcoef(rh[band], tr[band])[0, 1]
                parts.append(f"{lbl} n={n} corr={corr:+.3f}")
        print(f"  {cell:>8}: " + " | ".join(parts))


def report_regret(dataset: str, d: dict, label: str) -> None:
    print(f"\n=== {dataset} ({label}): oracle regret (mean/p50/p90) ===")
    q_deg = np.isclose(d['query_budget'], float(d['degenerate_budget']))
    q_mask = q_deg if label == 'degenerate' else ~q_deg
    for cell in CELLS:
        sel = d[f'query_selected_true_recall_{cell}'][q_mask]
        oracle = d[f'query_oracle_best_true_recall_{cell}'][q_mask]
        regret = oracle - sel
        print(f"  {cell:>8}: n={len(regret)} mean={regret.mean():.4f} "
              f"p50={np.median(regret):.4f} p90={np.percentile(regret, 90):.4f}")

#-------------------------------------------------------------------------------
# Plots
#-------------------------------------------------------------------------------


def _q_mask(d: dict, label: str) -> np.ndarray:
    q_deg = np.isclose(d['query_budget'], float(d['degenerate_budget']))
    return q_deg if label == 'degenerate' else ~q_deg


def _draw_calibration(ax, d: dict, q_mask: np.ndarray, n_bins: int = 8) -> None:
    for cell in CELLS:
        rh = d[f'query_selected_recall_hat_{cell}'][q_mask]
        tr = d[f'query_selected_true_recall_{cell}'][q_mask]
        if len(rh) < n_bins:
            continue
        order = np.argsort(rh)
        bins = np.array_split(order, n_bins)
        bx, my, sy = [], [], []
        for idx in bins:
            if len(idx) == 0:
                continue
            bx.append(rh[idx].mean())
            my.append(tr[idx].mean())
            sy.append(tr[idx].std(ddof=0) / np.sqrt(len(idx)))
        style = _style(cell)
        ax.errorbar(bx, my, yerr=sy, fmt='o', capsize=2, markersize=4, linewidth=1.3,
                    label=CELL_LABELS[cell], **style)
    ax.plot([0, 1], [0, 1], color='black', linewidth=1, linestyle=':', alpha=0.6, label='y = x (perfect)')
    ax.set_xlabel('mean estimated recall in bin')
    ax.set_ylabel('mean true recall (actual)')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)


def plot_calibration(dataset: str, d: dict, out_dir: Path) -> None:
    """Binned calibration curve, typical vs extreme cache size side by side -
    'closer to the dotted y=x line is better calibrated'. Known-broken cells
    (Hybrid) render dashed gray so they read as a reference, not a
    competitor, per this round's finding that they need a redesign first."""
    info = BUDGET_GROUP_INFO[dataset]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    _draw_calibration(axes[0], d, ~_q_mask(d, 'degenerate'))
    axes[0].set_title(f"Typical cache size ({info['typical']})")
    _draw_calibration(axes[1], d, _q_mask(d, 'degenerate'))
    axes[1].set_title(f"Extreme cache size ({info['extreme']})")
    axes[0].legend(fontsize=8, loc='upper left')
    fig.suptitle(f'{dataset}: calibration (mean estimated vs mean actual recall)')
    plt.tight_layout()
    out_path = out_dir / f'calibration_{dataset}.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_path}")


def _draw_ambiguous_band(ax, d: dict, q_mask: np.ndarray) -> None:
    lo, hi = 0.3, 0.8
    markers = ['o', '^', 's', 'D', 'v', 'P', 'X']
    for cell, marker in zip(CELLS, markers):
        rh = d[f'query_selected_recall_hat_{cell}'][q_mask]
        tr = d[f'query_selected_true_recall_{cell}'][q_mask]
        band = (tr >= lo) & (tr < hi)
        n = int(band.sum())
        if n < 2:
            continue
        corr = np.corrcoef(rh[band], tr[band])[0, 1] if rh[band].std() > 0 else float('nan')
        flag = ' (few samples)' if n < MIN_BAND_N else ''
        style = _style(cell)
        ax.scatter(rh[band], tr[band], s=24, marker=marker, color=style['color'], alpha=style['alpha'],
                   label=f'{CELL_LABELS[cell]}: n={n}, corr={corr:+.2f}{flag}')
    ax.plot([lo, hi], [lo, hi], color='black', linewidth=1, linestyle='--', alpha=0.6)
    ax.set_xlabel('estimated recall')
    ax.set_ylabel('true recall (actual)')
    ax.grid(alpha=0.3)


def plot_ambiguous_band(dataset: str, d: dict, out_dir: Path) -> None:
    """Restricted to true_recall in [0.3,0.8) - the genuinely hard cases a
    hit policy needs to discriminate, not the easy obviously-good/bad ones
    that inflate pooled correlation (a repeated finding on this branch)."""
    info = BUDGET_GROUP_INFO[dataset]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
    _draw_ambiguous_band(axes[0], d, ~_q_mask(d, 'degenerate'))
    axes[0].set_title(f"Typical cache size ({info['typical']})")
    _draw_ambiguous_band(axes[1], d, _q_mask(d, 'degenerate'))
    axes[1].set_title(f"Extreme cache size ({info['extreme']})")
    axes[0].legend(fontsize=7, loc='best')
    fig.suptitle(f'{dataset}: the hard cases only (true recall between 0.3 and 0.8)')
    plt.tight_layout()
    out_path = out_dir / f'ambiguous_band_{dataset}.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_path}")


def _draw_regret_bar(ax, d: dict, q_mask: np.ndarray) -> None:
    means, sems, p90s = [], [], []
    for cell in CELLS:
        sel = d[f'query_selected_true_recall_{cell}'][q_mask]
        oracle = d[f'query_oracle_best_true_recall_{cell}'][q_mask]
        regret = oracle - sel
        means.append(regret.mean())
        sems.append(regret.std(ddof=0) / np.sqrt(len(regret)))
        p90s.append(np.percentile(regret, 90))
    x = np.arange(len(CELLS))
    colors = [_style(c)['color'] for c in CELLS]
    alphas = [_style(c)['alpha'] for c in CELLS]
    for xi, m, s, a, c in zip(x, means, sems, alphas, colors):
        ax.bar(xi, m, yerr=s, capsize=4, color=c, alpha=a)
    ax.scatter(x, p90s, color='black', marker='_', s=150, zorder=3, label='p90 regret')
    ax.set_xticks(x)
    ax.set_xticklabels([CELL_LABELS[c] for c in CELLS], rotation=30, ha='right', fontsize=7)
    ax.set_ylabel('oracle regret (lower is better)')
    ax.grid(alpha=0.3, axis='y')


def _draw_bias_bar(ax, d: dict, pair_mask: np.ndarray, q_mask: np.ndarray) -> None:
    x = np.arange(len(CELLS))
    width = 0.35
    bias_all, bias_sel = [], []
    for cell in CELLS:
        rh, tr = d[f'pair_recall_hat_{cell}'][pair_mask], d['pair_true_recall'][pair_mask]
        bias_all.append(np.mean(rh - tr))
        sel_rh = d[f'query_selected_recall_hat_{cell}'][q_mask]
        sel_tr = d[f'query_selected_true_recall_{cell}'][q_mask]
        bias_sel.append(np.mean(sel_rh - sel_tr))
    ax.bar(x - width / 2, bias_all, width, label='bias, all candidates evaluated', color='tab:gray', alpha=0.85)
    ax.bar(x + width / 2, bias_sel, width, label='bias, only the one served', color='tab:red', alpha=0.85)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([CELL_LABELS[c] for c in CELLS], rotation=30, ha='right', fontsize=7)
    ax.set_ylabel('bias (estimated - actual recall)')
    ax.grid(alpha=0.3, axis='y')


def plot_regret_and_bias(dataset: str, d: dict, out_dir: Path) -> None:
    """Top row: how much recall is left on the table by trusting each
    estimator's pick over the best available candidate. Bottom row: whether
    picking the best-looking candidate (vs. averaging over all of them)
    makes an estimator's bias look artificially better - the winner's-curse
    check from this branch's prior round."""
    info = BUDGET_GROUP_INFO[dataset]
    deg = float(d['degenerate_budget'])
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    _draw_regret_bar(axes[0, 0], d, ~_q_mask(d, 'degenerate'))
    axes[0, 0].set_title(f"Regret - typical cache size ({info['typical']})")
    _draw_regret_bar(axes[0, 1], d, _q_mask(d, 'degenerate'))
    axes[0, 1].set_title(f"Regret - extreme cache size ({info['extreme']})")
    axes[0, 0].legend(fontsize=8)

    _draw_bias_bar(axes[1, 0], d, ~np.isclose(d['pair_budget'], deg), ~_q_mask(d, 'degenerate'))
    axes[1, 0].set_title(f"Bias - typical cache size ({info['typical']})")
    _draw_bias_bar(axes[1, 1], d, np.isclose(d['pair_budget'], deg), _q_mask(d, 'degenerate'))
    axes[1, 1].set_title(f"Bias - extreme cache size ({info['extreme']})")
    axes[1, 0].legend(fontsize=8)

    fig.suptitle(f'{dataset}: regret from trusting the estimate, and whether "only look at the '
                 f'winner" flatters the bias')
    plt.tight_layout()
    out_path = out_dir / f'regret_and_bias_{dataset}.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_all(dataset: str, d: dict, out_dir: Path) -> None:
    plot_calibration(dataset, d, out_dir)
    plot_ambiguous_band(dataset, d, out_dir)
    plot_regret_and_bias(dataset, d, out_dir)

#-------------------------------------------------------------------------------


def report_dataset(dataset: str) -> None:
    path = RESULTS_DIR / f'estimator_comparison_{dataset}.npz'
    if not path.exists():
        print(f"skipping {dataset}: {path} not found")
        return
    d = load(dataset)
    nondeg_mask, deg_mask = _split_degenerate(d)

    report_all_pairs_calibration(dataset, d, nondeg_mask, 'non-degenerate')
    report_winners_curse(dataset, d, 'non-degenerate')
    report_stratified(dataset, d, nondeg_mask, 'non-degenerate')
    report_regret(dataset, d, 'non-degenerate')

    print(f"\n{'='*70}\n{dataset}: DEGENERATE BUDGET ({float(d['degenerate_budget'])}) - "
          f"reported separately, not pooled above\n{'='*70}")
    report_all_pairs_calibration(dataset, d, deg_mask, 'degenerate')
    report_winners_curse(dataset, d, 'degenerate')
    report_stratified(dataset, d, deg_mask, 'degenerate')
    report_regret(dataset, d, 'degenerate')

    plot_all(dataset, d, OUT_DIR)

#-------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', default=['sift', 'esci'])
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        report_dataset(dataset)


if __name__ == '__main__':
    main()
