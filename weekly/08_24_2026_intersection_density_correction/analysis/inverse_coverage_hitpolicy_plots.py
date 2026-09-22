#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Report for the real-population inverse-coverage hit policy (see
../mechanism_spec_v2.md and
../schemes/inverse_coverage_population_hitpolicy_run_experiment.py). Reads
hitpolicy_{dataset}.npz's per (query, candidate) pair arrays and reduces them,
per query, to the argmax-by-recall_hat candidate - the one the hit policy
would actually serve - not all evaluated candidates pooled together.

H2: recall_hat-vs-true_recall correlation of the selected candidate, across
all queries (reported, not primary).
H2b: same, restricted to queries where the selected candidate's true_recall
falls in [0.3, 0.8) - the primary result.
H3: oracle regret distribution (mean/p50/p90)."""

import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'schemes'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from inverse_coverage_population_hitpolicy_run_experiment import (
    DATASETS, BUDGET_FRACTION, BUDGET_SWEEP_LEVELS, SEED,
    load_population_inputs, propose_and_select, extract_entries,
)

#-------------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent.parent / 'results' / 'inverse_coverage_feasibility'
OUT_DIR = Path(__file__).parent.parent / 'plots' / 'inverse_coverage_hitpolicy'
AMBIGUOUS_BAND = (0.3, 0.8)

#-------------------------------------------------------------------------------


def per_query_selected(d: dict, estimate_key: str = 'recall_hat') -> dict:
    """Reduces the raw (query, candidate) pair arrays to one row per query:
    the argmax-by-estimate_key candidate (what that estimator's hit policy
    would actually serve) and, for comparison, the max-true_recall candidate
    (oracle best). estimate_key lets the same reduction serve either
    inverse-coverage ('recall_hat') or vanilla IF ('recall_hat_if') - each
    estimator picks its own candidate independently, since a fair comparison
    means each is evaluated as its own hit policy, not both scored against
    inverse-coverage's pick. Derived directly from the pair arrays rather
    than the separately saved selected_true_recall summaries, to avoid any
    dependence on their storage order matching query_idx."""
    query_idx = d['query_idx']
    estimate, true_recall = d[estimate_key], d['true_recall']
    unique_q = np.unique(query_idx)

    sel_estimate = np.empty(len(unique_q))
    sel_true_recall = np.empty(len(unique_q))
    oracle_true_recall = np.empty(len(unique_q))
    for i, qi in enumerate(unique_q):
        mask = query_idx == qi
        est, tr = estimate[mask], true_recall[mask]
        amax = np.argmax(est)
        sel_estimate[i] = est[amax]
        sel_true_recall[i] = tr[amax]
        oracle_true_recall[i] = tr.max()

    return dict(query_idx=unique_q, recall_hat=sel_estimate, true_recall=sel_true_recall,
                oracle_true_recall=oracle_true_recall)


def report_h2_h2b(dataset: str, sel: dict) -> None:
    print(f"\n=== {dataset}: H2 / H2b - selected candidate's recall_hat vs true_recall ===")
    recall_hat, true_recall = sel['recall_hat'], sel['true_recall']
    n = len(true_recall)
    corr_pooled = np.corrcoef(recall_hat, true_recall)[0, 1]
    print(f"  n_queries={n}, H2 corr(recall_hat, true_recall) = {corr_pooled:+.4f}")

    lo, hi = AMBIGUOUS_BAND
    mask = (true_recall >= lo) & (true_recall < hi)
    n_band = mask.sum()
    if n_band >= 2:
        corr_band = np.corrcoef(recall_hat[mask], true_recall[mask])[0, 1]
        print(f"  H2b band [{lo},{hi}) n={n_band} corr(recall_hat, true_recall) = {corr_band:+.4f}")
    else:
        print(f"  H2b band [{lo},{hi}) n={n_band} - too few queries to correlate")


def report_h3_regret(dataset: str, sel: dict) -> None:
    print(f"\n=== {dataset}: H3 - oracle regret (selected argmax vs oracle-best) ===")
    selected, oracle = sel['true_recall'], sel['oracle_true_recall']
    regret = oracle - selected
    n = len(regret)
    print(f"  n_queries={n}")
    print(f"  selected_true_recall: mean={selected.mean():.4f} p50={np.median(selected):.4f}")
    print(f"  oracle_best_true_recall: mean={oracle.mean():.4f} p50={np.median(oracle):.4f}")
    print(f"  regret: mean={regret.mean():.4f} p50={np.median(regret):.4f} "
          f"p90={np.percentile(regret, 90):.4f} max={regret.max():.4f}")


def report_min_coverage_summary(dataset: str, d: dict) -> None:
    min_cov = d['min_coverage']
    trust = min_cov >= 0.5
    print(f"\n=== {dataset}: min_coverage summary (recorded, not gated this pass) ===")
    print(f"  mean={min_cov.mean():.6f} median={np.median(min_cov):.6f} "
          f"trustworthy_frac(>=0.5)={trust.mean():.4f} ({trust.sum()}/{len(trust)})")

#-------------------------------------------------------------------------------


def plot_recall_hat_vs_true_recall(dataset: str, sel: dict, out_dir: Path) -> None:
    """x = estimated recall_hat of the selected candidate, y = its actual
    true_recall - one point per query, not one per (query, candidate) pair."""
    recall_hat, true_recall = sel['recall_hat'], sel['true_recall']
    lo, hi = AMBIGUOUS_BAND
    band_mask = (true_recall >= lo) & (true_recall < hi)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.scatter(recall_hat, true_recall, s=18, alpha=0.5, color='tab:blue')
    ax.plot([0, 1], [0, 1], color='black', linewidth=1, linestyle='--', alpha=0.6, label='y=x')
    corr = np.corrcoef(recall_hat, true_recall)[0, 1]
    ax.set_xlabel('recall_hat (estimated)')
    ax.set_ylabel('true_recall (actual)')
    ax.set_title(f'{dataset}: selected candidate, all queries (n={len(recall_hat)}), corr={corr:+.3f}')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    n_band = band_mask.sum()
    if n_band >= 2:
        ax.scatter(recall_hat[band_mask], true_recall[band_mask], s=22, alpha=0.6, color='tab:orange')
        corr_band = np.corrcoef(recall_hat[band_mask], true_recall[band_mask])[0, 1]
    else:
        corr_band = float('nan')
    ax.plot([lo, hi], [lo, hi], color='black', linewidth=1, linestyle='--', alpha=0.6, label='y=x')
    ax.set_xlabel('recall_hat (estimated)')
    ax.set_ylabel('true_recall (actual)')
    ax.set_title(f'{dataset}: selected, ambiguous band [{lo},{hi}) (n={n_band}), corr={corr_band:+.3f}')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / f'recall_hat_vs_true_recall_{dataset}.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_binned(dataset: str, sel: dict, out_dir: Path, n_bins: int = 8) -> None:
    """Bins queries by the selected candidate's recall_hat (equal-count bins)
    and plots, per bin, the mean +/- SEM of the selected candidate's actual
    true_recall alongside the mean +/- SEM of the oracle-best true_recall
    available for that same query - shows how much headroom above the
    served estimate the oracle leaves, as a function of confidence level."""
    recall_hat, true_recall = sel['recall_hat'], sel['true_recall']
    oracle = sel['oracle_true_recall']
    n = len(recall_hat)
    n_bins = max(1, min(n_bins, n))

    order = np.argsort(recall_hat)
    bin_edges = np.array_split(order, n_bins)

    bin_x, sel_mean, sel_sem, oracle_mean, oracle_sem, bin_n = [], [], [], [], [], []
    for idx in bin_edges:
        if len(idx) == 0:
            continue
        bin_x.append(recall_hat[idx].mean())
        sel_mean.append(true_recall[idx].mean())
        sel_sem.append(true_recall[idx].std(ddof=0) / np.sqrt(len(idx)))
        oracle_mean.append(oracle[idx].mean())
        oracle_sem.append(oracle[idx].std(ddof=0) / np.sqrt(len(idx)))
        bin_n.append(len(idx))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.errorbar(bin_x, sel_mean, yerr=sel_sem, fmt='o-', color='tab:blue',
                label='selected (served) true_recall', capsize=3)
    ax.errorbar(bin_x, oracle_mean, yerr=oracle_sem, fmt='s--', color='tab:green',
                label='oracle-best true_recall', capsize=3)
    ax.plot([0, 1], [0, 1], color='black', linewidth=1, linestyle=':', alpha=0.6, label='y=x')
    ax.set_xlabel('mean recall_hat in bin (estimated)')
    ax.set_ylabel('mean true_recall (actual)')
    ax.set_title(f'{dataset}: binned by recall_hat, {len(bin_x)} bins '
                 f'(~{bin_n[0] if bin_n else 0}/bin, n={n} queries)')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / f'binned_recall_hat_{dataset}.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_regret(dataset: str, sel: dict, out_dir: Path) -> None:
    regret = sel['oracle_true_recall'] - sel['true_recall']
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(regret, bins=30, color='tab:red', alpha=0.75)
    ax.axvline(regret.mean(), color='black', linestyle='--', linewidth=1,
               label=f'mean={regret.mean():.4f}')
    ax.axvline(np.percentile(regret, 90), color='gray', linestyle=':', linewidth=1,
               label=f'p90={np.percentile(regret, 90):.4f}')
    ax.set_xlabel('regret (oracle_best_true_recall - selected_true_recall)')
    ax.set_ylabel('count')
    ax.set_title(f'{dataset}: oracle regret (n={len(regret)} queries)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / f'regret_{dataset}.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_min_coverage(dataset: str, d: dict, out_dir: Path) -> None:
    min_cov = d['min_coverage']
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(min_cov, bins=40, color='tab:green', alpha=0.75)
    ax.axvline(0.5, color='black', linestyle='--', linewidth=1, label='trustworthy threshold (0.5)')
    trust_frac = (min_cov >= 0.5).mean()
    ax.set_xlabel('min_coverage')
    ax.set_ylabel('count')
    ax.set_title(f'{dataset}: min_coverage distribution (trustworthy frac={trust_frac:.3f})')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / f'min_coverage_{dataset}.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_rc_size(dataset: str, out_dir: Path, seed: int = SEED) -> None:
    """Regenerates the real greedy-selected candidates (cheap - population
    construction, not the per-query evaluation loop) purely to plot the
    rc_size distribution, which isn't saved in hitpolicy_{dataset}.npz."""
    cfg = DATASETS[dataset]
    D, K = cfg['D'], cfg['K']
    repo_root = Path(__file__).parent.parent.parent.parent
    inputs = load_population_inputs(dataset, repo_root)
    selected = propose_and_select(inputs, D, K, BUDGET_FRACTION, seed)
    entries = extract_entries(selected)
    rc_sizes = np.array([e['rc_size'] for e in entries])
    n_base = inputs['base'].shape[0]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(rc_sizes, bins=30, color='tab:purple', alpha=0.75)
    ax.axvline(rc_sizes.mean(), color='black', linestyle='--', linewidth=1,
               label=f'mean={rc_sizes.mean():.0f} ({rc_sizes.mean()/n_base:.1%} of corpus)')
    ax.set_xlabel('rc_size (stored points per candidate)')
    ax.set_ylabel('count')
    ax.set_title(f'{dataset}: candidate storage size (n_candidates={len(entries)}, n_base={n_base})')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / f'rc_size_{dataset}.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_path}")

#-------------------------------------------------------------------------------
# Budget sweep aggregation
#-------------------------------------------------------------------------------


def per_budget_selected(sweep_d: dict, estimate_key: str = 'recall_hat') -> dict:
    """Splits the budget-sweep pair arrays by budget_fraction and reduces each
    slice to one row per query (argmax-selected + oracle-best), via the same
    reduction as per_query_selected. Grouping must be by (query_idx, budget),
    not query_idx alone, since the same query index is evaluated separately
    against each budget level's own candidate set."""
    budget_levels = sweep_d['budget_levels']
    n_candidates_by_budget = sweep_d['n_candidates_by_budget']
    pair_budget = sweep_d['pair_budget_fraction']

    out = {}
    for b, n_cand in zip(budget_levels, n_candidates_by_budget):
        b = float(b)
        mask = np.isclose(pair_budget, b)
        if mask.sum() == 0:
            out[b] = dict(recall_hat=np.array([]), true_recall=np.array([]),
                           oracle_true_recall=np.array([]), n_candidates=int(n_cand))
            continue
        sub = dict(query_idx=sweep_d['query_idx'][mask], cand_idx=sweep_d['cand_idx'][mask],
                   recall_hat=sweep_d['recall_hat'][mask], recall_hat_if=sweep_d['recall_hat_if'][mask],
                   true_recall=sweep_d['true_recall'][mask])
        sel = per_query_selected(sub, estimate_key=estimate_key)
        sel['n_candidates'] = int(n_cand)
        out[b] = sel
    return out


def load_combined_budget_data(dataset: str, estimate_key: str = 'recall_hat') -> dict:
    """per_budget_selected on the sweep results (0.01-0.9), plus the existing
    single-budget hitpolicy_{dataset}.npz (0.99) tagged in as one more level -
    that run is not repeated, per the standing instruction not to rerun 0.99."""
    sweep_path = RESULTS_DIR / f'hitpolicy_budget_sweep_{dataset}.npz'
    base_path = RESULTS_DIR / f'hitpolicy_{dataset}.npz'
    combined = {}
    if sweep_path.exists():
        combined.update(per_budget_selected(np.load(sweep_path), estimate_key=estimate_key))
    if base_path.exists():
        base_d = np.load(base_path)
        base_sel = per_query_selected(base_d, estimate_key=estimate_key)
        base_sel['n_candidates'] = int(base_d['n_candidates'])
        combined[0.99] = base_sel
    return combined


def report_budget_sweep_table(dataset: str, combined: dict) -> None:
    """H2 (pooled per level) is reported alongside H2b (restricted to the
    ambiguous true_recall in [0.3, 0.8) band) because H2 alone is misleading
    here: with few distinct candidates per query, the argmax pick is often
    either clearly right or clearly wrong, and that corner separation alone
    can produce a high Pearson r with no real discrimination in between -
    H2b is what actually tests discrimination in the cases that matter."""
    print(f"\n=== {dataset}: budget sweep - H2 (pooled) vs H2b (ambiguous band) per level ===")
    print(f"{'budget':>8} {'n_cand':>8} {'n_q':>6} {'n_uniq_rh':>10} {'H2':>9} {'n_band':>7} {'H2b':>9}")
    for b in sorted(combined):
        s = combined[b]
        rh, tr = s['recall_hat'], s['true_recall']
        n = len(rh)
        n_uniq = len(np.unique(np.round(rh, 6))) if n else 0
        h2 = np.corrcoef(rh, tr)[0, 1] if n >= 2 else float('nan')
        lo, hi = AMBIGUOUS_BAND
        band = (tr >= lo) & (tr < hi)
        h2b = np.corrcoef(rh[band], tr[band])[0, 1] if band.sum() >= 2 else float('nan')
        print(f"{b:>8.2f} {s['n_candidates']:>8} {n:>6} {n_uniq:>10} {h2:>+9.4f} {band.sum():>7} {h2b:>+9.4f}")

    all_rh = np.concatenate([combined[b]['recall_hat'] for b in combined if len(combined[b]['recall_hat'])])
    all_tr = np.concatenate([combined[b]['true_recall'] for b in combined if len(combined[b]['true_recall'])])
    lo, hi = AMBIGUOUS_BAND
    band = (all_tr >= lo) & (all_tr < hi)
    print(f"  pooled across all budgets: n={len(all_rh)}, H2={np.corrcoef(all_rh, all_tr)[0, 1]:+.4f}, "
          f"H2b (n={band.sum()})={np.corrcoef(all_rh[band], all_tr[band])[0, 1] if band.sum() >= 2 else float('nan'):+.4f}")


def compute_candidate_stats_by_budget(dataset: str, budget_levels, seed: int = SEED) -> dict:
    """Regenerates candidates at each budget level (cheap - population
    construction only, confirmed in the sweep pilot) purely to get mean
    rc_size per level, which isn't saved in the sweep npz (only n_candidates
    and per-query recall_hat/true_recall are)."""
    cfg = DATASETS[dataset]
    D, K = cfg['D'], cfg['K']
    repo_root = Path(__file__).parent.parent.parent.parent
    inputs = load_population_inputs(dataset, repo_root)
    stats = {}
    for b in budget_levels:
        selected = propose_and_select(inputs, D, K, b, seed)
        entries = extract_entries(selected)
        rc_sizes = np.array([e['rc_size'] for e in entries])
        stats[float(b)] = dict(n_candidates=len(entries),
                                mean_rc_size=float(rc_sizes.mean()) if len(rc_sizes) else 0.0)
    return stats


def pooled_arrays(combined: dict) -> dict:
    """Concatenates recall_hat/true_recall/oracle_true_recall across every
    budget level with data - the pooled-across-budget dataset used by the
    binned and ambiguous-band plots."""
    rh = np.concatenate([combined[b]['recall_hat'] for b in combined if len(combined[b]['recall_hat'])])
    tr = np.concatenate([combined[b]['true_recall'] for b in combined if len(combined[b]['true_recall'])])
    oracle = np.concatenate([combined[b]['oracle_true_recall'] for b in combined
                              if len(combined[b]['oracle_true_recall'])])
    return dict(recall_hat=rh, true_recall=tr, oracle_true_recall=oracle)


def plot_ambiguous_band_scatter(dataset: str, combined: dict, out_dir: Path) -> None:
    """The plot that actually matters: restricted to true_recall in the
    ambiguous [0.3, 0.8) band, pooled across every budget level for real
    statistical power - the pooled all-data correlation is inflated by
    corner separation (obviously-bad vs obviously-good picks) and isn't a
    fair read of whether recall_hat discriminates within genuinely hard
    cases."""
    pooled = pooled_arrays(combined)
    rh, tr = pooled['recall_hat'], pooled['true_recall']
    lo, hi = AMBIGUOUS_BAND
    band = (tr >= lo) & (tr < hi)
    n_band = band.sum()

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if n_band >= 2:
        ax.scatter(rh[band], tr[band], s=20, alpha=0.5, color='tab:orange')
        corr = np.corrcoef(rh[band], tr[band])[0, 1]
    else:
        corr = float('nan')
    ax.plot([lo, hi], [lo, hi], color='black', linewidth=1, linestyle='--', alpha=0.6, label='y=x')
    ax.set_xlabel('recall_hat (estimated)')
    ax.set_ylabel('true_recall (actual)')
    ax.set_title(f'{dataset}: ambiguous band [{lo},{hi}), pooled across budgets '
                 f'(n={n_band}), corr={corr:+.3f}')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / f'budget_sweep_ambiguous_band_{dataset}.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_pooled_budget_scatter(dataset: str, combined: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    budgets = sorted(combined)
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(budgets), vmax=max(budgets))

    all_rh, all_tr = [], []
    for b in budgets:
        s = combined[b]
        if len(s['recall_hat']) == 0:
            continue
        ax.scatter(s['recall_hat'], s['true_recall'], s=22, alpha=0.6, color=cmap(norm(b)))
        all_rh.append(s['recall_hat'])
        all_tr.append(s['true_recall'])

    ax.plot([0, 1], [0, 1], color='black', linewidth=1, linestyle='--', alpha=0.6, label='y=x')
    if all_rh:
        rh, tr = np.concatenate(all_rh), np.concatenate(all_tr)
        corr = np.corrcoef(rh, tr)[0, 1]
        n = len(rh)
    else:
        corr, n = float('nan'), 0

    ax.set_xlabel('recall_hat (estimated)')
    ax.set_ylabel('true_recall (actual)')
    ax.set_title(f'{dataset}: pooled across {len(budgets)} budget levels (n={n}), corr={corr:+.3f}')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('budget_fraction')
    plt.tight_layout()
    out_path = out_dir / f'budget_sweep_pooled_scatter_{dataset}.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_regret_by_budget(dataset: str, combined: dict, out_dir: Path) -> None:
    budgets = sorted(combined)
    ncols = min(5, len(budgets))
    nrows = int(np.ceil(len(budgets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3 * nrows), squeeze=False)

    for i, b in enumerate(budgets):
        ax = axes[i // ncols][i % ncols]
        s = combined[b]
        if len(s['recall_hat']) == 0:
            ax.set_title(f'budget={b:.2f}\n(no candidates)')
            ax.axis('off')
            continue
        regret = s['oracle_true_recall'] - s['true_recall']
        ax.hist(regret, bins=15, color='tab:red', alpha=0.75)
        ax.axvline(regret.mean(), color='black', linestyle='--', linewidth=1)
        ax.set_title(f'budget={b:.2f} (n={len(regret)})\nmean regret={regret.mean():.3f}')
        ax.set_xlabel('regret')

    for j in range(len(budgets), nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')
    fig.suptitle(f'{dataset}: oracle regret by budget level')
    plt.tight_layout()
    out_path = out_dir / f'budget_sweep_regret_grid_{dataset}.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_budget_relationship(dataset: str, combined: dict, candidate_stats: dict, out_dir: Path) -> None:
    budgets = sorted(combined)
    n_cand = [candidate_stats[b]['n_candidates'] for b in budgets]
    rc_size = [candidate_stats[b]['mean_rc_size'] for b in budgets]
    corr, regret = [], []
    for b in budgets:
        s = combined[b]
        corr.append(np.corrcoef(s['recall_hat'], s['true_recall'])[0, 1] if len(s['recall_hat']) >= 2 else np.nan)
        regret.append((s['oracle_true_recall'] - s['true_recall']).mean() if len(s['recall_hat']) >= 1 else np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(budgets, n_cand, 'o-', color='tab:purple')
    axes[0].set_xlabel('budget_fraction')
    axes[0].set_ylabel('n_candidates')
    axes[0].set_title(f'{dataset}: n_candidates vs budget')
    axes[0].grid(alpha=0.3)

    axes[1].plot(budgets, rc_size, 'o-', color='tab:orange')
    axes[1].set_xlabel('budget_fraction')
    axes[1].set_ylabel('mean rc_size')
    axes[1].set_title(f'{dataset}: mean rc_size vs budget')
    axes[1].grid(alpha=0.3)

    ax2 = axes[2]
    l1 = ax2.plot(budgets, corr, 'o-', color='tab:blue', label='corr(recall_hat, true_recall)')
    ax2.set_xlabel('budget_fraction')
    ax2.set_ylabel('correlation')
    ax2.grid(alpha=0.3)
    ax2b = ax2.twinx()
    l2 = ax2b.plot(budgets, regret, 's--', color='tab:red', label='mean regret')
    ax2b.set_ylabel('mean regret')
    lines = l1 + l2
    ax2.legend(lines, [line.get_label() for line in lines], loc='best', fontsize=8)
    ax2.set_title(f'{dataset}: correlation & regret vs budget')

    plt.tight_layout()
    out_path = out_dir / f'budget_sweep_relationship_{dataset}.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_pooled_binned(dataset: str, combined: dict, out_dir: Path, n_bins: int = 10) -> None:
    """The binned recall_hat-vs-true_recall view, done on the pooled-across-
    budget data - reuses the same binning/errorbar approach as plot_binned
    (the single-budget version), just fed the wider, more representative
    spread of recall_hat values the sweep produces."""
    pooled = pooled_arrays(combined)
    plot_binned(f'{dataset}_budget_sweep_pooled', pooled, out_dir, n_bins=n_bins)


def report_and_plot_budget_sweep(dataset: str, out_dir: Path = OUT_DIR,
                                  include_candidate_stats: bool = False) -> None:
    sweep_path = RESULTS_DIR / f'hitpolicy_budget_sweep_{dataset}.npz'
    if not sweep_path.exists():
        print(f"skipping {dataset} budget sweep: {sweep_path} not found")
        return
    combined = load_combined_budget_data(dataset)
    report_budget_sweep_table(dataset, combined)

    # Primary: does recall_hat actually track true_recall, pooled and (more
    # importantly) restricted to the ambiguous band where it matters.
    plot_pooled_budget_scatter(dataset, combined, out_dir)
    plot_ambiguous_band_scatter(dataset, combined, out_dir)
    plot_pooled_binned(dataset, combined, out_dir)
    plot_regret_by_budget(dataset, combined, out_dir)

    # Secondary/context only - off by default, since it was crowding out the
    # plots above without being the actual question under test.
    if include_candidate_stats:
        candidate_stats = compute_candidate_stats_by_budget(dataset, sorted(combined.keys()))
        plot_budget_relationship(dataset, combined, candidate_stats, out_dir)

#-------------------------------------------------------------------------------
# Inverse-coverage vs vanilla intersection-fraction comparison
#-------------------------------------------------------------------------------


def report_estimator_comparison(dataset: str, combined_ic: dict, combined_if: dict) -> None:
    """Compares the two estimators as independent hit policies: each picks
    its own argmax candidate, so this is a fair head-to-head on actual
    serving quality, not just on estimate accuracy in isolation."""
    print(f"\n=== {dataset}: inverse-coverage (IC) vs vanilla intersection-fraction (IF) ===")
    pooled_ic, pooled_if = pooled_arrays(combined_ic), pooled_arrays(combined_if)
    lo, hi = AMBIGUOUS_BAND

    for name, pooled in (('IC', pooled_ic), ('IF', pooled_if)):
        tr, oracle = pooled['true_recall'], pooled['oracle_true_recall']
        regret = oracle - tr
        band = (tr >= lo) & (tr < hi)
        corr_all = np.corrcoef(pooled['recall_hat'], tr)[0, 1]
        corr_band = np.corrcoef(pooled['recall_hat'][band], tr[band])[0, 1] if band.sum() >= 2 else float('nan')
        print(f"  {name}: n={len(tr)} selected_true_recall_mean={tr.mean():.4f} "
              f"regret_mean={regret.mean():.4f} regret_p90={np.percentile(regret, 90):.4f} "
              f"corr_pooled={corr_all:+.4f} corr_ambiguous_band(n={band.sum()})={corr_band:+.4f}")


def plot_estimator_comparison_ambiguous_band(dataset: str, combined_ic: dict, combined_if: dict,
                                              out_dir: Path) -> None:
    """The plot that actually settles the comparison: both estimators'
    argmax-selected candidates, restricted to the ambiguous true_recall
    band, pooled across every budget level, on the same axes."""
    pooled_ic, pooled_if = pooled_arrays(combined_ic), pooled_arrays(combined_if)
    lo, hi = AMBIGUOUS_BAND
    band_ic = (pooled_ic['true_recall'] >= lo) & (pooled_ic['true_recall'] < hi)
    band_if = (pooled_if['true_recall'] >= lo) & (pooled_if['true_recall'] < hi)

    fig, ax = plt.subplots(figsize=(7, 6))
    if band_ic.sum() >= 2:
        corr_ic = np.corrcoef(pooled_ic['recall_hat'][band_ic], pooled_ic['true_recall'][band_ic])[0, 1]
        ax.scatter(pooled_ic['recall_hat'][band_ic], pooled_ic['true_recall'][band_ic], s=18, alpha=0.4,
                   color='tab:blue', label=f'inverse-coverage (n={band_ic.sum()}, corr={corr_ic:+.3f})')
    if band_if.sum() >= 2:
        corr_if = np.corrcoef(pooled_if['recall_hat'][band_if], pooled_if['true_recall'][band_if])[0, 1]
        ax.scatter(pooled_if['recall_hat'][band_if], pooled_if['true_recall'][band_if], s=18, alpha=0.4,
                   color='tab:red', marker='^', label=f'vanilla IF (n={band_if.sum()}, corr={corr_if:+.3f})')
    ax.plot([lo, hi], [lo, hi], color='black', linewidth=1, linestyle='--', alpha=0.6)
    ax.set_xlabel('estimated recall')
    ax.set_ylabel('true_recall (actual)')
    ax.set_title(f'{dataset}: ambiguous band [{lo},{hi}), pooled across budgets')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / f'estimator_comparison_ambiguous_band_{dataset}.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_estimator_comparison_binned(dataset: str, combined_ic: dict, combined_if: dict,
                                      out_dir: Path, n_bins: int = 10) -> None:
    pooled_ic, pooled_if = pooled_arrays(combined_ic), pooled_arrays(combined_if)

    def bin_stats(rh, tr):
        order = np.argsort(rh)
        bins = np.array_split(order, min(n_bins, len(rh)))
        bx, my, sy = [], [], []
        for idx in bins:
            if len(idx) == 0:
                continue
            bx.append(rh[idx].mean())
            my.append(tr[idx].mean())
            sy.append(tr[idx].std(ddof=0) / np.sqrt(len(idx)))
        return np.array(bx), np.array(my), np.array(sy)

    bx_ic, my_ic, sy_ic = bin_stats(pooled_ic['recall_hat'], pooled_ic['true_recall'])
    bx_if, my_if, sy_if = bin_stats(pooled_if['recall_hat'], pooled_if['true_recall'])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.errorbar(bx_ic, my_ic, yerr=sy_ic, fmt='o-', color='tab:blue', label='inverse-coverage', capsize=3)
    ax.errorbar(bx_if, my_if, yerr=sy_if, fmt='^--', color='tab:red', label='vanilla IF', capsize=3)
    ax.plot([0, 1], [0, 1], color='black', linewidth=1, linestyle=':', alpha=0.6, label='y=x')
    ax.set_xlabel('mean estimated recall in bin')
    ax.set_ylabel('mean true_recall (actual)')
    ax.set_title(f'{dataset}: binned comparison, pooled across budgets')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / f'estimator_comparison_binned_{dataset}.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_path}")


def run_estimator_comparison(dataset: str, out_dir: Path = OUT_DIR) -> None:
    sweep_path = RESULTS_DIR / f'hitpolicy_budget_sweep_{dataset}.npz'
    if not sweep_path.exists():
        print(f"skipping {dataset} estimator comparison: {sweep_path} not found")
        return
    combined_ic = load_combined_budget_data(dataset, estimate_key='recall_hat')
    combined_if = load_combined_budget_data(dataset, estimate_key='recall_hat_if')
    report_estimator_comparison(dataset, combined_ic, combined_if)
    plot_estimator_comparison_ambiguous_band(dataset, combined_ic, combined_if, out_dir)
    plot_estimator_comparison_binned(dataset, combined_ic, combined_if, out_dir)

#-------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', default=['sift', 'esci'])
    parser.add_argument('--skip-rc-size', action='store_true',
                         help='skip regenerating candidates for the rc_size plot (saves time)')
    parser.add_argument('--budget-sweep', action='store_true',
                         help='also report/plot the budget-sweep aggregation')
    parser.add_argument('--compare-if', action='store_true',
                         help='also compare inverse-coverage against vanilla intersection-fraction')
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        path = RESULTS_DIR / f'hitpolicy_{dataset}.npz'
        if not path.exists():
            print(f"skipping {dataset}: {path} not found")
            continue
        d = np.load(path)
        sel = per_query_selected(d)
        report_h2_h2b(dataset, sel)
        report_h3_regret(dataset, sel)
        report_min_coverage_summary(dataset, d)
        plot_recall_hat_vs_true_recall(dataset, sel, OUT_DIR)
        plot_binned(dataset, sel, OUT_DIR)
        plot_regret(dataset, sel, OUT_DIR)
        plot_min_coverage(dataset, d, OUT_DIR)
        if not args.skip_rc_size:
            plot_rc_size(dataset, OUT_DIR)

    if args.budget_sweep:
        for dataset in args.datasets:
            report_and_plot_budget_sweep(dataset, OUT_DIR)

    if args.compare_if:
        for dataset in args.datasets:
            run_estimator_comparison(dataset, OUT_DIR)


if __name__ == '__main__':
    main()
