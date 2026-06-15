#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys
import json
import re
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.dpi':      150,
    'font.size':       13,
    'axes.titlesize':  13,
    'axes.labelsize':  13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
})

#-------------------------------------------------------------------------------

POPULATION_RAW_BASE = Path('simulations/sift/raw/population')
POPULATION_PROCESSED_BASE = Path('simulations/sift/processed/population/main')

ALPHA_VALUES = [1.05, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5]

STRATEGY_ORDER = ['random', 'greedy_naive', 'greedy_cost_static', 'greedy_cost']
STRATEGY_LABELS = {
    'random':              'Random',
    'greedy_naive':        'Greedy Naive',
    'greedy_cost_static':  'Greedy Cost (Static)',
    'greedy_cost':         'Greedy Cost',
}
ALGO_LABELS = {
    'lemma1':   'CIG',
    'lemma2':   'HGG',
    'combined': 'Combined',
}
ALGO_ORDER = ['lemma1', 'lemma2', 'combined']

#-------------------------------------------------------------------------------
# Helpers
#-------------------------------------------------------------------------------

def _default_raw_dir() -> Path:
    base = POPULATION_RAW_BASE
    stamped = sorted(
        d for d in base.iterdir()
        if d.is_dir() and re.fullmatch(r'\d{8}_\d{6}', d.name)
    ) if base.exists() else []
    if stamped:
        chosen = stamped[-1]
        print(f"Auto-selected run: {chosen.name}")
        return chosen
    return base

def _train_size_int(benchmark: str) -> int:
    m = re.search(r'_c(\d+)k', benchmark)
    return int(m.group(1)) if m else 0

def _train_size(benchmark: str) -> str:
    n = _train_size_int(benchmark)
    return f"c{n}" if n else benchmark

def _bench_title(benchmark: str, mode: str, budget: str = None) -> str:
    ts = _train_size_int(benchmark)
    mode_str = 'Oracle' if mode == 'oracle' else 'Non-Oracle'
    if budget is not None:
        return f"{mode_str}, Train Size = {ts}, Budget = {_budget_int(budget)}"
    return f"{mode_str}, Train Size = {ts}"

def _budget_int(b: str) -> int:
    return int(b[1:])

def _budget_label(b: str) -> str:
    return str(_budget_int(b))

def save_fig(fig: plt.Figure, path: Path, name: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / name, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  {name}")

def pct_fmt(ax, axis: str = 'y') -> None:
    fmt = mticker.FuncFormatter(lambda v, _: f'{v:.0%}')
    if axis == 'y':
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)

def _strategy_palette():
    colors = sns.color_palette('tab10', len(STRATEGY_ORDER))
    return {s: c for s, c in zip(STRATEGY_ORDER, colors)}

def _algo_palette():
    colors = sns.color_palette('Set2', len(ALGO_ORDER))
    return {a: c for a, c in zip(ALGO_ORDER, colors)}

def _annotate_last(ax, x_vals, y_vals, label, color, fontsize=18, stack_idx=0):
    """Annotate the last point of a line. stack_idx staggers overlapping labels."""
    if not x_vals:
        return
    near_top = y_vals[-1] > 0.85
    base = -18 if near_top else 6
    direction = -1 if near_top else 1
    y_offset = base + stack_idx * 22 * direction
    ax.annotate(label,
                xy=(x_vals[-1], y_vals[-1]),
                xytext=(4, y_offset), textcoords='offset points',
                fontsize=fontsize, color=color, clip_on=False)

def _annotate_each(ax, b_vals, n_entries, y_vals, color, stack_idx=0, fontsize=9):
    """Annotate every budget point with its cache-entry count, staggered by stack_idx."""
    for x, y, n in zip(b_vals, y_vals, n_entries):
        near_top = y > 0.85
        base = -14 if near_top else 5
        direction = -1 if near_top else 1
        y_off = base + stack_idx * 18 * direction
        ax.annotate(f"n={n}", xy=(x, y), xytext=(3, y_off),
                    textcoords='offset points', fontsize=fontsize,
                    color=color, clip_on=False)

#-------------------------------------------------------------------------------
# Loaders
#-------------------------------------------------------------------------------

def load_trajectories(results_dir: Path) -> dict:
    """Returns {benchmark: {budget: {strategy: [entry, ...]}}}."""
    trajectories: dict = {}
    for traj_file in sorted(results_dir.glob('*/B*/*_trajectory.json')):
        benchmark = traj_file.parent.parent.name
        budget = traj_file.parent.name
        strategy = traj_file.stem.replace('_trajectory', '')
        with open(traj_file) as f:
            data = json.load(f)
        trajectories.setdefault(benchmark, {}).setdefault(budget, {})[strategy] = data
    return trajectories

def load_population_summaries(results_dir: Path, mode: str) -> list:
    """Load summary.json for the given eval mode. Path: {benchmark}/B{val}/{mode}/{strategy}/summary.json."""
    summaries = []
    for summary_file in sorted(results_dir.glob(f'*/B*/{mode}/*/summary.json')):
        strategy = summary_file.parent.name
        budget = summary_file.parent.parent.parent.name
        benchmark = summary_file.parent.parent.parent.parent.name
        with open(summary_file) as f:
            data = json.load(f)
        data['_strategy'] = strategy
        data['_budget'] = budget
        data['_benchmark'] = benchmark
        data['_eval_mode'] = mode
        summaries.append(data)
    return summaries

def load_selection_logs(results_dir: Path) -> dict:
    """Returns {benchmark: {budget: {strategy: [step_dict, ...]}}}."""
    logs: dict = {}
    for f in sorted(results_dir.glob('*/B*/*_selection.json')):
        benchmark = f.parent.parent.name
        budget = f.parent.name
        strategy = f.stem.replace('_selection', '')
        with open(f) as fh:
            data = json.load(fh)
        logs.setdefault(benchmark, {}).setdefault(budget, {})[strategy] = data
    return logs

#-------------------------------------------------------------------------------
# Per-B trajectory plots
#-------------------------------------------------------------------------------

def plot_cache_entries_vs_cig_coverage(
    traj_by_strategy: dict,
    benchmark: str,
    budget: str,
    outdir: Path,
    mode: str,
) -> None:
    if not traj_by_strategy:
        return
    palette = _strategy_palette()
    fig, ax = plt.subplots(figsize=(8, 5))
    has_data = False
    for strategy in STRATEGY_ORDER:
        traj = traj_by_strategy.get(strategy, [])
        if not traj:
            continue
        n_entries = [t['num_entries'] for t in traj]
        cov_vals = [t['cig_train_coverage'] for t in traj]
        color = palette[strategy]
        ax.plot(n_entries, cov_vals, marker='o', markersize=4,
                label=STRATEGY_LABELS.get(strategy, strategy), color=color)
        _annotate_last(ax, n_entries, cov_vals, f"n={n_entries[-1]}", color)
        has_data = True
    if not has_data:
        plt.close(fig)
        return
    ax.set_xlabel('Cache Entries')
    ax.set_ylabel('CIG Coverage (theoretical)')
    pct_fmt(ax)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title(f'{_bench_title(benchmark, mode, budget)} - Cache Entries vs CIG Coverage')
    save_fig(fig, outdir, 'cache_entries_vs_cig_coverage.png')

#-------------------------------------------------------------------------------

def plot_B_vs_cig_coverage(
    traj_by_strategy: dict,
    benchmark: str,
    budget: str,
    outdir: Path,
    mode: str,
) -> None:
    if not traj_by_strategy:
        return
    palette = _strategy_palette()
    fig, ax = plt.subplots(figsize=(8, 5))
    has_data = False
    for strategy in STRATEGY_ORDER:
        traj = traj_by_strategy.get(strategy, [])
        if not traj:
            continue
        b_vals = [t['vectors_cached'] for t in traj]
        cov_vals = [t['cig_train_coverage'] for t in traj]
        color = palette[strategy]
        ax.plot(b_vals, cov_vals, marker='o', markersize=4,
                label=STRATEGY_LABELS.get(strategy, strategy), color=color)
        _annotate_last(ax, b_vals, cov_vals, f"n={traj[-1]['num_entries']}", color)
        has_data = True
    if not has_data:
        plt.close(fig)
        return
    ax.set_xlabel('Vectors Cached (running total)')
    ax.set_ylabel('CIG Coverage (theoretical)')
    pct_fmt(ax)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title(f'{_bench_title(benchmark, mode, budget)} - Vectors Cached vs CIG Coverage')
    save_fig(fig, outdir, 'vectors_cached_vs_cig_coverage.png')

#-------------------------------------------------------------------------------
# Per-B simulation plot
#-------------------------------------------------------------------------------

def plot_sim_results(
    summaries: list,
    benchmark: str,
    budget: str,
    outdir: Path,
    mode: str,
) -> None:
    bench_summaries = [s for s in summaries
                       if s['_benchmark'] == benchmark
                       and s['_budget'] == budget
                       and s['_eval_mode'] == mode]
    strategies_present = [s for s in STRATEGY_ORDER
                          if any(b['_strategy'] == s for b in bench_summaries)]
    if not strategies_present:
        return

    algo_colors = _algo_palette()
    n = len(strategies_present)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 9), squeeze=False)

    for col, strategy in enumerate(strategies_present):
        s_data = next((s for s in bench_summaries if s['_strategy'] == strategy), None)
        if s_data is None:
            continue
        results_by_algo = {r['algorithm']: r for r in s_data['results']}
        algos = [a for a in ALGO_ORDER if a in results_by_algo]
        labels = [ALGO_LABELS.get(a, a) for a in algos]
        colors = [algo_colors[a] for a in algos]

        ax_hr = axes[0][col]
        hit_rates = [results_by_algo[a].get('hit_rate', 0.0) for a in algos]
        ax_hr.bar(labels, hit_rates, color=colors)
        pct_fmt(ax_hr)
        ax_hr.set_ylim(0, 1)
        ax_hr.set_title(STRATEGY_LABELS.get(strategy, strategy))
        ax_hr.set_ylabel('Simulation Hit Rate' if col == 0 else '')

        ax_dc = axes[1][col]
        avg_dc = [results_by_algo[a].get('avg_distance_calcs') or 0 for a in algos]
        ax_dc.bar(labels, avg_dc, color=colors)
        ax_dc.set_ylim(bottom=0)
        ax_dc.set_ylabel('Avg Distance Calcs' if col == 0 else '')
        ax_dc.set_xlabel('Algorithm')

    plt.tight_layout(pad=2.5, h_pad=3.5, w_pad=2.5)
    fig.suptitle(f'{_bench_title(benchmark, mode, budget)} - Simulation Results', y=1.02)
    save_fig(fig, outdir, 'sim_results.png')

#-------------------------------------------------------------------------------
# Per-benchmark aggregate plots
#-------------------------------------------------------------------------------

def plot_cig_coverage_vs_budget(
    trajectories: dict,
    benchmark: str,
    outdir: Path,
    mode: str,
) -> None:
    bdata = trajectories.get(benchmark, {})
    if not bdata:
        return
    budgets = sorted(bdata.keys(), key=_budget_int)
    palette = _strategy_palette()
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, strategy in enumerate(STRATEGY_ORDER):
        b_vals, cov_vals, n_entries = [], [], []
        for budget in budgets:
            traj = bdata[budget].get(strategy, [])
            b_vals.append(_budget_int(budget))
            cov_vals.append(traj[-1]['cig_train_coverage'] if traj else 0.0)
            n_entries.append(traj[-1]['num_entries'] if traj else 0)
        color = palette[strategy]
        ax.plot(b_vals, cov_vals, marker='o', markersize=5,
                label=STRATEGY_LABELS.get(strategy, strategy), color=color)
        _annotate_each(ax, b_vals, n_entries, cov_vals, color, stack_idx=i)
    ax.set_xlabel('Budget B (vectors)')
    ax.set_ylabel('CIG Coverage (theoretical)')
    pct_fmt(ax)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title(f'{_bench_title(benchmark, mode)} - Budget vs CIG Coverage')
    save_fig(fig, outdir, 'cig_coverage_vs_budget.png')

#-------------------------------------------------------------------------------

def plot_cig_test_coverage_vs_budget(
    trajectories: dict,
    benchmark: str,
    outdir: Path,
    mode: str,
) -> None:
    """Non-oracle CIG coverage (cig_test_coverage) at final budget step, by strategy."""
    bdata = trajectories.get(benchmark, {})
    if not bdata:
        return
    has_test_cov = any(
        'cig_test_coverage' in t
        for bgt_data in bdata.values()
        for traj in bgt_data.values()
        for t in traj
    )
    if not has_test_cov:
        return
    budgets = sorted(bdata.keys(), key=_budget_int)
    palette = _strategy_palette()
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, strategy in enumerate(STRATEGY_ORDER):
        b_vals, cov_vals, n_entries = [], [], []
        for budget in budgets:
            traj = bdata[budget].get(strategy, [])
            b_vals.append(_budget_int(budget))
            cov_vals.append(traj[-1]['cig_test_coverage'] if traj else 0.0)
            n_entries.append(traj[-1]['num_entries'] if traj else 0)
        color = palette[strategy]
        ax.plot(b_vals, cov_vals, marker='o', markersize=5,
                label=STRATEGY_LABELS.get(strategy, strategy), color=color)
        _annotate_each(ax, b_vals, n_entries, cov_vals, color, stack_idx=i)
    ax.set_xlabel('Budget B (vectors)')
    ax.set_ylabel('CIG Test Coverage (non-oracle)')
    pct_fmt(ax)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title(f'{_bench_title(benchmark, mode)} - Budget vs CIG Test Coverage')
    save_fig(fig, outdir, 'cig_test_coverage_vs_budget.png')

#-------------------------------------------------------------------------------

def plot_sim_hit_rate_vs_budget(
    summaries: list,
    benchmark: str,
    outdir: Path,
    mode: str,
    trajectories: dict = None,
) -> None:
    bench_summaries = [s for s in summaries
                       if s['_benchmark'] == benchmark and s['_eval_mode'] == mode]
    if not bench_summaries:
        return

    budgets = sorted({s['_budget'] for s in bench_summaries}, key=_budget_int)
    algo_colors = _algo_palette()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), squeeze=False)
    axes_flat = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]

    bdata = (trajectories or {}).get(benchmark, {})

    for i, (ax, strategy) in enumerate(zip(axes_flat, STRATEGY_ORDER)):
        n_entries = [bdata.get(bgt, {}).get(strategy, []) for bgt in budgets]
        n_entries = [t[-1]['num_entries'] if t else 0 for t in n_entries]
        cig_b_vals, cig_hr_vals = [], []
        for j, algo in enumerate(ALGO_ORDER):
            b_vals, hr_vals = [], []
            for budget in budgets:
                s_data = next((s for s in bench_summaries
                               if s['_strategy'] == strategy
                               and s['_budget'] == budget), None)
                r = next((r for r in s_data['results'] if r['algorithm'] == algo),
                         None) if s_data else None
                b_vals.append(_budget_int(budget))
                hr_vals.append(r.get('hit_rate', 0.0) if r else 0.0)
            color = algo_colors[algo]
            ax.plot(b_vals, hr_vals, marker='o', markersize=4,
                    label=ALGO_LABELS.get(algo, algo), color=color)
            if algo == ALGO_ORDER[0]:
                cig_b_vals, cig_hr_vals = b_vals, hr_vals
        _annotate_each(ax, cig_b_vals, n_entries, cig_hr_vals,
                       algo_colors[ALGO_ORDER[0]], stack_idx=0)
        ax.set_xlabel('Budget B (vectors)')
        ax.set_ylabel('Simulation Hit Rate')
        pct_fmt(ax)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1)
        ax.set_title(STRATEGY_LABELS.get(strategy, strategy))
        ax.legend()

    plt.tight_layout(pad=2.5, h_pad=3.5, w_pad=2.5)
    fig.subplots_adjust(top=0.90)
    fig.suptitle(f'{_bench_title(benchmark, mode)} - Budget vs Sim Hit Rate')
    save_fig(fig, outdir, 'sim_hit_rate_vs_budget.png')

#-------------------------------------------------------------------------------

def plot_selection_trajectory(
    trajectories: dict,
    benchmark: str,
    outdir: Path,
    mode: str,
) -> None:
    bdata = trajectories.get(benchmark, {})
    if not bdata:
        return
    budgets = sorted(bdata.keys(), key=_budget_int)
    tab_colors = sns.color_palette('tab10', len(budgets))
    budget_colors = {b: tab_colors[i] for i, b in enumerate(budgets)}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), squeeze=False)
    axes_flat = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]

    for ax_idx, (ax, strategy) in enumerate(zip(axes_flat, STRATEGY_ORDER)):
        for i, budget in enumerate(budgets):
            traj = bdata[budget].get(strategy, [])
            color = budget_colors[budget]
            if traj:
                b_vals = [t['vectors_cached'] for t in traj]
                cov_vals = [t['cig_train_coverage'] for t in traj]
                ax.plot(b_vals, cov_vals, marker='o', markersize=3,
                        color=color, label=_budget_label(budget))
                _annotate_last(ax, b_vals, cov_vals,
                               f"n={traj[-1]['num_entries']}", color, stack_idx=i)
            else:
                ax.plot([0], [0.0], marker='o', markersize=3, color=color,
                        label=_budget_label(budget))
                y_off = 6 + i * 22
                ax.annotate("n=0", xy=(0, 0.0), xytext=(4, y_off),
                            textcoords='offset points', fontsize=18,
                            color=color, clip_on=False)
        ax.set_xlabel('Vectors Cached (running total)')
        ax.set_ylabel('CIG Coverage (theoretical)')
        pct_fmt(ax)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1)
        ax.set_title(STRATEGY_LABELS.get(strategy, strategy))
        if ax_idx == 0:
            ax.legend(title='Budget B', fontsize=8, title_fontsize=9,
                      loc='upper left', ncol=2)

    plt.tight_layout(pad=2.5, h_pad=3.5, w_pad=2.5)
    fig.subplots_adjust(top=0.90)
    fig.suptitle(f'{_bench_title(benchmark, mode)} - Selection Trajectories')
    save_fig(fig, outdir, 'selection_trajectory.png')

#-------------------------------------------------------------------------------
# Cross-benchmark aggregate plots
#-------------------------------------------------------------------------------

def _cross_cig_all_benchmarks(
    trajectories: dict,
    benchmarks: list,
    outdir: Path,
    mode: str,
) -> None:
    """2×2 by strategy; each panel shows budget vs CIG coverage for all train sizes."""
    bm_colors = sns.color_palette('tab10', len(benchmarks))
    bm_palette = {b: c for b, c in zip(benchmarks, bm_colors)}
    mode_str = 'Oracle' if mode == 'oracle' else 'Non-Oracle'
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), squeeze=False)
    axes_flat = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]
    for ax, strategy in zip(axes_flat, STRATEGY_ORDER):
        for i, benchmark in enumerate(benchmarks):
            bdata = trajectories.get(benchmark, {})
            budgets = sorted(bdata.keys(), key=_budget_int)
            b_vals, cov_vals, n_entries = [], [], []
            for bgt in budgets:
                traj = bdata[bgt].get(strategy, [])
                b_vals.append(_budget_int(bgt))
                cov_vals.append(traj[-1]['cig_train_coverage'] if traj else 0.0)
                n_entries.append(traj[-1]['num_entries'] if traj else 0)
            ts = _train_size_int(benchmark)
            color = bm_palette[benchmark]
            ax.plot(b_vals, cov_vals, marker='o', markersize=5,
                    label=f'Train={ts}', color=color)
            _annotate_each(ax, b_vals, n_entries, cov_vals, color, stack_idx=i)
        ax.set_xlabel('Budget B (vectors)')
        ax.set_ylabel('CIG Coverage (theoretical)')
        pct_fmt(ax)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)
        ax.set_title(STRATEGY_LABELS[strategy])
    plt.tight_layout(pad=2.5, h_pad=3.5, w_pad=2.5)
    fig.subplots_adjust(top=0.90)
    fig.suptitle(f'{mode_str} - CIG Coverage vs Budget (All Train Sizes)')
    save_fig(fig, outdir, 'cross.png')

def _cross_cig_by_strategy(
    trajectories: dict,
    benchmarks: list,
    outdir: Path,
    mode: str,
) -> None:
    bm_colors = sns.color_palette('tab10', len(benchmarks))
    bm_palette = {b: c for b, c in zip(benchmarks, bm_colors)}
    mode_str = 'Oracle' if mode == 'oracle' else 'Non-Oracle'
    for strategy in STRATEGY_ORDER:
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, benchmark in enumerate(benchmarks):
            bdata = trajectories.get(benchmark, {})
            budgets = sorted(bdata.keys(), key=_budget_int)
            b_vals, cov_vals, n_entries = [], [], []
            for bgt in budgets:
                traj = bdata[bgt].get(strategy, [])
                b_vals.append(_budget_int(bgt))
                cov_vals.append(traj[-1]['cig_train_coverage'] if traj else 0.0)
                n_entries.append(traj[-1]['num_entries'] if traj else 0)
            ts = _train_size_int(benchmark)
            color = bm_palette[benchmark]
            ax.plot(b_vals, cov_vals, marker='o', markersize=5,
                    label=f'Train Size = {ts}', color=color)
            _annotate_each(ax, b_vals, n_entries, cov_vals, color, stack_idx=i)
        ax.set_xlabel('Budget B (vectors)')
        ax.set_ylabel('CIG Coverage (theoretical)')
        pct_fmt(ax)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.set_title(f'{mode_str}, {STRATEGY_LABELS[strategy]} - Budget vs CIG Coverage')
        save_fig(fig, outdir, f'{strategy.replace("_", "-")}.png')

def _cross_sim_by_train_size(
    summaries: list,
    benchmarks: list,
    outdir: Path,
    mode: str,
    trajectories: dict = None,
) -> None:
    palette = _strategy_palette()
    mode_str = 'Oracle' if mode == 'oracle' else 'Non-Oracle'
    for benchmark in benchmarks:
        bench_sums = [s for s in summaries if s['_benchmark'] == benchmark]
        budgets = sorted({s['_budget'] for s in bench_sums}, key=_budget_int)
        bdata = (trajectories or {}).get(benchmark, {})
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, algo in zip(axes, ALGO_ORDER):
            for i, strategy in enumerate(STRATEGY_ORDER):
                b_vals, hr_vals, n_entries = [], [], []
                for bgt in budgets:
                    s_data = next((s for s in bench_sums
                                   if s['_strategy'] == strategy
                                   and s['_budget'] == bgt), None)
                    r = next((r for r in s_data['results'] if r['algorithm'] == algo),
                             None) if s_data else None
                    b_vals.append(_budget_int(bgt))
                    hr_vals.append(r.get('hit_rate', 0.0) if r else 0.0)
                    traj = bdata.get(bgt, {}).get(strategy, [])
                    n_entries.append(traj[-1]['num_entries'] if traj else 0)
                color = palette[strategy]
                ax.plot(b_vals, hr_vals, marker='o', markersize=4,
                        label=STRATEGY_LABELS[strategy], color=color)
                _annotate_each(ax, b_vals, n_entries, hr_vals, color, stack_idx=i)
            ax.set_title(ALGO_LABELS[algo])
            ax.set_xlabel('Budget B (vectors)')
            ax.set_ylabel('Simulation Hit Rate')
            pct_fmt(ax)
            ax.set_xlim(left=0)
            ax.set_ylim(0, 1)
            ax.legend(fontsize=9)
        ts = _train_size_int(benchmark)
        plt.tight_layout(pad=2.5, w_pad=3.0)
        fig.subplots_adjust(top=0.88)
        fig.suptitle(f'{mode_str}, Train Size = {ts} - Budget vs Sim Hit Rate')
        save_fig(fig, outdir, f'{_train_size(benchmark)}.png')

def _cross_sim_by_strategy(
    summaries: list,
    benchmarks: list,
    outdir: Path,
    mode: str,
    trajectories: dict = None,
) -> None:
    bm_colors = sns.color_palette('tab10', len(benchmarks))
    bm_palette = {b: c for b, c in zip(benchmarks, bm_colors)}
    mode_str = 'Oracle' if mode == 'oracle' else 'Non-Oracle'
    for strategy in STRATEGY_ORDER:
        strat_sums = [s for s in summaries if s['_strategy'] == strategy]
        budgets = sorted({s['_budget'] for s in strat_sums}, key=_budget_int)
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, algo in zip(axes, ALGO_ORDER):
            for i, benchmark in enumerate(benchmarks):
                bench_sums = [s for s in strat_sums if s['_benchmark'] == benchmark]
                bdata = (trajectories or {}).get(benchmark, {})
                b_vals, hr_vals, n_entries = [], [], []
                for bgt in budgets:
                    s_data = next((s for s in bench_sums if s['_budget'] == bgt), None)
                    r = next((r for r in s_data['results'] if r['algorithm'] == algo),
                             None) if s_data else None
                    b_vals.append(_budget_int(bgt))
                    hr_vals.append(r.get('hit_rate', 0.0) if r else 0.0)
                    traj = bdata.get(bgt, {}).get(strategy, [])
                    n_entries.append(traj[-1]['num_entries'] if traj else 0)
                ts = _train_size_int(benchmark)
                color = bm_palette[benchmark]
                ax.plot(b_vals, hr_vals, marker='o', markersize=4,
                        label=f'Train Size = {ts}', color=color)
                _annotate_each(ax, b_vals, n_entries, hr_vals, color, stack_idx=i)
            ax.set_title(ALGO_LABELS[algo])
            ax.set_xlabel('Budget B (vectors)')
            ax.set_ylabel('Simulation Hit Rate')
            pct_fmt(ax)
            ax.set_xlim(left=0)
            ax.set_ylim(0, 1)
            ax.legend(fontsize=9)
        plt.tight_layout(pad=2.5, w_pad=3.0)
        fig.subplots_adjust(top=0.88)
        fig.suptitle(f'{mode_str}, {STRATEGY_LABELS[strategy]} - Budget vs Sim Hit Rate')
        save_fig(fig, outdir, f'{strategy.replace("_", "-")}.png')

def plot_cache_entries_vs_budget(
    trajectories: dict,
    benchmark: str,
    outdir: Path,
    mode: str,
) -> None:
    """Budget B vs number of cache entries selected, one line per strategy."""
    bdata = trajectories.get(benchmark, {})
    if not bdata:
        return
    budgets = sorted(bdata.keys(), key=_budget_int)
    palette = _strategy_palette()
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, strategy in enumerate(STRATEGY_ORDER):
        b_vals, n_entries = [], []
        for budget in budgets:
            traj = bdata[budget].get(strategy, [])
            b_vals.append(_budget_int(budget))
            n_entries.append(traj[-1]['num_entries'] if traj else 0)
        color = palette[strategy]
        ax.plot(b_vals, n_entries, marker='o', markersize=5,
                label=STRATEGY_LABELS[strategy], color=color)
        _annotate_last(ax, b_vals, n_entries, f"n={n_entries[-1]}", color, stack_idx=i)
    ax.set_xlabel('Budget B (vectors)')
    ax.set_ylabel('Cache Entries Selected')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.set_title(f'{_bench_title(benchmark, mode)} - Budget vs Cache Entries')
    save_fig(fig, outdir, 'cache_entries_vs_budget.png')

def _cross_cache_entries_vs_budget(
    trajectories: dict,
    benchmarks: list,
    outdir: Path,
    mode: str,
) -> None:
    """2×2 by strategy; each panel shows budget vs cache entries for all train sizes."""
    bm_colors = sns.color_palette('tab10', len(benchmarks))
    bm_palette = {b: c for b, c in zip(benchmarks, bm_colors)}
    mode_str = 'Oracle' if mode == 'oracle' else 'Non-Oracle'
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), squeeze=False)
    axes_flat = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]
    for ax, strategy in zip(axes_flat, STRATEGY_ORDER):
        for i, benchmark in enumerate(benchmarks):
            bdata = trajectories.get(benchmark, {})
            budgets = sorted(bdata.keys(), key=_budget_int)
            b_vals, n_entries = [], []
            for bgt in budgets:
                traj = bdata[bgt].get(strategy, [])
                b_vals.append(_budget_int(bgt))
                n_entries.append(traj[-1]['num_entries'] if traj else 0)
            ts = _train_size_int(benchmark)
            color = bm_palette[benchmark]
            ax.plot(b_vals, n_entries, marker='o', markersize=5,
                    label=f'Train={ts}', color=color)
            _annotate_last(ax, b_vals, n_entries, f'n={n_entries[-1]}', color, stack_idx=i)
        ax.set_xlabel('Budget B (vectors)')
        ax.set_ylabel('Cache Entries Selected')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9)
        ax.set_title(STRATEGY_LABELS[strategy])
    plt.tight_layout(pad=2.5, h_pad=3.5, w_pad=2.5)
    fig.subplots_adjust(top=0.90)
    fig.suptitle(f'{mode_str} - Cache Entries vs Budget (All Train Sizes)')
    save_fig(fig, outdir, 'cross.png')

#-------------------------------------------------------------------------------
# Selection stats plots
#-------------------------------------------------------------------------------

def plot_selection_stats(
    sel_logs: dict,
    benchmark: str,
    outdir: Path,
) -> None:
    """Write selection stat figures for a single benchmark."""
    outdir.mkdir(parents=True, exist_ok=True)
    strategies = STRATEGY_ORDER
    budgets = sorted(sel_logs.keys(), key=_budget_int)
    ts_label = _train_size_int(benchmark)

    # alpha distribution
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Train={ts_label} - Alpha Distribution')
    for ax, strat in zip(axes.flat, strategies):
        alpha_counts = {a: 0 for a in ALPHA_VALUES}
        total = 0
        for budget in budgets:
            for step in sel_logs.get(budget, {}).get(strat, []):
                a = step['alpha']
                alpha_counts[a] = alpha_counts.get(a, 0) + 1
                total += 1
        fracs = [alpha_counts.get(a, 0) / max(total, 1) for a in ALPHA_VALUES]
        ax.bar([str(a) for a in ALPHA_VALUES], fracs)
        pct_fmt(ax)
        ax.set_title(STRATEGY_LABELS.get(strat, strat))
        ax.set_xlabel('alpha')
        ax.set_ylabel('fraction of selections')
    plt.tight_layout()
    fig.savefig(outdir / 'sel_alpha_distribution.png', dpi=120)
    plt.close(fig)

    # R_c size box plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Train={ts_label} - R_c Size by Budget')
    for ax, strat in zip(axes.flat, strategies):
        pairs = [
            (b, [s['vectors_in_R_c'] for s in sel_logs.get(b, {}).get(strat, [])])
            for b in budgets
        ]
        pairs = [(b, d) for b, d in pairs if d]
        if pairs:
            lbls, data = zip(*pairs)
            ax.boxplot(list(data), tick_labels=list(lbls))
            ax.tick_params(axis='x', rotation=45)
        ax.set_title(STRATEGY_LABELS.get(strat, strat))
        ax.set_xlabel('budget')
        ax.set_ylabel('|R_c|')
    plt.tight_layout()
    fig.savefig(outdir / 'sel_rc_size.png', dpi=120)
    plt.close(fig)

#-------------------------------------------------------------------------------
# PDF builder
#-------------------------------------------------------------------------------

def _divider_fig(title: str, subtitle: str = '') -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_facecolor('#f0f0f0')
    fig.patch.set_facecolor('#f0f0f0')
    ax.axis('off')
    y_title = 0.55 if subtitle else 0.50
    ax.text(0.5, y_title, title, ha='center', va='center',
            fontsize=28, fontweight='bold', transform=ax.transAxes)
    if subtitle:
        ax.text(0.5, 0.40, subtitle, ha='center', va='center',
                fontsize=16, color='#555555', transform=ax.transAxes)
    return fig

def _numeric_key(p: Path) -> int:
    m = re.search(r'\d+', p.stem)
    return int(m.group()) if m else 0

def _strategy_sort_key(p: Path) -> int:
    stem = p.stem.replace('-', '_')
    return STRATEGY_ORDER.index(stem) if stem in STRATEGY_ORDER else len(STRATEGY_ORDER)

def _save_pdf(outdir_mode: Path, benchmarks: list, mode: str, pdf_path: Path) -> None:
    """Build a curated, structured PDF with section dividers (aggregate plots only)."""
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.image as mpimg

    mode_str = 'Oracle' if mode == 'oracle' else 'Non-Oracle'
    cross = outdir_mode / 'cross'
    PAGE_SIZE = (14, 9)

    def _add_png(pdf, path):
        if not path.exists():
            return
        img = mpimg.imread(str(path))
        fig = plt.figure(figsize=PAGE_SIZE)
        ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
        ax.imshow(img)
        ax.axis('off')
        pdf.savefig(fig, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

    with PdfPages(pdf_path) as pdf:
        # title
        fig = _divider_fig(f'{mode_str} Analysis', 'Population Cache Study')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # cross-benchmark CIG coverage (single multi-benchmark figure)
        cig_train_size_png = cross / 'cig_by_train_size' / 'cross.png'
        if cig_train_size_png.exists():
            fig = _divider_fig('Cross-Benchmark: CIG Coverage by Train Size')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            _add_png(pdf, cig_train_size_png)

        # cross-benchmark CIG by strategy (one file per strategy, sorted by STRATEGY_ORDER)
        cig_by_strat_dir = cross / 'cig_by_strategy'
        if cig_by_strat_dir.exists():
            fig = _divider_fig('Cross-Benchmark: CIG Coverage by Strategy')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            for png in sorted(cig_by_strat_dir.glob('*.png'), key=_strategy_sort_key):
                _add_png(pdf, png)

        # cross-benchmark sim hit rate by train size (sorted numerically)
        sim_train_size_dir = cross / 'sim_by_train_size'
        if sim_train_size_dir.exists():
            fig = _divider_fig('Cross-Benchmark: Sim Hit Rate by Train Size')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            for png in sorted(sim_train_size_dir.glob('*.png'), key=_numeric_key):
                _add_png(pdf, png)

        # cross-benchmark sim hit rate by strategy (sorted by STRATEGY_ORDER)
        sim_by_strat_dir = cross / 'sim_by_strategy'
        if sim_by_strat_dir.exists():
            fig = _divider_fig('Cross-Benchmark: Sim Hit Rate by Strategy')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            for png in sorted(sim_by_strat_dir.glob('*.png'), key=_strategy_sort_key):
                _add_png(pdf, png)

        # cross-benchmark cache entries vs budget (single multi-benchmark figure)
        entries_train_size_png = cross / 'cache_entries_by_train_size' / 'cross.png'
        if entries_train_size_png.exists():
            fig = _divider_fig('Cross-Benchmark: Cache Entries vs Budget')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            _add_png(pdf, entries_train_size_png)

        # per-benchmark sections (aggregate plots only)
        for benchmark in benchmarks:
            bm_dir = outdir_mode / benchmark
            if not bm_dir.exists():
                continue
            ts = _train_size_int(benchmark)
            fig = _divider_fig(f'Train Size = {ts}', 'Per-benchmark aggregate results')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            for fname in ['cig_coverage_vs_budget.png',
                          'cig_test_coverage_vs_budget.png',
                          'cache_entries_vs_budget.png',
                          'selection_trajectory.png',
                          'sim_hit_rate_vs_budget.png']:
                _add_png(pdf, bm_dir / fname)
            # selection stats only in oracle PDF (oracle-agnostic but generated once)
            if mode == 'oracle':
                sel_dir = outdir_mode.parent / benchmark
                for fname in ['sel_alpha_distribution.png', 'sel_rc_size.png']:
                    _add_png(pdf, sel_dir / fname)

    print(f"  {pdf_path.name}")

#-------------------------------------------------------------------------------

def _plot_cross_benchmark(
    trajectories: dict,
    summaries: list,
    benchmarks: list,
    outdir: Path,
    mode: str,
) -> None:
    _cross_cig_all_benchmarks(trajectories, benchmarks, outdir / 'cig_by_train_size', mode)
    _cross_cig_by_strategy(trajectories, benchmarks, outdir / 'cig_by_strategy', mode)
    _cross_sim_by_train_size(summaries, benchmarks, outdir / 'sim_by_train_size', mode,
                              trajectories)
    _cross_sim_by_strategy(summaries, benchmarks, outdir / 'sim_by_strategy', mode,
                            trajectories)
    _cross_cache_entries_vs_budget(trajectories, benchmarks,
                                   outdir / 'cache_entries_by_train_size', mode)

#-------------------------------------------------------------------------------

def analyze(results_dir: str, output_dir: str) -> None:
    results_path = Path(results_dir)
    output_path = Path(output_dir)

    print(f"Loading trajectory data from {results_path}...")
    trajectories = load_trajectories(results_path)

    print("Loading simulation summaries...")
    oracle_summaries = load_population_summaries(results_path, 'oracle')
    non_oracle_summaries = load_population_summaries(results_path, 'non_oracle')

    print("Loading selection logs...")
    selection_logs = load_selection_logs(results_path)

    benchmarks = sorted(set(list(trajectories.keys()) +
                            [s['_benchmark'] for s in oracle_summaries] +
                            [s['_benchmark'] for s in non_oracle_summaries]),
                        key=_train_size_int)

    if not benchmarks:
        print("No data found.")
        return

    print(f"Found {len(benchmarks)} benchmarks, "
          f"{len(oracle_summaries)} oracle summaries, "
          f"{len(non_oracle_summaries)} non-oracle summaries.")

    for mode, sums in [('oracle', oracle_summaries), ('non_oracle', non_oracle_summaries)]:
        outdir_mode = output_path / mode
        for benchmark in benchmarks:
            print(f"\n[{mode}] {benchmark}")
            bdata = trajectories.get(benchmark, {})
            budgets_sorted = sorted(bdata.keys(), key=_budget_int)
            outdir_bm = outdir_mode / benchmark

            for budget in budgets_sorted:
                outdir_b = outdir_bm / budget
                plot_cache_entries_vs_cig_coverage(bdata[budget], benchmark, budget,
                                                   outdir_b, mode)
                plot_B_vs_cig_coverage(bdata[budget], benchmark, budget, outdir_b, mode)
                plot_sim_results(sums, benchmark, budget, outdir_b, mode)

            plot_cig_coverage_vs_budget(trajectories, benchmark, outdir_bm, mode)
            plot_cig_test_coverage_vs_budget(trajectories, benchmark, outdir_bm, mode)
            plot_cache_entries_vs_budget(trajectories, benchmark, outdir_bm, mode)
            plot_selection_trajectory(trajectories, benchmark, outdir_bm, mode)
            plot_sim_hit_rate_vs_budget(sums, benchmark, outdir_bm, mode, trajectories)

            if mode == 'oracle':
                plot_selection_stats(
                    selection_logs.get(benchmark, {}),
                    benchmark,
                    output_path / benchmark,
                )

        print(f"\n[{mode}] Generating cross-benchmark plots...")
        _plot_cross_benchmark(trajectories, sums, benchmarks,
                              outdir_mode / 'cross', mode)

        print(f"\n[{mode}] Building PDF...")
        _save_pdf(outdir_mode, benchmarks, mode, outdir_mode / f'{mode}.pdf')

    print(f"\nAll plots written to {output_path}")

#-------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze SIFT population pipeline results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--results-dir', type=str, default=None,
        help='Timestamped raw run dir (default: latest under raw/population/)',
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output dir for plots (default: processed/population/main/<timestamp>)',
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else _default_raw_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else POPULATION_PROCESSED_BASE / ts
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing processed output to: {output_dir}")

    analyze(str(results_dir), str(output_dir))

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
