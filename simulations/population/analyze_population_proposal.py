#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Analyze proposal-based population pipeline results.

Reads raw outputs from run_population.py and produces, per dataset:

  ball/
    memory_vs_hitrate.png               (sigma x K_test grid)
    memory_vs_candidates_cached.png     (sigma x K_test grid)
    recall_vs_sigma_k{kt}.png           (one per K_test, one subplot per budget)

  debug/ (if debug_dir provided)
    {dataset}_ball_debug.json

point-like proposal was removed (Phase 3 Round 12) -- it was exactly ball
with K=0, see utils/base/population.py and docs/POPULATION_DESIGN.md --
run_population.py only ever produces `ball/` output, and K_cache always
equals K_test (no separate K_cache dimension).

n_max is fixed at 20 in every real run (never actually swept -- --n-max
takes a single value per CLI invocation), so it is not a grid dimension
here; it's noted in each plot's title instead. Phase 3 Round 12 replaced the
previous n x K_test grid with sigma x K_test (rows=sigma, columns=K_test),
and replaced the old per-sigma-directory separate plot files with one
combined file per metric per dataset -- no more sigma_*/ output nesting.
"""

import re
import sys
import json
import argparse
import statistics
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 11,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
})

#-------------------------------------------------------------------------------
# Constants
#-------------------------------------------------------------------------------

STRATEGY = 'greedy_cost'
ALGORITHM = 'lemma1_no_union'

# One distinct color per cache size (sorted ascending: 256, 512, 1024, 2048).
CACHE_SIZE_COLORS = ['#e63946', '#f4a261', '#2a9d8f', '#4361ee']

#-------------------------------------------------------------------------------
# Helpers
#-------------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, path: Path, name: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  {path / name}")


def _pct_fmt(ax, axis: str = 'y') -> None:
    fmt = mticker.FuncFormatter(lambda v, _: f'{v:.0%}')
    if axis == 'y':
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)


def _pct_fmt_fine(ax, axis: str = 'y') -> None:
    fmt = mticker.FuncFormatter(lambda v, _: f'{v:.2%}')
    if axis == 'y':
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)


def _cache_size_from_name(name: str) -> int:
    m = re.search(r'_c(\d+)k', name)
    return int(m.group(1)) if m else 0


def _budget_int(b: str) -> int:
    return int(b[1:])


def _dedup_legend(ax, **kwargs) -> None:
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), **kwargs)


def _shared_legend(fig, axes_2d, n_rows, n_cols, **kwargs) -> None:
    for r in range(n_rows):
        for c in range(n_cols):
            handles, labels = axes_2d[r][c].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            if by_label:
                fig.legend(
                    by_label.values(), by_label.keys(),
                    loc='lower center', ncol=len(by_label),
                    fontsize=8, bbox_to_anchor=(0.5, -0.02),
                    **kwargs,
                )
                return


def _shared_legend_train_test(fig, axes_2d, n_rows, n_cols, cache_sizes: list) -> None:
    """Two-column legend for train/test comparison plots: column 1 is every
    cache size's train entry, column 2 is every cache size's test entry,
    one row per cache size (256, 512, 1024, 2048 top to bottom). matplotlib's
    ncol=2 legend fills column-major (top-to-bottom down column 1 first,
    then column 2), so the handle/label list must list every train label
    (in cache-size order) before any test label to land train in column 1
    and test in column 2 -- passing ncol=2 alone does not group by column
    on its own, and interleaving train/test per cache size (the reverse
    grouping) mixes both into each column instead."""
    for r in range(n_rows):
        for c in range(n_cols):
            handles, labels = axes_2d[r][c].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            if not by_label:
                continue
            ordered_handles, ordered_labels = [], []
            for suffix in ('train', 'test'):
                for cs in cache_sizes:
                    label = f'Cache {cs} - {suffix}'
                    if label in by_label:
                        ordered_labels.append(label)
                        ordered_handles.append(by_label[label])
            if ordered_handles:
                fig.legend(
                    ordered_handles, ordered_labels,
                    loc='lower center', ncol=2,
                    fontsize=8, bbox_to_anchor=(0.5, -0.02),
                )
                return


def _make_subplots(n_rows: int, n_cols: int, row_h: float = 4.0, col_w: float = 4.5):
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(col_w * n_cols, row_h * n_rows),
        squeeze=False,
    )
    return fig, axes


def _max_budget_entry(bm_data: dict) -> tuple:
    if not bm_data:
        return None, {}
    max_b = max(bm_data.keys(), key=_budget_int)
    return max_b, bm_data[max_b]


def _size_color_map(all_data: dict) -> tuple[list, dict]:
    """Return (sorted_cache_sizes, size_color) from all_data."""
    cache_sizes = sorted({
        _cache_size_from_name(bm)
        for rd in all_data.values()
        for bm in rd
    })
    size_color = dict(zip(cache_sizes, CACHE_SIZE_COLORS[:len(cache_sizes)]))
    return cache_sizes, size_color


#-------------------------------------------------------------------------------
# Data loading
#-------------------------------------------------------------------------------

def _read_summary_hit_rate(summary_path: Path) -> tuple:
    """Return (hit_rate, mean_recall) for ALGORITHM from a summary.json, or
    (0.0, 0.0) if the file/algorithm entry is missing."""
    if not summary_path.exists():
        return 0.0, 0.0
    with open(summary_path) as f:
        summary = json.load(f)
    for r in summary.get('results', []):
        if r.get('algorithm') == ALGORITHM:
            return float(r.get('hit_rate', 0.0)), float(r.get('mean_recall', 0.0))
    return 0.0, 0.0


def _load_run_data(param_run_dir: Path, sigma: float = None) -> dict:
    """Load greedy_cost results from one param directory.

    sigma: if given, read the sigma-swept summary at B{budget}/sigma_{sigma:.2f}/
        instead of the default (sigma=1.0, unnested) location. The selection
        trajectory is sigma-independent and always read from the unnested path.

    Reads both non_oracle (test) and oracle (train) summaries -- oracle
    output has always been written by population_runner.py but was never
    read here before (Phase 3 Round 14); same schema, same file layout,
    just under oracle/{strategy}/ instead of non_oracle/{strategy}/.

    Returns:
        {benchmark: {budget_str: {
            'test_hit_rate', 'train_hit_rate', 'mean_recall', 'vectors_cached',
            'num_entries', 'train_coverage', 'test_coverage', 'trajectory'
        }}}
    """
    data: dict = {}
    for bm_dir in sorted(param_run_dir.iterdir()):
        if not bm_dir.is_dir():
            continue
        bm = bm_dir.name
        data[bm] = {}
        for b_dir in sorted(bm_dir.glob('B*')):
            budget_str = b_dir.name
            sim_dir = b_dir / f'sigma_{sigma:.2f}' if sigma is not None else b_dir
            non_oracle_path = sim_dir / 'non_oracle' / STRATEGY / 'summary.json'
            oracle_path = sim_dir / 'oracle' / STRATEGY / 'summary.json'
            if not non_oracle_path.exists():
                continue
            test_hit_rate, mean_recall = _read_summary_hit_rate(non_oracle_path)
            train_hit_rate, _ = _read_summary_hit_rate(oracle_path)

            traj_path = b_dir / f'{STRATEGY}_trajectory.json'
            traj = []
            if traj_path.exists():
                with open(traj_path) as f:
                    traj = json.load(f)

            if traj:
                last = traj[-1]
                vectors_cached = int(last['vectors_cached'])
                num_entries = int(last.get('num_entries', last.get('step', 0)))
                train_coverage = float(last['cig_train_coverage'])
                test_coverage = float(last['cig_test_coverage'])
            else:
                vectors_cached = num_entries = 0
                train_coverage = test_coverage = 0.0

            data[bm][budget_str] = {
                'test_hit_rate': test_hit_rate,
                'train_hit_rate': train_hit_rate,
                'mean_recall': mean_recall,
                'vectors_cached': vectors_cached,
                'num_entries': num_entries,
                'train_coverage': train_coverage,
                'test_coverage': test_coverage,
                'trajectory': traj,
            }
    return data


def _base_size(param_run_dir: Path) -> dict:
    """Return {cache_size_int: n_base} where n_base = benchmark base vector count.

    Budget fractions always include 1.0, so max_budget_dir = benchmark_n_base.
    This is the count of vectors in base_vectors.npy for the benchmark, NOT the
    full dataset size stored in summary.json's num_base_vectors field.
    """
    sizes: dict = {}
    for bm_dir in sorted(param_run_dir.iterdir()):
        if not bm_dir.is_dir():
            continue
        cs = _cache_size_from_name(bm_dir.name)
        if cs in sizes:
            continue
        b_dirs = [d for d in bm_dir.iterdir()
                  if d.is_dir() and d.name.startswith('B')]
        if b_dirs:
            max_b = max(b_dirs, key=lambda d: int(d.name[1:]))
            sizes[cs] = int(max_b.name[1:])
    return sizes

#-------------------------------------------------------------------------------
# Plot functions
#-------------------------------------------------------------------------------

def _plot_grid(
    all_data: dict,
    base_sizes: dict,
    sigma_vals: list,
    kt_vals: list,
    outdir: Path,
    x_fn,
    y_fn,
    title: str,
    fname: str,
    ylabel: str,
    xlabel: str,
    ref_line: bool = False,
    pct_x: bool = True,
    pct_y: bool = True,
    ylim=None,
    int_y: bool = False,
) -> None:
    """Generic sigma x K_test grid plot (K_cache always equals K_test in the
    real pipeline, so there is no separate K_cache dimension to plot; n_max
    is fixed in every real run, so it's noted in the title, not gridded).

    When ylim isn't given, a shared y-axis is computed from the global max
    y-value across every cell instead of letting each subplot auto-scale to
    its own local max -- otherwise cells with very different magnitudes
    (e.g. candidate counts across cache sizes) become visually misleading
    to compare."""
    _, size_color = _size_color_map(all_data)

    if ylim is None:
        global_max = 0.0
        for (sigma, kt), run_data in all_data.items():
            for bm, bm_data in run_data.items():
                nb = base_sizes.get(_cache_size_from_name(bm))
                if not nb:
                    continue
                for b, entry in bm_data.items():
                    y = y_fn(entry, b, nb)
                    if y is not None:
                        global_max = max(global_max, y)
        ylim = (0, global_max * 1.05) if global_max > 0 else None

    n_rows, n_cols = len(sigma_vals), len(kt_vals)
    fig, axes = _make_subplots(n_rows, n_cols)
    fig.suptitle(title, fontsize=12)

    for r, sigma in enumerate(sigma_vals):
        for c, kt in enumerate(kt_vals):
            ax = axes[r][c]
            run_data = all_data.get((sigma, kt), {})
            cell_bms = sorted(run_data.keys(), key=_cache_size_from_name)
            for bm in cell_bms:
                cs = _cache_size_from_name(bm)
                nb = base_sizes.get(cs)
                budgets = sorted(run_data[bm].keys(), key=_budget_int)
                xs, ys = [], []
                for b in budgets:
                    entry = run_data[bm].get(b)
                    if not entry or not nb:
                        continue
                    x = x_fn(entry, b, nb)
                    y = y_fn(entry, b, nb)
                    if x is not None and y is not None:
                        xs.append(x)
                        ys.append(y)
                if xs:
                    ax.plot(xs, ys, marker='o', markersize=3,
                            color=size_color[cs], label=f'Cache {cs}')
            if ref_line:
                ax.plot([0, 1], [0, 1], linestyle='--', color='gray',
                        linewidth=0.8, alpha=0.5)
            if pct_x:
                _pct_fmt(ax, 'x')
                ax.set_xlim(0, 1)
            if pct_y:
                _pct_fmt(ax, 'y')
            if ylim is not None:
                ax.set_ylim(*ylim)
            if int_y:
                ax.yaxis.set_major_locator(
                    mticker.MaxNLocator(integer=True, min_n_ticks=2)
                )
                ax.set_ylim(bottom=0)
            ax.set_title(f'$\\sigma={sigma:.2f},\\ K_{{test}}={kt}$', fontsize=9)
            if c == 0:
                ax.set_ylabel(ylabel)
            if r == n_rows - 1:
                ax.set_xlabel(xlabel)

    _shared_legend(fig, axes, n_rows, n_cols)
    plt.tight_layout()
    _save_fig(fig, outdir, fname)


def _plot_grid_train_test(
    all_data: dict,
    base_sizes: dict,
    sigma_vals: list,
    kt_vals: list,
    outdir: Path,
    x_fn,
    y_fn_train,
    y_fn_test,
    title: str,
    fname: str,
    ylabel: str,
    xlabel: str,
    ref_line: bool = True,
    ylim=(0, 1.05),
) -> None:
    """sigma x K_test grid, train and test series overlaid per cache size
    on the same axes -- train: triangle marker, dashed line; test: square
    marker, solid line; same color per cache size for both series. Avoids
    the plot-per-quantity-per-side proliferation the single-series
    _plot_grid would otherwise require (one file for train, one for test)."""
    cache_sizes, size_color = _size_color_map(all_data)
    n_rows, n_cols = len(sigma_vals), len(kt_vals)
    fig, axes = _make_subplots(n_rows, n_cols)
    fig.suptitle(title, fontsize=12)

    for r, sigma in enumerate(sigma_vals):
        for c, kt in enumerate(kt_vals):
            ax = axes[r][c]
            run_data = all_data.get((sigma, kt), {})
            cell_bms = sorted(run_data.keys(), key=_cache_size_from_name)
            for bm in cell_bms:
                cs = _cache_size_from_name(bm)
                nb = base_sizes.get(cs)
                budgets = sorted(run_data[bm].keys(), key=_budget_int)
                xs, ys_train, ys_test = [], [], []
                for b in budgets:
                    entry = run_data[bm].get(b)
                    if not entry or not nb:
                        continue
                    x = x_fn(entry, b, nb)
                    yt = y_fn_train(entry, b, nb)
                    ys = y_fn_test(entry, b, nb)
                    if x is not None and yt is not None and ys is not None:
                        xs.append(x)
                        ys_train.append(yt)
                        ys_test.append(ys)
                if xs:
                    ax.plot(xs, ys_train, marker='^', markersize=3, linestyle='--',
                            color=size_color[cs], label=f'Cache {cs} - train')
                    ax.plot(xs, ys_test, marker='s', markersize=3, linestyle='-',
                            color=size_color[cs], label=f'Cache {cs} - test')
            if ref_line:
                ax.plot([0, 1], [0, 1], linestyle=':', color='gray',
                        linewidth=0.8, alpha=0.5)
            _pct_fmt(ax, 'x')
            ax.set_xlim(0, 1)
            _pct_fmt(ax, 'y')
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.set_title(f'$\\sigma={sigma:.2f},\\ K_{{test}}={kt}$', fontsize=9)
            if c == 0:
                ax.set_ylabel(ylabel)
            if r == n_rows - 1:
                ax.set_xlabel(xlabel)

    _shared_legend_train_test(fig, axes, n_rows, n_cols, cache_sizes)
    plt.tight_layout()
    _save_fig(fig, outdir, fname)


def _plot_hitrate(all_data, base_sizes, sigma_vals, kt_vals, outdir, n_max) -> None:
    _plot_grid_train_test(
        all_data, base_sizes, sigma_vals, kt_vals, outdir,
        x_fn=lambda e, b, nb: e['vectors_cached'] / nb,
        y_fn_train=lambda e, b, nb: e['train_hit_rate'],
        y_fn_test=lambda e, b, nb: e['test_hit_rate'],
        title=f'Memory Utilization vs. Hit Rate, Train vs. Test (n={n_max})',
        fname='memory_vs_hitrate.png',
        ylabel='Hit Rate',
        xlabel='Memory Utilization',
    )


def _plot_candidates_cached(all_data, base_sizes, sigma_vals, kt_vals, outdir, n_max) -> None:
    _plot_grid(
        all_data, base_sizes, sigma_vals, kt_vals, outdir,
        x_fn=lambda e, b, nb: e['vectors_cached'] / nb,
        y_fn=lambda e, b, nb: e['num_entries'],
        title=f'Memory Utilization vs. Candidates Cached (n={n_max})',
        fname='memory_vs_candidates_cached.png',
        ylabel='Candidates Cached',
        xlabel='Memory Utilization',
        ref_line=False,
        pct_x=True, pct_y=False,
        ylim=None,
        int_y=True,
    )


def _plot_recall_vs_sigma(all_data: dict, base_sizes: dict, sigma_vals: list,
                           kt_vals: list, outdir: Path) -> None:
    """One combined grid file: rows=K_test, columns=budget fraction, x=sigma,
    y=mean_recall, one line per cache/train size per cell."""
    if len(sigma_vals) < 2:
        return  # nothing to plot sigma-vs-anything against with a single sigma
    cache_sizes, size_color = _size_color_map(all_data)

    all_budgets = sorted(
        {b for key, run_data in all_data.items() for bm in run_data for b in run_data[bm]},
        key=_budget_int,
    )
    if not all_budgets:
        return

    n_rows, n_cols = len(kt_vals), len(all_budgets)
    fig, axes = _make_subplots(n_rows, n_cols, row_h=4.0, col_w=4.0)
    fig.suptitle('Recall vs. Sigma', fontsize=12)

    for r, kt in enumerate(kt_vals):
        for c, budget_str in enumerate(all_budgets):
            ax = axes[r][c]
            panel_ys = []
            for cs in cache_sizes:
                xs, ys = [], []
                for sigma in sigma_vals:
                    run_data = all_data.get((sigma, kt), {})
                    bm = next(
                        (bm for bm in run_data if _cache_size_from_name(bm) == cs), None
                    )
                    if bm is None:
                        continue
                    entry = run_data[bm].get(budget_str)
                    if not entry:
                        continue
                    xs.append(sigma)
                    ys.append(entry['mean_recall'])
                if xs:
                    ax.plot(xs, ys, marker='o', markersize=3,
                            color=size_color[cs], label=f'Cache {cs}')
                    panel_ys.extend(ys)
            ax.set_xlim(min(sigma_vals) - 0.02, 1.02)
            # Adaptive y-axis, not a fixed (0, 1.05) range: the real sigma
            # effect on recall is small (often < 1 percentage point), and a
            # fixed 0-100% axis flattens it to invisibility. Zoom to the
            # panel's own data range with a small margin instead.
            if panel_ys:
                y_min, y_max = min(panel_ys), max(panel_ys)
                if y_max - y_min < 1e-6:
                    margin = 0.05 if y_max > 0 else 0.05
                    ax.set_ylim(max(0.0, y_min - margin), min(1.05, y_max + margin))
                else:
                    pad = (y_max - y_min) * 0.15
                    ax.set_ylim(max(0.0, y_min - pad), min(1.05, y_max + pad))
            else:
                ax.set_ylim(0, 1.05)
            _pct_fmt(ax, 'y')
            ax.set_title(f'$K_{{test}}={kt},\\ B={_budget_int(budget_str)}$', fontsize=9)
            if c == 0:
                ax.set_ylabel('Mean Recall')
            if r == n_rows - 1:
                ax.set_xlabel('Sigma')

    _shared_legend(fig, axes, n_rows, n_cols)
    plt.tight_layout()
    _save_fig(fig, outdir, 'recall_vs_sigma.png')


#-------------------------------------------------------------------------------
# Debug output
#-------------------------------------------------------------------------------

def _write_debug(
    all_data: dict,
    base_sizes: dict,
    debug_dir: Path,
    label: str,
    n_max,
) -> None:
    """Write per-entry and aggregate debug JSON."""
    debug_dir.mkdir(parents=True, exist_ok=True)
    params_out: dict = {}
    combo_stats: list = []
    max_budget_hit_records: list = []  # (param_key, bm, exact_test_hit_rate)

    for key, run_data in sorted(all_data.items()):
        sigma, kt = key
        param_key = f'sigma{sigma:.2f}_k{kt}'

        benchmarks_out: dict = {}
        max_budget_hits: list = []

        for bm, bm_data in sorted(run_data.items()):
            cs = _cache_size_from_name(bm)
            n_base = base_sizes.get(cs)
            n_train_q = cs
            budgets_out: dict = {}
            bm_hits: list = []

            for budget_str in sorted(bm_data.keys(), key=_budget_int):
                entry = bm_data[budget_str]
                budget = _budget_int(budget_str)
                vc = entry['vectors_cached']
                ne = entry['num_entries']
                hr = entry['test_hit_rate']
                thr = entry['train_hit_rate']
                mr = entry['mean_recall']
                tc = entry['train_coverage']
                tesc = entry['test_coverage']
                traj = entry.get('trajectory', [])

                mem_util = vc / n_base if n_base else 0.0
                bud_util = vc / budget if budget else 0.0

                step1 = traj[0] if traj else {}
                tc_vals = [t['cig_train_coverage'] for t in traj]
                tc_p50 = statistics.median(tc_vals) if tc_vals else 0.0
                step1_vc = int(step1.get('vectors_cached', 0))
                step1_tc = float(step1.get('cig_train_coverage', 0.0))

                budgets_out[budget_str] = {
                    'budget': budget,
                    'test_hit_rate': round(hr, 6),
                    'train_hit_rate': round(thr, 6),
                    'mean_recall': round(mr, 6),
                    'memory_util': round(mem_util, 6),
                    'budget_utilization': round(bud_util, 4),
                    'vectors_cached': vc,
                    'num_entries': ne,
                    'train_coverage_final': round(tc, 6),
                    'test_coverage_final': round(tesc, 6),
                    'trajectory_steps': len(traj),
                    'traj_vectors_at_step1': step1_vc,
                    'traj_train_cov_at_step1': round(step1_tc, 6),
                    'traj_train_cov_p50': round(tc_p50, 6),
                    'traj_step1_frac_of_base': round(step1_vc / n_base, 6) if n_base else 0.0,
                }
                bm_hits.append(hr)

            max_b, max_entry = _max_budget_entry(bm_data)
            any_nonzero = any(h > 1e-4 for h in bm_hits)
            max_hit = max(bm_hits) if bm_hits else 0.0
            max_vc = max_entry.get('vectors_cached', 0)
            max_tc = max_entry.get('train_coverage', 0.0)
            max_mem = max_vc / n_base if n_base else 0.0
            bud_at_max = _budget_int(max_b) if max_b else 1
            bud_util_at_max = max_vc / bud_at_max if bud_at_max else 0.0

            notes = []
            if not any_nonzero:
                notes.append('hit_rate near zero across all budgets')
            if max_tc < 1e-4:
                notes.append('train_coverage near zero at max budget')
            if max_entry.get('num_entries', 0) == 0:
                notes.append('no candidates cached at max budget')
            if bud_util_at_max < 0.5:
                notes.append(f'budget utilization low at max budget ({bud_util_at_max:.1%})')

            benchmarks_out[bm] = {
                'n_base': n_base,
                'n_train_queries': n_train_q,
                'budgets': budgets_out,
                'per_benchmark_sanity': {
                    'any_nonzero_hitrate': any_nonzero,
                    'max_hit_rate': round(max_hit, 6),
                    'max_memory_util': round(max_mem, 6),
                    'max_train_coverage': round(max_tc, 6),
                    'budget_utilization_at_max_budget': round(bud_util_at_max, 4),
                    'num_entries_at_max_budget': max_entry.get('num_entries', 0),
                    'note': '; '.join(notes) if notes else 'ok',
                },
            }
            max_budget_hits.append(max_hit)
            # unrounded, for the exact-1.0 sanity check below (round(x,6)
            # above could theoretically mask a value that's close to but
            # not exactly 1.0 -- this uses the raw float)
            max_budget_hit_records.append((param_key, bm, max_hit))

        max_tc_vals = [
            _max_budget_entry(bm_data)[1].get('train_coverage', 0.0)
            for bm_data in run_data.values() if bm_data
        ]
        max_mem_vals = [
            _max_budget_entry(bm_data)[1].get('vectors_cached', 0)
            / base_sizes[_cache_size_from_name(bm)]
            for bm, bm_data in run_data.items()
            if bm_data and base_sizes.get(_cache_size_from_name(bm))
        ]

        mean_hr = statistics.mean(max_budget_hits) if max_budget_hits else 0.0
        max_hr = max(max_budget_hits) if max_budget_hits else 0.0
        frac_nonzero = (
            sum(1 for h in max_budget_hits if h > 1e-4) / len(max_budget_hits)
            if max_budget_hits else 0.0
        )

        params_out[param_key] = {
            'benchmarks': benchmarks_out,
            'param_combo_summary': {
                'n_benchmarks': len(run_data),
                'mean_hit_rate_at_max_budget': round(mean_hr, 6),
                'max_hit_rate_at_max_budget': round(max_hr, 6),
                'frac_benchmarks_with_nonzero_hitrate': round(frac_nonzero, 4),
                'mean_memory_util_at_max_budget': round(statistics.mean(max_mem_vals), 6) if max_mem_vals else 0.0,
                'mean_train_coverage_at_max_budget': round(statistics.mean(max_tc_vals), 6) if max_tc_vals else 0.0,
            },
        }
        combo_stats.append((param_key, mean_hr, max_hr))

    best_combo = max(combo_stats, key=lambda x: x[2], default=(None, 0, 0))
    worst_combo = min(combo_stats, key=lambda x: x[1], default=(None, 0, 0))
    frac_combos_nz = sum(1 for _, _, mh in combo_stats if mh > 1e-4) / max(len(combo_stats), 1)

    # Sanity check (Phase 3 Round 14): literal 1.0 hit rate at max budget
    # would be surprising -- the CIG condition isn't automatically satisfied
    # just because the base is fully covered, so this flags any combo that
    # lands there for a closer look, using the exact (unrounded) value.
    exact_ones = [
        {'param': pk, 'benchmark': bm, 'test_hit_rate': hr}
        for pk, bm, hr in max_budget_hit_records
        if hr == 1.0
    ]

    out = {
        'dataset': label,
        'method': 'ball',
        'n_max': n_max,
        'params': params_out,
        'dataset_summary': {
            'total_param_combos': len(params_out),
            'frac_combos_with_nonzero_hitrate': round(frac_combos_nz, 4),
            'any_nonzero_hitrate': any(mh > 1e-4 for _, _, mh in combo_stats),
            'best_combo': best_combo[0],
            'global_max_hit_rate': round(best_combo[2], 6),
            'worst_mean_combo': worst_combo[0],
        },
        'sanity_checks': {
            'n_combos_with_exact_1.0_hit_rate_at_max_budget': len(exact_ones),
            'exact_1.0_hit_rate_combos': exact_ones,
        },
    }

    out_path = debug_dir / f'{label}_ball_debug.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  debug: {out_path}")

#-------------------------------------------------------------------------------
# Main analysis
#-------------------------------------------------------------------------------

def analyze_dataset(
    run_dir: Path,
    processed_base: Path,
    dataset_label: str,
    debug_dir: Path | None = None,
    sigma_values: list | None = None,
) -> None:
    """sigma_values: swept sigma values (e.g. [0.7, 0.8, 0.9, 1.0]) if the raw
    output has a sigma_* nesting level, else None/empty for the flat (single
    default sigma=1.0, unnested) layout. All sigma values are combined into
    one set of plots per dataset (rows=sigma), not one plot set per sigma."""
    print(f"\n=== {dataset_label} ===")
    base_sizes_cache: dict = {}

    if not (run_dir and run_dir.exists()):
        return
    method_raw = run_dir / 'ball'
    if not method_raw.exists():
        return

    print("\n-- ball --")
    all_data: dict = {}
    n_vals_seen: set = set()
    sweep_sigmas = sigma_values if sigma_values else [1.0]
    flat_layout = not sigma_values  # no sigma_*/ nesting present on disk

    for param_dir in sorted(method_raw.iterdir()):
        if not param_dir.is_dir():
            continue
        m = re.fullmatch(r'nmax(\d+)_k(\d+)', param_dir.name)
        if not m:
            continue
        n, k_test = int(m.group(1)), int(m.group(2))
        n_vals_seen.add(n)
        for sigma in sweep_sigmas:
            run_data = _load_run_data(param_dir, sigma=None if flat_layout else sigma)
            all_data[(sigma, k_test)] = run_data
        base_sizes_cache.update(_base_size(param_dir))

    if len(n_vals_seen) > 1:
        print(f"  WARNING: multiple n_max values found {sorted(n_vals_seen)}, "
              f"expected exactly one (n_max is not swept in real runs)")
    n_max = next(iter(sorted(n_vals_seen)), None)

    sigma_vals = sorted({s for (s, _) in all_data})
    kt_vals = sorted({k for (_, k) in all_data})
    outdir = processed_base / 'ball'

    if all_data:
        if debug_dir:
            _write_debug(all_data, base_sizes_cache, debug_dir, dataset_label, n_max)
        _plot_hitrate(all_data, base_sizes_cache, sigma_vals, kt_vals, outdir, n_max)
        _plot_candidates_cached(all_data, base_sizes_cache, sigma_vals, kt_vals, outdir, n_max)
        _plot_recall_vs_sigma(all_data, base_sizes_cache, sigma_vals, kt_vals, outdir)


def _latest_with(base: Path, method: str) -> Path | None:
    """Latest run_ts dir containing `method/` directly under it."""
    if not base.exists():
        return None
    stamped = sorted(
        d for d in base.iterdir()
        if d.is_dir() and re.fullmatch(r'\d{8}_\d{6}', d.name) and (d / method).exists()
    )
    return stamped[-1] if stamped else None


def _sigma_values_in(method_dir: Path | None) -> list:
    """Sigma values swept for this method, found by peeking into one representative
    B{budget}/ dir (sigma nests as B{budget}/sigma_{val}/, see population_runner.py).
    Returns [] for a flat (single default sigma=1.0, unnested) layout."""
    if not method_dir or not method_dir.exists():
        return []
    for param_dir in sorted(method_dir.iterdir()):
        if not param_dir.is_dir():
            continue
        for bm_dir in sorted(param_dir.iterdir()):
            if not bm_dir.is_dir():
                continue
            for b_dir in sorted(bm_dir.glob('B*')):
                sigmas = sorted(
                    float(re.fullmatch(r'sigma_(\d+\.\d+)', d.name).group(1))
                    for d in b_dir.iterdir()
                    if d.is_dir() and re.fullmatch(r'sigma_\d+\.\d+', d.name)
                )
                if sigmas:
                    return sigmas
    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze proposal-based population pipeline results"
    )
    parser.add_argument('--esci-dir', type=Path, default=None,
                        help='Raw ESCI run dir (default: latest with ball/ output)')
    parser.add_argument('--sift-dir', type=Path, default=None,
                        help='Raw SIFT run dir (default: latest with ball/ output)')
    parser.add_argument('--run-ts', type=str, default=None)
    parser.add_argument('--no-debug', action='store_true',
                        help='Skip writing debug JSON files')
    args = parser.parse_args()

    from datetime import datetime
    run_ts = args.run_ts or datetime.now().strftime("%Y%m%d_%H%M%S")

    esci_raw = Path('simulations/esci/raw/population_proposal')
    sift_raw = Path('simulations/sift/raw/population_proposal')

    esci_dir = args.esci_dir or _latest_with(esci_raw, 'ball')
    sift_dir = args.sift_dir or _latest_with(sift_raw, 'ball')

    esci_processed_base = Path('simulations/esci/processed/population_proposal') / run_ts
    sift_processed_base = Path('simulations/sift/processed/population_proposal') / run_ts
    debug_base = None if args.no_debug else Path('simulations/debug/population_proposal') / run_ts

    for dataset_label, run_dir, processed_base in [
        ('ESCI', esci_dir, esci_processed_base),
        ('SIFT', sift_dir, sift_processed_base),
    ]:
        if run_dir is None:
            print(f"\n{dataset_label}: no raw run dir found, skipping")
            continue

        sigma_values = _sigma_values_in(run_dir / 'ball')
        if sigma_values:
            print(f"\n{dataset_label}: found sigma values: {sigma_values}")
        analyze_dataset(run_dir, processed_base, dataset_label, debug_base,
                         sigma_values=sigma_values)

    print(f"\nDone. run_ts={run_ts}")


if __name__ == '__main__':
    main()
