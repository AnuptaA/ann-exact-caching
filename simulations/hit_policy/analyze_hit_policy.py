#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys, json, re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import argparse
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import seaborn as sns

from simulations.core.analysis_helpers import (
    ALGO_ORDER, ALGO_LABELS, lbl, lbls, cap_metric, cap_metrics,
    save_fig, pct_fmt, load_summaries,
)

sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.dpi':      150,
    'font.size':       13,
    'axes.titlesize':  15,
    'axes.labelsize':  14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
})

#-------------------------------------------------------------------------------

NO_UNION_ALGOS: Set[str] = {'lemma1_no_union', 'lemma2_no_union'}

#-------------------------------------------------------------------------------
# ESCI benchmark name parsing, unused for SIFT

_BENCH_RE = re.compile(r'esci_c(\d+)k(\d+)_t(\d+)n(\d+)_s(\d+)')

def _parse_esci_name(name: str) -> Dict:
    """Parse ESCI benchmark name into component integers."""
    m = _BENCH_RE.search(name)
    if not m:
        return {}
    return dict(
        num_cache=int(m.group(1)), cache_K=int(m.group(2)),
        num_test=int(m.group(3)),  test_N=int(m.group(4)),
        seed=int(m.group(5)),
    )

def _config_key(name: str) -> str:
    """Strip seed suffix to get a config-level key for seed aggregation."""
    return re.sub(r'_s\d+$', '', name)

def _classify_set(num_test: int) -> str:
    """Classify a benchmark into set2 (cache scaling) or set3 (K/N ratios)."""
    return 'set3' if num_test == 256 else 'set2'

#-------------------------------------------------------------------------------
# style / ordering helpers

def ordered_algos(algos, include_no_union: bool = False) -> List[str]:
    algos = set(algos)
    if not include_no_union:
        algos -= NO_UNION_ALGOS
    return [a for a in ALGO_ORDER if a in algos] + sorted(algos - set(ALGO_ORDER))

def ordered_metrics(metrics) -> List[str]:
    preferred = ['euclidean', 'angular', 'cosine']
    metrics = set(metrics)
    return [m for m in preferred if m in metrics] + sorted(metrics - set(preferred))

def algo_palette(algos) -> Dict[str, tuple]:
    ordered = ordered_algos(algos, include_no_union=True)
    colors = sns.color_palette('tab10', len(ordered))
    return {a: c for a, c in zip(ordered, colors)}

def _set_cache_size_xaxis(ax, cache_sizes):
    """Log2 x-axis for cache size plots - ticks show actual cache size values."""
    ax.set_xscale('log', base=2)
    ax.set_xticks(cache_sizes)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: str(int(x))))
    ax.tick_params(axis='x', labelrotation=45)

def _add_footnote(fig: plt.Figure, text: str):
    fig.text(0.01, -0.01, text, fontsize=7.5, color='#555555',
             va='top', ha='left', transform=fig.transFigure)

def _left_cbar_fig(n_rows, n_cols, cmap, vmin, vmax, cbar_label,
                   fig_w, fig_h, wspace=0.05):
    fig = plt.figure(figsize=(fig_w + 1.2, fig_h))
    gs = GridSpec(n_rows, n_cols + 1, figure=fig,
                  width_ratios=[0.05] + [1] * n_cols,
                  hspace=0.4, wspace=wspace)
    cbar_ax = fig.add_subplot(gs[:, 0])
    sm = mpl.cm.ScalarMappable(cmap=cmap,
                                norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label=cbar_label)
    cbar_ax.yaxis.set_label_position('left')
    cbar_ax.yaxis.tick_left()
    axes = [[fig.add_subplot(gs[r, c + 1]) for c in range(n_cols)]
            for r in range(n_rows)]
    for row in axes:
        for ax in row:
            ax.yaxis.set_label_position('right')
            ax.yaxis.tick_right()
    return fig, axes

#-------------------------------------------------------------------------------
# data loading

def build_summary_df(summaries: List[Dict], parse_name_fn=None) -> pd.DataFrame:
    """
    parse_name_fn: optional callable(name) -> dict with 'num_test'/'seed' keys,
    used to classify ESCI benchmarks into set2/set3 and to expose the seed.
    """
    rows = []
    for ds in summaries:
        name = ds['dataset']
        cfg = ds['config']
        pars = parse_name_fn(name) if parse_name_fn else {}
        set_id = None
        if parse_name_fn:
            num_test = pars.get('num_test', cfg.get('num_test_queries', 0))
            set_id = _classify_set(num_test)
        for res in ds['results']:
            rows.append({
                'dataset':            name,
                'config_key':         _config_key(name),
                'set_id':             set_id,
                'seed':               pars.get('seed'),
                'algorithm':          res['algorithm'],
                'metric':             res['metric'],
                'hit_rate':           res['hit_rate'],
                'total_queries':      res['total_queries'],
                'cache_hits':         res['cache_hits'],
                'lemma1_hits':        res.get('lemma1_hits', 0),
                'lemma2_hits':        res.get('lemma2_hits', 0),
                'avg_distance_calcs': res.get('avg_distance_calcs'),
                'avg_time_us':        res.get('avg_time_us'),
                'cache_K':            cfg.get('cache_K'),
                'test_N':             cfg.get('test_N'),
                'num_cache_queries':  cfg.get('num_cache_queries'),
                'num_base_vectors':   cfg.get('num_base_vectors'),
                'dimension':          cfg.get('dimension'),
            })
    return pd.DataFrame(rows)

def load_per_query_df(raw_dir: str, summaries: List[Dict], parse_name_fn=None) -> pd.DataFrame:
    raw_path = Path(raw_dir)
    cfg_lookup = {ds['dataset']: ds['config'] for ds in summaries}
    dfs = []
    for p in sorted(raw_path.glob('*/per_query.csv')):
        name = p.parent.name
        try:
            df   = pd.read_csv(p)
            cfg  = cfg_lookup.get(name, {})
            pars = parse_name_fn(name) if parse_name_fn else {}
            set_id = None
            if parse_name_fn:
                num_test = pars.get('num_test', cfg.get('num_test_queries', 0))
                set_id = _classify_set(num_test)
            df['dataset']           = name
            df['config_key']        = _config_key(name)
            df['set_id']            = set_id
            df['num_cache_queries'] = cfg.get('num_cache_queries')
            df['cache_K']           = cfg.get('cache_K')
            df['test_N']            = cfg.get('test_N')
            dfs.append(df)
        except Exception as e:
            print(f"  Could not load {p}: {e}")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame(columns=[
        'dataset', 'algorithm', 'metric', 'query_id',
        'cache_hit', 'hit_source', 'distance_calcs', 'time_us',
        'config_key', 'set_id', 'num_cache_queries', 'cache_K', 'test_N',
    ])

#-------------------------------------------------------------------------------
# aggregation helper

def _get_agg(a_sub: pd.DataFrame, value_col: str,
             group_vals: list, group_col: str):
    """Average value_col per benchmark (seed isolation), then mean +/- std across seeds.

    Groups by 'dataset' (the full benchmark directory name) to isolate seeds so this is
    equivalent to grouping by config_key for either dataset.

    Returns (ys, errs) aligned to group_vals; NaN for missing values, 0 for missing std.
    """
    a_sub = a_sub.copy()
    per_seed = a_sub.groupby([group_col, 'dataset'])[value_col].mean().reset_index()
    agg = per_seed.groupby(group_col)[value_col].agg(['mean', 'std'])
    ys = [float(agg.loc[v, 'mean']) if v in agg.index else np.nan for v in group_vals]
    errs = [float(agg.loc[v, 'std'])  if v in agg.index else 0.0    for v in group_vals]
    errs = [0.0 if np.isnan(e) else e for e in errs]
    return ys, errs

#-------------------------------------------------------------------------------
# coverage computation

def compute_coverage(
    dataset_name: str,
    K: int,
    num_cache_queries: int,
    num_base_vectors: int,
    datasets_base_dir: Path,
    gt_filename_fn,
) -> Optional[float]:
    npz = datasets_base_dir / dataset_name / gt_filename_fn(K)
    if not npz.exists():
        print(f'  [coverage] missing: {npz}')
        return None
    data = np.load(npz)
    all_idx = np.concatenate([data[f'indices_{i}'] for i in range(num_cache_queries)])
    return float(len(np.unique(all_idx)) / num_base_vectors)


def compute_set2_coverage_df(sub: pd.DataFrame, datasets_base_dir: Path, gt_filename_fn) -> pd.DataFrame:
    unique = (
        sub[['dataset', 'num_cache_queries', 'cache_K', 'num_base_vectors']]
        .drop_duplicates()
        .copy()
    )
    rows = []
    for _, row in unique.iterrows():
        name = row['dataset']
        C = int(row['num_cache_queries'])
        K = int(row['cache_K'])
        B = int(row['num_base_vectors'])
        ac = compute_coverage(name, K, C, B, datasets_base_dir, gt_filename_fn)
        ec = 1.0 - (1.0 - K / B) ** C
        rows.append({'dataset': name, 'cache_size': C,
                     'coverage_actual': ac, 'coverage_estimated': ec})
    return pd.DataFrame(rows)

#-------------------------------------------------------------------------------
# coverage plots

def plot_set2_coverage_and_hit_rate(
    sub: pd.DataFrame,
    cov_df: pd.DataFrame,
    outdir: Path,
):
    algos = ordered_algos(sub['algorithm'].unique(), include_no_union=False)
    cache_sizes = sorted(cov_df['cache_size'].unique().astype(int))

    cov_agg = (
        cov_df.dropna(subset=['coverage_actual'])
        .groupby('cache_size')['coverage_actual']
        .mean()
        .reindex(cache_sizes)
    )
    cov_mean = cov_agg.values
    est_vals = (
        cov_df.groupby('cache_size')['coverage_estimated']
        .first()
        .reindex(cache_sizes)
        .values
    )
    cs_arr = np.array(cache_sizes, dtype=float)

    n_rows = len(algos)
    fig, axes = plt.subplots(n_rows, 1, figsize=(6, 4 * n_rows), sharex=True, sharey=True)
    if n_rows == 1:
        axes = [axes]

    h_cov_handle = None
    h_est_handle = None
    h_hr_handle = None
    for row_idx, algo in enumerate(algos):
        ax = axes[row_idx]
        h_cov, = ax.plot(cs_arr, cov_mean, color='red', linewidth=2.5,
                         label='Coverage (actual)')
        if h_cov_handle is None:
            h_cov_handle = h_cov
        h_est, = ax.plot(cs_arr, est_vals, color='blue', linewidth=2,
                         linestyle='--', label='Coverage (estimated)')
        if h_est_handle is None:
            h_est_handle = h_est
        a_sub = sub[sub['algorithm'] == algo].copy()
        if not a_sub.empty:
            ys, _ = _get_agg(a_sub, 'hit_rate', cache_sizes, 'num_cache_queries')
            h_hr, = ax.plot(cs_arr, ys, color='green', linewidth=1.8, marker='o', markersize=4,
                            label='Hit Rate')
            if h_hr_handle is None:
                h_hr_handle = h_hr
        pct_fmt(ax)
        ax.set_ylim(0, 1.05)
        _set_cache_size_xaxis(ax, cache_sizes)
        ax.set_ylabel(lbl(algo))
        if row_idx == n_rows - 1:
            ax.set_xlabel('Cache Size')

    legend_handles = [h for h in [h_cov_handle, h_est_handle, h_hr_handle] if h is not None]
    legend_labels = ['Coverage (actual)', 'Coverage (estimated)', 'Hit Rate'][:len(legend_handles)]
    if legend_handles:
        fig.legend(legend_handles, legend_labels,
                   loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.suptitle('Set 2 - Coverage and Hit Rate vs Cache Size', y=1.02)
    plt.tight_layout()
    save_fig(fig, outdir, 'set2_coverage_and_hit_rate.png')


def plot_set2_coverage_estimated_vs_actual(cov_df: pd.DataFrame, outdir: Path):
    cache_sizes = sorted(cov_df['cache_size'].unique().astype(int))
    cs_arr = np.array(cache_sizes, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 5))
    cov_agg = (
        cov_df.dropna(subset=['coverage_actual'])
        .groupby('cache_size')['coverage_actual']
        .mean()
        .reindex(cache_sizes)
    )
    h_mean, = ax.plot(cs_arr, cov_agg.values, color='steelblue', linewidth=2.5,
                      label='Actual coverage (mean)')
    est = cov_df.groupby('cache_size')['coverage_estimated'].first().reindex(cache_sizes).values
    h_est, = ax.plot(cs_arr, est, color='darkorange', linewidth=2,
                     linestyle='--', label='Estimated: 1-(1-K/B)^C')
    pct_fmt(ax)
    ax.set_ylim(0, 1.05)
    _set_cache_size_xaxis(ax, cache_sizes)
    ax.set_xlabel('Cache Size')
    ax.set_ylabel('Database Coverage')
    ax.legend([h_mean, h_est], ['Actual coverage (mean)', 'Estimated: 1-(1-K/B)^C'], fontsize=9)
    fig.suptitle('Set 2 - Estimated vs Actual Coverage', y=1.02)
    plt.tight_layout()
    save_fig(fig, outdir, 'set2_coverage_estimated_vs_actual.png')

#-------------------------------------------------------------------------------
# global plots

def plot_global_hit_rate_heatmap(df: pd.DataFrame, outdir: Path):
    """Hit rate heatmap averaged across all benchmarks: algo (rows) x metric (cols)."""
    sub = df[~df['algorithm'].isin(NO_UNION_ALGOS)]
    algos = ordered_algos(sub['algorithm'].unique())
    metrics = ordered_metrics(sub['metric'].unique())

    mat = np.full((len(algos), len(metrics)), np.nan)
    for i, a in enumerate(algos):
        for j, m in enumerate(metrics):
            vals = sub[(sub['algorithm'] == a) & (sub['metric'] == m)]['hit_rate']
            if len(vals):
                mat[i, j] = vals.mean()

    fig_w = max(5, len(metrics) * 2.5)
    fig_h = max(4, len(algos) * 0.9)
    fig, ax_grid = _left_cbar_fig(1, 1, 'YlOrRd', 0, 1, 'Hit Rate', fig_w, fig_h)
    ax = ax_grid[0][0]
    sns.heatmap(mat, annot=True, fmt='.1%', cmap='YlOrRd', vmin=0, vmax=1,
                xticklabels=cap_metrics(metrics), yticklabels=lbls(algos),
                ax=ax, cbar=False)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title('Hit Rate by Algorithm - All Benchmarks', pad=12)
    save_fig(fig, outdir, 'hit_rate_heatmap_global.png')

#-------------------------------------------------------------------------------

def plot_lemma_breakdown_global(df: pd.DataFrame, outdir: Path, group_col: str, xtick_fmt):
    """Stacked bar: CIG / HGG share within Combined algorithm hits, grouped by group_col.

    When combined cache hits are zero for a group the bars show 0% with a footnote.
    """
    comb = df[df['algorithm'] == 'combined'].copy()
    if comb.empty:
        print("  No combined algorithm data, skipping lemma breakdown.")
        return

    comb['l1_share'] = np.where(comb['cache_hits'] > 0,
                                comb['lemma1_hits'] / comb['cache_hits'], 0.0)
    comb['l2_share'] = np.where(comb['cache_hits'] > 0,
                                comb['lemma2_hits'] / comb['cache_hits'], 0.0)
    has_zero_hits = (comb['cache_hits'] == 0).any()

    metrics = ordered_metrics(comb['metric'].unique())
    if group_col == 'num_cache_queries':
        group_vals = sorted(comb[group_col].dropna().unique().astype(int))
    else:
        group_vals = sorted(comb[group_col].dropna().unique())
    pal = sns.color_palette('tab10', 2)

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(max(5, len(group_vals) * 1.8 + 2) * len(metrics), 5),
                             sharey=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        sub = comb[comb['metric'] == metric].copy()
        agg = sub.groupby(group_col)[['l1_share', 'l2_share']].mean()
        agg = agg.reindex([v for v in group_vals if v in agg.index])
        x = np.arange(len(agg))
        ax.bar(x, agg['l1_share'], label='CIG', color=pal[0])
        ax.bar(x, agg['l2_share'], bottom=agg['l1_share'], label='HGG', color=pal[1])
        ax.set_xticks(x)
        ax.set_xticklabels([xtick_fmt(v) for v in agg.index], rotation=0)
        if group_col == 'num_cache_queries':
            ax.set_xlabel('Cache Size')
        ax.set_title(cap_metric(metric))
        pct_fmt(ax)

    axes[0].set_ylabel('Share of Combined Hits')
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in pal[:2]]
    axes[-1].legend(handles, ['CIG', 'HGG'],
                    bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    if has_zero_hits:
        _add_footnote(fig, '* Bars at 0% indicate zero combined cache hits for that group.')
    fig.suptitle('CIG vs HGG Contribution Within Combined Algorithm', y=1.02)
    plt.tight_layout()
    save_fig(fig, outdir, 'lemma_breakdown_global.png')

#-------------------------------------------------------------------------------

def plot_avg_time(df: pd.DataFrame, outdir: Path, group_col: str, xtick_fmt):
    """Log-scale bar chart of average query time per algorithm, grouped by group_col."""
    sub = df[~df['algorithm'].isin(NO_UNION_ALGOS)].copy()
    if sub.empty or sub['avg_time_us'].isna().all():
        print("  No timing data, skipping.")
        return

    algos = ordered_algos(sub['algorithm'].unique())
    metrics = ordered_metrics(sub['metric'].unique())
    if group_col == 'num_cache_queries':
        group_vals = sorted(sub[group_col].dropna().unique().astype(int))
    else:
        group_vals = sorted(sub[group_col].dropna().unique())
    pal = algo_palette(algos)

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(max(6, len(group_vals) * 2.5 + 2) * len(metrics), 5),
                             sharey=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        m_sub = sub[sub['metric'] == metric]
        n_grps = len(algos)
        width = 0.8 / n_grps
        x = np.arange(len(group_vals))
        for i, algo in enumerate(algos):
            ys_us, _ = _get_agg(m_sub[m_sub['algorithm'] == algo],
                                'avg_time_us', group_vals, group_col)
            ys = [v / 1000 if not np.isnan(v) else v for v in ys_us]
            ax.bar(x + (i - n_grps / 2 + 0.5) * width, ys, width,
                   label=lbl(algo), color=pal.get(algo))
        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels([xtick_fmt(v) for v in group_vals])
        if group_col == 'num_cache_queries':
            ax.set_xlabel('Cache Size')
        ax.set_title(cap_metric(metric))

    axes[0].set_ylabel('Mean Query Time (ms)')
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                    borderaxespad=0, fontsize=8)
    fig.suptitle('Average Query Time by Algorithm and Group', y=1.02)
    plt.tight_layout()
    save_fig(fig, outdir, 'avg_time_by_algorithm.png')

#-------------------------------------------------------------------------------
# cache-size scaling plots (SIFT: whole dataset; ESCI: pre-filtered to set2)

def plot_cache_size_scaling(
    sub: pd.DataFrame,
    outdir: Path,
    datasets_base_dir: Path,
    gt_filename_fn,
    title_prefix: str = '',
):
    """Line plots (hit rate, distance calcs) and heatmap vs cache size, seeds aggregated."""
    sub = sub[~sub['algorithm'].isin(NO_UNION_ALGOS)].copy()
    sub['num_cache_queries'] = sub['num_cache_queries'].astype(int)

    cache_sizes = sorted(sub['num_cache_queries'].dropna().unique().astype(int))
    if len(cache_sizes) < 2:
        print("  Only one cache size found, skipping cache scaling plots.")
        return

    algos = ordered_algos(sub['algorithm'].unique())
    metrics = ordered_metrics(sub['metric'].unique())
    pal = algo_palette(algos)

    # --- hit rate vs cache size ---
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics) + 2, 5), sharey=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        m_sub = sub[sub['metric'] == metric]
        for algo in algos:
            ys, _ = _get_agg(m_sub[m_sub['algorithm'] == algo],
                             'hit_rate', cache_sizes, 'num_cache_queries')
            ls = '--' if algo == 'lemma1' else '-'
            ax.plot(cache_sizes, ys, marker='o', label=lbl(algo),
                    color=pal[algo], linestyle=ls)
        pct_fmt(ax)
        ax.set_ylim(0, 1.05)
        _set_cache_size_xaxis(ax, cache_sizes)
        ax.set_xlabel('Cache Size')
        ax.set_title(cap_metric(metric))
    axes[0].set_ylabel('Hit Rate')
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                    borderaxespad=0, fontsize=8)
    fig.suptitle(f'{title_prefix}Hit Rate vs Cache Size', y=1.02)
    plt.tight_layout()
    save_fig(fig, outdir, 'hit_rate_vs_cache_size.png')

    # --- distance calcs vs cache size ---
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics) + 2, 5), sharey=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        m_sub = sub[sub['metric'] == metric]
        for algo in algos:
            ys, _ = _get_agg(m_sub[m_sub['algorithm'] == algo],
                             'avg_distance_calcs', cache_sizes, 'num_cache_queries')
            ls = '--' if algo == 'lemma1' else '-'
            ax.plot(cache_sizes, ys, marker='o', label=lbl(algo),
                    color=pal.get(algo), linestyle=ls)
        ax.set_yscale('log')
        _set_cache_size_xaxis(ax, cache_sizes)
        ax.set_xlabel('Cache Size')
        ax.set_title(cap_metric(metric))
    axes[0].set_ylabel('Mean Distance Calculations per Query')
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                    borderaxespad=0, fontsize=8)
    fig.suptitle(f'{title_prefix}Mean Distance Calculations vs Cache Size', y=1.02)
    plt.tight_layout()
    save_fig(fig, outdir, 'distance_calcs_vs_cache_size.png')

    # --- coverage plots ---
    if datasets_base_dir is not None:
        cov_df = compute_set2_coverage_df(sub, datasets_base_dir, gt_filename_fn)
        if not cov_df.empty:
            plot_set2_coverage_and_hit_rate(sub, cov_df, outdir)
            plot_set2_coverage_estimated_vs_actual(cov_df, outdir)

    # --- heatmap: algo x cache size, one per metric ---
    for metric in metrics:
        m_sub = sub[sub['metric'] == metric]
        algos_m = ordered_algos(m_sub['algorithm'].unique())
        mat = np.full((len(algos_m), len(cache_sizes)), np.nan)
        for i, algo in enumerate(algos_m):
            ys, _ = _get_agg(m_sub[m_sub['algorithm'] == algo],
                             'hit_rate', cache_sizes, 'num_cache_queries')
            for j, y in enumerate(ys):
                mat[i, j] = y

        fig_w = max(7, len(cache_sizes) * 1.5)
        fig_h = max(4, len(algos_m) * 0.9)
        fig, ax_grid = _left_cbar_fig(1, 1, 'YlOrRd', 0, 1, 'Hit Rate', fig_w, fig_h)
        ax = ax_grid[0][0]
        sns.heatmap(mat, annot=True, fmt='.1%', cmap='YlOrRd', vmin=0, vmax=1,
                    xticklabels=[int(c) for c in cache_sizes],
                    yticklabels=lbls(algos_m), ax=ax, cbar=False)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xlabel('Cache Size')
        ax.set_title(
            f'{title_prefix}Hit Rate: Algorithm vs Cache Size ({cap_metric(metric)})',
            pad=12,
        )
        save_fig(fig, outdir, f'hit_rate_heatmap_{metric}.png')

#-------------------------------------------------------------------------------
# ESCI-only: Set 1 baseline (c=1024, K=99/100, N=20, 3 seeds)
#-------------------------------------------------------------------------------

def plot_set1_baseline(df: pd.DataFrame, outdir: Path):
    """Grouped bar: hit rate and distance calcs at baseline config, +/-1 sigma across 3 seeds."""
    sub = df[
        (df['set_id'] == 'set2') &
        (df['num_cache_queries'] == 1024) &
        (~df['algorithm'].isin(NO_UNION_ALGOS))
    ]
    if sub.empty:
        print("  No Set 1 (baseline) data, skipping.")
        return

    algos = ordered_algos(sub['algorithm'].unique())
    metrics = ordered_metrics(sub['metric'].unique())
    pal = algo_palette(algos)

    for value_col, ylabel, yscale, fname in [
        ('hit_rate', 'Hit Rate', 'linear', 'baseline_hit_rate.png'),
        ('avg_distance_calcs', 'Mean Distance Calculations per Query', 'log', 'baseline_distance_calcs.png'),
    ]:
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics) + 2, 5),
                                 sharey=True)
        if len(metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            m_sub = sub[sub['metric'] == metric]
            per_seed = m_sub.groupby(['algorithm', 'config_key'])[value_col].mean().reset_index()
            stats = per_seed.groupby('algorithm')[value_col].agg(['mean', 'std']).reindex(algos)
            for i, algo in enumerate(algos):
                if algo not in stats.index or pd.isna(stats.loc[algo, 'mean']):
                    continue
                mean = stats.loc[algo, 'mean']
                ax.bar(i, mean, 0.6, color=pal[algo])
            ax.set_xticks(range(len(algos)))
            ax.set_xticklabels(lbls(algos), rotation=30, ha='right')
            ax.set_yscale(yscale)
            if yscale == 'linear':
                pct_fmt(ax)
                ax.set_ylim(0, 1.05)
            ax.set_title(cap_metric(metric))

        axes[0].set_ylabel(ylabel)
        fig.suptitle(
            f'Set 1 Baseline - {ylabel}',
            y=1.02,
        )
        plt.tight_layout()
        save_fig(fig, outdir, fname)

#-------------------------------------------------------------------------------
# ESCI-only: Set 3, K/N ratios (num_cache=1024, num_test=256, 2 seeds)
#-------------------------------------------------------------------------------

def plot_set3_kn(df: pd.DataFrame, outdir: Path):
    """KxN hit-rate heatmap (Combined only) and K/N ratio matrix (all algorithms)."""
    sub = df[df['set_id'] == 'set3']
    if sub.empty:
        print("  No Set 3 data, skipping.")
        return

    metrics = ordered_metrics(sub['metric'].unique())
    non_union_algos = ordered_algos(sub['algorithm'].unique(), include_no_union=False)

    def _fmt_ratio(r):
        return str(int(r)) if r == int(r) else f'{r:.2f}'.rstrip('0').rstrip('.')

    for metric in metrics:
        m_sub = sub[sub['metric'] == metric]

        # --- K x N heatmap (Combined only, seeds averaged) ---
        comb = m_sub[m_sub['algorithm'] == 'combined']
        # average per config_key first (seed isolation), then across seeds
        agg = (comb.groupby(['cache_K', 'test_N', 'config_key'])['hit_rate']
                   .mean()
                   .reset_index()
                   .groupby(['cache_K', 'test_N'])['hit_rate']
                   .mean()
                   .reset_index())
        K_vals = sorted(agg['cache_K'].dropna().unique().astype(int))
        N_vals = sorted(agg['test_N'].dropna().unique().astype(int))

        if K_vals and N_vals:
            mat = np.full((len(K_vals), len(N_vals)), np.nan)
            for _, row in agg.iterrows():
                i = K_vals.index(int(row['cache_K']))
                j = N_vals.index(int(row['test_N']))
                mat[i, j] = row['hit_rate']

            fig_w = max(7, len(N_vals) * 1.5)
            fig_h = max(5, len(K_vals) * 0.9)
            fig, ax_grid = _left_cbar_fig(1, 1, 'YlOrRd', 0, 1, 'Hit Rate', fig_w, fig_h)
            ax = ax_grid[0][0]
            sns.heatmap(mat, annot=True, fmt='.1%', cmap='YlOrRd', vmin=0, vmax=1,
                        xticklabels=N_vals, yticklabels=K_vals,
                        ax=ax, cbar=False)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            ax.set_xlabel('N')
            ax.set_ylabel('K')
            ax.set_title(
                f'Set 3 - Hit Rate vs K and N (Combined, {cap_metric(metric)})',
                pad=12,
            )
            save_fig(fig, outdir, f'kn_heatmap_combined_{metric}.png')

        # --- K/N ratio matrix: ratio (rows) x algorithm (cols), seeds averaged ---
        ratio_sub = m_sub[m_sub['algorithm'].isin(set(non_union_algos))].copy()
        ratio_sub['kn_ratio'] = (ratio_sub['cache_K'] / ratio_sub['test_N']).round(4)
        agg2 = (ratio_sub.groupby(['kn_ratio', 'algorithm', 'config_key'])['hit_rate']
                         .mean()
                         .reset_index()
                         .groupby(['kn_ratio', 'algorithm'])['hit_rate']
                         .mean()
                         .reset_index())
        kn_ratios = sorted(agg2['kn_ratio'].unique())

        if kn_ratios and non_union_algos:
            mat2 = np.full((len(kn_ratios), len(non_union_algos)), np.nan)
            for i, ratio in enumerate(kn_ratios):
                for j, algo in enumerate(non_union_algos):
                    row = agg2[(agg2['kn_ratio'] == ratio) & (agg2['algorithm'] == algo)]
                    if len(row):
                        mat2[i, j] = row.iloc[0]['hit_rate']

            fig_w = max(7, len(non_union_algos) * 2.0)
            fig_h = max(5, len(kn_ratios) * 0.6)
            fig, ax_grid = _left_cbar_fig(1, 1, 'YlOrRd', 0, 1, 'Hit Rate', fig_w, fig_h)
            ax = ax_grid[0][0]
            sns.heatmap(mat2, annot=True, fmt='.1%', cmap='YlOrRd', vmin=0, vmax=1,
                        xticklabels=lbls(non_union_algos),
                        yticklabels=[_fmt_ratio(r) for r in kn_ratios],
                        ax=ax, cbar=False)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            ax.set_xlabel('Algorithm')
            ax.set_ylabel('K/N Ratio')
            ax.set_title(
                f'Set 3 - Hit Rate by K/N Ratio and Algorithm ({cap_metric(metric)})',
                pad=12,
            )
            save_fig(fig, outdir, f'kn_ratio_matrix_{metric}.png')

#-------------------------------------------------------------------------------
# DATASET_CONFIG -- what differs between SIFT and ESCI analysis

DATASET_CONFIG = {
    'sift': dict(
        label='SIFT',
        parse_name_fn=None,
        coverage_gt_filename=lambda K: f'cache_gt_population_K{K}.npz',
        datasets_base_dir=Path('datasets/sift'),
        group_col='num_cache_queries',
        lemma_breakdown_xtick_fmt=lambda v: str(int(v)),
        avg_time_xtick_fmt=lambda v: str(int(v)),
        cache_scaling_subdir='cache_scaling',
        cache_scaling_title_prefix='',
    ),
    'esci': dict(
        label='ESCI',
        parse_name_fn=_parse_esci_name,
        coverage_gt_filename=lambda K: f'cache_gt_K{K}.npz',
        datasets_base_dir=Path('datasets/esci'),
        group_col='set_id',
        lemma_breakdown_xtick_fmt=lambda s: s.upper(),
        avg_time_xtick_fmt=lambda s: s.replace('set', 'Set '),
        cache_scaling_subdir='set2',
        cache_scaling_title_prefix='Set 2 - ',
    ),
}

#-------------------------------------------------------------------------------
# analysis entry point

def analyze(dataset: str, raw_dir: str, output_dir: str):
    """Load simulation results, export CSVs, and produce all plots."""
    cfg = DATASET_CONFIG[dataset]
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dirs = {'global': output_path / 'global'}
    if dataset == 'esci':
        dirs['set1'] = output_path / 'set1'
        dirs['set2'] = output_path / 'set2'
        dirs['set3'] = output_path / 'set3'
    else:
        dirs['cache_scaling'] = output_path / cfg['cache_scaling_subdir']
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"{cfg['label']} Simulation Analysis")
    print(f"{'='*80}")
    print(f"Raw dir:    {raw_path}")
    print(f"Output dir: {output_path}")
    print()

    summaries = load_summaries(str(raw_path), kind='benchmark summaries')
    if not summaries:
        print("No summaries found. Exiting.")
        return

    df = build_summary_df(summaries, parse_name_fn=cfg['parse_name_fn'])
    per_query_df = load_per_query_df(str(raw_path), summaries, parse_name_fn=cfg['parse_name_fn'])

    # data exports at output root
    df.to_csv(output_path / "summary.csv", index=False)
    print(f"Exported summary.csv ({len(df)} rows)")

    if not per_query_df.empty:
        per_query_df.to_csv(output_path / "per_query_all.csv", index=False)
        print(f"Exported per_query_all.csv ({len(per_query_df)} rows)")

    with open(output_path / "summaries.json", 'w') as f:
        json.dump(summaries, f, indent=2)
    print(f"Exported summaries.json ({len(summaries)} benchmarks)")

    # for ESCI's global plots, surface the baseline (set2 @ cache=1024) as a
    # standalone "set1" group so it's distinguished from the rest of set2
    df_global = df.copy()
    if dataset == 'esci':
        df_global.loc[
            (df_global['set_id'] == 'set2') & (df_global['num_cache_queries'] == 1024),
            'set_id'
        ] = 'set1'

    print("\n[Global]")
    plot_global_hit_rate_heatmap(df_global, dirs['global'])
    plot_lemma_breakdown_global(df_global, dirs['global'], cfg['group_col'], cfg['lemma_breakdown_xtick_fmt'])
    plot_avg_time(df_global, dirs['global'], cfg['group_col'], cfg['avg_time_xtick_fmt'])

    if dataset == 'esci':
        print("\n[Set 1: Baseline (c=1024)]")
        plot_set1_baseline(df, dirs['set1'])

        print("\n[Set 2: Cache size scaling]")
        set2_sub = df[df['set_id'] == 'set2']
        plot_cache_size_scaling(
            set2_sub, dirs['set2'], cfg['datasets_base_dir'],
            cfg['coverage_gt_filename'], title_prefix=cfg['cache_scaling_title_prefix'],
        )

        print("\n[Set 3: K/N ratios]")
        plot_set3_kn(df, dirs['set3'])
    else:
        print("\n[Cache scaling]")
        plot_cache_size_scaling(
            df, dirs['cache_scaling'], cfg['datasets_base_dir'],
            cfg['coverage_gt_filename'], title_prefix=cfg['cache_scaling_title_prefix'],
        )

    print(f"\n{'='*80}")
    print(f"Analysis complete. Results saved to {output_path}")
    print(f"{'='*80}\n")

#-------------------------------------------------------------------------------

def _default_raw_dir(dataset: str) -> str:
    """Auto-detect the raw dir: picks the latest timestamped subdir if present."""
    base = Path(__file__).parent.parent / dataset / 'raw'
    stamped = sorted(
        d for d in base.iterdir()
        if d.is_dir() and re.fullmatch(r'\d{8}_\d{6}', d.name)
    ) if base.exists() else []
    if stamped:
        chosen = stamped[-1]
        print(f"Auto-selected run: {chosen}")
        return str(chosen)
    return str(base)


def main():
    parser = argparse.ArgumentParser(description="Analyze SIFT/ESCI simulation results")
    parser.add_argument(
        '--dataset', type=str, choices=['sift', 'esci'], required=True,
        help='dataset to analyze',
    )
    parser.add_argument(
        "--raw_dir", type=str,
        default=None,
        help="Directory containing raw simulation results (default: latest timestamped run)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: simulations/<dataset>/processed/<timestamp>)",
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir or _default_raw_dir(args.dataset)

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = str(Path(__file__).parent.parent / args.dataset / "processed" / ts)
    else:
        output_dir = args.output_dir

    analyze(args.dataset, raw_dir, output_dir)

if __name__ == "__main__":
    main()

#-------------------------------------------------------------------------------
