#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Query-Clustering Ratio Benchmark.

For each dataset, computes for every query q:
  r(q) = Euclidean distance to its nearest neighbor in the database
  d(q) = Euclidean distance to its nearest other query in Q (excluding self)
  ratio(q) = d(q) / r(q)

ratio < 1 means queries cluster tighter than the database is sampled — the
regime where the caching correctness guarantees deliver the most benefit.
"""

import sys
import json
import csv
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from datasets.dataloaders import load_sift_dataset, load_esci_data

#-------------------------------------------------------------------------------

SIFT_DIR = Path('datasets/sift')
ESCI_DIR = Path('datasets/esci')
ESCI_SAMPLE_SIZE = 20_000
ESCI_SAMPLE_SEED = 42
OUTPUT_BASE = Path('simulations/characterization')

sns.set_style('whitegrid')
plt.rcParams.update({
    'figure.dpi':      150,
    'font.size':       12,
    'axes.titlesize':  12,
    'axes.labelsize':  12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
})

#-------------------------------------------------------------------------------
# Data loading
#-------------------------------------------------------------------------------

def _load_sift() -> tuple[np.ndarray, np.ndarray]:
    """Return (queries, r_q) for SIFT using precomputed groundtruth."""
    base, _, queries, groundtruth = load_sift_dataset(str(SIFT_DIR))
    q64 = queries.astype(np.float64)
    nn64 = base[groundtruth[:, 0]].astype(np.float64)
    r_q = np.sqrt(np.sum((q64 - nn64) ** 2, axis=1))
    print(f"  {len(queries)} queries, r range [{r_q.min():.4f}, {r_q.max():.4f}]")
    return queries, r_q


def _load_esci(
    n_sample: int = ESCI_SAMPLE_SIZE,
    seed: int = ESCI_SAMPLE_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (queries, r_q) for ESCI (downsampled). Avoids loading 2.8 GB embeddings."""
    _, queries, _, topk_scores = load_esci_data(str(ESCI_DIR), load_embeddings=False)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(queries), size=n_sample, replace=False))
    queries = queries[idx]
    # topk_scores are dot-product similarities for L2-normalized vectors
    # Euclidean: ||a - b||^2 = 2 - 2*(a . b)
    r_q = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * topk_scores[idx, 0].astype(np.float64)))
    print(f"  {len(queries)} queries (sample of {n_sample}), r range [{r_q.min():.4f}, {r_q.max():.4f}]")
    return queries, r_q

#-------------------------------------------------------------------------------
# Exact pairwise nearest-other-query distance
#-------------------------------------------------------------------------------

def _nearest_other_query(Q: np.ndarray, chunk_size: int = 1000) -> np.ndarray:
    """Exact nearest-other-query Euclidean distance for every query, self excluded.

    Uses chunked matrix computation to avoid materializing the full n x n matrix.
    Peak memory: chunk_size * n * 8 bytes (float64).
    """
    Q64 = Q.astype(np.float64)
    n = len(Q64)
    sq_norms = np.sum(Q64 ** 2, axis=1)
    d_min = np.full(n, np.inf, dtype=np.float64)
    n_chunks = (n + chunk_size - 1) // chunk_size

    for ci, i in enumerate(range(0, n, chunk_size)):
        end = min(i + chunk_size, n)
        chunk = Q64[i:end]
        # shape (end-i, n): squared Euclidean distances from chunk to all queries
        dists_sq = sq_norms[i:end, np.newaxis] + sq_norms - 2.0 * (chunk @ Q64.T)
        np.clip(dists_sq, 0.0, None, out=dists_sq)
        for j in range(end - i):
            dists_sq[j, i + j] = np.inf  # exclude self
        d_min[i:end] = np.sqrt(dists_sq.min(axis=1))
        if ci % 5 == 0 or ci == n_chunks - 1:
            print(f"    chunk {ci + 1}/{n_chunks}")

    return d_min

#-------------------------------------------------------------------------------
# Computation helpers
#-------------------------------------------------------------------------------

def _compute_ratio(r_q: np.ndarray, d_q: np.ndarray) -> np.ndarray:
    """ratio = d/r; nan where r == 0 (query is exact duplicate of a base vector)."""
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(r_q > 0, d_q / r_q, np.nan)


def _summary_stats(ratio: np.ndarray) -> dict:
    valid = ratio[np.isfinite(ratio)]
    return {
        'n_queries': int(len(ratio)),
        'n_valid': int(len(valid)),
        'frac_ratio_lt_1': float((valid < 1).mean()),
        'mean': float(valid.mean()),
        'median': float(np.median(valid)),
        'p10': float(np.percentile(valid, 10)),
        'p25': float(np.percentile(valid, 25)),
        'p75': float(np.percentile(valid, 75)),
        'p90': float(np.percentile(valid, 90)),
        'p99': float(np.percentile(valid, 99)),
    }


def _save_csv(r_q: np.ndarray, d_q: np.ndarray, ratio: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['query_id', 'r', 'd', 'ratio'])
        for i, (r, d, rat) in enumerate(zip(r_q, d_q, ratio)):
            writer.writerow([i, float(r), float(d), '' if np.isnan(rat) else float(rat)])
    print(f"  {path.name}")

#-------------------------------------------------------------------------------
# Plots
#-------------------------------------------------------------------------------

def _pct_fmt(ax, axis: str = 'y') -> None:
    fmt = mticker.FuncFormatter(lambda v, _: f'{v:.0%}')
    if axis == 'y':
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)


def _plot_cdf(datasets: dict, outdir: Path) -> None:
    """Overlay empirical CDFs of ratio for each dataset on a log x-axis."""
    palette = sns.color_palette('tab10', len(datasets))
    fig, ax = plt.subplots(figsize=(8, 5))

    for (label, res), color in zip(datasets.items(), palette):
        valid = np.sort(res['ratio'][np.isfinite(res['ratio'])])
        cdf = np.arange(1, len(valid) + 1) / len(valid)
        ax.plot(valid, cdf, color=color, linewidth=1.5, label=label)

        # annotate the CDF value at ratio=1
        frac_lt1 = float((valid < 1).mean())
        ax.annotate(
            f'{frac_lt1:.0%} < 1',
            xy=(1.0, frac_lt1),
            xytext=(-6, 0), textcoords='offset points',
            ha='right', va='center', fontsize=9, color=color,
        )

    ax.axvline(1.0, color='red', linestyle='--', linewidth=1.2, alpha=0.7, label='ratio = 1')
    ax.set_xscale('log')
    ax.set_xlabel(r"$d(q) / r(q)$")
    ax.set_ylabel("Fraction of queries")
    ax.set_ylim(0, 1)
    _pct_fmt(ax, 'y')
    ax.legend()
    plt.tight_layout()
    fig.suptitle("Query Clustering Ratio — Empirical CDF", y=1.02)
    fig.savefig(outdir / 'ratio_cdf.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ratio_cdf.png")


def _plot_geometry_scatter(datasets: dict, outdir: Path) -> None:
    """r(q) vs d(q) scatter on log-log axes; points coloured by ratio < 1 vs >= 1."""
    labels = list(datasets.keys())
    fig, axes = plt.subplots(1, len(labels), figsize=(6 * len(labels), 5), squeeze=False)

    for col, label in enumerate(labels):
        ax = axes[0][col]
        res = datasets[label]
        r_q, d_q, ratio = res['r'], res['d'], res['ratio']

        # restrict to finite, positive values for log scale
        mask = np.isfinite(ratio) & (r_q > 0) & (d_q > 0)
        below = mask & (ratio < 1)
        above = mask & (ratio >= 1)

        frac_below = below.sum() / mask.sum() if mask.sum() > 0 else 0.0
        frac_above = above.sum() / mask.sum() if mask.sum() > 0 else 0.0

        ax.scatter(r_q[below], d_q[below], c='steelblue', alpha=0.25, s=3,
                   label=f'ratio < 1 ({frac_below:.0%})', rasterized=True)
        ax.scatter(r_q[above], d_q[above], c='tomato', alpha=0.25, s=3,
                   label=f'ratio >= 1 ({frac_above:.0%})', rasterized=True)

        # diagonal y = x corresponds to ratio = 1
        r_v = r_q[mask]
        d_v = d_q[mask]
        lo = min(r_v.min(), d_v.min()) * 0.8
        hi = max(r_v.max(), d_v.max()) * 1.25
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=0.8, alpha=0.6, label='ratio = 1')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(r"$r(q)$ — database NN distance")
        if col == 0:
            ax.set_ylabel(r"$d(q)$ — query-set NN distance")
        ax.set_title(label)
        ax.legend(fontsize=8, markerscale=3)

    plt.tight_layout()
    fig.suptitle("Query-Clustering Geometry", y=1.02)
    fig.savefig(outdir / 'geometry_scatter.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  geometry_scatter.png")


def _plot_distance_distributions(datasets: dict, outdir: Path) -> None:
    """Histogram of r(q) and d(q) per dataset, one file each with its own scale.

    If d(q) distribution lies left of r(q), queries cluster more tightly than the
    database is sampled — the caching regime.
    """
    for label, res in datasets.items():
        r_q = res['r'][res['r'] > 0]
        d_q = res['d'][np.isfinite(res['d']) & (res['d'] > 0)]

        all_vals = np.concatenate([r_q, d_q])
        lo = np.percentile(all_vals, 1)
        hi = np.percentile(all_vals, 99)
        bins = np.logspace(np.log10(lo), np.log10(hi), 50)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(r_q, bins=bins, density=True, alpha=0.55, color='steelblue',
                label=r"$r(q)$ — database NN")
        ax.hist(d_q, bins=bins, density=True, alpha=0.55, color='darkorange',
                label=r"$d(q)$ — query-set NN")
        ax.set_xscale('log')
        ax.set_xlabel("Distance")
        ax.set_ylabel("Density")
        ax.set_title(label)
        ax.legend()
        plt.tight_layout()
        fname = f"distance_distributions_{label.lower()}.png"
        fig.savefig(outdir / fname, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  {fname}")


def _plot_r_by_regime(datasets: dict, outdir: Path) -> None:
    """r(q) distribution split by caching regime (ratio < 1 vs >= 1), per dataset."""
    labels = list(datasets.keys())
    fig, axes = plt.subplots(1, len(labels), figsize=(6 * len(labels), 5), squeeze=False)

    for col, label in enumerate(labels):
        ax = axes[0][col]
        res = datasets[label]
        mask = np.isfinite(res['ratio']) & (res['r'] > 0)
        r_all = res['r'][mask]
        below = res['ratio'][mask] < 1
        r_below = r_all[below]
        r_above = r_all[~below]
        frac_below = below.mean()

        lo = np.percentile(r_all, 1)
        hi = np.percentile(r_all, 99)
        bins = np.logspace(np.log10(lo), np.log10(hi), 50)

        ax.hist(r_below, bins=bins, density=True, alpha=0.55, color='steelblue',
                label=f'ratio < 1 ({frac_below:.0%})')
        ax.hist(r_above, bins=bins, density=True, alpha=0.55, color='tomato',
                label=f'ratio >= 1 ({1 - frac_below:.0%})')
        ax.set_xscale('log')
        ax.set_xlabel(r"$r(q)$")
        if col == 0:
            ax.set_ylabel("Density")
        ax.set_title(label)
        ax.legend()

    plt.tight_layout()
    fig.suptitle(r"$r(q)$ Distribution by Caching Regime", y=1.02)
    fig.savefig(outdir / 'r_by_regime.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  r_by_regime.png")


def _plot_ratio_vs_r_density(datasets: dict, outdir: Path) -> None:
    """2D hexbin of (r(q), ratio) in log-log space, per dataset."""
    labels = list(datasets.keys())
    fig, axes = plt.subplots(1, len(labels), figsize=(6 * len(labels), 5), squeeze=False)

    for col, label in enumerate(labels):
        ax = axes[0][col]
        res = datasets[label]
        mask = np.isfinite(res['ratio']) & (res['r'] > 0) & (res['ratio'] > 0)
        r_v = res['r'][mask]
        rat_v = res['ratio'][mask]

        hb = ax.hexbin(r_v, rat_v, xscale='log', yscale='log',
                       gridsize=50, cmap='YlOrRd', mincnt=1)
        fig.colorbar(hb, ax=ax, label='Count')
        ax.axhline(1.0, color='red', linestyle='--', linewidth=1.2, alpha=0.7,
                   label='ratio = 1')
        ax.set_xlabel(r"$r(q)$ — database NN distance")
        if col == 0:
            ax.set_ylabel(r"$d(q)/r(q)$")
        ax.set_title(label)
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.suptitle(r"Query Clustering Ratio vs. DB NN Distance", y=1.02)
    fig.savefig(outdir / 'ratio_vs_r_density.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ratio_vs_r_density.png")

#-------------------------------------------------------------------------------

def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = OUTPUT_BASE / ts
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {outdir}\n")

    datasets = {}

    print("[SIFT]")
    queries_s, r_s = _load_sift()
    print("  Computing d(q)...")
    d_s = _nearest_other_query(queries_s)
    ratio_s = _compute_ratio(r_s, d_s)
    datasets['SIFT'] = {'r': r_s, 'd': d_s, 'ratio': ratio_s}
    _save_csv(r_s, d_s, ratio_s, outdir / 'sift_per_query.csv')

    print("\n[ESCI]")
    queries_e, r_e = _load_esci()
    print("  Computing d(q)...")
    d_e = _nearest_other_query(queries_e)
    ratio_e = _compute_ratio(r_e, d_e)
    datasets['ESCI'] = {'r': r_e, 'd': d_e, 'ratio': ratio_e}
    _save_csv(r_e, d_e, ratio_e, outdir / 'esci_per_query.csv')

    print("\nSummary stats:")
    stats = {}
    for label, res in datasets.items():
        s = _summary_stats(res['ratio'])
        stats[label] = s
        print(f"  {label}: {s['frac_ratio_lt_1']:.1%} have ratio < 1, median = {s['median']:.3f}")
    with open(outdir / 'summary_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print("  summary_stats.json")

    print("\nGenerating plots...")
    _plot_cdf(datasets, outdir)
    _plot_geometry_scatter(datasets, outdir)
    _plot_distance_distributions(datasets, outdir)
    _plot_r_by_regime(datasets, outdir)
    _plot_ratio_vs_r_density(datasets, outdir)

    print(f"\nDone. Results in {outdir}")

#-------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
