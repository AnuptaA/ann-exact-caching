#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Characterize all workloads and produce cross-workload comparison plots.
"""

import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

from datasets.dataloaders import (
    load_sift_dataset,
    load_sift_benchmark,
    load_esci_data,
    load_esci_benchmark,
)
from datasets.synthetic.generator import load_synthetic_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

#-------------------------------------------------------------------------------
# Constants
#-------------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent.parent

SYNTHETIC_BASELINE_SEEDS = [42, 43, 44]
SIFT_BASELINE_SEEDS = [42, 43, 44]
ESCI_BASELINE_SEEDS = [42, 43, 44]

SYNTHETIC_BASELINE_TEMPLATE = "set1_baseline_seed{seed}"
SIFT_BASELINE_TEMPLATE = "sift_b50k_c1024k100_t512n20_s{seed}"
ESCI_BASELINE_TEMPLATE = "esci_c1024k99_t512n20_s{seed}"

ESCI_FULL_SAMPLE_SIZE = 10_000
ESCI_FULL_SAMPLE_SEED = 0
NN_SCORE_RANKS = [1, 5, 25, 50, 99]

WORKLOAD_COLORS = {
    'synthetic': 'steelblue',
    'sift': 'darkorange',
    'esci': 'forestgreen',
}

WORKLOAD_LABELS = {
    'synthetic': 'Synthetic',
    'sift': 'SIFT',
    'esci': 'ESCI',
}

POPULATION_LABELS = {
    'sift': 'SIFT',
    'esci': 'ESCI (sub-sampled 10k)',
}

RANK_COLORS = {
    1: 'tab:blue',
    5: 'tab:orange',
    25: 'tab:green',
    50: 'tab:red',
    99: 'tab:purple',
}

PCT_COLORS = {
    'p10': 'tab:cyan',
    'median': 'tab:red',
    'p90': 'tab:purple',
}

BINS_DIST = np.linspace(0, 1, 51)

#-------------------------------------------------------------------------------
# Geometry helpers
#-------------------------------------------------------------------------------

def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms < 1e-10, 1e-10, norms)

def angular_pairwise_nn(vectors: np.ndarray) -> np.ndarray:
    """For each vector, angular distance to its nearest other vector in the same set."""
    vn = _normalize(vectors)
    gram = vn @ vn.T
    np.fill_diagonal(gram, -2.0)
    return np.arccos(np.clip(gram, -1.0, 1.0)).min(axis=1) / np.pi

def angular_pairwise_topk(vectors: np.ndarray, ranks: list) -> dict:
    """
    For each vector, angular distance to its k-th nearest other vector in the set,
    for each k in ranks (1-based).
    """
    vn = _normalize(vectors)
    gram = vn @ vn.T
    np.fill_diagonal(gram, -2.0)
    np.clip(gram, -1.0, 1.0, out=gram)
    np.arccos(gram, out=gram)
    gram /= np.pi
    sorted_dists = np.sort(gram, axis=1)
    del gram
    return {r: sorted_dists[:, r - 1].copy() for r in ranks}

def test_to_cache_nn(test_q: np.ndarray, cache_q: np.ndarray) -> np.ndarray:
    """For each test query, angular distance to its nearest cache query."""
    test_norm = _normalize(test_q)
    cache_norm = _normalize(cache_q)
    sims = test_norm @ cache_norm.T
    return np.arccos(np.clip(sims, -1.0, 1.0)).min(axis=1) / np.pi

def compute_containment_counts(
    test_q: np.ndarray,
    cache_q: np.ndarray,
    rq_values: np.ndarray,
) -> np.ndarray:
    """For each test query, count cache entries whose r_Q radius contains it."""
    test_norm = _normalize(test_q)
    cache_norm = _normalize(cache_q)
    sims = test_norm @ cache_norm.T
    dists = np.arccos(np.clip(sims, -1.0, 1.0)) / np.pi
    return (dists <= rq_values[np.newaxis, :]).sum(axis=1)

#-------------------------------------------------------------------------------
# Ground truth loaders
#-------------------------------------------------------------------------------

def load_gt_distances_at_ranks(
    benchmark_dir: Path,
    dataset: str,
    cache_K: int,
    ranks: list,
) -> dict:
    """
    Load angular distances from precomputed ground truth at specific 1-based ranks.
    Returns {rank: np.ndarray(n_cache,)} or {} if the file is missing.

    SIFT's cache_gt_population_K{K}.npz only stores the K-th (last) distance per
    query (r_Q), not a full per-rank distance array like ESCI's cache_gt_K{K}.npz
    does -- so for SIFT, only rank == cache_K is supported.
    """
    if dataset == 'sift':
        gt_file = benchmark_dir / f"cache_gt_population_K{cache_K}.npz"
        if not gt_file.exists():
            logger.warning("Ground truth not found: %s", gt_file)
            return {}
        if ranks != [cache_K]:
            logger.warning(
                "SIFT population GT only stores the K-th distance; requested "
                "ranks %s unsupported (only [cache_K]=[%d] is available)",
                ranks, cache_K,
            )
            return {}
        data = np.load(gt_file)
        values = []
        i = 0
        while f'r_Q_{i}' in data:
            values.append(float(data[f'r_Q_{i}']))
            i += 1
        return {cache_K: np.array(values)}
    elif dataset == 'esci':
        gt_file = benchmark_dir / f"cache_gt_K{cache_K}.npz"
    else:
        return {}

    if not gt_file.exists():
        logger.warning("Ground truth not found: %s", gt_file)
        return {}

    data = np.load(gt_file)
    result = {r: [] for r in ranks}
    i = 0
    while f'distances_{i}' in data:
        dists = data[f'distances_{i}']
        for r in ranks:
            idx = min(r - 1, len(dists) - 1)
            result[r].append(float(dists[idx]))
        i += 1
    return {r: np.array(v) for r, v in result.items()}

def load_rq_values(benchmark_dir: Path, dataset: str, cache_K: int):
    rd = load_gt_distances_at_ranks(benchmark_dir, dataset, cache_K, [cache_K])
    return rd.get(cache_K)

#-------------------------------------------------------------------------------
# Data loading - aggregated across seeds
#-------------------------------------------------------------------------------

def load_synthetic_set1():
    """Returns (nn_dists, containment_counts) over seeds 42/43/44."""
    nn_parts, count_parts = [], []
    base = ROOT / 'datasets' / 'synthetic' / 'data'

    for seed in SYNTHETIC_BASELINE_SEEDS:
        logger.info("Synthetic Set 1 seed %d", seed)
        corpus, cache_q, test_q, config = load_synthetic_dataset(
            str(base / SYNTHETIC_BASELINE_TEMPLATE.format(seed=seed))
        )
        cache_K = config.get('cache_K', 100)

        nn_parts.append(test_to_cache_nn(test_q, cache_q))

        cache_norm = _normalize(cache_q)
        corpus_norm = _normalize(corpus)
        full_dists = np.arccos(np.clip(cache_norm @ corpus_norm.T, -1.0, 1.0)) / np.pi
        sorted_dists = np.sort(full_dists, axis=1)
        rq = sorted_dists[:, min(cache_K - 1, sorted_dists.shape[1] - 1)]
        count_parts.append(compute_containment_counts(test_q, cache_q, rq))

    return (
        np.concatenate(nn_parts),
        np.concatenate(count_parts),
    )

def load_sift_set1():
    """Returns (nn_dists, containment_counts) over seeds 42/43/44."""
    nn_parts, count_parts = [], []

    for seed in SIFT_BASELINE_SEEDS:
        bm = SIFT_BASELINE_TEMPLATE.format(seed=seed)
        bd = ROOT / 'datasets' / 'sift' / bm
        logger.info("SIFT Set 1: %s", bm)
        _, cache_q, test_q, config = load_sift_benchmark(str(bd))
        cache_K = config.get('cache_K', 100)

        nn_parts.append(test_to_cache_nn(test_q, cache_q))

        rq = load_rq_values(bd, 'sift', cache_K)
        if rq is not None:
            count_parts.append(compute_containment_counts(test_q, cache_q, rq))

    return (
        np.concatenate(nn_parts),
        np.concatenate(count_parts) if count_parts else np.array([]),
    )

def load_esci_set1():
    """Returns (nn_dists, containment_counts) over seeds 42/43/44."""
    nn_parts, count_parts = [], []

    for seed in ESCI_BASELINE_SEEDS:
        bm = ESCI_BASELINE_TEMPLATE.format(seed=seed)
        bd = ROOT / 'datasets' / 'esci' / bm
        logger.info("ESCI Set 1: %s (loading ~141 MB ground truth)", bm)
        _, cache_q, test_q, config = load_esci_benchmark(str(bd))
        cache_K = config.get('cache_K', 99)

        nn_parts.append(test_to_cache_nn(test_q, cache_q))

        rq = load_rq_values(bd, 'esci', cache_K)
        if rq is not None:
            count_parts.append(compute_containment_counts(test_q, cache_q, rq))

    return (
        np.concatenate(nn_parts),
        np.concatenate(count_parts) if count_parts else np.array([]),
    )

SYNTHETIC_SET2_LEVELS = [
    ('Small Perturbation',  'set2_pert_small_seed{seed}'),
    ('Medium Perturbation', 'set2_pert_medium_seed{seed}'),
    ('Large Perturbation',  'set2_pert_large_seed{seed}'),
]

def load_sift_population() -> dict:
    """Intra-set NN distances at multiple ranks within the full 10K SIFT query set."""
    logger.info("Loading SIFT full query population")
    _, _, queries, _ = load_sift_dataset(str(ROOT / 'datasets' / 'sift'))
    logger.info("Computing intra-set NN for %d SIFT queries", len(queries))
    return angular_pairwise_topk(queries, NN_SCORE_RANKS)

def load_esci_population() -> dict:
    """Intra-set NN distances at multiple ranks within a sub-sampled 10K ESCI query set."""
    logger.info("Loading ESCI query population (sub-sampling %d)", ESCI_FULL_SAMPLE_SIZE)
    _, queries, _, _ = load_esci_data(str(ROOT / 'datasets' / 'esci'), load_embeddings=False)
    rng = np.random.default_rng(ESCI_FULL_SAMPLE_SEED)
    idx = rng.choice(len(queries), size=ESCI_FULL_SAMPLE_SIZE, replace=False)
    idx.sort()
    queries = queries[idx]
    logger.info("Computing intra-set NN for %d ESCI queries", len(queries))
    return angular_pairwise_topk(queries, NN_SCORE_RANKS)

def load_synthetic_set2() -> dict:
    """
    Test-to-cache NN angular distances for Synthetic Set 2, aggregated over seeds 42/43/44.
    Returns {'Small Perturbation': array, 'Medium Perturbation': array, 'Large Perturbation': array}.
    """
    base = ROOT / 'datasets' / 'synthetic' / 'data'
    result = {}
    for label, template in SYNTHETIC_SET2_LEVELS:
        parts = []
        for seed in SYNTHETIC_BASELINE_SEEDS:
            _, cache_q, test_q, _ = load_synthetic_dataset(
                str(base / template.format(seed=seed))
            )
            parts.append(test_to_cache_nn(test_q, cache_q))
        result[label] = np.concatenate(parts)
    return result

#-------------------------------------------------------------------------------
# Style helpers
#-------------------------------------------------------------------------------

def _cdf(data: np.ndarray):
    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y

def _add_hist_percentile_lines(ax, data: np.ndarray):
    """
    Draw colored vertical dashed lines at p10, median, p90 of data.
    Returns list of (Line2D, label) for legend construction.
    """
    specs = [
        ('p10', np.percentile(data, 10), 'p10'),
        ('median', np.median(data), 'p50'),
        ('p90', np.percentile(data, 90), 'p90'),
    ]
    handles = []
    for key, val, label in specs:
        line = ax.axvline(val, color=PCT_COLORS[key], linestyle='--', linewidth=1.0, alpha=0.85)
        handles.append((line, label))
    return handles

def _pct_legend(ax, pct_handles, title=None, loc='upper right'):
    ax.legend(
        [h for h, _ in pct_handles],
        [l for _, l in pct_handles],
        fontsize=8, loc=loc, title=title, title_fontsize=8,
    )

#-------------------------------------------------------------------------------
# Plot 1: Baseline query similarity -- CDF
#-------------------------------------------------------------------------------

def plot_baseline_query_similarity_cdf(nn_syn, nn_sift, nn_esci, out_path: Path):
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    fig.suptitle("Baseline - Test-to-Cache Query NN Angular Distance Frequency", fontsize=13)

    for ax, (wl, data) in zip(axes, [
        ('synthetic', nn_syn), ('sift', nn_sift), ('esci', nn_esci)
    ]):
        ax.set_title(WORKLOAD_LABELS[wl], fontsize=10)
        x, y = _cdf(data)
        ax.plot(x, y, color=WORKLOAD_COLORS[wl], linewidth=1.8)
        pct_handles = _add_hist_percentile_lines(ax, data)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Cumulative Probability", fontsize=9)
        ax.grid(True, alpha=0.2)
        _pct_legend(ax, pct_handles, loc='lower right')

    axes[-1].set_xlabel("Angular Distance", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved: %s", out_path)

#-------------------------------------------------------------------------------
# Plot 2: Baseline query similarity -- histogram
#-------------------------------------------------------------------------------

def plot_baseline_query_similarity_hist(nn_syn, nn_sift, nn_esci, out_path: Path):
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    fig.suptitle("Baseline - Test-to-Cache Query NN Angular Distance Frequency", fontsize=13)

    for ax, (wl, data) in zip(axes, [
        ('synthetic', nn_syn), ('sift', nn_sift), ('esci', nn_esci)
    ]):
        ax.set_title(WORKLOAD_LABELS[wl], fontsize=10)
        ax.hist(data, bins=BINS_DIST, color=WORKLOAD_COLORS[wl], alpha=0.7, density=True)
        pct_handles = _add_hist_percentile_lines(ax, data)
        ax.set_xlim(0, 1)
        ax.set_ylabel("Density", fontsize=9)
        ax.grid(True, alpha=0.2, axis='y')
        _pct_legend(ax, pct_handles, loc='upper right')

    axes[-1].set_xlabel("Angular Distance", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved: %s", out_path)

#-------------------------------------------------------------------------------
# Plot 3: Inter-query (population) similarity -- CDF
#-------------------------------------------------------------------------------

def plot_inter_query_similarity_cdf(nn_sift, nn_esci, out_path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    fig.suptitle("Full Query Set - NN Angular Distance Frequency", fontsize=13)

    for ax, (wl, data) in zip(axes, [('sift', nn_sift), ('esci', nn_esci)]):
        ax.set_title(POPULATION_LABELS[wl], fontsize=10)
        x, y = _cdf(data)
        ax.plot(x, y, color=WORKLOAD_COLORS[wl], linewidth=1.8)
        pct_handles = _add_hist_percentile_lines(ax, data)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Cumulative Probability", fontsize=9)
        ax.grid(True, alpha=0.2)
        _pct_legend(ax, pct_handles, loc='lower right')

    axes[-1].set_xlabel("Angular Distance", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved: %s", out_path)

#-------------------------------------------------------------------------------
# Plot 4: Inter-query (population) similarity -- histogram
#-------------------------------------------------------------------------------

def plot_inter_query_similarity_hist(nn_sift, nn_esci, out_path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    fig.suptitle("Full Query Set - NN Angular Distance Frequency", fontsize=13)

    for ax, (wl, data) in zip(axes, [('sift', nn_sift), ('esci', nn_esci)]):
        ax.set_title(POPULATION_LABELS[wl], fontsize=10)
        ax.hist(data, bins=BINS_DIST, color=WORKLOAD_COLORS[wl], alpha=0.7, density=True)
        pct_handles = _add_hist_percentile_lines(ax, data)
        ax.set_xlim(0, 1)
        ax.set_ylabel("Density", fontsize=9)
        ax.grid(True, alpha=0.2, axis='y')
        _pct_legend(ax, pct_handles, loc='upper right')

    axes[-1].set_xlabel("Angular Distance", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved: %s", out_path)

#-------------------------------------------------------------------------------
# Plot 5: Intra-set NN at multiple ranks -- 5x2 grid (SIFT and ESCI)
#-------------------------------------------------------------------------------

def plot_query_population_multirank_grid(pop_sift, pop_esci, out_path: Path):
    n_ranks = len(NN_SCORE_RANKS)
    fig, axes = plt.subplots(
        n_ranks, 2,
        figsize=(10, 3 * n_ranks + 1),
        sharex='row',
    )
    fig.suptitle("Full Query Set - Angular Distances to Top-1/5/25/50/99 NN by Rank", fontsize=13)

    populations = [('sift', pop_sift), ('esci', pop_esci)]
    for col_idx, (wl, pop) in enumerate(populations):
        color = WORKLOAD_COLORS[wl]
        if col_idx == 0:
            axes[0, col_idx].set_title(POPULATION_LABELS[wl], fontsize=11)
        else:
            axes[0, col_idx].set_title(POPULATION_LABELS[wl], fontsize=11)

        for row_idx, rank in enumerate(NN_SCORE_RANKS):
            ax = axes[row_idx, col_idx]
            data = pop.get(rank)
            if data is None or len(data) == 0:
                ax.set_visible(False)
                continue

            ax.hist(data, bins=BINS_DIST, color=color, alpha=0.7, density=True)
            _add_hist_percentile_lines(ax, data)
            ax.set_xlim(0, 1)
            ax.grid(True, alpha=0.2, axis='y')

            if col_idx == 0:
                ax.set_ylabel(f"Rank {rank}\nDensity", fontsize=9)
            if row_idx == n_ranks - 1:
                ax.set_xlabel("Angular Distance", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved: %s", out_path)

#-------------------------------------------------------------------------------
# Plot 6: Intra-set NN at multiple ranks -- 2x1 by-workload overlay
#-------------------------------------------------------------------------------

def plot_query_population_multirank_by_workload(pop_sift, pop_esci, out_path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
    fig.suptitle("Full Query Set - Angular Distance to Top-1/5/25/50/99 NN Consolidated", fontsize=13)

    for ax, (wl, pop) in zip(axes, [('sift', pop_sift), ('esci', pop_esci)]):
        ax.set_title(POPULATION_LABELS[wl], fontsize=10)
        for rank in NN_SCORE_RANKS:
            data = pop.get(rank)
            if data is None or len(data) == 0:
                continue
            ax.hist(
                data, bins=BINS_DIST,
                color=RANK_COLORS[rank], alpha=0.55,
                density=True, label=f"Rank {rank}",
                histtype='step', linewidth=1.4,
            )
        ax.set_xlim(0, 1)
        ax.set_ylabel("Density", fontsize=9)
        ax.grid(True, alpha=0.2, axis='y')
        ax.legend(fontsize=8, loc='upper right')

    axes[-1].set_xlabel("Angular Distance", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved: %s", out_path)

#-------------------------------------------------------------------------------
# Plot 7: Containment count -- histogram
#-------------------------------------------------------------------------------

def plot_containment_hist(counts_syn, counts_sift, counts_esci, out_path: Path):
    all_counts = [c for c in [counts_syn, counts_sift, counts_esci] if len(c) > 0]
    if not all_counts:
        logger.warning("No containment data; skipping %s", out_path)
        return

    global_max = int(max(c.max() for c in all_counts))
    bins = np.arange(0, global_max + 2) - 0.5

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    fig.suptitle("Baseline - Counts of Cache Radii Containing Test Query", fontsize=13)

    for ax, (wl, data) in zip(axes, [
        ('synthetic', counts_syn), ('sift', counts_sift), ('esci', counts_esci)
    ]):
        if len(data) == 0:
            ax.set_visible(False)
            continue
        ax.set_title(WORKLOAD_LABELS[wl], fontsize=10)
        ax.hist(data, bins=bins, color=WORKLOAD_COLORS[wl], alpha=0.7, density=True)
        pct_handles = _add_hist_percentile_lines(ax, data)
        ax.set_ylabel("Density", fontsize=9)
        ax.grid(True, alpha=0.2, axis='y')
        _pct_legend(ax, pct_handles, loc='upper right')

    axes[-1].set_xlabel("Cache Entries Containing Test Query", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved: %s", out_path)

#-------------------------------------------------------------------------------
# Plot 8: Synthetic Set 2 perturbation -- 3x1 stacked histograms
#-------------------------------------------------------------------------------

def plot_synthetic_set2_perturbation(nn_by_level: dict, out_path: Path):
    labels = [label for label, _ in SYNTHETIC_SET2_LEVELS]
    fig, axes = plt.subplots(1, len(labels), figsize=(14, 4), sharey=True)
    fig.suptitle("Perturbation Sensitivity - Test-to-Cache NN Angular Distance by Perturbation Level", fontsize=13)

    color = WORKLOAD_COLORS['synthetic']
    for ax, label in zip(axes, labels):
        data = nn_by_level.get(label)
        ax.set_title(label, fontsize=10)
        if data is None or len(data) == 0:
            ax.set_visible(False)
            continue
        ax.hist(data, bins=BINS_DIST, color=color, alpha=0.7, density=True)
        pct_handles = _add_hist_percentile_lines(ax, data)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Angular Distance", fontsize=9)
        ax.grid(True, alpha=0.2, axis='y')
        _pct_legend(ax, pct_handles, loc='upper right')

    axes[0].set_ylabel("Density", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved: %s", out_path)

#-------------------------------------------------------------------------------
# PDF helper
#-------------------------------------------------------------------------------

def _add_image_page(pdf: PdfPages, img_path: Path):
    img = mpimg.imread(str(img_path))
    h, w = img.shape[:2]
    page_w, page_h = 11.0, 8.5
    scale = min(page_w / (w / 100), page_h / (h / 100))
    fig, ax = plt.subplots(figsize=((w / 100) * scale, (h / 100) * scale))
    ax.imshow(img)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    pdf.savefig(fig, bbox_inches='tight', dpi=150)
    plt.close(fig)

#-------------------------------------------------------------------------------
# Entry point
#-------------------------------------------------------------------------------

def run(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir)

    logger.info("=== Loading Synthetic Set 1 ===")
    nn_syn, counts_syn = load_synthetic_set1()

    logger.info("=== Loading SIFT Set 1 ===")
    nn_sift, counts_sift = load_sift_set1()

    logger.info("=== Loading ESCI Set 1 ===")
    nn_esci, counts_esci = load_esci_set1()

    logger.info("=== Loading SIFT full query population ===")
    pop_sift = load_sift_population()

    logger.info("=== Loading ESCI full query population ===")
    pop_esci = load_esci_population()

    logger.info("=== Loading Synthetic Set 2 ===")
    set2_syn = load_synthetic_set2()

    logger.info("=== Generating plots ===")

    p = [
        out_dir / "baseline_test_to_cache_cdf.png",
        out_dir / "baseline_test_to_cache_hist.png",
        out_dir / "inter_query_nn_cdf.png",
        out_dir / "inter_query_nn_hist.png",
        out_dir / "query_population_multirank_grid.png",
        out_dir / "query_population_multirank_by_workload.png",
        out_dir / "containment_hist.png",
        out_dir / "synthetic_set2_perturbation.png",
    ]

    plot_baseline_query_similarity_cdf(nn_syn, nn_sift, nn_esci, p[0])
    plot_baseline_query_similarity_hist(nn_syn, nn_sift, nn_esci, p[1])
    plot_inter_query_similarity_cdf(pop_sift[1], pop_esci[1], p[2])
    plot_inter_query_similarity_hist(pop_sift[1], pop_esci[1], p[3])
    plot_query_population_multirank_grid(pop_sift, pop_esci, p[4])
    plot_query_population_multirank_by_workload(pop_sift, pop_esci, p[5])
    plot_containment_hist(counts_syn, counts_sift, counts_esci, p[6])
    plot_synthetic_set2_perturbation(set2_syn, p[7])

    pdf_path = out_dir / "characterization.pdf"
    logger.info("Building PDF: %s", pdf_path)
    with PdfPages(str(pdf_path)) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor('#0d1b2a')
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor('#0d1b2a')
        ax.axis('off')
        ax.text(0.5, 0.62, 'Characterization Analysis', transform=ax.transAxes,
                ha='center', va='center', fontsize=32, color='white', fontweight='bold')
        ax.text(0.5, 0.48, datetime.now().strftime('%B %d, %Y'), transform=ax.transAxes,
                ha='center', va='center', fontsize=16, color='#b0c4de')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        for png_path in p:
            if png_path.exists():
                _add_image_page(pdf, png_path)

    size_mb = pdf_path.stat().st_size / 1_000_000
    logger.info("Done: %s  (%.1f MB)", pdf_path, size_mb)
    print(f"\nOutput: {out_dir}\nPDF:    {pdf_path}  ({size_mb:.1f} MB)")

def main():
    parser = argparse.ArgumentParser(
        description="Characterize all workloads and produce cross-workload comparison plots."
    )
    parser.add_argument(
        '--output-dir', default=None,
        help='Output directory (default: simulations/characterization/<timestamp>)',
    )
    args = parser.parse_args()

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = ROOT / 'simulations' / 'characterization' / ts

    run(out_dir)

if __name__ == '__main__':
    main()
