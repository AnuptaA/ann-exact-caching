#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Examine raw candidate properties across K values at fixed alphas."""

import re
import sys
import csv
from datetime import datetime
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from datasets.dataloaders import load_sift_benchmark
from utils.base.population import propose_candidates

#-------------------------------------------------------------------------------

sns.set_style('whitegrid')

K_VALUES = [1, 5, 10, 20]
ALPHA_VALUES = [1.05, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5]
METRIC = 'angular'
BENCHMARK_BASE_DIR = Path('datasets/sift')
OUTPUT_BASE = Path('simulations/sift/processed/population/k-microbenchmarks')

MICROBENCHMARKS = [
    'sift_b50k_c256k100_t512n20_s42',
    'sift_b50k_c1024k100_t512n20_s42',
    'sift_b50k_c2048k100_t512n20_s42',
]

#-------------------------------------------------------------------------------

def _train_size_int(name: str) -> int:
    m = re.search(r'_c(\d+)k', name)
    return int(m.group(1)) if m else 0

def _load_gt_for_k(
    bdir: Path,
    k: int,
    n_cache: int,
    base_vectors: np.ndarray,
) -> tuple[list, list, np.ndarray, np.ndarray]:
    """Load per-K GT file. Returns (tv_all, ti_all, hg_all, rQ_all)."""
    data = np.load(bdir / f'cache_gt_population_K{k}.npz')
    tv_all, ti_all, hg_all, rQ_all = [], [], [], []
    for i in range(n_cache):
        indices = data[f'indices_{i}'].tolist()
        tv_all.append([base_vectors[j] for j in indices])
        ti_all.append(indices)
        hg_all.append(float(data[f'half_gap_{i}']))
        rQ_all.append(float(data[f'r_Q_{i}']))
    return tv_all, ti_all, np.array(hg_all, dtype=np.float32), np.array(rQ_all, dtype=np.float32)

def _collect_candidates_for_k(
    benchmark_name: str,
    benchmark_base_dir: Path = BENCHMARK_BASE_DIR,
) -> dict[int, list[dict]]:
    """Return {target_k: [row_dicts]} for all K_VALUES with on-disk GT files."""
    bdir = benchmark_base_dir / benchmark_name
    base, cache_q, _, config = load_sift_benchmark(str(bdir))
    n_cache = len(cache_q)

    result: dict[int, list[dict]] = {}
    for target_k in K_VALUES:
        gt_path = bdir / f'cache_gt_population_K{target_k}.npz'
        if not gt_path.exists():
            print(f"  SKIP k={target_k}: missing {gt_path.name}")
            continue

        tv_all, ti_all, hg_all, rQ_all = _load_gt_for_k(bdir, target_k, n_cache, base)

        print(f"  K={target_k}: proposing candidates ({n_cache} queries x {len(ALPHA_VALUES)} alphas)...")
        candidates = propose_candidates(
            cache_q, rQ_all, tv_all, ti_all, hg_all,
            cache_q, rQ_all,
            cache_q, rQ_all,
            base, ALPHA_VALUES, METRIC,
            store_rc=False,
        )
        print(f"    {len(candidates)} candidates")

        rows = []
        for cand in candidates:
            rc_size = cand.rc_size
            n_covered = len(cand.covered_train_indices)
            rows.append({
                'k': target_k,
                'alpha': float(cand.alpha),
                'r_Q': float(cand.r_Q),
                'r_c': float(cand.r_c),
                'R_c_size': rc_size,
                'n_covered': n_covered,
                'coverage_rate': n_covered / max(n_cache, 1),
            })
        result[target_k] = rows
    return result

def _save_csv(rows_by_k: dict[int, list[dict]], path: Path) -> None:
    all_rows = [r for rows in rows_by_k.values() for r in rows]
    if not all_rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

def _cross_k_plots(rows_by_k: dict[int, list[dict]], label: str, outdir: Path) -> None:
    """Line plots: alpha on x-axis, one line per K value, showing median."""
    outdir.mkdir(parents=True, exist_ok=True)
    palette = sns.color_palette('tab10', len(K_VALUES))
    k_color = {k: c for k, c in zip(sorted(rows_by_k), palette)}
    ts = _train_size_int(label)
    alpha_labels = [str(a) for a in ALPHA_VALUES]

    fields = [
        ('R_c_size', 'k_vs_rc_size.png', '|R_c| (median)', 'R_c Size vs Alpha'),
        ('coverage_rate', 'k_vs_coverage.png', 'Coverage Rate (median)', 'Coverage Rate vs Alpha'),
    ]
    x_pos = list(range(1, len(ALPHA_VALUES) + 1))
    for field, fname, ylabel, title in fields:
        fig, ax = plt.subplots(figsize=(9, 5))
        for target_k, rows in sorted(rows_by_k.items()):
            ys = []
            for a in ALPHA_VALUES:
                vals = [r[field] for r in rows if r['alpha'] == a]
                ys.append(np.median(vals) if vals else None)
            ys_nan = [y if y is not None else np.nan for y in ys]
            ax.plot(x_pos, ys_nan, marker='o', label=f'K={target_k}',
                    color=k_color[target_k])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(alpha_labels)
        ax.set_xlabel('alpha')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Train={ts} - {title} by K')
        ax.legend()
        plt.tight_layout()
        fig.savefig(outdir / fname, dpi=120)
        plt.close(fig)

def _aggregate_plots(
    all_data: dict[str, dict[int, list[dict]]],
    outdir: Path,
) -> None:
    """Cross-benchmark aggregate: one line per (benchmark, K) combination."""
    outdir.mkdir(parents=True, exist_ok=True)
    labels = list(all_data.keys())
    palette = sns.color_palette('tab10', len(labels) * len(K_VALUES))
    alpha_labels = [str(a) for a in ALPHA_VALUES]

    fields = [
        ('R_c_size', 'rc_size.png', '|R_c| (median)', 'R_c Size vs Alpha'),
        ('coverage_rate', 'coverage.png', 'Coverage Rate (median)', 'Coverage Rate vs Alpha'),
    ]
    x_pos = list(range(1, len(ALPHA_VALUES) + 1))
    for field, fname, ylabel, title in fields:
        fig, ax = plt.subplots(figsize=(11, 6))
        ci = 0
        for label in labels:
            ts = _train_size_int(label)
            bm_data = all_data[label]
            for target_k, rows in sorted(bm_data.items()):
                ys = []
                for a in ALPHA_VALUES:
                    vals = [r[field] for r in rows if r['alpha'] == a]
                    ys.append(np.median(vals) if vals else None)
                ys_nan = [y if y is not None else np.nan for y in ys]
                ax.plot(x_pos, ys_nan, marker='o',
                        label=f'Train={ts} K={target_k}',
                        color=palette[ci], linewidth=1.5)
                ci += 1
        ax.set_xticks(x_pos)
        ax.set_xticklabels(alpha_labels)
        ax.set_xlabel('alpha')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Aggregate - {title}')
        ax.legend(fontsize=7)
        plt.tight_layout()
        fig.savefig(outdir / fname, dpi=120)
        plt.close(fig)

    # E2: aggregate scatter
    fig, ax = plt.subplots(figsize=(11, 6))
    ci = 0
    for label in labels:
        ts = _train_size_int(label)
        for target_k, rows in sorted(all_data[label].items()):
            xs = [r['R_c_size'] for r in rows]
            ys = [r['coverage_rate'] for r in rows]
            ax.scatter(xs, ys, color=palette[ci], alpha=0.3, s=10,
                       label=f'Train={ts} K={target_k}')
            ci += 1
    ax.set_xlabel('|R_c|')
    ax.set_ylabel('Coverage Rate')
    ax.set_title('Aggregate - R_c Size vs Coverage Rate')
    ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(outdir / 'rc_vs_coverage_scatter.png', dpi=120)
    plt.close(fig)

#-------------------------------------------------------------------------------

def main() -> None:
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outbase = OUTPUT_BASE / run_ts
    print(f"Run timestamp: {run_ts}")

    all_data: dict[str, dict[int, list[dict]]] = {}

    for benchmark_name in MICROBENCHMARKS:
        print(f"\n[{benchmark_name}]")
        rows_by_k = _collect_candidates_for_k(benchmark_name, BENCHMARK_BASE_DIR)
        if not rows_by_k:
            continue
        label = benchmark_name
        outdir = outbase / label
        _save_csv(rows_by_k, outdir / 'candidates.csv')
        _cross_k_plots(rows_by_k, label, outdir)
        all_data[label] = rows_by_k
        print(f"  Written to {outdir}")

    if len(all_data) > 1:
        print("\n[aggregate]")
        _aggregate_plots(all_data, outbase / 'aggregate')
        print(f"  Written to {outbase / 'aggregate'}")

if __name__ == '__main__':
    main()
