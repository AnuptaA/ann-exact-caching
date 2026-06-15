#!/usr/bin/env python3
"""Examine raw candidate properties before selection across alpha values."""

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

ALPHA_VALUES = [1.05, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5]
METRIC = 'angular'
LARGE_BUDGET = 200_000
BENCHMARK_BASE_DIR = Path('datasets/sift')
OUTPUT_BASE = Path('simulations/sift/processed/population/alpha-microbenchmarks')

MICROBENCHMARKS = [
    'sift_b50k_c256k100_t512n20_s42',
    'sift_b50k_c1024k100_t512n20_s42',
    'sift_b50k_c2048k100_t512n20_s42',
]

#-------------------------------------------------------------------------------

def _cache_k_from_name(benchmark_name: str) -> int:
    """Extract cache K from benchmark name (e.g. sift_b50k_c256k100_... -> 100)."""
    m = re.search(r'c\d+k(\d+)', benchmark_name)
    if m is None:
        print(f"Warning: could not parse K from benchmark name '{benchmark_name}', defaulting to 100")
    return int(m.group(1)) if m else 100

def _output_label(benchmark_name: str, top_k: int) -> str:
    """Derive output label from benchmark name, normalising the embedded k-value."""
    return re.sub(r'k\d+', f'k{top_k}', benchmark_name)

def _load_cache_gt_arrays(benchmark_dir: Path, K: int, n_cache: int,
                           base_vectors: np.ndarray) -> tuple:
    data = np.load(benchmark_dir / f'cache_gt_population_K{K}.npz')
    tv_all, ti_all, hg_all = [], [], []
    for i in range(n_cache):
        indices = data[f'indices_{i}'].tolist()
        tv_all.append([base_vectors[idx] for idx in indices])
        ti_all.append(indices)
        hg_all.append(float(data[f'half_gap_{i}']))
    return tv_all, ti_all, np.array(hg_all, dtype=np.float32)

def _collect_candidates(
    benchmark_name: str,
    benchmark_base_dir: Path = BENCHMARK_BASE_DIR,
) -> tuple[list[dict], int]:
    """Return (rows, n_queries) for all candidates of this benchmark."""
    bdir = benchmark_base_dir / benchmark_name
    if not (bdir / 'cache_r_Q.npy').exists():
        print(f"  SKIP {benchmark_name}: missing cache_r_Q.npy")
        return [], 0

    base, cache_q, _, config = load_sift_benchmark(str(bdir))
    K = config['cache_K']
    n_cache = len(cache_q)

    r_Q_all = np.load(bdir / 'cache_r_Q.npy')
    tv_all, ti_all, hg_all = _load_cache_gt_arrays(bdir, K, n_cache, base)

    print(f"  Proposing candidates ({n_cache} queries x {len(ALPHA_VALUES)} alphas)...")
    candidates = propose_candidates(
        cache_q, r_Q_all, tv_all, ti_all, hg_all,
        cache_q, r_Q_all,
        cache_q, r_Q_all,
        base, ALPHA_VALUES, METRIC,
        store_rc=False,
    )
    print(f"  {len(candidates)} candidates")

    rows = []
    for cand in candidates:
        rc_size = cand.rc_size
        n_covered = len(cand.covered_train_indices)
        rows.append({
            'alpha': float(cand.alpha),
            'r_Q': float(cand.r_Q),
            'r_c': float(cand.r_c),
            'R_c_size': rc_size,
            'n_covered': n_covered,
            'coverage_rate': n_covered / max(n_cache, 1),
        })
    return rows, n_cache

def _save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def _per_benchmark_plots(rows: list[dict], label: str, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    alpha_groups = {a: [r for r in rows if r['alpha'] == a] for a in ALPHA_VALUES}
    positions = list(range(1, len(ALPHA_VALUES) + 1))

    fields = [
        ('R_c_size', 'rc_size_vs_alpha.png', '|R_c|', 'R_c Size vs Alpha'),
        ('n_covered', 'coverage_vs_alpha.png', 'queries covered', 'Coverage vs Alpha'),
    ]
    for field, fname, ylabel, title in fields:
        fig, ax = plt.subplots(figsize=(9, 5))
        data = [[r[field] for r in alpha_groups[a]] for a in ALPHA_VALUES]
        non_empty = [(pos, d) for pos, d in zip(positions, data) if d]
        if non_empty:
            pos_list, data_list = zip(*non_empty)
            ax.boxplot(list(data_list), positions=list(pos_list))
        ax.set_xticks(positions)
        ax.set_xticklabels([str(a) for a in ALPHA_VALUES])
        ax.set_xlabel('alpha')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{label} - {title}')
        plt.tight_layout()
        fig.savefig(outdir / fname, dpi=120)
        plt.close(fig)

    cmap = plt.cm.tab10
    alpha_color = {a: cmap(i / max(len(ALPHA_VALUES) - 1, 1)) for i, a in enumerate(ALPHA_VALUES)}
    fig, ax = plt.subplots(figsize=(9, 5))
    for a in ALPHA_VALUES:
        xs = [r['R_c_size'] for r in alpha_groups[a]]
        ys = [r['n_covered'] for r in alpha_groups[a]]
        ax.scatter(xs, ys, color=alpha_color[a], alpha=0.5, s=15, label=str(a))
    ax.set_xlabel('|R_c|')
    ax.set_ylabel('queries covered')
    ax.set_title(f'{label} - R_c Size vs Coverage')
    ax.legend(title='alpha', fontsize=9)
    plt.tight_layout()
    fig.savefig(outdir / 'rc_vs_coverage_scatter.png', dpi=120)
    plt.close(fig)

def _aggregate_plots(all_rows: dict[str, list[dict]], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    benchmarks = list(all_rows.keys())
    palette = sns.color_palette('tab10', len(benchmarks))
    bm_color = {b: c for b, c in zip(benchmarks, palette)}
    alpha_labels = [str(a) for a in ALPHA_VALUES]

    fields = [
        ('R_c_size', 'rc_size_vs_alpha.png', '|R_c| (median)', 'R_c Size vs Alpha'),
        ('coverage_rate', 'coverage_vs_alpha.png', 'Coverage Rate (median)', 'Coverage Rate vs Alpha'),
    ]
    x_pos = list(range(1, len(ALPHA_VALUES) + 1))
    for field, fname, ylabel, title in fields:
        fig, ax = plt.subplots(figsize=(9, 5))
        for bm in benchmarks:
            rows = all_rows[bm]
            ys = []
            for a in ALPHA_VALUES:
                vals = [r[field] for r in rows if r['alpha'] == a]
                ys.append(np.median(vals) if vals else None)
            ys_nan = [y if y is not None else np.nan for y in ys]
            ax.plot(x_pos, ys_nan, marker='o', label=bm, color=bm_color[bm])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(alpha_labels)
        ax.set_xlabel('alpha')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Aggregate - {title}')
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(outdir / fname, dpi=120)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for bm in benchmarks:
        xs = [r['R_c_size'] for r in all_rows[bm]]
        ys = [r['coverage_rate'] for r in all_rows[bm]]
        ax.scatter(xs, ys, color=bm_color[bm], alpha=0.3, s=10, label=bm)
    ax.set_xlabel('|R_c|')
    ax.set_ylabel('Coverage Rate')
    ax.set_title('Aggregate - R_c Size vs Coverage Rate')
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(outdir / 'rc_vs_coverage_scatter.png', dpi=120)
    plt.close(fig)

#-------------------------------------------------------------------------------

def main() -> None:
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outbase = OUTPUT_BASE / run_ts
    print(f"Run timestamp: {run_ts}")

    all_rows: dict[str, list[dict]] = {}

    for benchmark_name in MICROBENCHMARKS:
        print(f"\n[{benchmark_name}]")
        rows, _ = _collect_candidates(benchmark_name, BENCHMARK_BASE_DIR)
        if not rows:
            continue
        label = _output_label(benchmark_name, _cache_k_from_name(benchmark_name))
        outdir = outbase / label
        _save_csv(rows, outdir / 'candidates.csv')
        _per_benchmark_plots(rows, label, outdir)
        all_rows[label] = rows
        print(f"  Written to {outdir}")

    if len(all_rows) > 1:
        print("\n[aggregate]")
        _aggregate_plots(all_rows, outbase / 'aggregate')
        print(f"  Written to {outbase / 'aggregate'}")

if __name__ == '__main__':
    main()
