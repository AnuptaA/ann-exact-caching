#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Sphere-aware candidate proposal: r_Q = D(Q, q_n) + D(q_n, E_{K_cache}).

Sweeps K_Cache x K_Test x n to measure hit rate and budget (mean |R_c|).
"""

import re
import sys
import csv
import argparse
from datetime import datetime
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from datasets.dataloaders import load_sift_benchmark, load_esci_benchmark

#-------------------------------------------------------------------------------

sns.set_style('whitegrid')

K_CACHE_VALUES = [1, 5, 10, 20]
K_TEST_VALUES = [0, 1, 5, 10]
N_VALUES = [1, 5, 10]

DATASET_CONFIG = {
    'sift': dict(
        label='SIFT',
        load_benchmark_fn=load_sift_benchmark,
        benchmark_base_dir=Path('datasets/sift'),
        output_base=Path('simulations/sift/processed/population/containment-microbenchmarks'),
        research_benchmarks=[
            'sift_b50k_c256k100_t512n20_s42',
            'sift_b50k_c512k100_t512n20_s42',
            'sift_b50k_c1024k100_t512n20_s42',
            'sift_b50k_c2048k100_t512n20_s42',
        ],
    ),
    'esci': dict(
        label='ESCI',
        load_benchmark_fn=load_esci_benchmark,
        benchmark_base_dir=Path('datasets/esci'),
        output_base=Path('simulations/esci/processed/population/containment-microbenchmarks'),
        research_benchmarks=[
            'esci_c256k100_t512n20_s42',
            'esci_c512k100_t512n20_s42',
            'esci_c1024k100_t512n20_s42',
            'esci_c2048k100_t512n20_s42',
        ],
    ),
}

#-------------------------------------------------------------------------------

def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return m / norms

def _parse_benchmark_info(label: str) -> dict:
    cache = re.search(r'_c(\d+)k', label)
    test = re.search(r'_t(\d+)', label)
    return {
        'cache_size': cache.group(1) if cache else '?',
        'test_size': test.group(1) if test else '?',
    }

def _load_test_radii(benchmark_dir: Path, K: int, n_test: int) -> np.ndarray:
    data = np.load(benchmark_dir / f'test_gt_population_K{K}.npz')
    return np.array([float(data[f'r_Q_{i}']) for i in range(n_test)], dtype=np.float32)

#-------------------------------------------------------------------------------

def _compute_ball_costs(cache_q: np.ndarray) -> tuple:
    """Pairwise angular distances between cache queries, sorted per row.

    Returns (sorted_dists, sorted_indices).
    sorted_dists[c, n-1] = D(q_c, n-th nearest cache neighbor).
    sorted_indices[c, n-1] = index of that neighbor.
    Self-distance is excluded (set to inf before sorting).
    """
    cache_norm = _normalize(cache_q)
    sims = np.clip(cache_norm @ cache_norm.T, -1.0, 1.0)
    dists = np.arccos(sims) / np.pi
    np.fill_diagonal(dists, np.inf)
    order = np.argsort(dists, axis=1)
    sorted_dists = np.take_along_axis(dists, order, axis=1)
    return sorted_dists, order

def _compute_rows(
    test_dists: np.ndarray,
    ball_costs: np.ndarray,
    nn_indices: np.ndarray,
    cb_dists: np.ndarray,
    r_t_by_k: dict,
    label: str,
) -> list:
    """Compute hit_rate, mean_r_q, mean_n_base_inside for all valid (n, k_cache, k_test).

    r_q[c] = D(Q_c, q_n) + D(q_n, E_{k_cache})
    D(q_n, E_{k_cache}) is q_n's distance to its k_cache-th nearest base vector
    (into the base dataset, not to other train/cache queries).
    """
    # d_base_sorted[c, k-1] = D(q_c, E_k): distance from cache query c to its
    # k-th nearest base vector. NOTE: base dataset distances, not train-to-train.
    d_base_sorted = np.sort(cb_dists, axis=1)   # (n_cache, n_base)

    rows = []
    for k_cache in K_CACHE_VALUES:
        d_base_kc = d_base_sorted[:, k_cache - 1]  # (n_cache,)
        for n in N_VALUES:
            nn_idx = nn_indices[:, n - 1]                       # (n_cache,)
            r_q = ball_costs[:, n - 1] + d_base_kc[nn_idx]     # (n_cache,)
            n_base_inside = (cb_dists <= r_q[:, np.newaxis]).sum(axis=1)

            for k_test, r_t in sorted(r_t_by_k.items()):
                if k_test >= k_cache:
                    continue
                # Non-strict <=, matching the real CIG algorithm's boundary
                # condition (fixed from strict < in an earlier round; this
                # standalone script had drifted out of sync with it).
                hit = (test_dists + r_t[:, np.newaxis] <= r_q[np.newaxis, :]).any(axis=1)
                rows.append({
                    'benchmark': label,
                    'n': n,
                    'k_cache': k_cache,
                    'k_test': k_test,
                    'hit_rate': float(hit.mean()),
                    'mean_r_q': float(r_q.mean()),
                    'mean_n_base_inside': float(n_base_inside.mean()),
                })
    return rows

#-------------------------------------------------------------------------------

def _save_csv(rows: list, path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

#-------------------------------------------------------------------------------

def _aggregate_plots(all_data: dict, outdir: Path, label: str) -> None:
    """Three figures: hit_rate_vs_n, hit_rate_vs_k_cache, budget_vs_k_cache."""
    outdir.mkdir(parents=True, exist_ok=True)

    all_rows = [r for rows in all_data.values() for r in rows]
    _save_csv(all_rows, outdir / 'containment.csv')

    labels = list(all_data.keys())
    n_bm = len(labels)
    n_kc = len(K_CACHE_VALUES)
    n_nv = len(N_VALUES)

    kt_palette = sns.color_palette('tab10', len(K_TEST_VALUES))
    kt_color = {k: c for k, c in zip(K_TEST_VALUES, kt_palette)}
    n_palette = sns.color_palette('tab10', n_nv)
    n_color = {n: c for n, c in zip(N_VALUES, n_palette)}

    # ── hit_rate_vs_n: rows=K_Cache, cols=benchmarks, lines=K_Test ───────────
    fig, axes = plt.subplots(n_kc, n_bm, figsize=(5 * n_bm, 3.5 * n_kc),
                             sharey=True, sharex=True)
    axes = np.array(axes).reshape(n_kc, n_bm)
    fig.suptitle(f"{label} Hit Rate vs. $n$ (sphere-aware proposal)", fontsize=13)

    for i_kc, k_cache in enumerate(K_CACHE_VALUES):
        valid_kts = [kt for kt in K_TEST_VALUES if kt < k_cache]
        for j, lbl in enumerate(labels):
            ax = axes[i_kc, j]
            info = _parse_benchmark_info(lbl)
            rows = all_data[lbl]
            kc_rows = [r for r in rows if r['k_cache'] == k_cache]
            for k_test in valid_kts:
                x = [r['n'] for r in kc_rows if r['k_test'] == k_test]
                y = [r['hit_rate'] for r in kc_rows if r['k_test'] == k_test]
                show_label = (j == n_bm - 1 and i_kc == 0)
                ax.plot(x, y, marker='o', color=kt_color[k_test],
                        label=f'$K_{{test}}={k_test}$' if show_label else None)
            if i_kc == 0:
                ax.set_title(f"Cache Size = {info['cache_size']}")
            if j == 0:
                ax.set_ylabel(f"$K_{{cache}}={k_cache}$\nHit Rate")
            ax.set_xticks(N_VALUES)
            ax.set_ylim(0, 1.05)
            if i_kc == n_kc - 1:
                ax.set_xlabel("$n$")

    axes[0, -1].legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    fig.savefig(outdir / 'hit_rate_vs_n.png', dpi=120)
    plt.close(fig)

    # ── hit_rate_vs_k_cache: rows=n, cols=benchmarks, lines=K_Test ──────────
    fig, axes = plt.subplots(n_nv, n_bm, figsize=(5 * n_bm, 3.5 * n_nv),
                             sharey=True, sharex=True)
    axes = np.array(axes).reshape(n_nv, n_bm)
    fig.suptitle(rf"{label} Hit Rate vs. $K_{{cache}}$ (sphere-aware proposal)", fontsize=13)

    for i_n, n in enumerate(N_VALUES):
        for j, lbl in enumerate(labels):
            ax = axes[i_n, j]
            info = _parse_benchmark_info(lbl)
            rows = all_data[lbl]
            n_rows = [r for r in rows if r['n'] == n]
            for k_test in K_TEST_VALUES:
                x = [r['k_cache'] for r in n_rows if r['k_test'] == k_test]
                y = [r['hit_rate'] for r in n_rows if r['k_test'] == k_test]
                if not x:
                    continue
                show_label = (j == n_bm - 1 and i_n == 0)
                ax.plot(x, y, marker='o', color=kt_color[k_test],
                        label=f'$K_{{test}}={k_test}$' if show_label else None)
            if i_n == 0:
                ax.set_title(f"Cache Size = {info['cache_size']}")
            if j == 0:
                ax.set_ylabel(f"$n={n}$\nHit Rate")
            ax.set_xticks(K_CACHE_VALUES)
            ax.set_ylim(0, 1.05)
            if i_n == n_nv - 1:
                ax.set_xlabel(r"$K_{cache}$")

    axes[0, -1].legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    fig.savefig(outdir / 'hit_rate_vs_k_cache.png', dpi=120)
    plt.close(fig)

    # ── budget_vs_k_cache: mean_n_base_inside vs K_Cache, lines=n ────────────
    fig, axes = plt.subplots(1, n_bm, figsize=(5 * n_bm, 4), sharey=True)
    if n_bm == 1:
        axes = [axes]
    fig.suptitle(rf"{label} Mean $|R_c|$ vs. $K_{{cache}}$ (sphere-aware proposal)", fontsize=13)

    for j, (lbl, ax) in enumerate(zip(labels, axes)):
        info = _parse_benchmark_info(lbl)
        rows = all_data[lbl]
        for n in N_VALUES:
            # mean_n_base_inside doesn't depend on k_test; pick k_test=0 rows
            kc_vals, budget_vals = [], []
            for k_cache in K_CACHE_VALUES:
                subset = [
                    r for r in rows
                    if r['n'] == n and r['k_cache'] == k_cache and r['k_test'] == 0
                ]
                if subset:
                    kc_vals.append(k_cache)
                    budget_vals.append(subset[0]['mean_n_base_inside'])
            ax.plot(kc_vals, budget_vals, marker='o', color=n_color[n],
                    label=f'$n={n}$' if j == n_bm - 1 else None)
        ax.set_title(f"Cache Size = {info['cache_size']}")
        ax.set_xlabel(r"$K_{cache}$")
        ax.set_xticks(K_CACHE_VALUES)
        ax.set_ylim(bottom=0)
        if j == 0:
            ax.set_ylabel(r"Mean $|R_c|$")

    axes[-1].legend()
    plt.tight_layout()
    fig.savefig(outdir / 'budget_vs_k_cache.png', dpi=120)
    plt.close(fig)

#-------------------------------------------------------------------------------

def run_benchmark(
    benchmark_name: str,
    load_benchmark_fn,
    benchmark_base_dir: Path,
    debug_data: bool = False,
) -> tuple:
    """Compute sphere-aware containment rows for one benchmark.

    Returns (rows, ball_costs, nn_indices). rows is [] if required GT files absent.
    If debug_data=True, returns a 4-tuple where the fourth element is a dict with
    keys test_dists, cb_dists, ball_costs, nn_indices, r_t_by_k.
    """
    bdir = benchmark_base_dir / benchmark_name
    gt_path = bdir / 'test_gt_population_K1.npz'
    if not gt_path.exists():
        print(f"  SKIP {benchmark_name}: missing {gt_path.name}")
        return ([], None, None, None) if debug_data else ([], None, None)

    base, cache_q, test_q, _ = load_benchmark_fn(str(bdir))
    n_test = len(test_q)

    r_t_by_k = {0: np.zeros(n_test, dtype=np.float32)}
    for k in [1, 5, 10]:
        k_path = bdir / f'test_gt_population_K{k}.npz'
        if k_path.exists():
            r_t_by_k[k] = _load_test_radii(bdir, k, n_test)
        else:
            print(f"  SKIP test K={k}: missing {k_path.name}")

    test_norm = _normalize(test_q)
    cache_norm = _normalize(cache_q)
    test_dists = np.arccos(np.clip(test_norm @ cache_norm.T, -1.0, 1.0)) / np.pi

    base_norm = _normalize(base)
    cb_dists = np.arccos(np.clip(cache_norm @ base_norm.T, -1.0, 1.0)) / np.pi

    ball_costs, nn_indices = _compute_ball_costs(cache_q)
    rows = _compute_rows(test_dists, ball_costs, nn_indices, cb_dists, r_t_by_k, benchmark_name)

    if debug_data:
        return rows, ball_costs, nn_indices, {
            'test_dists': test_dists,
            'cb_dists': cb_dists,
            'ball_costs': ball_costs,
            'nn_indices': nn_indices,
            'r_t_by_k': r_t_by_k,
            'cache_norm': cache_norm,
        }
    return rows, ball_costs, nn_indices

#-------------------------------------------------------------------------------

def run_debug(
    ts: str,
    load_benchmark_fn,
    benchmark_base_dir: Path,
    research_benchmarks: list,
    label: str,
) -> tuple:
    """Run all benchmarks for this dataset and write per-candidate debug output under ts."""
    from simulations.debug import containment_debug

    dataset_dir = Path('simulations/debug') / ts / 'containment' / label.lower()

    all_data: dict = {}
    all_ball_costs: dict = {}
    all_nn_indices: dict = {}
    all_debug_dicts: dict = {}

    for bm in research_benchmarks:
        print(f"\n  [{label}] {bm}")
        rows, ball_costs, nn_indices, dbg = run_benchmark(
            bm, load_benchmark_fn, benchmark_base_dir, debug_data=True,
        )
        if not rows:
            continue
        all_data[bm] = rows
        all_ball_costs[bm] = ball_costs
        all_nn_indices[bm] = nn_indices
        all_debug_dicts[bm] = dbg
        sanity = containment_debug.sanity_check(dbg['cache_norm'], dbg['test_dists'], bm)
        containment_debug.write_section_outputs(
            sanity_lines=sanity,
            ball_costs=dbg['ball_costs'],
            nn_indices=dbg['nn_indices'],
            cb_dists=dbg['cb_dists'],
            r_t_by_k=dbg['r_t_by_k'],
            test_dists=dbg['test_dists'],
            benchmark_dir=dataset_dir / bm,
            label=bm,
        )

    if all_data:
        containment_debug.aggregate_plots(
            all_data, all_debug_dicts, label, dataset_dir,
        )
        print(f"\n  Debug output: {dataset_dir}")

    return all_data, all_ball_costs, all_nn_indices, all_debug_dicts

#-------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run containment microbenchmark (sphere-aware proposal)"
    )
    parser.add_argument('--dataset', choices=['sift', 'esci'], required=True)
    parser.add_argument('--benchmark', type=str, default=None,
                        help='Single benchmark name (default: all)')
    parser.add_argument('--timestamp', type=str, default=None,
                        help='Timestamp string for output dir (default: auto-generated)')
    parser.add_argument('--debug', action='store_true',
                        help='Write per-candidate debug output to simulations/debug/ '
                             'instead of the aggregate hit-rate/budget plots')
    args = parser.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    run_ts = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Run timestamp: {run_ts}")

    if args.debug:
        run_debug(run_ts, cfg['load_benchmark_fn'], cfg['benchmark_base_dir'],
                  cfg['research_benchmarks'], cfg['label'])
        return

    outbase = cfg['output_base'] / run_ts
    benchmarks = [args.benchmark] if args.benchmark else cfg['research_benchmarks']
    all_data: dict = {}

    for benchmark_name in benchmarks:
        print(f"\n[{benchmark_name}]")
        rows, _, _ = run_benchmark(benchmark_name, cfg['load_benchmark_fn'], cfg['benchmark_base_dir'])
        if rows:
            all_data[benchmark_name] = rows

    if all_data:
        _aggregate_plots(all_data, outbase, cfg['label'])
        print(f"\nOutput: {outbase}")

#-------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
