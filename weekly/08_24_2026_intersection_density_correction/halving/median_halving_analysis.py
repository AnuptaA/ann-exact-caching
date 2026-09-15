#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Analysis of the median-halving tree grid: H1-H5 verdicts, depth
table, query cost, count accuracy (cluster/background split for
clustered datasets), memory, degenerate-leaf accounting, and a final
feasibility verdict."""

import sys
import math
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'common'))

import numpy as np

from median_halving_grid_run_experiment import (
    N, C_VALUES, DIMS_SYNTHETIC, W_MULTIPLIERS, OUT_DIR,
)

#-------------------------------------------------------------------------------


def load_result(dataset_key: str, C: int, halving: bool) -> dict:
    variant = 'H' if halving else 'N'
    path = OUT_DIR / dataset_key / f'C{C}_{variant}.npz'
    if not path.exists():
        return None
    data = dict(np.load(path, allow_pickle=True))
    return data


def all_dataset_keys() -> list:
    keys = ['sift', 'esci']
    for d in DIMS_SYNTHETIC:
        keys.append(f'uniform_d{d}')
    for d in DIMS_SYNTHETIC:
        for wi in range(len(W_MULTIPLIERS)):
            keys.append(f'clustered_d{d}_w{wi}')
    return keys


def dataset_dim(dataset_key: str) -> int:
    if dataset_key == 'sift':
        return 128
    if dataset_key == 'esci':
        return 384
    return int(dataset_key.split('_d')[1].split('_')[0])

#-------------------------------------------------------------------------------


def h1_check() -> None:
    print("=== H1: Variant H depth is geometry-independent (ceil(log2(N/C)) +-1) ===")
    mismatches = []
    for dataset_key in all_dataset_keys():
        for C in C_VALUES:
            r = load_result(dataset_key, C, True)
            if r is None:
                continue
            expected = math.ceil(math.log2(N / C))
            actual = int(r['max_depth_reached'])
            if abs(actual - expected) > 1:
                mismatches.append((dataset_key, C, expected, actual))
    if mismatches:
        print(f"  MISMATCHES (|actual-expected|>1): {len(mismatches)}")
        for m in mismatches[:20]:
            print(f"    {m}")
    else:
        print(f"  CONFIRMED: every Variant H config's depth is within +-1 of ceil(log2(N/C)), "
              f"across all {len(all_dataset_keys())} dataset settings x {len(C_VALUES)} C values.")


def h2_check() -> None:
    print("\n=== H2: Variant N depth/resolution is geometry-dependent, generally worse ===")
    rows = []
    for dataset_key in all_dataset_keys():
        for C in C_VALUES:
            rh = load_result(dataset_key, C, True)
            rn = load_result(dataset_key, C, False)
            if rh is None or rn is None:
                continue
            worse_depth = int(rn['max_depth_reached']) > int(rh['max_depth_reached'])
            capped_frac = float(rn['n_capped_points']) / N
            rows.append((dataset_key, C, int(rh['max_depth_reached']), int(rn['max_depth_reached']),
                         worse_depth, capped_frac))

    n_worse = sum(1 for r in rows if r[4])
    print(f"  Variant N depth exceeds Variant H depth in {n_worse}/{len(rows)} configs.")
    thin_capped = [r for r in rows if r[5] > 0.01]
    print(f"  Configs with >1% of points left in depth-capped leaves (Variant N): {len(thin_capped)}")
    for r in sorted(thin_capped, key=lambda x: -x[5])[:15]:
        print(f"    {r[0]} C={r[1]}: depth H={r[2]} N={r[3]}, capped_frac={r[5]:.4f}")
    if not thin_capped:
        print("  No configuration left a meaningful fraction of points unresolved for Variant N.")


def h3_check() -> None:
    print("\n=== H3: Variant H query-time branching is regime-dependent ===")
    for C in C_VALUES:
        rows = []
        for dataset_key in ['sift', 'esci'] + [f'uniform_d{d}' for d in DIMS_SYNTHETIC]:
            r = load_result(dataset_key, C, True)
            if r is None:
                continue
            touches = r['leaf_touch_counts']
            n_leaves = int(r['n_leaves'])
            rows.append((dataset_key, touches.mean(), touches.mean() / n_leaves))
        print(f"  C={C}:")
        for dk, mean_touch, frac in rows:
            print(f"    {dk:>14}: mean leaves touched={mean_touch:.1f} ({frac:.2%} of {'' } leaves)")


def h4_check() -> None:
    print("\n=== H4: count estimator is essentially exact for Variant H ===")
    all_err = []
    for dataset_key in all_dataset_keys():
        for C in C_VALUES:
            r = load_result(dataset_key, C, True)
            if r is None:
                continue
            err = r['est_counts'].astype(np.int64) - r['true_counts'].astype(np.int64)
            all_err.append(err)
    all_err = np.concatenate(all_err)
    print(f"  Across all Variant H configs (n={len(all_err)} query evaluations): "
          f"mean={all_err.mean():.6f}, p10={np.percentile(all_err,10):.4f}, "
          f"p90={np.percentile(all_err,90):.4f}, max_abs={np.abs(all_err).max()}")
    if np.abs(all_err).max() == 0:
        print("  CONFIRMED: exact in every single query evaluation across the full grid.")
    else:
        print("  DEVIATION FOUND - treat as an implementation bug, not acceptable error, per H4.")


def h5_check() -> None:
    print("\n=== H5: memory dominated by direction storage unless mitigated ===")
    rows = []
    for dataset_key in ['sift', 'esci']:
        for C in C_VALUES:
            r = load_result(dataset_key, C, True)
            if r is None:
                continue
            rows.append((dataset_key, C, int(r['mem_full_direction']), int(r['mem_pooled_index'])))
    print("  Measured at N=50,000:")
    for dk, C, full, pooled in rows:
        ratio = full / pooled if pooled else float('nan')
        print(f"    {dk} C={C}: full_direction={full:,} bytes, pooled_index={pooled:,} bytes, "
              f"ratio={ratio:.1f}x")

    print("\n  Extrapolation to N=100,000,000 (labeled estimate, not measured):")
    for dk, C, full, pooled in rows:
        scale = 100_000_000 / N
        d = 128 if dk == 'sift' else 384
        pool_fixed_cost = 256 * d * 8  # P * d * float64, independent of N
        full_scaled = full * scale
        pooled_scaled = (pooled - pool_fixed_cost) * scale + pool_fixed_cost
        print(f"    {dk} C={C}: full_direction ~{full_scaled/1e9:.2f} GB, "
              f"pooled_index ~{pooled_scaled/1e9:.2f} GB (pool itself stays fixed at "
              f"{pool_fixed_cost/1e6:.2f} MB regardless of N)")


def degenerate_leaf_accounting() -> None:
    print("\n=== Degenerate-leaf accounting (duplicate-exact vs depth-capped-unresolved) ===")
    total_unique_H = total_capped_H = total_unique_N = total_capped_N = 0
    for dataset_key in all_dataset_keys():
        for C in C_VALUES:
            rh = load_result(dataset_key, C, True)
            rn = load_result(dataset_key, C, False)
            if rh is not None:
                total_unique_H += int(rh['n_leaf_unique'])
                total_capped_H += int(rh['n_capped_leaves'])
            if rn is not None:
                total_unique_N += int(rn['n_leaf_unique'])
                total_capped_N += int(rn['n_capped_leaves'])
    print(f"  Variant H: {total_unique_H} duplicate-degenerate leaves (exact), "
          f"{total_capped_H} depth-capped leaves (unresolved) across the whole grid.")
    print(f"  Variant N: {total_unique_N} duplicate-degenerate leaves (exact), "
          f"{total_capped_N} depth-capped leaves (unresolved) across the whole grid.")

#-------------------------------------------------------------------------------


def clustered_cluster_vs_background() -> None:
    print("\n=== Synthetic-clustered: cluster-region vs background-region queries (not pooled) ===")
    for d in [10, 128, 384]:
        if d not in DIMS_SYNTHETIC:
            continue
        for wi, wm in enumerate(W_MULTIPLIERS):
            dataset_key = f'clustered_d{d}_w{wi}'
            r = load_result(dataset_key, 20, True)
            if r is None or 'tags' not in r:
                continue
            tags = r['tags'].astype(bool)
            n_cluster_q = int(tags.sum())
            n_bg_q = int((~tags).sum())
            err = r['est_counts'].astype(np.int64) - r['true_counts'].astype(np.int64)
            touches = r['leaf_touch_counts']
            print(f"  d={d}, W_mult={wm}: n_cluster_queries={n_cluster_q}, n_background_queries={n_bg_q}")
            if n_cluster_q > 0:
                print(f"    cluster region: mean touch={touches[tags].mean():.1f}, "
                      f"count err mean={err[tags].mean():.4f}")
            if n_bg_q > 0:
                print(f"    background region: mean touch={touches[~tags].mean():.1f}, "
                      f"count err mean={err[~tags].mean():.4f}")

#-------------------------------------------------------------------------------


def main() -> None:
    print(f"Loaded grid from {OUT_DIR}\n")
    h1_check()
    h2_check()
    h3_check()
    h4_check()
    h5_check()
    degenerate_leaf_accounting()
    clustered_cluster_vs_background()
    print("\nDone.")


if __name__ == '__main__':
    main()
