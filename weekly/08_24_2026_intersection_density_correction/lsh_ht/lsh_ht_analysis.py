#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""E5 (selection-procedure validation) + H1-H6 hypothesis checks for
the Horvitz-Thompson LSH grid."""

import sys
import json
import math
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'common'))

import numpy as np

from lsh_ht_grid_run_experiment import (
    OUT_DIR, M_VALUES, X_VALUES, L_VALUES, E3_DATASETS, K, N, _direct_radii,
)
from lsh_horvitz_thompson import select_hyperparameters, p_single

#-------------------------------------------------------------------------------


def load_e3(dataset_key: str, m: int, x: int, L: int) -> dict:
    path = OUT_DIR / 'E3' / dataset_key / f'm{m}_x{x}_L{L}.npz'
    if not path.exists():
        return None
    return dict(np.load(path))


def all_e3_configs(dataset_key: str) -> list:
    configs = []
    for m in M_VALUES:
        for x in X_VALUES:
            if x >= m:
                continue
            for L in L_VALUES:
                if load_e3(dataset_key, m, x, L) is not None:
                    configs.append((m, x, L))
    return configs

#-------------------------------------------------------------------------------


def h1_unbiasedness() -> None:
    print("=== H1: unbiasedness ===")
    for dataset_key in E3_DATASETS:
        ratios = []
        for m, x, L in all_e3_configs(dataset_key):
            r = load_e3(dataset_key, m, x, L)
            valid = r['true_counts'] > 0
            if valid.any():
                ratios.append(np.nanmean(r['ratio'][valid]))
        ratios = np.array(ratios)
        print(f"  {dataset_key}: mean ratio across all configs = {ratios.mean():.4f} "
              f"(std={ratios.std():.4f}, n_configs={len(ratios)})")
    overall = []
    for dataset_key in E3_DATASETS:
        for m, x, L in all_e3_configs(dataset_key):
            r = load_e3(dataset_key, m, x, L)
            valid = r['true_counts'] > 0
            if valid.any():
                overall.append(np.nanmean(r['ratio'][valid]))
    overall = np.array(overall)
    verdict = 'CONFIRMED' if abs(overall.mean() - 1.0) < 0.10 else 'DEVIATION - check theta-vs-r and dedup'
    print(f"  Overall mean ratio across ALL {len(overall)} configs/datasets: {overall.mean():.4f} -> {verdict}")


def h2_fixed_table_bias() -> None:
    print("\n=== H2: fixed-table bias shrinks ~1/sqrt(L) ===")
    stds = {}
    fracs = {}
    for L in L_VALUES:
        path = OUT_DIR / 'E2' / f'L{L}.npz'
        if not path.exists():
            continue
        data = np.load(path)
        stds[L] = float(data['mean_ratios'].std())
        if 'mean_retrieved_fracs' in data:
            fracs[L] = float(data['mean_retrieved_fracs'].mean())
        frac_str = f", mean retrieved fraction={fracs[L]:.4f}" if L in fracs else ""
        print(f"  L={L}: sigma across 30 independent table-set draws = {stds[L]:.4f}{frac_str}")

    if fracs and max(fracs.values()) > 0.3:
        print(f"  CAVEAT: retrieved fraction climbs to {max(fracs.values()):.1%} at the highest L tested - "
              f"sigma collapsing toward 0 is largely the estimator approaching a full scan, not a free "
              f"variance-reduction benefit. Read jointly with cost, per external review.")

    Ls = sorted(stds.keys())
    if len(Ls) >= 2:
        log_L = np.log(Ls)
        log_std = np.log([stds[L] for L in Ls])
        slope = np.polyfit(log_L, log_std, 1)[0]
        print(f"  Fitted slope of log(sigma) vs log(L): {slope:.3f} (expect ~-0.5 for 1/sqrt(L))")
        verdict = 'CONFIRMED' if -0.8 < slope < -0.2 else 'NOT a clean 1/sqrt(L) fit'
        print(f"  -> {verdict}")


def h3_variance_formula() -> None:
    """Corrected per external review: the formula predicts a standard
    deviation. Comparing it against mean(|relative error|) (a mean
    absolute deviation, computed per earlier draft of this check) mixes
    units - those differ by a roughly constant factor for well-behaved
    noise, but comparing against percentiles of that mixed quantity
    (as the earlier draft implicitly did by pooling across configs)
    mostly measures the width of the error distribution, not the
    formula's accuracy. Redone here as std(ratio) vs predicted std,
    matching units directly."""
    print("\n=== H3 (corrected): variance formula vs std(N_hat/true_count), matching units ===")
    factors = []
    for dataset_key in E3_DATASETS:
        for m, x, L in all_e3_configs(dataset_key):
            r = load_e3(dataset_key, m, x, L)
            valid = r['true_counts'] > 0
            if valid.sum() < 5:
                continue
            observed_std = float(np.nanstd(r['ratio'][valid]))
            if observed_std == 0 or np.isnan(observed_std):
                continue
            retrieved_mean = r['retrieved_counts'].mean()
            p_union_est = retrieved_mean / N if N > 0 else np.nan
            if p_union_est <= 0:
                continue
            predicted_std = math.sqrt(max((1 - p_union_est) / (p_union_est * K), 0))
            if predicted_std > 0:
                factors.append(predicted_std / observed_std)
    factors = np.array(factors)
    factors = factors[np.isfinite(factors)]
    print(f"  Conservatism factor (predicted/observed) across {len(factors)} configs: "
          f"mean={factors.mean():.2f}, median={np.median(factors):.2f}, "
          f"p10={np.percentile(factors,10):.2f}, p90={np.percentile(factors,90):.2f}")
    spread = np.percentile(factors, 90) / max(np.percentile(factors, 10), 1e-6)
    verdict = 'CONFIRMED, roughly constant factor' if spread < 5 else 'FACTOR VARIES WIDELY - formula not usable as design rule'
    print(f"  p90/p10 spread = {spread:.2f} -> {verdict}")


def h4_scaling() -> None:
    print("\n=== H4: sublinear retrieval scaling ===")
    n_values = [4000, 8000, 16000, 32000, 50000]
    for label in ['fixedL', 'targeterr']:
        counts = []
        for n_pts in n_values:
            path = OUT_DIR / 'E4' / f'{label}_N{n_pts}.npz'
            if not path.exists():
                continue
            data = np.load(path)
            counts.append(float(data['retrieved_counts'].mean()))
        if len(counts) >= 2:
            log_n = np.log(n_values[:len(counts)])
            log_c = np.log(counts)
            alpha = np.polyfit(log_n, log_c, 1)[0]
            print(f"  {label}: retrieved counts = {[f'{c:.0f}' for c in counts]}")
            print(f"  {label}: fitted exponent alpha (retrieved ~ N^alpha) = {alpha:.3f}")
            if label == 'targeterr':
                Ls = []
                for n_pts in n_values:
                    path = OUT_DIR / 'E4' / f'{label}_N{n_pts}.npz'
                    if path.exists():
                        Ls.append(int(np.load(path)['L']))
                print(f"  {label}: L required to hold error <=20% as N grows = {Ls}")


def h5_real_vs_isotropic() -> None:
    print("\n=== H5: real data outperforms isotropic synthetic ===")
    for real_key, iso_key in [('sift', 'isotropic_d128'), ('esci', 'isotropic_d384')]:
        real_radii_summary = []
        for label, key in [('real', real_key), ('isotropic', iso_key)]:
            best_frac_at_20pct = None
            for m, x, L in all_e3_configs(key):
                r = load_e3(key, m, x, L)
                valid = r['true_counts'] > 0
                if not valid.any():
                    continue
                err = np.nanmean(r['rel_err'][valid])
                frac = r['retrieved_counts'].mean() / N
                if err <= 0.20:
                    if best_frac_at_20pct is None or frac < best_frac_at_20pct:
                        best_frac_at_20pct = frac
            print(f"  {label} ({key}): lowest retrieval fraction achieving <=20% error = "
                  f"{best_frac_at_20pct if best_frac_at_20pct is not None else 'none achieved'}")


def baseline_comparison() -> None:
    """Headline result per external review: does the LSH structure earn
    its existence over trivial uniform sampling? Lowest retrieval
    fraction at each error threshold, LSH vs uniform, per dataset."""
    print("\n=== Baseline comparison: LSH-HT vs uniform sampling ===")
    for dataset_key in E3_DATASETS:
        lsh_best = {}
        for thresh in [0.10, 0.15, 0.20]:
            best = None
            for m, x, L in all_e3_configs(dataset_key):
                r = load_e3(dataset_key, m, x, L)
                valid = r['true_counts'] > 0
                if not valid.any():
                    continue
                err = np.nanmean(r['rel_err'][valid])
                frac = r['retrieved_counts'].mean() / N
                if err <= thresh and (best is None or frac < best):
                    best = frac
            lsh_best[thresh] = best

        path = OUT_DIR / 'uniform_baseline' / f'{dataset_key}.npz'
        uniform_best = {}
        if path.exists():
            data = np.load(path)
            for thresh in [0.10, 0.15, 0.20]:
                qualifying = data['fracs'][data['mean_rel_err'] <= thresh]
                uniform_best[thresh] = float(qualifying.min()) if len(qualifying) else None

        print(f"  {dataset_key}:")
        for thresh in [0.10, 0.15, 0.20]:
            lsh_f = lsh_best.get(thresh)
            uni_f = uniform_best.get(thresh)
            if lsh_f is not None and uni_f is not None:
                ratio = uni_f / lsh_f
                print(f"    <=  {thresh:.0%} error: LSH-HT frac={lsh_f:.4f}, uniform frac={uni_f:.4f} "
                      f"-> LSH needs {ratio:.1f}x less data")
            else:
                print(f"    <=  {thresh:.0%} error: LSH-HT frac={lsh_f}, uniform frac={uni_f} "
                      f"(one or both not achieved in tested range)")


def h6_selection_procedure(dataset_radii: dict, dataset_loaders: dict) -> None:
    """The derived L rarely lands exactly on the measured grid's L
    values, so the derived config's own (frac, error) is actually run
    (cheap - a few seconds), not looked up - comparing against
    arbitrary existing grid points would not be a fair Pareto check."""
    print("\n=== H6: derived hyperparameters vs hand-picked grid (E5) ===")
    from lsh_horvitz_thompson import build_tables, query_retrieval, estimate_count_batch

    for dataset_key, r in dataset_radii.items():
        print(f"\n  {dataset_key} (mean cap radius={r:.4f} rad):")
        frontier = []
        for m, x, L in all_e3_configs(dataset_key):
            res = load_e3(dataset_key, m, x, L)
            valid = res['true_counts'] > 0
            if not valid.any():
                continue
            err = np.nanmean(res['rel_err'][valid])
            frac = res['retrieved_counts'].mean() / N
            frontier.append((frac, err, m, x, L))

        data = dataset_loaders[dataset_key]()
        radii = _direct_radii(data['points'], data['queries'], K)
        true_counts = np.array([
            int((np.arccos(np.clip(data['points'] @ data['queries'][i], -1.0, 1.0)) <= radii[i] + 1e-9).sum())
            for i in range(len(data['queries']))
        ])
        valid_tc = true_counts > 0

        for eps in [0.10, 0.15, 0.20]:
            for L_max in [16, 64]:
                derived = select_hyperparameters(K=K, epsilon=eps, r=r, m_candidates=M_VALUES,
                                                  x_candidates=X_VALUES, L_max=L_max)
                if derived is None:
                    print(f"    eps={eps} L_max={L_max}: no valid (m,x,L) found in candidate grid")
                    continue
                rng = np.random.default_rng(999)
                tables = build_tables(data['points'], derived['m'], derived['L'], rng)
                retrieved = query_retrieval(tables, data['queries'], derived['m'], derived['x'])
                n_hat = estimate_count_batch(data['queries'], retrieved, data['points'], radii,
                                              derived['m'], derived['x'], derived['L'])
                actual_frac = float(np.mean([len(rr) for rr in retrieved])) / N
                actual_err = float(np.nanmean(np.abs(n_hat[valid_tc] - true_counts[valid_tc]) / true_counts[valid_tc]))

                dominators = [f for f in frontier if f[0] <= actual_frac and f[1] <= actual_err
                              and (f[0] < actual_frac or f[1] < actual_err)]
                verdict = 'NOT dominated by any hand-picked grid point' if not dominators else \
                          f'DOMINATED by {len(dominators)} grid point(s)'
                print(f"    eps={eps} L_max={L_max}: derived m={derived['m']} x={derived['x']} "
                      f"L={derived['L']} | predicted_frac={derived['retrieved_frac']:.4f}, "
                      f"ACTUAL frac={actual_frac:.4f} err={actual_err:.4f} (target eps={eps}) -> {verdict}")

#-------------------------------------------------------------------------------


def main() -> None:
    print(f"Loaded grid from {OUT_DIR}\n")
    h1_unbiasedness()
    h2_fixed_table_bias()
    h3_variance_formula()
    h4_scaling()
    h5_real_vs_isotropic()
    baseline_comparison()

    dataset_radii = {}
    for dataset_key, loader in E3_DATASETS.items():
        data = loader()
        radii = _direct_radii(data['points'], data['queries'], K)
        dataset_radii[dataset_key] = float(radii.mean())
    h6_selection_procedure(dataset_radii, E3_DATASETS)

    print("\n=== E6 memory ===")
    mem = json.load(open(OUT_DIR / 'E6' / 'memory.json'))
    for row in mem:
        ratio = row['full_direction_bytes'] / row['pooled_direction_bytes']
        print(f"  {row['dataset']} m={row['m']} L={row['L']}: point_id={row['point_id_bytes_measured']:,}B, "
              f"full_dir={row['full_direction_bytes']:,}B, pooled_dir={row['pooled_direction_bytes']:,}B, "
              f"ratio={ratio:.1f}x")

    print("\nDone.")


if __name__ == '__main__':
    main()
