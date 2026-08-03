#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Shared debug utilities for containment microbenchmarks (SIFT and ESCI).

Key definitions used throughout this module:

  ball_costs[c, j]: Angular distance from cache query c to its j-th nearest
                    other cache query (sorted ascending, diagonal=inf).
                    This is a distance matrix.

  r_c_n[c]:         ball_costs[c, n-1]
                    The ball radius for candidate c at parameter n.
                    It is the angular distance to the n-th nearest cache
                    neighbor, the ball just large enough to contain
                    n other cache queries.

  n_test_inside[c]: Number of test queries whose centers fall inside the ball
                    around c: count(d(test, cache_c) < r_c_n[c]).

  n_base_inside[c]: Number of base (data) vectors inside the ball around c:
                    count(d(base, cache_c) <= r_c_n[c]).

  r_t[t]:           Distance from test query t to its K-th nearest base vector.
                    Used in the CIG hit condition.

  Hit condition (CIG, K> 0): d(test_t, cache_c) + r_t[t] < r_c_n[c]
  K=0 drops r_t because it is a test of pure containment, not CIG.
"""

import re
from pathlib import Path

import numpy as np

from simulations.population.microbenchmarks.debug_utils import write_txt as _write_txt, write_csv as _write_csv

#-------------------------------------------------------------------------------

def _parse_benchmark_info(label: str) -> dict:
    dataset = label.split('_')[0].upper()
    cache = re.search(r'_c(\d+)k', label)
    test = re.search(r'_t(\d+)', label)
    return {
        'dataset': dataset,
        'cache_size': cache.group(1) if cache else '?',
        'test_size': test.group(1) if test else '?',
    }

#-------------------------------------------------------------------------------
# Compute functions

def sanity_check(cache_norm: np.ndarray, test_dists: np.ndarray, label: str) -> list:
    """Verify basic distance invariants. Returns list of result strings (ASCII only).

    Checks:
      1. Cache queries are unit-normalized (self-similarity == 1.0).
      2. All test-to-cache distances are non-negative.
      3. Minimum test-to-cache distance -- near zero means a near-duplicate
         test/train pair, which would inflate K=0 hit rates.
    """
    lines = []
    lines.append(f"Benchmark: {label}")
    lines.append(f"  cache_norm shape : {cache_norm.shape}")
    lines.append(f"  test_dists shape : {test_dists.shape}")
    lines.append(f"  test_dists range : [{test_dists.min():.8f}, {test_dists.max():.8f}]")

    self_sims = np.diag(cache_norm @ cache_norm.T)
    max_dev = float(np.abs(self_sims - 1.0).max())
    tag = "PASS" if max_dev < 1e-5 else "WARN"
    lines.append(
        f"  {tag}  cache self-similarity: max |sim - 1.0| = {max_dev:.2e}"
        f"  (expected < 1e-5; self-distance = arccos(1.0)/pi = 0.0)"
    )

    neg = int((test_dists < 0).sum())
    if neg == 0:
        lines.append("  PASS  all test-to-cache distances >= 0")
    else:
        lines.append(f"  FAIL  {neg} negative test-to-cache distances detected")

    min_dist = float(test_dists.min())
    tag = "WARN" if min_dist < 1e-4 else "PASS"
    note = "  <-- possible near-duplicate (may inflate K=0 hit rate)" if min_dist < 1e-4 else ""
    lines.append(f"  {tag}  min test-to-cache distance = {min_dist:.8f}{note}")

    return lines

def _scale_summary(
    ball_costs: np.ndarray,
    nn_indices: np.ndarray,
    cb_dists: np.ndarray,
    r_t_by_k: dict,
    k_cache: int,
) -> list:
    """Summary lines for improved procedure at a specific K_cache."""
    imp_n_vals = [1, 5, 10]
    d_base_sorted = np.sort(cb_dists, axis=1)
    d_base_kc = d_base_sorted[:, k_cache - 1]

    lines = [f"K_cache = {k_cache}  (d_base_kc = D(q_c, E_{{K_cache}}))",
             ""]
    lines.append("r_t values by K_test (angular distance to K-th nearest base for test query):")
    for k in sorted(r_t_by_k.keys()):
        r_t = r_t_by_k[k]
        if k == 0:
            lines.append("  K_test=0 : all zeros")
        else:
            lines.append(
                f"  K_test={k} : mean={r_t.mean():.6f}  median={float(np.median(r_t)):.6f}"
                f"  max={r_t.max():.6f}"
            )

    lines.append("")
    lines.append("r_q values by n (= D(Q_c, q_n) + D(q_n, E_{K_cache})):")
    for n_val in imp_n_vals:
        if n_val > ball_costs.shape[1]:
            break
        nn_idx = nn_indices[:, n_val - 1]
        r_q = ball_costs[:, n_val - 1] + d_base_kc[nn_idx]
        lines.append(
            f"  n={n_val}: mean={r_q.mean():.6f}  median={float(np.median(r_q)):.6f}"
            f"  max={r_q.max():.6f}"
        )

    lines.append("")
    lines.append("Ratio r_t_mean / r_q_mean per (K_test, n):")
    for k in sorted(r_t_by_k.keys()):
        if k == 0:
            continue
        r_t_mean = float(r_t_by_k[k].mean())
        for n_val in imp_n_vals:
            if n_val > ball_costs.shape[1]:
                break
            nn_idx = nn_indices[:, n_val - 1]
            r_q = ball_costs[:, n_val - 1] + d_base_kc[nn_idx]
            rq_mean = float(r_q.mean())
            ratio = r_t_mean / rq_mean if rq_mean > 0 else float('inf')
            flag = "  <-- r_t >> r_q" if ratio > 2 else ""
            lines.append(
                f"  K_test={k}, n={n_val}:  ratio={ratio:.4f}"
                f"  (r_t_mean={r_t_mean:.6f}  r_q_mean={rq_mean:.6f}){flag}"
            )
    return lines

def write_section_outputs(
    sanity_lines: list,
    ball_costs: np.ndarray,
    nn_indices: np.ndarray,
    cb_dists: np.ndarray,
    r_t_by_k: dict,
    test_dists: np.ndarray,
    benchmark_dir: Path,
    label: str,
) -> None:
    """Write per-section directories for the improved procedure under benchmark_dir.

    Creates:
      section1_sanity/summary.txt
      section2_scale_kc{k}/summary.txt + scale_ratios_improved.csv  (per K_cache)
      section3_candidates_n1_kc{k}/summary.txt + candidates.csv     (per K_cache)
      section4_candidates_n10_kc{k}/summary.txt + candidates.csv    (per K_cache)
    """
    K_CACHE_VALUES = [1, 5, 10, 20]
    IMP_N_VALUES = [1, 5, 10]
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    _write_txt(
        benchmark_dir / 'section1_sanity' / 'summary.txt',
        ['===== SECTION 1: SANITY CHECKS (improved) =====', ''] + sanity_lines,
    )

    d_base_sorted = np.sort(cb_dists, axis=1)
    k_sorted = sorted(r_t_by_k.keys())

    for k_cache in K_CACHE_VALUES:
        if k_cache > d_base_sorted.shape[1]:
            continue
        d_base_kc = d_base_sorted[:, k_cache - 1]

        sec2_dir = benchmark_dir / f'section2_scale_kc{k_cache}'
        scale_lines = _scale_summary(ball_costs, nn_indices, cb_dists, r_t_by_k, k_cache)
        scale_rows = []
        for n_val in IMP_N_VALUES:
            if n_val > ball_costs.shape[1]:
                break
            nn_idx = nn_indices[:, n_val - 1]
            r_q = ball_costs[:, n_val - 1] + d_base_kc[nn_idx]
            scale_rows.append({
                'k_cache': k_cache, 'n': n_val,
                'r_q_mean': float(r_q.mean()),
                'r_q_median': float(np.median(r_q)),
                'r_q_max': float(r_q.max()),
            })
        _write_txt(
            sec2_dir / 'summary.txt',
            [f'===== SECTION 2: SCALE SUMMARY (improved, K_cache={k_cache}) =====',
             f'Benchmark: {label}', ''] + scale_lines + ['', 'See scale_ratios_improved.csv.'],
        )
        _write_csv(sec2_dir / 'scale_ratios_improved.csv', scale_rows)

        for n_val, sec_num, sec_name in [
            (1, 3, f'section3_candidates_n1_kc{k_cache}'),
            (10, 4, f'section4_candidates_n10_kc{k_cache}'),
        ]:
            if n_val > ball_costs.shape[1]:
                continue
            stats = per_candidate_stats(
                test_dists, ball_costs, nn_indices, cb_dists, d_base_kc, r_t_by_k, n=n_val,
            )
            sec_dir = benchmark_dir / sec_name
            header = (
                f'===== SECTION {sec_num}: CANDIDATES AT n={n_val},'
                f' K_cache={k_cache} (improved) ====='
            )
            cand_rows = []
            for s in stats:
                row = {
                    'candidate_idx': s['candidate_idx'],
                    'r_q': s['r_q'],
                    'n_test_inside': s['n_test_inside'],
                    'n_base_inside': s['n_base_inside'],
                }
                for k in k_sorted:
                    row[f'n_hits_K{k}'] = s.get(f'n_hits_K{k}', 0)
                cand_rows.append(row)
            n_test = test_dists.shape[0]
            n_base = cb_dists.shape[1]
            mean_test = float(np.mean([s['n_test_inside'] for s in stats]))
            mean_base = float(np.mean([s['n_base_inside'] for s in stats]))
            summ_lines = [
                header, f'Benchmark: {label}', '',
                f'n_candidates: {len(stats)}',
                f'n_test: {n_test}',
                f'n_base: {n_base}',
                f'mean n_test_inside: {mean_test:.2f}  ({100 * mean_test / n_test:.2f}%)',
                f'mean n_base_inside: {mean_base:.2f}  ({100 * mean_base / n_base:.2f}%)',
                '',
                'See candidates.csv for per-candidate data.',
            ]
            _write_txt(sec_dir / 'summary.txt', summ_lines)
            _write_csv(sec_dir / 'candidates.csv', cand_rows)

    print(f"  Section outputs (improved): {benchmark_dir}")

#-------------------------------------------------------------------------------

def per_candidate_stats(
    test_dists: np.ndarray,
    ball_costs: np.ndarray,
    nn_indices: np.ndarray,
    cb_dists: np.ndarray,
    d_base_kc: np.ndarray,
    r_t_by_k: dict,
    n: int,
) -> list:
    """Per-candidate stats for the improved (sphere-aware) procedure.

    r_q[c] = ball_costs[c, n-1] + d_base_kc[nn_indices[c, n-1]]
    where d_base_kc[c] is the k_cache-th nearest base vector distance from c.

    Args:
        test_dists: (n_test, n_cache)
        ball_costs: (n_cache, n_cache) sorted pairwise cache distances
        nn_indices: (n_cache, n_cache) argsorted indices from _compute_ball_costs
        cb_dists:   (n_cache, n_base)
        d_base_kc:  (n_cache,) k_cache-th nearest base distance per cache query
        r_t_by_k:   {K: (n_test,)}
        n:          ball-radius parameter

    Returns list of dicts, one per cache candidate.
    """
    nn_idx = nn_indices[:, n - 1]
    r_q = ball_costs[:, n - 1] + d_base_kc[nn_idx]
    k_sorted = sorted(r_t_by_k.keys())
    stats = []
    for c in range(ball_costs.shape[0]):
        r_c = float(r_q[c])
        d_tc = test_dists[:, c]
        row = {
            'candidate_idx': c,
            'r_q': r_c,
            'n_test_inside': int((d_tc < r_c).sum()),
            'n_base_inside': int((cb_dists[c] <= r_c).sum()),
        }
        for k in k_sorted:
            if k >= 0:
                row[f'n_hits_K{k}'] = int(((d_tc + r_t_by_k[k]) < r_c).sum())
        stats.append(row)
    return stats

#-------------------------------------------------------------------------------

def aggregate_plots(
    all_data: dict,
    all_debug_dicts: dict,
    dataset_label: str,
    outdir: Path,
) -> None:
    """Write diagnostic plots for the sphere procedure.

    all_data: {benchmark: rows} where rows come from _compute_rows (improved).
      Each row has: benchmark, n, k_cache, k_test, hit_rate, mean_r_q, mean_n_base_inside.
    all_debug_dicts: {benchmark: dict} with keys test_dists, cb_dists, ball_costs,
      nn_indices, r_t_by_k.

    Layout convention: rows=k_cache, cols=benchmark, lines=k_test.
    """
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    outdir.mkdir(parents=True, exist_ok=True)
    sns.set_style('whitegrid')

    K_CACHE_VALUES = [1, 5, 10, 20]
    K_TEST_VALUES = [0, 1, 5, 10]
    IMP_N_VALUES = [1, 5, 10]

    labels = list(all_data.keys())
    n_bm = len(labels)
    n_kc = len(K_CACHE_VALUES)
    n_nv = len(IMP_N_VALUES)

    kt_palette = sns.color_palette('tab10', len(K_TEST_VALUES))
    kt_color = {k: c for k, c in zip(K_TEST_VALUES, kt_palette)}

    # -- hit_rate_vs_n_improved -------------------------------------------------
    fig, axes = plt.subplots(n_kc, n_bm, figsize=(5 * n_bm, 3.5 * n_kc),
                             sharey=True, sharex=True)
    axes = np.array(axes).reshape(n_kc, n_bm)
    fig.suptitle(f"{dataset_label} Hit Rate vs. $n$ (improved)", fontsize=13)

    import matplotlib.patches as mpatches

    for i_kc, k_cache in enumerate(K_CACHE_VALUES):
        valid_kts = [kt for kt in K_TEST_VALUES if kt < k_cache]
        for j, lbl in enumerate(labels):
            cache_m = re.search(r'_c(\d+)k', lbl)
            cs = cache_m.group(1) if cache_m else '?'
            ax = axes[i_kc, j]
            rows = all_data[lbl]
            kc_rows = [r for r in rows if r['k_cache'] == k_cache]
            for k_test in valid_kts:
                x = [r['n'] for r in kc_rows if r['k_test'] == k_test]
                y = [r['hit_rate'] for r in kc_rows if r['k_test'] == k_test]
                ax.plot(x, y, marker='o', color=kt_color[k_test])
            if i_kc == 0:
                ax.set_title(f"Cache Size = {cs}")
            if j == 0:
                ax.set_ylabel(f"$K_{{cache}}={k_cache}$\nHit Rate")
            ax.set_xticks(IMP_N_VALUES)
            ax.set_ylim(0, 1.05)
            if i_kc == n_kc - 1:
                ax.set_xlabel("$n$")

    # proxy artists for complete legend (placed in bottom-left where K_cache=20 shows all K_test)
    kt_proxies = [
        mpatches.Patch(color=kt_color[k], label=f'$K_{{test}}={k}$')
        for k in K_TEST_VALUES
    ]
    axes[-1, 0].legend(handles=kt_proxies, loc='lower right', fontsize=8)
    plt.tight_layout()
    fig.savefig(outdir / 'hit_rate_vs_n_improved.png', dpi=120)
    plt.close(fig)

    # -- hit_rate_vs_k_cache_improved -------------------------------------------
    fig, axes = plt.subplots(n_nv, n_bm, figsize=(5 * n_bm, 3.5 * n_nv),
                             sharey=True, sharex=True)
    axes = np.array(axes).reshape(n_nv, n_bm)
    fig.suptitle(r"" + f"{dataset_label} Hit Rate vs. $K_{{cache}}$ (improved)", fontsize=13)

    for i_n, n_val in enumerate(IMP_N_VALUES):
        for j, lbl in enumerate(labels):
            cache_m = re.search(r'_c(\d+)k', lbl)
            cs = cache_m.group(1) if cache_m else '?'
            ax = axes[i_n, j]
            rows = all_data[lbl]
            n_rows = [r for r in rows if r['n'] == n_val]
            for k_test in K_TEST_VALUES:
                x = [r['k_cache'] for r in n_rows if r['k_test'] == k_test]
                y = [r['hit_rate'] for r in n_rows if r['k_test'] == k_test]
                if not x:
                    continue
                show_label = (j == 0 and i_n == 0)
                ax.plot(x, y, marker='o', color=kt_color[k_test],
                        label=f'$K_{{test}}={k_test}$' if show_label else None)
            if i_n == 0:
                ax.set_title(f"Cache Size = {cs}")
            if j == 0:
                ax.set_ylabel(f"$n={n_val}$\nHit Rate")
            ax.set_xticks(K_CACHE_VALUES)
            ax.set_ylim(0, 1.05)
            if i_n == n_nv - 1:
                ax.set_xlabel(r"$K_{cache}$")

    axes[0, 0].legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    fig.savefig(outdir / 'hit_rate_vs_k_cache_improved.png', dpi=120)
    plt.close(fig)

    # -- boundary_proximity_improved --------------------------------------------
    # One file per K_test. Rows=valid K_cache (k_test < k_cache), cols=benchmark.
    # Proximity = slack_Ktest / r_q at best candidate; filter to prox > 0.
    for k_test_bp in K_TEST_VALUES:
        valid_kcs_bp = [kc for kc in K_CACHE_VALUES if k_test_bp < kc]
        if not valid_kcs_bp:
            continue
        n_rows_bp = len(valid_kcs_bp)
        fig_bp, axes_bp = plt.subplots(
            n_rows_bp, n_bm, figsize=(5 * n_bm, 3.5 * n_rows_bp), sharey=True,
        )
        axes_bp = np.array(axes_bp).reshape(n_rows_bp, n_bm)
        fig_bp.suptitle(
            f"{dataset_label} Boundary Proximity $(D(Q, q_n) - D(q, Q)) / D(Q, q_n)$"
            f" ($K_{{test}}={k_test_bp}$)",
            fontsize=13,
        )
        for i_kc_bp, k_cache in enumerate(valid_kcs_bp):
            for j, lbl in enumerate(labels):
                cache_m = re.search(r'_c(\d+)k', lbl)
                cs = cache_m.group(1) if cache_m else '?'
                ax_bp = axes_bp[i_kc_bp, j]
                dbg = all_debug_dicts[lbl]
                bc = dbg['ball_costs']
                nn_idx_arr = dbg['nn_indices']
                cb = dbg['cb_dists']
                td = dbg['test_dists']
                r_t = dbg['r_t_by_k'].get(
                    k_test_bp, np.zeros(td.shape[0], dtype=np.float32)
                )
                n_test_q = td.shape[0]
                d_base_sorted = np.sort(cb, axis=1)
                d_base_kc = d_base_sorted[:, k_cache - 1]
                prox_by_n = []
                for n_val in IMP_N_VALUES:
                    nn_idx = nn_idx_arr[:, n_val - 1]
                    r_q = bc[:, n_val - 1] + d_base_kc[nn_idx]
                    all_slacks = r_q[np.newaxis, :] - td - r_t[:, np.newaxis]
                    best_c_bp = all_slacks.argmax(axis=1)
                    best_slack = all_slacks[np.arange(n_test_q), best_c_bp]
                    best_rq = r_q[best_c_bp]
                    prox = np.where(
                        (best_slack > 0) & (best_rq > 0),
                        best_slack / best_rq,
                        np.nan,
                    )
                    prox_by_n.append(prox[~np.isnan(prox)])
                ax_bp.boxplot(prox_by_n, tick_labels=[str(n) for n in IMP_N_VALUES],
                              showfliers=True,
                              flierprops={'marker': '.', 'markersize': 3, 'alpha': 0.3})
                ax_bp.axhline(0.0, color='black', linewidth=0.8, linestyle=':')
                ax_bp.axhline(1.0, color='gray', linewidth=0.8, linestyle=':')
                if i_kc_bp == 0:
                    ax_bp.set_title(f"Cache Size = {cs}")
                if j == 0:
                    ax_bp.set_ylabel(f"$K_{{cache}}={k_cache}$\nProximity (0=boundary, 1=center)")
                if i_kc_bp == n_rows_bp - 1:
                    ax_bp.set_xlabel("$n$")
                ax_bp.set_ylim(-0.1, 1.1)
        plt.tight_layout()
        fig_bp.savefig(outdir / f'boundary_proximity_improved_kt{k_test_bp}.png', dpi=120)
        plt.close(fig_bp)

    # -- rc_distribution_improved ----------------------------------------------
    # Boxplot of n_base_inside per candidate vs n. Rows=k_cache, cols=benchmark.
    # Mirrors r_c_distribution.png in the original procedure.
    fig, axes = plt.subplots(n_kc, n_bm, figsize=(5 * n_bm, 3.5 * n_kc), sharey='row')
    axes = np.array(axes).reshape(n_kc, n_bm)
    fig.suptitle(f"{dataset_label} " + r"$|R_c|$ Distribution (improved)", fontsize=13)

    for i_kc, k_cache in enumerate(K_CACHE_VALUES):
        for j, lbl in enumerate(labels):
            cache_m = re.search(r'_c(\d+)k', lbl)
            cs = cache_m.group(1) if cache_m else '?'
            ax = axes[i_kc, j]
            dbg = all_debug_dicts[lbl]
            bc = dbg['ball_costs']
            nn_idx_arr = dbg['nn_indices']
            cb = dbg['cb_dists']
            d_base_sorted = np.sort(cb, axis=1)
            d_base_kc = d_base_sorted[:, k_cache - 1]
            data = []
            for n_val in IMP_N_VALUES:
                nn_idx = nn_idx_arr[:, n_val - 1]
                r_q = bc[:, n_val - 1] + d_base_kc[nn_idx]
                n_base_in = (cb <= r_q[:, np.newaxis]).sum(axis=1).astype(float)
                data.append(n_base_in)
            ax.boxplot(data, tick_labels=[str(n) for n in IMP_N_VALUES],
                       showfliers=True,
                       flierprops={'marker': '.', 'markersize': 3, 'alpha': 0.3})
            ax.set_ylim(bottom=0)
            if i_kc == 0:
                ax.set_title(f"Cache Size = {cs}")
            if j == 0:
                ax.set_ylabel(f"$K_{{cache}}={k_cache}$\n" + r"$|R_c|$")
            if i_kc == n_kc - 1:
                ax.set_xlabel("$n$")

    plt.tight_layout()
    fig.savefig(outdir / 'rc_distribution_improved.png', dpi=120)
    plt.close(fig)

    # -- scale_ratio_heatmap_improved ------------------------------------------
    # mean D(q,E_K_test) / mean r_q. One figure per k_cache.
    K_TEST_GT0 = [k for k in K_TEST_VALUES if k > 0]
    for k_cache in K_CACHE_VALUES:
        heat_arrays_imp = {}
        all_ratios_imp = []
        for lbl in labels:
            dbg = all_debug_dicts[lbl]
            bc = dbg['ball_costs']
            nn_idx_arr = dbg['nn_indices']
            cb = dbg['cb_dists']
            r_t_by_k = dbg['r_t_by_k']
            d_base_sorted = np.sort(cb, axis=1)
            d_base_kc = d_base_sorted[:, k_cache - 1]
            mat = np.full((len(K_TEST_GT0), len(IMP_N_VALUES)), np.nan)
            for i_k, k_test in enumerate(K_TEST_GT0):
                if k_test not in r_t_by_k or k_test >= k_cache:
                    continue
                r_t_mean = float(r_t_by_k[k_test].mean())
                for i_n, n_val in enumerate(IMP_N_VALUES):
                    nn_idx = nn_idx_arr[:, n_val - 1]
                    r_q = bc[:, n_val - 1] + d_base_kc[nn_idx]
                    rq_mean = float(r_q.mean())
                    if rq_mean > 0:
                        mat[i_k, i_n] = r_t_mean / rq_mean
            heat_arrays_imp[lbl] = mat
            all_ratios_imp.extend(mat[~np.isnan(mat)].tolist())

        if not all_ratios_imp:
            continue

        import matplotlib as mpl_imp
        half_imp = max(abs(max(all_ratios_imp) - 1.0), abs(1.0 - min(all_ratios_imp)))
        vmin_i = 1.0 - half_imp
        vmax_i = 1.0 + half_imp

        fig_h, axes_h = plt.subplots(1, n_bm, figsize=(3.5 * n_bm + 1.5, 4.5))
        if n_bm == 1:
            axes_h = [axes_h]
        fig_h.suptitle(
            f"{dataset_label} Scale Ratio $D(q,E_K) / D(Q,q_n)$"
            f" (improved, $K_{{cache}}={k_cache}$)",
            fontsize=13, y=1.01,
        )
        import seaborn as sns_h
        for j, (lbl, ax_h) in enumerate(zip(labels, axes_h)):
            cache_m = re.search(r'_c(\d+)k', lbl)
            cs = cache_m.group(1) if cache_m else '?'
            mat = heat_arrays_imp[lbl]
            ytick_labels = [f'$K_{{test}}={k}$' for k in K_TEST_GT0] if j == 0 else False
            sns_h.heatmap(
                mat, ax=ax_h,
                annot=True, fmt='.2f', annot_kws={'size': 10},
                xticklabels=[str(n) for n in IMP_N_VALUES],
                yticklabels=ytick_labels,
                center=1.0, cmap='RdBu_r', vmin=vmin_i, vmax=vmax_i,
                cbar=False,
                linewidths=0.5, linecolor='white',
            )
            ax_h.set_xlabel("$n$")
            if j == 0:
                ax_h.set_ylabel("$K_{test}$")
                ax_h.yaxis.set_tick_params(rotation=0)
            ax_h.set_title(f"Cache Size = {cs}")

        fig_h.subplots_adjust(right=0.85, top=0.88)
        cbar_ax_h = fig_h.add_axes([0.87, 0.15, 0.025, 0.65])
        sm_h = plt.cm.ScalarMappable(
            cmap='RdBu_r',
            norm=mpl_imp.colors.Normalize(vmin=vmin_i, vmax=vmax_i),
        )
        sm_h.set_array([])
        fig_h.colorbar(sm_h, cax=cbar_ax_h, label='$D(q,E_K)$ / $r_q$')
        fig_h.savefig(outdir / f'scale_ratio_heatmap_improved_kc{k_cache}.png',
                      dpi=120, bbox_inches='tight')
        plt.close(fig_h)

    # -- hit_count_distribution_improved ---------------------------------------
    # One file per K_cache. rows=n, cols=benchmark, box per valid K_test.
    import matplotlib.patches as mpatches_hc
    for k_cache_hc in K_CACHE_VALUES:
        valid_kts_hc = [kt for kt in K_TEST_VALUES if kt < k_cache_hc]
        fig_hc, axes_hc = plt.subplots(n_nv, n_bm, figsize=(5 * n_bm, 4 * n_nv), sharey='row')
        axes_hc = np.array(axes_hc).reshape(n_nv, n_bm)
        fig_hc.suptitle(
            f"{dataset_label} Hit Count per Candidate (improved, $K_{{cache}}={k_cache_hc}$)",
            fontsize=13,
        )
        legend_patches_hc = [
            mpatches_hc.Patch(facecolor=kt_color[k], alpha=0.7, label=f'$K_{{test}}={k}$')
            for k in valid_kts_hc
        ]
        for i_n, n_val in enumerate(IMP_N_VALUES):
            for j, lbl in enumerate(labels):
                cache_m = re.search(r'_c(\d+)k', lbl)
                cs = cache_m.group(1) if cache_m else '?'
                ax = axes_hc[i_n, j]
                dbg = all_debug_dicts[lbl]
                bc = dbg['ball_costs']
                nn_idx_arr = dbg['nn_indices']
                cb = dbg['cb_dists']
                r_t_by_k = dbg['r_t_by_k']
                d_base_sorted = np.sort(cb, axis=1)
                d_base_kc = d_base_sorted[:, k_cache_hc - 1]
                stats = per_candidate_stats(
                    dbg['test_dists'], bc, nn_idx_arr, cb, d_base_kc, r_t_by_k, n=n_val,
                )
                data_hc = [
                    [s['n_test_inside'] if k == 0 else s.get(f'n_hits_K{k}', 0) for s in stats]
                    for k in valid_kts_hc
                ]
                if data_hc:
                    bp = ax.boxplot(
                        data_hc,
                        tick_labels=[f'$K_{{test}}={k}$' for k in valid_kts_hc],
                        patch_artist=True, showfliers=True,
                        flierprops={'marker': '.', 'markersize': 3, 'alpha': 0.4},
                    )
                    for patch, k in zip(bp['boxes'], valid_kts_hc):
                        patch.set_facecolor(kt_color[k])
                        patch.set_alpha(0.7)
                if i_n == 0:
                    ax.set_title(f"Cache Size = {cs}")
                if j == 0:
                    ax.set_ylabel(f"$n={n_val}$\nHit count per candidate")
                ax.set_ylim(bottom=0)
                if i_n == n_nv - 1:
                    ax.set_xlabel("$K_{test}$")
        if legend_patches_hc:
            axes_hc[0, 0].legend(handles=legend_patches_hc, fontsize=8, loc='upper right')
        plt.tight_layout()
        fig_hc.savefig(outdir / f'hit_count_distribution_improved_kc{k_cache_hc}.png', dpi=120)
        plt.close(fig_hc)

    # -- aggregate_coverage_improved_kc{k_cache} -------------------------------
    # One file per K_cache. rows=valid K_test, cols=benchmark.
    # Solid=test hit_rate, dashed gray=base coverage (independent of K_test).
    import matplotlib.lines as mlines_ac
    for k_cache_cov in K_CACHE_VALUES:
        valid_kts_cov = [kt for kt in K_TEST_VALUES if kt < k_cache_cov]
        n_rows_cov = len(valid_kts_cov)
        fig_cov, axes_cov = plt.subplots(
            n_rows_cov, n_bm, figsize=(5 * n_bm, 4 * n_rows_cov), sharey=True,
        )
        axes_cov = np.array(axes_cov).reshape(n_rows_cov, n_bm)
        fig_cov.suptitle(
            f"{dataset_label} Coverage ($K_{{cache}}={k_cache_cov}$, improved)",
            fontsize=13,
        )
        for i_kt_cov, k_test_cov in enumerate(valid_kts_cov):
            for j, lbl in enumerate(labels):
                cache_m = re.search(r'_c(\d+)k', lbl)
                cs = cache_m.group(1) if cache_m else '?'
                ax_cov = axes_cov[i_kt_cov, j]
                dbg = all_debug_dicts[lbl]
                bc = dbg['ball_costs']
                nn_idx_arr = dbg['nn_indices']
                cb = dbg['cb_dists']
                d_base_kc_cov = np.sort(cb, axis=1)[:, k_cache_cov - 1]
                base_cov_vals, test_cov_vals = [], []
                for n_val in IMP_N_VALUES:
                    nn_idx = nn_idx_arr[:, n_val - 1]
                    r_q = bc[:, n_val - 1] + d_base_kc_cov[nn_idx]
                    base_cov_vals.append(
                        float(np.any(cb <= r_q[:, np.newaxis], axis=0).mean())
                    )
                    row_cov = next(
                        (r for r in all_data[lbl]
                         if r['n'] == n_val and r['k_cache'] == k_cache_cov
                         and r['k_test'] == k_test_cov),
                        None,
                    )
                    test_cov_vals.append(row_cov['hit_rate'] if row_cov else np.nan)
                ax_cov.plot(IMP_N_VALUES, test_cov_vals, marker='o', color='black',
                            label='test' if (i_kt_cov == 0 and j == 0) else None)
                ax_cov.plot(IMP_N_VALUES, base_cov_vals, marker='s', linestyle='--',
                            color='gray',
                            label=r'base ($\cup R_q$)' if (i_kt_cov == 0 and j == 0) else None)
                if i_kt_cov == 0:
                    ax_cov.set_title(f"Cache Size = {cs}")
                if j == 0:
                    ax_cov.set_ylabel(f"$K_{{test}}={k_test_cov}$\nFraction")
                ax_cov.set_xticks(IMP_N_VALUES)
                ax_cov.set_ylim(0, 1.05)
                if i_kt_cov == n_rows_cov - 1:
                    ax_cov.set_xlabel("$n$")
        legend_handles = [
            mlines_ac.Line2D([], [], color='black', marker='o', label='test coverage'),
            mlines_ac.Line2D([], [], color='gray', marker='s', linestyle='--',
                             label=r'base ($\cup R_q$)'),
        ]
        axes_cov[0, 0].legend(handles=legend_handles, fontsize=7, loc='lower right')
        plt.tight_layout()
        fig_cov.savefig(outdir / f'aggregate_coverage_improved_kc{k_cache_cov}.png', dpi=120)
        plt.close(fig_cov)

    # -- cache_size_comparison_improved ----------------------------------------
    # rows=n, x=cache_size, lines=K_test (one color), one panel per k_cache column.
    # Layout: rows=k_cache, one panel per benchmark is too wide; do rows=n, cols=k_cache.
    fig, axes = plt.subplots(n_nv, n_kc, figsize=(5 * n_kc, 4 * n_nv), sharey=True)
    axes = np.array(axes).reshape(n_nv, n_kc)
    fig.suptitle(f"{dataset_label} Hit Rate vs. Cache Size (improved)", fontsize=13)

    cache_sizes_imp = []
    for lbl in labels:
        m = re.search(r'_c(\d+)k', lbl)
        cache_sizes_imp.append(int(m.group(1)) if m else 0)

    for i_n, n_val in enumerate(IMP_N_VALUES):
        for i_kc, k_cache in enumerate(K_CACHE_VALUES):
            ax = axes[i_n, i_kc]
            for k_test in K_TEST_VALUES:
                y_vals = []
                x_vals = []
                for lbl, cs_int in zip(labels, cache_sizes_imp):
                    rows_bm = all_data[lbl]
                    row = next(
                        (r for r in rows_bm
                         if r['n'] == n_val and r['k_cache'] == k_cache and r['k_test'] == k_test),
                        None,
                    )
                    if row is None:
                        continue
                    y_vals.append(row['hit_rate'])
                    x_vals.append(cs_int)
                if x_vals:
                    ax.plot(x_vals, y_vals, marker='o', color=kt_color[k_test],
                            linestyle='--' if k_test == 0 else '-')
            if i_n == 0:
                ax.set_title(f"$K_{{cache}}={k_cache}$")
            if i_kc == 0:
                ax.set_ylabel(f"$n={n_val}$\nHit Rate")
            ax.set_ylim(0, 1.05)
            if i_n == n_nv - 1:
                ax.set_xlabel("Cache Size")

    import matplotlib.patches as mpatches_cs
    kt_proxies_cs = [
        mpatches_cs.Patch(color=kt_color[k], label=f'$K_{{test}}={k}$')
        for k in K_TEST_VALUES
    ]
    axes[0, -1].legend(handles=kt_proxies_cs, fontsize=8)
    plt.tight_layout()
    fig.savefig(outdir / 'cache_size_comparison_improved.png', dpi=120)
    plt.close(fig)

    # -- recall_degradation_improved -------------------------------------------
    # x=n, y=recall@K_test. Rows=k_cache, cols=benchmark.
    has_knn_imp = any('knn_idx_by_k' in all_debug_dicts[lbl] for lbl in labels)
    if has_knn_imp:
        fig, axes = plt.subplots(n_kc, n_bm, figsize=(5 * n_bm, 3.5 * n_kc), sharey=True)
        axes = np.array(axes).reshape(n_kc, n_bm)
        fig.suptitle(f"{dataset_label} Recall@K (union coverage) vs. $n$ (improved)", fontsize=13)

        for i_kc, k_cache in enumerate(K_CACHE_VALUES):
            for j, lbl in enumerate(labels):
                cache_m = re.search(r'_c(\d+)k', lbl)
                cs = cache_m.group(1) if cache_m else '?'
                ax = axes[i_kc, j]
                dbg = all_debug_dicts[lbl]
                bc = dbg['ball_costs']
                nn_idx_arr = dbg['nn_indices']
                cb = dbg['cb_dists']
                knn_idx_by_k = dbg.get('knn_idx_by_k', {})
                d_base_sorted = np.sort(cb, axis=1)
                d_base_kc = d_base_sorted[:, k_cache - 1]

                for k_test in K_TEST_GT0:
                    if k_test not in knn_idx_by_k or k_test >= k_cache:
                        continue
                    knn = knn_idx_by_k[k_test]
                    recalls = []
                    for n_val in IMP_N_VALUES:
                        nn_idx = nn_idx_arr[:, n_val - 1]
                        r_q = bc[:, n_val - 1] + d_base_kc[nn_idx]
                        base_covered = np.any(cb <= r_q[:, np.newaxis], axis=0)
                        recalls.append(float(base_covered[knn].all(axis=1).mean()))
                    ax.plot(IMP_N_VALUES, recalls, marker='o', color=kt_color[k_test],
                            label=f'$K_{{test}}={k_test}$' if (i_kc == n_kc - 1 and j == 0) else None)

                if i_kc == 0:
                    ax.set_title(f"Cache Size = {cs}")
                if j == 0:
                    ax.set_ylabel(f"$K_{{cache}}={k_cache}$\nRecall@K")
                ax.set_xticks(IMP_N_VALUES)
                ax.set_ylim(0, 1.05)
                if i_kc == n_kc - 1:
                    ax.set_xlabel("$n$")

        axes[-1, 0].legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(outdir / 'recall_degradation_improved.png', dpi=120)
        plt.close(fig)

    # -- utility_ratio_improved ------------------------------------------------
    # ratio = (n_test_inside/n_test) / (n_base_inside/n_base) per candidate.
    # Rows=K_cache, cols=benchmark, boxplot per IMP_N_VALUES.
    fig, axes = plt.subplots(n_kc, n_bm, figsize=(5 * n_bm, 3.5 * n_kc), sharey='row')
    axes = np.array(axes).reshape(n_kc, n_bm)
    fig.suptitle(f"{dataset_label} Utility Ratio (test cov / base cov) per Candidate (improved)",
                 fontsize=12)

    for i_kc, k_cache in enumerate(K_CACHE_VALUES):
        for j, lbl in enumerate(labels):
            cache_m = re.search(r'_c(\d+)k', lbl)
            cs = cache_m.group(1) if cache_m else '?'
            ax = axes[i_kc, j]
            dbg = all_debug_dicts[lbl]
            bc = dbg['ball_costs']
            nn_idx_arr = dbg['nn_indices']
            cb = dbg['cb_dists']
            td = dbg['test_dists']
            n_test_total = td.shape[0]
            n_base_total = cb.shape[1]
            d_base_sorted = np.sort(cb, axis=1)
            d_base_kc = d_base_sorted[:, k_cache - 1]
            data_ur = []
            for n_val in IMP_N_VALUES:
                nn_idx = nn_idx_arr[:, n_val - 1]
                r_q = bc[:, n_val - 1] + d_base_kc[nn_idx]
                n_test_in = (td < r_q[np.newaxis, :]).sum(axis=0).astype(float)
                n_base_in = (cb <= r_q[:, np.newaxis]).sum(axis=1).astype(float)
                valid = n_base_in > 0
                if valid.any():
                    ratio = (n_test_in[valid] / n_test_total) / (n_base_in[valid] / n_base_total)
                else:
                    ratio = np.array([])
                data_ur.append(ratio)
            ax.boxplot(data_ur, tick_labels=[str(n) for n in IMP_N_VALUES],
                       showfliers=True,
                       flierprops={'marker': '.', 'markersize': 3, 'alpha': 0.3})
            ax.axhline(1.0, color='red', linewidth=1.0, linestyle='--',
                       label='ratio = 1' if (i_kc == n_kc - 1 and j == 0) else None)
            if i_kc == 0:
                ax.set_title(f"Cache Size = {cs}")
            if j == 0:
                ax.set_ylabel(f"$K_{{cache}}={k_cache}$\nUtility Ratio")
            ax.set_ylim(bottom=0)
            if i_kc == n_kc - 1:
                ax.set_xlabel("$n$")

    for ax in axes.flat:
        ax.set_yticks(sorted(set(list(ax.get_yticks()) + [1.0])))
    axes[-1, 0].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(outdir / 'utility_ratio_improved.png', dpi=120)
    plt.close(fig)

    print(f"  Improved debug plots: {outdir}")

#-------------------------------------------------------------------------------

