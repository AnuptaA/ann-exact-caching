#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Vector-coverage ceiling vs. actual CIG-certified hit rate (sigma=1.0 only).

Standalone debug script, not part of the main population pipeline: answers
"how much of the current hit-rate shortfall is a certification problem
(recoverable via a different/relaxed certification scheme) vs. a population
problem (the right vectors just aren't cached, no certification scheme can
help)" by comparing two quantities at matched budgets, for test queries:

- vector coverage (ceiling): fraction of test queries whose full true top-K
  neighbor set is physically present somewhere in the cache, regardless of
  whether CIG's distance-threshold check would certify them.
- actual test hit rate: the real CIG-certified hit rate, computed via
  debug_trajectory.py's monotonic_hit_rate_pass (exact, not simulated --
  see that function's docstring for the equivalence proof to real
  simulation, which only holds at sigma=1.0).

Sweeps real budget caps -- the same BUDGET_FRACTIONS the main pipeline uses
(run_population.py) -- calling select_candidates once per fraction, not a
single unconstrained run interpolated after the fact. An earlier version of
this script derived every budget level from one unconstrained
(budget_B=n_base) trajectory, matching debug_trajectory.py's cheap "one run,
re-derive every budget" pattern; that turned out to be unreliable here
whenever a single oversized candidate's own R_c already covered most/all of
the base in one greedy pick, which happens for several real (benchmark, K)
combos -- the interpolation then had no earlier trajectory state to report
for any budget below that pick's, producing a flat-then-cliff artifact
instead of a real answer. A real budget cap simply excludes an unaffordable
candidate and selects from smaller ones instead, so re-running per fraction
gives the trustworthy picture. Candidate proposal is budget-independent and
still runs once per (benchmark, K) combo; only selection repeats (15x
instead of 1x) -- still no CacheSimulator/simulation stage, so still cheaper
than the main pipeline, just no longer a single-run approximation.
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent.parent))

from simulations.population import debug_common as dc
from simulations.population.debug_common import parse_benchmark_sizes
from simulations.population.debug_trajectory import monotonic_hit_rate_pass
from simulations.population.run_population import BUDGET_FRACTIONS
from utils.base.selection import select_candidates

#-------------------------------------------------------------------------------

K_VALUES = [0, 1, 5, 10]
SIGMA = 1.0
OUT_DIR = Path('simulations/debug/population_vector_coverage')

#-------------------------------------------------------------------------------


def compute_combo(
    dataset: str, benchmark_name: str, K: int, invariants: dict = None,
    budget_fractions: list = None,
) -> dict:
    """Propose once, then select once per real budget-fraction value against
    that same candidate pool -- each budget_pct point is an independent,
    genuinely budget-constrained selection run, not an interpolation.

    budget_fractions: defaults to run_population.py's full 15-value
    BUDGET_FRACTIONS; pass a shorter list to trade resolution for speed."""
    budget_fractions = budget_fractions if budget_fractions is not None else BUDGET_FRACTIONS
    base, cache_q, test_q, bdir = dc.load_benchmark(dataset, benchmark_name)
    n_cache, n_test = len(cache_q), len(test_q)
    train_r_t, test_r_t = dc.load_gt_r_t(bdir, K, n_cache, n_test)
    n_base = len(base)

    candidates = dc.propose_full_pool(
        base, cache_q, test_q, train_r_t, test_r_t, K, invariants=invariants,
    )
    if not candidates:
        print(f"    {dataset}/{benchmark_name} K={K}: no candidates proposed, skipping")
        return None

    test_dist_matrix = invariants['test_dist_only'] if invariants is not None else None
    test_gt_indices = dc.load_gt_indices(bdir, K, n_test) if K > 0 else None

    budget_pct, test_hit_curve, vector_coverage = [], [], []
    rho_curve, median_overshoot_curve = [], []
    overshoot_arrays = []
    for frac in budget_fractions:
        budget_B = max(1, int(frac * n_base))
        selected, _traj = select_candidates(
            candidates, budget_B, strategy=dc.STRATEGY,
            n_train_queries=n_cache, n_test_queries=n_test, seed=dc.SEED,
        )
        rel_overshoot = np.array([], dtype=np.float64)
        if not selected:
            hit_rate, coverage, rho, median_overshoot = 0.0, 0.0, 0.0, float('nan')
        else:
            test_cert = monotonic_hit_rate_pass(
                selected, test_q, test_r_t, K, SIGMA, dc.METRIC, dist_matrix=test_dist_matrix,
            )
            hit_rate = float(np.mean(test_cert > 0))
            if K == 0:
                # Same identity as always: at K=0 there's no separate
                # "neighbor vector presence" question -- the CIG hit
                # condition is pure point-containment, which coverage
                # computes identically. rho is trivially 1.0 (hit_rate ==
                # coverage exactly); "overshoot" past a candidate's r_c
                # isn't a meaningful concept the same way at K=0, so it's
                # left undefined here.
                coverage = hit_rate
                rho = 1.0
                median_overshoot = float('nan')
            else:
                cached_mask = np.zeros(n_base, dtype=bool)
                for c in selected:
                    cached_mask |= c.R_c
                gt_matrix = np.array(test_gt_indices, dtype=np.int32)
                covered = cached_mask[gt_matrix].all(axis=1)
                coverage = float(covered.mean())
                rho = hit_rate / coverage if coverage > 0 else 0.0

                # Median relative overshoot among covered-but-uncertified
                # queries: (D(t,Q)+r_t-r_c)/r_c for each query's best
                # (closest-to-certifying) selected candidate. Reuses
                # test_dist_matrix via each candidate's source_index --
                # same optimization monotonic_hit_rate_pass's dist_matrix
                # param already relies on -- instead of a fresh matmul.
                covered_and_missed = covered & (test_cert <= 0)
                if covered_and_missed.any():
                    if test_dist_matrix is not None:
                        src_idx = [c.source_index for c in selected]
                        D = test_dist_matrix[src_idx, :].T
                    else:
                        q_hat = test_q / np.linalg.norm(test_q, axis=1, keepdims=True)
                        cand_centers = np.array([c.query for c in selected])
                        cand_centers_hat = cand_centers / np.linalg.norm(
                            cand_centers, axis=1, keepdims=True,
                        )
                        dots = np.clip(q_hat @ cand_centers_hat.T, -1.0, 1.0)
                        D = np.arccos(dots) / np.pi
                    cand_rc = np.array([c.r_c for c in selected])
                    overshoot_matrix = D + test_r_t[:, None] - cand_rc[None, :]
                    min_overshoot = overshoot_matrix.min(axis=1)
                    best_idx = overshoot_matrix.argmin(axis=1)
                    rel_overshoot = (
                        min_overshoot[covered_and_missed]
                        / cand_rc[best_idx[covered_and_missed]]
                    )
                    median_overshoot = float(np.median(rel_overshoot))
                else:
                    median_overshoot = float('nan')
        budget_pct.append(frac * 100)
        test_hit_curve.append(hit_rate)
        vector_coverage.append(coverage)
        rho_curve.append(rho)
        median_overshoot_curve.append(median_overshoot)
        overshoot_arrays.append(rel_overshoot)

    return {
        'budget_pct': np.array(budget_pct),
        'test_hit_curve': np.array(test_hit_curve),
        'vector_coverage': np.array(vector_coverage),
        'rho_curve': np.array(rho_curve),
        'overshoot_arrays': overshoot_arrays,
        'median_overshoot_curve': np.array(median_overshoot_curve),
        'final_test_hit_rate': test_hit_curve[-1],
        'final_vector_coverage': vector_coverage[-1],
    }


def plot_dataset_coverage_ceiling(
    dataset: str, data_by_bm_k: dict, test_n: int, k_values: list = None,
) -> None:
    """rows = cache size, columns = K (matching debug_trajectory.py's grid
    convention), two series per cell: vector coverage (ceiling, dashed) and
    actual test hit rate (solid), x-axis = % of budget. Markers at each
    swept budget-fraction point -- these are discrete independent
    measurements, not a continuous interpolated curve.

    One row per benchmark in data_by_bm_k -- passing multiple benchmark
    names (e.g. two different cache sizes) in a single call naturally
    produces one combined multi-row comparison plot."""
    k_values = k_values if k_values is not None else K_VALUES
    benchmarks = [bm for bm in data_by_bm_k if any(data_by_bm_k[bm].get(k) for k in k_values)]
    if not benchmarks:
        return
    benchmarks = sorted(benchmarks, key=lambda bm: parse_benchmark_sizes(bm)[0])

    n_rows, n_cols = len(benchmarks), len(k_values)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.8 * n_rows), squeeze=False)

    for r, bm in enumerate(benchmarks):
        train_n, _ = parse_benchmark_sizes(bm)
        for c, K in enumerate(k_values):
            ax = axes[r][c]
            d = data_by_bm_k[bm].get(K)
            if d is None:
                ax.axis('off')
                continue
            ax.plot(d['budget_pct'], d['vector_coverage'], marker='o', markersize=3,
                     label='vector coverage (ceiling)', linestyle='--')
            ax.plot(d['budget_pct'], d['test_hit_curve'], marker='s', markersize=3,
                     label='actual test hit rate', linestyle='-')
            ax.set_ylim(-0.02, 1.02)

            if r == 0:
                ax.set_title(f'$K={K}$', fontsize=10)
            if c == 0:
                ax.set_ylabel(f'{train_n} Train Queries\nfraction')
            if r == n_rows - 1:
                ax.set_xlabel('% of budget')

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               fontsize=8, bbox_to_anchor=(0.5, -0.02 / n_rows))
    fig.suptitle(
        f'{dataset.upper()}: Vector Coverage Ceiling vs. Actual Test Hit Rate '
        f'({test_n} test queries, sigma=1.0)', fontsize=13,
    )
    plt.tight_layout()

    outdir = OUT_DIR / dataset
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / 'coverage_ceiling_vs_budget_pct.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")


def plot_dataset_rho(
    dataset: str, data_by_bm_k: dict, test_n: int, k_values: list = None,
) -> None:
    """rows = cache size, columns = K, single series: rho = hit_rate /
    vector_coverage (of queries whose neighbors are fully cached, what
    fraction actually get CIG-certified). y=1 reference line marks perfect
    certification efficiency."""
    k_values = k_values if k_values is not None else K_VALUES
    benchmarks = [bm for bm in data_by_bm_k if any(data_by_bm_k[bm].get(k) for k in k_values)]
    if not benchmarks:
        return
    benchmarks = sorted(benchmarks, key=lambda bm: parse_benchmark_sizes(bm)[0])

    n_rows, n_cols = len(benchmarks), len(k_values)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.8 * n_rows), squeeze=False)

    for r, bm in enumerate(benchmarks):
        train_n, _ = parse_benchmark_sizes(bm)
        for c, K in enumerate(k_values):
            ax = axes[r][c]
            d = data_by_bm_k[bm].get(K)
            if d is None:
                ax.axis('off')
                continue
            ax.plot(d['budget_pct'], d['rho_curve'], marker='o', markersize=3,
                     color='tab:green', label='rho = hit_rate / coverage')
            ax.axhline(1.0, linestyle=':', color='gray', linewidth=0.8, alpha=0.5)
            ax.set_ylim(-0.02, 1.05)

            if r == 0:
                ax.set_title(f'$K={K}$', fontsize=10)
            if c == 0:
                ax.set_ylabel(f'{train_n} Train Queries\nrho')
            if r == n_rows - 1:
                ax.set_xlabel('% of budget')

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               fontsize=8, bbox_to_anchor=(0.5, -0.02 / n_rows))
    fig.suptitle(
        f'{dataset.upper()}: Certification Efficiency (rho) vs. Budget '
        f'({test_n} test queries, sigma=1.0)', fontsize=13,
    )
    plt.tight_layout()

    outdir = OUT_DIR / dataset
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / 'rho_vs_budget_pct.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")


def plot_dataset_overshoot(
    dataset: str, data_by_bm_k: dict, test_n: int, k_values: list = None,
) -> None:
    """rows = cache size, columns = K, single series: median relative
    overshoot -- (D(t,Q)+r_t-r_c)/r_c for each covered-but-uncertified
    query's best candidate -- among queries whose neighbors are fully
    cached but not CIG-certified. NaN (no covered-and-missed queries at
    that budget) leaves a gap in the line, matplotlib's native NaN
    handling."""
    k_values = k_values if k_values is not None else K_VALUES
    benchmarks = [bm for bm in data_by_bm_k if any(data_by_bm_k[bm].get(k) for k in k_values)]
    if not benchmarks:
        return
    benchmarks = sorted(benchmarks, key=lambda bm: parse_benchmark_sizes(bm)[0])

    n_rows, n_cols = len(benchmarks), len(k_values)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.8 * n_rows), squeeze=False)

    for r, bm in enumerate(benchmarks):
        train_n, _ = parse_benchmark_sizes(bm)
        for c, K in enumerate(k_values):
            ax = axes[r][c]
            d = data_by_bm_k[bm].get(K)
            if d is None:
                ax.axis('off')
                continue
            ax.plot(d['budget_pct'], d['median_overshoot_curve'], marker='o', markersize=3,
                     color='tab:red', label='median relative overshoot')

            if r == 0:
                ax.set_title(f'$K={K}$', fontsize=10)
            if c == 0:
                ax.set_ylabel(f'{train_n} Train Queries\n(D+r_t-r_c)/r_c')
            if r == n_rows - 1:
                ax.set_xlabel('% of budget')

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               fontsize=8, bbox_to_anchor=(0.5, -0.02 / n_rows))
    fig.suptitle(
        f'{dataset.upper()}: Median Relative Overshoot, Covered-but-Missed Queries '
        f'({test_n} test queries, sigma=1.0)', fontsize=13,
    )
    plt.tight_layout()

    outdir = OUT_DIR / dataset
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / 'median_overshoot_vs_budget_pct.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")


def recovery_fraction(overshoot_array: np.ndarray, slack_grid: np.ndarray) -> np.ndarray:
    """fraction of overshoot_array <= s for each s in slack_grid -- the
    empirical CDF of relative overshoot among covered-but-missed queries,
    i.e. what fraction of today's coverable-but-uncertified queries would
    flip to certified under a CIG tolerance of `1 + s` instead of exact
    (sigma=1.0)."""
    if len(overshoot_array) == 0:
        return np.full(len(slack_grid), np.nan)
    return np.array([np.mean(overshoot_array <= s) for s in slack_grid])


RECOVERY_SLACK_GRID = np.arange(0.0, 1.05, 0.05)
RECOVERY_BUDGET_PCTS_TO_SHOW = [30.0, 65.0, 85.0]
RECOVERED_HIT_RATE_SLACK_LEVELS = [0.25, 0.75]


def plot_dataset_recovery_curves(
    dataset: str, data_by_bm_k: dict, test_n: int, k_values: list = None,
    budget_pcts_to_show: list = None,
) -> None:
    """rows = cache size, columns = K, one line per shown budget fraction:
    fraction of covered-but-missed queries recovered vs. relaxation slack
    s (CIG tolerance 1+s instead of exact). Direct quantification of what
    relaxed certification would buy at each budget level."""
    k_values = k_values if k_values is not None else K_VALUES
    budget_pcts_to_show = budget_pcts_to_show if budget_pcts_to_show is not None else RECOVERY_BUDGET_PCTS_TO_SHOW
    benchmarks = [bm for bm in data_by_bm_k if any(data_by_bm_k[bm].get(k) for k in k_values)]
    if not benchmarks:
        return
    benchmarks = sorted(benchmarks, key=lambda bm: parse_benchmark_sizes(bm)[0])

    n_rows, n_cols = len(benchmarks), len(k_values)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.8 * n_rows), squeeze=False)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(budget_pcts_to_show)))

    for r, bm in enumerate(benchmarks):
        train_n, _ = parse_benchmark_sizes(bm)
        for c, K in enumerate(k_values):
            ax = axes[r][c]
            d = data_by_bm_k[bm].get(K)
            if d is None or K == 0:
                ax.axis('off')
                continue
            for pct, color in zip(budget_pcts_to_show, colors):
                idx = np.argmin(np.abs(d['budget_pct'] - pct))
                frac = recovery_fraction(d['overshoot_arrays'][idx], RECOVERY_SLACK_GRID)
                ax.plot(RECOVERY_SLACK_GRID, frac, marker='o', markersize=2,
                         color=color, label=f'budget={d["budget_pct"][idx]:.0f}%')
            ax.set_ylim(-0.02, 1.02)

            if r == 0:
                ax.set_title(f'$K={K}$', fontsize=10)
            if c == 0:
                ax.set_ylabel(f'{train_n} Train Queries\nfraction recovered')
            if r == n_rows - 1:
                ax.set_xlabel('relaxation slack s (CIG tolerance = 1+s)')

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               fontsize=8, bbox_to_anchor=(0.5, -0.02 / n_rows))
    fig.suptitle(
        f'{dataset.upper()}: Fraction of Covered-but-Missed Queries Recovered vs. Relaxation Slack',
        fontsize=13,
    )
    plt.tight_layout()

    outdir = OUT_DIR / dataset
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / 'recovery_vs_slack.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")


def plot_dataset_recovered_hit_rate(
    dataset: str, data_by_bm_k: dict, test_n: int, k_values: list = None,
    slack_levels: list = None,
) -> None:
    """Extends the coverage-ceiling grid with recovered hit rate at 1-2
    slack levels: hit_rate + fraction of covered-and-missed queries with
    overshoot <= slack. Shows whether even generous relaxation closes the
    gap to the coverage ceiling."""
    k_values = k_values if k_values is not None else K_VALUES
    slack_levels = slack_levels if slack_levels is not None else RECOVERED_HIT_RATE_SLACK_LEVELS
    benchmarks = [bm for bm in data_by_bm_k if any(data_by_bm_k[bm].get(k) for k in k_values)]
    if not benchmarks:
        return
    benchmarks = sorted(benchmarks, key=lambda bm: parse_benchmark_sizes(bm)[0])

    n_rows, n_cols = len(benchmarks), len(k_values)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.8 * n_rows), squeeze=False)
    slack_colors = plt.cm.autumn(np.linspace(0.2, 0.8, len(slack_levels)))

    for r, bm in enumerate(benchmarks):
        train_n, _ = parse_benchmark_sizes(bm)
        for c, K in enumerate(k_values):
            ax = axes[r][c]
            d = data_by_bm_k[bm].get(K)
            if d is None or K == 0:
                ax.axis('off')
                continue
            ax.plot(d['budget_pct'], d['vector_coverage'], marker='o', markersize=3,
                     label='vector coverage (ceiling)', linestyle='--', color='tab:blue')
            ax.plot(d['budget_pct'], d['test_hit_curve'], marker='s', markersize=3,
                     label='actual test hit rate (exact)', linestyle='-', color='tab:orange')
            for slack, color in zip(slack_levels, slack_colors):
                # ov holds relative overshoot only for the covered-and-missed
                # subset (see compute_combo), so a recovered count from it
                # must be rescaled by len(ov)/test_n before adding to hit
                # (a fraction of *all* test queries) -- otherwise this
                # silently mixes a subset-fraction with a whole-population
                # fraction and can push "recovered" above the coverage
                # ceiling, which is impossible by construction (recovered
                # queries are always a subset of covered queries).
                recovered = np.array([
                    hit + np.sum(ov <= slack) / test_n if len(ov) > 0 else hit
                    for hit, ov in zip(d['test_hit_curve'], d['overshoot_arrays'])
                ])
                ax.plot(d['budget_pct'], recovered, marker='^', markersize=2,
                         linestyle=':', color=color, label=f'recovered hit rate (slack={slack})')
            ax.set_ylim(-0.02, 1.02)

            if r == 0:
                ax.set_title(f'$K={K}$', fontsize=10)
            if c == 0:
                ax.set_ylabel(f'{train_n} Train Queries\nfraction')
            if r == n_rows - 1:
                ax.set_xlabel('% of budget')

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               fontsize=8, bbox_to_anchor=(0.5, -0.03 / n_rows))
    fig.suptitle(
        f'{dataset.upper()}: Recovered Hit Rate Under Relaxed Certification vs. Budget '
        f'({test_n} test queries)', fontsize=13,
    )
    plt.tight_layout()

    outdir = OUT_DIR / dataset
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / 'recovered_hit_rate_vs_budget_pct.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['sift', 'esci'], default=None,
                         help='Default: both')
    parser.add_argument('--benchmarks', nargs='+', default=None,
                         help="Override this run's benchmark list "
                              "(default: DATASET_CONFIG's list)")
    parser.add_argument('--k-values', type=int, nargs='+', default=None,
                         dest='k_values',
                         help=f'Override K_VALUES (default: {K_VALUES})')
    parser.add_argument('--budget-fractions', type=float, nargs='+', default=None,
                         dest='budget_fractions',
                         help="Override run_population.py's 15-value BUDGET_FRACTIONS")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    k_values = args.k_values or K_VALUES
    budget_fractions = args.budget_fractions or BUDGET_FRACTIONS
    datasets = [args.dataset] if args.dataset else list(dc.DATASET_CONFIG.keys())

    summary = {}
    for dataset in datasets:
        cfg = dc.DATASET_CONFIG[dataset]
        print(f"\n=== {dataset.upper()} ===")
        summary[dataset] = {}
        data_by_bm_k = {}
        test_n = None
        benchmarks = args.benchmarks or cfg['benchmarks']
        for benchmark_name in benchmarks:
            bdir = cfg['base_dir'] / benchmark_name
            if not (bdir / 'cache_queries.npy').exists():
                print(f"  {benchmark_name}: not prepared, skipping")
                continue
            print(f"  {benchmark_name}")
            _, test_n = parse_benchmark_sizes(benchmark_name)
            summary[dataset][benchmark_name] = {}
            data_by_bm_k[benchmark_name] = {}

            base, cache_q, test_q, _bdir = dc.load_benchmark(dataset, benchmark_name)
            invariants = dc.precompute_ball_invariants_for_benchmark(base, cache_q, test_q)

            for K in k_values:
                result = compute_combo(
                    dataset, benchmark_name, K, invariants=invariants,
                    budget_fractions=budget_fractions,
                )
                data_by_bm_k[benchmark_name][K] = result
                if result is not None:
                    summary[dataset][benchmark_name][f'K={K}'] = {
                        'final_test_hit_rate': result['final_test_hit_rate'],
                        'final_vector_coverage': result['final_vector_coverage'],
                    }
        plot_dataset_coverage_ceiling(dataset, data_by_bm_k, test_n, k_values=k_values)
        plot_dataset_rho(dataset, data_by_bm_k, test_n, k_values=k_values)
        plot_dataset_overshoot(dataset, data_by_bm_k, test_n, k_values=k_values)
        plot_dataset_recovery_curves(dataset, data_by_bm_k, test_n, k_values=k_values)
        plot_dataset_recovered_hit_rate(dataset, data_by_bm_k, test_n, k_values=k_values)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. Summary: {OUT_DIR / 'summary.json'}")


if __name__ == '__main__':
    main()
