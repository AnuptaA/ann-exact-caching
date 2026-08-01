#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Trajectory diagnostics at budget=100% (Phase 3 Round 14, plan Step C).

Entirely separate from population_runner.py/selection.py: calls
propose_candidates_ball + select_candidates directly (via debug_common.py),
then computes hit-rate-vs-iteration in a single monotonic pass exploiting
CIG's monotonicity (see lemma1_no_union_sigma docstring / utils/base/
lemmas_sigma.py) instead of N repeated re-simulations -- once a query is
certified by some entry, it stays certified for the rest of the trajectory
since entries are never removed and lemma1_no_union checks each entry
independently.

A step is classified "deferred_flush" post-hoc, purely from existing
trajectory data: vectors_cached unchanged from the previous step while
num_entries increases (see utils/base/selection.py's _select_greedy_cost --
zero-marginal-cost candidates are collected during the main loop and
flushed for free at the end).
"""

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.base.population import _normalize
from simulations.population import debug_common as dc
from simulations.population.debug_common import parse_benchmark_sizes, _boxplot_grid, _density_grid

#-------------------------------------------------------------------------------

K_VALUES = [0, 1, 5, 10]
SIGMA = 1.0
OUT_DIR = Path('simulations/debug/population_trajectory')

#-------------------------------------------------------------------------------


def classify_phases(trajectory: list) -> list:
    """"deferred_flush" iff vectors_cached is unchanged from the previous
    step (every main-loop pick has strictly positive marginal cost by
    construction -- selection.py filters zero-marginal candidates into
    `deferred` before scoring)."""
    phases = []
    prev_vectors_cached = None
    for step in trajectory:
        vc = step['vectors_cached']
        phases.append('deferred_flush' if vc == prev_vectors_cached else 'greedy')
        prev_vectors_cached = vc
    return phases


def monotonic_hit_rate_pass(
    selected: list, queries: np.ndarray, r_t: np.ndarray, K: int, sigma: float, metric: str,
    dist_matrix: np.ndarray = None,
) -> np.ndarray:
    """certifying_step[i] = 1-indexed step (in `selected` order) at which
    query i first got certified, or -1 if never certified by the end.
    One pass through `selected`: each candidate is checked only against
    still-uncertified queries, exact (not sampled), by CIG monotonicity.

    Uses the coverage condition D(Q,t) + r_t <= r_c directly instead of
    gathering and counting R_c membership -- these are the exact same
    condition, not an approximation: if D(Q,t)+r_t<=r_c, every one of t's
    true K nearest neighbors (all within r_t of t) is within r_c of Q by
    the triangle inequality, so it's in R_c *and* within threshold of t --
    that's >= K matches. Conversely, if some K points inside R_c are
    within threshold of t, the full (unrestricted) base set has >= K
    points within threshold too, so t's true K-th-nearest distance is at
    most threshold, which rearranges to the same inequality. Both
    directions hold, so this replaces an O(rc_size) gather+count per step
    with an O(1) scalar comparison, using r_t (already loaded by
    compute_combo) instead of gathering entry.top_k_indices at all -- no
    to_cache_entry() call needed either, just cand.query/cand.r_c
    directly off the CandidateEntry.

    Verified against direct lemma1_no_union_sigma calls in
    tests/population/test_debug_trajectory.py, and broadly across real
    data this session (both datasets, multiple K/cache sizes): 0/N test-
    side mismatches in every combo tested.

    dist_matrix: optional precomputed (n_cache, n_queries) distance matrix
    (from utils/base/population.py's precompute_ball_invariants --
    train_dist_only or test_dist_only), indexed by each candidate's
    source_index. When provided, d_Q is looked up from this shared matrix
    instead of recomputed via dc.pairwise_dist -- the same matrix
    _propose_nested_spheres used to build r_c/covered_train_indices, so a
    self-referential candidate's own coverage check reads the literal same
    value at proposal and simulate time, closing the previously-documented
    train-side ULP mismatch to true zero. Default None recomputes d_Q
    fresh (unchanged behavior, e.g. for callers without an invariants dict).

    sigma=1.0 only: the equivalence proof above is for the exact case
    (threshold=K). It does not generalize to sigma<1.0 (threshold=
    ceil(sigma*K)<K), which would need the distance to the
    ceil(sigma*K)-th nearest neighbor, not r_t (the full K-th) -- a value
    not loaded anywhere today. Asserted rather than silently wrong.
    """
    assert sigma == 1.0, (
        f"monotonic_hit_rate_pass's scalar coverage check is only proven "
        f"exact at sigma=1.0 (got sigma={sigma}); see docstring"
    )
    n_queries = len(queries)
    r_t = r_t.astype(np.float64)

    if dist_matrix is None:
        query_hat = _normalize(queries.copy().astype(np.float64))

    certifying_step = np.full(n_queries, -1, dtype=np.int64)
    uncertified = np.arange(n_queries)

    for step_idx, cand in enumerate(selected, start=1):
        if len(uncertified) == 0:
            break

        if dist_matrix is not None:
            d_Q = dist_matrix[cand.source_index, uncertified].astype(np.float64)
        else:
            cand_q = cand.query.astype(np.float64)
            cand_q_hat = cand_q / np.linalg.norm(cand_q)
            d_Q = dc.pairwise_dist(query_hat[uncertified], cand_q_hat[None, :], metric)[:, 0]

        covered = (d_Q + r_t[uncertified]) <= cand.r_c
        newly_certified = uncertified[covered]

        if len(newly_certified):
            certifying_step[newly_certified] = step_idx
            uncertified = uncertified[~np.isin(uncertified, newly_certified)]

    return certifying_step


def hit_rate_curve(certifying_step: np.ndarray, n_steps: int) -> np.ndarray:
    """Cumulative hit rate at each step index (1..n_steps)."""
    n_queries = len(certifying_step)
    curve = np.zeros(n_steps, dtype=np.float64)
    hits_so_far = 0
    step_to_new_hits = np.zeros(n_steps + 1, dtype=np.int64)
    for s in certifying_step:
        if s > 0:
            step_to_new_hits[s] += 1
    for step in range(1, n_steps + 1):
        hits_so_far += step_to_new_hits[step]
        curve[step - 1] = hits_so_far / n_queries if n_queries > 0 else 0.0
    return curve


#-------------------------------------------------------------------------------


def compute_combo(dataset: str, benchmark_name: str, K: int, invariants: dict = None) -> dict:
    """All data needed to plot + summarize one (benchmark, K) combo, no
    plotting. num_entries is deliberately NOT included here -- it equals
    the 1-indexed step index exactly, always (every trajectory step, greedy
    or deferred-flush, appends exactly one CandidateEntry -- see _record()
    in utils/base/selection.py), so it carries no information beyond the
    x-axis itself and isn't worth plotting.

    invariants: optional precomputed K-independent proposal quantities
    (see debug_common.precompute_ball_invariants_for_benchmark), shared
    across the K-sweep for this benchmark. Default None recomputes them
    fresh for this call only, unchanged from before."""
    base, cache_q, test_q, bdir = dc.load_benchmark(dataset, benchmark_name)
    n_cache, n_test = len(cache_q), len(test_q)
    train_r_t, test_r_t = dc.load_gt_r_t(bdir, K, n_cache, n_test)

    selected, trajectory = dc.propose_and_select_full_budget(
        base, cache_q, test_q, train_r_t, test_r_t, K, invariants=invariants,
    )
    if not selected:
        print(f"    {dataset}/{benchmark_name} K={K}: no candidates selected, skipping")
        return None

    phases = classify_phases(trajectory)
    deferred_start = next((i for i, p in enumerate(phases) if p == 'deferred_flush'), None)

    n_steps = len(selected)

    test_dist_matrix = invariants['test_dist_only'] if invariants is not None else None
    train_dist_matrix = invariants['train_dist_only'] if invariants is not None else None
    test_cert = monotonic_hit_rate_pass(
        selected, test_q, test_r_t, K, SIGMA, dc.METRIC, dist_matrix=test_dist_matrix,
    )
    train_cert = monotonic_hit_rate_pass(
        selected, cache_q, train_r_t, K, SIGMA, dc.METRIC, dist_matrix=train_dist_matrix,
    )

    test_hit_curve = hit_rate_curve(test_cert, n_steps)
    train_hit_curve = hit_rate_curve(train_cert, n_steps)

    iters = np.arange(1, n_steps + 1)
    vectors_cached = np.array([t['vectors_cached'] for t in trajectory], dtype=np.float64)
    mem_util = vectors_cached / len(base)
    train_cov = np.array([t['cig_train_coverage'] for t in trajectory], dtype=np.float64)
    # test coverage isn't plotted -- at sigma=1.0 it's mathematically the
    # same quantity as test hit rate (both check the same CIG containment
    # condition, once at proposal time and once at simulate time), and
    # confirmed empirically to have zero divergence on test queries (unlike
    # train, where self-referential candidates cause real floating-point
    # boundary divergence). Not worth a redundant line on the plot.

    # benefit = newly covered train queries / marginal cost, the exact
    # score _select_greedy_cost ranks candidates by (selection.py's
    # `scores = newly / marginal`). Both quantities are already in
    # trajectory -- marginal cost per step is the vectors_cached delta
    # (0 for deferred_flush steps by construction, see classify_phases).
    new_train = np.array([t['new_train_coverage'] for t in trajectory], dtype=np.float64)
    delta_cost = np.diff(vectors_cached, prepend=0.0)
    deferred_mask = delta_cost <= 0
    benefit = np.where(delta_cost > 0, new_train / np.where(delta_cost > 0, delta_cost, 1.0), np.nan)

    # Actual radius/neighbor-rank each selected candidate was built at --
    # distinct from r_t (a per-query scalar independent of which neighbor
    # rank n was used): r_c = D(query, n-th neighbor) + D(neighbor, E_K),
    # so it depends on n (swept 1..n_max at proposal time, chosen freely by
    # greedy selection) as well as r_t. See utils/base/population.py's
    # _propose_nested_spheres.
    selected_r_c = np.array([c.r_c for c in selected], dtype=np.float64)
    selected_n = np.array([c.n for c in selected], dtype=np.float64)

    return {
        'iters': iters,
        'mem_util': mem_util,
        'train_cov': train_cov,
        'train_hit_curve': train_hit_curve,
        'test_hit_curve': test_hit_curve,
        'benefit': benefit,
        'deferred_mask': deferred_mask,
        'selected_r_c': selected_r_c,
        'selected_n': selected_n,
        'deferred_start': deferred_start,
        'n_steps': n_steps,
        'final_train_hit_rate': float(train_hit_curve[-1]) if n_steps else 0.0,
        'final_test_hit_rate': float(test_hit_curve[-1]) if n_steps else 0.0,
        'final_train_coverage': float(train_cov[-1]) if n_steps else 0.0,
        'final_memory_utilization': float(mem_util[-1]) if n_steps else 0.0,
    }


def plot_dataset_trajectory(dataset: str, data_by_bm_k: dict, test_n: int) -> None:
    """One grid figure per dataset: rows = cache size (train query count),
    columns = K, matching the sigma x K_test grid shape used throughout
    analyze_population_proposal.py. Each cell shows memory utilization /
    train coverage / train hit rate / test hit rate vs iteration, with the
    deferred-flush start marked. test coverage is intentionally omitted
    (see compute_combo's docstring -- mathematically identical to test hit
    rate at sigma=1.0)."""
    benchmarks = [bm for bm in data_by_bm_k if any(data_by_bm_k[bm].get(k) for k in K_VALUES)]
    if not benchmarks:
        return
    benchmarks = sorted(benchmarks, key=lambda bm: parse_benchmark_sizes(bm)[0])

    n_rows, n_cols = len(benchmarks), len(K_VALUES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.8 * n_rows), squeeze=False)

    for r, bm in enumerate(benchmarks):
        train_n, _ = parse_benchmark_sizes(bm)
        for c, K in enumerate(K_VALUES):
            ax = axes[r][c]
            d = data_by_bm_k[bm].get(K)
            if d is None:
                ax.axis('off')
                continue
            ax.plot(d['iters'], d['mem_util'], label='memory utilization')
            ax.plot(d['iters'], d['train_cov'], label='train coverage')
            ax.plot(d['iters'], d['train_hit_curve'], label='train hit rate', linestyle='--')
            ax.plot(d['iters'], d['test_hit_curve'], label='test hit rate', linestyle='--')
            if d['deferred_start'] is not None:
                ax.axvline(d['deferred_start'] + 1, color='gray', linestyle=':',
                           label='deferred_flush start')
            if r == 0:
                ax.set_title(f'$K={K}$', fontsize=10)
            if c == 0:
                ax.set_ylabel(f'{train_n} Train Queries\nfraction')
            if r == n_rows - 1:
                ax.set_xlabel('iteration (selection order)')

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               fontsize=8, bbox_to_anchor=(0.5, -0.02 / n_rows))
    fig.suptitle(f'{dataset.upper()}: Trajectory vs. Iteration ({test_n} test queries)', fontsize=13)
    plt.tight_layout()

    outdir = OUT_DIR / dataset
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / 'trajectory.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")


def plot_dataset_benefit(dataset: str, data_by_bm_k: dict, test_n: int) -> None:
    """Same grid layout as plot_dataset_trajectory (rows = cache size,
    columns = K) but plots benefit (newly covered train queries / marginal
    cost) vs iteration -- the exact score _select_greedy_cost ranks
    candidates by (utils/base/selection.py's `scores = newly / marginal`).
    Log-scale y-axis: the ratio spans orders of magnitude (large early,
    when a candidate's own dense neighborhood is mostly uncovered; near-
    zero once most queries are already certified).

    deferred_flush steps have zero marginal cost by construction, so their
    benefit (newly / 0) is unboundedly large, not zero -- plotted as
    triangle markers pinned just above the panel's own max finite benefit
    (a sentinel, not a real ratio) rather than left as a gap, so the
    x-range still reflects every step even when most of a combo's steps
    are deferred (previously made the line look truncated at 1-2
    iterations when in fact thousands of further, free picks followed)."""
    benchmarks = [bm for bm in data_by_bm_k if any(data_by_bm_k[bm].get(k) for k in K_VALUES)]
    if not benchmarks:
        return
    benchmarks = sorted(benchmarks, key=lambda bm: parse_benchmark_sizes(bm)[0])

    n_rows, n_cols = len(benchmarks), len(K_VALUES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.8 * n_rows), squeeze=False)

    for r, bm in enumerate(benchmarks):
        train_n, _ = parse_benchmark_sizes(bm)
        for c, K in enumerate(K_VALUES):
            ax = axes[r][c]
            d = data_by_bm_k[bm].get(K)
            if d is None:
                ax.axis('off')
                continue
            ax.plot(d['iters'], d['benefit'], label='benefit (newly covered / cost)')
            ax.set_yscale('log')

            finite_benefit = d['benefit'][~np.isnan(d['benefit'])]
            sentinel = float(finite_benefit.max()) * 1.5 if len(finite_benefit) else 1.0
            deferred_iters = d['iters'][d['deferred_mask']]
            if len(deferred_iters):
                ax.scatter(deferred_iters, np.full(len(deferred_iters), sentinel),
                           marker='^', s=10, color='tab:red', alpha=0.5,
                           label='deferred (zero-cost, unbounded benefit)')

            if d['deferred_start'] is not None:
                ax.axvline(d['deferred_start'] + 1, color='gray', linestyle=':',
                           label='deferred_flush start')
            if r == 0:
                ax.set_title(f'$K={K}$', fontsize=10)
            if c == 0:
                ax.set_ylabel(f'{train_n} Train Queries\nbenefit')
            if r == n_rows - 1:
                ax.set_xlabel('iteration (selection order)')

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               fontsize=8, bbox_to_anchor=(0.5, -0.02 / n_rows))
    fig.suptitle(f'{dataset.upper()}: Marginal Benefit vs. Iteration ({test_n} test queries)', fontsize=13)
    plt.tight_layout()

    outdir = OUT_DIR / dataset
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / 'benefit_vs_iteration.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")


def compute_budget_pct_view(combo_result: dict) -> dict:
    """Re-derive train/test hit rate as a function of % of budget (1%-100%,
    1% steps) from the single budget=100% trajectory already computed by
    compute_combo -- no re-simulation at intermediate budgets needed,
    since mem_util already tracks cumulative memory utilization
    monotonically across the full run (vectors_cached only grows).

    For each 1% target, we want the *best* (most-converged) result
    achievable using at most that much memory -- the *last* step whose
    mem_util is <= the target, not the first step that reaches it. Many
    real combos saturate mem_util to 1.0 within the first step or two
    (a single early candidate can cover the whole base) and then spend
    thousands of further steps on deferred_flush picks that add no memory
    but keep certifying more queries -- looking up the *first* crossing
    would report step 1's hit rate at every budget_pct from 1% to 100%,
    silently discarding that entire free tail (this was a real bug, not
    just a design choice: it made this plot disagree with
    final_train_hit_rate/final_test_hit_rate at budget_pct=100).

    `searchsorted(..., side='right') - 1` gives the last index with
    mem_util <= target. If that's negative (even the cheapest candidate
    costs more than the target budget), no candidate is affordable at
    that budget, so hit rate is genuinely 0 -- not step 1's value.
    """
    mem_util = combo_result['mem_util']
    n_steps = combo_result['n_steps']
    budget_pct = np.arange(1, 101)

    if n_steps == 0:
        nan_arr = np.full(100, np.nan)
        return {
            'budget_pct': budget_pct,
            'train_hit_curve': nan_arr,
            'test_hit_curve': nan_arr,
        }

    targets = budget_pct / 100.0
    idx = np.searchsorted(mem_util, targets, side='right') - 1
    affordable = idx >= 0
    safe_idx = np.where(affordable, idx, 0)

    train_hit_curve = np.where(affordable, combo_result['train_hit_curve'][safe_idx], 0.0)
    test_hit_curve = np.where(affordable, combo_result['test_hit_curve'][safe_idx], 0.0)

    return {
        'budget_pct': budget_pct,
        'train_hit_curve': train_hit_curve,
        'test_hit_curve': test_hit_curve,
    }


def plot_dataset_budget_pct(dataset: str, data_by_bm_k: dict, test_n: int) -> None:
    """Same grid layout as plot_dataset_trajectory/plot_dataset_benefit
    (rows = cache size, columns = K), but x-axis = % of budget (0-100%)
    instead of raw iteration -- re-derived from the same budget=100% run
    via compute_budget_pct_view, not a separate simulation at each
    budget. Plots only train hit rate and test hit rate (0-100% y-axis)."""
    benchmarks = [bm for bm in data_by_bm_k if any(data_by_bm_k[bm].get(k) for k in K_VALUES)]
    if not benchmarks:
        return
    benchmarks = sorted(benchmarks, key=lambda bm: parse_benchmark_sizes(bm)[0])

    n_rows, n_cols = len(benchmarks), len(K_VALUES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.8 * n_rows), squeeze=False)

    for r, bm in enumerate(benchmarks):
        train_n, _ = parse_benchmark_sizes(bm)
        for c, K in enumerate(K_VALUES):
            ax = axes[r][c]
            d = data_by_bm_k[bm].get(K)
            if d is None:
                ax.axis('off')
                continue
            view = compute_budget_pct_view(d)
            ax.plot(view['budget_pct'], view['train_hit_curve'], label='train hit rate', linestyle='--')
            ax.plot(view['budget_pct'], view['test_hit_curve'], label='test hit rate', linestyle='--')
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
    fig.suptitle(f'{dataset.upper()}: Metrics vs. % of Budget ({test_n} test queries)', fontsize=13)
    plt.tight_layout()

    outdir = OUT_DIR / dataset
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / 'budget_pct_vs_metrics.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")


def _selected_panels(dataset: str, data_by_bm_k: dict, field: str) -> tuple:
    """Shared setup for plot_dataset_selected_radius/plot_dataset_selected_n_density:
    (benchmarks, test_n_seen, panels) where panels is a list of
    (data_by_label, title, ylabel_or_xlabel) tuples -- one per cache size,
    K values as the grouped label within each panel, matching
    debug_ball_geometry.py's margin_vs_k.png/rt_vs_k.png convention."""
    benchmarks = [bm for bm in data_by_bm_k if any(data_by_bm_k[bm].get(k) for k in K_VALUES)]
    benchmarks = sorted(benchmarks, key=lambda bm: parse_benchmark_sizes(bm)[0])
    test_n_seen = None
    panels = []
    for bm in benchmarks:
        train_n, test_n = parse_benchmark_sizes(bm)
        test_n_seen = test_n_seen or test_n
        data_by_label = {}
        for K in K_VALUES:
            d = data_by_bm_k[bm].get(K)
            if d is not None and len(d[field]):
                data_by_label[f'$K={K}$'] = d[field]
        panels.append((data_by_label, f'{train_n} Train Queries', None))
    return benchmarks, test_n_seen, panels


def plot_dataset_selected_radius(dataset: str, data_by_bm_k: dict) -> None:
    """Boxplot grid (one panel per cache size, K as the grouped x-axis) of
    the actual radius (r_c) each SELECTED candidate was built at --
    distinct from rt_vs_k.png's r_t (a per-query scalar independent of
    which neighbor rank n a candidate used). Log-scale y-axis, same
    convention as margin_vs_k.png/rt_vs_k.png."""
    _, test_n_seen, panels = _selected_panels(dataset, data_by_bm_k, 'selected_r_c')
    panels = [(d, title, 'selected candidate radius ($r_c$)', True) for d, title, _ in panels]
    if not any(d for d, _, _, _ in panels):
        return
    test_note = f' ({test_n_seen} test queries)' if test_n_seen else ''
    _boxplot_grid(
        panels,
        f'{dataset.upper()}: Selected Candidate Radius vs. $K${test_note}',
        OUT_DIR / dataset / 'selected_radius_vs_k.png',
    )


def plot_dataset_selected_n_density(dataset: str, data_by_bm_k: dict) -> None:
    """Density grid (one panel per cache size, K as overlapping KDE curves)
    of n (neighbor rank, 1..n_max) each SELECTED candidate's sphere was
    built from. Combos where budget saturates in 1-2 steps (see
    plot_dataset_benefit's docstring) will have very few selected
    candidates and a correspondingly sparse/degenerate curve for those
    panels -- expected, not a bug."""
    _, test_n_seen, panels = _selected_panels(dataset, data_by_bm_k, 'selected_n')
    panels = [(d, title, 'n (neighbor rank)') for d, title, _ in panels]
    if not any(d for d, _, _ in panels):
        return
    test_note = f' ({test_n_seen} test queries)' if test_n_seen else ''
    _density_grid(
        panels,
        f'{dataset.upper()}: Selected Candidate n Density vs. $K${test_note}',
        OUT_DIR / dataset / 'selected_n_density_vs_k.png',
    )


def main() -> None:
    summary = {}
    for dataset, cfg in dc.DATASET_CONFIG.items():
        print(f"\n=== {dataset.upper()} ===")
        summary[dataset] = {}
        data_by_bm_k = {}
        test_n = None
        for benchmark_name in cfg['benchmarks']:
            bdir = cfg['base_dir'] / benchmark_name
            if not (bdir / 'cache_queries.npy').exists():
                print(f"  {benchmark_name}: not prepared, skipping")
                continue
            print(f"  {benchmark_name}")
            _, test_n = parse_benchmark_sizes(benchmark_name)
            summary[dataset][benchmark_name] = {}
            data_by_bm_k[benchmark_name] = {}

            # K-independent proposal invariants computed once per benchmark
            # and reused across the whole K-sweep below, instead of every
            # compute_combo call redoing the same O(n_cache * n_base) work
            # -- see utils/base/population.py's precompute_ball_invariants.
            base, cache_q, test_q, _bdir = dc.load_benchmark(dataset, benchmark_name)
            invariants = dc.precompute_ball_invariants_for_benchmark(base, cache_q, test_q)

            for K in K_VALUES:
                result = compute_combo(dataset, benchmark_name, K, invariants=invariants)
                data_by_bm_k[benchmark_name][K] = result
                if result is not None:
                    summary[dataset][benchmark_name][f'K={K}'] = {
                        k: v for k, v in result.items()
                        if k not in ('iters', 'mem_util', 'train_cov',
                                     'train_hit_curve', 'test_hit_curve', 'benefit',
                                     'deferred_mask', 'selected_r_c', 'selected_n')
                    }
        plot_dataset_trajectory(dataset, data_by_bm_k, test_n)
        plot_dataset_benefit(dataset, data_by_bm_k, test_n)
        plot_dataset_budget_pct(dataset, data_by_bm_k, test_n)
        plot_dataset_selected_radius(dataset, data_by_bm_k)
        plot_dataset_selected_n_density(dataset, data_by_bm_k)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. Summary: {OUT_DIR / 'summary.json'}")


if __name__ == '__main__':
    main()
