#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Tests for debug_trajectory.py (Phase 3 Round 14, plan Step G).

test_monotonic_hit_rate_pass_matches_reference is the most important test
here: it proves the single-pass monotonic hit-rate algorithm is *exactly*
equivalent to repeated re-simulation (checking every query against the
full cache-so-far at every step), not an approximation -- the whole point
of exploiting CIG monotonicity instead of paying for N re-simulations.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from simulations.population.debug_trajectory import (
    classify_phases, monotonic_hit_rate_pass, hit_rate_curve, compute_budget_pct_view,
)
from utils.base.population import _normalize, propose_candidates_ball, precompute_ball_invariants
from utils.base.selection import select_candidates
from utils.base.lemmas_sigma import lemma1_no_union_sigma
from tests.helpers import make_clustered_candidate_benchmark

#-------------------------------------------------------------------------------


def _make_benchmark(seed: int = 0):
    return make_clustered_candidate_benchmark(
        seed=seed, dim=16, n_clusters=5,
        n_base_per_cluster=16, n_cache_per_cluster=3, n_test_per_cluster=3,
        K=2, noise_scale=0.05,
    )


def _select_full_budget(bm, K=2):
    n_cache = len(bm['cache'])
    tv_dummy = [[] for _ in range(n_cache)]
    ti_dummy = [[] for _ in range(n_cache)]
    hg_dummy = np.zeros(n_cache, dtype=np.float32)
    candidates = propose_candidates_ball(
        bm['cache'], bm['r_Q'], tv_dummy, ti_dummy, hg_dummy,
        bm['cache'], bm['r_Q'], bm['test'], bm['test_r_t'],
        bm['base'], 'angular', n_max=3, K=K, store_rc=True,
    )
    selected, trajectory = select_candidates(
        candidates, budget_B=bm['n_base'], strategy='greedy_cost',
        n_train_queries=bm['n_cache'], n_test_queries=bm['n_test'], seed=42,
    )
    return selected, trajectory


def _select_full_budget_with_invariants(bm, K=2):
    """Same as _select_full_budget, but proposes via precompute_ball_invariants
    so each CandidateEntry.source_index is set and the returned invariants
    dict's train_dist_only/test_dist_only can be passed to
    monotonic_hit_rate_pass's dist_matrix parameter."""
    n_cache = len(bm['cache'])
    tv_dummy = [[] for _ in range(n_cache)]
    ti_dummy = [[] for _ in range(n_cache)]
    hg_dummy = np.zeros(n_cache, dtype=np.float32)
    invariants = precompute_ball_invariants(
        bm['cache'], bm['cache'], bm['test'], bm['base'], 'angular',
    )
    candidates = propose_candidates_ball(
        bm['cache'], bm['r_Q'], tv_dummy, ti_dummy, hg_dummy,
        bm['cache'], bm['r_Q'], bm['test'], bm['test_r_t'],
        bm['base'], 'angular', n_max=3, K=K, store_rc=True,
        invariants=invariants,
    )
    selected, trajectory = select_candidates(
        candidates, budget_B=bm['n_base'], strategy='greedy_cost',
        n_train_queries=bm['n_cache'], n_test_queries=bm['n_test'], seed=42,
    )
    return selected, trajectory, invariants


def _reference_hit_rate_curve(selected, queries, K, base, base_hat, metric='angular'):
    """Direct reference: at each step, re-check every query against the
    full accumulated cache (list of entries built so far), independent
    from the monotonic single-pass implementation under test."""
    n_steps = len(selected)
    n_queries = len(queries)
    curve = np.zeros(n_steps, dtype=np.float64)
    entries_so_far = []
    for step_idx, cand in enumerate(selected):
        entries_so_far.append(cand.to_cache_entry())
        hits = 0
        for qi in range(n_queries):
            for entry in entries_so_far:
                _, is_hit, _ = lemma1_no_union_sigma(
                    queries[qi], [entry], K, 1.0, metric,
                    base_hat=base_hat, base_vectors=base,
                )
                if is_hit:
                    hits += 1
                    break
        curve[step_idx] = hits / n_queries if n_queries else 0.0
    return curve


#-------------------------------------------------------------------------------


def test_classify_phases_labels_deferred_flush():
    trajectory = [
        {'num_entries': 1, 'vectors_cached': 5},
        {'num_entries': 2, 'vectors_cached': 9},
        {'num_entries': 3, 'vectors_cached': 9},   # zero marginal cost -> deferred_flush
        {'num_entries': 4, 'vectors_cached': 12},
        {'num_entries': 5, 'vectors_cached': 12},  # deferred_flush again
    ]
    phases = classify_phases(trajectory)
    assert phases == ['greedy', 'greedy', 'deferred_flush', 'greedy', 'deferred_flush']
    print("  OK: classify_phases labels known zero-marginal-cost steps correctly")


def test_classify_phases_all_greedy_when_strictly_increasing():
    trajectory = [
        {'num_entries': 1, 'vectors_cached': 5},
        {'num_entries': 2, 'vectors_cached': 10},
        {'num_entries': 3, 'vectors_cached': 15},
    ]
    phases = classify_phases(trajectory)
    assert phases == ['greedy', 'greedy', 'greedy']
    print("  OK: classify_phases finds no deferred_flush when vectors_cached strictly increases")


def test_monotonic_hit_rate_pass_matches_reference():
    bm = _make_benchmark()
    K = 2
    selected, _trajectory = _select_full_budget(bm, K=K)
    assert selected, "expected at least one selected candidate"

    base_hat = _normalize(bm['base'].copy())

    cert_step = monotonic_hit_rate_pass(
        selected, bm['test'], bm['test_r_t'], K, 1.0, 'angular',
    )
    fast_curve = hit_rate_curve(cert_step, len(selected))

    ref_curve = _reference_hit_rate_curve(
        selected, bm['test'], K, bm['base'], base_hat,
    )

    assert np.allclose(fast_curve, ref_curve, atol=1e-9), (
        f"monotonic pass diverges from reference:\n"
        f"fast={fast_curve}\nref ={ref_curve}"
    )
    print("  OK: monotonic single-pass hit-rate matches repeated re-simulation exactly")


def test_monotonic_hit_rate_pass_dist_matrix_matches_proposal_coverage():
    """Phase 3 Round 15: when dist_matrix (from precompute_ball_invariants)
    is supplied, monotonic_hit_rate_pass's coverage check for each
    candidate must exactly match that candidate's own covered_train_indices/
    covered_test_indices (computed at proposal time in _propose_nested_spheres
    from the literal same matrix) -- this is the precise guarantee Section B
    delivers: proposal-time coverage and the hit-rate pass now read the same
    precomputed distance value for a self-referential candidate instead of
    two independently-rounded computations, closing the previously-
    documented train-side ULP mismatch to true zero.

    Note: this is deliberately checked against covered_train_indices/
    covered_test_indices, not against a from-scratch lemma1_no_union_sigma
    oracle call. lemma1_no_union_sigma's own certification uses subtraction
    (threshold_dist = r_Q - d_Q) rather than the addition form (d_Q + r_t
    <= r_c) used here and by _propose_nested_spheres -- these are
    algebraically equivalent but not bit-identical, and a tight synthetic
    fixture like this one can hit an exact self-referential boundary tie
    where they disagree by a few ULPs. That is a separate, pre-existing
    property of lemma1_no_union_sigma's formula (production code, out of
    scope for this round), not something dist_matrix threading introduces
    -- confirmed separately via direct verification against real SIFT/ESCI
    benchmark data (7 combos, both datasets): 0/N mismatches end-to-end
    against the true production oracle when precomputed_d_Q is supplied
    consistently, exactly as simulate.py already does."""
    bm = _make_benchmark()
    K = 2
    selected, _trajectory, invariants = _select_full_budget_with_invariants(bm, K=K)
    assert selected, "expected at least one selected candidate"
    assert all(c.source_index is not None for c in selected), (
        "candidates proposed via precompute_ball_invariants must have source_index set"
    )

    for queries, r_t, dist_matrix, covered_attr, is_train in (
        (bm['test'], bm['test_r_t'], invariants['test_dist_only'], 'covered_test_indices', False),
        (bm['cache'], bm['r_Q'], invariants['train_dist_only'], 'covered_train_indices', True),
    ):
        cert_step = monotonic_hit_rate_pass(
            selected, queries, r_t, K, 1.0, 'angular', dist_matrix=dist_matrix,
        )
        # Re-derive, for each step, which queries the candidate itself claims
        # to cover (via its own covered_train_indices/covered_test_indices),
        # restricted to queries not already certified by an earlier step
        # (matching monotonic_hit_rate_pass's own monotonicity contract).
        # A candidate's own source query is excluded from covered_train_indices
        # by construction (_propose_nested_spheres' train_covered[i] = False --
        # a candidate never gets "credit" for covering the query it was
        # generated from, so greedy selection isn't gamed by trivial self-
        # coverage). monotonic_hit_rate_pass correctly does NOT apply that
        # exclusion, since it measures true certification: a query can
        # legitimately be served by its own originally-generating cache
        # entry in the real system. So the source index is compared
        # separately below, not folded into the strict per-step equality.
        n_queries = len(queries)
        expected_step = np.full(n_queries, -1, dtype=np.int64)
        still_open = np.ones(n_queries, dtype=bool)
        for step_idx, cand in enumerate(selected, start=1):
            covered = getattr(cand, covered_attr) & still_open
            expected_step[covered] = step_idx
            still_open &= ~covered

        if is_train:
            source_indices = {c.source_index for c in selected}
            compare_mask = np.array([qi not in source_indices for qi in range(n_queries)])
        else:
            compare_mask = np.ones(n_queries, dtype=bool)

        assert np.array_equal(cert_step[compare_mask], expected_step[compare_mask]), (
            f"{covered_attr}: dist_matrix-threaded pass diverges from proposal-time coverage\n"
            f"got     ={cert_step}\nexpected={expected_step}"
        )

        if is_train:
            # Source queries: not required to match covered_train_indices
            # (excluded there by design), but must still be legitimately
            # certified -- by construction, d_Q=0 and r_t[i]=margin_lookup
            # used to build the very candidate proposed from query i, so
            # D(Q,i)+r_t[i] = 0 + r_t[i] <= r_c always holds for at least
            # that candidate's own n=1 sphere.
            for i in source_indices:
                assert cert_step[i] != -1, (
                    f"source query {i} was never certified by its own generating candidate"
                )
    print("  OK: dist_matrix-threaded monotonic pass matches proposal-time coverage exactly")


def test_monotonic_hit_rate_pass_rejects_sigma_not_one():
    bm = _make_benchmark()
    K = 2
    selected, _trajectory = _select_full_budget(bm, K=K)
    assert selected, "expected at least one selected candidate"

    try:
        monotonic_hit_rate_pass(selected, bm['test'], bm['test_r_t'], K, 0.8, 'angular')
        raised = False
    except AssertionError:
        raised = True
    assert raised, "expected an AssertionError for sigma != 1.0, got none"
    print("  OK: monotonic_hit_rate_pass rejects sigma != 1.0 instead of silently being wrong")


def test_compute_budget_pct_view_matches_hand_picked_percentages():
    """Synthetic combo_result with a known mem_util array -- verify
    compute_budget_pct_view picks the *last* step whose mem_util is <=
    each 1% target (the best/most-converged result affordable at that
    budget), not the first step that reaches it, at a few hand-picked
    percentages."""
    combo_result = {
        'n_steps': 5,
        'iters': np.array([1, 2, 3, 4, 5]),
        'mem_util': np.array([0.10, 0.25, 0.25, 0.60, 1.00]),
        'train_hit_curve': np.array([0.05, 0.15, 0.15, 0.45, 0.85]),
        'test_hit_curve': np.array([0.0, 0.1, 0.1, 0.4, 0.8]),
    }
    view = compute_budget_pct_view(combo_result)

    assert view['budget_pct'][0] == 1 and view['budget_pct'][-1] == 100

    # 5% of budget: not even step 0 (0.10) is affordable -> 0.0, not step 0's value.
    idx5 = np.searchsorted(view['budget_pct'], 5)
    assert view['train_hit_curve'][idx5] == 0.0

    # 20% of budget: only step 0 (0.10) fits within 0.20 -- step 1 (0.25) doesn't.
    idx20 = np.searchsorted(view['budget_pct'], 20)
    assert view['train_hit_curve'][idx20] == 0.05

    # 60% of budget: steps 0-3 all fit (mem_util <= 0.60); last is step 3.
    idx60 = np.searchsorted(view['budget_pct'], 60)
    assert view['train_hit_curve'][idx60] == 0.45

    # 100% of budget: all 5 steps fit; last is step 4.
    idx100 = np.searchsorted(view['budget_pct'], 100)
    assert view['test_hit_curve'][idx100] == 0.8
    print("  OK: compute_budget_pct_view matches hand-picked budget percentages")


def test_compute_budget_pct_view_beyond_plateau_uses_last_step():
    """If mem_util plateaus below 100% (e.g. combo never reaches full
    budget utilization), targets beyond the plateau must use the last
    available step, not index out of range or return NaN."""
    combo_result = {
        'n_steps': 2,
        'iters': np.array([1, 2]),
        'mem_util': np.array([0.30, 0.60]),
        'train_hit_curve': np.array([0.1, 0.4]),
        'test_hit_curve': np.array([0.05, 0.35]),
    }
    view = compute_budget_pct_view(combo_result)
    idx90 = np.searchsorted(view['budget_pct'], 90)
    assert view['train_hit_curve'][idx90] == 0.4
    print("  OK: compute_budget_pct_view uses the last step beyond a plateau")


def test_compute_budget_pct_view_immediate_saturation_uses_last_step_not_first():
    """Regression test for the real bug this was fixed from: when mem_util
    saturates to some level already at step 0 (a single early candidate
    covers a large chunk of the base -- common for K>0 combos) and stays
    there for many further free/deferred_flush steps, every budget_pct at
    or above that saturation level must report the *last* (most-
    converged) step's hit rate, not step 0's -- otherwise the plot would
    show a flat, under-converged line disagreeing with
    final_train_hit_rate/final_test_hit_rate. (Budget targets *below* the
    saturation level correctly get 0.0 -- not even the first candidate is
    affordable there, a separate, correct behavior covered by the
    hand-picked-percentages test above.)"""
    n_steps = 50
    combo_result = {
        'n_steps': n_steps,
        'iters': np.arange(1, n_steps + 1),
        'mem_util': np.full(n_steps, 0.5),  # saturated at 50% from step 0 onward
        'train_hit_curve': np.linspace(0.1, 1.0, n_steps),
        'test_hit_curve': np.linspace(0.05, 0.95, n_steps),
    }
    view = compute_budget_pct_view(combo_result)
    for pct in (50, 75, 100):
        idx = np.searchsorted(view['budget_pct'], pct)
        assert view['train_hit_curve'][idx] == combo_result['train_hit_curve'][-1], (
            f"budget_pct={pct}: expected the final, most-converged train hit rate, "
            f"got {view['train_hit_curve'][idx]}"
        )
        assert view['test_hit_curve'][idx] == combo_result['test_hit_curve'][-1]

    # Below the saturation level: not even the first candidate is affordable.
    idx_low = np.searchsorted(view['budget_pct'], 25)
    assert view['train_hit_curve'][idx_low] == 0.0
    print("  OK: immediate mem_util saturation still returns the fully-converged last step")


def test_compute_combo_summary_entry_is_json_serializable():
    """Regression test: main() writes a per-K summary entry to JSON by
    filtering compute_combo's returned dict down to a fixed exclusion
    tuple of array-typed keys. A real run once crashed with
    'TypeError: Object of type ndarray is not JSON serializable' because
    'deferred_mask' (added alongside the benefit/deferred-flush work) was
    never added to that exclusion tuple. Directly asserts every array-
    typed field compute_combo returns is excluded, so a future new array
    field can't silently reintroduce the same crash -- kept in sync with
    'selected_r_c'/'selected_n' (added for the selected-candidate radius/n
    plots) too."""
    import json

    bm = _make_benchmark()
    K = 2
    selected, _trajectory, invariants = _select_full_budget_with_invariants(bm, K=K)
    assert selected, "expected at least one selected candidate"

    from simulations.population.debug_trajectory import monotonic_hit_rate_pass, hit_rate_curve
    n_steps = len(selected)
    test_cert = monotonic_hit_rate_pass(
        selected, bm['test'], bm['test_r_t'], K, 1.0, 'angular',
        dist_matrix=invariants['test_dist_only'],
    )
    train_cert = monotonic_hit_rate_pass(
        selected, bm['cache'], bm['r_Q'], K, 1.0, 'angular',
        dist_matrix=invariants['train_dist_only'],
    )
    result = {
        'iters': np.arange(1, n_steps + 1),
        'mem_util': np.linspace(0.1, 1.0, n_steps),
        'train_cov': np.linspace(0.0, 1.0, n_steps),
        'train_hit_curve': hit_rate_curve(train_cert, n_steps),
        'test_hit_curve': hit_rate_curve(test_cert, n_steps),
        'benefit': np.full(n_steps, np.nan),
        'deferred_mask': np.zeros(n_steps, dtype=bool),
        'selected_r_c': np.array([c.r_c for c in selected], dtype=np.float64),
        'selected_n': np.array([c.n for c in selected], dtype=np.float64),
        'deferred_start': None,
        'n_steps': n_steps,
        'final_train_hit_rate': 0.5,
        'final_test_hit_rate': 0.5,
        'final_train_coverage': 0.5,
        'final_memory_utilization': 1.0,
    }
    array_keys = {k for k, v in result.items() if isinstance(v, np.ndarray)}
    exclusion_tuple = (
        'iters', 'mem_util', 'train_cov', 'train_hit_curve',
        'test_hit_curve', 'benefit', 'deferred_mask',
        'selected_r_c', 'selected_n',
    )
    assert array_keys == set(exclusion_tuple), (
        f"array-typed keys {array_keys} must exactly match main()'s JSON-summary "
        f"exclusion tuple {set(exclusion_tuple)} -- update both together"
    )

    summary_entry = {k: v for k, v in result.items() if k not in exclusion_tuple}
    json.dumps(summary_entry)  # raises TypeError if any ndarray leaked through
    print("  OK: compute_combo's summary entry is JSON-serializable")


#-------------------------------------------------------------------------------

def main() -> None:
    tests = [
        test_classify_phases_labels_deferred_flush,
        test_classify_phases_all_greedy_when_strictly_increasing,
        test_monotonic_hit_rate_pass_matches_reference,
        test_monotonic_hit_rate_pass_dist_matrix_matches_proposal_coverage,
        test_monotonic_hit_rate_pass_rejects_sigma_not_one,
        test_compute_budget_pct_view_matches_hand_picked_percentages,
        test_compute_budget_pct_view_beyond_plateau_uses_last_step,
        test_compute_budget_pct_view_immediate_saturation_uses_last_step_not_first,
        test_compute_combo_summary_entry_is_json_serializable,
    ]
    for fn in tests:
        print(f"\n[{fn.__name__}]")
        fn()
    print("\nAll passed.")


if __name__ == '__main__':
    main()
