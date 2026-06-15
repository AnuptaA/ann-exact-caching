#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from utils.base.population import CandidateEntry
from utils.base.selection import select_candidates

#-------------------------------------------------------------------------------

def _make_candidate(
    query_dim: int = 4,
    R_c: set = None,
    covered: set = None,
    r_Q: float = 0.2,
    alpha: float = 1.0,
    seed: int = 0,
) -> CandidateEntry:
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(query_dim).astype(np.float32)
    q /= np.linalg.norm(q)
    if R_c is None:
        R_c = set()
    if covered is None:
        covered = set()
    return CandidateEntry(
        query=q,
        r_Q=r_Q,
        alpha=alpha,
        r_c=alpha * r_Q,
        R_c=R_c,
        rc_size=len(R_c),
        covered_train_indices=covered,
        covered_test_indices=covered,
        top_k_vectors=[q],
        top_k_indices=[0],
        half_gap=0.01,
    )

def _build_candidates(n_test: int = 20) -> list:
    """Build a fixed small candidate set with predictable coverage."""
    candidates = [
        # covers 10 test queries, needs 5 vectors
        _make_candidate(R_c=set(range(5)), covered=set(range(10)), seed=0),
        # covers 6 test queries, needs 3 vectors
        _make_candidate(R_c=set(range(3)), covered=set(range(6)), seed=1),
        # covers 4 test queries, needs 8 vectors
        _make_candidate(R_c=set(range(8)), covered=set(range(4)), seed=2),
        # covers 2 test queries, needs 1 vector
        _make_candidate(R_c={10}, covered={15, 16}, seed=3),
        # covers 5 test queries, overlaps with first, needs 2 new vectors
        _make_candidate(R_c={5, 6}, covered=set(range(5, 10)), seed=4),
        # zero coverage: useless
        _make_candidate(R_c={20}, covered=set(), seed=5),
        # large coverage, expensive
        _make_candidate(R_c=set(range(15)), covered=set(range(12)), seed=6),
    ]
    return candidates

#-------------------------------------------------------------------------------

def _check_invariants(selected, trajectory, budget_B, n_test_queries, label):
    # budget invariant
    cached = set()
    for c in selected:
        cached |= c.R_c
    assert len(cached) <= budget_B, \
        f"{label}: cached={len(cached)} > budget_B={budget_B}"

    # trajectory length
    assert len(trajectory) == len(selected), \
        f"{label}: trajectory length {len(trajectory)} != selected length {len(selected)}"

    # monotone non-decreasing
    for i in range(1, len(trajectory)):
        assert trajectory[i]['cig_train_coverage'] >= trajectory[i-1]['cig_train_coverage'] - 1e-9, \
            f"{label}: cig_train_coverage not monotone at step {i}"
        assert trajectory[i]['vectors_cached'] >= trajectory[i-1]['vectors_cached'], \
            f"{label}: vectors_cached not monotone at step {i}"

    # cig_train_coverage in [0, 1]
    for t in trajectory:
        assert 0.0 <= t['cig_train_coverage'] <= 1.0 + 1e-9, \
            f"{label}: cig_train_coverage {t['cig_train_coverage']} out of range"

#-------------------------------------------------------------------------------

def test_select_invariants_all_strategies():
    """All four strategies satisfy budget and trajectory invariants."""
    print("Test: invariants for all strategies")
    n_test = 20
    candidates = _build_candidates(n_test)

    for strategy in ('random', 'greedy_naive', 'greedy_cost_static', 'greedy_cost'):
        selected, trajectory = select_candidates(
            candidates, budget_B=20,
            strategy=strategy, n_train_queries=n_test, n_test_queries=n_test, seed=42,
        )
        _check_invariants(selected, trajectory, 20, n_test, strategy)
        print(f"  {strategy}: {len(selected)} selected, {len(trajectory)} steps")

    print("  Passed.")

#-------------------------------------------------------------------------------

def test_select_random_reproducible():
    """Same seed → same result; different seed → usually different result."""
    print("Test: random reproducibility")
    n_test = 20
    candidates = _build_candidates(n_test)

    sel1, _ = select_candidates(candidates, 20, 'random', n_test, n_test, seed=42)
    sel2, _ = select_candidates(candidates, 20, 'random', n_test, n_test, seed=42)

    assert [id(c) for c in sel1] == [id(c) for c in sel2], "Same seed must give same result"
    print("  Same seed: reproducible. Passed.")

#-------------------------------------------------------------------------------

def test_select_greedy_naive_picks_best_first():
    """Greedy naive picks the candidate with highest coverage first."""
    print("Test: greedy_naive picks best coverage first")
    n_test = 20
    candidates = [
        _make_candidate(R_c={0}, covered=set(range(15)), seed=0),   # best: 15
        _make_candidate(R_c={1}, covered=set(range(5)), seed=1),    # 5
        _make_candidate(R_c={2}, covered=set(range(3)), seed=2),    # 3
    ]
    selected, _ = select_candidates(candidates, 20, 'greedy_naive', n_test, n_test, seed=42)

    assert len(selected) > 0
    assert selected[0].covered_train_indices == set(range(15)), \
        "First selection should be the candidate with coverage 15"
    print("  Passed.")

#-------------------------------------------------------------------------------

def test_select_greedy_cost_deferred_flush():
    """greedy_cost adds zero-marginal-cost candidates in the deferred flush."""
    print("Test: greedy_cost deferred flush")
    n_test = 10
    # Candidate A: covers {0,1,2,3}, needs vectors {0,1}
    # Candidate B: covers {4,5,6},   needs vectors {0,1} (same vectors — zero marginal after A)
    A = _make_candidate(R_c={0, 1}, covered={0, 1, 2, 3}, seed=10)
    B = _make_candidate(R_c={0, 1}, covered={4, 5, 6}, seed=11)
    C = _make_candidate(R_c={2, 3}, covered={7, 8, 9}, seed=12)

    candidates = [A, B, C]
    selected, trajectory = select_candidates(
        candidates, budget_B=10,
        strategy='greedy_cost', n_train_queries=n_test, n_test_queries=n_test, seed=42,
    )

    assert any(c is B for c in selected), "B should be added via deferred flush after A"

    # B's trajectory entry should have same vectors_cached as previous entry
    b_step = next(i for i, c in enumerate(selected) if c is B)
    if b_step > 0:
        assert trajectory[b_step]['vectors_cached'] == trajectory[b_step - 1]['vectors_cached'], \
            "Deferred addition should not increase vectors_cached"

    print("  Passed.")

#-------------------------------------------------------------------------------

def test_select_empty_candidates():
    """Empty candidate list → empty selection and trajectory."""
    print("Test: empty candidates")
    for strategy in ('random', 'greedy_naive', 'greedy_cost_static', 'greedy_cost'):
        selected, trajectory = select_candidates([], 100, strategy, 20, 20, seed=42)
        assert selected == [], f"{strategy}: expected empty selected"
        assert trajectory == [], f"{strategy}: expected empty trajectory"
    print("  Passed.")

#-------------------------------------------------------------------------------

def test_select_budget_zero():
    """budget_B=0: candidates with non-empty R_c cannot be added."""
    print("Test: budget_B=0")
    n_test = 10
    candidates = [_make_candidate(R_c={0, 1}, covered={0, 1}, seed=i) for i in range(5)]
    for strategy in ('random', 'greedy_naive', 'greedy_cost_static', 'greedy_cost'):
        selected, trajectory = select_candidates(candidates, 0, strategy, n_test, n_test, seed=42)
        assert len(selected) == 0, f"{strategy}: expected no selections with budget_B=0"
    print("  Passed.")

#-------------------------------------------------------------------------------

def test_select_full_coverage_possible():
    """When full coverage is achievable within budget, greedy strategies reach 1.0."""
    print("Test: full coverage achievable")
    n_test = 5
    candidates = [
        _make_candidate(R_c={i}, covered={i}, seed=i)
        for i in range(n_test)
    ]
    for strategy in ('greedy_naive', 'greedy_cost_static', 'greedy_cost'):
        selected, trajectory = select_candidates(
            candidates, budget_B=100,
            strategy=strategy, n_train_queries=n_test, n_test_queries=n_test, seed=42,
        )
        if trajectory:
            final_cov = trajectory[-1]['cig_train_coverage']
            assert abs(final_cov - 1.0) < 1e-9, \
                f"{strategy}: expected cig_train_coverage=1.0, got {final_cov}"
    print("  Passed.")

#-------------------------------------------------------------------------------

def test_select_runs_to_budget_exhaustion():
    """Selection continues until budget exhausted — no cardinality cap."""
    print("Test: runs to budget exhaustion")
    n_test = 30
    # 10 candidates each needing exactly 1 unique vector; budget=10 fits all
    candidates = [
        _make_candidate(R_c={i}, covered={i}, seed=i)
        for i in range(10)
    ]
    for strategy in ('random', 'greedy_naive', 'greedy_cost_static', 'greedy_cost'):
        selected, trajectory = select_candidates(
            candidates, budget_B=10,
            strategy=strategy, n_train_queries=n_test, n_test_queries=n_test, seed=42,
        )
        assert len(selected) == 10, \
            f"{strategy}: expected all 10 selected (budget fits all), got {len(selected)}"
    print("  Passed.")

#-------------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Testing selection.py")
    print("=" * 70)

    test_select_invariants_all_strategies()
    test_select_random_reproducible()
    test_select_greedy_naive_picks_best_first()
    test_select_greedy_cost_deferred_flush()
    test_select_empty_candidates()
    test_select_budget_zero()
    test_select_full_coverage_possible()
    test_select_runs_to_budget_exhaustion()

    print("\n" + "=" * 70)
    print("All tests passed.")
    print("=" * 70)

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
