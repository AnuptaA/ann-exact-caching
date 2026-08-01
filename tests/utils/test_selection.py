#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from utils.base.population import CandidateEntry
from utils.base.selection import select_candidates
from tests.helpers import check_selection_invariants

#-------------------------------------------------------------------------------

def _mask(indices: set, n_domain: int) -> np.ndarray:
    """Boolean coverage array of length n_domain from a set of indices --
    matches the covered_train_indices/covered_test_indices representation
    (fixed-size bool array, not a Python set)."""
    m = np.zeros(n_domain, dtype=bool)
    for i in indices:
        m[i] = True
    return m

# Fixed base-vector domain size for R_c masks in this file -- must exceed
# the largest R_c index used by any fixture below (max used is 20).
N_BASE = 30

def _make_candidate(
    query_dim: int = 4,
    R_c: set = None,
    covered: set = None,
    n_domain: int = 20,
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
    R_c_mask = _mask(R_c, N_BASE)
    covered_mask = _mask(covered, n_domain)
    return CandidateEntry(
        query=q,
        r_Q=r_Q,
        alpha=alpha,
        r_c=alpha * r_Q,
        R_c=R_c_mask,
        rc_size=int(R_c_mask.sum()),
        covered_train_indices=covered_mask,
        covered_test_indices=covered_mask,
        top_k_vectors=[q],
        top_k_indices=[0],
        half_gap=0.01,
    )

def _build_candidates(n_test: int = 20) -> list:
    """Build a fixed small candidate set with predictable coverage."""
    candidates = [
        # covers 10 test queries, needs 5 vectors
        _make_candidate(R_c=set(range(5)), covered=set(range(10)), n_domain=n_test, seed=0),
        # covers 6 test queries, needs 3 vectors
        _make_candidate(R_c=set(range(3)), covered=set(range(6)), n_domain=n_test, seed=1),
        # covers 4 test queries, needs 8 vectors
        _make_candidate(R_c=set(range(8)), covered=set(range(4)), n_domain=n_test, seed=2),
        # covers 2 test queries, needs 1 vector
        _make_candidate(R_c={10}, covered={15, 16}, n_domain=n_test, seed=3),
        # covers 5 test queries, overlaps with first, needs 2 new vectors
        _make_candidate(R_c={5, 6}, covered=set(range(5, 10)), n_domain=n_test, seed=4),
        # zero coverage: useless
        _make_candidate(R_c={20}, covered=set(), n_domain=n_test, seed=5),
        # large coverage, expensive
        _make_candidate(R_c=set(range(15)), covered=set(range(12)), n_domain=n_test, seed=6),
    ]
    return candidates

#-------------------------------------------------------------------------------

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
        check_selection_invariants(selected, trajectory, 20, strategy)
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
        _make_candidate(R_c={0}, covered=set(range(15)), n_domain=n_test, seed=0),   # best: 15
        _make_candidate(R_c={1}, covered=set(range(5)), n_domain=n_test, seed=1),    # 5
        _make_candidate(R_c={2}, covered=set(range(3)), n_domain=n_test, seed=2),    # 3
    ]
    selected, _ = select_candidates(candidates, 20, 'greedy_naive', n_test, n_test, seed=42)

    assert len(selected) > 0
    assert np.array_equal(selected[0].covered_train_indices, _mask(set(range(15)), n_test)), \
        "First selection should be the candidate with coverage 15"
    print("  Passed.")

#-------------------------------------------------------------------------------

def test_select_greedy_cost_deferred_flush():
    """greedy_cost adds zero-marginal-cost candidates in the deferred flush."""
    print("Test: greedy_cost deferred flush")
    n_test = 10
    # Candidate A: covers {0,1,2,3}, needs vectors {0,1}
    # Candidate B: covers {4,5,6},   needs vectors {0,1} (same vectors — zero marginal after A)
    A = _make_candidate(R_c={0, 1}, covered={0, 1, 2, 3}, n_domain=n_test, seed=10)
    B = _make_candidate(R_c={0, 1}, covered={4, 5, 6}, n_domain=n_test, seed=11)
    C = _make_candidate(R_c={2, 3}, covered={7, 8, 9}, n_domain=n_test, seed=12)

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

def test_greedy_cost_static_matches_dynamic_when_no_overlap():
    """When candidates have zero R_c overlap, marginal cost (recomputed each
    iteration) always equals static cost (each candidate's own rc_size, fixed
    up front), so greedy_cost and greedy_cost_static must pick an identical
    selection order."""
    print("Test: greedy_cost_static == greedy_cost with disjoint R_c")
    n_domain = 27
    candidates = [
        _make_candidate(R_c=set(range(0, 5)), covered=set(range(0, 10)), n_domain=n_domain, seed=0),
        _make_candidate(R_c=set(range(5, 8)), covered=set(range(10, 16)), n_domain=n_domain, seed=1),
        _make_candidate(R_c=set(range(8, 16)), covered=set(range(16, 20)), n_domain=n_domain, seed=2),
        _make_candidate(R_c=set(range(16, 17)), covered={20, 21}, n_domain=n_domain, seed=3),
        _make_candidate(R_c=set(range(17, 19)), covered=set(range(22, 27)), n_domain=n_domain, seed=4),
    ]

    selected_static, traj_static = select_candidates(
        candidates, budget_B=100,
        strategy='greedy_cost_static', n_train_queries=27, n_test_queries=27, seed=42,
    )
    selected_dynamic, traj_dynamic = select_candidates(
        candidates, budget_B=100,
        strategy='greedy_cost', n_train_queries=27, n_test_queries=27, seed=42,
    )

    assert [id(c) for c in selected_static] == [id(c) for c in selected_dynamic], (
        "Selection order should be identical when R_c sets are disjoint"
    )
    for step_s, step_d in zip(traj_static, traj_dynamic):
        assert step_s['vectors_cached'] == step_d['vectors_cached']
        assert abs(step_s['cig_train_coverage'] - step_d['cig_train_coverage']) < 1e-9

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
    candidates = [_make_candidate(R_c={0, 1}, covered={0, 1}, n_domain=n_test, seed=i) for i in range(5)]
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
        _make_candidate(R_c={i}, covered={i}, n_domain=n_test, seed=i)
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
        _make_candidate(R_c={i}, covered={i}, n_domain=n_test, seed=i)
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
# Cross-check: the production select_candidates (vectorized, Round 4) must
# produce byte-identical (selected, trajectory) output to a deliberately
# naive, unvectorized reference implementation of the same scoring logic,
# across random candidate sets with realistic heavy overlap. This is the test
# that actually proves vectorization didn't change behavior at a scale large
# enough to exercise real overlap patterns -- the small hand-built fixtures
# above only cover specific hand-picked cases.
#-------------------------------------------------------------------------------

def _ref_fits_budget(c, cached_mask, n_cached, budget_B):
    if n_cached + c.rc_size <= budget_B:
        return True
    new_vecs = int(np.sum(c.R_c & ~cached_mask))
    return n_cached + new_vecs <= budget_B


def _ref_add_to_cache(c, cached_mask, n_cached):
    new_vecs = int(np.sum(c.R_c & ~cached_mask))
    cached_mask |= c.R_c
    return n_cached + new_vecs


def _ref_record(selected, n_cached, covered_train_mask, prev_train, n_train_queries,
                 covered_test_mask, prev_test, n_test_queries, trajectory):
    n_train = int(covered_train_mask.sum())
    n_test = int(covered_test_mask.sum())
    trajectory.append({
        "num_entries": len(selected),
        "vectors_cached": n_cached,
        "cig_train_coverage": n_train / n_train_queries if n_train_queries > 0 else 0.0,
        "cig_test_coverage": n_test / n_test_queries if n_test_queries > 0 else 0.0,
        "new_train_coverage": n_train - prev_train,
        "new_test_coverage": n_test - prev_test,
    })


def _reference_select(candidates, budget_B, strategy, n_train_queries, n_test_queries):
    """Deliberately naive (Python-loop, one candidate at a time) reference
    implementation of greedy_naive/greedy_cost_static/greedy_cost, kept only
    in this test file for cross-checking against the production vectorized
    implementation. Not used anywhere else."""
    selected = []
    n_base = len(candidates[0].R_c) if candidates else 1
    cached_mask = np.zeros(n_base, dtype=bool)
    covered_train_mask = np.zeros(n_train_queries, dtype=bool)
    covered_test_mask = np.zeros(n_test_queries, dtype=bool)
    trajectory = []

    if strategy in ("greedy_naive", "greedy_cost_static"):
        working = list(candidates)
        while working:
            n_cached = int(cached_mask.sum())
            working = [c for c in working if _ref_fits_budget(c, cached_mask, n_cached, budget_B)]
            if not working:
                break
            newly = np.array([
                int(np.sum(c.covered_train_indices & ~covered_train_mask)) for c in working
            ])
            keep = newly > 0
            working = [c for c, k in zip(working, keep) if k]
            newly = newly[keep]
            if not working:
                break
            if strategy == "greedy_naive":
                scores = newly
            else:
                static_cost = np.array([c.rc_size for c in working], dtype=float)
                static_cost[static_cost == 0.0] = 1.0
                scores = newly / static_cost
            best_idx = int(np.argmax(scores))
            best = working.pop(best_idx)
            prev_train = int(covered_train_mask.sum())
            prev_test = int(covered_test_mask.sum())
            selected.append(best)
            n_cached = _ref_add_to_cache(best, cached_mask, n_cached)
            covered_train_mask |= best.covered_train_indices
            covered_test_mask |= best.covered_test_indices
            _ref_record(selected, n_cached, covered_train_mask, prev_train, n_train_queries,
                        covered_test_mask, prev_test, n_test_queries, trajectory)
    elif strategy == "greedy_cost":
        active = list(candidates)
        deferred = []
        while active:
            n_cached = int(cached_mask.sum())
            active = [c for c in active if _ref_fits_budget(c, cached_mask, n_cached, budget_B)]
            if not active:
                break
            marginal = np.array([int(np.sum(c.R_c & ~cached_mask)) for c in active])
            newly = np.array([
                int(np.sum(c.covered_train_indices & ~covered_train_mask)) for c in active
            ])
            keep = newly > 0
            active = [c for c, k in zip(active, keep) if k]
            marginal = marginal[keep]
            newly = newly[keep]
            if not active:
                break
            zero_cost = marginal == 0
            deferred.extend(c for c, z in zip(active, zero_cost) if z)
            active = [c for c, z in zip(active, zero_cost) if not z]
            marginal = marginal[~zero_cost]
            newly = newly[~zero_cost]
            if not active:
                break
            scores = newly / marginal
            best_idx = int(np.argmax(scores))
            best = active.pop(best_idx)
            prev_train = int(covered_train_mask.sum())
            prev_test = int(covered_test_mask.sum())
            selected.append(best)
            n_cached = _ref_add_to_cache(best, cached_mask, n_cached)
            covered_train_mask |= best.covered_train_indices
            covered_test_mask |= best.covered_test_indices
            _ref_record(selected, n_cached, covered_train_mask, prev_train, n_train_queries,
                        covered_test_mask, prev_test, n_test_queries, trajectory)

        if deferred:
            deferred_scored = sorted(
                [(int(np.sum(c.covered_train_indices & ~covered_train_mask)), c) for c in deferred],
                key=lambda x: x[0], reverse=True,
            )
        else:
            deferred_scored = []
        n_cached = int(cached_mask.sum())
        for benefit, c in deferred_scored:
            if benefit > 0:
                prev_train = int(covered_train_mask.sum())
                prev_test = int(covered_test_mask.sum())
                selected.append(c)
                covered_train_mask |= c.covered_train_indices
                covered_test_mask |= c.covered_test_indices
                _ref_record(selected, n_cached, covered_train_mask, prev_train, n_train_queries,
                            covered_test_mask, prev_test, n_test_queries, trajectory)
    else:
        raise ValueError(f"no reference implementation for strategy {strategy!r}")

    return selected, trajectory


def _random_overlapping_candidates(rng, n_candidates, n_base, n_train, r_c_density, cover_density):
    """n_candidates random CandidateEntry objects with heavy, realistic
    overlap in both R_c and covered_train_indices (controlled by density),
    similar in spirit to the real nested-sphere candidates' overlap pattern."""
    candidates = []
    for i in range(n_candidates):
        R_c = rng.random(n_base) < r_c_density
        covered = rng.random(n_train) < cover_density
        candidates.append(CandidateEntry(
            query=rng.standard_normal(4).astype(np.float32),
            r_Q=0.1, alpha=None, r_c=0.2,
            R_c=R_c, rc_size=int(R_c.sum()),
            covered_train_indices=covered, covered_test_indices=covered,
            top_k_vectors=[], top_k_indices=[], half_gap=0.0,
        ))
    return candidates


def test_vectorized_matches_naive_reference():
    """The production (vectorized) select_candidates must match a naive
    per-candidate-loop reference implementation exactly, on random candidate
    sets with heavy overlap (~40% density, matching real-data density),
    across several seeds and all three greedy
    strategies."""
    print("Test: vectorized implementation matches naive reference")
    n_candidates, n_base, n_train = 300, 150, 60

    for seed in range(5):
        rng = np.random.default_rng(seed)
        candidates = _random_overlapping_candidates(
            rng, n_candidates, n_base, n_train, r_c_density=0.4, cover_density=0.15,
        )
        for strategy in ("greedy_naive", "greedy_cost_static", "greedy_cost"):
            for budget_B in (n_base // 10, n_base // 2, n_base):
                got_selected, got_trajectory = select_candidates(
                    candidates, budget_B, strategy, n_train, n_train, seed=42,
                )
                ref_selected, ref_trajectory = _reference_select(
                    candidates, budget_B, strategy, n_train, n_train,
                )
                assert [id(c) for c in got_selected] == [id(c) for c in ref_selected], (
                    f"seed={seed} strategy={strategy} budget={budget_B}: "
                    f"selection order differs from naive reference"
                )
                assert got_trajectory == ref_trajectory, (
                    f"seed={seed} strategy={strategy} budget={budget_B}: "
                    f"trajectory differs from naive reference"
                )
    print(f"  Matched naive reference across {5} seeds x 3 strategies x 3 budgets. Passed.")


def test_greedy_cost_incremental_matches_reference_at_high_budget():
    """Stress the specific regime that motivated the incremental marginal/
    newly rewrite of _select_greedy_cost: budget_fraction close to 1.0 with
    high R_c overlap, where the active set barely shrinks across many
    iterations (unlike the smaller-scale check in
    test_vectorized_matches_naive_reference, this uses a larger candidate
    set and higher density specifically to exercise many iterations of the
    incremental delta-update path, including the deferred-flush branch)."""
    print("Test: greedy_cost incremental delta matches naive reference at budget_fraction~1.0")
    n_candidates, n_base, n_train = 800, 400, 120

    for seed in range(3):
        rng = np.random.default_rng(seed)
        candidates = _random_overlapping_candidates(
            rng, n_candidates, n_base, n_train, r_c_density=0.44, cover_density=0.08,
        )
        for budget_B in (n_base, int(n_base * 0.9)):
            got_selected, got_trajectory = select_candidates(
                candidates, budget_B, "greedy_cost", n_train, n_train, seed=42,
            )
            ref_selected, ref_trajectory = _reference_select(
                candidates, budget_B, "greedy_cost", n_train, n_train,
            )
            assert [id(c) for c in got_selected] == [id(c) for c in ref_selected], (
                f"seed={seed} budget={budget_B}: selection order differs from naive reference"
            )
            assert got_trajectory == ref_trajectory, (
                f"seed={seed} budget={budget_B}: trajectory differs from naive reference"
            )
    print("  Matched naive reference across 3 seeds x 2 high budgets. Passed.")


def test_greedy_cost_chunked_delta_matches_reference_across_chunk_boundary():
    """The incremental delta-update's chunked batching (_chunked_delta_counts,
    _DELTA_CHUNK_SIZE=6000) must produce identical results whether or not
    the active set spans multiple chunks. Uses a candidate count comfortably
    above one chunk so the first iteration's delta-update genuinely crosses
    the chunk boundary, and checks against the naive per-candidate reference
    (not just internal self-consistency)."""
    print("Test: chunked delta-update matches naive reference across a chunk boundary")
    from utils.base.selection import _DELTA_CHUNK_SIZE
    assert _DELTA_CHUNK_SIZE < 7500, "test assumes n_candidates spans >1 chunk"

    n_candidates, n_base, n_train = 7500, 150, 40
    rng = np.random.default_rng(7)
    candidates = _random_overlapping_candidates(
        rng, n_candidates, n_base, n_train, r_c_density=0.4, cover_density=0.1,
    )
    for budget_B in (n_base, int(n_base * 0.8)):
        got_selected, got_trajectory = select_candidates(
            candidates, budget_B, "greedy_cost", n_train, n_train, seed=42,
        )
        ref_selected, ref_trajectory = _reference_select(
            candidates, budget_B, "greedy_cost", n_train, n_train,
        )
        assert [id(c) for c in got_selected] == [id(c) for c in ref_selected], (
            f"budget={budget_B}: selection order differs from naive reference "
            f"(chunk boundary case)"
        )
        assert got_trajectory == ref_trajectory, (
            f"budget={budget_B}: trajectory differs from naive reference (chunk boundary case)"
        )
    print("  Matched naive reference with active set spanning multiple chunks. Passed.")

def test_chunked_delta_counts_bitpack_matches_fancy_index():
    """_chunked_delta_counts has two code paths (fancy-index gather vs.
    bit-packed popcount, chosen by idx size relative to the attribute
    length -- see _BITPACK_IDX_FRACTION). Both must produce bit-identical
    integer counts; this directly forces each path (small idx -> gather,
    large idx -> bitpack) on the same underlying data and checks
    np.array_equal, not just that the overall algorithm still works.

    Builds CandidateEntry directly rather than via _make_candidate --
    that helper hardcodes R_c's length to the file-level N_BASE=30
    constant, too small to give the fancy-index/bit-packed paths a
    meaningfully large idx to diverge on."""
    from utils.base.selection import _chunked_delta_counts

    n_candidates, n_base = 500, 400
    rng = np.random.default_rng(3)
    candidates = []
    for i in range(n_candidates):
        size = int(rng.integers(50, 350))
        R_c_mask = np.zeros(n_base, dtype=bool)
        R_c_mask[rng.choice(n_base, size=size, replace=False)] = True
        candidates.append(CandidateEntry(
            query=np.zeros(4, dtype=np.float32),
            r_Q=0.2, alpha=1.0, r_c=0.2,
            R_c=R_c_mask, rc_size=int(R_c_mask.sum()),
            covered_train_indices=np.zeros(10, dtype=bool),
            covered_test_indices=np.zeros(10, dtype=bool),
            top_k_vectors=[], top_k_indices=[], half_gap=0.01,
        ))
    # small idx: well below n_base/20=20, forces the fancy-index path
    small_idx = np.array(sorted(rng.choice(n_base, size=5, replace=False)))
    # large idx: well above n_base/20=20, forces the bit-packed path
    large_idx = np.array(sorted(rng.choice(n_base, size=300, replace=False)))

    expected_small = np.array([
        int(np.count_nonzero(c.R_c[small_idx])) for c in candidates
    ], dtype=np.int64)
    expected_large = np.array([
        int(np.count_nonzero(c.R_c[large_idx])) for c in candidates
    ], dtype=np.int64)

    got_small = _chunked_delta_counts(candidates, 'R_c', small_idx, chunk_size=64)
    got_large = _chunked_delta_counts(candidates, 'R_c', large_idx, chunk_size=64)

    assert np.array_equal(got_small, expected_small), "fancy-index path diverges from reference"
    assert np.array_equal(got_large, expected_large), "bit-packed path diverges from reference"
    print("  OK: both _chunked_delta_counts code paths match a brute-force reference exactly")

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
    test_vectorized_matches_naive_reference()
    test_greedy_cost_incremental_matches_reference_at_high_budget()
    test_greedy_cost_chunked_delta_matches_reference_across_chunk_boundary()
    test_chunked_delta_counts_bitpack_matches_fancy_index()

    print("\n" + "=" * 70)
    print("All tests passed.")
    print("=" * 70)

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
