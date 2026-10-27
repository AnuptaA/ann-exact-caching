#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Tests for ee_incidence: toy states X, Z, W from the spec, provenance, cost-0 priority, the incremental engine
against the slow reference, and the group-aware LSH counts. Run with pytest or `python3 -B test_ee_incidence.py`."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import ee_incidence as ei

K = 10

#-------------------------------------------------------------------------------
# Toy states: n = 10 queries with K = 10 slots, every slot has its own vector id q * 10 + j, query q is
# "pre-covered" in its first (10 - deficit) slots. Candidates have group {0} (a query with deficit 0).

def toy(deficits_tenths, candidates, shared=None, group=(0,)):
    n = len(deficits_tenths)
    slot_vec = np.arange(n * K, dtype=np.int32).reshape(n, K)
    cnt0 = np.zeros((n, K), np.int32)
    for q, d in enumerate(deficits_tenths):
        cnt0[q, :K - d] = 1
    n_vec = n * K
    if shared:  # vectors present in the slots of several queries: {extra id: [(q, j), ...]}
        for v, pairs in shared.items():
            n_vec = max(n_vec, v + 1)
            for q, j in pairs:
                slot_vec[q, j] = v
    ptr, vec = ei.csr_from_lists(candidates)
    gptr, gq = ei.csr_from_lists([list(group)] * len(candidates))
    inc = ei.Incidence(n=n, K=K, n_vec=n_vec, slot_vec=slot_vec, cand_ptr=ptr, cand_vec=vec, grp_ptr=gptr, grp_q=gq,
                       cand_source=np.zeros(len(candidates), np.int32), cnt0=cnt0)
    return inc, ei.build_index(inc)

def slots(q, js):
    return [q * K + j for j in js]

def state_x():
    # deficits: four at 0, a = 0.5, b1 = b2 = 0.6, c1 = c2 = c3 = 1.0 (query ids 4, 5, 6, 7, 8, 9)
    A = slots(4, range(5, 10))
    B = slots(5, [4, 5]) + slots(6, [4, 5])
    C = slots(7, [0, 1]) + slots(8, [0, 1]) + slots(9, [0, 1])
    D = slots(7, range(2, 10))
    return toy([0, 0, 0, 0, 5, 6, 6, 10, 10, 10], [A, B, C, D])

def gains(inc, idx, score, T, **kw):
    g, c = ei.score_gains(inc, idx, score, T, **kw)
    return g.tolist(), c.tolist()

#-------------------------------------------------------------------------------

def test_state_x_gains_and_hstar():
    inc, idx = state_x()
    T = 0.8
    assert ei.replay_h_star(inc, idx, (), T) == 7
    assert [ei.replay_h_star(inc, idx, (c,), T) for c in range(4)] == [7, 7, 7, 8]  # h* after each alone
    g1, cost = gains(inc, idx, "SEL1", T)
    assert cost == [5, 4, 6, 8] and g1 == [1, 0, 0, 1]                              # A 0.20 and D 0.125 per vector
    assert gains(inc, idx, "SEL2", T)[0] == [0, 0, 0, 1]                            # D 0.125 per vector
    assert gains(inc, idx, "SEL3", T)[0] == [5, 4, 6, 8]                            # 0.1 per vector (tenths per vector = 1)
    g4 = gains(inc, idx, "SEL4", T, h0=7)[0]
    assert g4 == [5, 4, 0, 4]                                                       # per vector 0.10, 0.10, 0, 0.05
    assert gains(inc, idx, "SEL5", T)[0] == g4                                      # SEL5 on this state equals SEL4 with h0 = 7
    assert [g / c for g, c in zip(g1, cost)] == [0.2, 0.0, 0.0, 0.125]            # SEL1 utility per vector
    assert [g / c for g, c in zip(gains(inc, idx, "SEL4", T, h0=7)[0], cost)] == [1.0, 1.0, 0.0, 0.5]

def test_state_x_after_buying_d():
    inc, idx = state_x()
    T = 0.8
    A, B, D = 0, 1, 3
    assert ei.replay_h_star(inc, idx, (D,), T) == 8
    assert ei.replay_h_star(inc, idx, (D, A), T) == 8 and ei.replay_h_star(inc, idx, (D, B), T) == 8
    assert gains(inc, idx, "SEL2", T, replay=(D,))[0][A] == 0 and gains(inc, idx, "SEL2", T, replay=(D,))[0][B] == 0
    assert ei.replay_h_star(inc, idx, (D, A, B), T) == 9

def test_state_z():
    P = slots(5, [7, 8, 9])
    shared = {100 + j: [(7, j), (8, j), (9, j)] for j in range(3)}  # each Q vector sits in a slot of all three c queries
    inc, idx = toy([0] * 5 + [3, 4, 10, 10, 10], [P, [100, 101, 102]], shared=shared)
    T = 0.95
    assert ei.replay_h_star(inc, idx, (), T) == 6
    assert gains(inc, idx, "SEL2", T)[0] == [1, 0]                                  # P +1, Q 0
    g3, cost = gains(inc, idx, "SEL3", T)
    assert [g / c for g, c in zip(g3, cost)] == [1.0, 3.0]                          # 0.1 and 0.3 per vector
    assert ei.replay_h_star(inc, idx, (0,), T) == 7

def test_state_w():
    G = slots(6, range(5))
    inc, idx = toy([0] * 6 + [10] * 4, [G])
    T = 0.95
    assert ei.replay_h_star(inc, idx, (), T) == 6
    assert gains(inc, idx, "SEL2", T)[0] == [1] and ei.replay_h_star(inc, idx, (0,), T) == 7
    assert gains(inc, idx, "SEL5", T)[0] == [0]   # h0 = current h* = 6, the six smallest deficits stay 0
    assert gains(inc, idx, "SEL4", T, h0=6)[0] == [0]

def test_greedy_on_state_x_picks_best_utilities():
    inc, idx = state_x()
    r = ei.greedy(inc, idx, "SEL1", 0.8, budget=13)
    assert r.selected.tolist()[0] == 0 and r.pool_size <= 13   # A (0.20 per vector) first
    r2 = ei.greedy(inc, idx, "SEL2", 0.8, budget=8)
    assert r2.selected.tolist() == [3] and r2.h_star == 8 and r2.pool_size == 8 and not r2.stalled

#-------------------------------------------------------------------------------

def test_provenance_group_queries_get_no_credit():
    # query 0 slot vectors {0, 1}; candidate A has group {0} and contains vector 0, candidate B has group {1}
    slot_vec = np.array([[0, 1], [2, 3], [4, 5]], np.int32)
    ptr, vec = ei.csr_from_lists([[0, 2], [0]])
    gptr, gq = ei.csr_from_lists([[0], [1]])
    inc = ei.Incidence(3, 2, 6, slot_vec, ptr, vec, gptr, gq, np.zeros(2, np.int32))
    idx = ei.build_index(inc)
    # A: vector 0 is query 0's slot but 0 is in A's group: no credit to query 0; vector 2 credits query 1
    # B: vector 0 credits query 0 (group {1} excludes it)
    assert gains(inc, idx, "SEL3", 0.9)[0] == [1, 1]
    r = ei.greedy(inc, idx, "SEL3", 0.9, budget=10)
    assert r.selected.tolist() == [1, 0] or r.selected.tolist() == [0, 1]
    run = ei._Run(inc, idx, 0.9)
    run.apply(0)
    assert run.covered.tolist() == [0, 1, 0]        # query 0 got nothing from candidate A
    assert run.compute_gains("SEL3", 10).tolist() == [0, 1]
    run.apply(1)
    assert run.covered.tolist() == [1, 1, 0]
    assert run.cost.tolist() == [0, 0]

def test_cost_zero_positive_gain_is_taken_first():
    # vector 0 sits in a slot of query 0 and of query 1; P (group {0}) credits query 1, then Q (group {1}) credits
    # query 0 at cost 0 because vector 0 is already in the pool
    slot_vec = np.array([[0, 1], [0, 3], [4, 5]], np.int32)
    ptr, vec = ei.csr_from_lists([[0], [0], [2]])
    gptr, gq = ei.csr_from_lists([[0], [1], [2]])
    inc = ei.Incidence(3, 2, 6, slot_vec, ptr, vec, gptr, gq, np.full(3, -1, np.int32))
    idx = ei.build_index(inc)
    r = ei.greedy(inc, idx, "SEL3", 0.9, budget=5)
    ref = ei.greedy_reference(inc, "SEL3", 0.9, 5)
    assert r.selected.tolist() == ref["selected"].tolist() == [0, 1]
    assert r.steps["cost"].tolist() == [1, 0] and r.pool_size == 1
    assert r.covered.tolist() == [1, 1, 0]
    # a cost-0 candidate beats a positive-cost candidate of higher raw gain
    ptr2, vec2 = ei.csr_from_lists([[0], [0], [1, 3]])
    inc2 = ei.Incidence(3, 2, 6, slot_vec, ptr2, vec2, gptr, gq, np.full(3, -1, np.int32))
    r2 = ei.greedy(inc2, ei.build_index(inc2), "SEL3", 0.9, budget=5)
    assert r2.steps["cost"].tolist()[:2] == [1, 0]

#-------------------------------------------------------------------------------

def random_instance(rng, n=12, K=4, n_vec=60, C=40, with_cnt0=False):
    slot_vec = np.stack([rng.choice(n_vec, K, replace=False) for _ in range(n)]).astype(np.int32)
    cands, groups = [], []
    for _ in range(C):
        r = rng.choice(n_vec, int(rng.integers(3, 11)), replace=False)
        g = rng.choice(n, int(rng.integers(1, 4)), replace=False)
        cands.append(np.sort(r))
        groups.append(g)
    ptr, vec = ei.csr_from_lists(cands)
    gptr, gq = ei.csr_from_lists(groups)
    cnt0 = (rng.random((n, K)) < 0.15).astype(np.int32) if with_cnt0 else None
    return ei.Incidence(n, K, n_vec, slot_vec, ptr, vec, gptr, gq, np.arange(C, dtype=np.int32) % n, cnt0)

def test_incremental_engine_equals_slow_reference_on_random_instances():
    rng = np.random.default_rng(0)
    checked = 0
    for trial in range(24):
        inc = random_instance(rng, with_cnt0=trial % 2 == 1)
        idx = ei.build_index(inc)
        for T in (0.8, 0.9):
            for score in ei.SCORES:
                h0 = int(rng.integers(1, inc.n + 1)) if score == "SEL4" else None
                budget = int(rng.integers(8, 30))
                r = ei.greedy(inc, idx, score, T, budget, h0)
                ref = ei.greedy_reference(inc, score, T, budget, h0)
                assert r.selected.tolist() == ref["selected"].tolist(), (trial, T, score)
                assert r.steps["gain"].tolist() == ref["gains"].tolist(), (trial, T, score)
                assert r.covered.tolist() == ref["covered"].tolist() and r.h_star == ref["h_star"]
                assert set(np.flatnonzero(r.pool_mask).tolist()) == ref["pool"] and r.pool_size <= budget
                checked += 1
    assert checked == 24 * 2 * 5

def test_leave_group_out_recall_matches_set_definition():
    rng = np.random.default_rng(1)
    inc = random_instance(rng)
    idx = ei.build_index(inc)
    r = ei.greedy(inc, idx, "SEL3", 0.9, 30)
    assert len(r.selected) >= 3
    for q in range(inc.n):
        avail = set()
        for c in r.selected:
            if q not in inc.grp_q[inc.grp_ptr[c]:inc.grp_ptr[c + 1]]:
                avail |= set(inc.cand_vec[inc.cand_ptr[c]:inc.cand_ptr[c + 1]].tolist())
        assert r.covered[q] == sum(int(v) in avail for v in inc.slot_vec[q])

def test_stall_and_budget():
    inc, idx = state_x()
    r = ei.greedy(inc, idx, "SEL2", 0.8, budget=100)  # only D has positive SEL2 gain, then nothing helps
    assert r.stalled and r.unused_budget == 100 - r.pool_size and r.pool_size < 100
    r0 = ei.greedy(inc, idx, "SEL2", 0.8, budget=3)   # nothing fits
    assert len(r0.selected) == 0 and r0.stalled and r0.unused_budget == 3

#-------------------------------------------------------------------------------

def test_group_aware_counts_match_brute_force():
    rng = np.random.default_rng(2)
    d, n_vec, n = 6, 80, 14
    base = rng.standard_normal((n_vec, d))
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    train = rng.standard_normal((n, d))
    train /= np.linalg.norm(train, axis=1, keepdims=True)
    dist = np.arccos(np.clip(train @ base.T, -1, 1)) / np.pi
    slot_vec = np.argsort(dist, axis=1)[:, :4].astype(np.int32)
    cands, groups = [], []
    for _ in range(30):
        q = int(rng.integers(n))
        near = np.argsort(dist[q])[:int(rng.integers(5, 25))]
        cands.append(np.sort(near))
        groups.append(np.unique(np.concatenate([[q], rng.choice(n, 2, replace=False)])))
    ptr, vec = ei.csr_from_lists(cands)
    gptr, gq = ei.csr_from_lists(groups)
    inc = ei.Incidence(n, 4, n_vec, slot_vec, ptr, vec, gptr, gq, np.arange(30, dtype=np.int32) % n)
    selected = np.array([0, 3, 4, 9, 12, 21, 25])
    pool_ids = np.unique(np.concatenate([cands[c] for c in selected]))
    r_hat = rng.uniform(0.2, 0.5, n)
    got = ei.group_aware_counts(inc, selected, pool_ids, r_hat, train, base, chunk=5)
    for q in range(n):
        avail = set()
        for c in selected:
            if q not in groups[c]:
                avail |= set(cands[c].tolist())
        expect = sum((v in avail) and dist[q, v] <= r_hat[q] for v in pool_ids)
        assert got[q] == expect, q
    # a huge radius counts exactly the available pool vectors, which can exceed K slots' worth
    big = ei.group_aware_counts(inc, selected, pool_ids, np.full(n, 0.99), train, base)
    assert big.max() > 4
    # with a full pool (no group exclusion) the count would be larger for some query: groups really exclude
    full = np.array([(dist[q, pool_ids] <= 0.99).sum() for q in range(n)])
    assert (big <= full).all() and (big < full).any()

#-------------------------------------------------------------------------------

def run_all() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:
            failed += 1
            import traceback
            print(f"FAIL {name}: {exc!r}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return failed

if __name__ == "__main__":
    sys.exit(1 if run_all() else 0)
