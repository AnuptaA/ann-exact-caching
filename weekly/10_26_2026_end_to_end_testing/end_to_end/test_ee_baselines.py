#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Tests for ee_baselines. Run with pytest or `python3 -B test_ee_baselines.py`."""

import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import ee_baselines as eb
import ee_estimators as est
import ee_hitpolicy as hp
import ee_incidence as ei
import ee_proposals as ep
import ee_workload as ew

D, N_BASE, N, K, PREFIX = 8, 500, 80, 10, 61

def workload(seed=0, lam=0.5):
    rng = np.random.default_rng(seed)
    unit = lambda n: (lambda v: v / np.linalg.norm(v, axis=1, keepdims=True))(rng.standard_normal((n, D)))
    base, pool = unit(N_BASE), unit(400)
    return base, ew.build_workload("sift", base, pool, seed, lam, n=N, K=K, per_anchor=10, prefix_len=PREFIX)

def test_no_cache_hit_rate_floor_and_zero_recall_hits():
    base, w = workload()
    pool = eb.no_cache(w, N_BASE)
    assert not pool.pool_mask.any() and (pool.train_true_counts == 0).all()
    true = est.true_recall_counts(pool.pool_mask, w.test_top_ids)
    assert (true == 0).all()
    n = 400
    zeros = np.zeros(n, dtype=np.int64)
    rows = hp.evaluate_pool(0, 0.9, zeros, zeros, zeros, zeros, K, oracle_arm=True)
    hp1, hp2, hp3 = rows
    assert hp1["hits"] == 0 and np.isclose(hp1["achieved_recall"], 1.0)
    assert hp2["hits"] == 40 and np.isclose(hp2["hit_rate"], 0.1) and np.isclose(hp2["frac_hits_zero_recall"], 1.0)
    assert np.isclose(hp2["achieved_recall"], 0.9) and hp2["frac_below_T"] == 0
    assert hp3["hits"] == 40 and hp3["learned_cap_units"] == 20

def test_popularity_pool_ties_and_leave_one_out():
    base, w = workload(1)
    for budget in (15, 40, 120):
        pool = eb.popularity(w, budget, N_BASE)
        counts = np.bincount(w.train_top_ids.ravel(), minlength=N_BASE)
        ids = np.arange(N_BASE)
        order = sorted(ids[counts > 0].tolist(), key=lambda v: (-counts[v], v))
        assert set(np.flatnonzero(pool.pool_mask).tolist()) == set(order[:budget])
        for q in (0, 9, 33, 70):
            rest = np.delete(w.train_top_ids, q, axis=0)
            c2 = np.bincount(rest.ravel(), minlength=N_BASE)
            o2 = sorted(ids[c2 > 0].tolist(), key=lambda v: (-c2[v], v))[:budget]
            assert pool.train_true_counts[q] == len(set(w.train_top_ids[q].tolist()) & set(o2)), (budget, q)

def test_random_own_budget_provenance_and_nothing_fits_at_end():
    base, w = workload(2)
    inc, _ = ep.make_incidence(w, "own", n_vec=N_BASE)
    idx = ei.build_index(inc)
    for budget in (20, 55, 200):
        pool = eb.random_own(w, budget, 3, inc, idx)
        assert pool.pool_mask.sum() <= budget and len(set(pool.selected.tolist())) == len(pool.selected)
        union = set()
        for c in pool.selected:
            union |= set(w.train_top_ids[c].tolist())
        assert union == set(np.flatnonzero(pool.pool_mask).tolist())
        left = budget - pool.pool_mask.sum()
        for c in set(range(N)) - set(pool.selected.tolist()):
            assert len(set(w.train_top_ids[c].tolist()) - union) > left   # nothing else fits
        for q in range(0, N, 7):
            avail = set()
            for c in pool.selected:
                if c != q:
                    avail |= set(w.train_top_ids[c].tolist())
            assert pool.train_true_counts[q] == len(set(w.train_top_ids[q].tolist()) & avail)
    a, b = eb.random_own(w, 55, 3, inc, idx), eb.random_own(w, 55, 3, inc, idx)
    assert np.array_equal(a.selected, b.selected) and not np.array_equal(a.selected, eb.random_own(w, 55, 4, inc, idx).selected)

def slow_threshold_greedy(cands, budget):
    R = [set(r.tolist()) for r in cands["R"]]
    pool, alive, picks = set(), set(range(len(R))), []
    while True:
        best, best_key = None, None
        for c in sorted(alive):
            cost = len(R[c] - pool)
            if cost == 0:
                alive.discard(c)
                continue
            if cost > budget - len(pool):
                continue
            key = (Fraction(int(cands["numer"][c]), cost), -cost, -c)
            if best_key is None or key > best_key:
                best, best_key = c, key
        if best is None:
            return picks
        picks.append(best)
        pool |= R[best]
        alive.discard(best)

def test_threshold_candidates_and_greedy():
    base, w = workload(3)
    dist = np.arccos(np.clip(w.train @ base.T, -1, 1)) / np.pi
    budget = 45
    cands = eb.threshold_candidates(w, budget)
    theta80 = w.train_top_dist[:, math.ceil(0.8 * K) - 1]
    assert math.ceil(0.8 * K) == 8
    k = 0
    for a in range(N):
        others = [j for j in range(N) if j != a]
        R_all = np.array([w.tt_dist[a, j] + theta80[j] for j in others])
        order = np.argsort(R_all, kind="stable")
        for r_i in R_all[order][:20]:
            members = np.flatnonzero(dist[a] <= r_i)
            if len(members) > budget:
                continue
            assert cands["source"][k] == a and cands["G"][k] == [a]
            assert set(cands["R"][k].tolist()) == set(members.tolist())
            assert cands["numer"][k] == int((R_all <= r_i).sum())
            k += 1
    assert k == len(cands["R"])
    picks = eb.threshold_greedy(cands, budget, N_BASE)
    assert picks.tolist() == slow_threshold_greedy(cands, budget)
    pool = eb.threshold_baseline(w, budget, N_BASE)
    assert pool.pool_mask.sum() <= budget and pool.selected.tolist() == picks.tolist()
    for q in range(0, N, 9):
        avail = set()
        for c in pool.selected:
            if cands["source"][c] != q:
                avail |= set(cands["R"][c].tolist())
        assert pool.train_true_counts[q] == len(set(w.train_top_ids[q].tolist()) & avail)

def test_threshold_requires_prefix_longer_than_budget():
    base, w = workload(4)
    try:
        eb.threshold_candidates(w, PREFIX)
        raise RuntimeError("expected AssertionError")
    except AssertionError:
        pass

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
