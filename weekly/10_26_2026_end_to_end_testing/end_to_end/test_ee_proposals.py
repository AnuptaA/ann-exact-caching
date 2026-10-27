#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Tests for ee_proposals on a small synthetic workload. Run with pytest or `python3 -B test_ee_proposals.py`."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import ee_incidence as ei
import ee_proposals as ep
import ee_workload as ew

D, N_BASE, N, K, MAX_SIZE = 8, 400, 60, 10, 49

def workload(seed=0, lam=0.5):
    rng = np.random.default_rng(seed)
    unit = lambda n: (lambda v: v / np.linalg.norm(v, axis=1, keepdims=True))(rng.standard_normal((n, D)))
    base, pool = unit(N_BASE), unit(300)
    w = ew.build_workload("sift", base, pool, seed, lam, n=N, K=K, per_anchor=10, prefix_len=MAX_SIZE + 1)
    return base, w

def test_own_candidates():
    base, w = workload()
    c = ep.own_candidates(w)
    assert len(c["R"]) == N and all(len(r) == K for r in c["R"])
    for q in (0, 7, 41):
        assert set(c["R"][q].tolist()) == set(w.train_top_ids[q].tolist()) and c["G"][q] == [q]

def test_group_union_levels_and_union_sets():
    base, w = workload(1)
    c = ep.group_union_candidates(w, max_i=5, max_size=1000)
    assert len(c["R"]) == N * 6 and c["dropped"] == 0
    assert np.array_equal(c["level"][:6], np.arange(6))
    own = ep.own_candidates(w)
    assert set(c["R"][0].tolist()) == set(own["R"][0].tolist())  # i = 0 equals Own
    for q in (3, 20):
        for i in range(6):
            k = q * 6 + i
            group = [q] + w.tt_order[q, :i].tolist()
            assert c["G"][k] == group
            assert set(c["R"][k].tolist()) == set(w.train_top_ids[group].ravel().tolist())
            assert len(set(c["R"][k].tolist())) == len(c["R"][k])
    small = ep.group_union_candidates(w, max_i=5, max_size=15)  # sets above the cap are dropped
    assert small["dropped"] > 0 and all(len(r) <= 15 for r in small["R"])
    assert len(small["R"]) + small["dropped"] == N * 6

def test_ball_radius_membership_and_drop_rule():
    base, w = workload(2)
    c = ep.ball_candidates(w, max_i=5, max_size=MAX_SIZE)
    dist = np.arccos(np.clip(w.train @ base.T, -1, 1)) / np.pi
    kept = dropped = 0
    k = 0
    for q in range(N):
        nbrs = w.tt_order[q, :5]
        for i in range(1, 6):
            radius = w.tt_dist[q, nbrs[i - 1]] + w.theta_train[nbrs[i - 1]]
            members = np.flatnonzero(dist[q] <= radius)
            if len(members) > MAX_SIZE:
                dropped += 1
                continue
            assert c["source"][k] == q and c["level"][k] == i
            assert c["G"][k] == [q] + nbrs[:i].tolist()
            assert set(c["R"][k].tolist()) == set(members.tolist())
            kept += 1
            k += 1
    assert kept == len(c["R"]) and dropped == c["dropped"] and kept + dropped == N * 5

def test_ball_prefix_must_hold_one_extra_vector():
    base, w = workload(3)
    try:
        ep.ball_candidates(w, max_i=3, max_size=MAX_SIZE - 1)
        raise RuntimeError("expected AssertionError")
    except AssertionError:
        pass

def test_make_incidence_runs_the_engine():
    base, w = workload(4)
    for kind, kw in (("own", {}), ("group_union", dict(max_i=3)), ("ball", dict(max_i=3, max_size=MAX_SIZE))):
        inc, cands = ep.make_incidence(w, kind, n_vec=N_BASE, **kw)
        idx = ei.build_index(inc)
        assert inc.n_cand == len(cands["R"]) and (inc.cand_source == cands["source"]).all()
        r = ei.greedy(inc, idx, "SEL3", 0.9, budget=60)
        assert r.pool_size <= 60 and len(r.selected) > 0
        ref = ei.greedy_reference(inc, "SEL3", 0.9, 60)
        assert r.selected.tolist() == ref["selected"].tolist()

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
