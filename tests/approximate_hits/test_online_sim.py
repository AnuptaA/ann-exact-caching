#!/usr/bin/env python3
"""Tests for the online simulator's cache, features, drift bound, streams."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from simulations.approximate_hits import online_prepare as op
from simulations.approximate_hits import online_sim as osim


def _data(n_db=400, n_q=30, d=4, R=60, seed=0):
    rng = np.random.default_rng(seed)
    db = rng.standard_normal((n_db, d)).astype(np.float32)
    q = rng.standard_normal((n_q, d)).astype(np.float32)
    top = np.argsort(((q[:, None] - db[None]) ** 2).sum(-1), axis=1)[:, :R]
    return {"db": db, "q": q, "top": top, "draw": rng.random(n_db)}


def test_add_replace_refcounts_and_fetch_counting():
    D = _data()
    c = osim.Cache(D, k=5, R=60, bits=4, budget=1e9)
    c.add(0, 20)
    c.add(1, 10)
    assert c.n_cached() == len(set(D["top"][0, :20]) | set(D["top"][1, :10]))
    fetched = c.vectors_fetched
    c.replace({0: 20, 2: 10})                   # keep 0, drop 1, add 2
    assert set(c.entries) == {0, 2}
    assert c.n_cached() == len(set(D["top"][0, :20]) | set(D["top"][2, :10]))
    # only vectors not already cached count as fetched
    assert c.vectors_fetched - fetched == len(set(D["top"][2, :10]) - set(D["top"][0, :20])
                                              - set(D["top"][1, :10]))
    c.add(0, 10)                                # resize down
    assert c.entries[0] == 10 and (c.ref >= 0).all()


def test_memory_counts_sample_at_bits_over_32():
    D = _data()
    c = osim.Cache(D, k=5, R=60, bits=8, budget=1e9)
    c.add(0, 20)
    c.p = 1.0
    shell = set(D["top"][0, 20:60])
    assert np.isclose(c.memory(), 20 + len(shell) * 8 / 32)
    c.p = 0.0
    assert np.isclose(c.memory(), 20)


def test_features_match_brute_force():
    D = _data()
    c = osim.Cache(D, k=5, R=60, bits=32, budget=1e9)
    for i in (0, 1, 2):
        c.add(i, 20)
    c.p = 1.0
    qidx = np.arange(3, 8)
    X, near = c.features(qidx)
    cached = np.flatnonzero(c.ref > 0)
    for row, qi in enumerate(qidx):
        d = np.linalg.norm(D["db"][cached] - D["q"][qi], axis=1)
        nr = np.sort(d)[4]
        assert set(near[row]) == set(cached[np.argsort(d)[:5]])
        rad = [np.linalg.norm(D["q"][e] - D["db"][D["top"][e, 19]]) for e in (0, 1, 2)]
        off = [np.linalg.norm(D["q"][qi] - D["q"][e]) for e in (0, 1, 2)]
        fits = sorted(((o + nr) / r, o / r, nr / r) for o, r in zip(off, rad))
        assert np.allclose(X[row, :6], [v for f in fits for v in f[1:]], atol=1e-4)
        sid = c.shell_ids()
        cnt = (np.linalg.norm(D["db"][sid] - D["q"][qi], axis=1) < nr).sum()
        assert np.isclose(X[row, 6], np.log1p(cnt))


def test_retention_and_hoeffding():
    D = _data()
    c = osim.Cache(D, k=5, R=60, bits=4, budget=1e9)
    c.add(3, 5)
    assert c.retention(np.array([3]))[0] == 1.0
    up = osim.hoeffding_upper(np.full(100, 0.8), 0.05)
    assert np.isclose(up, 0.8 + np.sqrt(np.log(20) / 200))


def test_split_clusters_balanced_and_stream_ordered():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 10, 5000)
    group = op.split_clusters(labels, rng)
    qg = group[labels]
    assert abs((qg == 0).sum() - (qg == 1).sum()) < 0.2 * len(labels)
    order, phase = op.shift_stream(labels, group, rng)
    assert sorted(order) == list(range(len(labels)))
    assert (np.diff(phase) >= 0).all() and (qg[order] == phase).all()


def _sim(D, **kw):
    import argparse
    a = dict(k=5, target=0.9, delta=0.05, budget=0.2, f=0.85, R=60, bits=4, audit_rate=0.1,
             cal_share=0.5, cal_span=1000, window=1000, c_max=50, s_max=1000, cand_share=0.2,
             min_window=5, gain=0.1, oracle_cal=False, k_lru=5, seed=0)
    a.update(kw)
    return osim.Sim(D, np.arange(len(D["q"])), argparse.Namespace(**a))


def test_weighted_bound_matches_unweighted_and_weights_matter():
    from simulations.approximate_hits import threshold_selection as ts
    rng = np.random.default_rng(0)
    s = rng.random(3000)
    y = np.clip(s + 0.1 * rng.standard_normal(3000), 0, 1)
    grid = ts.threshold_grid(s)
    assert ts.select_weighted(s, y, np.ones(3000), grid, 0.8, 0.05) == \
        ts.select("normal_fixed", s, y, grid, 0.8, 0.05)
    # up-weighting low-retention queries can only make lambda stricter
    w = np.where(y < 0.5, 10.0, 1.0)
    lw = ts.select_weighted(s, y, w, grid, 0.8, 0.05)
    lu = ts.select_weighted(s, y, np.ones(3000), grid, 0.8, 0.05)
    assert lw is None or lw >= lu


def test_two_prop_pvalue():
    assert osim.two_prop_pvalue(50, 100, 50, 100) == 1.0
    assert osim.two_prop_pvalue(80, 100, 20, 100) < 1e-6
    assert osim.two_prop_pvalue(0, 100, 0, 100) == 1.0


def test_lru_eviction_never_empties_cache():
    D = _data(n_q=30)
    sim = _sim(D, budget=0.05)                     # M = 20 vectors
    for c in range(5):
        sim.cache.add(c, 20)                       # far over budget
        sim.cache.last_used[c] = 0
    qb = np.arange(5, 30)
    serve = np.zeros(len(qb), bool)
    sim.lru_update(qb, serve, np.zeros((len(qb), 5), np.int64), 1, 0.0)
    assert len(sim.cache.entries) >= 1 and sim.cache.memory() <= sim.M + 5
    # the newest entries survive, the old ones were evicted first
    assert not set(range(5)) & set(sim.cache.entries)


def test_rebuild_holds_out_calibration_and_weights_hits(monkeypatch):
    D = _data(n_q=30)
    sim = _sim(D, cal_share=0.3)
    sim.model = lambda X: -X[:, 0]
    sim.cache.add(0, 20)
    sim.window = [(q, q % 3 == 0, q % 6 == 0, 0.2) for q in range(30)]
    seen = []

    def fake(qs, wts):
        seen.append((set(qs.tolist()), dict(zip(qs.tolist(), wts.tolist()))))
        return {int(qs[0]): 20}
    monkeypatch.setattr(sim, "_greedy", fake)
    monkeypatch.setattr(osim, "hits_of_entries", lambda *a: 10 ** 6)
    assert sim.try_rebuild(1)
    built, w = seen[-1]
    # unaudited hits never build; audited hits weigh 10 / 0.2, misses 10
    assert not any(q % 3 == 0 and q % 6 != 0 for q in built)
    assert all(w[q] == (50 if q % 6 == 0 else 10) for q in built)
    # calibration of the new cache uses only held-out queries, none built on
    held = 30 - len(built) - sum(q % 3 == 0 and q % 6 != 0 for q in range(30))
    assert len(sim.cal) == 1 and len(sim.ref) == 1
    assert len(sim.cal[0][1]) <= held + 5 and len(sim.ref[0][1]) >= len(sim.cal[0][1])


def test_greedy_candidates_disjoint_from_scoring(monkeypatch):
    D = _data(n_q=30)
    sim = _sim(D)
    seen = {}

    def fake(Dd, cand, score, score_w, budget, k, target):
        seen["cand"] = {c for c, _ in cand}
        seen["score"] = set(np.asarray(score).tolist())
        return {}, 0
    monkeypatch.setattr(osim, "greedy_entries", fake)
    sim._greedy(np.arange(30), np.ones(30, dtype=np.int64))
    assert seen["cand"] and not (seen["cand"] & seen["score"])


def test_write_streams_trace_keeps_order_and_repeats(tmp_path):
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 6, 50)
    requests = rng.integers(0, 50, 400)              # repeated requests
    natural = (np.arange(400) >= 300).astype(np.int8)
    op.write_streams(tmp_path, labels, requests, rng, natural)
    z = np.load(tmp_path / "streams.npz")
    assert sorted(z["random"]) == sorted(requests) and sorted(z["shift"]) == sorted(requests)
    assert np.array_equal(z["trace"], requests) and np.array_equal(z["phase_trace"], natural)
    g = z["group"][labels[z["shift"]]]
    assert np.array_equal(g, z["phase_shift"]) and (np.diff(g) >= 0).all()
    # original order kept within each half
    for h in (0, 1):
        sub = requests[z["group"][labels[requests]] == h]
        assert np.array_equal(z["shift"][g == h], sub)


def test_multishift_phases_mix_and_no_reuse():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 12, 6000)
    requests = rng.permutation(6000)
    order, phase, group, mild, L = op.multishift_stream(labels, requests, rng, warmup=1000,
                                                       stable=200, mild=0.2)
    assert len(np.unique(order)) == len(order)                 # each request at most once
    assert (np.diff(phase) >= 0).all() and set(phase.tolist()) == {0, 1, 2, 3}
    lab = labels[requests[order]]
    A = group[lab] == 0
    assert A[phase == 0].all() and A[phase == 3].all() and not A[phase == 2].any()
    assert abs(np.isin(lab[phase == 1], mild["add"]).mean() - 0.2) < 0.02
    assert not np.isin(lab[phase == 1], mild["drop"]).any()
    # the return phase has the start mix
    mix = lambda m: np.bincount(lab[m], minlength=12) / m.sum()
    assert np.abs(mix(phase == 0) - mix(phase == 3)).sum() < 0.02
    assert all(abs((phase == i).sum() - L) <= 12 for i in (2, 3))


def _dyn_data(n_db=300, n_q=20, d=3, seed=1):
    """Database with a full ranking per query and a presence mask."""
    rng = np.random.default_rng(seed)
    db = rng.standard_normal((n_db, d)).astype(np.float32)
    q = rng.standard_normal((n_q, d)).astype(np.float32)
    rank = np.argsort(((q[:, None] - db[None]) ** 2).sum(-1), axis=1).astype(np.int32)
    present = rng.random(n_db) < 0.8
    return {"db": db, "q": q, "top": rank, "rank": rank, "present": present,
            "draw": rng.random(n_db)}


def test_topk_present_matches_brute_force():
    D = _dyn_data()
    ids = osim.topk_present(D, np.arange(5), 7)
    for i in range(5):
        pres = np.flatnonzero(D["present"])
        d = ((D["db"][pres] - D["q"][i]) ** 2).sum(-1)
        assert list(ids[i]) == list(pres[np.argsort(d)[:7]])


def _ball_exact(c, D, center):
    """Every present vector inside the entry's ball is cached."""
    pe, ps, rad = c.geo[center]
    d = np.sqrt(((D["db"] - D["q"][center]) ** 2).sum(-1))
    inside = D["present"] & (d <= rad + 1e-6)
    return bool(c.cached[inside].all())


def test_maintained_ball_stays_exact_under_inserts_and_deletes():
    D = _dyn_data()
    c = osim.Cache(D, k=3, R=40, bits=32, budget=1e9)
    c.add(0, 10)
    assert c.n_cached() == 10 and _ball_exact(c, D, 0)
    row = D["rank"][0]
    absent_inside = [i for i in row[:c.geo[0][0]] if not D["present"][i]]
    gone = int(osim.topk_present(D, [0], 1)[0, 0])
    # insert absent vectors inside the ball, delete the nearest member
    D["present"][absent_inside] = True
    D["present"][gone] = False
    c.data_changed()
    assert _ball_exact(c, D, 0) and not c.cached[gone]
    assert c.n_cached() == 10 + len(absent_inside) - 1
    # radius is fixed at creation
    c.prepare()
    assert np.isclose(c.geo[0][2], c.radius[0])


def test_unmaintained_ball_ignores_inserts_but_drops_deletes():
    D = _dyn_data()
    c = osim.Cache(D, k=3, R=40, bits=32, budget=1e9, maintain=False)
    c.add(0, 10)
    row = D["rank"][0]
    absent_inside = [i for i in row[:c.geo[0][0]] if not D["present"][i]]
    member = int(osim.topk_present(D, [0], 1)[0, 0])
    D["present"][absent_inside] = True
    D["present"][member] = False
    c.data_changed()
    assert not c.cached[absent_inside].any() and not c.cached[member]
    assert c.n_cached() == 9


def test_apply_events_updates_presence_and_costs():
    D = _dyn_data()
    pres0 = D["present"].copy()
    ins = np.flatnonzero(~pres0)[:4]
    dele = np.flatnonzero(pres0)[:3]
    D["events"] = {"t": np.array([5, 5, 5, 5, 9, 9, 9]), "id": np.r_[ins, dele],
                   "insert": np.r_[np.ones(4, bool), np.zeros(3, bool)]}
    sim = _sim(D, R=40, method="greedy")
    sim.cache.add(0, 10)
    sim.cache.add(1, 10)
    assert sim.apply_events(4) == 0
    assert sim.apply_events(6) == 4 and D["present"][ins].all()
    assert sim.cost["center_checks"] == 4 * 2
    assert sim.apply_events(20) == 3 and not D["present"][dele].any()


def test_retention_alarm_fires_only_on_low_retention():
    D = _data()
    sim = _sim(D, target=0.9)
    sim.audits = [(1000, np.full(400, 0.95), np.ones(400))]
    assert not sim.retention_alarm(1000, 0.01)
    sim.audits = [(1000, np.full(400, 0.8), np.ones(400))]
    assert sim.retention_alarm(1000, 0.01)


def test_approximate_rows_recall_order_and_padding():
    rng = np.random.default_rng(0)
    rows = np.stack([rng.permutation(5000)[:400] for _ in range(300)]).astype(np.int32)
    rows[:5, 380:] = -1                                     # padded rows stay padded
    assert np.array_equal(osim.approximate_rows(rows, 1.0), rows)
    out = osim.approximate_rows(rows, 0.9)
    for r, o in zip(rows[:20], out[:20]):
        keep = o[o >= 0]
        assert list(keep) == [x for x in r if x in set(keep.tolist())]    # subsequence, in order
        assert (o[len(keep):] == -1).all()                                # tail padded
    # recall at k equals the keep probability, at any k
    for k in (10, 50):
        assert abs(osim.measured_recall(rows, out, k, n=300) - 0.9) < 0.02
    assert np.array_equal(osim.approximate_rows(rows, 0.9), out)          # deterministic
    assert not np.array_equal(osim.approximate_rows(rows, 0.9, seed=1), out)


def test_backend_answers_vs_truth_retention():
    D = _dyn_data()
    osim.db_view(D)
    assert D["truth"] is D["rank"]                          # exact backend: same array
    c = osim.Cache(D, k=3, R=40, bits=32, budget=1e9)
    c.add(0, 10)
    q = np.arange(10)
    assert np.array_equal(c.retention(q), c.retention_exact(q))
    D["truth"] = D["rank"]
    D["rank"] = D["answer"] = osim.approximate_rows(D["truth"], 0.7)   # approximate backend
    c2 = osim.Cache(D, k=3, R=40, bits=32, budget=1e9)
    c2.add(0, 10)
    # entries are built from the backend's answer; the exact truth is separate
    members = set(c2.members(0).tolist())
    assert members <= set(D["rank"][0].tolist())
    assert not np.array_equal(c2.retention(q), c2.retention_exact(q))


def test_shell_scope_balls_matches_all_with_three_entries_and_is_subset():
    D = _data()
    c = osim.Cache(D, k=5, R=60, bits=32, budget=1e9)
    for i in (0, 1, 2):
        c.add(i, 20)
    c.p = 1.0
    qidx = np.arange(3, 12)
    X_all, _ = c.features(qidx)
    c.shell_scope = "balls"
    X_b, _ = c.features(qidx)
    assert np.allclose(X_all, X_b)                 # 3 entries: the 3 balls are all of them
    c.add(5, 20)
    c.add(6, 20)
    c.shell_scope = "all"
    X_all, _ = c.features(qidx)
    c.shell_scope = "balls"
    X_b, _ = c.features(qidx)
    assert np.allclose(X_all[:, :6], X_b[:, :6])   # ball features unchanged
    assert (X_b[:, 6] <= X_all[:, 6] + 1e-12).all()   # restricted count <= full count
