#!/usr/bin/env python3
"""Tests for the online simulator's cache, features, drift bound, streams."""


import numpy as np

from simulations.approximate_hits import online_prepare as op
from simulations.approximate_hits import online_sim as osim


def _data(n_db=400, n_q=30, d=4, R=60, seed=0):
    rng = np.random.default_rng(seed)
    db = rng.standard_normal((n_db, d)).astype(np.float32)
    q = rng.standard_normal((n_q, d)).astype(np.float32)
    top = np.argsort(((q[:, None] - db[None]) ** 2).sum(-1), axis=1)[:, :R]
    return {"db": db, "q": q, "top": top, "draw": rng.random(n_db)}


def test_split_clusters_balanced_and_stream_ordered():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 10, 5000)
    group = op.split_clusters(labels, rng)
    qg = group[labels]
    assert abs((qg == 0).sum() - (qg == 1).sum()) < 0.2 * len(labels)
    order, phase = op.shift_stream(labels, group, rng)
    assert sorted(order) == list(range(len(labels)))
    assert (np.diff(phase) >= 0).all() and (qg[order] == phase).all()


def _sim(D):
    import argparse
    a = argparse.Namespace(k=5, target=0.9, delta=0.05, budget=0.2, f=0.85, R=60, bits=4,
                           audit_rate=0.1, cal_share=0.5, n_cal=100, window=1000, c_max=50,
                           s_max=1000, cand_share=0.2, seed=0)
    return osim.Sim(D, np.arange(len(D["q"])), a)


def test_rebuild_inputs_exclude_calibration_role_and_weight_served():
    D = _data(n_q=30)
    sim = _sim(D)
    # (query, served, audited, cal role)
    sim.window = [(0, False, False, False), (1, True, False, False), (2, True, True, False),
                  (3, True, True, True), (4, False, True, True), (5, False, True, False)]
    qs, wts = sim.rebuild_inputs()
    assert set(qs.tolist()) == {0, 2, 5}            # unaudited served and cal role excluded
    w = dict(zip(qs.tolist(), wts.tolist()))
    assert w[0] == 1 and w[5] == 1 and w[2] == round((1 - 0.05) / 0.05)


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
