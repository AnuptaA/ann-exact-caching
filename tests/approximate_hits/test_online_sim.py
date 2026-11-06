#!/usr/bin/env python3
"""Tests for the online simulator's cache, features, drift bound, streams."""


import argparse

import numpy as np
import pytest

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


def _micro(seed=0, n_db=600, n_q=400, d=6, R=60, n_clusters=4):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, d)) * 3
    db = (centers[rng.integers(0, n_clusters, n_db)] + rng.standard_normal((n_db, d))).astype(np.float32)
    lab = rng.integers(0, n_clusters, n_q)
    q = (centers[lab] + rng.standard_normal((n_q, d))).astype(np.float32)
    top = np.argsort(((q[:, None, :] - db[None]) ** 2).sum(-1), axis=1)[:, :R].astype(np.int32)
    return {"db": db, "q": q, "top": top, "draw": np.random.default_rng(1).random(n_db)}, rng.permutation(n_q)


def _args(**kw):
    a = dict(k=5, target=0.9, delta=0.05, budget=0.2, f=0.85, R=60, bits=4, batch=40, warmup=100,
             audit_rate=0.2, cal_share=0.5, n_cal=80, epoch=100, window=300, cand_share=0.2,
             drift_window=30, drift_queries=80, min_window=100, check_every=100, gain=0.1,
             delta_alarm=0.05, c_max=50, s_max=300, k_lru=20, aker_alpha_l=1.1, aker_alpha_t=0.25,
             qv_mini=4, qv_D=0.075, qv_alpha=0.9, qv_dims=4, qv_buckets=4, no_recal=False,
             no_ret_alarm=False, no_score_alarm=False, no_shadow=False, tag="", seed=0,
             max_queries=None, out=None)
    a.update(kw)
    return argparse.Namespace(**a)


@pytest.mark.parametrize("method", ["static", "periodic", "triggered", "lru"])
def test_every_method_runs_and_logs_consistently(method):
    D, stream = _micro()
    res = osim.Sim(D, stream, _args()).run(method)
    n = sum(b["n"] for b in res["log"])
    assert n == len(stream) - 100                             # every post-warm-up query logged
    for b in res["log"]:
        assert 0 <= b["served"] <= b["n"] and 0 <= b["audited_served"] <= b["served"]
        assert b["served_retention_sum"] <= b["served"] + 1e-9
    assert res["cost"]["topk"] >= 0 and res["cost"]["warmup_topR"] == 100


@pytest.mark.parametrize("method", ["static", "periodic", "triggered", "lru"])
def test_unreachable_target_and_tiny_budget_do_not_crash(method):
    D, stream = _micro()
    for kw in (dict(target=1.0), dict(budget=0.005), dict(f=0.0)):
        res = osim.Sim(D, stream, _args(**kw)).run(method)
        assert sum(b["served"] for b in res["log"]) >= 0


def test_stream_barely_longer_than_warmup_and_ragged_last_batch():
    D, stream = _micro()
    res = osim.Sim(D, stream[:101], _args(batch=40)).run("triggered")
    assert [b["n"] for b in res["log"]] == [1]
    res = osim.Sim(D, stream[:237], _args(batch=40)).run("periodic")
    assert [b["n"] for b in res["log"]] == [40, 40, 40, 17]


def test_invalid_configurations_are_rejected_up_front():
    D, stream = _micro()
    with pytest.raises(ValueError, match="warm-up"):
        osim.Sim(D, stream[:100], _args()).run("static")
    with pytest.raises(ValueError, match="at least 5"):
        osim.Sim(D, stream, _args(warmup=3)).run("static")
    with pytest.raises(ValueError, match="k_lru"):
        osim.Sim(D, stream, _args(k_lru=61)).run("lru")
    with pytest.raises(ValueError, match="k=70"):
        osim.Sim(D, stream, _args(k=70)).run("static")
    with pytest.raises(ValueError, match="audit_rate"):
        osim.Sim(D, stream, _args(cal_share=1.0)).run("static")


def test_rebuilds_need_build_role_audits():
    D, stream = _micro()
    sim = osim.Sim(D, stream, _args(audit_rate=0.0))
    sim.window = [(0, False, False, False)]
    with pytest.raises(ValueError, match="build-role audits"):
        sim.rebuild_inputs()

