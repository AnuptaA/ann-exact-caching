#!/usr/bin/env python3
"""Tests for the Aker and QVCache baselines of the online simulator."""

import argparse


import numpy as np

from simulations.approximate_hits import online_baselines as ob


def _data(n_db=300, n_q=40, d=3, seed=0):
    rng = np.random.default_rng(seed)
    db = rng.standard_normal((n_db, d)).astype(np.float32)
    q = rng.standard_normal((n_q, d)).astype(np.float32)
    top = np.argsort(((q[:, None] - db[None]) ** 2).sum(-1), axis=1)[:, :50]
    return {"db": db, "q": q, "top": top}


def _args(**kw):
    a = dict(k=5, budget=0.1, warmup=10, batch=5, aker_alpha_l=1.1, aker_alpha_t=0.25,
             qv_mini=2, qv_D=0.0, qv_alpha=0.9, qv_dims=2, qv_buckets=4)
    a.update(kw)
    return argparse.Namespace(**a)


def test_aker_insert_tau_and_equality_hit():
    D = _data()
    ak = ob.Aker(D, np.arange(40), _args())
    ak._insert(0)
    tau, min_d = ak.reps[0]
    assert np.isclose(tau, min_d / 4) and ak.n_cached() == 5
    served, ret = ak.batch(np.array([0]), 0)             # exact repeat
    assert served[0] and ret[0] == 1.0 and np.isclose(ak.reps[0][0], tau * 1.1)


def test_aker_approximate_hit_shrinks_tau_and_aliases():
    D = _data()
    D["q"][1] = D["q"][0] + 1e-6                          # near-duplicate of query 0
    ak = ob.Aker(D, np.arange(40), _args())
    ak._insert(0)
    tau = ak.reps[0][0]
    served, _ = ak.batch(np.array([1]), 0)
    assert served[0] and np.isclose(ak.reps[0][0], tau * 0.25) and ak.alias[1] == 0
    served, _ = ak.batch(np.array([1]), 1)                # alias hit
    assert served[0] and np.isclose(ak.reps[0][0], tau * 0.25 ** 2)


def test_aker_fifo_eviction_within_budget():
    D = _data()
    ak = ob.Aker(D, np.arange(40), _args(budget=12 / 300))  # room for about 2 entries
    for c in (0, 1, 2, 3):
        ak._insert(c)
    assert ak.n_cached() <= 12 and 0 not in ak.reps and 3 in ak.reps
    assert ak.n_cached() == int((ak.ref > 0).sum())


def test_qvcache_threshold_learning_and_mini_eviction():
    D = _data()
    qv = ob.QVCache(D, np.arange(40), _args(budget=20 / 300))  # 2 mini-indexes of 10
    served, _ = qv.batch(np.array([0]), 0)               # empty cache: miss
    assert not served[0]
    key = qv.region(0)
    dk = np.linalg.norm(D["q"][0] - D["db"][D["top"][0, 4]])
    assert np.isclose(qv.theta[key], dk)
    served, ret = qv.batch(np.array([0]), 1)              # same query: d_cache[k] = theta
    assert served[0] and ret[0] == 1.0
    for q in range(1, 8):                                 # fill and force evictions
        qv._insert(q, 10 + q)
    assert qv.n_cached() <= 20 and qv.n_cached() == int((qv.ref > 0).sum())
    assert any(e["event"] == "evict_mini" for e in qv.events)
