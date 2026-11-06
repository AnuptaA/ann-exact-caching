#!/usr/bin/env python3
"""Tests for the Aker and QVCache baselines of the online simulator."""

import argparse


import numpy as np
import pytest

from utils.cache import baselines as ob


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


def _consistent(b):
    cached = b.ref > 0
    assert b.n_c == int(cached.sum()) and (b.ref >= 0).all()


def test_qvcache_every_insert_leaves_its_own_topk_fully_cached():
    D = _data(n_db=300, n_q=40)
    qv = ob.QVCache(D, np.arange(40), _args(budget=40 / 300, qv_mini=2))   # 2 minis of 20, k = 5
    evictions = 0
    for t, q in enumerate(range(40)):
        qv._insert(q, t)
        ids = D["top"][q, :5]
        assert (qv.ref[ids] > 0).all(), f"insert {q} left part of its top-k uncached"
        _consistent(qv)
        assert (qv.owner[ids] >= 0).all()
        assert qv.n_c <= qv.M + 5                           # bounded by the budget (plus one entry)
        for m, members in enumerate(qv.mini):
            assert all(qv.owner[v] == m for v in members) and qv.size[m] == len(members)
        evictions = sum(e["event"] == "evict_mini" for e in qv.events)
    assert evictions > 0


def test_qvcache_single_mini_index_and_too_small_budget():
    D = _data(n_db=300, n_q=40)
    qv = ob.QVCache(D, np.arange(40), _args(budget=12 / 300, qv_mini=1))
    for t, q in enumerate(range(30)):
        qv._insert(q, t)
        assert (qv.ref[D["top"][q, :5]] > 0).all()
        _consistent(qv)
    with pytest.raises(ValueError, match="cannot hold one entry"):
        ob.QVCache(D, np.arange(40), _args(budget=20 / 300, qv_mini=8))      # 2.5 < k = 5
    with pytest.raises(ValueError, match="cannot hold one entry"):
        ob.Aker(D, np.arange(40), _args(budget=3 / 300))
    with pytest.raises(ValueError, match="warm-up"):
        ob.QVCache(D, np.arange(40), _args(warmup=1))


def test_aker_eviction_never_leaves_dangling_aliases_or_counts():
    D = _data(n_db=300, n_q=40)
    D["q"][1] = D["q"][0] + 1e-6                             # near-duplicates become aliases
    D["q"][3] = D["q"][2] + 1e-6
    ak = ob.Aker(D, np.arange(40), _args(budget=12 / 300))   # room for about two entries
    for rnd in range(3):
        ak.batch(np.arange(40), 0)
        _consistent(ak)
        assert all(rep in ak.reps for rep in ak.alias.values())
        assert set(ak.children) <= set(ak.reps)
        assert set(ak.fifo) >= set(ak.reps)
        assert ak.n_cached() <= ak.M + 5


def test_baselines_handle_repeated_queries_in_one_batch():
    D = _data(n_db=300, n_q=40)
    stream = np.tile(np.arange(8), 6)
    for cls in (ob.Aker, ob.QVCache):
        b = cls(D, stream, _args(budget=60 / 300, batch=12, warmup=10))
        out = b.run()
        _consistent(b)
        assert sum(r["served"] for r in out["log"]) <= len(stream)

