#!/usr/bin/env python3
"""Tests for the Aker and QVCache baselines of the online simulator."""

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from simulations.approximate_hits import online_baselines as ob
from simulations.approximate_hits import online_sim as osim


def _data(n_db=300, n_q=40, d=3, seed=0, full=False):
    rng = np.random.default_rng(seed)
    db = rng.standard_normal((n_db, d)).astype(np.float32)
    q = rng.standard_normal((n_q, d)).astype(np.float32)
    top = np.argsort(((q[:, None] - db[None]) ** 2).sum(-1), axis=1)
    return {"db": db, "q": q, "top": top if full else top[:, :50]}


def _args(**kw):
    a = dict(k=5, budget=0.1, warmup=10, batch=5, aker_alpha_l=1.1, aker_alpha_t=0.25,
             aker_tau_init=0.25, aker_cap=1.0, aker_reserve=0, aker_risk=0.3, aker_batch=16,
             qv_mini=2, qv_D=0.0, qv_alpha=0.9, qv_dims=2, qv_buckets=4, qv_deletes="keep")
    a.update(kw)
    return argparse.Namespace(**a)


def _truth(D, qs, k=5):
    return osim.topk_present(osim.db_view(D), np.asarray(qs), k)


def test_aker_insert_tau_and_equality_hit():
    D = _data()
    ak = ob.Aker(D, np.arange(40), _args())
    ak._insert(0)
    e = ak.reps[0]
    assert np.isclose(e["tau"], e["d"][0] / 4) and ak.n_cached() == 5
    tau = e["tau"]
    served, ret, _x = ak.batch(np.array([0]), 0, _truth(D, [0]))     # exact repeat
    assert served[0] and ret[0] == 1.0 and np.isclose(ak.reps[0]["tau"], tau * 1.1)


def test_aker_approximate_hit_shrinks_tau_and_aliases():
    D = _data()
    D["q"][1] = D["q"][0] + 1e-6                          # near-duplicate of query 0
    ak = ob.Aker(D, np.arange(40), _args())
    ak._insert(0)
    tau = ak.reps[0]["tau"]
    served, _, _x = ak.batch(np.array([1]), 0, _truth(D, [1]))
    assert served[0] and np.isclose(ak.reps[0]["tau"], tau * 0.25) and ak.alias[1] == 0
    served, _, _x = ak.batch(np.array([1]), 1, _truth(D, [1]))      # alias hit
    assert served[0] and np.isclose(ak.reps[0]["tau"], tau * 0.25 ** 2)


def test_aker_fifo_eviction_within_budget():
    D = _data()
    ak = ob.Aker(D, np.arange(40), _args(budget=12 / 300))  # room for about 2 entries
    for c in (0, 1, 2, 3):
        ak._insert(c)
    assert ak.n_cached() <= 12 and 0 not in ak.reps and 3 in ak.reps
    assert ak.n_cached() == int((ak.ref > 0).sum())


def test_aker_delete_promotes_reserve_then_drops_entry():
    D = _data(full=True)
    ak = ob.Aker(D, np.arange(40), _args(aker_reserve=2))
    ak._insert(0)
    ids = ak.reps[0]["ids"].copy()
    assert len(ids) == 7
    D["present"][ids[0]] = False
    ak.on_delete(ids[:1])
    assert list(ak.reps[0]["ids"]) == list(ids[1:])            # reserve moved up
    assert ak.n_cached() == 6 and ak.ref[ids[0]] == 0
    D["present"][ids[1:3]] = False
    ak.on_delete(ids[1:3])                                      # 4 left < k = 5
    assert 0 not in ak.reps and ak.n_cached() == 0 and not ak.holders


def test_aker_fast_path_takes_closer_insert():
    D = _data(full=True)
    D["present"] = np.ones(300, dtype=bool)
    new = int(D["top"][0, 0])                  # nearest vector to query 0, absent at first
    D["present"][new] = False
    ak = ob.Aker(D, np.arange(40), _args())
    ak._insert(0)
    assert new not in ak.reps[0]["ids"]
    D["present"][new] = True
    ak.on_insert(np.array([new]))
    assert ak.reps[0]["ids"][0] == new and len(ak.reps[0]["ids"]) == 5
    assert ak.n_cached() == int((ak.ref > 0).sum()) == 5


def test_aker_slow_path_reaches_other_entries():
    D = _data(full=True)
    D["present"] = np.ones(300, dtype=bool)
    new = int(D["top"][1, 0])
    D["present"][new] = False
    ak = ob.Aker(D, np.arange(40), _args(aker_risk=0.0))       # always refresh
    ak._insert(0)
    ak._insert(1)
    D["present"][new] = True
    ak.on_insert(np.array([new]))
    for _ in range(4):
        ak.slow_step()
    assert ak.reps[1]["ids"][0] == new
    assert all(e["lc"] == 1 for e in ak.reps.values())


def test_qvcache_threshold_learning_and_mini_eviction():
    D = _data()
    qv = ob.QVCache(D, np.arange(40), _args(budget=20 / 300))  # 2 mini-indexes of 10
    served, _, _x = qv.batch(np.array([0]), 0, _truth(D, [0]))     # empty cache: miss
    assert not served[0]
    key = qv.region(0)
    dk = np.linalg.norm(D["q"][0] - D["db"][D["top"][0, 4]])
    assert np.isclose(qv.theta[key], dk)
    served, ret, _x = qv.batch(np.array([0]), 1, _truth(D, [0]))    # d_cache[k] = theta
    assert served[0] and ret[0] == 1.0
    for q in range(1, 8):                                 # fill and force evictions
        qv._insert(q, 10 + q)
    assert qv.n_cached() <= 20 and qv.n_cached() == int((qv.ref > 0).sum())
    assert any(e["event"] == "evict_mini" for e in qv.events)


def test_qvcache_deletes_kept_or_removed():
    for mode, cached_after in (("keep", 5), ("remove", 4)):
        D = _data(full=True)
        qv = ob.QVCache(D, np.arange(40), _args(qv_deletes=mode))
        qv._insert(0, 0)
        gone = int(_truth(D, [0])[0, 0])
        D["present"][gone] = False
        qv.on_delete(np.array([gone]))
        assert qv.n_cached() == cached_after == int((qv.ref > 0).sum())
        served, ret, _x = qv.batch(np.array([0]), 1, _truth(D, [0]))
        if mode == "keep" and served[0]:
            assert qv.cost["deleted_returned"] == 1 and ret[0] < 1
