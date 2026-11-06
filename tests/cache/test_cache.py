#!/usr/bin/env python3
"""Tests for the online cache state (utils.cache.cache)."""

import numpy as np
import pytest

from utils.base.knn import knn_dists
from utils.cache.cache import Cache
from utils.cache.features import N_BALLS, POOR_FIT, ball_features


def _data(n_db=400, n_q=30, d=4, R=60, seed=0):
    rng = np.random.default_rng(seed)
    db = rng.standard_normal((n_db, d)).astype(np.float32)
    q = rng.standard_normal((n_q, d)).astype(np.float32)
    top = np.argsort(((q[:, None] - db[None]) ** 2).sum(-1), axis=1)[:, :R]
    return {"db": db, "q": q, "top": top, "draw": rng.random(n_db)}


def test_add_replace_refcounts_and_fetch_counting():
    D = _data()
    c = Cache(D, k=5, R=60, bits=4, budget=1e9)
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
    c = Cache(D, k=5, R=60, bits=8, budget=1e9)
    c.add(0, 20)
    c.p = 1.0
    shell = set(D["top"][0, 20:60])
    assert np.isclose(c.memory(), 20 + len(shell) * 8 / 32)
    c.p = 0.0
    assert np.isclose(c.memory(), 20)


def test_features_match_brute_force():
    D = _data()
    c = Cache(D, k=5, R=60, bits=32, budget=1e9)
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


def test_retention_of_cached_query_is_one():
    D = _data()
    c = Cache(D, k=5, R=60, bits=4, budget=1e9)
    c.add(3, 5)
    assert c.retention(np.array([3]))[0] == 1.0


def test_ball_features_agree_with_offline_extractor():
    D = _data()
    c = Cache(D, k=5, R=60, bits=32, budget=1e9)
    for i in (0, 1, 2):
        c.add(i, 20)
    c.p = 1.0
    qidx = np.arange(3, 15)
    X, _ = c.features(qidx)
    q = D["q"][qidx].astype(np.float64)
    bl = {"c": c.cq.astype(np.float64), "c_sq": c.cq_sq.astype(np.float64), "r": c.radius}
    nr = knn_dists(q, c.cv.astype(np.float64), (c.cv.astype(np.float64) ** 2).sum(1), 5)[:, 4]
    B = np.nan_to_num(ball_features(q, nr + 1e-12, bl, N_BALLS), nan=POOR_FIT)
    assert np.allclose(X[:, :2 * N_BALLS], B, atol=1e-4)


def test_entry_size_must_fit_the_stored_neighbors():
    D = _data(R=60)
    c = Cache(D, k=5, R=60, bits=4, budget=1e9)
    for bad in (0, 61):
        with pytest.raises(ValueError, match="between 1 and"):
            c.add(0, bad)


def test_cache_with_fewer_than_k_vectors_cannot_serve_and_features_do_not_crash():
    D = _data()
    c = Cache(D, k=5, R=60, bits=4, budget=1e9)
    X, near = c.features(np.arange(6))                       # no entries at all
    assert not c.can_serve and X.shape == (6, 7) and (near == -1).all()
    assert np.allclose(X, 10.0)
    c.add(0, 3)                                              # 3 cached vectors < k = 5
    assert not c.can_serve
    X, near = c.features(np.arange(6))
    assert np.allclose(X, 10.0) and (near == -1).all()
    c.add(1, 10)
    assert c.can_serve
    X, near = c.features(np.arange(6))
    assert (near >= 0).all() and not np.allclose(X, 10.0)


def test_remove_and_resize_keep_reference_counts_consistent():
    D = _data()
    c = Cache(D, k=5, R=60, bits=4, budget=1e9)
    c.add(0, 20)
    c.add(1, 20)
    c.add(0, 10)                                              # shrink an entry
    assert (c.ref >= 0).all()
    c.remove(1)
    assert c.n_cached() == 10 and set(np.flatnonzero(c.ref > 0)) == set(D["top"][0, :10])
    c.remove(0)
    assert c.n_cached() == 0 and (c.ref == 0).all()

