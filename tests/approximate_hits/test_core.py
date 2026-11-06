#!/usr/bin/env python3
"""Tests for eval_core: balls, ball features, shell sample, quantization."""


import numpy as np

from simulations.approximate_hits import eval_core as ec
from utils.cache.features import ball_features, balls_of
from utils.cache.shell import quantize, shell_sample
from utils.cache.workloads import RunData


def _setup():
    rng = np.random.default_rng(0)
    db = rng.standard_normal((600, 3))
    cand_q = rng.standard_normal((4, 3))
    k_values = [10, 50, 500]
    d = np.linalg.norm(cand_q[:, None] - db[None], axis=2)
    order = np.argsort(d, axis=1)
    cached_ids = [order[i, :K] for i in range(4) for K in k_values]
    radius = np.array([np.sort(d[i])[K - 1] for i in range(4) for K in k_values])
    run = RunData(k=10, k_values=k_values, db=db, cand_q=cand_q, sets={}, topk={},
                  cached_ids=cached_ids, radius=radius)
    run.draw = rng.random(len(db))
    return run, order


def test_one_ball_per_center_with_largest_k():
    run, _ = _setup()
    entries = [0 * 3 + 0, 0 * 3 + 1, 2 * 3 + 0]  # center 0 at K = 10 and 50; center 2 at 10
    bl = balls_of(run, entries)
    assert bl["cand"].tolist() == [0, 2] and bl["K"].tolist() == [50, 10]
    assert np.isclose(bl["r"][0], run.radius[0 * 3 + 1])


def test_ball_features_rank_by_fit():
    _setup()
    bl = {"c": np.array([[0.0, 0, 0], [10.0, 0, 0]]), "r": np.array([5.0, 1.0])}
    bl["c_sq"] = (bl["c"] ** 2).sum(axis=1)
    F = ball_features(np.array([[1.0, 0, 0]]), np.array([1.0]), bl, 3)
    # ball 0 fits best: offset 1 / 5, size 1 / 5; ball 1: 9 / 1, 1 / 1; third missing
    assert np.allclose(F[0, :4], [0.2, 0.2, 9.0, 1.0])
    assert np.isnan(F[0, 4:]).all()


def test_shell_sample_rank_cap_and_memory():
    run, order = _setup()
    db = run.db
    entries = [1 * 3 + 0]                          # center 1, K = 10
    in_cache = np.zeros(len(db), dtype=bool)
    in_cache[order[1, :10]] = True
    bl = balls_of(run, entries)
    samp = shell_sample(run, bl, in_cache, 0.5, 100, 8)
    shell = set(order[1, 10:100].tolist())
    assert set(samp["ids"].tolist()) == {v for v in shell if run.draw[v] < 0.5}
    assert np.isclose(samp["memory"], len(samp["ids"]) * 8 / 32 / len(db))


def test_quantize_error_bounded():
    x = np.random.default_rng(1).standard_normal((200, 4))
    for bits in (2, 4, 8):
        step = (x.max(axis=0) - x.min(axis=0)) / (2 ** bits - 1)
        assert (np.abs(quantize(x, bits) - x) <= step / 2 + 1e-12).all()
    assert quantize(x, 32) is x


def test_oracle_at_interpolates_in_log_size():
    pts = [{"cache_frac": 0.01, "test_hit_rate": 0.1}, {"cache_frac": 0.1, "test_hit_rate": 0.3}]
    assert np.isclose(ec.oracle_at(pts, 10 ** -1.5), 0.2)
    assert ec.oracle_at(pts, 0.2) is None
