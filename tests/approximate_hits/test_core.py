#!/usr/bin/env python3
"""Tests for eval_core: balls, ball features, shell sample, quantization."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from simulations.approximate_hits import eval_calibrated as ecal
from simulations.approximate_hits import eval_core as ec


def _setup():
    rng = np.random.default_rng(0)
    db = rng.standard_normal((600, 3))
    cand_q = rng.standard_normal((4, 3))
    k_values = [10, 50, 500]
    d = np.linalg.norm(cand_q[:, None] - db[None], axis=2)
    order = np.argsort(d, axis=1)
    cached_ids = [order[i, :K] for i in range(4) for K in k_values]
    radius = np.array([np.sort(d[i])[K - 1] for i in range(4) for K in k_values])
    ecal._G.update(db=db, cand_q=cand_q, k_values=k_values, cached_ids=cached_ids,
                   radius=radius)
    ec._R["draw"] = rng.random(len(db))
    return db, cand_q, order, d


def test_one_ball_per_center_with_largest_k():
    _setup()
    entries = [0 * 3 + 0, 0 * 3 + 1, 2 * 3 + 0]  # center 0 at K = 10 and 50; center 2 at 10
    bl = ec.balls_of(entries)
    assert bl["cand"].tolist() == [0, 2] and bl["K"].tolist() == [50, 10]
    assert np.isclose(bl["r"][0], ecal._G["radius"][0 * 3 + 1])


def test_ball_features_rank_by_fit():
    _setup()
    bl = {"c": np.array([[0.0, 0, 0], [10.0, 0, 0]]), "r": np.array([5.0, 1.0])}
    bl["c_sq"] = (bl["c"] ** 2).sum(axis=1)
    F = ec.ball_features(np.array([[1.0, 0, 0]]), np.array([1.0]), bl, 3)
    # ball 0 fits best: offset 1 / 5, size 1 / 5; ball 1: 9 / 1, 1 / 1; third missing
    assert np.allclose(F[0, :4], [0.2, 0.2, 9.0, 1.0])
    assert np.isnan(F[0, 4:]).all()


def test_shell_sample_rank_cap_and_memory():
    db, _, order, _ = _setup()
    entries = [1 * 3 + 0]                          # center 1, K = 10
    in_cache = np.zeros(len(db), dtype=bool)
    in_cache[order[1, :10]] = True
    bl = ec.balls_of(entries)
    samp = ec.shell_sample(bl, in_cache, 0.5, 100, 8)
    shell = set(order[1, 10:100].tolist())
    assert set(samp["ids"].tolist()) == {v for v in shell if ec._R["draw"][v] < 0.5}
    assert np.isclose(samp["memory"], len(samp["ids"]) * 8 / 32 / len(db))


def test_quantize_error_bounded():
    x = np.random.default_rng(1).standard_normal((200, 4))
    for bits in (2, 4, 8):
        step = (x.max(axis=0) - x.min(axis=0)) / (2 ** bits - 1)
        assert (np.abs(ec.quantize(x, bits) - x) <= step / 2 + 1e-12).all()
    assert ec.quantize(x, 32) is x


def test_oracle_at_interpolates_in_log_size():
    pts = [{"cache_frac": 0.01, "test_hit_rate": 0.1}, {"cache_frac": 0.1, "test_hit_rate": 0.3}]
    assert np.isclose(ec.oracle_at(pts, 10 ** -1.5), 0.2)
    assert ec.oracle_at(pts, 0.2) is None
