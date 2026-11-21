"""Tests for harness config, thresholds, scoring and f32bin files."""

import numpy as np
import pytest

from harness import config, data, scoring, thresholds
from workloads.q1.distance import sq_l2_rows


def test_pool_size_is_exactly_one_five_or_ten_percent():
    assert config.pool_size_for(10_000_000, 1) == 100_000
    assert config.pool_size_for(10_000_000, 10) == 1_000_000
    assert config.pool_size_for(1_814_924, 1) == 18_149           # rounds half up, within one vector
    for bad in (0, 2, 3, 15):
        with pytest.raises(ValueError):
            config.pool_size_for(10_000_000, bad)


def test_settings_validation_and_driver_args():
    std = config.AkerSettings(mode="standard", pool_size=1000)
    args = std.driver_args()
    assert args[args.index("--alpha-tighten") + 1] == "0.25" and "--global-thresh" not in args
    assert args[args.index("--filter-ef-search") + 1] == "16"
    prox = config.AkerSettings(mode="proximity", pool_size=1000, global_thresh=0.5)
    assert "--global-thresh" in prox.driver_args()
    with pytest.raises(ValueError):
        config.AkerSettings(mode="proximity", pool_size=1000)            # needs a threshold
    with pytest.raises(ValueError):
        config.AkerSettings(mode="standard", pool_size=1000, global_thresh=1.0)
    with pytest.raises(ValueError):
        config.AkerSettings(mode="standard", pool_size=5)                # smaller than k
    with pytest.raises(ValueError):
        config.AkerSettings(mode="lru", pool_size=1000)


def test_percentiles_all_pairs_match_brute_force():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(40, 5)).astype(np.float32)
    got = thresholds.pairwise_sq_percentiles(v, (0.1, 1.0, 50.0))
    d = [float(((v[i] - v[j]) ** 2).sum()) for i in range(40) for j in range(i + 1, 40)]
    for p, value in got.items():
        assert value == pytest.approx(np.percentile(d, p), rel=1e-4)


def test_percentiles_sampled_are_seeded_and_close():
    rng = np.random.default_rng(1)
    v = rng.normal(size=(300, 6)).astype(np.float32)
    a = thresholds.pairwise_sq_percentiles(v, (1.0, 50.0), max_pairs=20_000, seed=3)
    b = thresholds.pairwise_sq_percentiles(v, (1.0, 50.0), max_pairs=20_000, seed=3)
    exact = thresholds.pairwise_sq_percentiles(v, (1.0, 50.0), max_pairs=10 ** 9)
    assert a == b
    assert a[50.0] == pytest.approx(exact[50.0], rel=0.05)
    with pytest.raises(ValueError):
        thresholds.pairwise_sq_percentiles(v[:1], (1.0,))


def test_history_vectors_uses_distinct_history_queries_only():
    ids = np.array([5, 5, 7, 9, 9, 11])
    got = thresholds.history_vectors(ids, 4, lambda u: u.reshape(-1, 1).astype(np.float32))
    assert got.ravel().tolist() == [5.0, 7.0, 9.0]


def test_scoring_uses_true_recall_not_hit_flags():
    base = np.zeros((30, 2), dtype=np.float32)
    base[:, 0] = np.arange(30)
    q = np.zeros((4, 2), dtype=np.float32)
    kth = np.full(4, 9.0)
    ids = np.stack([np.arange(10), np.arange(10, 20), np.arange(10), np.arange(5, 15)])
    outcome = np.array([0, 2, 1, 2])         # request 1 is a hit but returns the wrong vectors
    stream = np.array([1, 2, 1, 3])
    score = scoring.score_run(outcome, ids, stream, q, base, kth, history_len=1)
    assert score.recall.tolist() == [1.0, 0.0, 1.0, 0.5]
    s = score.summary()
    assert s["requests_measured"] == 3 and s["hit_rate"] == 1.0
    assert s["delivered_recall"] == pytest.approx((0.0 + 1.0 + 0.5) / 3)
    assert score.equality.tolist() == [False, False, True, False]   # request 2 repeats vector 1
    assert score.approximate.tolist() == [False, True, False, True]
    assert score.seen_before_hit.tolist() == [False, False, True, False]
    assert s["recall_on_misses"] is None
    with pytest.raises(ValueError):
        scoring.score_run(outcome[:3], ids, stream, q, base, kth, 1)


def test_seen_before():
    assert scoring.seen_before(np.array([3, 4, 3, 3, 5, 4])).tolist() == [False, False, True, True, False, True]


def test_f32bin_roundtrip_from_int8_and_float64(tmp_path):
    rng = np.random.default_rng(0)
    for src in (rng.integers(-100, 100, size=(25, 7)).astype(np.int8), rng.normal(size=(25, 7))):
        path = data.write_f32bin(tmp_path / "x.f32bin", src, chunk_rows=10)
        out = data.read_f32bin(path)
        assert out.shape == (25, 7) and out.dtype == np.float32
        assert np.array_equal(out, src.astype(np.float32))
        path.unlink()
    assert sq_l2_rows(out[:3], out[3:6]).shape == (3,)


def test_equality_is_the_driver_code_not_seen_before_and_linked_counts_as_approximate():
    base = np.zeros((30, 2), dtype=np.float32)
    base[:, 0] = np.arange(30)
    ids = np.tile(np.arange(10), (4, 1))
    outcome = np.array([0, 1, 3, 2])
    stream = np.array([5, 5, 5, 5])          # every request repeats, but only code 1 is equality
    score = scoring.score_run(outcome, ids, stream, np.zeros((4, 2), np.float32), base, np.full(4, 9.0), 0)
    assert score.equality.tolist() == [False, True, False, False]
    assert score.approximate.tolist() == [False, False, True, True]
    assert score.seen_before_hit.tolist() == [False, True, True, True]
