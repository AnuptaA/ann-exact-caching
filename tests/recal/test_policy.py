import numpy as np
import pytest

from recal import policy
from recal.history import reuse_distances
from tests.recal.reference import prototype as ref


def synthetic_items(seed=0, n=4000):
    rng = np.random.default_rng(seed)
    score = rng.uniform(size=n)
    y = np.clip(0.55 + 0.45 * score + rng.normal(scale=0.08, size=n), 0, 1)
    w = np.where(rng.uniform(size=n) < 0.2, 10.0, 1.0)
    return score, y, w


def test_certify_matches_prototype_binned_deque():
    score, y, w = synthetic_items()
    train = np.random.default_rng(1).uniform(size=3000)
    edges = policy.threshold_edges(train, 100)
    window = policy.CalibrationWindow(edges, W=10**9, target=0.9, delta=0.05)
    proto = ref.BinnedDeque(edges[::-1], size=len(score), target=0.9)
    for i, (s, yy, ww) in enumerate(zip(score, y, w)):
        window.add(i, s, yy, ww)
        proto.push(int(proto.bin_of(s)), yy, ww)
    got = window.certify()
    lam, ess = proto.lam()
    assert got.lam == lam and got.lam is not None
    assert got.ess == pytest.approx(ess)


def test_certify_undefined_when_nothing_passes():
    score, y, w = synthetic_items()
    edges = policy.threshold_edges(score, 100)
    window = policy.CalibrationWindow(edges, W=10**9, target=0.99, delta=0.05)
    for i, (s, yy, ww) in enumerate(zip(score, y * 0.5, w)):
        window.add(i, s, yy, ww)
    assert window.certify().lam is None


def test_ess_floor_skips_strict_thresholds():
    edges = np.linspace(0, 1, 100, endpoint=False)
    window = policy.CalibrationWindow(edges, W=10**9, target=0.5, delta=0.05, min_ess=100)
    rng = np.random.default_rng(2)
    for i in range(150):
        window.add(i, rng.uniform(), 1.0, 1.0)
    got = window.certify()
    assert got.lam is not None and got.ess >= 100
    assert got.lam <= np.sort([s for _, s, _, _ in window.items])[0] + 1.0   # sanity: a real threshold


def test_window_expires_by_main_path_requests():
    edges = np.linspace(0, 1, 100, endpoint=False)
    window = policy.CalibrationWindow(edges, W=100, target=0.5, delta=0.05)
    for i in range(300):
        window.add(i, 0.5, 1.0, 1.0)
    assert len(window) == 100
    assert window.sums[0].sum() == pytest.approx(100.0)
    window.expire(10_000)
    assert len(window) == 0 and np.allclose(window.sums, 0)


def test_window_size_rule():
    assert policy.window_size(30_000, 0.1, 0.2) == (10_000, False)
    assert policy.window_size(30_000, 0.1, 0.01) == (30_000, True)
    assert policy.window_size(30_000, 0.1, 0.9) == (2223, False)   # rounded up


def test_logistic_scores_rank_and_signs():
    rng = np.random.default_rng(3)
    F = rng.normal(size=(2000, 3))
    y = 1 / (1 + np.exp(-(1.0 - 1.5 * F[:, 0] - 0.5 * F[:, 1] - 0.8 * F[:, 2])))
    model = policy.logistic_model(F, y)
    assert np.all(model["coef"] < 0)
    s = policy.score_features(F, model)
    assert np.corrcoef(s, y)[0, 1] > 0.9


def test_reuse_distances():
    seq = np.array([1, 2, 3, 1, 2, 2, 4, 1], dtype=np.uint64)
    # repeats: 1 at 3 (distinct between: 2,3 -> 2), 2 at 4 (3,1 -> 2), 2 at 5 (none -> 0), 1 at 7 (2,4 -> 2)
    assert reuse_distances(seq).tolist() == [2, 2, 0, 2]
    assert len(reuse_distances(np.arange(10, dtype=np.uint64))) == 0
