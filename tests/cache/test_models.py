#!/usr/bin/env python3
"""Tests for utils.cache.models."""

import numpy as np
import pytest

from utils.cache.models import fit_logistic, fit_ridge, gbt, gbt_regressor, logistic_scores


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    y = 1 / (1 + np.exp(-(X[:, 0] - 0.5 * X[:, 1])))          # soft labels in (0, 1)
    return X, y


def test_logistic_orders_scores_like_the_signal_and_unit_weights_match_default():
    X, y = _data()
    score = fit_logistic(X, y)
    assert np.corrcoef(score(X), X[:, 0] - 0.5 * X[:, 1])[0, 1] > 0.95
    assert np.allclose(score(X), fit_logistic(X, y, np.ones(len(y)))(X))


def test_logistic_scores_fit_on_fit_split_and_score_every_split():
    X, y = _data()
    w = np.ones(len(y))
    out = logistic_scores({"fit": X, "cal": X[:50]}, y, w)
    assert set(out) == {"fit", "cal"} and np.allclose(out["cal"], out["fit"][:50])


def test_zero_weights_are_rejected():
    X, y = _data(20)
    with pytest.raises(ValueError, match="positive"):
        fit_ridge(X, y, np.zeros(20))
    with pytest.raises(ValueError, match="positive"):
        fit_logistic(X, y, np.zeros(20))
    with pytest.raises(ValueError):
        fit_logistic(X[:0], y[:0])


def test_ridge_recovers_a_quadratic_signal_and_weights_matter():
    rng = np.random.default_rng(1)
    F = rng.standard_normal((500, 2))
    y = 0.3 + 0.5 * F[:, 0] + 0.2 * F[:, 1] ** 2
    pred = fit_ridge(F, y, np.ones(500))(F)
    assert np.corrcoef(pred, y)[0, 1] > 0.99
    w = np.where(F[:, 0] > 0, 100.0, 1.0)
    biased = fit_ridge(F, y + (F[:, 0] < 0) * 5.0, w)(F)
    assert np.abs(biased[F[:, 0] > 0] - y[F[:, 0] > 0]).mean() < np.abs(
        biased[F[:, 0] < 0] - y[F[:, 0] < 0]).mean()


def test_gbt_factories():
    assert type(gbt("reg")).__name__ == "HistGradientBoostingRegressor"
    assert type(gbt("cls")).__name__ == "HistGradientBoostingClassifier"
    q = gbt("quantile", 0.2)
    assert q.loss == "quantile" and q.quantile == 0.2
    assert gbt_regressor(3).random_state == 3
