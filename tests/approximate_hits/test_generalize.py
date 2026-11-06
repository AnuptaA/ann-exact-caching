#!/usr/bin/env python3
"""Tests for helpers in eval generalize."""


import numpy as np

from simulations.approximate_hits.eval import generalize as eg


def test_optimal_rate_splits_ties():
    y = np.array([1.0, 1.0, 0.8, 0.8, 0.8, 0.0])
    # top 4 average 0.9 (ties at 0.8 may be split)
    assert eg.optimal_rate(y, 0.9) == 4 / 6


def test_stack_appends_constant_column():
    a = (np.ones((2, 3)), np.array([0.1, 0.2]))
    b = (np.zeros((1, 3)), np.array([0.3]))
    X, y = eg._stack([a, b], [5.0, 7.0])
    assert X.shape == (3, 4) and np.allclose(X[:, 3], [5, 5, 7])
    assert np.allclose(y, [0.1, 0.2, 0.3])
    assert np.allclose(eg._with(np.ones((2, 1)), 2.0), [[1, 2], [1, 2]])


def test_fit_logistic_ranks_monotone_signal():
    import pytest
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    X = rng.random((500, 2))
    y = np.clip(X[:, 0] + 0.05 * rng.standard_normal(500), 0, 1)
    s = eg.fit_logistic(X, y)(X)
    assert np.corrcoef(s, X[:, 0])[0, 1] > 0.95
