#!/usr/bin/env python3
"""Tests for utils.cache.metrics."""

import numpy as np

from utils.cache.metrics import best_cutoff_rate, interp_log


def test_best_cutoff_rate():
    s = np.array([5, 4, 3, 2, 1.0])
    y = np.array([1, 1, 0.8, 0.0, 1.0])
    # top 3 average 0.933 >= 0.9; top 4 average 0.7
    assert best_cutoff_rate(s, y, 0.9) == 0.6


def test_best_cutoff_rate_respects_ties():
    s = np.array([2, 1, 1, 1.0])
    y = np.array([1, 1, 0.0, 0.0])
    # cannot split the tied group: only the top query qualifies
    assert best_cutoff_rate(s, y, 0.9) == 0.25


def test_interp_log():
    assert np.isclose(interp_log([0.01, 0.1], [0.1, 0.3], 10 ** -1.5), 0.2)
    assert interp_log([0.01, 0.1], [0.1, 0.3], 0.2) is None
    assert interp_log([0.1, 0.01], [0.3, 0.1], 0.01) == 0.1   # unsorted input
