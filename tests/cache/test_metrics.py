#!/usr/bin/env python3
"""Tests for utils.cache.metrics."""

import numpy as np
import pytest

from utils.cache.metrics import best_cutoff_rate, expand_index, interp_log


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


def test_degenerate_inputs():
    assert best_cutoff_rate(np.array([]), np.array([]), 0.9) == 0.0
    assert best_cutoff_rate(np.array([1.0]), np.array([0.0]), 0.9) == 0.0
    assert interp_log([], [], 0.1) is None
    assert interp_log([0.01, 0.1], [0.1, 0.3], 0.0) is None
    assert interp_log([0.0, 0.1], [0.1, 0.3], 0.05) is None


def test_expand_index_repeats_rows_and_rejects_bad_weights():
    assert expand_index(np.array([2, 0, 1])).tolist() == [0, 0, 2]
    assert expand_index(np.array([2.0, 1.0])).tolist() == [0, 0, 1]
    assert len(expand_index(np.array([]))) == 0
    for bad in ([0.5, 1], [-1, 2]):
        with pytest.raises(ValueError, match="non-negative integers"):
            expand_index(np.array(bad))

