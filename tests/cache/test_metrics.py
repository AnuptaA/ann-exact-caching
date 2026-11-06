#!/usr/bin/env python3
"""Tests for utils.cache.metrics."""

import numpy as np
import pytest

from utils.cache.metrics import (
    admit, best_cutoff_rate, expand_index, interp_log, nearest_point, result_for_target, state_tasks,
)


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


def test_admit_handles_no_threshold_and_empty_admission():
    s, y = np.array([0.9, 0.5, 0.1]), np.array([1.0, 0.8, 0.0])
    assert admit(0.4, s, y) == (2 / 3, 0.9)
    assert admit(None, s, y) == (0.0, None)
    assert admit(2.0, s, y) == (0.0, None)
    assert admit(0.0, s, y) == (1.0, 0.6)


def _data():
    pts = [{"cache_frac": 0.01, "n_entries": 2}, {"cache_frac": 0.1, "n_entries": 5}]
    return {"results": [{"target": 0.9, "points": pts, "selected": list(range(10))},
                        {"target": 0.7, "points": [], "selected": []}]}


def test_result_lookup_nearest_point_and_state_tasks():
    data = _data()
    res = result_for_target(data, 0.9)
    with pytest.raises(ValueError, match="not in this run"):
        result_for_target(data, 0.95)
    assert nearest_point(res, 0.02)["n_entries"] == 2 and nearest_point(res, 0.5)["n_entries"] == 5
    with pytest.raises(ValueError, match="no checkpoints"):
        nearest_point(data["results"][1], 0.1)
    tasks = state_tasks(res, [0.01, 0.1])
    assert [(t, p["n_entries"], e) for t, p, e in tasks] == [(0.9, 2, [0, 1]), (0.9, 5, [0, 1, 2, 3, 4])]
    assert state_tasks(res, [0.1], 0.8)[0][0] == 0.8

