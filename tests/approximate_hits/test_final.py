#!/usr/bin/env python3
"""Tests for helpers in eval_final."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from simulations.approximate_hits import eval_final as ef
from simulations.approximate_hits.threshold_selection import threshold_grid


def test_interp_log():
    assert np.isclose(ef.interp_log([0.01, 0.1], [0.1, 0.3], 10 ** -1.5), 0.2)
    assert ef.interp_log([0.01, 0.1], [0.1, 0.3], 0.2) is None
    assert ef.interp_log([0.1, 0.01], [0.3, 0.1], 0.01) == 0.1   # unsorted input


def test_admit_and_none_lambda():
    s, y = np.array([0.9, 0.5, 0.1]), np.array([1.0, 0.8, 0.0])
    assert ef.admit(0.4, s, y) == (2 / 3, 0.9)
    assert ef.admit(None, s, y) == (0.0, None)


def test_calibration_sweep_skips_large_n_and_is_sane():
    rng = np.random.default_rng(0)
    s = rng.random(4000)
    y = np.clip(s + 0.05 * rng.standard_normal(4000), 0, 1)
    out = ef.calibration_sweep(s[:2000], y[:2000], threshold_grid(s), s[2000:], y[2000:], 0.9,
                               [500, 2000, 5000], 5, np.random.default_rng(1))
    assert set(out) == {"500", "2000"}
    assert out["2000"]["mean_hit_rate"] >= out["500"]["min_hit_rate"]
    assert 0 <= out["500"]["frac_retention_below_T"] <= 1
