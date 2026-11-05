#!/usr/bin/env python3
"""Tests for the continuous cached-vector count in eval_radius_models."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from simulations.approximate_hits.eval_radius_models import soft_count


def test_soft_count_integer_points_and_interpolation():
    dc = np.array([[1.0, 2.0, 4.0, 8.0]] * 5)
    r = np.array([0.5, 2.0, 3.0, 8.0, 9.0])
    # 0.5: halfway to the 1st; 2.0: exactly 2; 3.0: halfway 2 -> 3; >= last: all
    assert np.allclose(soft_count(dc, r), [0.125, 0.5, 0.625, 1.0, 1.0])


def test_soft_count_monotone():
    dc = np.sort(np.random.default_rng(0).random((1, 10)), axis=1).repeat(200, axis=0)
    r = np.linspace(0, 1.2, 200)
    assert np.all(np.diff(soft_count(dc, r)) >= -1e-12)
