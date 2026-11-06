#!/usr/bin/env python3
"""Tests for the top-3 and region features in eval_features_v2."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from simulations.approximate_hits import eval_features_v2 as ef
from utils.base.knn import sq_norms


def _state(centers, radii):
    cq = np.asarray(centers, dtype=np.float64)
    return {"ent_q": cq, "ent_q_sq": sq_norms(cq), "ent_r": np.asarray(radii, dtype=np.float64)}


def _setup(n_probes=64):
    ang = np.linspace(0, 2 * np.pi, n_probes, endpoint=False)
    ef._R["U"] = np.column_stack([np.cos(ang), np.sin(ang)])


def test_region_fully_covered_and_uncovered():
    _setup()
    st = _state([[0.0, 0.0], [100.0, 0.0]], [10.0, 1.0])
    q = np.array([[0.0, 0.0], [50.0, 0.0]])
    ck = np.array([2.0, 2.0])
    top, region = ef.geometry_features(q, ck, st)
    assert np.allclose(region[0], 0.0)        # neighborhood inside the big ball
    assert np.allclose(region[1], 1.0)        # far from every ball
    # best-fitting ball for query 0 is ball 0: depth 0, size 0.2
    assert np.isclose(top[0, 0], 0.0) and np.isclose(top[0, 1], 0.2)


def test_region_half_covered_by_union():
    _setup(400)
    # two unit balls side by side cover the left and right halves of the
    # boundary of a radius-1 neighborhood around the origin only partly
    st = _state([[-1.0, 0.0], [1.0, 0.0]], [1.0, 1.0])
    _, region = ef.geometry_features(np.zeros((1, 2)), np.array([1.0]), st)
    # boundary point at angle a is inside the right ball iff |a| < 60 degrees
    assert abs(region[0, 0] - 1.0 / 3.0) < 0.02
