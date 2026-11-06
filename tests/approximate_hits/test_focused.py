#!/usr/bin/env python3
"""Tests for focus weighting in eval_focused."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from simulations.approximate_hits.eval_focused import focus_weights


def test_focus_weights_boost_top_region_only():
    rng = np.random.default_rng(0)
    score = rng.random(2000)
    y = np.clip(score + 0.05 * rng.standard_normal(2000), 0, 1)
    w = np.ones(2000)
    w2 = focus_weights(score, y, w, 0.9, 5.0)
    assert set(np.unique(w2)) == {1.0, 5.0}
    assert score[w2 == 5.0].min() >= score[w2 == 1.0].max()
    assert 0.02 <= np.mean(w2 == 5.0) <= 0.5
