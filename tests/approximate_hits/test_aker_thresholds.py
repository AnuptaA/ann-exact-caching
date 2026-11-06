#!/usr/bin/env python3
"""Tests for the Aker hit-rule replay in eval_aker_thresholds."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from simulations.approximate_hits.eval_aker_thresholds import simulate


def _run(stream, alias_update="approx"):
    # one representative (global id 0) with minD = 4, so tau starts at 1
    rep_of_gid = {0: 0}
    near = {1: 0, 2: 0, 3: 0}
    near_d = {1: 0.9, 2: 0.5, 3: 2.0}
    return simulate(stream, len(stream), rep_of_gid, near, near_d,
                    np.array([1.0]), np.array([4.0]), 0.25, 1.1, alias_update)


def test_hit_types_and_tau_updates():
    # 1: approx hit (0.9 < 1), tau -> 0.25; 2: miss (0.5 >= 0.25);
    # 1 again: alias hit despite tau; 0: equality hit; 3: miss (beyond cap)
    kind, used = _run([1, 2, 1, 0, 3])
    assert kind.tolist() == [3, 0, 2, 1, 0]
    assert used.tolist() == [0, -1, 0, 0, -1]


def test_equality_hits_grow_tau_up_to_cap():
    # tau 1 -> 1.1^n, capped at minD = 4; then distance 2.0 becomes a hit
    kind, _ = _run([0] * 20 + [3])
    assert kind[-1] == 3


def test_reported_suffix_only():
    rep_of_gid, near, near_d = {0: 0}, {1: 0}, {1: 0.5}
    kind, used = simulate([0, 1, 1], 1, rep_of_gid, near, near_d,
                          np.array([1.0]), np.array([4.0]), 0.25, 1.1, "approx")
    assert kind.tolist() == [2] and used.tolist() == [0]
