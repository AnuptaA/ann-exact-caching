#!/usr/bin/env python3
"""Tests for cache perturbation in eval_threshold_transfer."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from simulations.approximate_hits.eval_threshold_transfer import perturbed


def test_perturbed_keeps_k_and_replaces_fraction():
    n_cand, n_kv = 200, 3
    rng = np.random.default_rng(0)
    base = sorted(rng.choice(n_cand * n_kv, 60, replace=False).tolist())
    out = perturbed(base, 0.3, n_cand, n_kv, np.random.default_rng(1))
    assert len(out) == len(base) == len(set(out))
    changed = [a != b for a, b in zip(base, out)]
    assert sum(changed) == 18
    assert all(a % n_kv == b % n_kv for a, b in zip(base, out))
    assert not set(o for o, c in zip(out, changed) if c) & set(base)
