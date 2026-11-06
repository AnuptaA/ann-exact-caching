#!/usr/bin/env python3
"""Tests for utils.cache.entries."""

import numpy as np

from utils.cache.entries import entry_kmax, perturbed


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


def test_entry_kmax_keeps_largest_k_per_candidate():
    k_values = [10, 20, 50]
    # candidate 0 at K=10 and K=50, candidate 2 at K=20
    entries = [0 * 3 + 0, 0 * 3 + 2, 2 * 3 + 1]
    cand, K = entry_kmax(entries, k_values)
    assert cand.tolist() == [0, 2] and K.tolist() == [50, 20]
