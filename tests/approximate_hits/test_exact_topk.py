#!/usr/bin/env python3
"""Tests for the blocked exact top-k used on large (10M) databases."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from simulations.approximate_hits.run_e2e_oracle_sift import exact_topk_blocked


def _brute(queries, db, k, tie_by_id):
    d = ((queries[:, None, :].astype(np.float64) - db[None, :, :].astype(np.float64)) ** 2).sum(-1)
    out = []
    for row in d:
        ids = np.arange(len(row))
        order = np.lexsort((ids, row)) if tie_by_id else np.argsort(row, kind="stable")
        out.append(order[:k])
    return np.array(out), d


def test_int8_matches_brute_force_with_id_tie_break():
    rng = np.random.default_rng(0)
    db = rng.integers(-3, 4, size=(3000, 8)).astype(np.int8)  # many exact ties
    q = rng.integers(-3, 4, size=(50, 8)).astype(np.int8)
    got = exact_topk_blocked(q, db, 20, q_chunk=16, db_block=700)
    want, _ = _brute(q, db, 20, tie_by_id=True)
    assert np.array_equal(got, want)


def test_float_matches_brute_force_distances():
    rng = np.random.default_rng(1)
    db = rng.standard_normal((5000, 16))
    q = rng.standard_normal((40, 16))
    got = exact_topk_blocked(q, db, 15, q_chunk=7, db_block=1100)
    want, d = _brute(q, db, 15, tie_by_id=False)
    # same neighbor distances (float32 arithmetic may swap exact near-ties)
    got_d = np.take_along_axis(d, got, axis=1)
    want_d = np.take_along_axis(d, want, axis=1)
    assert np.allclose(got_d, want_d, rtol=1e-5, atol=1e-6)
    assert np.mean(np.sort(got, axis=1) == np.sort(want, axis=1)) > 0.99
