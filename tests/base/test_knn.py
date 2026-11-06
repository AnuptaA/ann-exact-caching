#!/usr/bin/env python3
"""Tests for utils.base.knn: distances and exact top-k (incl. blocked, for large databases)."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from utils.base.knn import (
    exact_topk, exact_topk_blocked, fetch_rows, pairwise_dist, sq_dist, sq_norms,
)


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


def test_sq_norms_and_distances_match_direct_formula():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((6, 5))
    b = rng.standard_normal((9, 5))
    direct = ((a[:, None, :] - b[None]) ** 2).sum(-1)
    assert np.allclose(sq_norms(a), (a ** 2).sum(1))
    assert np.allclose(pairwise_dist(a, b), np.sqrt(direct))
    assert np.allclose(pairwise_dist(a, b, sq_norms(b)), np.sqrt(direct))
    assert np.allclose(sq_dist(a, b, sq_norms(b)), direct)
    assert (sq_dist(a, a, sq_norms(a)) >= 0).all()          # clipped at zero


def test_exact_topk_matches_brute_force_and_blocked():
    rng = np.random.default_rng(3)
    db = rng.standard_normal((800, 12))
    q = rng.standard_normal((30, 12))
    got = exact_topk(q, db, 10, chunk=7)
    want, _ = _brute(q, db, 10, tie_by_id=False)
    assert np.array_equal(got, want)
    blocked = exact_topk_blocked(q, db, 10, q_chunk=8, db_block=250)
    assert np.mean(np.sort(blocked, axis=1) == np.sort(got, axis=1)) > 0.99


def test_fetch_rows_returns_rows_in_id_order_float64():
    db = np.arange(40, dtype=np.float32).reshape(10, 4)
    ids = np.array([7, 2, 2, 9])
    rows = fetch_rows(db, ids)
    assert rows.dtype == np.float64
    assert np.array_equal(rows, db[ids].astype(np.float64))

