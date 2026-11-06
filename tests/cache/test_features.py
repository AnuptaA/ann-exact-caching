#!/usr/bin/env python3
"""Tests for the per-split feature inputs (utils.cache.features)."""

import numpy as np
import pytest

from utils.base.knn import exact_topk
from utils.cache.entries import build_entry_tables
from utils.cache.features import N_BALLS, POOR_FIT, SplitInputs, ball_features, shell_block, split_inputs
from utils.cache.shell import SAMPLE_J, shell_features, shell_sample
from utils.cache.workloads import RunData


def _run(n_db=600, n_cand=15, k_values=(5, 30, 100), k=3, n_q=30, seed=0):
    rng = np.random.default_rng(seed)
    db = rng.standard_normal((n_db, 4))
    cand_q = rng.standard_normal((n_cand, 4))
    cached_ids, radius, _, _ = build_entry_tables(cand_q, db, list(k_values), 1)
    sets = {s: (rng.standard_normal((n_q, 4)), rng.integers(1, 4, n_q)) for s in ("fit", "cal", "test")}
    topk = {s: exact_topk(q, db, k) for s, (q, _) in sets.items()}
    return RunData(k=k, k_values=list(k_values), db=db, cand_q=cand_q, sets=sets, topk=topk,
                   cached_ids=cached_ids, radius=radius)


def test_split_inputs_match_direct_computation():
    run = _run()
    entries = [0, 4, 8, 12]                         # candidates 0 (K=5), 1 (K=30), 2 (K=100), 4 (K=30)
    inp = split_inputs(run, entries, ("fit", "cal", "test"))
    assert isinstance(inp, SplitInputs)
    in_cache = np.zeros(len(run.db), dtype=bool)
    for j in entries:
        in_cache[run.cached_ids[j]] = True
    for sp in ("fit", "cal", "test"):
        assert np.allclose(inp.y[sp], in_cache[run.topk[sp]].mean(axis=1))
        assert np.array_equal(inp.w[sp], run.sets[sp][1].astype(float))
        assert inp.B[sp].shape == (len(run.sets[sp][0]), 2 * N_BALLS) and np.isfinite(inp.B[sp]).all()
        d = np.linalg.norm(run.sets[sp][0][:, None] - run.db[np.flatnonzero(in_cache)][None], axis=2)
        assert np.allclose(inp.nr[sp], np.sort(d, axis=1)[:, run.k - 1])
    assert inp.bl["cand"].tolist() == [0, 1, 2, 4]


def test_split_inputs_none_for_tiny_cache_and_optional_balls():
    run = _run(k=5)
    assert split_inputs(run, [], ("fit",)) is None
    assert split_inputs(run, [0], ("fit",)) is None              # exactly k = 5 cached vectors
    inp = split_inputs(run, [4], ("fit",), n_balls=None)
    assert inp.B == {} and set(inp.nr) == {"fit"}
    one = split_inputs(run, [4], ("fit",))                       # one ball: remaining columns POOR_FIT
    assert (one.B["fit"][:, 2:] == POOR_FIT).all()


def test_shell_block_matches_manual_columns():
    run = _run()
    run.cand_ext = None
    inp = split_inputs(run, [0, 4], ("fit", "test"))
    samp, S = shell_block(run, inp, 1.0, 100, 32)
    direct = shell_sample(run, inp.bl, inp.st["in_cache"], 1.0, 100, 32)
    assert samp is not None and set(samp["ids"]) == set(direct["ids"])
    for sp in ("fit", "test"):
        full = shell_features(run.sets[sp][0], inp.nr[sp], direct)
        assert np.allclose(S[sp], full)
    _, one = shell_block(run, inp, 1.0, 100, 32, [SAMPLE_J])
    assert one["fit"].shape[1] == 1 and np.allclose(one["fit"][:, 0], S["fit"][:, SAMPLE_J])
    none_samp, zero = shell_block(run, inp, 0.0, 100, 32)
    assert none_samp is None and (zero["fit"] == 0).all()


def test_ball_features_pad_when_fewer_balls_than_requested():
    bl = {"c": np.zeros((1, 2)), "c_sq": np.zeros(1), "r": np.array([2.0])}
    F = ball_features(np.array([[1.0, 0.0]]), np.array([0.5]), bl, 3)
    assert np.allclose(F[0, :2], [0.5, 0.25]) and np.isnan(F[0, 2:]).all()
