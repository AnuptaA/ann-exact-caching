#!/usr/bin/env python3
"""Tests for utils.cache.shell and the feature kernels in utils.cache.features."""

import numpy as np
import pytest

from utils.base.knn import sq_norms
from utils.cache.features import fit_balls, neighborhood_radius
from utils.cache.shell import (
    COUNT_SCALES, SAMPLE_J, ensure_shell_depth, load_cand_ext, quantize, shell_features, shell_ids, shell_sample,
)
from utils.cache.workloads import RunData


def _run(n_db=300, k_values=(10, 50), seed=0):
    rng = np.random.default_rng(seed)
    db = rng.standard_normal((n_db, 3))
    cand_q = rng.standard_normal((3, 3))
    d = np.linalg.norm(cand_q[:, None] - db[None], axis=2)
    order = np.argsort(d, axis=1)
    cached_ids = [order[i, :K] for i in range(3) for K in k_values]
    radius = np.array([np.sort(d[i])[K - 1] for i in range(3) for K in k_values])
    return RunData(k=5, k_values=list(k_values), db=db, cand_q=cand_q, sets={}, topk={},
                   cached_ids=cached_ids, radius=radius), order


def test_quantize_validates_bits_and_handles_constant_columns():
    x = np.ones((5, 2))
    with pytest.raises(ValueError, match="bits"):
        quantize(x, 0)
    assert np.allclose(quantize(x, 4), 1.0)              # hi == lo: no division by zero
    y = np.random.default_rng(0).standard_normal((50, 2))
    one_bit = quantize(y, 1)
    assert all(len(np.unique(one_bit[:, c])) <= 2 for c in range(2))


def test_shell_ids_requires_extended_neighbors_beyond_cached_top():
    run, order = _run()
    assert np.array_equal(shell_ids(run, 1, 10, 50), order[1, 10:50])
    with pytest.raises(ValueError, match="load_cand_ext"):
        shell_ids(run, 1, 10, 80)                         # deeper than the cached top-50
    run.cand_ext = order[:, :60]
    with pytest.raises(ValueError, match="load_cand_ext"):
        shell_ids(run, 1, 10, 80)                         # extension too narrow
    run.cand_ext = order[:, :100]
    assert np.array_equal(shell_ids(run, 1, 10, 80), order[1, 10:80])


def test_load_cand_ext_caches_and_recomputes_on_shape_mismatch(tmp_path):
    run, order = _run()
    ids = load_cand_ext(run, tmp_path, 100)
    assert ids.shape == (3, 100) and np.array_equal(ids, order[:, :100])
    assert (tmp_path / "cand_top100.npy").exists()
    np.save(tmp_path / "cand_top100.npy", np.zeros((2, 100), dtype=int))   # stale file
    again = load_cand_ext(run, tmp_path, 100)
    assert np.array_equal(again, order[:, :100])


def test_shell_sample_extremes_and_features():
    run, order = _run()
    bl = {"cand": np.array([0]), "K": np.array([10])}
    in_cache = np.zeros(len(run.db), dtype=bool)
    in_cache[run.cached_ids[0]] = True
    assert shell_sample(run, bl, in_cache, 0.0, 50, 32) is None          # p = 0: empty
    full = shell_sample(run, bl, in_cache, 1.0, 50, 32)
    assert set(full["ids"].tolist()) == set(order[0, 10:50].tolist())
    assert np.isclose(full["memory"], 40 / len(run.db))
    q = run.db[:4]
    F = shell_features(q, np.full(4, 0.5), full)
    assert F.shape == (4, SAMPLE_J + len(COUNT_SCALES)) and np.isfinite(F).all()
    assert (F[:, SAMPLE_J:] >= 0).all()
    assert (shell_features(q, np.ones(4), None) == 0).all()                # no sample


def test_fit_balls_pads_missing_balls_and_survives_zero_radius():
    nr = np.array([1.0, 2.0])
    de = np.array([[0.5], [3.0]])                       # one ball only
    F = fit_balls(nr, de, np.array([2.0]), 3, np.nan)
    assert np.allclose(F[:, :2], [[0.25, 0.5], [1.5, 1.0]]) and np.isnan(F[:, 2:]).all()
    F0 = fit_balls(nr, de, np.array([0.0]), 3, -1.0)    # duplicate vectors: radius 0
    assert np.isfinite(F0[:, :2]).all()
    assert fit_balls(nr, np.zeros((2, 0)), np.zeros(0), 2, 7.0).tolist() == [[7.0] * 4] * 2


def test_neighborhood_radius_needs_k_cached_vectors():
    db_c = np.random.default_rng(0).standard_normal((4, 3))
    st = {"db_c": db_c, "db_c_sq": sq_norms(db_c)}
    with pytest.raises(ValueError, match="fewer than k"):
        neighborhood_radius(db_c[:2], st, 5)
    assert (neighborhood_radius(db_c[:2], st, 1) < 1e-6).all()     # self distance, never negative


def test_ensure_shell_depth_loads_extended_lists_only_when_needed(tmp_path):
    run, order = _run()
    ensure_shell_depth(run, tmp_path, 50)                 # within the saved top-50: nothing to load
    assert run.cand_ext is None and not list(tmp_path.glob("cand_top*.npy"))
    ensure_shell_depth(run, tmp_path, 80)
    assert run.cand_ext.shape == (3, 80) and np.array_equal(shell_ids(run, 1, 10, 80), order[1, 10:80])
    kept = run.cand_ext
    ensure_shell_depth(run, tmp_path, 60)                 # already deep enough
    assert run.cand_ext is kept
