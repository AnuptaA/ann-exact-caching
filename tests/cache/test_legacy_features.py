#!/usr/bin/env python3
"""Tests for the earlier 12 + 8 features (utils.cache.legacy_features)."""

import numpy as np

from utils.base.knn import exact_topk
from utils.cache.entries import build_entry_tables, state_tables
from utils.cache.legacy_features import (
    EXT_NAMES, FEATURE_NAMES, SPLITS, build_state, entry_history, features, full_features,
)
from utils.cache.workloads import RunData


def _run(n_db=400, n_cand=12, k_values=(5, 20), k=3, n_q=40, seed=0):
    rng = np.random.default_rng(seed)
    db = rng.standard_normal((n_db, 4))
    cand_q = rng.standard_normal((n_cand, 4))
    cached_ids, radius, _, _ = build_entry_tables(cand_q, db, list(k_values), 1)
    sets = {s: (rng.standard_normal((n_q, 4)), rng.integers(1, 4, n_q)) for s in SPLITS}
    topk = {s: exact_topk(q, db, k) for s, (q, _) in sets.items()}
    return RunData(k=k, k_values=list(k_values), db=db, cand_q=cand_q, sets=sets, topk=topk,
                   cached_ids=cached_ids, radius=radius)


def test_base_features_have_expected_shape_ranges_and_ordering():
    run = _run()
    st = state_tables(run, [0, 3, 6, 9, 12, 15])
    q = run.sets["fit"][0]
    F = features(run, q, st)
    assert F.shape == (len(q), len(FEATURE_NAMES)) and np.isfinite(F).all()
    col = {n: i for i, n in enumerate(FEATURE_NAMES)}
    assert (F[:, col["log_D1"]] <= F[:, col["log_Dk"]] + 1e-12).all()      # D1 <= Dk
    assert ((F[:, col["a1_frac"]] >= 0) & (F[:, col["a1_frac"]] <= 1)).all()
    assert ((F[:, col["frac_topk_from_entry"]] >= 0) & (F[:, col["frac_topk_from_entry"]] <= 1)).all()


def test_full_features_extended_columns_and_nearest_entry():
    run = _run()
    st = state_tables(run, [0, 3, 6, 9, 12, 15])
    q = run.sets["dev"][0]
    base, ext, jstar = full_features(run, q, st, 0.05)
    assert ext.shape == (len(q), len(EXT_NAMES)) and (jstar >= 0).all() and (jstar < 6).all()
    assert np.allclose(ext[:, EXT_NAMES.index("log_cache_frac")], np.log(0.05 + 1e-12))
    assert ((ext[:, 3] >= 0) & (ext[:, 3] <= 1)).all()                       # covered fraction


def test_entry_history_out_of_fold_and_smoothing():
    rng = np.random.default_rng(0)
    n, n_entries = 200, 4
    jstar = rng.integers(0, n_entries, n)
    y = rng.random(n)
    w = rng.integers(1, 4, n).astype(float)
    (oof_mean, oof_cnt), full_mean, full_cnt = entry_history(jstar, y, w, n_entries, smooth=1e9)
    prior = np.average(y, weights=w)
    assert np.allclose(oof_mean, prior, atol=1e-3) and np.allclose(full_mean, prior, atol=1e-3)
    (_, oof_cnt), _, full_cnt = entry_history(jstar, y, w, n_entries, smooth=20.0)
    assert (oof_cnt <= full_cnt[jstar] + 1e-9).all()                         # fold excludes rows
    assert np.isclose(full_cnt.sum(), w.sum())


def test_build_state_labels_match_brute_force_and_handle_empty_or_tiny_caches():
    run = _run()
    point = {"cache_frac": 0.1, "test_hit_rate_online": 0.2, "test_hit_rate": 0.25}
    assert build_state((0.9, point, []), run) is None                         # empty cache
    tiny = _run(k=5)
    assert build_state((0.9, point, [0]), tiny) is None                       # 5 cached <= k = 5
    entries = [0, 3, 6, 9]
    out = build_state((0.9, point, entries), run)
    in_cache = np.zeros(len(run.db), dtype=bool)
    for j in entries:
        in_cache[run.cached_ids[j]] = True
    for s in SPLITS:
        assert np.allclose(out[s]["y"], in_cache[run.topk[s]].mean(axis=1))
        assert out[s]["base"].shape[1] == len(FEATURE_NAMES)
        assert out[s]["ext"].shape[1] == len(EXT_NAMES) and np.isfinite(out[s]["ext"]).all()
    assert out["cache_frac"] == 0.1 and out["oracle_hit_rate"] == 0.25
