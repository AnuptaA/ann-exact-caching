#!/usr/bin/env python3
"""Tests for the best-cutoff hit rate in eval_gap."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from simulations.approximate_hits.eval_gap import best_cutoff_rate


def test_best_cutoff_rate():
    s = np.array([5, 4, 3, 2, 1.0])
    y = np.array([1, 1, 0.8, 0.0, 1.0])
    # top 3 average 0.933 >= 0.9; top 4 average 0.7
    assert best_cutoff_rate(s, y, 0.9) == 0.6


def test_best_cutoff_rate_respects_ties():
    s = np.array([2, 1, 1, 1.0])
    y = np.array([1, 1, 0.0, 0.0])
    # cannot split the tied group: only the top query qualifies
    assert best_cutoff_rate(s, y, 0.9) == 0.25


def test_profile_features_offsets_and_missing_ranks():
    from simulations.approximate_hits import eval_calibrated as ecal
    from simulations.approximate_hits import eval_gap as eg

    rng = np.random.default_rng(0)
    db = rng.standard_normal((400, 8))
    cand_q = rng.standard_normal((5, 8))
    k_values = [10, 20, 50]
    d_all = np.linalg.norm(cand_q[:, None, :] - db[None], axis=2)
    order = np.argsort(d_all, axis=1)
    cached_ids = [order[i, :K] for i in range(5) for K in k_values]
    ecal._G.update(db=db, cand_q=cand_q, k_values=k_values, cached_ids=cached_ids)
    eg._R["center_prof"] = eg.center_profiles()
    assert np.allclose(eg._R["center_prof"], np.sort(d_all, axis=1)[:, :50])

    # cache: candidate 0 with K = 50, candidate 1 with K = 10
    entries = [0 * 3 + 2, 1 * 3 + 0]
    q = cand_q[[0]] + 1e-3
    ck = np.array([1.0])
    F, first = eg.profile_features(q, entries, ck)
    assert first[0, 0] == 0 and first[0, 1] == 50
    assert np.isclose(first[0, 2], np.linalg.norm(q[0] - cand_q[0]))
    # nearest entry has all ranks; second (K = 10) lacks ranks 20 and 50
    assert not np.isnan(F[0, :4]).any()
    assert not np.isnan(F[0, 5]) and np.isnan(F[0, 6:8]).all()
    # only two cached centers: third slot missing entirely
    assert np.isnan(F[0, 8:]).all()


def test_sample_features_counts_only_uncached_intruders():
    from simulations.approximate_hits import eval_gap as eg
    from utils.base.knn import sq_norms

    # sample: 1-d points at distances 1, 2, 3, 4, 5, 6 from the query at 0
    vecs = np.arange(1.0, 7.0)[:, None]
    samp = {"ids": np.arange(6), "vecs": vecs, "sq": sq_norms(vecs)}
    in_cache = np.array([True, False, True, False, False, False])
    q = np.zeros((1, 1))
    F = eg.sample_features(q, np.array([4.5]), in_cache, samp)
    assert np.allclose(np.exp(F[0, :5]) * 4.5, [1, 2, 3, 4, 5])
    # uncached within 4.5: ids 1, 3 -> 2; within 4.95: same; all within 5.625: 5
    assert np.allclose(np.expm1(F[0, 5:]), [2, 2, 5])


def test_shell_sample_draws_only_uncached_shell_vectors():
    from simulations.approximate_hits import eval_calibrated as ecal
    from simulations.approximate_hits import eval_gap as eg

    rng = np.random.default_rng(0)
    db = rng.standard_normal((1000, 4))
    cand_q = rng.standard_normal((3, 4))
    k_values = [10, 500]
    order = np.argsort(np.linalg.norm(cand_q[:, None] - db[None], axis=2), axis=1)
    cached_ids = [order[i, :K] for i in range(3) for K in k_values]
    ecal._G.update(db=db, cand_q=cand_q, k_values=k_values, cached_ids=cached_ids)
    eg._R["draw"] = rng.random(len(db))
    entries = [0 * 2 + 0]  # candidate 0 with K = 10
    in_cache = np.zeros(len(db), dtype=bool)
    in_cache[order[0, :10]] = True
    samp = eg.shell_sample(entries, in_cache, 0.3)
    shell = set(order[0, 10:500].tolist())
    assert set(samp["ids"].tolist()) <= shell
    assert not in_cache[samp["ids"]].any()
    assert (eg._R["draw"][samp["ids"]] < 0.3).all()
    expected = {v for v in shell if eg._R["draw"][v] < 0.3}
    assert set(samp["ids"].tolist()) == expected
    assert np.allclose(samp["vecs"], db[samp["ids"]])
