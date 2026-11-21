#!/usr/bin/env python3
"""Tests for utils.cache.entries."""

import numpy as np
import pytest

from utils.cache.entries import (
    build_entry_tables, entry_kmax, is_member, perturbed, state_tables,
)
from utils.cache.workloads import RunData


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


def _run(n_db=200, n_cand=5, k_values=(5, 20), seed=0):
    rng = np.random.default_rng(seed)
    db = rng.standard_normal((n_db, 4))
    cand_q = rng.standard_normal((n_cand, 4))
    cached_ids, radius, _, _ = build_entry_tables(cand_q, db, list(k_values), 1)
    return RunData(k=3, k_values=list(k_values), db=db, cand_q=cand_q, sets={}, topk={},
                   cached_ids=cached_ids, radius=radius), db, cand_q


def test_build_entry_tables_radius_is_kth_neighbor_distance():
    run, db, cand_q = _run()
    d = np.linalg.norm(cand_q[:, None] - db[None], axis=2)
    for i in range(5):
        for t, K in enumerate((5, 20)):
            j = i * 2 + t
            assert len(run.cached_ids[j]) == K
            assert np.isclose(run.radius[j], np.sort(d[i])[K - 1])
            assert set(run.cached_ids[j].tolist()) == set(np.argsort(d[i])[:K].tolist())


def test_build_entry_tables_validates_inputs():
    db = np.random.default_rng(0).standard_normal((50, 3))
    q = db[:2]
    for bad in ([], [0, 5], [10, 5], [5, 5]):
        with pytest.raises(ValueError, match="strictly increasing"):
            build_entry_tables(q, db, bad, 1)
    with pytest.raises(ValueError, match="columns"):
        build_entry_tables(q, db, [5, 20], 1, cached_top=np.zeros((2, 10), dtype=int))
    with pytest.raises(ValueError, match="rows"):
        build_entry_tables(q, db, [5], 1, cached_top=np.zeros((3, 10), dtype=int))


def test_state_tables_membership_matches_brute_force():
    run, db, _ = _run()
    entries = [0, 3, 4, 9]                     # candidates 0 (K=5), 1 (K=20), 2 (K=5), 4 (K=20)
    st = state_tables(run, entries)
    union = set()
    for j in entries:
        union |= set(run.cached_ids[j].tolist())
    assert set(st["cached"].tolist()) == union == set(np.flatnonzero(st["in_cache"]).tolist())
    assert np.allclose(st["db_c"], db[st["cached"]])
    for e, j in enumerate(entries):
        member = set(run.cached_ids[j].tolist())
        probe = np.arange(len(db))
        assert is_member(st, e, probe).tolist() == [v in member for v in probe]


def test_state_tables_empty_and_single_entry():
    run, _, _ = _run()
    st = state_tables(run, [])
    assert len(st["cached"]) == 0 and len(st["member_keys"]) == 0
    assert not is_member(st, 0, np.arange(5)).any()
    one = state_tables(run, [1])
    assert len(one["cached"]) == 20 and one["ent_q"].shape == (1, 4) and one["ent_r"].shape == (1,)


def test_perturbed_rejects_bad_fraction_and_exhausted_candidates():
    with pytest.raises(ValueError, match="frac"):
        perturbed([0, 1], 1.5, 5, 2, np.random.default_rng(0))
    # every candidate already selected at k index 0: nothing left to swap in
    n_cand, n_kv = 3, 2
    full = [c * n_kv for c in range(n_cand)]
    with pytest.raises(ValueError, match="no unselected candidate"):
        perturbed(full, 1.0, n_cand, n_kv, np.random.default_rng(0))
    assert perturbed([], 0.5, n_cand, n_kv, np.random.default_rng(0)) == []

