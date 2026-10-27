#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Tests for ee_estimators. Run with pytest or `python3 -B test_ee_estimators.py` (temp dirs only)."""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import ee_config as cfg
import ee_estimators as est
import ee_workload as ew
from rt_data import sample_permutation
from rt_engine import build_seed_state, make_hyperplanes, sign_codes, table_estimates, table_hamming

D, N_BASE, K = 16, 120, 10

def unit(rng, n, d=D):
    v = rng.standard_normal((n, d))
    return v / np.linalg.norm(v, axis=1, keepdims=True)

def fixture(seed=0):
    rng = np.random.default_rng(seed)
    base, pool = unit(rng, N_BASE), unit(rng, 300)
    w = ew.build_workload("sift", base, pool, seed, 0.5, n=40, K=K, per_anchor=10, prefix_len=30)
    return base, w

def brute_value(W, L, m, labeled, labels, query, nu, exclude=None):
    vals = []
    for t in range(L):
        Wt = W[t * m:(t + 1) * m]
        cl, qc = sign_codes(labeled, Wt), sign_codes(query[None, :], Wt)[0]
        h = (cl != qc).sum(axis=1)
        lab = labels
        if exclude is not None:
            keep = np.arange(len(labeled)) != exclude
            h, lab = h[keep], labels[keep]
        cutoff = np.sort(h)[min(nu, len(h)) - 1]
        vals.append(lab[h <= cutoff].mean())
    return float(np.mean(vals))

#-------------------------------------------------------------------------------

def test_hash_conventions():
    W = make_hyperplanes(3, D)
    assert W.shape == (2048, D) and set(np.unique(W).tolist()) == {-1.0, 1.0}
    assert (sign_codes(np.zeros((2, D)), W) == 1).all()  # a zero projection counts as +1
    base, w = fixture()
    labeled = np.vstack([w.train, base[:30]])
    labels = np.concatenate([w.theta_train, np.linspace(0.2, 0.4, 30)])
    store = est.LSHStore(W, 4, 32, labeled, labels)
    assert [t.rows for t in store.tables] == [(0, 32), (32, 32), (64, 32), (96, 32)]
    for t in store.tables:
        assert t.counts.sum() == len(labeled) and np.isclose(t.sums.sum(), labels.sum())
        assert np.array_equal(t.codes_pm[t.inverse], sign_codes(labeled, W[t.rows[0]:t.rows[0] + t.rows[1]]))

def test_estimator_matches_brute_force_with_ties_and_leave_one_out():
    base, w = fixture(1)
    W = make_hyperplanes(1, D)
    labeled = np.vstack([w.train, base[:50]])
    labels = np.concatenate([w.theta_train, np.linspace(0.15, 0.45, 50)])
    n_train = len(w.train)
    for L, m in ((1, 8), (4, 8), (3, 32)):
        store = est.LSHStore(W, L, m, labeled, labels)
        for nu in (1, 3, 8, 40, 1000):
            got_test = store.estimate(w.test, nu)
            got_train = store.estimate(w.train, nu, own=np.arange(n_train))
            for i in (0, 5, 17):
                assert np.isclose(got_test[i], brute_value(W, L, m, labeled, labels, w.test[i], nu), rtol=1e-12), (L, m, nu)
                assert np.isclose(got_train[i], brute_value(W, L, m, labeled, labels, w.train[i], nu, exclude=i),
                                  rtol=1e-12), (L, m, nu)

def test_leave_one_out_restores_store_and_equals_rebuild():
    base, w = fixture(2)
    W = make_hyperplanes(2, D)
    labeled, labels = np.vstack([w.train, base[:40]]), np.concatenate([w.theta_train, np.linspace(0.2, 0.4, 40)])
    store = est.LSHStore(W, 4, 16, labeled, labels)
    before = [(t.sums.copy(), t.counts.copy()) for t in store.tables]
    loo = store.estimate(w.train, 4, own=np.arange(len(w.train)))
    for t, (s, c) in zip(store.tables, before):
        assert np.array_equal(t.sums, s) and np.array_equal(t.counts, c)
    for i in (0, 9, 25):
        keep = np.arange(len(labeled)) != i
        rebuilt = est.LSHStore(W, 4, 16, labeled[keep], labels[keep])
        assert np.isclose(rebuilt.estimate(w.train[i:i + 1], 4)[0], loo[i], rtol=1e-12)

def test_matches_rt_engine_table_values_for_plain_mean():
    rng = np.random.default_rng(5)
    database = unit(rng, 60)
    train = unit(rng, 20)
    queries = unit(rng, 15)
    theta_db, theta_train = rng.uniform(0.2, 0.4, 60), rng.uniform(0.2, 0.4, 20)
    state = build_seed_state(4, database, theta_db, train, theta_train, [60], bits_max=2048)
    W = state.W
    labeled = np.vstack([database[state.perm], train])
    labels = np.concatenate([theta_db[state.perm], theta_train])
    q_codes = sign_codes(queries, W).astype(np.float32)
    for L, m in ((1, 64), (4, 64)):  # rt_engine blocks are 64 bits wide
        tables = [(t * m, m) for t in range(L)]
        H = table_hamming(q_codes, state, tables)
        for nu in (1, 4, 16):
            rt_vals = np.mean([table_estimates(H[tb], state, m, [nu])[(60, "sample_train")][0][0] for tb in tables], axis=0)
            ours = est.LSHStore(W, L, m, labeled, labels).estimate(queries, nu)
            assert np.allclose(ours, rt_vals, rtol=1e-12), (L, m, nu)

def test_estimate_multi_equals_single_nu_with_and_without_leave_one_out():
    base, w = fixture(7)
    W = make_hyperplanes(7, D)
    labeled = np.vstack([w.train, base[:50]])
    labels = np.concatenate([w.theta_train, np.linspace(0.15, 0.45, 50)])
    nus = [1, 2, 4, 8, 16, 32]
    for L, m in ((1, 8), (4, 16), (3, 32)):
        store = est.LSHStore(W, L, m, labeled, labels)
        multi = store.estimate_multi(w.test, nus, chunk=7)
        multi_loo = store.estimate_multi(w.train, nus, own=np.arange(len(w.train)), chunk=11)
        for j, nu in enumerate(nus):
            assert np.allclose(multi[j], store.estimate(w.test, nu), rtol=1e-12, atol=0)
            assert np.allclose(multi_loo[j], store.estimate(w.train, nu, own=np.arange(len(w.train))), rtol=1e-12, atol=0)

def test_oracle_estimated_recall_equals_count_within_theta():
    base, w = fixture(3)
    rng = np.random.default_rng(9)
    pool_ids = rng.choice(N_BASE, 50, replace=False)
    mask = est.pool_mask(pool_ids, N_BASE)
    for split, queries, ids, theta in (("train", w.train, w.train_top_ids, w.theta_train),
                                       ("test", w.test, w.test_top_ids, w.theta_test)):
        by_set = est.true_recall_counts(mask, ids)
        by_distance = est.vectors_within_counts(queries, theta, base[pool_ids])
        assert np.array_equal(by_set, by_distance), split
        assert est.OracleArm(w).recall_counts(split, mask).tolist() == by_set.tolist()
    assert np.array_equal(est.OracleArm(w).radius("test"), w.theta_test)

def test_lsh_estimated_recall_is_unclipped_count_over_pool():
    base, w = fixture(4)
    mask = est.pool_mask(np.arange(80), N_BASE)
    big = np.full(len(w.test), 0.99)
    counts = est.vectors_within_counts(w.test, big, base[:80])
    assert counts.max() > K and (counts == 80).all()  # every pool vector is within 0.99
    d = np.arccos(np.clip(w.test @ base[:80].T, -1, 1)) / np.pi
    r = np.full(len(w.test), 0.3)
    assert np.array_equal(est.vectors_within_counts(w.test, r, base[:80]), (d <= 0.3).sum(axis=1))

def test_lsh_arm_labeled_set_and_loo_radius():
    base, w = fixture(6)
    with tempfile.TemporaryDirectory() as td:
        theta_db = ew.db_theta("sift", base, K, Path(td), batch_size=64)
    arm = est.LSHArm(w, base, seed=6, L=4, m=16, nu=4, sample_size=40, db_theta=theta_db)
    assert len(arm.store.labels) == len(w.train) + 40 and len(arm.sample_ids) == len(set(arm.sample_ids.tolist())) == 40
    labeled = np.vstack([w.train, base[arm.sample_ids]])
    labels = np.concatenate([w.theta_train, theta_db[arm.sample_ids]])
    W = make_hyperplanes(6, D)
    for i in (0, 11, 30):
        assert np.isclose(arm.radius("train")[i], brute_value(W, 4, 16, labeled, labels, w.train[i], 4, exclude=i), rtol=1e-12)
        assert np.isclose(arm.radius("test")[i], brute_value(W, 4, 16, labeled, labels, w.test[i], 4), rtol=1e-12)
    mask = est.pool_mask(np.arange(60), N_BASE)
    counts = arm.recall_counts("test", mask)
    assert np.array_equal(counts, est.vectors_within_counts(w.test, arm.radius("test"), base[:60]))
    none = est.LSHArm(w, base, seed=6, L=1, m=16, nu=2)
    assert len(none.store.labels) == len(w.train)

#-------------------------------------------------------------------------------

def run_all() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:
            failed += 1
            import traceback
            print(f"FAIL {name}: {exc!r}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return failed

if __name__ == "__main__":
    sys.exit(1 if run_all() else 0)
