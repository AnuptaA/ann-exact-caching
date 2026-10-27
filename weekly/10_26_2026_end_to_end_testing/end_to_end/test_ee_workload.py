#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Tests for ee_workload. Run with pytest or `python3 -B test_ee_workload.py` (temp dirs only)."""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import ee_config as cfg
import ee_workload as ew

D, N_BASE, N_POOL, N, PER_ANCHOR, K, PREFIX = 16, 300, 400, 200, 50, 10, 40

def unit(rng, n, d=D):
    v = rng.standard_normal((n, d))
    return v / np.linalg.norm(v, axis=1, keepdims=True)

def fixture():
    rng = np.random.default_rng(0)
    return unit(rng, N_BASE), unit(rng, N_POOL)

def make(seed, lam, base=None, pool=None):
    b, p = fixture()
    return ew.build_workload("sift", b if base is None else base, p if pool is None else pool, seed, lam, n=N, K=K,
                             per_anchor=PER_ANCHOR, prefix_len=PREFIX)

def angle_deg(a, b):
    return np.degrees(np.arccos(np.clip((a * b).sum(axis=1), -1, 1)))

#-------------------------------------------------------------------------------

def test_regime_counts_and_anchor_nesting():
    w0, w5, w1 = make(3, 0.0), make(3, 0.5), make(3, 1.0)
    assert (w0.n_anchors, w5.n_anchors, w1.n_anchors) == (0, 2, 4)
    assert (w5.train_anchor >= 0).sum() == 2 * PER_ANCHOR and (w5.train_anchor < 0).sum() == N - 2 * PER_ANCHOR
    assert (w1.train_anchor >= 0).sum() == N and (w0.train_anchor < 0).sum() == N
    assert np.array_equal(w5.anchor_source, w1.anchor_source[:2])
    # anchors and their copies are nested across lambda
    assert np.array_equal(w5.train[:2 * PER_ANCHOR], w1.train[:2 * PER_ANCHOR])
    assert np.array_equal(w5.test[:2 * PER_ANCHOR], w1.test[:2 * PER_ANCHOR])

def test_diffuse_regime_counts_and_cache_names():
    b, p = fixture()
    w = ew.build_workload("sift", b, p, 2, 1.0, n=N, K=K, per_anchor=5, prefix_len=PREFIX)
    assert w.n_anchors == 40 and (w.train_anchor >= 0).all() and (w.test_anchor >= 0).all()
    assert np.bincount(w.train_anchor).tolist() == [5] * 40 == np.bincount(w.test_anchor).tolist()
    assert set(w.train_anchor.tolist()) == set(w.test_anchor.tolist())
    assert cfg.regime_tag(0.0) == "lam0" and cfg.regime_tag(0.5) == "lam0.5" and cfg.regime_tag(1.0) == "lam1"
    assert cfg.regime_tag(1.0, 5) == "diffuse" and cfg.regime_tag(0.5, 5) == "lam0.5_pa5"
    assert ew.cache_path("sift", 0, 1.0, 2500, 10, Path("x"), 5).name == "sift_diffuse_seed0_n2500_K10.npz"
    assert ew.cache_path("sift", 0, 0.5, 2500, 10, Path("x")).name == "sift_lam0.5_seed0_n2500_K10.npz"

def test_train_test_disjoint_and_random_queries_exclude_anchors():
    b, p = fixture()
    for lam in (0.0, 0.5):
        w = make(1, lam)
        tr_rand = w.train[w.train_anchor < 0]
        te_rand = w.test[w.test_anchor < 0]
        pool_ids = {tuple(np.round(x, 12)): i for i, x in enumerate(p)}
        tr_ids = [pool_ids[tuple(np.round(x, 12))] for x in tr_rand]
        te_ids = [pool_ids[tuple(np.round(x, 12))] for x in te_rand]
        assert len(set(tr_ids)) == len(tr_ids) and len(set(te_ids)) == len(te_ids)
        assert not set(tr_ids) & set(te_ids)
        assert not (set(tr_ids) | set(te_ids)) & set(w.anchor_source.tolist())
    both = np.vstack([w.train, w.test])
    assert len(np.unique(np.round(both, 12), axis=0)) == len(both)

def test_base_coinciding_queries_are_skipped_in_the_same_permutation():
    b, p = fixture()
    plain = ew.build_queries(p, 3, 0.0, 100, PER_ANCHOR)
    same = ew.build_queries(p, 3, 0.0, 100, PER_ANCHOR, base=b)
    assert np.array_equal(same["random_source"], plain["random_source"])
    drawn = plain["random_source"]
    bad = drawn[[0, 5, 17]]
    p2 = p.copy()
    p2[bad] = b[:3]
    bad = bad[ew.coincides_with_base(p2[bad], b)]  # float rounding can leave a copy at distance ~1e-8
    assert len(bad) >= 1
    skip = ew.build_queries(p2, 3, 0.0, 100, PER_ANCHOR, base=b)
    expected = [i for i in np.random.default_rng([3, 13, 0]).permutation(np.arange(N_POOL)) if i not in set(bad)][:200]
    assert skip["random_source"].tolist() == expected
    assert not set(bad) & set(skip["random_source"].tolist())
    assert not ew.coincides_with_base(p[drawn[:20]], b).any()

def test_pool_too_small_raises_instead_of_resampling():
    b, p = fixture()
    try:
        ew.build_queries(p[:150], 0, 0.0, N, PER_ANCHOR)
        raise RuntimeError("expected ValueError")
    except ValueError as exc:
        assert "too small" in str(exc)

def test_local_copies_angles_in_range_and_independent():
    b, p = fixture()
    w = make(2, 1.0)
    for queries, anchors, angles in ((w.train, w.train_anchor, w.train_angle_deg),
                                     (w.test, w.test_anchor, w.test_angle_deg)):
        for rank in range(w.n_anchors):
            sel = anchors == rank
            assert sel.sum() == PER_ANCHOR
            a = p[w.anchor_source[rank]][None, :].repeat(PER_ANCHOR, axis=0)
            ang = angle_deg(queries[sel], a)
            assert ang.min() >= 0 and ang.max() <= cfg.MAX_ANGLE_DEG + 1e-9
            assert np.allclose(ang, angles[sel], atol=1e-6)
            assert len(np.unique(np.round(ang, 9))) == PER_ANCHOR  # no shared angles
    assert np.allclose(np.linalg.norm(w.train, axis=1), 1.0) and np.allclose(np.linalg.norm(w.test, axis=1), 1.0)

def test_ground_truth_matches_brute_force_and_theta():
    b, p = fixture()
    w = make(4, 0.5)
    for queries, ids, dist in ((w.train, w.prefix_ids, w.prefix_dist), (w.test, w.test_top_ids, w.test_top_dist)):
        full = np.arccos(np.clip(queries @ b.T, -1, 1)) / np.pi
        order = np.argsort(full, axis=1, kind="stable")[:, :ids.shape[1]]
        assert np.array_equal(ids, order)
        assert np.allclose(dist, np.take_along_axis(full, order, axis=1), atol=1e-12)
    assert np.array_equal(w.train_top_ids, w.prefix_ids[:, :K])
    assert np.allclose(w.theta_train, w.train_top_dist[:, K - 1]) and np.allclose(w.theta_test, w.test_top_dist[:, K - 1])
    assert (np.diff(w.prefix_dist, axis=1) >= 0).all()

def test_train_train_matrix_and_order():
    w = make(5, 0.5)
    full = np.arccos(np.clip(w.train @ w.train.T, -1, 1)) / np.pi
    np.fill_diagonal(full, 0.0)
    assert np.allclose(w.tt_dist, full) and (np.diag(w.tt_dist) == 0).all()
    assert w.tt_order.shape == (N, N - 1)
    for i in (0, 7, 120):
        assert i not in w.tt_order[i]
        assert (np.diff(w.tt_dist[i, w.tt_order[i]]) >= 0).all()
        assert len(set(w.tt_order[i].tolist())) == N - 1

def test_determinism_and_seed_sensitivity():
    a, b, c = make(7, 0.5), make(7, 0.5), make(8, 0.5)
    assert np.array_equal(a.train, b.train) and np.array_equal(a.test, b.test) and np.array_equal(a.prefix_ids, b.prefix_ids)
    assert not np.array_equal(a.train, c.train)

def test_check_queries_rejects_duplicates_and_base_members():
    b, p = fixture()
    train, test = p[:5].copy(), p[5:10].copy()
    ew.check_queries(train, test, b)
    bad = test.copy()
    bad[0] = train[0]
    for args in ((train, bad, b), (train, np.vstack([b[:1], test[1:]]), b)):
        try:
            ew.check_queries(*args)
            raise RuntimeError("expected AssertionError")
        except AssertionError:
            pass

def test_cache_roundtrip_dedupe_and_db_theta_excludes_self():
    b, p = fixture()
    w = make(9, 0.5)
    with tempfile.TemporaryDirectory() as td:
        path = ew.cache_path("sift", 9, 0.5, N, K, Path(td))
        ew.save_workload(w, path)
        r = ew.load_workload(path, b)
        for f in ew.CACHE_FIELDS + ["prefix_ids", "prefix_dist", "tt_dist", "tt_order"]:
            np.testing.assert_array_equal(getattr(w, f), getattr(r, f), err_msg=f)
        assert path.stat().st_size < 2_000_000
        theta = ew.db_theta("sift", b, K, Path(td), batch_size=64)
        full = np.arccos(np.clip(b @ b.T, -1, 1)) / np.pi
        np.fill_diagonal(full, np.inf)
        assert np.allclose(theta, np.sort(full, axis=1)[:, K - 1], atol=1e-12)
        assert np.array_equal(theta, ew.db_theta("sift", b, K, Path(td)))
    dup = np.vstack([p, p[3:6], p[10:11]])
    uniq, kept = ew.dedupe_pool(dup)
    assert len(uniq) == N_POOL and np.array_equal(uniq, p) and np.array_equal(kept, np.arange(N_POOL))
    w2 = ew.build_workload("sift", b, uniq, 1, 0.0, n=N, K=K, per_anchor=PER_ANCHOR, prefix_len=PREFIX)
    assert w2.n_anchors == 0

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
