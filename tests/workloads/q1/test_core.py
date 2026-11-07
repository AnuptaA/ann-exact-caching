"""Tests for the Q1 distance, recall, universe, streams, stats and ground truth."""

import numpy as np
import pytest

from workloads.q1 import groundtruth, stats, streams, universe
from workloads.q1.distance import l2_rows, l2_to_ids, sq_l2_rows
from workloads.q1.recall import recall_at_k, recall_batch

RNG = np.random.default_rng(0)


def test_distance_matches_numpy_and_squared_variant():
    a = RNG.normal(size=(50, 8)).astype(np.float32)
    b = RNG.normal(size=(50, 8)).astype(np.float32)
    want = np.linalg.norm(a.astype(np.float64) - b, axis=1)
    assert np.allclose(l2_rows(a, b), want, rtol=1e-5)
    assert np.allclose(sq_l2_rows(a, b), want ** 2, rtol=1e-5)
    assert l2_rows(a, b).dtype == np.float32
    with pytest.raises(ValueError):
        sq_l2_rows(a, b[:10])


def _line_base(n=30):
    base = np.zeros((n, 2), dtype=np.float32)
    base[:, 0] = np.arange(n)
    return base


def test_recall_divides_by_ten_in_every_case():
    base = _line_base()
    q = np.zeros(2, dtype=np.float32)
    kth = 9.0   # exact top-10 are ids 0..9
    assert recall_at_k(q, np.arange(10), base, kth) == 1.0
    assert recall_at_k(q, np.arange(5), base, kth) == 0.5          # short answer
    assert recall_at_k(q, np.array([], dtype=int), base, kth) == 0.0  # empty answer
    assert recall_at_k(q, np.array([0, 0, 0, 0, 0, 1, 2, 3, 4, 5]), base, kth) == 0.6  # duplicates
    assert recall_at_k(q, np.array([0, 1, 2, -1, -1, -1, -1, -1, -1, -1]), base, kth) == 0.3
    assert recall_at_k(q, np.arange(10, 20), base, kth) == 0.0
    assert recall_at_k(q, np.arange(30), base, kth) == 1.0         # only first 10 are read


def test_recall_counts_ties_as_correct():
    base = np.zeros((12, 2), dtype=np.float32)
    base[:, 0] = 1.0   # every vector is at distance 1: all tie with the 10th
    q = np.zeros(2, dtype=np.float32)
    assert recall_at_k(q, np.arange(2, 12), base, kth_dist=1.0) == 1.0
    assert recall_at_k(q, np.arange(2, 12), base, kth_dist=0.99) == 0.0
    ans = np.tile(np.arange(10), (3, 1))
    assert np.allclose(recall_batch(np.zeros((3, 2), np.float32), ans, base, np.ones(3)), 1.0)


def test_universe_structure_and_determinism():
    official = RNG.normal(size=(40, 4)).astype(np.float32)
    uni = universe.build_universe(40, n_groups=25, group_size=7, seed=3)
    again = universe.build_universe(40, n_groups=25, group_size=7, seed=3)
    assert np.array_equal(uni.partner_idx, again.partner_idx) and np.array_equal(uni.lam, again.lam)
    assert len(set(uni.base_idx.tolist())) == 25                       # sample without replacement
    assert (uni.partner_idx != uni.base_idx[:, None]).all()           # partner never the base
    assert uni.partner_idx.min() >= 0 and uni.partner_idx.max() < 40
    assert (uni.lam > 0).all() and (uni.lam < 0.5).all()
    assert len({(g, tuple(r)) for g, r in enumerate(uni.partner_idx)}) == 25
    members = universe.member_vectors(uni, official, slice(0, 25))
    assert members.shape == (25 * 7, 4)
    ids = np.array([0, 8, uni.n_members + 3, uni.n_members + uni.n_groups + 5])
    got = universe.request_vectors(uni, official, ids)
    assert np.allclose(got[:2], members[[0, 8]], atol=1e-6)
    assert np.array_equal(got[2], official[uni.base_idx[3]])
    assert np.array_equal(got[3], official[5])
    with pytest.raises(ValueError):
        universe.request_vectors(uni, official, np.array([uni.n_ids]))


def test_universe_all_official_as_bases_and_member_formula():
    official = RNG.normal(size=(10, 3)).astype(np.float32)
    uni = universe.build_universe(10, n_groups=10, group_size=4, seed=0)
    assert sorted(uni.base_idx.tolist()) == list(range(10))
    g, j = 2, 3
    want = (1 - uni.lam[g, j]) * official[uni.base_idx[g]] + uni.lam[g, j] * official[uni.partner_idx[g, j]]
    got = universe.request_vectors(uni, official, np.array([uni.member_id(g, j)]))[0]
    assert np.allclose(got, want, atol=1e-6)
    with pytest.raises(ValueError):
        universe.build_universe(5, n_groups=10)


def test_zipf_stream_matches_theory_and_is_seeded():
    uni = universe.build_universe(200, n_groups=100, group_size=5, seed=0)
    s1 = streams.zipf_stream(uni, 50_000, 0.99, seed=0)
    assert np.array_equal(s1, streams.zipf_stream(uni, 50_000, 0.99, seed=0))
    assert not np.array_equal(s1, streams.zipf_stream(uni, 50_000, 0.99, seed=1))
    assert s1.min() >= 0 and s1.max() < uni.n_members
    st = stats.stream_stats(uni, s1, 0.99)
    top = st["zipf"][0]
    assert abs(top["empirical"] - top["theoretical"]) < 0.01
    assert st["repeat_share"] > 0.9 and "rank" in stats.format_stats("zipf", st)


def test_uniform_and_no_locality_streams():
    uni = universe.build_universe(60, n_groups=20, group_size=3, seed=0)
    u = streams.uniform_stream(uni, 1000, seed=0)
    assert ((u >= uni.n_members) & (u < uni.n_members + uni.n_groups)).all()
    nl = streams.no_locality_stream(uni, seed=0)
    assert len(nl) == 60 and len(set(nl.tolist())) == 60
    assert nl.min() == uni.pool_id(0) and nl.max() == uni.pool_id(59)
    assert not np.array_equal(nl, np.sort(nl))
    st = stats.stream_stats(uni, nl)
    assert st["distinct"] == 60 and st["repeat_share"] == 0.0


def test_history_split():
    assert streams.history_split(100_000) == 30_000
    assert streams.history_split(100, 0.5) == 50
    with pytest.raises(ValueError):
        streams.history_split(10, 1.0)


def _brute(queries, base, k):
    d = np.linalg.norm(queries[:, None, :].astype(np.float64) - base[None, :, :], axis=2)
    return np.argsort(d, axis=1, kind="stable")[:, :k]


def test_ground_truth_matches_brute_force_and_resumes(tmp_path):
    base = RNG.normal(size=(300, 6)).astype(np.float32)
    queries = RNG.normal(size=(25, 6)).astype(np.float32)
    ids, dist = groundtruth.build_ground_truth(queries, base, tmp_path / "gt", k=20,
                                               query_chunk=10, base_chunk=64)
    assert np.array_equal(ids, _brute(queries, base, 20))
    assert (np.diff(dist, axis=1) >= 0).all() and dist.dtype == np.float32
    assert len(list((tmp_path / "gt").glob("chunk_*.npz"))) == 3
    (tmp_path / "gt" / "chunk_000000010.npz").unlink()   # simulate an interrupted run
    ids2, dist2 = groundtruth.build_ground_truth(queries, base, tmp_path / "gt", k=20,
                                                 query_chunk=10, base_chunk=64)
    assert np.array_equal(ids, ids2) and np.array_equal(dist, dist2)
    with pytest.raises(ValueError):
        groundtruth.build_ground_truth(queries, base, tmp_path / "gt", k=19, query_chunk=10)
    with pytest.raises(ValueError):
        groundtruth.exact_topk_chunk(queries, base[:5], k=10)


def test_ground_truth_with_exact_ties_keeps_distances_exact():
    base = np.ones((40, 3), dtype=np.float32)   # every vector ties
    q = np.zeros((2, 3), dtype=np.float32)
    ids, dist = groundtruth.exact_topk_chunk(q, base, k=10)
    # which tied ids are kept may vary; the distances (what recall uses) may not
    assert len(set(ids[0].tolist())) == 10 and np.allclose(dist, np.sqrt(3), rtol=1e-6)


def test_ground_truth_int8_base():
    base = RNG.integers(-100, 100, size=(120, 5)).astype(np.int8)
    q = RNG.integers(-100, 100, size=(4, 5)).astype(np.float32)
    ids, _ = groundtruth.exact_topk_chunk(q, base, k=10, base_chunk=50)
    assert np.array_equal(ids, _brute(q, base.astype(np.float32), 10))
