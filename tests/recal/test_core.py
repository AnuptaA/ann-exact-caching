import math

import numpy as np
import pytest

from recal.core import (
    MAIN_HIT,
    MINI_HIT,
    MISS_ABSTAIN,
    MISS_K_MAX,
    MISS_NO_CANDIDATE,
    MISS_SCORE,
    Core,
)
from tests.recal.helpers import ID_BASE, build_core, exact_order, pool_ids
from tests.recal.reference import prototype as ref


def random_case(seed=0, n=600, d=8, n_entries=24, K=5):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n, d)).astype(np.float32)
    centers = rng.normal(size=(n_entries, d)).astype(np.float32)
    js = rng.integers(1, 6, size=n_entries)
    return rng, data, centers, js, K


def reference_serving(q, K, data, core_ids, centers, js, lists, radii, m):
    """Candidate, nr, best-fit offset and size, intruder count by the prototype's functions."""
    q64 = q.astype(np.float64)
    de = np.linalg.norm(centers.astype(np.float64) - q64, axis=1)
    read = np.argsort(de, kind="stable")[: (m if m else len(centers))]
    pool = set()
    for e in read:
        pool.update(lists[e][: js[e] * K].tolist())
    pool = np.array(sorted(pool))
    dist = np.linalg.norm(data[pool].astype(np.float64) - q64, axis=1)
    order = np.argsort(dist, kind="stable")[:K]
    nr = max(dist[order][-1], ref.EPS)
    fit = ref.fit_balls(np.array([nr]), de[read][None, :], radii[read], 1, np.nan)[0]
    return pool[order], dist[order], nr, fit[0], fit[1], read


def radii_of(data, centers, js, K):
    out = []
    for c, j in zip(centers, js):
        order = exact_order(data, c)[: j * K]
        out.append(np.linalg.norm(data[order[-1]].astype(np.float64) - c.astype(np.float64)))
    return np.array(out)


def test_candidate_features_match_prototype():
    rng, data, centers, js, K = random_case()
    core, ids, lists = build_core(data, centers, K, js)
    radii = radii_of(data, centers, js, K)
    for m in (0, 6):
        queries = rng.normal(size=(60, data.shape[1])).astype(np.float32)
        feat, cand_ids, cand_d = core.candidates_batch(queries, K, m)
        for i, q in enumerate(queries):
            want_ids, want_d, nr, off, size, read = reference_serving(q, K, data, ids, centers, js, lists, radii, m)
            assert feat[i, 0] == 1.0
            assert feat[i, 6] == len(read)
            np.testing.assert_array_equal(cand_ids[i], want_ids + ID_BASE)
            np.testing.assert_allclose(cand_d[i], want_d, rtol=1e-5)
            np.testing.assert_allclose(feat[i, 1], nr, rtol=1e-6)
            np.testing.assert_allclose(feat[i, 2], off, rtol=1e-5)
            np.testing.assert_allclose(feat[i, 3], size, rtol=1e-5)


def test_intruder_count_matches_prototype():
    rng, data, centers, js, K = random_case(seed=1)
    core, ids, lists = build_core(data, centers, K, js)
    radii = radii_of(data, centers, js, K)
    pool = pool_ids(core, K, js, lists)
    lo, hi = data.min(axis=0), data.max(axis=0)
    mismatches = 0
    queries = rng.normal(size=(80, data.shape[1])).astype(np.float32)
    feat, _, _ = core.candidates_batch(queries, K, 0)
    for i, q in enumerate(queries):
        _, _, nr, _, _, read = reference_serving(q, K, data, ids, centers, js, lists, radii, 0)
        shell = set()
        for e in read:
            shell.update((lists[e][js[e] * K:] + ID_BASE).tolist())
        shell = sorted(shell - pool)
        if shell:
            vecs = data[np.array(shell) - ID_BASE]
            rec = ref.quantize(np.vstack([vecs, lo, hi]), 4)[: len(shell)]
            count = int((np.linalg.norm(rec - q.astype(np.float64), axis=1) < nr).sum())
        else:
            count = 0
        mismatches += int(feat[i, 5] != count)
        assert feat[i, 4] == pytest.approx(math.log((feat[i, 5] + 1) / K))
    assert mismatches <= 1  # a quantization tie can round differently in float32


def test_containment_gives_exact_response():
    rng = np.random.default_rng(3)
    data = rng.normal(size=(400, 4)).astype(np.float32)
    center = np.zeros(4, np.float32)
    core, ids, lists = build_core(data, center[None, :], 5, [5], k_max=20)
    radius = core.entry(0)["radius"]
    exact_hits = 0
    for _ in range(200):
        q = (center + rng.normal(scale=0.05 * radius, size=4)).astype(np.float32)
        feat, cand, _ = core.candidates_batch(q[None, :], 5, 0)
        if feat[0, 2] + feat[0, 3] <= 1.0:
            truth = exact_order(data, q)[:5] + ID_BASE
            np.testing.assert_array_equal(np.sort(cand[0]), np.sort(truth))
            exact_hits += 1
    assert exact_hits > 50


def test_k_limits_and_missing_candidates():
    rng, data, centers, js, K = random_case(seed=4)
    core, _, _ = build_core(data, centers, K, js, k_max=10)
    q = rng.normal(size=data.shape[1]).astype(np.float32)
    assert core.lookup(q, 11, 0.5).path == MISS_K_MAX
    assert core.lookup(q, 0, 0.5).path == MISS_K_MAX
    one = Core(data.shape[1], 10)
    one.finish_build()
    assert one.lookup(q, 3, 0.5).path == MISS_NO_CANDIDATE
    # an entry list shorter than K cannot answer
    tiny, _, _ = build_core(data, centers[:1], 2, [1], k_max=10)
    assert tiny.lookup(q, 5, 0.5).path == MISS_NO_CANDIDATE


def test_score_decision_and_abstention():
    rng, data, centers, js, K = random_case(seed=5)
    mean, scale, coef, intercept = np.array([1.0, 2.0, 0.0]), np.array([0.5, 1.0, 2.0]), np.array([-1.0, -0.5, -0.25]), 0.3
    core, _, _ = build_core(data, centers, K, js, model=(mean, scale, coef, intercept))
    q = rng.normal(size=data.shape[1]).astype(np.float32)
    r = core.lookup(q, K, None)
    assert r.path == MISS_ABSTAIN and r.has_score
    z = intercept + np.dot(coef, (np.array([r.offset, r.size, r.rate]) - mean) / scale)
    assert r.score == pytest.approx(1 / (1 + math.exp(-z)))
    assert core.lookup(q, K, r.score - 1e-9).path == MAIN_HIT
    hit = core.lookup(q, K, r.score - 1e-9)
    assert len(hit.ids) == K and np.all(np.diff(hit.dists) >= 0)
    assert core.lookup(q, K, r.score + 1e-9).path == MISS_SCORE
    unset, _, _ = build_core(data, centers, K, js)
    with pytest.raises(RuntimeError):
        unset.lookup(q, K, 0.5)


def test_m_caps_entries_read():
    rng, data, centers, js, K = random_case(seed=6)
    core, _, _ = build_core(data, centers, K, js)
    q = rng.normal(size=data.shape[1]).astype(np.float32)
    feat, _, _ = core.candidates_batch(q[None, :], K, 4)
    assert feat[0, 6] == 4
    feat, _, _ = core.candidates_batch(q[None, :], K, 0)
    assert feat[0, 6] == len(centers)


def test_dead_slots_are_skipped_and_counted():
    rng, data, centers, js, K = random_case(seed=7)
    core, ids, lists = build_core(data, centers, K, js)
    q = rng.normal(size=data.shape[1]).astype(np.float32)
    before = core.lookup(q, K, None)
    victim = int(before.ids[0])
    live = core.n_data_slots()
    assert core.delete_id(victim) and not core.delete_id(victim)
    assert core.n_data_slots() == live - 1 and core.n_dead_slots() == 1
    after = core.lookup(q, K, None)
    assert victim not in after.ids
    assert core.bytes_by_structure()["data_pool"] == (live) * core.data_slot_bytes(data.shape[1])


def test_shared_shell_slots_counted_once_and_pool_vectors_excluded():
    rng, data, centers, js, K = random_case(seed=8)
    core, ids, lists = build_core(data, centers, K, js)
    pool = pool_ids(core, K, js, lists)
    shell_ids = set()
    total_refs = 0
    for j, order in zip(js, lists):
        listed = (order[j * K:] + ID_BASE).tolist()
        shell_ids.update(i for i in listed if i not in pool)
        total_refs += sum(1 for i in listed if i not in pool)
    assert core.n_shell_slots() == len(shell_ids)
    assert total_refs > len(shell_ids)  # some vectors are shared by several entries
    assert core.n_data_slots() == len(pool)


def test_mini_cache_lru_and_prefix_serving():
    rng = np.random.default_rng(9)
    d = 6
    core = Core(d, 20)
    core.mini_set_capacity(3)
    qs = [rng.normal(size=d).astype(np.float32) for _ in range(5)]
    resp = lambda k: (np.arange(k, dtype=np.int64), np.linspace(0.1, 1.0, k).astype(np.float32))  # noqa: E731
    for q in qs[:3]:
        core.mini_put(q, *resp(10))
    assert core.mini_size() == 3
    assert core.mini_get(qs[0], 4)[0].tolist() == [0, 1, 2, 3]  # prefix of a longer entry, touches qs[0]
    assert core.mini_get(qs[1], 11) is None  # an entry fetched for fewer than K cannot answer
    core.mini_put(qs[3], *resp(10))  # evicts the least recently used: qs[1]
    assert core.mini_get(qs[1], 5) is None and core.mini_get(qs[0], 5) is not None
    core.mini_set_capacity(1)
    assert core.mini_size() == 1
    assert core.hash_vector(qs[0]) == core.hash_vector(qs[0].copy())
    assert len({core.hash_vector(q) for q in qs}) == len(qs)


def test_mini_cache_order_matches_prototype_lru():
    rng = np.random.default_rng(10)
    d, R = 5, 7
    core = Core(d, 20)
    core.mini_set_capacity(R)
    proto = ref.MiniCache(R, k=3, policy="lru")
    pool = [rng.normal(size=d).astype(np.float32) for _ in range(20)]
    for step in range(400):
        q = pool[rng.integers(0, len(pool))]
        key = core.hash_vector(q)
        got = core.mini_get(q, 3) is not None
        want = proto.lookup(key, 3, None) == "backend"
        assert got == want
        if not got:
            core.mini_put(q, np.arange(3, dtype=np.int64), np.ones(3, np.float32))
            proto.write_backend(key, 3)
    assert core.mini_size() == len(proto)


def test_mini_hit_precedes_main_path_and_misses_do_not_touch_it():
    rng, data, centers, js, K = random_case(seed=11)
    core, _, _ = build_core(data, centers, K, js)
    core.mini_set_capacity(4)
    q = rng.normal(size=data.shape[1]).astype(np.float32)
    ids = np.arange(100, 100 + K, dtype=np.int64)
    core.mini_put(q, ids, np.linspace(0.1, 0.5, K).astype(np.float32))
    r = core.lookup(q, K, 2.0)  # lambda above any score: only the mini cache can answer
    assert r.path == MINI_HIT and r.ids.tolist() == ids.tolist() and not r.has_score
    assert core.lookup(q, K, None, use_mini=False).path == MISS_ABSTAIN


def test_entries_needed_matches_brute_force():
    rng, data, centers, js, K = random_case(seed=12)
    core, ids, lists = build_core(data, centers, K, js)
    pool = pool_ids(core, K, js, lists)
    queries = rng.normal(size=(40, data.shape[1])).astype(np.float32)
    truth = np.stack([exact_order(data, q)[:K] + ID_BASE for q in queries])
    got = core.entries_needed_batch(queries, truth)
    for i, q in enumerate(queries):
        present = [t for t in truth[i] if t in pool]
        if not present:
            assert got[i] == -1
            continue
        de = np.linalg.norm(centers.astype(np.float64) - q.astype(np.float64), axis=1)
        order = np.argsort(de, kind="stable")
        need = 0
        for t in present:
            first = next(r for r, e in enumerate(order) if t in set((lists[e][: js[e] * K] + ID_BASE).tolist()))
            need = max(need, first + 1)
        assert got[i] == need


def test_byte_accounting_sums_to_structures():
    rng, data, centers, js, K = random_case(seed=13)
    d = data.shape[1]
    core, ids, lists = build_core(data, centers, K, js, budget=1_000_000.0)
    core.mini_set_capacity(5)
    for _ in range(3):
        core.mini_put(rng.normal(size=d).astype(np.float32), np.arange(K, dtype=np.int64), np.ones(K, np.float32))
    core.set_external_bytes("calibration_window", 1234.0)
    b = core.bytes_by_structure()
    pool = pool_ids(core, K, js, lists)
    assert b["data_pool"] == len(pool) * (4 * d + 12)
    assert b["shell_pool"] == core.n_shell_slots() * ((d + 1) // 2) + 8 * d
    refs = sum(j * K for j in js) + sum(len(core.entry(e)["shell_slots"]) for e in range(len(js)))
    assert b["entries"] == len(js) * (4 * d + 20) + 4 * refs
    assert b["reverse_map"] == 12 * len(pool)
    assert b["mini_cache"] == 3 * (40 + 8 * K)
    assert b["headroom"] == 0.02 * 1_000_000
    assert b["center_index"] > 0 and b["calibration_window"] == 1234.0
    parts = [v for k, v in b.items() if k not in ("total", "budget")]
    assert b["total"] == pytest.approx(sum(parts))


def test_ten_thousand_vector_synthetic_run():
    rng = np.random.default_rng(14)
    data = rng.normal(size=(10_000, 16)).astype(np.float32)
    centers = rng.normal(size=(120, 16)).astype(np.float32)
    js = rng.integers(1, 6, size=120)
    core, ids, lists = build_core(data, centers, 10, js, budget=50_000_000.0)
    core.mini_set_capacity(256)
    queries = rng.normal(size=(300, 16)).astype(np.float32)
    feat, cand, _ = core.candidates_batch(queries, 10, 30)
    assert (feat[:, 0] == 1).mean() > 0.9
    for q in queries[:50]:
        r = core.lookup(q, 10, None, 30)
        assert r.path in (MISS_ABSTAIN, MISS_NO_CANDIDATE)
        core.mini_put(q, np.arange(10, dtype=np.int64), np.ones(10, np.float32))
    b = core.bytes_by_structure()
    assert b["total"] == pytest.approx(sum(v for k, v in b.items() if k not in ("total", "budget")))
    assert b["total"] > b["data_pool"]


def test_mini_cache_stores_four_byte_ids_and_payload_matches_formula():
    rng, data, centers, js, K = random_case(seed=14)
    d = data.shape[1]
    core, *_ = build_core(data, centers, K, js, budget=1_000_000.0)
    core.mini_set_capacity(4)
    for i in range(4):
        core.mini_put(rng.normal(size=d).astype(np.float32), np.arange(K, dtype=np.int64) + i, np.ones(K, np.float32))
    assert core.mini_payload_bytes() == 4 * 8 * K
    assert core.bytes_by_structure()["mini_cache"] == 4 * (40 + 8 * K)
    h, ids, dists = core.mini_dump()[0]
    assert ids.dtype == np.int64 and len(ids) == K


def test_mini_cache_rejects_ids_beyond_four_bytes():
    rng, data, centers, js, K = random_case(seed=15)
    core, *_ = build_core(data, centers, K, js, budget=1_000_000.0)
    core.mini_set_capacity(2)
    q = rng.normal(size=data.shape[1]).astype(np.float32)
    with pytest.raises(ValueError):
        core.mini_put(q, np.array([2**40] * K, dtype=np.int64), np.ones(K, np.float32))


def test_center_index_keeps_no_copy_of_the_centers():
    rng = np.random.default_rng(3)
    n, d, K = 400, 128, 5
    data = rng.normal(size=(n, d)).astype(np.float32)
    centers = rng.normal(size=(30, d)).astype(np.float32)
    core, *_ = build_core(data, centers, K, [1] * 30, budget=50_000_000.0)
    assert core.center_index_payload_per_element() == 12
    index_bytes = core.bytes_by_structure()["center_index"]
    assert index_bytes < 30 * 4 * d    # a copy of the centers alone would exceed this per element sum
    # allocated size: level-0 blocks and pointers charged for every slot, links above level 0 on top
    assert index_bytes >= 30 * (12 + 8 + 4)
