#!/usr/bin/env python3
"""Tests for utils.cache.workloads on tiny fake data (no external files)."""

import argparse
import json

import numpy as np
import pytest

from utils.base.knn import exact_topk
from utils.cache import workloads as wl


def _esci_dir(tmp_path, n_db=100, n_q=500, d=4, seed=0):
    rng = np.random.default_rng(seed)
    np.save(tmp_path / f"esci_db{n_db}_s{seed}.npy", rng.standard_normal((n_db, d)).astype(np.float32))
    np.save(tmp_path / "esci_queries.npy", rng.standard_normal((n_q, d)).astype(np.float32))
    return tmp_path


def _esci_cfg(tmp_path, **kw):
    cfg = dict(dataset="esci", esci_dir=str(tmp_path), n_db=100, seed=0, n_test=50,
               n_cand_queries=100, n_score_queries=100, k=3, k_values=[5, 20])
    cfg.update(kw)
    return cfg


def test_run_data_validation():
    base = dict(k=3, k_values=[5, 20], db=np.zeros((10, 2)), cand_q=np.zeros((2, 2)), sets={},
                topk={}, cached_ids=[np.arange(5)] * 4, radius=np.ones(4))
    wl.RunData(**base)
    for bad_kv in ([], [0, 5], [20, 5], [5, 5]):
        with pytest.raises(ValueError, match="k_values"):
            wl.RunData(**dict(base, k_values=bad_kv))
    with pytest.raises(ValueError, match="k must"):
        wl.RunData(**dict(base, k=0))
    with pytest.raises(ValueError, match="cached_ids"):
        wl.RunData(**dict(base, cached_ids=[np.arange(5)] * 3))
    with pytest.raises(ValueError, match="radius"):
        wl.RunData(**dict(base, radius=np.ones(5)))
    with pytest.raises(ValueError, match="weights"):
        wl.RunData(**dict(base, sets={"fit": (np.zeros((3, 2)), np.ones(2))}))
    with pytest.raises(ValueError, match="top-k"):
        wl.RunData(**dict(base, sets={"fit": (np.zeros((3, 2)), np.ones(3))},
                          topk={"fit": np.zeros((3, 2), dtype=int)}))


def test_load_inputs_esci_is_deterministic_and_splits_are_disjoint(tmp_path):
    _esci_dir(tmp_path)
    args = argparse.Namespace(**_esci_cfg(tmp_path))
    a = wl.load_inputs(args)
    b = wl.load_inputs(args)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    db, cand_q, score_q, test = a
    assert (len(db), len(cand_q), len(score_q), len(test)) == (100, 100, 100, 50)
    rows = np.vstack([cand_q, score_q, test])
    assert len(np.unique(rows, axis=0)) == len(rows)           # no query appears twice


def test_short_query_pool_is_an_error_not_a_short_split(tmp_path):
    _esci_dir(tmp_path, n_q=200)
    with pytest.raises(ValueError, match="pool has only 200"):
        wl.load_inputs(argparse.Namespace(**_esci_cfg(tmp_path)))


def test_load_run_data_esci_splits_and_tables(tmp_path):
    _esci_dir(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "results.json").write_text(json.dumps({"config": _esci_cfg(tmp_path), "results": []}))
    data, cfg, run = wl.load_run_data(run_dir, 30)
    assert set(run.sets) == {"fit", "dev", "cal", "test"}
    assert [len(run.sets[s][0]) for s in ("fit", "dev", "cal", "test")] == [30, 30, 30, 50]
    assert len(run.cached_ids) == 100 * 2 and run.radius.shape == (200,)
    assert run.topk["fit"].shape == (30, 3) and len(run.cached_ids[1]) == 20
    train = {tuple(r) for r in np.vstack([run.sets[s][0] for s in ("fit", "dev", "cal")])}
    assert len(train) == 90 and not train & {tuple(r) for r in run.sets["test"][0]}
    with pytest.raises(ValueError, match="pool has only"):
        wl.load_run_data(run_dir, 100)                          # 250 + 300 > 500


def test_simzipf_groups_dedup_and_validation():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((20, 4))
    w = wl.SimZipfWorkload(base, 5, 0.99, "interp", 0.5, 0.1, seed=1)
    ids = w.sample(200, np.random.default_rng(2))
    assert ids.min() >= 0 and ids.max() < 20 * 5
    assert not (w.partners == np.arange(20)[:, None]).any()    # never interpolates with itself
    v = w.vectors(np.array([0, 1, 7]))
    assert v.shape == (3, 4)
    anchor = base[[0, 0, 1]]
    assert (np.linalg.norm(v - anchor, axis=1) <= np.linalg.norm(
        base[w.partners[[0, 0, 1], [0, 1, 2]]] - anchor, axis=1) * 0.5 + 1e-9).all()
    g = wl.SimZipfWorkload(base, 5, 0.99, "gauss", 0.5, 0.1, seed=1)
    assert np.array_equal(g.vectors(np.array([3, 3])), g.vectors(np.array([3, 3])))
    uniq, counts, inv = wl.dedup(ids)
    assert counts.sum() == 200 and np.array_equal(uniq[inv], ids)
    with pytest.raises(ValueError, match="at least 2"):
        wl.SimZipfWorkload(base[:1], 5, 0.99, "interp", 0.5, 0.1, seed=1)


def _fake_aker(tmp_path, monkeypatch, n_req=400, n_distinct=40, d=6, n_base=50):
    rng = np.random.default_rng(0)
    base = rng.integers(-50, 50, size=(n_base, d)).astype(np.int8)
    pool = rng.integers(-50, 50, size=(n_distinct, d)).astype(np.int8)
    trace = pool[rng.integers(0, n_distinct, n_req)]
    np.save(tmp_path / "spacev-10m.npy", base)
    np.save(tmp_path / "spacev-sim-100k-0.3.npy", trace)
    monkeypatch.setattr(wl, "AKER", tmp_path)
    return base, trace


def test_trace_splits_weights_and_base_dtype(tmp_path, monkeypatch):
    base, trace = _fake_aker(tmp_path, monkeypatch)
    name = "spacev-aker-0.3"
    assert wl.load_base(name).dtype == np.int8
    uniq, inv, bounds = wl.build_trace(name)
    assert len(inv) == 400 and len(uniq) <= 40 and np.array_equal(uniq[inv], trace)
    ends = [bounds[s] for s in wl.TRACE_SPLITS]
    assert ends[0][0] == 0 and ends[-1][1] == 400
    assert all(a[1] == b[0] for a, b in zip(ends, ends[1:]))   # contiguous, covering
    total = 0
    for s in wl.TRACE_SPLITS:
        ids, weights, local = wl.split_view(inv, bounds, s)
        lo, hi = bounds[s]
        assert weights.sum() == hi - lo and len(local) == hi - lo
        assert np.array_equal(ids[local], inv[lo:hi])
        total += weights.sum()
    assert total == 400
    float_base = np.arange(12, dtype=np.float64).reshape(4, 3)
    np.save(tmp_path / "spacev-10m.npy", float_base)
    assert wl.load_base(name).dtype == np.float32


def test_too_short_trace_is_rejected(tmp_path, monkeypatch):
    _fake_aker(tmp_path, monkeypatch, n_req=5)
    with pytest.raises(ValueError, match="too short"):
        wl.build_trace("spacev-aker-0.3")
    with pytest.raises(ValueError, match="too short"):
        wl.split_boundaries("spacev-real", [np.zeros(1000), np.zeros(10)])


def test_spacev_bin_roundtrip(tmp_path):
    arr = np.arange(24, dtype=np.int8).reshape(4, 6)
    path = tmp_path / "q.bin"
    path.write_bytes(np.array([4, 6], dtype=np.int32).tobytes() + arr.tobytes())
    assert np.array_equal(wl._load_vectors(path), arr)


def test_load_run_data_for_a_trace_run(tmp_path, monkeypatch):
    base, _ = _fake_aker(tmp_path, monkeypatch)
    name = "spacev-aker-0.3"
    db = wl.load_base(name)
    uniq, inv, bounds = wl.build_trace(name)
    k_values, k = [5, 10], 3
    cand_ids = wl.split_view(inv, bounds, "cand")[0]
    cand_top = exact_topk(uniq[cand_ids], db, max(k_values))
    topk_all = exact_topk(uniq, db, k)
    run_dir = tmp_path / "trace_run"
    run_dir.mkdir()
    np.savez(run_dir / "topk.npz", uniq=uniq, inv=inv, topk=topk_all, cand_top=cand_top,
             **{f"bounds_{s}": np.array(b) for s, b in bounds.items()})
    cfg = dict(trace=name, k=k, k_values=k_values)
    (run_dir / "results.json").write_text(json.dumps({"config": cfg, "results": []}))
    data, _, run = wl.load_run_data(run_dir, 0)
    assert set(run.sets) == {"fit", "dev", "cal", "test"}
    for s in ("fit", "dev", "cal", "test"):
        assert len(run.sets[s][0]) == len(run.topk[s]) and run.topk[s].shape[1] == k
    assert len(run.cached_ids) == len(cand_ids) * 2
    # a stale neighbor file narrower than max(k_values) is rejected instead of truncated
    np.savez(run_dir / "topk.npz", uniq=uniq, inv=inv, topk=topk_all, cand_top=cand_top[:, :6],
             **{f"bounds_{s}": np.array(b) for s, b in bounds.items()})
    with pytest.raises(ValueError, match="columns"):
        wl.load_run_data(run_dir, 0)
