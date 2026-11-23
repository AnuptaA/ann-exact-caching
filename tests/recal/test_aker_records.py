import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from recal import records as rec

ROOT = Path(__file__).resolve().parents[2]


def load_driver():
    path = ROOT / "simulations/approximate_hits/aker_real/eval_aker_real.py"
    spec = importlib.util.spec_from_file_location("eval_aker_real_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeAker:
    """Hits on a query id seen before (exact), misses otherwise."""

    def __init__(self, *args):
        self.cache = {}
        self.inserted = 0

    def lookup(self, key, vec):
        if key in self.cache:
            return 1, self.cache[key], 1e-6, 0.0
        return 0, None, 1e-6, 0.0

    def insert(self, key, vec, nid, nvec):
        self.cache[key] = nid
        self.inserted += len(nid)
        return True, 1e-6

    def status(self):
        n = len(self.cache)
        pool_mb = self.inserted * (48 + 4 * 3) / 1e6
        return f"  Total entries: {n}\n  Physical entries: {n}\n    Pool: {pool_mb:.6f} MB\n"

    def close(self):
        pass

    def process(self):
        return 0.0


class FakeWorld:
    ev = None
    present = None
    live = None
    recall_measured = 1.0

    def __init__(self, rows):
        self.rows = rows

    def truth(self, qi, n):
        return self.rows[qi][:n]

    def answer(self, qi, n):
        return self.rows[qi][:n]

    def events(self, i):
        return None, None


def test_baseline_records_match_the_quality_log(monkeypatch):
    mod = load_driver()
    monkeypatch.setattr(mod, "Aker", FakeAker)
    rng = np.random.default_rng(0)
    rows = np.stack([rng.permutation(50)[:10] for _ in range(20)]).astype(np.int64)
    # the fake returns ids + 1 shifted back by the driver; store ids as given
    stream = rng.integers(0, 20, 400)
    args = argparse.Namespace(k=5, delta=2, budget=0.2, risk=0.1, alpha_tighten=0.25, alpha_loosen=1.1,
                              warmup=100, batch=100, progress=10**9, max_requests=None,
                              equal_bytes=False, coord_bytes=4.0, pool_scale=1.0, pool_capacity=None)
    db = np.zeros((50, 3), np.float32)
    q = np.zeros((20, 3), np.float32)
    records = rec.Records()
    res = mod.run_quality(args, db, q, FakeWorld(rows), stream, records)
    cols = records.arrays()
    n = len(cols["index"])
    assert n == 300 and cols["index"][0] == 100
    hits = cols["path"] == rec.MAIN
    assert hits.sum() == sum(b["served"] for b in res["log"])
    assert np.isclose(cols["recall"][hits].sum(), sum(b["served_retention_sum"] for b in res["log"]))
    miss = cols["path"] == rec.MISS
    assert (cols["backend_kind"][miss] == rec.BACKEND_MISS).all()
    assert (cols["rereads"][miss] == 7).all() and (cols["rereads"][hits] == 0).all()


def make_esci_prep(tmp_path):
    from recal.backend import ExactBackend
    from tests.recal.test_system import make_stream, make_world
    rng, data, centers = make_world(seed=5, n=1500)
    stream = make_stream(rng, centers, 2400, repeat_share=0.3)
    uniq, inverse = np.unique(np.stack(stream), axis=0, return_inverse=True)
    uniq = uniq.astype(np.float32)
    esci = tmp_path / "esci"
    esci.mkdir()
    np.save(esci / "esci_db1500_s42.npy", data)
    np.save(esci / "esci_queries.npy", uniq)
    prep = tmp_path / "prep"
    prep.mkdir()
    np.save(prep / "top500.npy", ExactBackend(data).search(uniq, 100)[0].astype(np.int32))
    np.savez(prep / "streams.npz", random=inverse.astype(np.int64).ravel())
    (prep / "meta.json").write_text('{"kind": "esci", "n_db": 1500}')
    return prep, esci, len(inverse)


def test_online_simulators_write_matching_records(tmp_path, monkeypatch):
    import sys
    from simulations.approximate_hits import online_sim
    prep, esci, n = make_esci_prep(tmp_path)
    monkeypatch.setattr(online_sim, "ESCI", esci)
    for method in ("aker", "qvcache"):
        out = tmp_path / f"out_{method}"
        argv = ["x", "--prep", str(prep), "--stream", "random", "--method", method, "--budget", "0.3",
                "--k", "5", "--warmup", "500", "--batch", "400", "--records", "--out", str(out)]
        monkeypatch.setattr(sys, "argv", argv)
        online_sim.main()
        res = json.loads((out / f"{method}.json").read_text())
        cols, _, meta = rec.load(out / f"{method}_records.npz")
        assert len(cols["index"]) == n - 500 and cols["index"][0] == 500
        hits = cols["path"] == rec.MAIN
        assert hits.sum() == sum(b["served"] for b in res["log"])
        assert np.isclose(cols["recall"][hits].sum(), sum(b["served_retention_exact_sum"] for b in res["log"]),
                          rtol=1e-3)
        assert (cols["rereads"][cols["path"] == rec.MISS] == 5).all()


def test_sim_record_batch_marks_mini_hits_and_misses():
    from simulations.approximate_hits import online_sim
    sim = object.__new__(online_sim.Sim)
    sim.k = 5
    sim.records = rec.Records()
    sim._tmask = np.array([True, False, False])
    serve = np.array([True, True, False])
    sim._record_batch(10, serve, np.array([1.0, 0.8, 0.0]), np.array([1.0, 1.0, 0.9]), "static")
    cols = sim.records.arrays()
    assert list(cols["index"]) == [10, 11, 12]
    assert list(cols["path"]) == [rec.MINI, rec.MAIN, rec.MISS]
    assert np.allclose(cols["recall"], [1.0, 0.8, 0.9]) and cols["rereads"].sum() == 0


def test_equal_bytes_mode_calibrates_capacity_and_reports_it(monkeypatch):
    mod = load_driver()
    monkeypatch.setattr(mod, "Aker", FakeAker)
    rng = np.random.default_rng(1)
    rows = np.stack([rng.permutation(50)[:10] for _ in range(20)]).astype(np.int64)
    stream = rng.integers(0, 20, 300)
    args = argparse.Namespace(k=5, delta=0, budget=0.5, risk=0.1, alpha_tighten=0.25, alpha_loosen=1.1,
                              warmup=50, batch=100, progress=10**9, max_requests=None,
                              equal_bytes=True, coord_bytes=1.0, pool_scale=1.0, pool_capacity=None)
    db = np.zeros((100_000, 3), np.float32)
    q = np.zeros((20, 3), np.float32)
    res = mod.run_quality(args, db, q, FakeWorld(rows), stream)
    from recal import accounting
    M = accounting.budget_bytes(100_000, 3, 0.5, 1.0)
    # the fake holds 5 pooled objects per entry
    expected = accounting.aker_capacity_for_budget(M, 3, 1.0, 5.0)
    assert res["M"] == M and res["configured_capacity"] == res["pool_slots"] == expected
    assert abs(res["bytes_at_full_capacity"] - M) <= 0.01 * M
    assert res["calibration"]["source"] == "calibrated_on_history" and res["calibration"]["objects_per_entry"] == 5.0
    assert res["bytes_used_end"] > 0 and res["bytes"]["total"] >= res["bytes_used_end"]
    assert res["within_M"] == (res["bytes"]["total"] <= M) and "pool_vectors" in res["bytes"]


def test_explicit_pool_capacity_skips_the_calibration(monkeypatch):
    mod = load_driver()
    monkeypatch.setattr(mod, "Aker", FakeAker)
    rows = np.tile(np.arange(10), (5, 1)).astype(np.int64)
    args = argparse.Namespace(k=5, delta=0, budget=0.5, risk=0.1, alpha_tighten=0.25, alpha_loosen=1.1,
                              warmup=5, batch=50, progress=10**9, max_requests=None,
                              equal_bytes=True, coord_bytes=1.0, pool_scale=1.0, pool_capacity=77)
    res = mod.run_quality(args, np.zeros((50, 3), np.float32), np.zeros((5, 3), np.float32), FakeWorld(rows),
                          np.arange(100) % 5)
    assert res["configured_capacity"] == 77 and res["calibration"]["source"] == "given"

def make_esci_prep(tmp_path):
    from recal.backend import ExactBackend
    from tests.recal.test_system import make_stream, make_world
    rng, data, centers = make_world(seed=5, n=1500)
    stream = make_stream(rng, centers, 2400, repeat_share=0.3)
    uniq, inverse = np.unique(np.stack(stream), axis=0, return_inverse=True)
    uniq = uniq.astype(np.float32)
    esci = tmp_path / "esci"
    esci.mkdir()
    np.save(esci / "esci_db1500_s42.npy", data)
    np.save(esci / "esci_queries.npy", uniq)
    prep = tmp_path / "prep"
    prep.mkdir()
    np.save(prep / "top500.npy", ExactBackend(data).search(uniq, 100)[0].astype(np.int32))
    np.savez(prep / "streams.npz", random=inverse.astype(np.int64).ravel())
    (prep / "meta.json").write_text('{"kind": "esci", "n_db": 1500}')
    return prep, esci, len(inverse)


def test_real_driver_parser_defaults_to_thirty_percent():
    import re
    text = (ROOT / "simulations/approximate_hits/aker_real/eval_aker_real.py").read_text()
    assert re.search(r'"--history-frac", type=float, default=0\.3', text)
    sim_text = (ROOT / "simulations/approximate_hits/online_sim.py").read_text()
    assert re.search(r'"--history-frac", type=float, default=0\.3', sim_text)


def test_online_simulator_warms_up_on_thirty_percent_by_default(tmp_path, monkeypatch):
    import sys
    from simulations.approximate_hits import online_sim
    prep, esci, n = make_esci_prep(tmp_path)
    monkeypatch.setattr(online_sim, "ESCI", esci)
    out = tmp_path / "o"
    monkeypatch.setattr(sys, "argv", ["x", "--prep", str(prep), "--stream", "random", "--method", "aker", "--budget",
                                      "0.3", "--k", "5", "--batch", "400", "--records", "--out", str(out)])
    online_sim.main()
    cols, _, meta = rec.load(out / "aker_records.npz")
    assert cols["index"][0] == int(0.3 * n) and meta["history_len"] == int(0.3 * n)
