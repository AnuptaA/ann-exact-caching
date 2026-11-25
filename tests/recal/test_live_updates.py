import argparse
import json
import sys

import numpy as np
import pytest

from recal import sim
from recal.backend import ExactBackend
from recal.update_schedule import build_schedule
from tests.recal.test_aker_records import FakeAker, load_driver
from tests.recal.test_prep_convert import make_prep, world  # noqa: F401  (fixture)
from workloads.q1 import shifts


class MockLive(ExactBackend):
    """An exact backend over the vectors present at the start, logging searches and data-change events."""

    def __init__(self, db, present):
        super().__init__(db[present], np.flatnonzero(present))
        self.search_kinds, self.events = [], []

    def search(self, queries, depth, kind="serve"):
        self.search_kinds.append(kind)
        return super().search(queries, depth, kind)

    def insert(self, ids, vectors):
        self.events += [("insert", int(i)) for i in ids]
        super().insert(ids, vectors)

    def delete(self, ids):
        self.events += [("delete", int(i)) for i in ids]
        super().delete(ids)


class UpdatingFakeAker(FakeAker):
    def write(self, key, vec):
        return 0.0

    def delete(self, key):
        return 0.0


@pytest.fixture
def update_prep(world, tmp_path):  # noqa: F811
    import shutil
    from recal import prep_convert
    prep = tmp_path / "prep"
    shutil.copytree(make_prep(world), prep)
    vc = shifts.assign(world["base"], world["centroids"])
    schedule = build_schedule(vc, np.ones(8), 0.05, 450, 1500, "uniform", seed=2)
    prep_convert.convert_updates(prep, schedule, world["base"], depth=500)
    present = schedule["initial"]
    return prep, world, present


def recal_events(update_prep, tmp_path, monkeypatch):
    prep, world, present = update_prep
    live = MockLive(world["base"], present)
    monkeypatch.setattr(sim, "make_backend", lambda kind, *a, **k: live)
    args = sim.parser().parse_args(["--prep", str(prep), "--stream", "abrupt", "--k", "5", "--backend", "pgvector",
                                    "--pg-dsn", "mock", "--budget", "0.6", "--history", "3000", "--min-ess", "30",
                                    "--updates", str(prep / "updates.npz"), "--strict-modes", "--out", str(tmp_path / "o")])
    out = sim.run(args)
    return live, out


def test_recal_sends_misses_and_events_to_the_live_backend(update_prep, tmp_path, monkeypatch):
    live, out = recal_events(update_prep, tmp_path, monkeypatch)
    assert out["backend"] == "pgvector" and out["mode"] == "deployable"
    assert "miss" in live.search_kinds and live.events and out["within_M"]


def test_real_aker_driver_uses_the_same_live_interface_and_events(update_prep, tmp_path, monkeypatch):
    prep, world_, present = update_prep
    live_recal, _ = recal_events(update_prep, tmp_path, monkeypatch)
    mod = load_driver()
    monkeypatch.setattr(mod, "Aker", UpdatingFakeAker)
    base = world_["base"]
    q = np.load(prep / "q.npy")
    top = np.load(prep / "top500.npy")
    stream = np.load(prep / "streams.npz")["abrupt"]
    args = argparse.Namespace(k=5, delta=0, budget=0.2, risk=0.1, alpha_tighten=0.25, alpha_loosen=1.1, warmup=450,
                              batch=300, progress=10**9, max_requests=None, equal_bytes=False, coord_bytes=4.0,
                              pool_scale=1.0, pool_capacity=500, data_change="churn", schedule="gradual",
                              backend_recall=1.0, backend_file=None, backend_target="0.95", pg_dsn="mock",
                              precomputed_answers=False)
    w = mod.World(args, prep, top)
    w.q = q
    w.live = MockLive(base, present)
    res = mod.run_quality(args, base, q, w, stream)
    assert res["backend"] == "pgvector" and "live pgvector adapter" in res["backend_adapter"]
    assert w.live.events == live_recal.events and "miss" in w.live.search_kinds
    assert res["summary"]["retention_vs_backend"] is None


def test_aker_simulator_replays_the_same_events_on_the_same_interface(update_prep, tmp_path, monkeypatch):
    from simulations.approximate_hits import online_sim
    prep, world_, present = update_prep
    live_recal, _ = recal_events(update_prep, tmp_path, monkeypatch)
    live = MockLive(world_["base"], present)
    monkeypatch.setattr(online_sim, "make_live_backend", lambda args, D: live)
    monkeypatch.setattr(sys, "argv", ["x", "--prep", str(prep), "--stream", "abrupt", "--method", "aker", "--budget",
                                      "0.3", "--k", "5", "--batch", "1", "--data-change", "churn", "--pg-dsn", "mock",
                                      "--out", str(tmp_path / "sim")])
    online_sim.main()
    assert live.events == live_recal.events and "miss" in live.search_kinds
    result = json.loads((tmp_path / "sim" / "aker.json").read_text())
    assert sum(b["events"] for b in result["log"]) > 0


def test_a_data_change_run_without_a_live_backend_is_refused(update_prep, tmp_path, monkeypatch):
    from simulations.approximate_hits import online_sim
    prep, _, _ = update_prep
    mod = load_driver()
    with pytest.raises(SystemExit, match="--pg-dsn"):
        mod.require_live_or_flag(argparse.Namespace(pg_dsn=None, precomputed_answers=False))
    mod.require_live_or_flag(argparse.Namespace(pg_dsn=None, precomputed_answers=True))
    monkeypatch.setattr(sys, "argv", ["x", "--prep", str(prep), "--stream", "abrupt", "--method", "aker", "--budget",
                                      "0.3", "--k", "5", "--data-change", "churn", "--out", str(tmp_path / "x")])
    with pytest.raises(SystemExit, match="--pg-dsn"):
        online_sim.main()


def test_every_update_job_line_carries_its_own_clone_dsn():
    from recal import matrix
    cfg = json.loads(open("experiments/matrix.json").read())
    jobs = matrix.cell_jobs(cfg, {"q1_budgets": [0.001], "standin_5pct": 0.0003})[0]["q4"]
    updates = [j.line for j in jobs if j.kind == "update"]
    assert len(updates) == 30 and all("--backend pgvector --pg-dsn @$PG_CLONES/" in x for x in updates)
    dsns = {x.split("--pg-dsn @")[1].split(" ")[0] for x in updates}
    assert len(dsns) == 30
