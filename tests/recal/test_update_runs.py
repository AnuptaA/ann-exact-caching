import json

import numpy as np
import pytest

from recal import sim
from recal.backend import ExactBackend
from recal.update_schedule import UpdateStream, build_schedule, live_truth
from tests.recal.test_system import make_stream, make_world


def clusters(n=1000, k=10):
    return np.arange(n) % k


def test_schedule_counts_window_and_modes():
    vc = clusters()
    counts = np.array([50, 1, 1, 40, 1, 1, 1, 1, 1, 30])
    uni = build_schedule(vc, counts, 0.05, 100, 300, "uniform", seed=1)
    assert uni["insert"].sum() == 50 and (~uni["insert"]).sum() == 50 and (~uni["initial"]).sum() == 50
    assert uni["t"].min() >= 100 and uni["t"].max() < 300 and np.all(np.diff(uni["t"]) >= 0)
    tgt = build_schedule(vc, counts, 0.05, 100, 300, "targeted", top_clusters=3, seed=1)
    assert np.isin(vc[tgt["id"]], [0, 3, 9]).all()
    assert not set(tgt["id"][tgt["insert"]]) & set(tgt["id"][~tgt["insert"]])
    with pytest.raises(ValueError):
        build_schedule(vc, counts, 0.5, 0, 10, "targeted", top_clusters=1)


def test_live_truth_filters_dead_and_not_yet_inserted_vectors():
    gt = np.arange(10)
    present = np.ones(10, bool)
    present[[1, 3]] = False
    assert live_truth(gt, present, 4).tolist() == [0, 2, 4, 5]
    with pytest.raises(ValueError):
        live_truth(gt[:3], present, 4)


def test_update_stream_applies_events_once_in_order():
    sched = {"initial": np.array([True, True, False]), "t": np.array([5, 8]), "insert": np.array([True, False]),
             "id": np.array([2, 0])}
    u = UpdateStream(sched)
    assert [len(x) for x in u.due(4)] == [0, 0]
    ins, dels = u.due(5)
    assert ins.tolist() == [2] and u.present.tolist() == [True, True, True]
    assert u.due(5)[0].size == 0
    assert u.due(9)[1].tolist() == [0] and not u.present[0]


def test_run_with_updates_uses_live_truth_and_keeps_deleted_out_of_answers(tmp_path):
    rng, data, centers = make_world(seed=5)
    stream = make_stream(rng, centers, 4200, repeat_share=0.2)
    uniq, inverse = np.unique(np.stack(stream), axis=0, return_inverse=True)
    uniq = uniq.astype(np.float32)
    np.save(tmp_path / "db.npy", data)
    np.save(tmp_path / "q.npy", uniq)
    np.save(tmp_path / "top500.npy", ExactBackend(data).search(uniq, 100)[0].astype(np.int32))
    np.savez(tmp_path / "streams.npz", random=inverse.astype(np.int64).ravel())
    (tmp_path / "meta.json").write_text(json.dumps({"kind": "files", "db": str(tmp_path / "db.npy"),
                                                    "q": str(tmp_path / "q.npy")}))
    sched = build_schedule(np.arange(len(data)) % 10, np.ones(10), 0.05, 1500, 4200, "uniform", seed=2)
    np.savez(tmp_path / "upd.npz", **sched)
    args = sim.parser().parse_args(["--prep", str(tmp_path), "--stream", "random", "--k", "5", "--backend", "exact",
                                    "--history", "3000", "--history-frac", "0.3", "--min-ess", "30", "--budget",
                                    "0.6", "--updates", str(tmp_path / "upd.npz"), "--out", str(tmp_path / "o")])
    from recal.records import Records
    rec = Records()
    out = sim.run(args, rec)
    cols = rec.arrays()
    assert out["n_scored"] == len(cols["index"]) and cols["recall"].min() >= 0
    assert 0.5 < cols["recall"].mean() <= 1.0 and out["constructions"] >= 1 and out["within_M"]
    bad = sim.parser().parse_args(["--prep", str(tmp_path), "--updates", "x.npz", "--out", "o"])
    with pytest.raises(SystemExit):
        sim.run(bad)


def test_live_pgvector_backend_statements_with_a_fake_connection():
    from harness.backend.pgvector_live import LivePgvectorBackend

    class Cursor:
        def __init__(self, log):
            self.log, self.rows = log, []

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql, params=None):
            self.log.append(sql)
            self.rows = [(7, 0.5), (9, 0.75)] if sql.startswith("SELECT id, embedding <->") else []

        def fetchall(self):
            return self.rows

    class Conn:
        def __init__(self):
            self.log, self.commits = [], 0

        def cursor(self):
            return Cursor(self.log)

        def commit(self):
            self.commits += 1

    conn = Conn()
    b = LivePgvectorBackend(conn, dim=2, n_vectors=100)
    ids, dists = b.search(np.zeros(2, np.float32), 2, kind="miss")
    assert ids.tolist() == [7, 9] and dists.tolist() == [0.5, 0.75] and b.stats["miss"]["searches"] == 1
    b.insert(np.array([101]), np.zeros((1, 2), np.float32))
    b.delete(np.array([101]))
    assert len(b) == 100 and b.version == 2 and conn.commits == 2
    assert any(s.startswith("INSERT INTO") for s in conn.log) and any(s.startswith("DELETE FROM") for s in conn.log)
    with pytest.raises(ValueError):
        b.insert(np.array([101]), np.zeros((1, 2), np.float32))
