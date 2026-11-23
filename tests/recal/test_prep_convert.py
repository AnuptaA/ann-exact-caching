import json
import sys

import numpy as np
import pytest

from recal import prep_convert, sim
from recal import records as rec
from recal.update_schedule import build_schedule
from workloads.q1 import shifts
from workloads.q1.universe import build_universe


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("conv")
    rng = np.random.default_rng(0)
    centers = rng.normal(scale=4.0, size=(8, 12))
    base = np.vstack([c + rng.normal(size=(750, 12)) for c in centers]).astype(np.float32)
    official = (centers[rng.integers(0, 8, 300)] + rng.normal(size=(300, 12))).astype(np.float32)
    np.save(tmp / "base.npy", base)
    uni = build_universe(len(official), 64, 4, seed=0)
    centroids = shifts.kmeans_centroids(base, k=8, seed=0, allow_fallback=True)
    gc = shifts.assign(official[uni.base_idx], centroids)
    shift = shifts.shift_stream(uni, gc, "abrupt", 1500, start=500, n_clusters=8, seed=1)
    return {"tmp": tmp, "base": base, "official": official, "uni": uni, "centroids": centroids, "shift": shift}


def make_prep(world, name="abrupt"):
    out = world["tmp"] / name
    if not out.exists():
        prep_convert.convert_shift(out, "abrupt", world["shift"], world["uni"], world["official"], world["base"],
                                   world["tmp"] / "base.npy", world["centroids"], depth=500)
    return out


def test_shift_prep_has_everything_every_driver_reads(world):
    prep = make_prep(world)
    meta = json.loads((prep / "meta.json").read_text())
    streams = np.load(prep / "streams.npz")
    q = np.load(prep / "q.npy")
    assert meta["kind"] == "files" and meta["history_len"] == 450 and meta["n_requests"] == 1500
    assert len(streams["abrupt"]) == 1500 and streams["abrupt"].max() < len(q)
    assert np.array_equal(streams["abrupt"], streams["random"])
    assert streams["phase_abrupt"].shape == (1500,) and streams["cluster_abrupt"].shape == (1500,)
    assert np.load(prep / "k.npy").shape == (1500,) and np.load(prep / "top500.npy").shape == (len(q), 500)
    from workloads.q1.universe import request_vectors
    expected = request_vectors(world["uni"], world["official"], world["shift"]["ids"])
    assert np.array_equal(q[streams["abrupt"]], expected)


def run_args(prep, tmp, extra=()):
    return sim.parser().parse_args(["--prep", str(prep), "--stream", "abrupt", "--k", "5", "--backend", "exact",
                                    "--budget", "0.6", "--history", "3000", "--min-ess", "30",
                                    "--out", str(tmp / "o"), *extra])


def test_shift_stream_runs_end_to_end_through_recal(world, tmp_path):
    prep = make_prep(world)
    records = rec.Records()
    out = sim.run(run_args(prep, tmp_path), records)
    assert out["history_len"] == 450 and out["n_scored"] == 1050 and out["within_M"]
    cols = records.arrays()
    assert len(cols["index"]) == 1050 and cols["index"][0] == 450 and out["constructions"] >= 1


def test_shift_stream_runs_through_the_aker_simulator(world, tmp_path, monkeypatch):
    from simulations.approximate_hits import online_sim
    prep = make_prep(world)
    out = tmp_path / "aker"
    monkeypatch.setattr(sys, "argv", ["x", "--prep", str(prep), "--stream", "abrupt", "--method", "aker",
                                      "--budget", "0.3", "--k", "5", "--batch", "300", "--records",
                                      "--out", str(out)])
    online_sim.main()
    cols, _, meta = rec.load(out / "aker_records.npz")
    assert cols["index"][0] == 450 and len(cols["index"]) == 1050 and meta["history_len"] == 450


def test_update_schedule_converts_for_recal_and_the_aker_formats(world, tmp_path, monkeypatch):
    import shutil
    prep = tmp_path / "prep"
    shutil.copytree(make_prep(world), prep)
    vc = shifts.assign(world["base"], world["centroids"])
    schedule = build_schedule(vc, np.ones(8), 0.05, 450, 1500, "uniform", seed=2)
    prep_convert.convert_updates(prep, schedule, world["base"], depth=500)
    for name in ("updates.npz", "data_churn.npz", "rank_extra_churn.npz", "topU1000.npy"):
        assert (prep / name).exists()
    dc = np.load(prep / "data_churn.npz")
    n = int(round(0.05 * len(world["base"])))
    assert len(dc["ins_ids"]) == len(dc["del_ids"]) == n and dc["initial"].sum() == len(world["base"]) - n
    assert np.all(dc["ins_t"] >= 450) and not set(dc["ins_ids"]) & set(dc["del_ids"])
    # ReCal replays the schedule with live ground truth
    args = run_args(prep, tmp_path, ["--updates", str(prep / "updates.npz")])
    records = rec.Records()
    out = sim.run(args, records)
    assert out["n_scored"] == 1050 and out["within_M"] and 0.5 < records.arrays()["recall"].mean() <= 1.0
    # the Aker simulator replays the same events from the Aker-format files
    from simulations.approximate_hits import online_sim
    res = tmp_path / "aker_updates"
    monkeypatch.setattr(sys, "argv", ["x", "--prep", str(prep), "--stream", "abrupt", "--method", "aker",
                                      "--budget", "0.3", "--k", "5", "--batch", "300", "--data-change", "churn",
                                      "--precomputed-answers", "--out", str(res)])
    online_sim.main()
    result = json.loads((res / "aker.json").read_text())
    assert sum(b["events"] for b in result["log"]) > 0


def test_cluster_runs_refuse_without_faiss():
    try:
        import faiss  # noqa: F401
        pytest.skip("faiss is installed")
    except ImportError:
        pass
    with pytest.raises(RuntimeError, match="faiss is required"):
        shifts.kmeans_centroids(np.zeros((100, 4), np.float32), k=4)


def test_convert_script_writes_a_shift_prep(world, tmp_path):
    from scripts import convert_streams
    np.save(tmp_path / "official.npy", world["official"])
    np.savez(tmp_path / "shift.npz", **world["shift"])
    np.save(tmp_path / "centroids.npy", world["centroids"])
    convert_streams.main(["shift", "--stream", str(tmp_path / "shift.npz"), "--kind", "abrupt", "--official",
                          str(tmp_path / "official.npy"), "--base", str(world["tmp"] / "base.npy"), "--out",
                          str(tmp_path / "p"), "--groups", "64", "--group-size", "4", "--centroids",
                          str(tmp_path / "centroids.npy"), "--k-mixed", "--depth", "200"])
    assert set(np.load(tmp_path / "p" / "k.npy").tolist()) == {5, 10, 50, 100}
    assert json.loads((tmp_path / "p" / "meta.json").read_text())["history_len"] == 450
