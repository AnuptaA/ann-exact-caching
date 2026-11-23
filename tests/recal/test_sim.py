import json

import numpy as np
import pytest

from recal import sim
from tests.recal.test_system import make_stream, make_world


@pytest.fixture
def prep(tmp_path):
    rng, data, centers = make_world(seed=5)
    stream = make_stream(rng, centers, 4200, repeat_share=0.2)
    uniq, inverse = np.unique(np.stack(stream), axis=0, return_inverse=True)
    np.save(tmp_path / "db.npy", data)
    np.save(tmp_path / "q.npy", uniq.astype(np.float32))
    from recal.backend import ExactBackend
    top = ExactBackend(data).search(uniq.astype(np.float32), 100)[0].astype(np.int32)
    np.save(tmp_path / "top500.npy", top)
    np.savez(tmp_path / "streams.npz", random=inverse.astype(np.int64).ravel())
    (tmp_path / "meta.json").write_text(json.dumps({"kind": "files", "db": str(tmp_path / "db.npy"),
                                                    "q": str(tmp_path / "q.npy")}))
    return tmp_path


@pytest.mark.parametrize("backend", ["exact", "stored"])
def test_cell_writes_the_result_shape_and_bytes(prep, tmp_path, backend):
    args = sim.parser().parse_args(["--prep", str(prep), "--stream", "random", "--budget", "0.6", "--k", "5",
                                    "--backend", backend, "--batch", "500", "--timing", "--log-served", "--history-frac", "0",
                                    "--cold-start", "1500", "--history", "3000", "--min-ess", "30",
                                    "--out", str(tmp_path / "out")])
    r = sim.run(args)
    assert r["method"] == "recal" and r["within_M"] and r["bytes"]["total"] <= r["M"]
    assert "bytes_first_construction" in r
    assert set(r["log"][0]) >= {"t", "n", "served", "served_retention_sum", "audited_served", "lambda", "memory",
                                "cached", "entries"}
    assert sum(b["n"] for b in r["log"]) == 4000 or sum(b["n"] for b in r["log"]) > 3000
    assert r["timing"]["n_timed"] == len(r["timing"]["hit"]) == len(r["timing"]["dt_ms"]) == 4200
    assert len(r["timing"]["rereads"]) == 4200 and sum(r["timing"]["rereads"]) > 0
    assert r["constructions"] >= 1 and r["args"]["warmup"] > 0
    assert {"miss", "audit", "candidate", "label"} >= set(r["background"]) - {"eval", "serve"}
    assert len(r["audit_pos"]) == r["counts"]["audits"]
    assert sum(b["served"] for b in r["log"]) > 0


def test_main_writes_json_atomically(prep, tmp_path):
    from recal import cells
    import sys
    out = tmp_path / "res"
    sys.argv = ["x", "--method", "recal", "--prep", str(prep), "--budget", "0.6", "--k", "5", "--backend", "exact", "--cold-start", "1500", "--history", "3000", "--min-ess", "30",
                "--out", str(out)]
    cells.main()
    data = json.loads((out / "recal.json").read_text())
    assert data["bytes"]["total"] <= data["M"] and not list(out.glob(".*tmp"))


def test_proximity_is_flagged_as_cluster_only(prep, tmp_path):
    from recal import cells
    import sys
    sys.argv = ["x", "--method", "proximity", "--prep", str(prep), "--out", str(tmp_path / "o")]
    with pytest.raises(SystemExit) as e:
        cells.main()
    assert "cluster" in str(e.value)


def test_next_fraction_stays_inside_the_bracket():
    from recal.cells import next_fraction
    assert next_fraction(0.05, 1.2e6, 1.0e6, None, 0.05) < 0.05
    f = next_fraction(0.05, 1.2e6, 1.0e6, 0.045, 0.05)
    assert 0.045 < f < 0.05
    assert 0.05 < next_fraction(0.05, 0.5e6, 1.0e6, 0.05, 0.08) < 0.08


def test_history_mode_builds_from_history_and_scores_the_rest(prep, tmp_path):
    args = sim.parser().parse_args(["--prep", str(prep), "--stream", "random", "--budget", "0.6", "--k", "5",
                                    "--backend", "exact", "--batch", "500", "--timing", "--history-frac", "0.3",
                                    "--history", "3000", "--min-ess", "30", "--out", str(tmp_path / "o")])
    r = sim.run(args)
    n = 4200
    assert r["history_len"] == int(0.3 * n) == r["first_scored"] == r["args"]["warmup"]
    assert r["n_scored"] == n - r["history_len"]
    assert sum(b["n"] for b in r["log"]) <= r["n_scored"]
    assert r["timing"]["request_ids"][0] == r["first_scored"] and r["timing"]["n_timed"] == r["n_scored"]
    assert r["constructions"] >= 1 and r["within_M"]


def test_history_zero_keeps_the_cold_start_behaviour(prep, tmp_path):
    args = sim.parser().parse_args(["--prep", str(prep), "--stream", "random", "--budget", "0.6", "--k", "5",
                                    "--backend", "exact", "--cold-start", "1500", "--history", "3000", "--history-frac", "0",
                                    "--min-ess", "30", "--out", str(tmp_path / "o")])
    r = sim.run(args)
    assert r["history_len"] == 0 and r["first_scored"] == 0 and r["n_scored"] == 4200
    assert r["args"]["warmup"] >= 1500


def test_baseline_gets_the_same_history_as_warmup(prep, tmp_path, monkeypatch):
    from recal import cells, history_mode
    seen = {}

    def fake_run(cmd, check, env):
        seen["cmd"] = cmd
        out = tmp_path / "o"
        out.mkdir(exist_ok=True)
        tag = cmd[cmd.index("--tag") + 1]
        res = {"args": {}, "log": [{"cached": 10, "entries": 2, "memory": 10.0}]}
        (out / f"static{tag}.json").write_text(json.dumps(res))

    monkeypatch.setattr(cells.subprocess, "run", fake_run)
    a = sim.parser().parse_args(["--prep", str(prep), "--stream", "random", "--budget", "0.6", "--k", "5",
                                 "--history-frac", "0.3", "--out", str(tmp_path / "o")])
    res = cells.run_baseline("prototype", a, attempts=1)
    n = 4200
    cmd = seen["cmd"]
    assert cmd[cmd.index("--warmup") + 1] == str(history_mode.history_length(n, 0.3))
    assert (res["history_len"], res["first_scored"], res["n_scored"]) == (1260, 1260, n - 1260)


def test_records_file_rows_match_the_run(prep, tmp_path):
    from recal import records as rec
    args = sim.parser().parse_args(["--prep", str(prep), "--stream", "random", "--budget", "0.6", "--k", "5",
                                    "--backend", "exact", "--history-frac", "0.3", "--history", "3000",
                                    "--min-ess", "30", "--out", str(tmp_path / "o")])
    records = rec.Records()
    r = sim.run(args, records)
    path = records.save(tmp_path / "r_records.npz")
    cols, constructions, meta = rec.load(path)
    assert len(cols["index"]) == r["n_scored"] and cols["index"][0] == r["first_scored"]
    n_logged = sum(b["n"] for b in r["log"])
    served = np.isin(cols["path"], (rec.MINI, rec.MAIN))
    assert served[:n_logged].sum() == sum(b["served"] for b in r["log"])
    assert (cols["path"] == rec.MINI).sum() == r["counts"]["mini_hits"]
    assert (cols["path"] == rec.MAIN).sum() == r["counts"]["main_hits"]
    ret = sum(b["served_retention_sum"] for b in r["log"])
    assert np.isclose(cols["recall"][:n_logged][served[:n_logged]].sum(), ret, rtol=1e-3)
    main = cols["path"] != rec.MINI
    assert np.isfinite(cols["cand_recall"][main]).sum() > 0
    assert (cols["n_contrib"][cols["path"] == rec.MAIN] >= 1).all()
    assert len(constructions) == r["constructions"] and "entries_by_size" in constructions[0]
    assert meta["history_len"] == r["history_len"]


def real_args(prep, tmp_path, method, extra=()):
    from recal import cells
    parser = sim.parser()
    parser.add_argument("--method")
    parser.add_argument("--prox-percentile", type=float, default=0.1)
    parser.add_argument("--ann-python", default=".venv-ann/bin/python")
    return parser.parse_args(["--method", method, "--prep", str(prep), "--budget", "0.05", "--k", "10",
                              "--history-frac", "0.3", "--coord-bytes", "1", "--out", str(tmp_path / "o"), *extra])


def test_real_aker_and_proximity_commands(prep, tmp_path):
    from recal import cells
    aker = cells.real_aker_command(real_args(prep, tmp_path, "aker"), 0.9, tmp_path / "x.json")
    assert aker[:2] == [".venv-ann/bin/python", cells.REAL_DRIVER] and "--prox-percentile" not in aker
    assert aker[aker.index("--history-frac") + 1] == "0.3" and "--equal-bytes" in aker
    assert aker[aker.index("--coord-bytes") + 1] == "1" and aker[aker.index("--pool-scale") + 1] == "0.9"
    prox = cells.real_aker_command(real_args(prep, tmp_path, "proximity", ["--prox-percentile", "1"]), 1.0,
                                   tmp_path / "x.json")
    assert prox[prox.index("--prox-percentile") + 1] == "1"


def test_real_aker_rescales_the_pool_until_within_m(prep, tmp_path, monkeypatch):
    from recal import cells
    monkeypatch.setattr(cells, "REAL_SHIM", tmp_path / "libakershim.so")
    (tmp_path / "libakershim.so").write_bytes(b"")
    (tmp_path / "o").mkdir()
    scales = []

    def fake_run(cmd, check, env):
        scale = float(cmd[cmd.index("--pool-scale") + 1])
        scales.append(scale)
        out = Path(cmd[cmd.index("--out") + 1])
        total = 1200.0 * scale
        out.write_text(json.dumps({"M": 1000.0, "bytes": {"total": total}, "within_M": total <= 1000.0}))

    from pathlib import Path
    monkeypatch.setattr(cells.subprocess, "run", fake_run)
    res = cells.run_real_aker(real_args(prep, tmp_path, "aker"))
    assert len(scales) == 2 and scales[1] < scales[0] and res["within_M"] and res["method"] == "aker"
    assert not list((tmp_path / "o").glob("*_try*.json"))


def test_missing_shim_is_flagged_as_cluster_only(prep, tmp_path, monkeypatch):
    from recal import cells
    monkeypatch.setattr(cells, "REAL_SHIM", tmp_path / "missing.so")
    with pytest.raises(SystemExit) as e:
        cells.run_real_aker(real_args(prep, tmp_path, "aker"))
    assert "cluster" in str(e.value)


def test_library_fingerprint_names_commit_and_vendored_tree():
    from recal.measure.aker_check import library_fingerprint
    fp = library_fingerprint(".")
    assert fp["source"] == "vendored" and fp["filter"]["efSearch"] == 16
    assert fp["code_commit"] and fp["aker_tree"]


def test_job_lines_cover_real_aker_and_both_proximity_settings():
    from recal import jobs
    out = jobs.lines("prepdir", "random", [0.05], ["aker", "proximity-p0.1", "proximity-p1"], "x")
    assert "--method aker " in out[0] and "aker.json" in out[0]
    assert "--method proximity --prox-percentile 0.1 --tag _p0.1" in out[1] and "proximity_p0.1.json" in out[1]
    assert "--prox-percentile 1 --tag _p1" in out[2]


def test_default_history_is_thirty_percent_without_the_flag(prep, tmp_path):
    args = sim.parser().parse_args(["--prep", str(prep), "--stream", "random", "--budget", "0.6", "--k", "5",
                                    "--backend", "exact", "--history", "3000", "--min-ess", "30",
                                    "--out", str(tmp_path / "o")])
    assert args.history_frac == 0.3
    r = sim.run(args)
    assert r["history_len"] == int(0.3 * 4200) and r["first_scored"] == r["history_len"]
    assert r["n_scored"] == 4200 - r["history_len"]


def test_baseline_cell_and_real_aker_commands_default_to_thirty_percent(prep, tmp_path, monkeypatch):
    from recal import cells
    seen = {}

    def fake_run(cmd, check, env):
        seen["cmd"] = cmd
        out = tmp_path / "o"
        out.mkdir(exist_ok=True)
        tag = cmd[cmd.index("--tag") + 1]
        (out / f"static{tag}.json").write_text(json.dumps({"args": {}, "log": [{"cached": 1, "entries": 1, "memory": 1.0}]}))

    monkeypatch.setattr(cells.subprocess, "run", fake_run)
    a = sim.parser().parse_args(["--prep", str(prep), "--stream", "random", "--budget", "0.6", "--k", "5",
                                 "--out", str(tmp_path / "o")])
    res = cells.run_baseline("prototype", a, attempts=1)
    assert res["history_len"] == 1260 and seen["cmd"][seen["cmd"].index("--warmup") + 1] == "1260"


def test_result_records_the_request_file_hash(prep, tmp_path):
    import hashlib
    from recal.measure import jobs_util as ju
    ju._data_files.clear()
    args = sim.parser().parse_args(["--prep", str(prep), "--stream", "random", "--budget", "0.6", "--k", "5",
                                    "--backend", "exact", "--history", "3000", "--min-ess", "30",
                                    "--out", str(tmp_path / "o")])
    result = ju.stamp_result(sim.run(args))
    entry = result["data_files"][0]
    assert entry["name"] == "q.npy"
    assert entry["sha256"] == hashlib.sha256((prep / "q.npy").read_bytes()).hexdigest()
    ju._data_files.clear()
