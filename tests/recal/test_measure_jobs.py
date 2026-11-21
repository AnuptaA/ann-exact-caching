import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

from recal.measure import jobs_util as ju

ROOT = Path(__file__).resolve().parents[2]
LINE = (
    "[ -f out/d/recal_t1.json ] || .venv-core/bin/python -u -m recal.cells "
    "--method recal --prep p --budget 0.05 --out out/d --tag _t1 > logs/x.txt 2>&1"
)


def load_run_jobs():
    path = ROOT / "simulations/approximate_hits/run_jobs.py"
    spec = importlib.util.spec_from_file_location("run_jobs_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_atomic_write_roundtrip(tmp_path):
    path = tmp_path / "sub" / "r.json"
    ju.atomic_write_json(path, {"a": 1})
    assert json.loads(path.read_text()) == {"a": 1}
    assert [p.name for p in path.parent.iterdir()] == ["r.json"]


def test_atomic_write_failure_keeps_old_file(tmp_path):
    path = tmp_path / "r.json"
    ju.atomic_write_json(path, {"a": 1})
    with pytest.raises(TypeError):
        ju.atomic_write_json(path, {"a": object()})
    assert json.loads(path.read_text()) == {"a": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["r.json"]


def test_atomic_write_no_partial_file_visible(tmp_path, monkeypatch):
    path = tmp_path / "r.json"
    seen = []
    real_replace = os.replace

    def spy(src, dst):
        seen.append(path.exists())
        real_replace(src, dst)

    monkeypatch.setattr(ju.os, "replace", spy)
    ju.atomic_write_json(path, {"a": 1})
    assert seen == [False]


def test_config_hash_stable_and_sensitive():
    base = {"budget": "0.05", "method": "recal", "out": "a", "tag": "_x"}
    same = {"method": "recal", "budget": 0.05, "out": "b", "tag": "_y", "log_level": "debug"}
    assert ju.config_hash(base) == ju.config_hash(same)
    assert ju.config_hash(base) != ju.config_hash(dict(base, budget="0.1"))
    assert ju.config_hash(base) != ju.config_hash(dict(base, extra=True))


def test_argv_forms_agree():
    a = ju.argv_to_dict(["--max-ball-k", "5", "--flag", "--x=1"])
    assert a == {"max_ball_k": "5", "flag": True, "x": "1"}
    assert ju.jobline_args(LINE) == {
        "method": "recal", "prep": "p", "budget": "0.05", "out": "out/d", "tag": "_t1",
    }


def test_check_exit_codes(tmp_path):
    path = tmp_path / "r.json"
    assert ju.main(["check", str(path), LINE]) == 1
    good = {"config_hash": ju.config_hash(ju.jobline_args(LINE))}
    path.write_text(json.dumps(good))
    assert ju.main(["check", str(path), LINE]) == 0
    path.write_text(json.dumps({"config_hash": "other"}))
    assert ju.main(["check", str(path), LINE]) == 2
    path.write_text('{"config_hash": ')
    assert ju.main(["check", str(path), LINE]) == 2
    path.write_text("[1]")
    assert ju.main(["check", str(path), LINE]) == 2


def test_check_no_hash_policy(tmp_path):
    path = tmp_path / "r.json"
    path.write_text("{}")
    assert ju.main(["check", str(path), LINE]) == 0
    assert ju.main(["check", str(path), LINE, "--strict-hash"]) == 2
    path.write_text(json.dumps({"config_hash": "other"}))
    assert ju.main(["check", str(path), LINE, "--trust-existing"]) == 0


def test_stamp_result_matches_jobline():
    argv = ["--method", "recal", "--prep", "p", "--budget", "0.05", "--out", "o", "--tag", "_z"]
    res = ju.stamp_result({}, argv)
    assert res["config_hash"] == ju.config_hash(ju.jobline_args(LINE))
    assert "code_commit" in res


@pytest.mark.parametrize(
    "code,log,expected",
    [
        (0, "", "ok"),
        (137, "", "retry"),
        (-9, "", "retry"),
        (137, "Traceback (most recent call last)", "fatal"),
        (1, "", "fatal"),
        (1, "error: machine not quiet", "retry"),
        (2, "Traceback\nmachine not quiet", "fatal"),
    ],
)
def test_classify_exit(code, log, expected):
    assert ju.classify_exit(code, log) == expected


def test_classify_cli(tmp_path, capsys):
    log = tmp_path / "l.txt"
    log.write_text("Traceback")
    assert ju.main(["classify", "1", str(log)]) == 0
    assert capsys.readouterr().out.strip() == "fatal"
    assert ju.main(["classify", "137", str(tmp_path / "missing.txt")]) == 0
    assert capsys.readouterr().out.strip() == "retry"


def test_job_line_detection():
    rj = load_run_jobs()
    for line in (
        "python online_sim.py --x 1",
        "python -m recal.cells --method recal",
        "python -m recal.sim --prep p",
        "python make.py --out hit_tables/a.npz",
    ):
        assert rj.is_job(line) and ju.is_job_line(line)
    assert not rj.is_job("mkdir -p results/x") and not ju.is_job_line("cp a b")


def test_assign_cores():
    rj = load_run_jobs()
    asg = rj.assign_cores(64, 8, 4, 8)
    assert (asg.workers, asg.first_core, asg.pin, asg.notice) == (8, 8, True, "")
    asg = rj.assign_cores(16, 8, 4, 8)
    assert (asg.workers, asg.first_core) == (4, 0) and asg.notice
    asg = rj.assign_cores(2, 8, 4, 8)
    assert asg.workers == 1 and not asg.pin


def test_pin_line_every_occurrence():
    rj = load_run_jobs()
    line = "a=1 .venv-core/bin/python x.py && .venv-core/bin/python y.py"
    out = rj.pin_line(line, 0, 3)
    assert out.count("taskset -c 0-3 .venv-core/bin/python") == 2
    assert rj.pin_line(out, 4, 7) == out


def test_run_jobs_status_stale_rerun(tmp_path):
    rj = load_run_jobs()
    path = tmp_path / "r.json"
    line = LINE.replace("out/d/recal_t1.json", str(path))
    assert rj.result_status(line, False, False) == (str(path), 1)
    path.write_text(json.dumps({"config_hash": "old"}))
    assert rj.result_status(line, False, False)[1] == 2
    assert rj.result_status(line, False, True)[1] == 0
