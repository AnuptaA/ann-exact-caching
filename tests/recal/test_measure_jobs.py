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


def stub_script(tmp_path, default):
    path = tmp_path / "stub_job.py"
    path.write_text(
        "import argparse, sys\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r})\n"
        "from recal.measure import jobs_util\n"
        "ap = argparse.ArgumentParser()\n"
        f"ap.add_argument('--beta', type=float, default={default})\n"
        "ap.add_argument('--alpha', type=float, default=0.5)\n"
        "ap.add_argument('--out', default='x')\n"
        "jobs_util.add_hash_flag(ap)\n"
        "args = ap.parse_args()\n"
        "jobs_util.register_args(args)\n"
        "print(jobs_util.stamp_result({})['config_hash'])\n"
    )
    return path


def stub_line(script, result):
    return f"[ -f {result} ] || {sys.executable} {script} --out o --alpha 0.5 > /dev/null"


def test_check_exit_codes(tmp_path):
    path = tmp_path / "r.json"
    line = f"[ -f {path} ] || x.py"
    assert ju.main(["check", str(path), line]) == 1
    path.write_text(json.dumps({"config_hash": "x", "hash_scheme": 2}))
    assert ju.check_result(path, line, hasher=lambda l: "x") == 0
    assert ju.check_result(path, line, hasher=lambda l: "other") == 2
    assert ju.check_result(path, line, hasher=lambda l: None) == 2
    path.write_text('{"config_hash": ')
    assert ju.main(["check", str(path), line]) == 2
    path.write_text("[1]")
    assert ju.main(["check", str(path), line]) == 2


def test_result_without_scheme_two_hash_is_stale_unless_trusted(tmp_path):
    path = tmp_path / "r.json"
    line = f"[ -f {path} ] || x.py"
    path.write_text("{}")
    assert ju.main(["check", str(path), line]) == 2
    path.write_text(json.dumps({"config_hash": "same"}))
    assert ju.check_result(path, line, hasher=lambda l: "same") == 2     # old scheme
    assert ju.main(["check", str(path), line, "--trust-existing"]) == 0


def test_changed_default_makes_existing_result_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(ju, "SCRIPT_MARKERS", ju.SCRIPT_MARKERS + ("stub_job.py",))
    result = tmp_path / "r.json"
    script = stub_script(tmp_path, default=1.0)
    line = stub_line(script, result)
    stored = ju.jobline_hash(line)
    assert stored is not None
    result.write_text(json.dumps({"config_hash": stored, "hash_scheme": 2}))
    assert ju.check_result(result, line) == 0
    stub_script(tmp_path, default=2.0)                                   # a default changes in code
    assert ju.check_result(result, line) == 2


def test_hash_ignores_noise_keys_and_follows_commit():
    a = {"alpha": 0.5, "out": "a", "threads": 4}
    b = {"alpha": 0.5, "out": "b", "threads": 8}
    assert ju.resolved_hash(a, "c1") == ju.resolved_hash(b, "c1")
    assert ju.resolved_hash(a, "c1") != ju.resolved_hash(a, "c2")
    assert ju.resolved_hash(a, "c1") != ju.resolved_hash(dict(a, alpha=0.6), "c1")


def test_stamp_result_scheme_two_after_register():
    import argparse
    ns = argparse.Namespace(alpha=0.5, out="o", print_config_hash=False)
    ju.register_args(ns)
    try:
        res = ju.stamp_result({})
        assert res["hash_scheme"] == 2 and res["config_hash"] == ju.resolved_hash(vars(ns))
        assert "code_commit" in res
    finally:
        ju._registered.clear()


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


def test_aker_filter_line_parsing_and_check():
    import pytest
    from recal.measure import aker_check
    text = "noise\nFILTER faiss_M=4 efConstruction=8 efSearch=16 metric=0\nmore"
    held = aker_check.parse_filter(text)
    assert held == {"faiss_M": 4, "efConstruction": 8, "efSearch": 16, "metric": 0}
    aker_check.check_filter(held)
    unpatched = aker_check.parse_filter("FILTER faiss_M=4 efConstruction=16 efSearch=8 metric=0")
    with pytest.raises(SystemExit):
        aker_check.check_filter(unpatched)
    assert aker_check.parse_filter("nothing here") is None


def test_capture_stderr_collects_descriptor_two_output():
    import os
    from recal.measure import aker_check
    with aker_check.CaptureStderr() as cap:
        os.write(2, b"FILTER faiss_M=4 efConstruction=8 efSearch=16 metric=0\n")
    assert aker_check.parse_filter(cap.text)["efSearch"] == 16


def test_aker_build_defaults_to_the_vendored_library():
    text = (ROOT / "simulations/approximate_hits/aker_real/build.sh").read_text()
    assert "AKER_SRC=${AKER_SRC:-vendored}" in text
    assert "AKER_SRC=${AKER_SRC:-clone}" not in text


def test_filter_check_fails_unless_allowed():
    from recal.measure import aker_check
    paper = "FILTER faiss_M=4 efConstruction=8 efSearch=16 metric=0"
    assert aker_check.verify_filter(paper)["efSearch"] == 16
    swapped = "FILTER faiss_M=4 efConstruction=16 efSearch=8 metric=0"
    with pytest.raises(SystemExit):
        aker_check.verify_filter(swapped)
    with pytest.raises(SystemExit):
        aker_check.verify_filter("no line here")
    assert aker_check.verify_filter("no line here", allow_unverified=True) is None
    assert aker_check.verify_filter(swapped, allow_unverified=True)["efConstruction"] == 16
