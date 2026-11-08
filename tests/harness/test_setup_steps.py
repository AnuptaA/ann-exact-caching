"""Setup runs every independent step, ends with the GPU check, and lists each result.

Dry run with fake system tools, as in test_deploy_scripts; a fake GPU check command stands in for torch.
"""

import re
import stat
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tests.harness.test_deploy_scripts import ROOT, _fake_system, _setup

STEPS = ["preconditions", "data_volume", "tools_dir", "first_boot_wait", "system_packages", "cmake", "boost",
         "faiss", "postgres", "pg_cluster", "python_env", "encode_env", "drivers", "pg_start", "preflight",
         "clean_tree", "gpu"]


def _env(tmp_path: Path, gpu_exit: int | None = 0, **extra: str) -> dict:
    env = _fake_system(tmp_path, "1.83.0.1ubuntu2", "3.28.3", "13.2.0")
    if gpu_exit is not None:
        script = tmp_path / "gpu_check.sh"
        marker = tmp_path / "gpu_check_ran"
        script.write_text(f"#!/bin/sh\ntouch {marker}\nexit {gpu_exit}\n")
        script.chmod(script.stat().st_mode | stat.S_IXUSR)
        env["Q1_GPU_CHECK_CMD"] = str(script)
    env.update(extra)
    return env


def _summary(out: subprocess.CompletedProcess) -> dict:
    text = out.stdout.split("setup summary")[1]
    rows = re.findall(r"^  (\w+)\s+(PASS|FAIL|WARN|SKIPPED|N/A|NOT RUN)\b", text, re.M)
    return dict(rows)


def test_steps_run_in_order_with_the_gpu_check_last_and_all_are_listed(tmp_path):
    out = _setup(_env(tmp_path))
    assert out.returncode == 0, out.stderr + out.stdout
    started = re.findall(r"step (\w+): start", out.stdout)
    assert started == STEPS and started.index("drivers") < started.index("gpu") and started[-1] == "gpu"
    summary = _summary(out)
    assert list(summary) == STEPS
    assert summary["gpu"] == "PASS" and summary["drivers"] == "PASS" and "setup done" in out.stdout


@pytest.mark.parametrize("code", [10, 11, 12, 13])
def test_a_gpu_failure_never_prevents_the_builds_and_fails_setup(tmp_path, code):
    env = _env(tmp_path, gpu_exit=code)
    out = _setup(env)
    summary = _summary(out)
    assert out.returncode == 1
    assert summary["gpu"] == "FAIL" and summary["drivers"] == "PASS" and summary["postgres"] == "PASS"
    assert summary["pg_start"] == "PASS" and summary["preflight"] == "PASS"
    assert "build_linux.sh" in out.stdout and "setup FAILED: gpu" in out.stdout
    assert (tmp_path / "gpu_check_ran").exists()
    assert {10: "no NVIDIA driver", 11: "driver too old", 12: "torch missing", 13: "cannot use"}[code] in out.stdout


def test_an_unrelated_failure_skips_only_its_dependents_and_the_gpu_check_still_runs(tmp_path):
    out = _setup(_env(tmp_path, Q1_PGDATA="/data"))
    summary = _summary(out)
    assert out.returncode == 1
    assert summary["pg_cluster"] == "FAIL" and "refusing a path" in out.stdout.split("setup summary")[1]
    assert summary["pg_start"] == "SKIPPED" and summary["preflight"] == "SKIPPED"
    assert summary["drivers"] == "PASS" and summary["python_env"] == "PASS" and summary["gpu"] == "PASS"
    assert "setup FAILED: pg_cluster" in out.stdout


def test_a_failed_toolchain_step_skips_the_drivers_but_not_the_rest(tmp_path):
    env = _fake_system(tmp_path, "1.83.0.1ubuntu2", "3.28.3", gcc="7.5.0")
    out = _setup(env)
    summary = _summary(out)
    assert out.returncode == 1 and "need >= 9" in out.stderr
    assert summary["system_packages"] == "FAIL" and summary["drivers"] == "SKIPPED"
    assert summary["preconditions"] == "PASS" and summary["data_volume"] == "PASS"
    assert "setup FAILED: system_packages" in out.stdout


def test_a_failed_precondition_stops_everything_and_still_prints_the_summary(tmp_path):
    env = _env(tmp_path, PYTHON_BIN="/nonexistent/python")
    out = _setup(env)
    summary = _summary(out)
    assert out.returncode == 1 and "not found" in out.stderr
    assert summary["preconditions"] == "FAIL" and summary["drivers"] == "NOT RUN" and summary["gpu"] == "NOT RUN"


@pytest.mark.parametrize("value", [None, "", "0", "true", "yes", "2", "no", " 1"])
def test_only_the_exact_value_1_turns_a_gpu_failure_into_a_warning(tmp_path, value):
    extra = {} if value is None else {"Q1_ALLOW_NO_GPU": value}
    out = _setup(_env(tmp_path, gpu_exit=10, **extra))
    assert out.returncode == 1 and _summary(out)["gpu"] == "FAIL"
    assert (tmp_path / "gpu_check_ran").exists()
    assert "WARNING" not in out.stderr


def test_the_flag_gives_a_loud_warning_and_exit_0_but_still_runs_the_check(tmp_path):
    out = _setup(_env(tmp_path, gpu_exit=10, Q1_ALLOW_NO_GPU="1"))
    assert out.returncode == 0, out.stderr + out.stdout
    assert _summary(out)["gpu"] == "WARN" and "Q1_ALLOW_NO_GPU=1" in out.stdout
    assert "!!! WARNING: no usable GPU" in out.stderr and "setup OK with a GPU warning" in out.stdout
    assert (tmp_path / "gpu_check_ran").exists()


def test_the_flag_does_not_hide_other_failures(tmp_path):
    out = _setup(_env(tmp_path, gpu_exit=10, Q1_ALLOW_NO_GPU="1", Q1_PGDATA="/data"))
    assert out.returncode == 1 and _summary(out)["gpu"] == "WARN" and _summary(out)["pg_cluster"] == "FAIL"


def test_ci_mode_marks_gpu_and_encoder_steps_not_applicable(tmp_path):
    out = _setup(_env(tmp_path, gpu_exit=10), "--ci")
    summary = _summary(out)
    assert out.returncode == 0 and summary["gpu"] == "N/A" and summary["encode_env"] == "N/A"
    assert not (tmp_path / "gpu_check_ran").exists()


def test_the_flag_is_read_only_by_the_setup_script():
    readers = []
    for path in list((ROOT / "harness").rglob("*.py")) + list((ROOT / "workloads").rglob("*.py")):
        if "Q1_ALLOW_NO_GPU" in path.read_text() and re.search(r"environ|getenv", path.read_text()):
            readers.append(path.name)
    assert readers == [] or readers == ["gpu_check.py"]            # gpu_check.py only mentions it in a hint
    assert "Q1_ALLOW_NO_GPU" not in (ROOT / "Makefile").read_text()
    for path in (ROOT / "harness").rglob("*.py"):
        assert not re.search(r"(environ|getenv)[^\n]*Q1_ALLOW_NO_GPU", path.read_text()), path


def test_gpu_dependent_stages_still_fail_loudly_with_the_flag_set(tmp_path, monkeypatch):
    from harness import data, pipeline
    monkeypatch.setenv("Q1_ALLOW_NO_GPU", "1")
    monkeypatch.setattr(pipeline, "auto_device", lambda: "numpy")
    with pytest.raises(pipeline.StageError, match="cannot see a CUDA device"):
        pipeline.make_context("tiny", tmp_path, device="cuda")
    ctx = pipeline.make_context("sphere", tmp_path, device="numpy")
    ctx.root.mkdir(parents=True)
    data.write_f32bin(ctx.root / "official.f32bin", np.zeros((10, 768), dtype=np.float32))
    with pytest.raises(pipeline.StageError, match="Q1_ALLOW_SLOW"):
        pipeline.check_resources(ctx, "groundtruth")
    for name in ("esci", "sphere"):
        with pytest.raises(pipeline.StageError, match="many hours"):
            pipeline.stage_embed(pipeline.make_context(name, tmp_path, device="numpy"))


def test_setup_datasets_none_reaches_preflight_as_an_empty_list(tmp_path):
    out = _setup(_env(tmp_path, Q1_SETUP_DATASETS="none"))
    line = [ln for ln in out.stdout.splitlines() if "harness preflight" in ln][0]
    assert "--datasets --data-dir" in line
    default = [ln for ln in _setup(_env(tmp_path / "b")).stdout.splitlines() if "harness preflight" in ln] \
        if (tmp_path / "b").mkdir() is None else []
    assert "--datasets spacev sift10m sphere esci --data-dir" in default[0]


def test_preflight_without_datasets_checks_no_disk_or_ram(tmp_path):
    from harness.deploy import preflight
    assert preflight.check([], tmp_path / "data", require_linux=False) == [
        p for p in preflight.check([], tmp_path / "data", require_linux=False) if "dirty" in p or "root volume" in p
    ]
    assert preflight.main(["--datasets", "--data-dir", str(tmp_path), "--allow-non-linux"]) in (0, 1)
