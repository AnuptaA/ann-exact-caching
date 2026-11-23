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


CAUSES = {10: "no NVIDIA driver", 11: "driver too old", 12: "torch missing", 13: "cannot use"}


@pytest.mark.parametrize("code", [10, 11, 12, 13])
def test_no_usable_gpu_is_a_warning_and_setup_still_succeeds(tmp_path, code):
    out = _setup(_env(tmp_path, gpu_exit=code))
    summary = _summary(out)
    assert out.returncode == 0, out.stderr + out.stdout
    assert summary["gpu"] == "WARN" and summary["drivers"] == "PASS" and summary["postgres"] == "PASS"
    assert summary["pg_start"] == "PASS" and summary["preflight"] == "PASS"
    assert "build_linux.sh" in out.stdout and "setup FAILED" not in out.stdout
    assert "setup OK in CPU mode (no usable GPU)" in out.stdout and "stages run on the CPU" in out.stdout
    assert "!!! No usable GPU: ground truth and text embedding will run on the CPU" in out.stderr
    assert (tmp_path / "gpu_check_ran").exists() and CAUSES[code] in out.stdout


def test_an_unexpected_gpu_check_failure_is_still_a_failure(tmp_path):
    out = _setup(_env(tmp_path, gpu_exit=1))
    assert out.returncode == 1 and _summary(out)["gpu"] == "FAIL" and "unexpectedly" in out.stdout


def test_a_working_gpu_gives_no_cpu_banner(tmp_path):
    out = _setup(_env(tmp_path, gpu_exit=0))
    assert "No usable GPU" not in out.stderr and "CPU mode" not in out.stdout


@pytest.mark.parametrize("value", [None, "", "0", "1", "yes"])
def test_the_old_flag_changes_nothing(tmp_path, value):
    extra = {} if value is None else {"Q1_ALLOW_NO_GPU": value}
    out = _setup(_env(tmp_path, gpu_exit=10, **extra))
    assert out.returncode == 0 and _summary(out)["gpu"] == "WARN"
    assert ("no longer needed" in out.stdout) == bool(value)


def test_other_failures_still_fail_setup_without_a_gpu(tmp_path):
    out = _setup(_env(tmp_path, gpu_exit=10, Q1_PGDATA="/data"))
    assert out.returncode == 1 and _summary(out)["gpu"] == "WARN" and _summary(out)["pg_cluster"] == "FAIL"


def test_ci_mode_marks_gpu_and_encoder_steps_not_applicable(tmp_path):
    out = _setup(_env(tmp_path, gpu_exit=10), "--ci")
    summary = _summary(out)
    assert out.returncode == 0 and summary["gpu"] == "N/A" and summary["encode_env"] == "N/A"
    assert not (tmp_path / "gpu_check_ran").exists()


def test_the_old_flag_is_not_read_by_any_code():
    for path in list((ROOT / "harness").rglob("*.py")) + list((ROOT / "workloads").rglob("*.py")):
        assert "Q1_ALLOW_NO_GPU" not in path.read_text(), path
    assert "Q1_ALLOW_NO_GPU" not in (ROOT / "Makefile").read_text()
    script = (ROOT / "harness/deploy/setup_instance.sh").read_text()
    assert not re.search(r"Q1_ALLOW_NO_GPU:-\}\"\s*(=|!=)", script)       # only mentioned in the ignore note


def test_gpu_dependent_stages_run_on_the_cpu_without_a_gpu_and_explicit_cuda_fails(tmp_path, monkeypatch):
    import sys
    from harness import data, pipeline
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr("harness.deploy.gpu_check.query_smi", lambda: None)
    with pytest.raises(pipeline.StageError, match="cuda requested"):
        pipeline.make_context("tiny", tmp_path, device="cuda")
    ctx = pipeline.make_context("sphere", tmp_path)
    assert ctx.device == "cpu" and ctx.device_info.requested == "auto"
    ctx.root.mkdir(parents=True)
    data.write_f32bin(ctx.root / "official.f32bin", np.zeros((10, 768), dtype=np.float32))
    pipeline.check_resources(ctx, "groundtruth")                 # never refused for being on the CPU


def test_setup_datasets_none_reaches_preflight_as_an_empty_list(tmp_path):
    out = _setup(_env(tmp_path, Q1_SETUP_DATASETS="none"))
    line = [ln for ln in out.stdout.splitlines() if "harness preflight" in ln][0]
    assert "--datasets --data-dir" in line
    default = [ln for ln in _setup(_env(tmp_path / "b")).stdout.splitlines() if "harness preflight" in ln] \
        if (tmp_path / "b").mkdir() is None else []
    assert "--datasets spacev sphere esci --data-dir" in default[0]


def test_preflight_without_datasets_checks_no_disk_or_ram(tmp_path):
    from harness.deploy import preflight
    assert preflight.check([], tmp_path / "data", require_linux=False) == [
        p for p in preflight.check([], tmp_path / "data", require_linux=False) if "dirty" in p or "root volume" in p
    ]
    assert preflight.main(["--datasets", "--data-dir", str(tmp_path), "--allow-non-linux"]) in (0, 1)
