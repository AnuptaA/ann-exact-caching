"""Preflight: strict by default; --allow-shortfall turns only disk and RAM shortfalls into warnings."""

import os
import re
import stat
import subprocess
from collections import namedtuple
from pathlib import Path

import pytest

from harness.deploy import preflight
from tests.harness.test_deploy_scripts import ROOT, _fake_system, _setup

GIB = 2 ** 30
Usage = namedtuple("Usage", "total used free")


@pytest.fixture()
def small_machine(tmp_path, monkeypatch):
    """A clean git checkout as the working directory, 18 GiB free disk, 15 GiB RAM, data volume accepted."""
    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (["init", "-q"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "i"]):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    monkeypatch.chdir(repo)
    monkeypatch.setenv("Q1_ALLOW_ROOT_VOLUME", "1")
    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda p: Usage(100 * GIB, 82 * GIB, 18 * GIB))
    monkeypatch.setattr(preflight.os, "sysconf", lambda name: {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 15 * GIB // 4096}[name])
    return tmp_path / "data"


def test_default_is_a_hard_failure_with_the_numbers(small_machine):
    problems = preflight.check(list(preflight.DATASETS), small_machine, require_linux=False)
    text = " ".join(problems)
    assert re.search(r"disk: 18 GiB free in .* need \d+ GiB", text) and re.search(r"RAM: 15 GiB, need \d+ GiB", text)
    assert preflight.main(["--data-dir", str(small_machine), "--allow-non-linux"]) == 1


def test_shortfalls_go_to_the_list_when_one_is_passed_and_other_problems_still_fail(small_machine):
    shortfalls: list[str] = []
    problems = preflight.check(list(preflight.DATASETS), small_machine, require_linux=False, shortfalls=shortfalls)
    assert problems == [] and len(shortfalls) == 2 and "disk: 18 GiB" in shortfalls[0] and "RAM: 15 GiB" in shortfalls[1]
    (Path.cwd() / "stray").write_text("x")                         # dirty tree
    shortfalls = []
    problems = preflight.check(list(preflight.DATASETS), small_machine, require_linux=False, shortfalls=shortfalls)
    assert any("dirty" in p for p in problems) and len(shortfalls) == 2


def test_the_flag_prints_warnings_with_numbers_and_exits_zero(small_machine, capsys):
    code = preflight.main(["--data-dir", str(small_machine), "--allow-non-linux", "--allow-shortfall"])
    out = capsys.readouterr()
    assert code == 0 and out.out.count("PREFLIGHT WARN") == 2
    assert "disk: 18 GiB free" in out.out and "RAM: 15 GiB, need" in out.out and "preflight ok with 2 shortfall" in out.out
    assert "PREFLIGHT FAIL" not in out.err


def test_the_flag_does_not_hide_a_root_volume_or_a_dirty_tree(small_machine, monkeypatch, capsys):
    monkeypatch.delenv("Q1_ALLOW_ROOT_VOLUME")
    monkeypatch.setattr(preflight, "device_of", lambda p: 1)
    code = preflight.main(["--data-dir", str(small_machine), "--allow-non-linux", "--allow-shortfall"])
    assert code == 1 and "root volume" in capsys.readouterr().err


def test_the_environment_alone_never_relaxes_python_preflight(small_machine, monkeypatch):
    monkeypatch.setenv("Q1_ALLOW_SHORTFALL", "1")
    assert preflight.main(["--data-dir", str(small_machine), "--allow-non-linux"]) == 1
    assert not re.search(r"environ[^\n]*Q1_ALLOW_SHORTFALL", (ROOT / "harness" / "deploy" / "preflight.py").read_text())


def test_tiny_and_single_datasets_can_be_sized(small_machine, capsys):
    tiny = preflight.need_bytes("tiny")
    assert tiny["disk"] < 2 * GIB and tiny["ram"] < 2 * GIB
    assert preflight.main(["--datasets", "tiny", "--data-dir", str(small_machine), "--allow-non-linux"]) == 0
    assert preflight.main(["--datasets", "esci", "--data-dir", str(small_machine), "--allow-non-linux"]) == 1
    assert preflight.main(["--datasets", "--data-dir", str(small_machine), "--allow-non-linux"]) == 0
    assert "tiny" not in preflight.DATASETS


def _fake_preflight(tmp_path: Path, output: str, code: int) -> dict:
    script = tmp_path / "fake_preflight.sh"
    argv = tmp_path / "preflight_argv"
    script.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$@\" > {argv}\nprintf '%s\\n' '{output}'\nexit {code}\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    env = _fake_system(tmp_path / "sys", "1.83.0.1ubuntu2", "3.28.3", "13.2.0") if (tmp_path / "sys").mkdir() is None else {}
    env["Q1_PREFLIGHT_CMD"] = str(script)
    env["Q1_ALLOW_NO_GPU"] = "1"
    return env


def _argv(tmp_path: Path) -> list[str]:
    return (tmp_path / "preflight_argv").read_text().split()


WARN = "PREFLIGHT WARN: disk: 18 GiB free in /data/q1, need 281 GiB (ignored: --allow-shortfall)"


def test_setup_passes_the_flag_only_for_the_exact_value_and_reports_the_numbers(tmp_path):
    env = _fake_preflight(tmp_path, WARN, 0)
    env["Q1_ALLOW_SHORTFALL"] = "1"
    out = _setup(env)
    assert out.returncode == 0, out.stderr + out.stdout
    assert "--allow-shortfall" in _argv(tmp_path)
    assert re.search(r"preflight\s+WARN\s+1 shortfall\(s\): disk: 18 GiB free in /data/q1, need 281 GiB", out.stdout)
    assert "mode: DRY RUN" in out.stdout and "!!! DRY-RUN MODE" in out.stderr
    assert "setup OK with" in out.stdout and "dry-run shortfall mode" in out.stdout


@pytest.mark.parametrize("value", [None, "", "0", "true", "yes", " 1"])
def test_other_values_are_a_strict_run_without_the_flag(tmp_path, value):
    env = _fake_preflight(tmp_path, "disk: 18 GiB", 1)
    if value is not None:
        env["Q1_ALLOW_SHORTFALL"] = value
    out = _setup(env)
    assert "--allow-shortfall" not in _argv(tmp_path)
    assert out.returncode == 1 and re.search(r"preflight\s+FAIL", out.stdout) and "DRY-RUN MODE" not in out.stderr


def test_with_the_flag_a_real_preflight_failure_still_fails_setup(tmp_path):
    env = _fake_preflight(tmp_path, "PREFLIGHT FAIL: git tree is dirty", 1)
    env["Q1_ALLOW_SHORTFALL"] = "1"
    out = _setup(env)
    assert out.returncode == 1 and re.search(r"preflight\s+FAIL", out.stdout)


def test_no_gpu_plus_shortfall_flags_let_a_small_cpu_machine_finish_setup(tmp_path):
    env = _fake_preflight(tmp_path, WARN, 0)
    env.update(Q1_ALLOW_SHORTFALL="1", Q1_SETUP_DATASETS="tiny", Q1_GPU_CHECK_CMD="false")
    out = _setup(env)
    assert out.returncode == 0, out.stderr + out.stdout
    assert "tiny" in _argv(tmp_path) and "--allow-shortfall" in _argv(tmp_path)
    assert re.search(r"gpu\s+WARN", out.stdout) and re.search(r"preflight\s+WARN", out.stdout)
    assert "a GPU warning" in out.stdout and "dry-run shortfall mode" in out.stdout
