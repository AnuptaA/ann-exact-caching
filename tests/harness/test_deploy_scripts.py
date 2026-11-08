"""Decision logic of the setup scripts, run in dry-run mode with fake system tools.

Nothing is installed: every state-changing command goes through `run`, which only prints under
Q1_DRYRUN. Fake `dpkg-query`, `cmake` and `gcc` stand in for stock Ubuntu 22.04 and 24.04.
"""

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "harness" / "deploy"


def _shim(path: Path, body: str) -> None:
    path.write_text("#!/bin/sh\n" + body + "\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _fake_system(tmp_path: Path, boost: str, cmake: str, gcc: str = "11.4.0") -> dict:
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    _shim(bin_dir / "dpkg-query", f'echo "{boost}"')
    _shim(bin_dir / "cmake", f'echo "cmake version {cmake}"')
    _shim(bin_dir / "gcc", f'echo "gcc (Ubuntu) {gcc}"')
    _shim(bin_dir / "lsb_release", 'echo jammy')
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CONDA", "Q1_"))}
    env.update(PATH=f"{bin_dir}:/opt/conda/bin:{os.environ['PATH']}", Q1_DRYRUN="1",
               Q1_SETUP_ALLOW_NON_LINUX="1", PYTHON_BIN=sys.executable,
               Q1_DATA=str(tmp_path / "data" / "q1"), Q1_TOOLS=str(tmp_path / "tools" / "q1"))
    return env


def _setup(env: dict, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(DEPLOY / "setup_instance.sh"), *args], env=env,
                          capture_output=True, text=True)


def _bash(snippet: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", f". {DEPLOY}/lib.sh; {snippet}"], capture_output=True, text=True)


def test_jammy_builds_boost_cmake_and_faiss_from_pinned_sources(tmp_path):
    out = _setup(_fake_system(tmp_path, "1.74.0.3ubuntu7", "3.22.1"))
    text = out.stdout
    assert out.returncode == 0, out.stderr + text
    assert "system Boost 1.74.0.3 is older than 1.80" in text and "building Boost 1.84.0" in text
    assert "cc4b893acf645c9d4b698e9a0f08ca8846aa5d6c68275c14c3e7949c24109454" not in text    # verified, not printed
    assert "cmake==3.27.9" in text and ".venv-build" in text
    assert "943d08bdad7946b22f56d040756669ee444dd681" in text                               # FAISS pinned commit
    assert "2627c5ff775ae6d7aef0c430121ccf857842d2f2" in text                               # pgvector pinned commit
    assert "postgresql-16" in text and "pg_dropcluster --stop 16 main" in text
    assert "port = 54320" in text and "wal_level = minimal" in text and "tcp_keepalives_idle = 60" in text
    assert "--require-hashes -r " in text and "requirements-q1.lock" in text and ".venv-encode" in text
    assert "/opt/conda/bin" not in text.split("PATH after removing conda entries:")[1].splitlines()[0]
    assert "/tools/q1/boost/.installed-1.84.0" in text


def test_noble_uses_system_boost_and_cmake_but_still_builds_pinned_faiss(tmp_path):
    out = _setup(_fake_system(tmp_path, "1.83.0.1ubuntu2", "3.28.3", "13.2.0"))
    text = out.stdout
    assert out.returncode == 0, out.stderr + text
    assert "libboost-log-dev" in text and "building Boost" not in text and "./b2" not in text
    assert "cmake==3.27.9" not in text and "cmake 3.28.3 (system) is new enough" in text
    assert "943d08bdad7946b22f56d040756669ee444dd681" in text


def test_ci_mode_skips_data_volume_gpu_and_encoder_venv(tmp_path):
    out = _setup(_fake_system(tmp_path, "1.83.0.1ubuntu2", "3.28.3"), "--ci")
    text = out.stdout
    assert out.returncode == 0, out.stderr + text
    assert "requirements-ci.lock" in text and ".venv-encode" not in text and "chown" not in text.split("tools")[0]


def test_setup_refuses_old_gcc_old_python_and_unknown_arguments(tmp_path):
    old = _setup(_fake_system(tmp_path, "1.83.0.1ubuntu2", "3.28.3", gcc="7.5.0"))
    assert old.returncode != 0 and "need >= 9" in old.stderr
    env = _fake_system(tmp_path / "b", "1.83.0.1ubuntu2", "3.28.3") if (tmp_path / "b").mkdir() is None else {}
    assert _setup(env, "--bogus").returncode != 0
    env["PYTHON_BIN"] = "/nonexistent/python"
    assert "not found" in _setup(env).stderr


def test_version_and_path_helpers():
    assert _bash("version_ge 1.74.0 1.80 && echo yes || echo no").stdout.strip() == "no"
    assert _bash("version_ge 1.84.0 1.80 && echo yes || echo no").stdout.strip() == "yes"
    assert _bash("version_ge 3.23.1 3.23.1 && echo yes || echo no").stdout.strip() == "yes"
    assert _bash("version_ge 3.9 3.10 && echo yes || echo no").stdout.strip() == "no"
    assert _bash("first_version 'cmake version 3.22.1'").stdout.strip() == "3.22.1"
    empty = _bash("set -euo pipefail; v=$(first_version ''); echo \"[$v]\"")
    assert empty.returncode == 0 and empty.stdout.strip() == "[]"
    for bad in ("''", "/", "/usr", "/usr/local", "relative/path", "/a/b", "/opt", "/data", "/x/../y/zzzzzz"):
        out = _bash(f"guard_path {bad}")
        assert out.returncode != 0, bad
    assert _bash("guard_path /data/q1/pgdata").returncode == 0
    assert _bash("guard_path /opt/q1/boost").returncode == 0


def test_safe_rm_rf_never_removes_unguarded_paths(tmp_path):
    victim = tmp_path / "keep" / "me" / "please"
    victim.mkdir(parents=True)
    assert _bash(f"safe_rm_rf {victim}").returncode == 0 and not victim.exists()
    assert _bash("safe_rm_rf ''").returncode != 0
    assert _bash("safe_rm_rf /usr").returncode != 0 and Path("/usr").exists()


def test_scripts_use_strict_mode_and_guard_every_rm_rf():
    for script in DEPLOY.glob("*.sh"):
        text = script.read_text()
        if script.name not in ("lib.sh", "toolchain.sh"):   # sourced libraries
            assert "set -euo pipefail" in text, script.name
        for line in text.splitlines():
            stripped = line.strip()
            if "rm -rf" in stripped and not stripped.startswith("#"):
                assert "--" in stripped and ("${" in stripped or "safe_rm_rf" in stripped or '"$1"' in stripped), \
                    f"{script.name}: unguarded {stripped}"


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_deploy_scripts_are_shellcheck_clean():
    scripts = sorted(str(p) for p in DEPLOY.glob("*.sh"))
    out = subprocess.run(["shellcheck", "-x", "-s", "bash", *scripts], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout
