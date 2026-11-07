"""Preflight and deploy scripts: they must refuse wrong machines and dirty trees."""

import subprocess
from pathlib import Path

import pytest

from harness.deploy import preflight

ROOT = Path(__file__).resolve().parents[2]


def _repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.email=a@b", "-c", "user.name=t",
                    "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    return tmp_path


def test_need_bytes_scale_with_sphere_graph():
    sphere = preflight.need_bytes("sphere")
    assert sphere["ram"] > 41 * 2 ** 30 and sphere["disk"] > 120 * 2 ** 30
    assert preflight.need_bytes("esci")["disk"] < preflight.need_bytes("spacev")["disk"]


def test_preflight_rejects_wrong_architecture_dirty_tree_and_small_disk(tmp_path):
    repo = _repo(tmp_path)
    problems = preflight.check(["sphere"], tmp_path, require_linux=True, repo=repo)
    assert any("Linux x86_64" in p for p in problems) or preflight.platform.machine() == "x86_64"
    (repo / "untracked.txt").write_text("x")
    problems = preflight.check(["esci"], tmp_path, require_linux=False, repo=repo)
    assert any("dirty" in p for p in problems)
    assert any("disk" in p or "RAM" in p for p in preflight.check(["sphere"], tmp_path, False, repo))
    assert preflight.main(["--datasets", "sphere", "--data-dir", str(tmp_path), "--allow-non-linux"]) == 1


def test_scripts_exist_and_refuse_dirty_trees(tmp_path):
    for name in ("build_linux.sh", "deploy.sh"):
        assert (ROOT / "harness" / "deploy" / name).stat().st_mode & 0o111
    repo = _repo(tmp_path / "ck")
    (repo / "dirty").write_text("x")
    out = subprocess.run(["bash", str(ROOT / "harness/deploy/deploy.sh"), "unused-url", "HEAD", str(repo)],
                         capture_output=True, text=True)
    assert out.returncode != 0 and "dirty" in out.stderr
