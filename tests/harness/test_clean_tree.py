"""Setup, builds and the smoke test must leave `git status --short` empty: every target refuses a dirty tree."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.harness.test_deploy_scripts import DEPLOY, ROOT, _fake_system

IN_REPO_DIRS = [".venv", ".venv-build", ".venv-encode", "build/replay", ".pytest_cache", "harness/__pycache__"]


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout


@pytest.mark.parametrize("path", [".venv/", ".venv-build/", ".venv-encode/", ".venv/bin/python",
                                  ".venv-encode/lib/x.py", "build/replay/ak_replay_standard",
                                  ".pytest_cache/v", "harness/__pycache__/x.pyc"])
def test_directories_created_inside_the_repo_are_ignored(path):
    out = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT)
    assert out.returncode == 0, f"{path} is not ignored"


def test_every_venv_the_scripts_create_inside_the_repo_is_ignored():
    created = set()
    for script in DEPLOY.glob("*.sh"):
        created.update(re.findall(r'-m venv "\$root/([\w.-]+)"', script.read_text()))
    assert {".venv", ".venv-build", ".venv-encode"} <= created
    for name in created:
        assert subprocess.run(["git", "check-ignore", "-q", f"{name}/bin/python"], cwd=ROOT).returncode == 0, name


@pytest.fixture()
def mini_repo(tmp_path):
    repo = tmp_path / "repo"
    shutil.copytree(DEPLOY, repo / "harness" / "deploy", ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("requirements-q1.lock", "requirements-encode.lock", "requirements-ci.lock"):
        (repo / name).write_text("")
    shutil.copy(ROOT / ".gitignore", repo / ".gitignore")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def _run_setup(repo: Path, tmp_path: Path) -> subprocess.CompletedProcess:
    env = _fake_system(tmp_path / "sys", "1.83.0.1ubuntu2", "3.28.3", "13.2.0") \
        if (tmp_path / "sys").mkdir() is None else {}
    env.update(Q1_CHECK_CLEAN_TREE="1", Q1_SETUP_DATASETS="none")
    return subprocess.run(["bash", str(repo / "harness" / "deploy" / "setup_instance.sh")], env=env,
                          capture_output=True, text=True)


def _create_setup_outputs(repo: Path) -> None:
    for name in IN_REPO_DIRS:
        (repo / name).mkdir(parents=True, exist_ok=True)
        (repo / name / "file").write_text("x")


def test_status_is_empty_after_the_directories_setup_and_smoke_create(mini_repo):
    _create_setup_outputs(mini_repo)
    assert _git(mini_repo, "status", "--short") == ""


def test_clean_tree_step_passes_on_a_clean_checkout_and_names_stray_files(mini_repo, tmp_path):
    _create_setup_outputs(mini_repo)
    ok = _run_setup(mini_repo, tmp_path)
    assert ok.returncode == 0, ok.stderr + ok.stdout
    assert re.search(r"clean_tree\s+PASS", ok.stdout)
    (mini_repo / "stray.txt").write_text("x")
    shutil.rmtree(tmp_path / "sys")
    bad = _run_setup(mini_repo, tmp_path)
    assert bad.returncode == 1 and re.search(r"clean_tree\s+FAIL\s+git status --short is not empty", bad.stdout)
    assert "stray.txt" in bad.stdout


def test_the_old_gitignore_would_have_failed_the_check(mini_repo, tmp_path):
    (mini_repo / ".gitignore").write_text("build/\n.venv\n")
    _git(mini_repo, "commit", "-qam", "old ignore")
    _create_setup_outputs(mini_repo)
    out = _run_setup(mini_repo, tmp_path)
    assert out.returncode == 1 and ".venv-encode" in out.stdout and ".venv-build" in out.stdout


def test_smoke_target_ends_with_a_clean_tree_assertion():
    out = subprocess.run(["make", "-n", "q1-smoke", "Q1_ALLOW_DIRTY="], cwd=ROOT, capture_output=True, text=True)
    lines = out.stdout.strip().splitlines()
    assert "git status --porcelain" in lines[-1] or "git status --porcelain" in lines[-2]
    assert any("pytest" in ln for ln in lines[:-1])
