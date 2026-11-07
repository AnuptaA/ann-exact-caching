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


def test_gpu_check_reports_missing_cuda(monkeypatch):
    import sys
    import types
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert preflight.check_gpu() == ["no CUDA device visible to torch"]
    fake.cuda.is_available = lambda: True
    assert preflight.check_gpu() == []


def test_postgres_check_reports_unreachable_server():
    problems = preflight.check_postgres("host=127.0.0.1 port=1 user=x connect_timeout=2")
    assert problems and "not reachable" in problems[0]


def test_postgres_check_against_a_real_server():
    import os
    dsn = os.environ.get("PGVECTOR_TEST_DSN")
    if not dsn:
        pytest.skip("needs PostgreSQL + pgvector")
    assert preflight.check_postgres(dsn) == []
    assert any("expected 0.7.0" in p for p in preflight.check_postgres(dsn, expected="0.7.0"))


def test_make_targets_expand_to_the_expected_commands():
    def dry(*args):
        out = subprocess.run(["make", "-n", "Q1_ALLOW_DIRTY=1", *args], cwd=ROOT, capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        return out.stdout
    aker = dry("q1-run-aker-b1")
    assert "--systems aker" in aker and "--settings paper" in aker and "--budgets 1" in aker
    prox = dry("q1-run-proximity-p0.1-b5")
    assert "--systems proximity" in prox and "--settings p0.1" in prox and "--budgets 5" in prox
    both = dry("q1-run", "SYSTEM=proximity", "BUDGET=10")
    assert "--settings" not in both and "--budgets 10" in both          # both percentiles
    gen = dry("q1-generate", "DATASET=esci")
    assert "generate --dataset esci" in gen and "--pg-workers" in gen
    assert "--stages groundtruth --force groundtruth" in dry("q1-stage", "DATASET=esci", "STAGE=groundtruth", "FORCE=1")
    everything = dry("q1-run-all")
    assert everything.count("harness run") == 9 and "--settings p1" in everything and "--budgets 10" in everything
    assert "--list-size 50" in gen and "/results/" in aker                  # results are per commit
    assert "nohup" in dry("q1-bg", "TARGET=q1-generate DATASET=esci")
    guard = subprocess.run(["make", "q1-run", "SYSTEM=aker", "BUDGET=1", "-n"], cwd=ROOT, capture_output=True, text=True)
    assert "git status --porcelain" in guard.stdout
    refused = subprocess.run(["make", "q1-guard"], cwd=ROOT, capture_output=True, text=True,
                             env={**__import__("os").environ})
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    assert (refused.returncode != 0) == dirty
