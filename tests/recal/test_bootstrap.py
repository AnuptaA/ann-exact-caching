import os
import re
import shutil
import subprocess
from pathlib import Path

from scripts import smoke_cells

ROOT = Path(__file__).resolve().parents[2]


def dry(*args):
    return subprocess.run(["bash", str(ROOT / "scripts/ec2_bootstrap.sh"), "--dry-run", *args], capture_output=True,
                          text=True, check=True).stdout


def test_dry_run_lists_every_step_and_changes_nothing():
    out = dry("--no-data")
    for step in ("base", "limits", "venvs", "core", "aker", "qvcache", "links", "smoke"):
        assert f"{step}: PASS" in out
    assert "fs.aio-max-nr=1048576" in out and "setup_instance.sh" in out and "aker_real/build.sh" in out
    assert "mkdir -p /data/datasets/aker" in out and "links: PASS" in out
    assert "data fetch skipped" in out and "[dry-run]" in out


def test_only_and_skip_select_steps():
    out = dry("--only", "limits")
    assert "limits: PASS" in out and "venvs: NOT RUN" in out and "base: NOT RUN" in out
    assert "venvs: NOT RUN" in dry("--skip", "venvs")


def test_data_step_calls_the_fetch_script_with_all():
    assert "fetch_data.sh --all" in dry()


def test_script_is_valid_shell():
    subprocess.run(["bash", "-n", str(ROOT / "scripts/ec2_bootstrap.sh")], check=True)


def test_smoke_cell_runs_and_reports_library_state(tmp_path):
    prep = smoke_cells.make_prep(tmp_path / "prep")
    result = smoke_cells.run_cell(prep, tmp_path / "o", ["--no-mini-cache"])
    assert result["within_M"] and result["args"]["no_mini_cache"] and (tmp_path / "o" / "recal_records.npz").exists()
    assert set(smoke_cells.check_libraries()) == {"libakershim.so", "libakershim_prox.so", "diskannpy"}


def test_qvcache_step_pins_the_artifact_commit_and_builds_against_the_ann_interpreter():
    out = dry("--only", "qvcache")
    pins = (ROOT / "harness/deploy/pins.env").read_text()
    assert "QVCACHE_COMMIT=39731c6cfa428a8386441ffbc3c43a6bc00dc3f9" in pins and "qvcache/qvcache-vldb26" in pins
    assert "git" in out and "checkout -q" in out and "-DPython3_EXECUTABLE=" in out and ".venv-ann/bin/python" in out
    assert "qvcache: PASS" in out


SCRIPT = ROOT / "scripts/ec2_bootstrap.sh"


def requirement(*args):
    out = dry("--no-smoke", *args)
    return int(re.search(r"disk requirement: (\d+) GB", out).group(1))


def stub_id(tmp_path, uid):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "id"
    stub.write_text(f"#!/bin/sh\necho {uid}\n")
    stub.chmod(0o755)
    return bin_dir


def run_script(tmp_path, uid, *args, sudo_user=None):
    env = {k: v for k, v in os.environ.items() if k != "SUDO_USER"}
    env.update(PATH=f"{stub_id(tmp_path, uid)}:{env['PATH']}", RECAL_TMP=str(tmp_path / "tmp"))
    if sudo_user:
        env["SUDO_USER"] = sudo_user
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, env=env)


def sourced(tmp_path, body, commit="c1"):
    env = dict(os.environ, RECAL_TMP=str(tmp_path / "tmp"), RECAL_BOOTSTRAP_COMMIT=commit)
    script = f"source {SCRIPT}\n{body}"
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)


def test_not_root_exits_at_once_with_the_sudo_line(tmp_path):
    result = run_script(tmp_path, 1000, "--role", "prepare", "--datasets", "tripclick")
    assert result.returncode != 0
    assert "sudo -E bash scripts/ec2_bootstrap.sh --role prepare --datasets tripclick" in result.stdout
    assert not (tmp_path / "tmp").exists() and "step " not in result.stdout


def test_root_without_a_sudo_user_is_refused(tmp_path):
    result = run_script(tmp_path, 0)
    assert result.returncode != 0 and "SUDO_USER" in result.stdout and not (tmp_path / "tmp").exists()


def test_a_failing_step_stops_the_script(tmp_path):
    marker = tmp_path / "marker"
    body = f"""
STEPS=(one two three)
step_one() {{ echo one >> {marker}; }}
step_two() {{ echo "output of two"; run false; }}
step_three() {{ echo three >> {marker}; }}
run_steps; rc=$?; print_summary; echo "rc=$rc"
"""
    out = sourced(tmp_path, body).stdout
    assert "step two FAILED" in out and "failed command: false" in out and "output of two" in out
    assert "rc=1" in out and "one: PASS" in out and "two: FAIL" in out and "three: NOT RUN" in out
    assert marker.read_text() == "one\n"


def test_an_inner_command_failure_fails_the_step(tmp_path):
    body = """
STEPS=(one two)
step_one() { echo before; run false; echo after; }
step_two() { echo two; }
run_steps; echo "rc=$?"
"""
    out = sourced(tmp_path, body).stdout
    assert "before" in out and "after" not in out and "rc=1" in out and "two: " not in out


def test_disk_requirement_follows_the_selected_datasets():
    assert requirement("--datasets", "tripclick") < 100
    assert requirement() > 393 and requirement("--datasets", "spacev,triviaqa,medrag,tripclick") > 393
    assert requirement("--datasets", "medrag") > 393 > requirement("--datasets", "spacev,tripclick")
    bad = subprocess.run(["bash", str(SCRIPT), "--dry-run", "--datasets", "nope"], capture_output=True, text=True)
    assert bad.returncode == 2 and "unknown dataset" in bad.stderr


def test_data_step_fetches_only_the_selected_datasets():
    assert "fetch_data.sh --tripclick" in dry("--datasets", "tripclick")
    assert "fetch_data.sh --spacev --triviaqa --sphere" in dry("--datasets", "spacev,triviaqa")
    assert "fetch_data.sh --all" in dry("--datasets", "spacev,triviaqa,medrag,tripclick")


def test_harness_sizing_covers_only_datasets_it_knows(tmp_path):
    body = 'DATASETS="tripclick medrag"; harness_datasets; DATASETS="spacev triviaqa"; harness_datasets'
    assert sourced(tmp_path, body).stdout.split("\n")[:2] == ["none", "spacev sphere"]


def interruption_body(extra=""):
    return f"""
ART=$RECAL_TMP/artifact
mkdir -p $RECAL_TMP
STEPS=(one two)
step_one() {{ echo one >> $RECAL_TMP/log; }}
step_two() {{ [ ! -e $ART ] || {{ echo "reused partial output"; return 1; }}; echo partial > $ART; {extra} echo full > $ART; echo two >> $RECAL_TMP/log; }}
clean_two() {{ rm -f $ART; echo cleaned >> $RECAL_TMP/log; }}
run_steps; echo "rc=$?"; print_summary
"""


def test_rerun_after_an_interruption_redoes_only_the_interrupted_step(tmp_path):
    first = sourced(tmp_path, interruption_body("kill -9 $$; exit 137;"))
    assert first.returncode != 0
    log = tmp_path / "tmp" / "log"
    second = sourced(tmp_path, interruption_body()).stdout
    assert "rc=0" in second and "one: SKIPPED (done)" in second and "two: PASS" in second
    assert "reused partial output" not in second
    assert log.read_text().split() == ["one", "cleaned", "two"]
    third = sourced(tmp_path, interruption_body()).stdout
    assert "one: SKIPPED (done)" in third and "two: SKIPPED (done)" in third
    assert log.read_text().split() == ["one", "cleaned", "two"]


def test_a_new_commit_redoes_every_step_and_from_resumes(tmp_path):
    body = """
STEPS=(one two)
step_one() { echo one >> $RECAL_TMP/log; }
step_two() { echo two >> $RECAL_TMP/log; }
run_steps; print_summary
"""
    log = tmp_path / "tmp" / "log"
    sourced(tmp_path, body, commit="c1")
    changed = sourced(tmp_path, body, commit="c2").stdout
    assert "one: PASS" in changed and "two: PASS" in changed
    assert log.read_text().split() == ["one", "two", "one", "two"]
    resumed = sourced(tmp_path, body.replace("run_steps", "FROM=two; run_steps"), commit="c2").stdout
    assert "one: SKIPPED (before --from)" in resumed and "two: PASS" in resumed
    assert log.read_text().split() == ["one", "two", "one", "two", "two"]


def test_clean_refuses_a_system_directory_as_the_build_prefix(tmp_path):
    env = dict(os.environ, TP="/tmp", RECAL_TMP=str(tmp_path / "tmp"))
    result = subprocess.run(["bash", str(SCRIPT), "--dry-run", "--clean"], capture_output=True, text=True, env=env)
    assert result.returncode != 0 and "refusing to remove /tmp" in result.stdout


def test_git_status_is_clean_after_the_links_step_and_the_build_outputs(tmp_path):
    repo, data = tmp_path / "repo", tmp_path / "data"
    tracked = subprocess.run(["git", "ls-files", "datasets", "simulations/approximate_hits/results"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.split()
    for rel in [*tracked, ".gitignore"]:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).touch()
    shutil.copy(ROOT / ".gitignore", repo / ".gitignore")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(repo)]
    subprocess.run([*git[:5], "init", "-q", str(repo)], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "x"], check=True)
    body = f"root={repo}; DATA_DIR={data}; RESULTS_DIR={data}/results; STEPS=(links); run_steps; echo rc=$?"
    assert "rc=0" in sourced(tmp_path, body).stdout
    outputs = (".venv-core/bin/python", ".venv-ann/x", ".venv/x", ".venv-encode/x", ".env.local",
               "recal/core/_recal_core.cpython-311-x86_64-linux-gnu.so",
               "simulations/approximate_hits/cachecore/_cachecore.cpython-311-x86_64-linux-gnu.so",
               "datasets/.drive_staging/part", "build/replay/driver", ".pytest_cache/x", "recal/__pycache__/x.pyc")
    for rel in outputs:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).touch()
    assert (repo / "datasets/aker").is_symlink()
    status = subprocess.run([*git, "status", "--porcelain"], capture_output=True, text=True).stdout
    assert status == ""
