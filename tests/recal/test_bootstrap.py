import subprocess
from pathlib import Path

from scripts import smoke_cells

ROOT = Path(__file__).resolve().parents[2]


def dry(*args):
    return subprocess.run(["bash", str(ROOT / "scripts/ec2_bootstrap.sh"), "--dry-run", *args], capture_output=True,
                          text=True, check=True).stdout


def test_dry_run_lists_every_step_and_changes_nothing():
    out = dry("--no-data")
    for step in ("base", "limits", "venvs", "core", "aker", "links", "smoke"):
        assert f"{step}: PASS" in out
    assert "fs.aio-max-nr=1048576" in out and "setup_instance.sh" in out and "aker_real/build.sh" in out
    assert "datasets/aker" in out and "simulations/approximate_hits/results" in out
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
