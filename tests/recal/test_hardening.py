import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from recal import artifacts, prepare, store as store_mod
from scripts import data_fetch, pg_load_base
from scripts import prepare as prepare_script

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/ec2_bootstrap.sh"
CONFIG = json.loads((ROOT / "experiments/matrix.tripclick-legacy.json").read_text())
MODULE = "qvcache.cpython-311-x86_64-linux-gnu.so"


def stub(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n" + body + "\n")
    path.chmod(0o755)
    return path


def bash(tmp_path: Path, body: str, **env: str) -> subprocess.CompletedProcess:
    full = {**os.environ, "TP": str(tmp_path / "tp"), "RECAL_TMP": str(tmp_path / "tmp"), **env}
    return subprocess.run(["bash", "-c", f"source {SCRIPT}\nroot={tmp_path / 'repo'}\n{body}"], capture_output=True,
                          text=True, env=full)


@pytest.fixture
def qv(tmp_path):
    """A fake artifact tree and a .venv-ann python that knows qvcache.pth."""
    site = tmp_path / "site"
    site.mkdir()
    stub(tmp_path / "repo/.venv-ann/bin/python",
         f'case "$2" in *getsitepackages*) echo {site};; *"import qvcache"*) [ -f {site}/qvcache.pth ] && '
         f'echo "$(cat {site}/qvcache.pth)/{MODULE}";; esac')
    return tmp_path


# --- QVCache module: found where the build puts it, never assumed ------------------------------------------

def test_the_module_is_looked_up_in_the_artifact_python_directory_first(qv):
    (qv / "tp/qvcache/python").mkdir(parents=True)
    (qv / f"tp/qvcache/python/{MODULE}").touch()
    (qv / f"tp/qvcache/build/{MODULE}").parent.mkdir(parents=True)
    (qv / f"tp/qvcache/build/{MODULE}").touch()
    assert bash(qv, "qvcache_module").stdout.strip() == f"{qv}/tp/qvcache/python/{MODULE}"


def test_the_module_is_found_anywhere_under_the_artifact(qv):
    (qv / f"tp/qvcache/build/deep/lib/{MODULE}").parent.mkdir(parents=True)
    (qv / f"tp/qvcache/build/deep/lib/{MODULE}").touch()
    assert bash(qv, "qvcache_module").stdout.strip().endswith(f"build/deep/lib/{MODULE}")


def test_a_missing_module_fails_with_a_clear_message(qv):
    (qv / "tp/qvcache").mkdir(parents=True)
    result = bash(qv, "qvcache_module; echo rc=$?; register_qvcache; echo rc=$?")
    assert "rc=1" in result.stdout.split("\n")[0]
    assert "no qvcache*.so found anywhere under" in result.stdout and result.stdout.strip().endswith("rc=1")


def test_registration_writes_the_pth_and_verifies_the_import(qv):
    (qv / "tp/qvcache/python").mkdir(parents=True)
    (qv / f"tp/qvcache/python/{MODULE}").touch()
    result = bash(qv, "register_qvcache; echo rc=$?")
    assert result.stdout.strip().endswith("rc=0")
    assert (qv / "site/qvcache.pth").read_text().strip() == f"{qv}/tp/qvcache/python"


def test_a_failing_import_after_registration_is_an_error(qv):
    (qv / "tp/qvcache/python").mkdir(parents=True)
    (qv / f"tp/qvcache/python/{MODULE}").touch()
    stub(qv / "repo/.venv-ann/bin/python", f'case "$2" in *getsitepackages*) echo {qv}/site;; *) exit 1;; esac')
    result = bash(qv, "register_qvcache; echo rc=$?")
    assert "import qvcache fails" in result.stdout and result.stdout.strip().endswith("rc=1")


def test_a_rerun_with_a_complete_build_only_registers(qv):
    (qv / "tp/qvcache/python").mkdir(parents=True)
    (qv / f"tp/qvcache/python/{MODULE}").touch()
    body = """
DRY=0
apt_run() { echo "apt: $*"; }
preseed_mkl() { echo preseed; }
as_user() { if [ "$1" = tee ]; then shift; tee "$@"; else echo "as_user: $*"; fi; }
step_qvcache; echo rc=$?
"""
    out = bash(qv, body, PATH=f"{qv}/repo/.venv-ann/bin:{os.environ['PATH']}").stdout
    assert "build already complete" in out and out.strip().endswith("rc=0")
    assert "cmake" not in out and "git" not in out


# --- the fetch -------------------------------------------------------------------------------------------

def run_fetch_sh(tmp_path, env=None):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True, exist_ok=True)
    return subprocess.run(["sh", str(ROOT / "scripts/fetch_data.sh"), "--list"], capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin", "RECAL_REPO": str(repo), **(env or {})})


def test_the_fetch_uses_the_core_venv_when_there_is_no_fetch_venv(tmp_path):
    stub(tmp_path / "repo/.venv-core/bin/python", 'echo "core python: $@"')
    assert "core python: -m scripts.data_fetch --list" in run_fetch_sh(tmp_path).stdout


def test_the_fetch_honors_fetch_py_and_names_the_problem_when_none_exists(tmp_path):
    stub(tmp_path / "mine/python", 'echo "mine: $@"')
    assert "mine:" in run_fetch_sh(tmp_path, {"FETCH_PY": str(tmp_path / "mine/python")}).stdout
    result = run_fetch_sh(tmp_path)
    assert result.returncode != 0 and "no Python with gdown found" in result.stderr


def test_quota_errors_stop_at_once_with_a_clear_message():
    calls = []

    def quota():
        calls.append(1)
        raise RuntimeError("Too many users have viewed or downloaded this file recently")
    with pytest.raises(SystemExit) as exc:
        data_fetch.with_retries(quota, "downloading x.npy", pause=0)
    assert len(calls) == 1 and "quota or access limit" in str(exc.value) and "downloading x.npy" in str(exc.value)


def test_transient_errors_are_retried_and_then_reported():
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("reset")
        return "ok"
    assert data_fetch.with_retries(flaky, "x", pause=0) == "ok" and len(attempts) == 3
    with pytest.raises(SystemExit) as exc:
        data_fetch.with_retries(lambda: (_ for _ in ()).throw(ConnectionError("down")), "x", attempts=2, pause=0)
    assert "failed after 2 attempts" in str(exc.value)


def fake_gdown(size_written: int):
    def download(id, output, quiet, resume):
        Path(output).write_bytes(b"x" * size_written)
    return types.SimpleNamespace(download=download)


def test_a_download_of_the_wrong_size_is_not_placed(tmp_path, monkeypatch):
    item = data_fetch.Item("drive", "f.npy", "datasets/aker/f.npy", None, "tripclick")
    action = {"item": item, "entry": {"id": "i", "size": 10}, "size": 10}
    monkeypatch.setattr(data_fetch, "import_gdown", lambda: fake_gdown(4))
    with pytest.raises(SystemExit) as exc:
        data_fetch.download_item(action, tmp_path, tmp_path / "staging")
    assert "got 4 bytes, expected 10" in str(exc.value) and not (tmp_path / "datasets/aker/f.npy").exists()
    monkeypatch.setattr(data_fetch, "import_gdown", lambda: fake_gdown(10))
    data_fetch.download_item(action, tmp_path, tmp_path / "staging")
    assert (tmp_path / "datasets/aker/f.npy").stat().st_size == 10


def test_a_missing_gdown_is_named(monkeypatch):
    monkeypatch.setitem(sys.modules, "gdown", None)
    with pytest.raises(SystemExit) as exc:
        data_fetch.import_gdown()
    assert "gdown is not installed" in str(exc.value)


def test_the_core_requirements_cover_the_scripts_that_run_in_it():
    text = (ROOT / "envs/core.txt").read_text()
    for package in ("gdown", "boto3", "psycopg", "scikit-learn", "faiss-cpu", "diskannpy", "scipy", "numpy"):
        assert package in text, package


# --- the store ---------------------------------------------------------------------------------------------

def test_the_s3_client_gets_the_region_and_retries(monkeypatch):
    seen = {}
    boto3 = types.SimpleNamespace(client=lambda name, **kw: seen.update(name=name, **kw))
    config = types.SimpleNamespace(Config=lambda **kw: kw)
    monkeypatch.setitem(sys.modules, "boto3", boto3)
    monkeypatch.setitem(sys.modules, "botocore", types.SimpleNamespace(config=config))
    monkeypatch.setitem(sys.modules, "botocore.config", config)
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    store_mod.S3Backend("bucket", "pilot")
    assert seen["region_name"] == "eu-west-1" and seen["config"]["retries"]["max_attempts"] == 8
    monkeypatch.delenv("AWS_REGION")
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.setattr(store_mod, "instance_region", lambda: "us-east-2")
    store_mod.S3Backend("bucket")
    assert seen["region_name"] == "us-east-2"


def test_the_instance_region_is_none_off_ec2(monkeypatch):
    import urllib.request

    def refuse(*a, **k):
        raise OSError("no route")
    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    assert store_mod.instance_region(timeout=0.1) is None


# --- artifacts and the prepare stages ------------------------------------------------------------------------

def test_an_artifact_path_that_was_not_written_is_an_error_not_an_empty_artifact(tmp_path):
    with pytest.raises(FileNotFoundError, match="diskann_answers.*does not exist"):
        artifacts.describe("diskann_answers", ["results/backends/x/diskann.npz"], "cmd", "c", tmp_path)
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        artifacts.describe("prep", ["empty"], "cmd", "c", tmp_path)


def all_commands(dataset):
    return [c for st in prepare.stages(CONFIG, dataset) for c in st.commands]


@pytest.mark.parametrize("dataset", ["spacev", "triviaqa", "medrag", "tripclick"])
def test_every_flag_of_every_prepare_command_exists_in_its_script(dataset):
    for command in all_commands(dataset):
        match = re.search(r"(?:python|PY) (-m )?([\w./]+)(.*)", command)
        if not match or match.group(2) in ("-c",):
            continue
        path = Path(match.group(2).replace(".", "/") + ".py") if match.group(1) else Path(match.group(2))
        source = (ROOT / path).read_text()
        known = set(re.findall(r"add_argument\(\s*[\"'](--[\w-]+)", source))
        used = re.findall(r"(?<=\s)(--[\w-]+)", match.group(3))
        assert [f for f in used if f not in known] == [], (path, command[:120])


def test_the_prep_depth_is_the_file_name_every_consumer_reads():
    commands = [c for c in all_commands("tripclick") if "online_prepare.py" in c]
    assert commands and all("--R 500" in c for c in commands)
    for consumer in ("recal/sim.py", "simulations/approximate_hits/eval_real_backend.py"):
        assert "top500.npy" in (ROOT / consumer).read_text()


def test_diskann_outputs_go_where_the_matrix_and_the_manifest_expect_them():
    w = CONFIG["workloads"]["tripclick"]
    commands = [c for c in all_commands("tripclick") if "eval_real_backend.py" in c]
    out_dir = str(Path(w["diskann_file"]).parent)
    assert commands and all(f"--out-dir {out_dir}" in c for c in commands)
    assert all("--build-gb $(awk" in c and "--threads $(getconf _NPROCESSORS_ONLN)" in c for c in commands)
    eval_source = (ROOT / "simulations/approximate_hits/eval_real_backend.py").read_text()
    assert "args.out_dir" in eval_source


def test_tripclick_loads_postgresql_from_its_base_file_and_starts_the_server_first():
    stage = next(s for s in prepare.stages(CONFIG, "tripclick") if s.name == "pgvector_db")
    assert stage.commands[0] == "make q1-pg-start"
    assert "scripts.pg_load_base --dataset tripclick --base datasets/aker/tripclick-ip-docs.npy" in stage.commands[1]
    spacev = next(s for s in prepare.stages(CONFIG, "spacev") if s.name == "pgvector_db")
    assert spacev.commands[0] == "make q1-pg-start" and "q1-stage DATASET=spacev STAGE=pg_load" in spacev.commands[1]
    from harness import datasets as harness_datasets
    assert set(prepare.HARNESS_DATASETS) <= set(harness_datasets.SPECS) and "tripclick" not in harness_datasets.SPECS


# --- the PostgreSQL loader -------------------------------------------------------------------------------------

def test_tripclick_fits_a_64_gb_machine_and_medrag_does_not():
    gib = 2 ** 30
    assert pg_load_base.resource_problems(1_523_878, 769, "spacev", 64 * gib, 300 * gib) == []
    problems = pg_load_base.resource_problems(23_888_832, 769, "spacev", 64 * gib, 100 * gib)
    assert any("GiB of memory" in p for p in problems) and any("GiB free" in p for p in problems)


def test_the_loader_names_a_missing_base_file_and_picks_index_parameters(tmp_path):
    with pytest.raises(SystemExit, match="does not exist"):
        pg_load_base.load_base_file(tmp_path / "nope.npy")
    np.save(tmp_path / "v.npy", np.zeros((3, 2), dtype=np.float32))
    assert pg_load_base.load_base_file(tmp_path / "v.npy").shape == (3, 2)
    assert pg_load_base.build_key("tripclick") == "spacev" and pg_load_base.build_key("sphere") == "sphere"
    assert 1 <= pg_load_base.default_workers() <= 15


# --- idempotence and --check of a prepare ---------------------------------------------------------------------

def tiny_world(tmp_path, monkeypatch):
    cfg = prepare.tiny_config()
    monkeypatch.chdir(tmp_path)
    return cfg


def test_a_finished_stage_is_skipped_until_its_commands_or_outputs_change(tmp_path, monkeypatch):
    out = tmp_path / "out"
    stage = prepare.Stage("diskann_index", "diskann_index", ["echo run"], ["out"])
    cfg = {"prepare": {"tiny": {"base": "base.npy"}}}
    assert not prepare_script.stage_done(tmp_path, "tiny", stage, cfg)
    out.mkdir()
    prepare_script.mark_done(tmp_path, "tiny", stage)
    assert prepare_script.stage_done(tmp_path, "tiny", stage, cfg)
    assert not prepare_script.stage_done(tmp_path, "tiny", prepare.Stage("diskann_index", "x", ["echo other"], ["out"]), cfg)
    out.rmdir()
    assert not prepare_script.stage_done(tmp_path, "tiny", stage, cfg)
    assert not prepare_script.stage_done(tmp_path, "tiny", prepare.Stage("fetch", None, ["x"], []), cfg)


def test_prepare_runs_only_the_incomplete_stages_on_a_rerun(tmp_path, monkeypatch):
    ran = []
    stages = [prepare.Stage("a", "prep", ["make a"], ["a_out"]), prepare.Stage("b", None, ["make b"], [])]
    monkeypatch.setattr(prepare, "stages", lambda cfg, dataset: stages)
    monkeypatch.setattr(prepare_script, "run_commands", lambda cmds: ran.extend(cmds) or (tmp_path / "a_out").write_text("x"))
    monkeypatch.setattr(prepare_script, "check_free_disk", lambda *a: None)
    cfg = {"prepare": {"d": {}}, "workloads": {}}
    for _ in range(2):
        prepare_script.prepare_dataset(cfg, "d", None, root=tmp_path)
    assert ran == ["make a", "make b", "make b"]
    prepare_script.prepare_dataset(cfg, "d", None, root=tmp_path, force=True)
    assert ran[-2:] == ["make a", "make b"]


def test_prepare_check_passes_on_an_intact_prepare_and_names_a_damaged_file(tmp_path, monkeypatch):
    cfg = prepare.tiny_config()
    monkeypatch.chdir(tmp_path)
    manifest = prepare_script.prepare_dataset(cfg, "tiny", None, root=tmp_path)
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests/tiny.json").write_text(json.dumps(manifest))
    assert all(ok for ok, _ in prepare_script.check_dataset(cfg, "tiny", tmp_path / "manifests", tmp_path))
    victim = tmp_path / manifest["artifacts"]["diskann_answers"]["files"][0]["path"]
    victim.write_bytes(b"damaged")
    results = prepare_script.check_dataset(cfg, "tiny", tmp_path / "manifests", tmp_path)
    assert [ok for ok, _ in results] == [True, False]
    assert "differs from the manifest" in results[1][1] and "make prepare DATASET=tiny" in results[1][1]
    victim.unlink()
    assert "is missing" in prepare_script.check_dataset(cfg, "tiny", tmp_path / "manifests", tmp_path)[1][1]
    assert not prepare_script.check_dataset(cfg, "tiny", tmp_path / "none", tmp_path)[0][0]


def test_the_disk_need_of_prepare_matches_the_bootstrap():
    out = subprocess.run(["bash", str(SCRIPT), "--dry-run", "--no-smoke", "--datasets", ",".join(prepare.DISK_NEED_GB)],
                         capture_output=True, text=True).stdout
    for dataset, gb in prepare.DISK_NEED_GB.items():
        assert f"disk need {dataset}: {gb} GB" in out


# --- bootstrap --check ------------------------------------------------------------------------------------------

def test_bootstrap_check_prints_one_line_per_check_with_a_fix_for_each_failure():
    result = subprocess.run(["bash", str(SCRIPT), "--check", "--datasets", "tripclick"], capture_output=True, text=True)
    lines = [x for x in result.stdout.splitlines() if x.startswith(("PASS", "FAIL"))]
    assert len(lines) >= 17
    assert all("->" in x for x in lines if x.startswith("FAIL"))
    assert (result.returncode == 1) == any(x.startswith("FAIL") for x in lines)
    names = " ".join(lines)
    for needle in ("cmake", ".venv-core imports", ".venv-ann imports", "qvcache", "Aker shims", "FILTER", "aio-max-nr",
                   "PostgreSQL answers on port", "store", "Drive variables", "disk:"):
        assert needle in names, needle


def test_the_smoke_step_knows_where_the_aker_libraries_are(tmp_path):
    out = bash(tmp_path, "aker_ld_library_path", LD_LIBRARY_PATH="/opt/x").stdout
    assert f"{tmp_path}/tp/aker_vendored/build/lib" in out and "faiss-install/lib64" in out and out.endswith(":/opt/x")


# --- the links, store and data steps run for real in a fake tree ----------------------------------------------

def test_links_store_and_data_steps_run_in_a_fake_tree(tmp_path):
    repo = tmp_path / "repo"
    (repo / "datasets/medrag").mkdir(parents=True)
    (repo / "simulations/approximate_hits/results").mkdir(parents=True)    # tracked in the repository
    stub(repo / ".venv-core/bin/python", 'echo "core python: $@"')
    stub(repo / ".venv-core/bin/pip", 'echo "pip: $@"')
    (repo / "scripts").mkdir()
    (repo / "scripts/fetch_data.sh").write_text((ROOT / "scripts/fetch_data.sh").read_text())
    body = f"""
DRY=0; DATA=1; DATA_DIR={tmp_path}/data; RESULTS_DIR={tmp_path}/data/results; DATASETS=tripclick
STEPS=(links store data)
fetch_flags() {{ echo --list; }}
run_steps; echo "rc=$?"; print_summary
"""
    out = bash(tmp_path, body, RECAL_STORE=str(tmp_path / "store")).stdout
    assert "rc=0" in out and "links: PASS" in out and "store: PASS" in out and "data: PASS" in out
    assert "core python: -m scripts.data_fetch --list" in out
    assert (repo / "datasets/aker").is_symlink() and (repo / "results").is_symlink()
    assert (tmp_path / "data/datasets/tripclick").is_dir() and (tmp_path / "data/results").is_dir()
    again = bash(tmp_path, body, RECAL_STORE=str(tmp_path / "store")).stdout
    assert "rc=0" in again and "links: SKIPPED (done)" in again


def test_the_tripclick_verifier_names_a_missing_file(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "scripts/verify_data.py"), "tripclick", "--dir", str(tmp_path)],
                            capture_output=True, text=True)
    assert result.returncode != 0 and "tripclick-ip-docs.npy" in result.stdout + result.stderr
    assert "Traceback" not in result.stderr


# --- the environments cover what the entry points import --------------------------------------------------------

IMPORT_TO_REQUIREMENT = {"sklearn": "scikit-learn", "faiss": "faiss-cpu", "psycopg": "psycopg", "yaml": "pyyaml"}


def module_level_third_party(entry: str) -> set:
    """Third-party packages imported at module level by an entry point and the repository modules it imports."""
    import ast
    std = set(sys.stdlib_module_names)

    def local(name):
        base = ROOT / name.replace(".", "/")
        for candidate in (base.with_suffix(".py"), base / "__init__.py"):
            if candidate.exists():
                return candidate

    seen, found, stack = set(), set(), [ROOT / entry]
    while stack:
        path = stack.pop()
        if path in seen:
            continue
        seen.add(path)
        for node in ast.parse(path.read_text()).body:
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            for name in names:
                top = name.split(".")[0]
                if top in std:
                    continue
                if local(name) is not None:
                    stack.append(local(name))
                elif not (ROOT / top).exists():
                    found.add(top)
    return found


def requirement_names(path: str) -> str:
    return (ROOT / path).read_text().lower()


@pytest.mark.parametrize("entry", ["simulations/approximate_hits/eval_real_backend.py",
                                   "simulations/approximate_hits/run_e2e_oracle_trace.py"])
def test_ann_requirements_cover_the_diskann_build_imports(entry):
    text = requirement_names("envs/ann.txt")
    for package in module_level_third_party(entry) | {"diskannpy", "faiss"}:
        assert IMPORT_TO_REQUIREMENT.get(package, package).lower() in text, (entry, package)


@pytest.mark.parametrize("entry", ["scripts/prepare.py", "simulations/approximate_hits/online_prepare.py",
                                   "scripts/precompute_answers.py", "scripts/convert_streams.py",
                                   "scripts/make_update_schedule.py", "scripts/pg_load_base.py", "scripts/data_fetch.py",
                                   "scripts/store.py", "scripts/smoke_cells.py", "recal/sim.py"])
def test_core_requirements_cover_the_prepare_imports(entry):
    text = requirement_names("envs/core.txt")
    for package in module_level_third_party(entry) - {"boto3"}:     # boto3 is imported lazily and is listed anyway
        assert IMPORT_TO_REQUIREMENT.get(package, package).lower() in text, (entry, package)
