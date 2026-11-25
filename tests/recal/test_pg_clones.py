import json
import os
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CLONES_SH = ROOT / "scripts" / "pg_clones.sh"
BASE_SH = ROOT / "scripts" / "pg_update_base.sh"

# Stubs keep their state in files: a dataset is a file under $STUB/ds, a clone's data directory is
# $PG_POOL_MOUNT/<job>/pgdata, a running server is the file "running" (with its port) in that directory.
ZFS = r'''
echo "zfs $*" >> "$STUB/calls.log"
name() { echo "$STUB/ds/$(echo "$1" | tr '/' '_')"; }
case "$1" in
  list) shift; [ -f "$(name "${@: -1}")" ] ;;
  clone) touch "$(name "$3")"; mkdir -p "$PG_POOL_MOUNT/${3#*/}/pgdata" ;;
  destroy) rm -f "$(name "${@: -1}")"; mkdir -p "$STUB/trash"; mv "$PG_POOL_MOUNT/${3#*/}" "$STUB/trash/$RANDOM$RANDOM" ;;
  snapshot) touch "$(name "$2")" ;;
  get) echo 1700000000 ;;
  *) exit 1 ;;
esac
'''
PG_CTL = r'''
echo "pg_ctl $*" >> "$STUB/calls.log"
dir=""; port=""; mode=""
while [ $# -gt 0 ]; do
  case "$1" in
    -D) dir="$2"; shift ;;
    -o) port=$(echo "$2" | sed -n 's/.*-p \([0-9]*\).*/\1/p'); shift ;;
    start|stop|status) mode="$1" ;;
  esac
  shift
done
case "$mode" in
  status) [ -f "$dir/running" ] ;;
  start) [ -z "${STUB_START_FAIL:-}" ] || exit 1; sleep 0.2; echo "${port:-5432}" > "$dir/running" ;;
  stop) rm -f "$dir/running" ;;
esac
'''
PSQL = r'''
[ -z "${STUB_NO_ANSWER:-}" ] || exit 1
port=""
while [ $# -gt 0 ]; do [ "$1" = "-p" ] && port="$2"; shift; done
for f in "$PG_POOL_MOUNT"/*/pgdata/running; do [ -f "$f" ] && [ "$(cat "$f")" = "$port" ] && exit 0; done
exit 1
'''
INITDB = r'''
echo "initdb $*" >> "$STUB/calls.log"
dir=""; while [ $# -gt 0 ]; do [ "$1" = "-D" ] && dir="$2"; shift; done
touch "$dir/PG_VERSION" "$dir/postgresql.conf"
'''
ZPOOL = '[ -z "${STUB_NO_POOL:-}" ]\n'


@pytest.fixture
def env(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("zfs", ZFS), ("pg_ctl", PG_CTL), ("psql", PSQL), ("initdb", INITDB), ("zpool", ZPOOL)):
        path = bin_dir / name
        path.write_text("#!/bin/bash\n" + body)
        path.chmod(0o755)
    (tmp_path / "stub" / "ds").mkdir(parents=True)
    (tmp_path / "mnt").mkdir()
    (tmp_path / "stub" / "ds" / "pgpool_base@built").touch()
    return {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "PG_BIN": str(bin_dir),
            "STUB": str(tmp_path / "stub"), "PG_POOL_MOUNT": str(tmp_path / "mnt"),
            "PG_CLONES": str(tmp_path / "clones"), "PG_CLONE_PORT_RANGE": "57000-57080", "PG_CLONE_WAIT": "3",
            "TMPDIR": str(tmp_path)}


def clones(env, *args):
    return subprocess.run(["bash", str(CLONES_SH), *args], capture_output=True, text=True, env=env)


def calls(env) -> str:
    return (Path(env["STUB"]) / "calls.log").read_text()


def dataset_exists(env, name):
    return (Path(env["STUB"]) / "ds" / f"pgpool_{name}").exists()


def clone_file(env, job, suffix):
    return Path(env["PG_CLONES"]) / f"{job}.{suffix}"


def dsn_port(env, job):
    return int(clone_file(env, job, "dsn").read_text().split("port=")[1].split()[0])


def test_scripts_are_valid_bash():
    for script in (CLONES_SH, BASE_SH):
        assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0


def test_create_clones_the_snapshot_and_writes_the_dsn_and_settings(env):
    job = "spacev-0.99_targeted10_recal_s0"
    result = clones(env, "create", job)
    assert result.returncode == 0, result.stderr
    assert dataset_exists(env, job)
    assert f"zfs clone pgpool/base@built pgpool/{job}" in calls(env)
    port = dsn_port(env, job)
    assert 57000 <= port <= 57080
    record = json.loads(clone_file(env, job, "json").read_text())
    assert record["shared_buffers"] == "256MB" and record["port"] == port and record["snapshot"] == "pgpool/base@built"
    assert "shared_buffers=256MB" in calls(env)
    assert "already runs" in clones(env, "create", job).stdout


def test_create_without_the_snapshot_says_what_to_do(env):
    (Path(env["STUB"]) / "ds" / "pgpool_base@built").unlink()
    result = clones(env, "create", "job1")
    assert result.returncode != 0 and "--zfs-size-gb" in result.stderr


def test_concurrent_creates_get_distinct_ports(env):
    jobs = [f"job{i}" for i in range(8)]
    procs = [subprocess.Popen(["bash", str(CLONES_SH), "create", j], env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True) for j in jobs]
    assert all(p.wait(timeout=60) == 0 for p in procs)
    ports = [dsn_port(env, j) for j in jobs]
    assert len(set(ports)) == len(jobs)


def test_a_server_that_does_not_answer_is_rolled_back(env):
    result = clones({**env, "STUB_NO_ANSWER": "1", "PG_CLONE_WAIT": "1"}, "create", "job1")
    assert result.returncode != 0 and "destroying it" in result.stderr
    assert not dataset_exists(env, "job1")
    assert not clone_file(env, "job1", "dsn").exists() and not clone_file(env, "job1", "port").exists()


def test_destroy_stops_the_server_and_is_safe_when_absent(env):
    clones(env, "create", "job1")
    assert clones(env, "destroy", "job1").returncode == 0
    assert not dataset_exists(env, "job1") and not clone_file(env, "job1", "dsn").exists()
    assert "-m immediate" in calls(env)
    assert clones(env, "destroy", "job1").returncode == 0
    assert clones(env, "destroy", "bad/job").returncode != 0


def test_run_gives_the_job_its_clone_and_destroys_it_after_success_and_failure(env):
    ok = clones(env, "run", "job1", "--", 'cat "$PG_CLONES/job1.dsn"')
    assert ok.returncode == 0 and "host=127.0.0.1 port=57" in ok.stdout
    assert not dataset_exists(env, "job1") and not clone_file(env, "job1", "pid").exists()
    bad = clones(env, "run", "job2", "--", "exit 3")
    assert bad.returncode == 3 and not dataset_exists(env, "job2")
    assert not clone_file(env, "job2", "dsn").exists()


def test_run_destroys_the_clone_when_it_cannot_be_created(env):
    result = clones({**env, "STUB_START_FAIL": "1"}, "run", "job1", "--", "echo should-not-run")
    assert result.returncode != 0 and "should-not-run" not in result.stdout and not dataset_exists(env, "job1")


def test_gc_destroys_clones_of_finished_jobs_and_keeps_running_ones(env):
    for job in ("live", "dead", "unowned"):
        assert clones(env, "create", job).returncode == 0
    finished = subprocess.Popen(["true"])
    finished.wait()
    clone_file(env, "live", "pid").write_text(str(os.getpid()))
    clone_file(env, "dead", "pid").write_text(str(finished.pid))
    old = time.time() - 600
    for job in ("live", "dead", "unowned"):
        os.utime(clone_file(env, job, "port"), (old, old))
    assert clones(env, "gc").returncode == 0
    assert dataset_exists(env, "live") and not dataset_exists(env, "dead") and not dataset_exists(env, "unowned")
    listing = clones(env, "list").stdout
    assert "live" in listing and "running" in listing and "dead" not in listing


def test_gc_leaves_a_clone_that_was_just_created(env):
    clones(env, "create", "fresh")
    clones(env, "gc")
    assert dataset_exists(env, "fresh")


def build(env, extra=None, loader="echo loaded:$Q1_PG_PORT:$Q1_PGDATA > $STUB/loader.out"):
    environment = {**env, "RECORD": str(Path(env["TMPDIR"]) / "record.json"), **(extra or {})}
    return subprocess.run(["bash", str(BASE_SH), "build", "spacev", "--", loader], capture_output=True, text=True,
                          env=environment, cwd=env["TMPDIR"])


def record(env):
    return json.loads((Path(env["TMPDIR"]) / "record.json").read_text())


def test_the_base_build_loads_snapshots_and_records(env):
    (Path(env["STUB"]) / "ds" / "pgpool_base@built").unlink()
    result = build(env)
    assert result.returncode == 0, result.stderr
    assert (Path(env["STUB"]) / "loader.out").read_text().startswith("loaded:54330:")
    log = calls(env)
    assert log.index("pg_ctl") < log.index("zfs snapshot pgpool/base@built") and "-m fast" in log
    assert record(env)["state"] == "built" and record(env)["snapshot"] == "pgpool/base@built"
    settings = record(env)["server_settings"]
    assert "maintenance_work_mem=" in settings and "max_parallel_maintenance_workers=" in settings
    conf = (Path(env["PG_POOL_MOUNT"]) / "base" / "pgdata" / "postgresql.conf").read_text()
    assert "maintenance_work_mem" in conf and "max_parallel_maintenance_workers" in conf


def test_an_existing_snapshot_is_kept_and_not_reloaded(env):
    result = build(env)
    assert result.returncode == 0 and "keeping it" in result.stdout
    assert not (Path(env["STUB"]) / "loader.out").exists()


def test_without_the_pool_the_build_says_so_and_skips(env):
    result = build(env, {"STUB_NO_POOL": "1"})
    assert result.returncode == 0 and "--zfs-size-gb" in result.stdout
    assert record(env)["state"] == "skipped"
    assert not (Path(env["STUB"]) / "loader.out").exists()


def test_a_failed_loader_stops_the_server_and_takes_no_snapshot(env):
    (Path(env["STUB"]) / "ds" / "pgpool_base@built").unlink()
    result = build(env, loader="exit 4")
    assert result.returncode != 0
    assert not dataset_exists(env, "base@built")
    assert "stop" in calls(env) and "zfs snapshot" not in calls(env)


MEDRAG = {"q1_budgets": [0.0001, 0.0003, 0.001], "standin_5pct": 0.0003}


def update_jobs():
    from recal import matrix
    cfg = json.loads((ROOT / "experiments" / "matrix.json").read_text())
    return [j for j in matrix.cell_jobs(cfg, MEDRAG)[0]["q4"] if j.kind == "update"]


def test_update_job_lines_run_between_clone_creation_and_destruction():
    jobs = update_jobs()
    assert jobs
    for job in jobs:
        out = job.cell.rstrip("/")
        name = Path(out).name
        assert f"bash scripts/pg_clones.sh run {name} -- '" in job.line
        assert f"@$PG_CLONES/{name}.dsn" in job.line and "'" in job.line.split("-- '")[1]
        assert job.line.startswith(f"[ -f {out}/") and f"--out {out}'" in job.line
        assert job.line.rstrip().endswith("# recal.cells")


def test_only_update_jobs_use_a_clone():
    from recal import matrix
    cfg = json.loads((ROOT / "experiments" / "matrix.json").read_text())
    for exp, jobs in matrix.cell_jobs(cfg, MEDRAG)[0].items():
        for job in jobs:
            assert ("pg_clones.sh" in job.line) == (job.kind == "update")


def test_update_job_cap_follows_the_setting_then_the_machine(monkeypatch):
    from simulations.approximate_hits import run_jobs
    monkeypatch.delenv("PG_CLONE_MAX_JOBS", raising=False)
    assert run_jobs.update_job_cap(8, 3) == 3
    monkeypatch.setenv("PG_CLONE_MAX_JOBS", "2")
    assert run_jobs.update_job_cap(8) == 2
    monkeypatch.delenv("PG_CLONE_MAX_JOBS")
    monkeypatch.setattr(run_jobs, "meminfo_gb", lambda key: 9.0)
    monkeypatch.setattr(run_jobs, "pool_free_gb", lambda: 1000.0)
    assert run_jobs.update_job_cap(8) == 4
    monkeypatch.setattr(run_jobs, "pool_free_gb", lambda: 25.0)
    assert run_jobs.update_job_cap(8) == 2
    monkeypatch.setattr(run_jobs, "meminfo_gb", lambda key: 0.5)
    assert run_jobs.update_job_cap(8) == 1


def test_run_lines_never_runs_more_update_jobs_than_the_cap(tmp_path, monkeypatch):
    from simulations.approximate_hits import run_jobs
    monkeypatch.setattr(run_jobs, "result_status", lambda *a: (None, 1))
    active = tmp_path / "active"
    active.mkdir()
    counts = tmp_path / "counts"
    script = (f"touch {active}/$$; ls {active} | wc -l >> {counts}; sleep 0.3; rm -f {active}/$$")
    lines = [f"sh -c '{script}' # pg_clones.sh run job{i} recal.cells" for i in range(6)]
    assert run_jobs.run_lines(lines, workers=6, cores_per_job=1, first_core=0, max_update_jobs=2) == []
    seen = [int(x) for x in counts.read_text().split()]
    assert len(seen) == 6 and max(seen) <= 2


def test_job_line_parsers_see_through_the_clone_wrapper():
    from recal import matrix_hash
    from recal.measure import jobs_util
    job = update_jobs()[0]
    plain = jobs_util.unwrap_clone(job.line)
    assert "pg_clones.sh" not in plain and "recal.cells" in plain
    assert jobs_util.jobline_args(job.line) == jobs_util.jobline_args(plain)
    assert matrix_hash.cell_hash(job.line) == matrix_hash.cell_hash(plain)
    assert matrix_hash.out_dir(job.line) == job.cell.rstrip("/")
