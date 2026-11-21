"""pg.sh start and stop can be repeated: a second `make q1-setup` must not fail on a cluster that already runs."""

import os
import shutil
import socket
import stat
import subprocess
from pathlib import Path

import pytest

from tests.harness.test_deploy_scripts import DEPLOY

PORT = "54399"


class FakePg:
    """A fake PG_BIN whose pg_ctl and pg_isready follow flag files in a state directory."""

    def __init__(self, tmp_path: Path):
        self.state = tmp_path / "state"
        self.state.mkdir()
        self.bin = tmp_path / "pgbin"
        self.bin.mkdir()
        self.data = tmp_path / "pgdata"
        self.data.mkdir()
        self.log = tmp_path / "pgdata.log"
        self._script("pg_ctl", '''for a in "$@"; do last="$a"; done
echo "$@" >> "$FAKE_PG_STATE/calls"
case "$last" in
  status) [ -f "$FAKE_PG_STATE/running" ] && exit 0; exit 3 ;;
  start) [ -f "$FAKE_PG_STATE/fail_start" ] && exit 1; touch "$FAKE_PG_STATE/running"; exit 0 ;;
  stop) rm -f "$FAKE_PG_STATE/running"; exit 0 ;;
esac''')
        self._script("pg_isready", '[ -f "$FAKE_PG_STATE/not_ready" ] && exit 2; exit 0')

    def _script(self, name: str, body: str) -> None:
        path = self.bin / name
        path.write_text("#!/bin/sh\n" + body + "\n")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)

    def set_running(self, port: str = PORT) -> None:
        (self.state / "running").write_text("")
        (self.data / "postmaster.pid").write_text(f"123\n{self.data}\n1700000000\n{port}\n\n127.0.0.1\n1 1\nready\n")

    def flag(self, name: str) -> None:
        (self.state / name).write_text("")

    def calls(self) -> list[str]:
        path = self.state / "calls"
        return path.read_text().splitlines() if path.exists() else []

    def run(self, command: str, **extra: str) -> subprocess.CompletedProcess:
        env = {**os.environ, "PG_BIN": str(self.bin), "Q1_PGDATA": str(self.data), "Q1_PG_PORT": PORT,
               "Q1_DATA": str(self.data.parent), "FAKE_PG_STATE": str(self.state), **extra}
        return subprocess.run(["bash", str(DEPLOY / "pg.sh"), command], env=env, capture_output=True, text=True,
                              timeout=60)


@pytest.fixture()
def pg(tmp_path):
    return FakePg(tmp_path)


def _starts(pg: FakePg) -> int:
    return sum(1 for call in pg.calls() if call.endswith(" start"))


def test_start_on_a_stopped_cluster_starts_it_with_a_bounded_wait(pg):
    out = pg.run("start")
    assert out.returncode == 0, out.stderr
    assert _starts(pg) == 1 and "-w -t 120" in [c for c in pg.calls() if c.endswith(" start")][0]


def test_start_twice_is_a_no_op_when_the_server_already_runs_on_the_same_port(pg):
    pg.set_running()
    out = pg.run("start")
    assert out.returncode == 0 and f"already running on port {PORT}" in out.stdout and _starts(pg) == 0


def test_start_refuses_a_server_running_on_another_port(pg):
    pg.set_running(port="5432")
    out = pg.run("start")
    assert out.returncode != 0 and "5432" in out.stderr and PORT in out.stderr and "make q1-pg-stop" in out.stderr
    assert _starts(pg) == 0


def test_a_stale_pid_file_does_not_block_the_start(pg):
    (pg.data / "postmaster.pid").write_text(f"999\n{pg.data}\n1700000000\n{PORT}\n")
    assert pg.run("start").returncode == 0 and _starts(pg) == 1


def test_a_failed_start_shows_the_end_of_the_server_log_and_names_the_reason(pg, tmp_path):
    pg.flag("fail_start")
    pg.log.write_text("LOG: starting\nFATAL:  could not create lock file: Address already in use\n")
    err_file = tmp_path / "err"
    out = pg.run("start", Q1_ERR_FILE=str(err_file))
    assert out.returncode != 0 and "FATAL:  could not create lock file" in out.stderr
    assert err_file.read_text().startswith("PostgreSQL did not start: FATAL:  could not create lock file")


def test_a_server_that_does_not_accept_connections_fails_loudly(pg):
    pg.flag("not_ready")
    out = pg.run("start")
    assert out.returncode != 0 and "does not accept connections" in out.stderr


def test_stop_twice_is_fine(pg):
    pg.set_running()
    first = pg.run("stop")
    second = pg.run("stop")
    assert first.returncode == 0 and second.returncode == 0 and "not running" in second.stdout
    assert sum(1 for c in pg.calls() if c.endswith(" stop")) == 1


def _free_port() -> str:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return str(s.getsockname()[1])


REAL_BIN = Path("/opt/homebrew/opt/postgresql@16/bin")


@pytest.mark.skipif(not (REAL_BIN / "pg_ctl").exists(), reason="needs PostgreSQL 16 binaries")
def test_a_real_cluster_can_be_started_and_stopped_twice(tmp_path):
    data = tmp_path / "pgdata"
    subprocess.run([str(REAL_BIN / "initdb"), "-D", str(data), "-U", "postgres", "--auth=trust", "-A", "trust"],
                   check=True, capture_output=True)
    env = {**os.environ, "PG_BIN": str(REAL_BIN), "Q1_PGDATA": str(data), "Q1_PG_PORT": _free_port(),
           "Q1_DATA": str(tmp_path)}
    run = lambda cmd: subprocess.run(["bash", str(DEPLOY / "pg.sh"), cmd], env=env, capture_output=True, text=True,
                                     timeout=120)
    try:
        first, second = run("start"), run("start")
        assert first.returncode == 0, first.stderr
        assert second.returncode == 0, second.stderr
        assert "already running" in second.stdout
    finally:
        stop_one, stop_two = run("stop"), run("stop")
    assert stop_one.returncode == 0 and stop_two.returncode == 0 and "not running" in stop_two.stdout
