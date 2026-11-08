"""apt/dpkg lock wait of the setup scripts: decided by lock ownership, never by process name.

A fake kernel lock table (Q1_PROC_LOCKS), real lock files and real background processes stand in for a
fresh Ubuntu instance. No root and no apt are needed.
"""

import os
import re
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.harness.test_deploy_scripts import DEPLOY, _fake_system, _setup

POLL = "0.2"


def _lock_line(pid: int, inode: int, waiter: bool = False) -> str:
    arrow = "-> " if waiter else ""
    return f"1: {arrow}POSIX  ADVISORY  WRITE {pid} 08:01:{inode} 0 EOF\n"


class Env:
    def __init__(self, tmp_path: Path):
        self.tmp = tmp_path
        self.locks = [tmp_path / "lock-frontend", tmp_path / "lists-lock"]
        for lock in self.locks:
            lock.write_text("")
        self.table = tmp_path / "proc_locks"
        self.table.write_text("")
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        self.procs = []

    def inode(self, index: int = 0) -> int:
        return os.stat(self.locks[index]).st_ino

    def spawn(self, *argv: str) -> subprocess.Popen:
        proc = subprocess.Popen(list(argv), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.procs.append(proc)
        return proc

    def hold(self, proc: subprocess.Popen, index: int = 0) -> None:
        self.table.write_text(_lock_line(proc.pid, self.inode(index)))

    def release(self) -> None:
        self.table.write_text("")

    def shim(self, name: str, body: str) -> None:
        path = self.bin / name
        path.write_text("#!/bin/sh\n" + body + "\n")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)

    def environ(self, **extra: str) -> dict:
        env = {k: v for k, v in os.environ.items() if not k.startswith("Q1_")}
        env.update(PATH=f"{self.bin}:{os.environ['PATH']}", Q1_APT_LOCKS=" ".join(map(str, self.locks)),
                   Q1_PROC_LOCKS=str(self.table), Q1_APT_POLL_SECONDS=POLL, Q1_APT_WAIT_SECONDS="20")
        env.update(extra)
        return env

    def bash(self, snippet: str, **extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", "-c", f". {DEPLOY}/lib.sh; {snippet}"], env=self.environ(**extra),
                              capture_output=True, text=True, timeout=60)

    def popen_bash(self, snippet: str, **extra: str) -> subprocess.Popen:
        return subprocess.Popen(["bash", "-c", f". {DEPLOY}/lib.sh; {snippet}"], env=self.environ(**extra),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def cleanup(self) -> None:
        for proc in self.procs:
            proc.kill()
            proc.wait()


@pytest.fixture()
def env(tmp_path):
    e = Env(tmp_path)
    yield e
    e.cleanup()


def test_idle_daemon_named_like_unattended_upgrades_does_not_block(env):
    daemon = env.bin / "unattended-upgrade-shutdown"
    daemon.symlink_to(shutil.which("sleep"))
    proc = env.spawn(str(daemon), "300")
    time.sleep(0.2)
    assert proc.poll() is None
    start = time.time()
    out = env.bash("wait_for_apt")
    assert out.returncode == 0, out.stderr
    assert time.time() - start < 3


def test_real_holder_blocks_until_the_lock_is_released(env):
    holder = env.spawn("sleep", "300")
    env.hold(holder)
    start = time.time()
    waiting = env.popen_bash("wait_for_apt")
    time.sleep(1.5)
    assert waiting.poll() is None
    env.release()
    out, err = waiting.communicate(timeout=30)
    assert waiting.returncode == 0, err
    assert time.time() - start >= 1.5
    assert f"pid {holder.pid}" in out and "sleep 300" in out
    assert holder.poll() is None


def test_timeout_names_the_holder_and_touches_nothing(env):
    holder = env.spawn("sleep", "301")
    env.hold(holder, index=1)
    out = env.bash("wait_for_apt", Q1_APT_WAIT_SECONDS="1")
    assert out.returncode != 0
    assert f"{env.locks[1]} held by pid {holder.pid}: sleep 301" in out.stderr
    assert holder.poll() is None
    assert all(lock.exists() for lock in env.locks)


def test_unrelated_inode_and_waiters_do_not_block(env):
    holder = env.spawn("sleep", "300")
    env.table.write_text(_lock_line(holder.pid, env.inode() + 99999)
                         + _lock_line(holder.pid, env.inode(), waiter=True))
    start = time.time()
    assert env.bash("wait_for_apt").returncode == 0
    assert time.time() - start < 3


def test_missing_lock_files_are_free(env):
    for lock in env.locks:
        lock.unlink()
    assert env.bash("wait_for_apt").returncode == 0


def test_unreadable_lock_table_warns_and_continues(env):
    out = env.bash("wait_for_apt", Q1_PROC_LOCKS=str(env.tmp / "nope"))
    assert out.returncode == 0 and "cannot read" in out.stdout


def _apt_shims(env: Env) -> Path:
    record = env.tmp / "apt_args"
    env.shim("sudo", 'exec "$@"')
    env.shim("apt-get", f'printf "%s\\n" "$@" > "{record}"')
    return record


def test_apt_get_waits_for_the_lock_and_is_time_bounded(env):
    record = _apt_shims(env)
    holder = env.spawn("sleep", "300")
    env.hold(holder)
    blocked = env.bash("apt_get install foo", Q1_APT_WAIT_SECONDS="1")
    assert blocked.returncode != 0 and not record.exists()
    env.release()
    free = env.bash("apt_get install foo")
    assert free.returncode == 0, free.stderr
    args = record.read_text().split()
    for want in ("-y", "DPkg::Lock::Timeout=900", "Acquire::Retries=3", "Acquire::http::Timeout=60",
                 "Acquire::https::Timeout=60", "install", "foo"):
        assert want in args


def test_cloud_init_is_awaited_by_status(env):
    counter = env.tmp / "calls"
    env.shim("cloud-init", f'n=$(cat "{counter}" 2>/dev/null || echo 0); echo $((n+1)) > "{counter}"\n'
             'if [ "$n" -lt 2 ]; then echo "status: running"; else echo "status: done"; fi')
    out = env.bash("wait_for_cloud_init")
    assert out.returncode == 0 and "cloud-init status: done" in out.stdout
    assert int(counter.read_text()) == 3


def test_cloud_init_that_never_finishes_fails(env):
    env.shim("cloud-init", 'echo "status: running"')
    out = env.bash("wait_for_cloud_init", Q1_CLOUD_INIT_WAIT_SECONDS="1")
    assert out.returncode != 0 and "cloud-init still running" in out.stderr


def test_cloud_init_error_status_does_not_block(env):
    env.shim("cloud-init", 'echo "status: error"; exit 1')
    assert env.bash("wait_for_cloud_init").returncode == 0


@pytest.mark.skipif(shutil.which("cloud-init") is not None, reason="a real cloud-init is installed")
def test_missing_cloud_init_is_skipped(env):
    assert env.bash("wait_for_cloud_init").returncode == 0


def test_require_sudo_fails_fast_without_passwordless_sudo(env):
    env.shim("sudo", "exit 1")
    out = env.bash("require_sudo")
    assert out.returncode != 0 and "passwordless sudo" in out.stderr
    env.shim("sudo", "exit 0")
    assert env.bash("require_sudo").returncode == 0


def test_deploy_scripts_never_decide_by_process_name():
    bad = re.compile(r"\b(pgrep|pidof|pkill|killall)\b|\bps\b[^|#]*\|\s*grep")
    for script in DEPLOY.glob("*.sh"):
        for line in script.read_text().splitlines():
            if not line.strip().startswith("#"):
                assert not bad.search(line), f"{script.name}: {line.strip()}"


def test_dry_run_setup_commands_are_time_bounded_and_non_interactive(tmp_path):
    out = _setup(_fake_system(tmp_path, "1.74.0.3ubuntu7", "3.22.1"))
    text = out.stdout
    assert out.returncode == 0, out.stderr + text
    assert "--connect-timeout 30 --max-time 1800" in text and "--max-time 120" in text
    assert text.count("http.lowSpeedTime=60") == 2
    assert "Acquire::Retries=3" in text and "DEBIAN_FRONTEND=noninteractive" in text


@pytest.mark.skipif(not Path("/proc/locks").exists(), reason="needs the Linux kernel lock table")
def test_real_kernel_lock_is_seen_and_released(tmp_path):
    lock = tmp_path / "dpkg-lock"
    lock.write_text("")
    code = ("import fcntl, sys, time\nf = open(sys.argv[1], 'a')\nfcntl.lockf(f, fcntl.LOCK_EX)\n"
            "print('locked', flush=True)\ntime.sleep(300)\n")
    child = subprocess.Popen([sys.executable, "-c", code, str(lock)], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "locked"
        run = ["bash", "-c", f". {DEPLOY}/lib.sh; wait_for_apt"]
        env = {k: v for k, v in os.environ.items() if not k.startswith("Q1_")}
        env.update(Q1_APT_LOCKS=str(lock), Q1_APT_WAIT_SECONDS="1", Q1_APT_POLL_SECONDS=POLL)
        held = subprocess.run(run, env=env, capture_output=True, text=True, timeout=30)
        assert held.returncode != 0 and f"pid {child.pid}" in held.stderr
        child.kill()
        child.wait()
        assert subprocess.run(run, env=env, capture_output=True, text=True, timeout=30).returncode == 0
    finally:
        child.kill()
        child.wait()
