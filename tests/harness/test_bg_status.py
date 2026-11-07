"""q1-bg survives its launching session; q1-status reads progress from files."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from harness import status

REPO = Path(__file__).resolve().parents[2]


def dead_pid() -> int:
    return int(subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True,
                              text=True).stdout)


def test_q1_bg_keeps_running_after_the_launching_session_is_hung_up(tmp_path):
    fake = tmp_path / "fake_make.sh"
    fake.write_text('#!/bin/sh\necho "args: $@"\necho "makeflags: ${MAKEFLAGS-unset}"\n'
                    'trap "" HUP\nsleep 8\necho finished\n')
    fake.chmod(0o755)
    data = tmp_path / "data"
    launcher = subprocess.Popen(
        ["make", "-C", str(REPO), "-j2", "q1-bg", f"MAKE={fake}", "TARGET=q1-generate DATASET=tiny",
         f"Q1_DATA={data}"], env={**os.environ, "MAKEFLAGS": ""}, start_new_session=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = launcher.communicate(timeout=30)[0]
    assert "started pid" in out and "make q1-status" in out, out
    pid_files = [p for p in (data / "logs").glob("*.pid") if p.name != "last.pid"]
    assert len(pid_files) == 1 and (data / "logs" / "last.pid").read_text() == pid_files[0].read_text()
    pid = int(pid_files[0].read_text())
    try:
        os.killpg(launcher.pid, signal.SIGHUP)       # the session the SSH connection would have owned
        time.sleep(1.0)
        os.kill(pid, 0)                                # still alive
        log = pid_files[0].with_suffix(".log").read_text()
        assert "args: q1-generate DATASET=tiny" in log and "makeflags: unset" in log
        run = status.background_runs(data)[0]
        assert run["running"] and run["pid"] == pid
    finally:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                break


def test_q1_bg_requires_a_target(tmp_path):
    out = subprocess.run(["make", "-C", str(REPO), "q1-bg", f"Q1_DATA={tmp_path}"], capture_output=True, text=True)
    assert out.returncode != 0 and "set TARGET" in out.stdout + out.stderr


def test_launcher_prefers_setsid_when_present():
    text = (REPO / "Makefile").read_text()
    assert "command -v setsid" in text and "setsid nohup" in text and "< /dev/null" in text


def test_status_lists_runs_stages_and_failures(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "a-1.pid").write_text(str(os.getpid()))
    (logs / "a-1.log").write_text("line one\nline two\n")
    (logs / "b-1.pid").write_text(str(dead_pid()))
    (logs / "b-1.log").write_text("old\n")
    done = tmp_path / "tiny" / ".done"
    done.mkdir(parents=True)
    (done / "fetch.json").write_text(json.dumps({"seconds": 12.0}))
    (done / "embed.json").write_text(json.dumps({"seconds": 1.0}))
    failed = tmp_path / "tiny" / ".failed"
    failed.mkdir()
    (failed / "to_f32.json").write_text(json.dumps({"error": "RuntimeError('boom')"}))
    text = status.format_status(tmp_path)
    assert "a-1  pid" in text and "RUNNING" in text and "not running" in text
    assert "line two" in text
    assert "tiny: done: fetch (12s), embed (1s); next: to_f32" in text
    assert "FAILED to_f32: RuntimeError('boom')" in text


def test_generation_log_is_rotated(tmp_path):
    from harness import pipeline
    ctx = pipeline.make_context("tiny", tmp_path)
    with pipeline.file_log(ctx) as handler:
        assert handler.maxBytes == pipeline.LOG_MAX_BYTES and handler.backupCount == pipeline.LOG_BACKUPS
