"""Progress of background runs and of dataset generation, from files only (no database needed)."""

import json
import os
import time
from pathlib import Path

from harness import datasets, pipeline

TAIL_LINES = 3


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def background_runs(data_dir: Path, limit: int = 5) -> list[dict]:
    """Newest runs started by `make q1-bg`: pid file, log file, whether it is still running."""
    runs = []
    for pid_file in sorted(Path(data_dir, "logs").glob("*.pid"), key=lambda p: p.stat().st_mtime, reverse=True):
        if pid_file.name == "last.pid":
            continue
        text = pid_file.read_text().strip()
        pid = int(text) if text.isdigit() else 0
        log = pid_file.with_suffix(".log")
        tail = pipeline.tail(log, TAIL_LINES) if log.exists() else "(no log)"
        age = time.time() - log.stat().st_mtime if log.exists() else None
        runs.append({"name": pid_file.stem, "pid": pid, "running": bool(pid) and _alive(pid),
                     "log": str(log), "seconds_since_log_write": age, "tail": tail})
        if len(runs) == limit:
            break
    return runs


def dataset_status(data_dir: Path, name: str) -> dict:
    ctx = pipeline.make_context(name, data_dir)
    done, failed = {}, None
    for stage in pipeline.STAGES:
        marker = pipeline._marker(ctx, stage)
        if marker.exists():
            done[stage] = json.loads(marker.read_text()).get("seconds")
    record = ctx.root / ".failed"
    for stage in pipeline.STAGES:
        if (record / f"{stage}.json").exists():
            failed = {"stage": stage, **json.loads((record / f"{stage}.json").read_text())}
    remaining = [s for s in pipeline.STAGES if s not in done]
    return {"dataset": name, "done": done, "next": remaining[0] if remaining else None, "failed": failed}


def format_status(data_dir: Path) -> str:
    lines = ["background runs:"]
    runs = background_runs(data_dir)
    for run in runs:
        state = "RUNNING" if run["running"] else "not running"
        lines.append(f"  {run['name']}  pid {run['pid']} {state}; log {run['log']}")
        lines += [f"      {line}" for line in run["tail"].splitlines()]
    if not runs:
        lines.append("  none")
    lines.append("datasets:")
    for name in datasets.PRODUCTION + ("tiny",):
        if not (Path(data_dir) / name).exists():
            continue
        info = dataset_status(data_dir, name)
        finished = ", ".join(f"{s} ({sec:.0f}s)" for s, sec in info["done"].items()) or "nothing"
        lines.append(f"  {name}: done: {finished}; next: {info['next'] or 'all done'}")
        if info["failed"]:
            lines.append(f"      FAILED {info['failed']['stage']}: {info['failed']['error'][:200]}")
    return "\n".join(lines)
