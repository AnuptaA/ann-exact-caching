"""Unit checkpoints and the progress heartbeat of a micro-experiment run.

A unit is one piece of an experiment (a budget, a setting, a chunk of queries). Units.cached(key, fn) returns the
stored payload of a finished unit and otherwise computes it, writes it atomically and pushes it. Every
`interval` seconds a heartbeat file progress.json records units done, queries per second, time left, peak
memory and the last error; units done are counted from the append-only done.jsonl, so worker processes count too.
"""

import json
import resource
import sys
import threading
import time
from pathlib import Path

from recal.micro import common


def peak_memory_mb() -> float:
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    scale = 1.0 if sys.platform == "darwin" else 1024.0     # bytes on macOS, KB on Linux
    return max(own, children) * scale / 2**20


class Heartbeat:
    """Writes <directory>/progress.json every `interval` seconds from a daemon thread."""

    def __init__(self, directory: Path, experiment: str, workload: str, total: int, interval: float = 30.0,
                 push=None):
        self.directory = Path(directory)
        self.info = {"experiment": experiment, "workload": workload, "units_total": total}
        self.interval = interval
        self.push = push
        self.started = time.time()
        self.last_error = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def __enter__(self):
        self.write()
        self._thread.start()
        return self

    def __exit__(self, kind, error, trace):
        if error is not None:
            self.last_error = f"{kind.__name__}: {error}"
        self._stop.set()
        self._thread.join(timeout=5)
        self.write(final=True)
        return False

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.write()

    def done_log(self) -> list[dict]:
        path = self.directory / common.UNITS_DIR / common.DONE_LOG
        if not path.exists():
            return []
        out = []
        for line in path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
        return out

    def snapshot(self, final: bool = False) -> dict:
        log = self.done_log()
        done = len({row["unit"] for row in log if row["t"] >= self.started})
        elapsed = max(time.time() - self.started, 1e-9)
        queries = sum(row["queries"] for row in log if row["t"] >= self.started)
        total = self.info["units_total"]
        left = elapsed / done * (total - done) if done and total > done else None
        return dict(self.info, units_done=done, queries_per_second=queries / elapsed, elapsed_s=elapsed,
                    eta_s=left, peak_memory_mb=peak_memory_mb(), last_error=self.last_error,
                    updated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), finished=final and not self.last_error)

    def write(self, final: bool = False) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / common.PROGRESS_FILE
        tmp = path.with_name("." + path.name + ".tmp")
        try:
            tmp.write_text(json.dumps(self.snapshot(final), default=str))
            tmp.replace(path)
            if self.push is not None and (final or self.last_error):
                self.push(path)
        except OSError:
            pass


class Units:
    """Checkpointed pieces of one experiment of one workload."""

    def __init__(self, directory: Path, experiment: str, settings: dict, push=None):
        self.directory = Path(directory)
        self.experiment = experiment
        self.settings = settings
        self.push = push

    def cached(self, key: str, fn, queries: int = 0):
        got = common.read_unit(self.directory, key, self.settings)
        if got is not None:
            print(f"[{self.experiment}] unit {key}: finished earlier, skipped", flush=True)
            return got
        payload = fn()
        path = common.write_unit(self.directory, key, self.settings, payload, queries)
        if self.push is not None:
            self.push(path)
        return payload

    def done(self, key: str) -> bool:
        return common.read_unit(self.directory, key, self.settings) is not None
