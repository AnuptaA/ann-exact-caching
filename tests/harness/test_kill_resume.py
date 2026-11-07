"""SIGKILL a long stage mid-run, resume, and compare with an uninterrupted run."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

GT_SCRIPT = """
import sys, time
import numpy as np
from workloads.q1 import groundtruth
out, slow = sys.argv[1], float(sys.argv[2])
rng = np.random.default_rng(0)
base = rng.normal(size=(400, 6)).astype(np.float32)
queries = rng.normal(size=(1200, 6)).astype(np.float32)
original = groundtruth.exact_topk_chunk
def paced(*a, **k):
    time.sleep(slow)
    return original(*a, **k)
groundtruth.exact_topk_chunk = paced
groundtruth.build_ground_truth(queries, base, out, k=10, query_chunk=50)
"""

TABLE_SCRIPT = """
import sys, time
import numpy as np
from harness.backend import answer_table
out, slow = sys.argv[1], float(sys.argv[2])
rng = np.random.default_rng(0)
base = rng.normal(size=(400, 6)).astype(np.float32)
vectors = rng.normal(size=(1200, 6)).astype(np.float32)
class Backend:
    name = "exact"
    def search(self, q, k, ef):
        time.sleep(slow)
        d = ((q[:, None, :] - base[None]) ** 2).sum(2)
        return np.argsort(d, axis=1, kind="stable")[:, :k]
answer_table.build_answer_table(Backend(), vectors, base, out, 20, 100, block=50)
"""


def run(script: str, out: Path, slow: float) -> subprocess.Popen:
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    return subprocess.Popen([sys.executable, "-c", script, str(out), str(slow)], env=env, cwd=REPO,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def kill_after(proc: subprocess.Popen, out: Path, pattern: str, count: int) -> int:
    deadline = time.time() + 60
    while time.time() < deadline and len(list(out.glob(pattern))) < count:
        time.sleep(0.01)
    proc.send_signal(signal.SIGKILL)
    proc.wait()
    return len(list(out.glob(pattern)))


def plant_partial_write(out: Path, final_name: str) -> None:
    """What a kill between the temporary write and the rename leaves behind."""
    dead = int(subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True,
                              text=True).stdout)
    (out / f"{final_name}.{dead}.tmp").write_bytes(b"partial")


def arrays_equal(a: Path, b: Path, pattern: str) -> None:
    files_a, files_b = sorted(a.glob(pattern)), sorted(b.glob(pattern))
    assert [p.name for p in files_a] == [p.name for p in files_b] and files_a
    for fa, fb in zip(files_a, files_b):
        za, zb = np.load(fa), np.load(fb)
        assert za.files == zb.files
        for key in za.files:
            assert za[key].dtype == zb[key].dtype and np.array_equal(za[key], zb[key]), fa.name


def test_ground_truth_killed_mid_run_resumes_to_the_same_arrays(tmp_path):
    whole, killed = tmp_path / "whole", tmp_path / "killed"
    assert run(GT_SCRIPT, whole, 0).wait() == 0
    done = kill_after(run(GT_SCRIPT, killed, 0.05), killed, "chunk_*.npz", 3)
    assert 3 <= done < 24
    plant_partial_write(killed, "chunk_000000600.npz")
    assert run(GT_SCRIPT, killed, 0).wait() == 0
    arrays_equal(whole, killed, "chunk_*.npz")
    assert sorted(p.name for p in killed.glob("chunk_*")) == sorted(p.name for p in whole.glob("chunk_*"))
    assert not list(killed.glob("*.tmp")) and not list(killed.glob("*.tmp.npz"))


def test_answer_table_killed_mid_run_resumes_to_the_same_arrays(tmp_path):
    whole, killed = tmp_path / "whole", tmp_path / "killed"
    assert run(TABLE_SCRIPT, whole, 0).wait() == 0
    done = kill_after(run(TABLE_SCRIPT, killed, 0.05), killed, "block_*.npz", 3)
    assert 3 <= done < 24
    plant_partial_write(killed, "block_0000010.npz")
    assert run(TABLE_SCRIPT, killed, 0).wait() == 0
    arrays_equal(whole, killed, "block_*.npz")
    assert not list(killed.glob("*.tmp")) and not list(killed.glob("*.tmp.npz"))
