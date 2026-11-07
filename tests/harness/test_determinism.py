"""The same cell twice, with other thread counts and hash seeds, gives byte-identical outputs."""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.harness.test_replay import DRIVERS, needs_driver

REPO = Path(__file__).resolve().parents[2]

SCRIPT = """
import hashlib, sys, tempfile
from pathlib import Path
import numpy as np
from harness.runner import manifest, replay, results
from harness.runner.prepare import prepare_dataset
from tests.harness.test_replay import ExactBackend
rng = np.random.default_rng(1)
centers = rng.normal(size=(8, 6)).astype(np.float32) * 5
base = centers[rng.integers(0, 8, 2000)] + rng.normal(size=(2000, 6)).astype(np.float32) * 0.5
off = centers[rng.integers(0, 8, 40)] + rng.normal(size=(40, 6)).astype(np.float32) * 0.5
root = Path(tempfile.mkdtemp())
prepare_dataset(root, "toy", base, off, ExactBackend(base), ef_search=50, list_size=20,
                n_groups=20, group_size=3, n_requests=400, zipf=(0.99,))
ds = replay.DatasetDir(root, "toy")
digest = hashlib.sha256()
for system, setting, budget in (("aker", "paper", 5), ("proximity", "p1", 5)):
    cell = manifest.Cell("toy", "zipf_0.99", system, setting, budget)
    header, arrays = results.read_result(replay.run_cell(cell, ds, Path(sys.argv[1]), root / "r", root / "w"))
    for key in ("outcome", "recall", "equality", "approximate", "stream_ids"):
        digest.update(arrays[key].tobytes())
    digest.update((root / "w" / cell.cell_id / "result_ids.i64").read_bytes())
    digest.update(str(header["settings"]["global_thresh"]).encode())
print(digest.hexdigest())
"""


def run(threads: str, hashseed: str) -> str:
    env = {**os.environ, "PYTHONPATH": str(REPO), "OMP_NUM_THREADS": threads, "PYTHONHASHSEED": hashseed}
    out = subprocess.run([sys.executable, "-c", SCRIPT, str(DRIVERS)], env=env, cwd=REPO, capture_output=True,
                         text=True, check=True)
    return out.stdout.strip().splitlines()[-1]


@needs_driver
def test_same_cell_is_byte_identical_across_runs_threads_and_hash_seeds():
    first = run("1", "0")
    assert len(first) == 64
    assert run("1", "0") == first
    assert run("4", "123") == first
