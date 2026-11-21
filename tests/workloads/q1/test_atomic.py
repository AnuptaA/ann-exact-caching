"""Atomic writes: a failure leaves the old file intact and no temporary file behind."""

import numpy as np
import pytest

from workloads.q1 import atomic


def test_writes_replace_files_and_leave_no_temporaries(tmp_path):
    atomic.save_npy(tmp_path / "a.npy", np.arange(3))
    atomic.save_npz(tmp_path / "b.npz", x=np.ones(2), y=np.zeros(1))
    atomic.write_json(tmp_path / "c.json", {"k": 1})
    assert np.load(tmp_path / "a.npy").tolist() == [0, 1, 2]
    assert np.load(tmp_path / "b.npz")["x"].tolist() == [1.0, 1.0]
    assert (tmp_path / "c.json").read_text() == '{"k": 1}'
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.npy", "b.npz", "c.json"]


def test_a_failed_write_keeps_the_old_content(tmp_path):
    atomic.write_text(tmp_path / "f.txt", "old")

    def boom(f):
        f.write(b"par")
        raise RuntimeError("killed")

    with pytest.raises(RuntimeError):
        atomic._replace(tmp_path / "f.txt", boom)
    assert (tmp_path / "f.txt").read_text() == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["f.txt"]


def test_stale_temporaries_of_dead_writers_are_removed_and_live_ones_kept(tmp_path):
    import os
    import subprocess
    import sys
    dead = int(subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True,
                              text=True).stdout)
    (tmp_path / f"a.npz.{dead}.tmp").write_bytes(b"x")
    (tmp_path / f"b.npz.{os.getpid()}.tmp").write_bytes(b"x")
    (tmp_path / "old.tmp.npz").write_bytes(b"x")
    os.utime(tmp_path / "old.tmp.npz", (1, 1))
    (tmp_path / "fresh.tmp.npz").write_bytes(b"x")
    removed = atomic.remove_stale_temporaries(tmp_path)
    assert sorted(p.name for p in removed) == [f"a.npz.{dead}.tmp", "old.tmp.npz"]
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"b.npz.{os.getpid()}.tmp", "fresh.tmp.npz"]
