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
