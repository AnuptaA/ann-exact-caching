"""CPU ground truth: RAM-aware tiles, threads, resume, progress and provenance."""

import json
import logging

import numpy as np
import pytest

from workloads.q1 import device, groundtruth


def _data(n_base=900, n_queries=70, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(n_queries, dim)).astype(np.float32),
            rng.normal(size=(n_base, dim)).astype(np.float32))


def _brute(queries, base, k):
    d = ((queries[:, None, :] - base[None]) ** 2).sum(2)
    return np.argsort(d, axis=1, kind="stable")[:, :k]


def test_cpu_alias_and_name_take_the_numpy_path():
    queries, base = _data()
    a = groundtruth.exact_topk_chunk(queries, base, k=20, device="numpy", base_chunk=200)
    b = groundtruth.exact_topk_chunk(queries, base, k=20, device="cpu", base_chunk=200)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert np.array_equal(a[0], _brute(queries, base, 20))


@pytest.mark.skipif(not device.can_cap_threads(), reason="needs threadpoolctl")
def test_threaded_path_equals_the_serial_path():
    queries, base = _data(n_queries=130)
    serial = groundtruth.exact_topk_chunk(queries, base, k=30, base_chunk=150, workers=1)
    threaded = groundtruth.exact_topk_chunk(queries, base, k=30, base_chunk=150, workers=4)
    assert groundtruth._usable_workers(4, 130) == 4
    assert np.array_equal(serial[0], threaded[0]) and np.array_equal(serial[1], threaded[1])
    assert np.array_equal(threaded[0], _brute(queries, base, 30))


def test_one_worker_without_a_way_to_cap_blas_threads(monkeypatch):
    monkeypatch.setattr(device, "can_cap_threads", lambda: False)
    assert groundtruth._usable_workers(8, 1000) == 1
    monkeypatch.setattr(device, "can_cap_threads", lambda: True)
    assert groundtruth._usable_workers(8, 1000) == 8
    assert groundtruth._usable_workers(8, 20) == 1     # too few queries to split


def test_cpu_tiles_shrink_to_the_available_ram():
    gib = 2 ** 30
    big = groundtruth.fit_cpu_chunks(1024, 262144, 768, available=64 * gib)
    assert big == (1024, 262144)
    q, b = groundtruth.fit_cpu_chunks(1024, 262144, 768, available=2 * gib)
    assert (q, b) != (1024, 262144) and q >= 64 and b >= 4096
    assert groundtruth.TILE_BYTES_PER_ELEMENT * q * b + 8 * b * 768 <= groundtruth.CPU_RAM_FRACTION * 2 * gib
    with pytest.raises(MemoryError, match="does not fit in RAM"):
        groundtruth.fit_cpu_chunks(1024, 262144, 768, available=10 ** 6)


def test_build_resumes_and_shrunk_tiles_give_the_same_answer(tmp_path, monkeypatch):
    queries, base = _data(n_queries=90)
    monkeypatch.setattr(device, "available_ram_bytes", lambda: 64 * 2 ** 20)
    out = tmp_path / "gt"
    ids, dist = groundtruth.build_ground_truth(queries, base, out, k=25, query_chunk=30, base_chunk=262144)
    assert np.array_equal(ids, _brute(queries, base, 25))
    (out / "chunk_000000030.npz").unlink()
    again = groundtruth.build_ground_truth(queries, base, out, k=25, query_chunk=30)
    assert np.array_equal(again[0], ids) and np.array_equal(again[1], dist)
    runs = json.loads((out / "provenance.json").read_text())["runs"]
    assert len(runs) == 2 and runs[0]["device"] == "numpy" and runs[0]["base_chunk"] < 262144


def test_cpu_and_alias_runs_resume_each_other(tmp_path):
    queries, base = _data(n_queries=60)
    out = tmp_path / "gt"
    first = groundtruth.build_ground_truth(queries, base, out, k=10, device="numpy", query_chunk=20)
    second = groundtruth.build_ground_truth(queries, base, out, k=10, device="cpu", query_chunk=20)
    assert np.array_equal(first[0], second[0])
    devices = [r["device"] for r in json.loads((out / "provenance.json").read_text())["runs"]]
    assert devices == ["numpy", "cpu"]
    with pytest.raises(ValueError, match="different run"):
        groundtruth.build_ground_truth(queries, base, out, k=10, query_chunk=30)


def test_progress_lines_carry_a_rate_and_an_eta(tmp_path, caplog):
    queries, base = _data(n_base=1200, n_queries=60)
    with caplog.at_level(logging.INFO, logger="workloads.q1.progress"):
        groundtruth.build_ground_truth(queries, base, tmp_path / "gt", k=10, query_chunk=30,
                                       base_chunk=300, progress_interval=0.0)
    lines = [r.message for r in caplog.records if r.message.startswith("ground truth:")]
    assert lines and any("ETA" in m and "queries/s" in m for m in lines)


def test_progress_starts_from_the_resumed_position(tmp_path, caplog):
    queries, base = _data(n_queries=60)
    out = tmp_path / "gt"
    groundtruth.build_ground_truth(queries, base, out, k=10, query_chunk=20)
    (out / "chunk_000000040.npz").unlink()
    with caplog.at_level(logging.INFO, logger="workloads.q1.progress"):
        groundtruth.build_ground_truth(queries, base, out, k=10, query_chunk=20, progress_interval=0.0)
    first = [r.message for r in caplog.records if r.message.startswith("ground truth:")][0]
    assert "/60 queries" in first and not first.startswith("ground truth: 0/")


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


@pytest.mark.skipif(not _cuda_available(), reason="needs a CUDA device")
def test_cpu_and_cuda_agree_within_the_tie_tolerance(tmp_path):
    queries, base = _data(n_base=5000, n_queries=200, dim=32, seed=3)
    cpu = groundtruth.build_ground_truth(queries, base, tmp_path / "cpu", k=50, device="cpu", query_chunk=64)
    gpu = groundtruth.build_ground_truth(queries, base, tmp_path / "gpu", k=50, device="cuda", query_chunk=64)
    assert np.array_equal(cpu[1], gpu[1])               # distances come from the shared float32 recompute
    assert (cpu[0] == gpu[0]).mean() > 0.999            # ids differ only where distances tie


def test_auto_resolves_to_a_device_kind(tmp_path, monkeypatch):
    monkeypatch.setattr(device, "resolve", lambda requested="auto": device.DeviceInfo(requested, "cpu", "cpu", 1))
    queries, base = _data(n_queries=20)
    ids, _ = groundtruth.build_ground_truth(queries, base, tmp_path / "gt", k=5, device="auto", query_chunk=20)
    assert np.array_equal(ids, _brute(queries, base, 5))
    assert json.loads((tmp_path / "gt" / "provenance.json").read_text())["runs"][0]["device"] == "cpu"
