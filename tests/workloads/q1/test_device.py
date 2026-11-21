"""Device selection: auto falls back to the CPU with a notice; only explicit cuda/mps can fail."""

import logging
import sys
import types

import pytest

from harness.deploy import gpu_check
from tests.harness.test_gpu_check import NO_DRIVER, fake_torch
from workloads.q1 import device


@pytest.fixture(autouse=True)
def no_threads_env(monkeypatch):
    monkeypatch.delenv("Q1_THREADS", raising=False)


def no_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)       # makes `import torch` raise ImportError
    monkeypatch.setattr(gpu_check, "query_smi", lambda: None)


def test_auto_without_a_gpu_picks_the_cpu_and_says_so(monkeypatch, caplog):
    no_torch(monkeypatch)
    info = device.resolve("auto")
    assert info.kind == "cpu" and info.reason == "" and info.threads >= 1
    with caplog.at_level(logging.WARNING, logger="workloads.q1.device"):
        text = device.announce(info, "ground truth")
    assert text.startswith("No GPU found: running ground truth on CPU") and text in caplog.text
    record = info.record("ground truth")
    assert record["device"] == "cpu" and record["requested_device"] == "auto"
    assert record["device_notice"] == text and record["threads"] == info.threads


def test_auto_with_torch_but_no_driver_is_also_a_plain_notice(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(error=NO_DRIVER))
    monkeypatch.setattr(gpu_check, "query_smi", lambda: None)
    info = device.resolve("auto")
    assert info.kind == "cpu" and info.reason == "" and "No GPU found" in info.notice("embedding")


def test_auto_picks_cuda_when_torch_can_use_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    info = device.resolve("auto")
    assert info.kind == "cuda" and info.notice("embedding").startswith("Using cuda")


def test_a_present_but_unusable_gpu_runs_on_the_cpu_with_the_diagnosis(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(error="CUDA driver version is insufficient"))
    monkeypatch.setattr(gpu_check, "query_smi", lambda: ("470.1", "NVIDIA A10G"))
    info = device.resolve("auto")
    assert info.kind == "cpu" and "too old" in info.reason
    text = info.notice("embedding")
    assert text.startswith("No usable GPU: running embedding on CPU") and "too old" in text
    with pytest.raises(device.DeviceError, match="too old"):
        device.resolve("cuda")


def test_a_gpu_that_torch_cannot_see_because_torch_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(gpu_check, "query_smi", lambda: ("550.1", "NVIDIA A10G"))
    info = device.resolve("auto")
    assert info.kind == "cpu" and "torch is not installed" in info.reason


def test_cpu_alias_and_explicit_requests(monkeypatch):
    no_torch(monkeypatch)
    assert device.resolve("cpu").kind == "cpu" and device.resolve("numpy").kind == "cpu"
    assert "as requested" in device.resolve("cpu").notice("embedding")
    with pytest.raises(device.DeviceError, match="cuda requested"):
        device.resolve("cuda")
    with pytest.raises(device.DeviceError, match="mps"):
        device.resolve("mps")
    with pytest.raises(ValueError, match="unknown device"):
        device.resolve("tpu")


def test_mps_is_explicit_only(monkeypatch):
    torch = fake_torch(error=NO_DRIVER)
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(gpu_check, "query_smi", lambda: None)
    assert device.resolve("auto").kind == "cpu"
    assert device.resolve("mps").kind == "mps"


def test_threads_default_to_all_cores_and_q1_threads_overrides(monkeypatch):
    assert device.default_threads() >= 1
    monkeypatch.setenv("Q1_THREADS", "3")
    assert device.default_threads() == 3
    monkeypatch.setenv("Q1_THREADS", "zero")
    with pytest.raises(ValueError, match="Q1_THREADS"):
        device.default_threads()


def test_available_ram_is_positive():
    assert device.available_ram_bytes() > 0
