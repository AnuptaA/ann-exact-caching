"""GPU check: a missing driver, a driver that is too old and a torch problem give different messages."""

import stat
import sys
from types import SimpleNamespace

import pytest

from harness.deploy import gpu_check


def fake_torch(cuda="12.4", error=None):
    class Tensor:
        def cuda(self):
            if error:
                raise RuntimeError(error)
            return self

    return SimpleNamespace(__version__="2.6.0+cu124", version=SimpleNamespace(cuda=cuda),
                           zeros=lambda n: Tensor(),
                           cuda=SimpleNamespace(get_device_name=lambda i: "NVIDIA A10G"))


NO_DRIVER = "Found no NVIDIA driver on your system."


def test_working_gpu_passes():
    code, message = gpu_check.diagnose(fake_torch(), lambda: None)
    assert code == gpu_check.EXIT_OK and "NVIDIA A10G" in message


def test_no_driver_is_reported_with_the_instance_fix():
    code, message = gpu_check.diagnose(fake_torch(error=NO_DRIVER), lambda: None)
    assert code == gpu_check.EXIT_NO_DRIVER
    assert "no NVIDIA driver or GPU" in message and "nvidia-driver-<N>" in message and "g5" in message
    assert "Q1_ALLOW_NO_GPU=1" in message and NO_DRIVER.rstrip(".") in message


def test_old_driver_is_reported_with_both_versions_and_the_fix():
    code, message = gpu_check.diagnose(fake_torch(error="CUDA initialization: unknown error"),
                                       lambda: ("470.82.01", "NVIDIA A10G"))
    assert code == gpu_check.EXIT_DRIVER_TOO_OLD
    assert "470.82.01" in message and "525.60.13" in message and "CUDA 12.4" in message
    assert "Upgrade the driver" in message


def test_torch_saying_the_driver_is_too_old_counts_even_without_nvidia_smi():
    error = "The NVIDIA driver on your system is too old (found version 11040)."
    assert gpu_check.diagnose(fake_torch(error=error), lambda: None)[0] == gpu_check.EXIT_DRIVER_TOO_OLD


def test_cpu_only_torch_build_is_its_own_problem():
    code, message = gpu_check.diagnose(fake_torch(cuda=None), lambda: ("550.54.14", "NVIDIA A10G"))
    assert code == gpu_check.EXIT_TORCH and "CPU-only build" in message and "requirements-q1.lock" in message


def test_recent_driver_that_torch_cannot_use_prints_the_context(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    code, message = gpu_check.diagnose(fake_torch(error="CUDA error: unknown error"),
                                       lambda: ("550.54.14", "NVIDIA A10G"))
    assert code == gpu_check.EXIT_CUDA_UNUSABLE
    assert "550.54.14" in message and "CUDA_VISIBLE_DEVICES=''" in message and "unknown error" in message


def test_the_four_failures_have_different_messages_and_codes():
    results = [
        gpu_check.diagnose(fake_torch(error=NO_DRIVER), lambda: None),
        gpu_check.diagnose(fake_torch(error="x"), lambda: ("470.82.01", "gpu")),
        gpu_check.diagnose(fake_torch(cuda=None), lambda: None),
        gpu_check.diagnose(fake_torch(error="x"), lambda: ("550.54.14", "gpu")),
    ]
    assert len({code for code, _ in results}) == 4 and len({msg for _, msg in results}) == 4


def test_version_order():
    assert gpu_check.parse_version("525.60.13") < gpu_check.parse_version("550.54.14")
    assert gpu_check.parse_version("470.82.01") < gpu_check.parse_version("525.60.13")


def _smi(tmp_path, monkeypatch, body):
    shim = tmp_path / "nvidia-smi"
    shim.write_text("#!/bin/sh\n" + body + "\n")
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")


def test_query_smi_parses_fails_and_times_out(tmp_path, monkeypatch):
    _smi(tmp_path, monkeypatch, 'echo "550.54.14, NVIDIA A10G"')
    assert gpu_check.query_smi() == ("550.54.14", "NVIDIA A10G")
    _smi(tmp_path, monkeypatch, "exit 9")
    assert gpu_check.query_smi() is None
    monkeypatch.setattr(gpu_check, "SMI_TIMEOUT_SECONDS", 1)
    _smi(tmp_path, monkeypatch, "sleep 30")
    assert gpu_check.query_smi() is None
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert gpu_check.query_smi() is None


def test_main_reports_an_unimportable_torch(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert gpu_check.main() == gpu_check.EXIT_TORCH
    assert "cannot be imported" in capsys.readouterr().err
