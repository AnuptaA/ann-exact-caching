"""GPU check for setup and preflight: a missing driver is not the same problem as a driver that is too old."""

import os
import re
import subprocess
import sys
from typing import Callable

EXIT_OK = 0
EXIT_NO_DRIVER = 10
EXIT_DRIVER_TOO_OLD = 11
EXIT_TORCH = 12
EXIT_CUDA_UNUSABLE = 13

# Minimum NVIDIA driver per CUDA major version (Linux, minor-version compatibility).
MIN_DRIVER = {12: "525.60.13", 11: "450.80.02"}
SMI_TIMEOUT_SECONDS = 20


def parse_version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", text))


def query_smi() -> tuple[str, str] | None:
    """(driver version, GPU name) from nvidia-smi; None when it is missing, failing or not answering."""
    cmd = ["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=SMI_TIMEOUT_SECONDS)
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = out.stdout.strip().splitlines() if out.returncode == 0 else []
    if not lines:
        return None
    driver, _, name = lines[0].partition(",")
    return driver.strip(), name.strip()


def diagnose(torch, smi: Callable[[], tuple[str, str] | None] = query_smi) -> tuple[int, str]:
    """Exit code and message: 0 when torch can use a GPU, else which of four causes applies."""
    cuda = getattr(torch.version, "cuda", None)
    if cuda is None:
        return EXIT_TORCH, (
            f"torch {torch.__version__} is a CPU-only build (torch.version.cuda is None). The lock file installs "
            "a CUDA build: reinstall with `.venv/bin/pip install --require-hashes -r requirements-q1.lock`.")
    try:
        torch.zeros(1).cuda()
        return EXIT_OK, f"torch {torch.__version__} cuda {cuda} {torch.cuda.get_device_name(0)}"
    except Exception as exc:
        error = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    found = smi()
    too_old_text = "too old" in error.lower()
    if found is None and not too_old_text:
        return EXIT_NO_DRIVER, (
            f"no NVIDIA driver or GPU found (nvidia-smi is missing or not answering; torch said: {error}). "
            "Use a GPU instance type with the NVIDIA driver installed (for example g5 or g6), or install "
            "nvidia-driver-<N> and reboot, then check that `nvidia-smi` lists the GPU. "
            "For a CPU-only dry run set Q1_ALLOW_NO_GPU=1.")
    driver, name = found if found is not None else ("unknown", "")
    required = MIN_DRIVER.get(int(cuda.split(".")[0]))
    if too_old_text or (required is not None and driver != "unknown"
                        and parse_version(driver) < parse_version(required)):
        return EXIT_DRIVER_TOO_OLD, (
            f"NVIDIA driver {driver} is too old for torch {torch.__version__} (built for CUDA {cuda}, needs a "
            f"driver >= {required or 'a newer version'}). Upgrade the driver (nvidia-driver-<N>, then reboot) "
            "or install a torch build for an older CUDA version that matches the driver. The lock file pins "
            "torch 2.6.0 with CUDA 12.4.")
    return EXIT_CUDA_UNUSABLE, (
        f"the NVIDIA driver ({driver}, {name}) is recent enough for CUDA {cuda}, but torch cannot use it: "
        f"{error}. CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}. Check the `nvidia-smi` "
        "output for errors, and in a container that the GPU device is passed through.")


def main() -> int:
    try:
        import torch
    except ImportError as exc:
        print(f"GPU CHECK FAILED ({EXIT_TORCH}): torch cannot be imported ({exc}); reinstall from "
              "requirements-q1.lock", file=sys.stderr)
        return EXIT_TORCH
    code, message = diagnose(torch)
    if code == EXIT_OK:
        print(message)
    else:
        print(f"GPU CHECK FAILED ({code}): {message}", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
