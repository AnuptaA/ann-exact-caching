"""Device selection for the compute stages (ground truth and encoders).

`auto` uses a CUDA GPU when torch can use one and the CPU otherwise. A missing or
unusable GPU is a notice, never an error; only an explicitly requested `cuda` or
`mps` that is unavailable fails.
"""

import contextlib
import logging
import os
import re
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)

CHOICES = ("auto", "cpu", "cuda", "mps")
ALIASES = {"numpy": "cpu"}   # the old name of the CPU ground-truth device


class DeviceError(RuntimeError):
    """An explicitly requested device is not available."""


@dataclass(frozen=True)
class DeviceInfo:
    requested: str
    kind: str              # cpu, cuda or mps
    name: str
    threads: int
    reason: str = ""       # why a GPU is not used (auto only); empty when none is present
    torch: str | None = None

    def notice(self, stage: str) -> str:
        if self.kind != "cpu":
            return f"Using {self.kind} ({self.name}) for {stage}"
        if self.requested == "cpu":
            return f"Running {stage} on CPU ({self.threads} threads, as requested)"
        if self.reason:
            return f"No usable GPU: running {stage} on CPU ({self.threads} threads). {self.reason}"
        return f"No GPU found: running {stage} on CPU ({self.threads} threads)"

    def record(self, stage: str) -> dict:
        """Stage metadata: where the stage ran and what the user was told."""
        return {"device": self.kind, "device_name": self.name, "requested_device": self.requested,
                "threads": self.threads, "torch": self.torch, "device_notice": self.notice(stage)}


def normalize(requested: str) -> str:
    name = ALIASES.get(requested.lower(), requested.lower())
    if name not in CHOICES:
        raise ValueError(f"unknown device {requested!r}; choose from {', '.join(CHOICES)}")
    return name


def default_threads() -> int:
    """All usable cores, or Q1_THREADS."""
    env = os.environ.get("Q1_THREADS", "")
    if env:
        if not env.isdigit() or int(env) < 1:
            raise ValueError(f"Q1_THREADS must be a positive integer, got {env!r}")
        return int(env)
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def available_ram_bytes() -> int:
    """RAM that can be used now: MemAvailable on Linux, free and inactive pages on macOS, else half of total."""
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout
        page = int(re.search(r"page size of (\d+) bytes", out).group(1))
        pages = sum(int(m.group(1)) for m in re.finditer(r"Pages (?:free|inactive|speculative):\s+(\d+)", out))
        if pages:
            return pages * page
    except (OSError, AttributeError, ValueError, subprocess.TimeoutExpired):
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 2
    except (ValueError, OSError, AttributeError):
        return 1 << 62


def can_cap_threads() -> bool:
    try:
        import threadpoolctl  # noqa: F401
    except ImportError:
        return False
    return True


def _import_torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def _gpu_state(torch) -> tuple[bool, str, str]:
    """(usable, GPU name, reason it is not used). An empty reason means no GPU is present."""
    from harness.deploy import gpu_check
    if torch is None:
        found = gpu_check.query_smi()
        if found is None:
            return False, "", ""
        return False, found[1], (f"nvidia-smi lists {found[1]} (driver {found[0]}) but torch is not "
                                 "installed in this environment")
    code, message = gpu_check.diagnose(torch, gpu_check.query_smi)
    if code == gpu_check.EXIT_OK:
        return True, torch.cuda.get_device_name(0), ""
    if code == gpu_check.EXIT_NO_DRIVER:
        return False, "", ""
    return False, "", message


def resolve(requested: str = "auto") -> DeviceInfo:
    """Pick the device. `cuda` and `mps` raise DeviceError when unavailable."""
    want = normalize(requested)
    threads = default_threads()
    torch = _import_torch()
    version = torch.__version__ if torch is not None else None
    if want == "cpu":
        return DeviceInfo(requested, "cpu", "cpu", threads, torch=version)
    if want == "mps":
        if torch is None or not torch.backends.mps.is_available():
            raise DeviceError("device mps requested but torch cannot see an MPS device")
        return DeviceInfo(requested, "mps", "mps", threads, torch=version)
    usable, name, reason = _gpu_state(torch)
    if usable:
        return DeviceInfo(requested, "cuda", name, threads, torch=version)
    if want == "cuda":
        raise DeviceError("device cuda requested but no GPU is usable: "
                          + (reason or "no NVIDIA driver or GPU found"))
    return DeviceInfo(requested, "cpu", "cpu", threads, reason=reason, torch=version)


def announce(info: DeviceInfo, stage: str) -> str:
    """Log the notice at stage start (a warning when a GPU was expected) and return it."""
    text = info.notice(stage)
    (logger.warning if info.kind == "cpu" and info.requested == "auto" else logger.info)(text)
    return text


@contextlib.contextmanager
def limited_threads(threads: int):
    """Cap BLAS and OpenMP threads with threadpoolctl (no cap is applied without it)."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        if threads < default_threads():
            logger.warning("threadpoolctl is not installed: the BLAS thread cap is not applied")
        yield
        return
    with threadpool_limits(limits=threads):
        yield
