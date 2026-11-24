"""The machine a result was produced on: instance type, CPU, cores, memory, disk type."""

import os
import platform
import re
import subprocess
import urllib.request
from pathlib import Path

IMDS = "http://169.254.169.254/latest/meta-data/instance-type"


def instance_type(timeout: float = 0.3) -> str:
    try:
        token_req = urllib.request.Request("http://169.254.169.254/latest/api/token", method="PUT",
                                           headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"})
        token = urllib.request.urlopen(token_req, timeout=timeout).read().decode()
        req = urllib.request.Request(IMDS, headers={"X-aws-ec2-metadata-token": token})
        return urllib.request.urlopen(req, timeout=timeout).read().decode().strip() or "unknown"
    except (OSError, ValueError):
        return "unknown"


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        m = re.search(r"model name\s*:\s*(.+)", cpuinfo.read_text(errors="ignore"))
        if m:
            return m.group(1).strip()
    try:
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or "unknown"


def memory_gb() -> float | None:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        m = re.search(r"MemTotal:\s*(\d+) kB", meminfo.read_text())
        if m:
            return round(int(m.group(1)) / 1024 ** 2, 1)
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3, 1)
    except (ValueError, OSError):
        return None


def disk_type(path: str = ".") -> str:
    """Best effort: nvme, ssd, hdd or unknown, for the block device under /sys/block that backs `path`."""
    block = Path("/sys/block")
    if not block.exists():
        return "unknown"
    names = sorted(p.name for p in block.iterdir() if not p.name.startswith(("loop", "ram", "dm-")))
    for name in names:
        if name.startswith("nvme"):
            return "nvme"
    for name in names:
        rotational = block / name / "queue" / "rotational"
        if rotational.exists():
            return "hdd" if rotational.read_text().strip() == "1" else "ssd"
    return "unknown"


def describe() -> dict:
    return {"instance_type": instance_type(), "cpu_model": cpu_model(), "cores": os.cpu_count(),
            "memory_gb": memory_gb(), "disk_type": disk_type()}
