"""Preflight for a Linux x86_64 run instance: architecture, disk, RAM, clean git tree."""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from harness.backend.build_index import BUILD_PARAMS, hnsw_graph_bytes

GIB = 2 ** 30
# dataset: (vectors, dim)
DATASETS = {"spacev": (10_000_000, 100), "sift10m": (10_000_000, 128),
            "sphere": (10_000_000, 768), "esci": (1_814_924, 384)}


# bytes per value of the downloaded source: spacev int8, sift10m uint8, sphere float64, esci float32
RAW_ITEMSIZE = {"spacev": 1, "sift10m": 1, "sphere": 8, "esci": 4}
MAX_WAL_BYTES = 64 * GIB      # max_wal_size set by setup_instance.sh
OTHER_BYTES = 8 * GIB         # answer tables, ground truth, streams, results


def need_bytes(dataset: str) -> dict:
    """Disk by component and the RAM needed for the HNSW build.

    The raw source, the float32 file and the pgvector table coexist. WAL is bounded by max_wal_size; wal_level=minimal
    should keep it far lower, but that is not verified at full size.
    """
    n, d = DATASETS[dataset]
    base_f32 = n * d * 4
    graph = hnsw_graph_bytes(n, d, BUILD_PARAMS[dataset]["m"])
    parts = {"raw": n * d * RAW_ITEMSIZE[dataset], "f32": base_f32, "table": n * (d * 4 + 64), "index": graph,
             "wal": min(MAX_WAL_BYTES, base_f32), "other": OTHER_BYTES}
    return {**parts, "disk": int(sum(parts.values())), "ram": int(graph * 1.15 + 4 * GIB)}


def device_of(path: Path) -> int:
    return os.stat(path).st_dev


def check_data_volume(data_dir: Path) -> list[str]:
    """The data directory must not sit on the root volume (an unmounted /data would fill it)."""
    if os.environ.get("Q1_ALLOW_ROOT_VOLUME") == "1":
        return []
    probe = Path(data_dir).resolve()
    while not probe.exists():
        probe = probe.parent
    if device_of(probe) == device_of(Path("/")):
        return [f"{data_dir} is on the root volume: mount the data volume there, or set "
                "Q1_ALLOW_ROOT_VOLUME=1 to accept it"]
    return []


def check(datasets, data_dir: Path, require_linux: bool = True, repo: Path | None = None) -> list[str]:
    """Return the list of problems; an empty list means the instance can run."""
    problems = []
    if require_linux and (platform.system() != "Linux" or platform.machine() != "x86_64"):
        problems.append(f"need Linux x86_64, found {platform.system()} {platform.machine()}")
    status = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True)
    if status.returncode != 0:
        problems.append("not a git checkout")
    elif status.stdout.strip():
        problems.append("git tree is dirty: commit or stash before deploying")
    problems += check_data_volume(data_dir)
    needs = [need_bytes(d) for d in datasets]
    disk_needed = sum(n["disk"] for n in needs)
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(data_dir).free
    if free < disk_needed:
        problems.append(f"disk: {free / GIB:.0f} GiB free in {data_dir}, need {disk_needed / GIB:.0f} GiB")
    ram_needed = max(n["ram"] for n in needs)
    total_ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") if hasattr(os, "sysconf") else 0
    if total_ram < ram_needed:
        problems.append(f"RAM: {total_ram / GIB:.0f} GiB, need {ram_needed / GIB:.0f} GiB for the HNSW build")
    return problems


def check_gpu() -> list[str]:
    """A CUDA device must be visible for GPU ground truth and encoding."""
    try:
        import torch
    except ImportError:
        return ["torch is not installed"]
    return [] if torch.cuda.is_available() else ["no CUDA device visible to torch"]


def check_postgres(dsn: str, expected: str = "0.8.0") -> list[str]:
    """PostgreSQL 16 reachable and pgvector available at the expected version."""
    try:
        import psycopg
        from psycopg.conninfo import make_conninfo
        with psycopg.connect(make_conninfo(dsn, dbname="postgres"), connect_timeout=10) as conn:
            major = conn.execute("SHOW server_version_num").fetchone()[0][:2]
            row = conn.execute("SELECT default_version FROM pg_available_extensions "
                               "WHERE name = 'vector'").fetchone()
    except Exception as exc:
        return [f"PostgreSQL not reachable: {exc}"]
    problems = []
    if major != "16":
        problems.append(f"PostgreSQL major version {major}, expected 16")
    if row is None or row[0] != expected:
        problems.append(f"pgvector {row[0] if row else 'missing'}, expected {expected}")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--data-dir", default="/data")
    ap.add_argument("--allow-non-linux", action="store_true", help="for dry runs on a dev machine")
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--dsn", default=None, help="also check PostgreSQL 16 and pgvector 0.8.0")
    args = ap.parse_args(argv)
    problems = check(args.datasets, Path(args.data_dir), not args.allow_non_linux)
    if args.require_gpu:
        problems += check_gpu()
    if args.dsn:
        problems += check_postgres(args.dsn)
    for p in problems:
        print(f"PREFLIGHT FAIL: {p}", file=sys.stderr)
    print("preflight ok" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
