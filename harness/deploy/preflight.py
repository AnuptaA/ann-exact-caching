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


def need_bytes(dataset: str) -> dict:
    """Disk and RAM needs: float32 base, pgvector table and index, working files."""
    n, d = DATASETS[dataset]
    base_f32 = n * d * 4
    graph = hnsw_graph_bytes(n, d, BUILD_PARAMS[dataset]["m"])
    return {"disk": int(2 * base_f32 + 2 * graph + 8 * GIB),   # f32 base + table + index + tables/GT/results
            "ram": int(graph * 1.15 + 4 * GIB)}                # build graph in maintenance_work_mem + OS


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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--data-dir", default="/data")
    ap.add_argument("--allow-non-linux", action="store_true", help="for dry runs on a dev machine")
    args = ap.parse_args(argv)
    problems = check(args.datasets, Path(args.data_dir), not args.allow_non_linux)
    for p in problems:
        print(f"PREFLIGHT FAIL: {p}", file=sys.stderr)
    print("preflight ok" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
