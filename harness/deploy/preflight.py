"""Preflight for a Linux x86_64 run instance: architecture, disk, RAM, clean git tree.

A missing GPU is a note, not a problem: the compute stages run on the CPU, and the notes give an estimate
of how long that takes per dataset. --require-gpu makes a usable GPU mandatory.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from harness.backend.build_index import BUILD_PARAMS, hnsw_graph_bytes
from harness.deploy import gpu_check
from workloads.q1 import cost, device as device_mod, universe
from workloads.q1.datasets import sift10m, spacev

GIB = 2 ** 30
# dataset: (vectors, dim)
DATASETS = {"spacev": (10_000_000, 100), "sift10m": (10_000_000, 128),
            "sphere": (10_000_000, 768), "esci": (1_814_924, 384)}


# the smoke dataset can be sized too (a dry run on a small machine), but is not part of the default list
SIZING = {**DATASETS, "tiny": (4000, 16)}

# bytes per value of the downloaded source: spacev int8, sift10m uint8, sphere float64, esci float32
RAW_ITEMSIZE = {"spacev": 1, "sift10m": 1, "sphere": 8, "esci": 4, "tiny": 4}
MAX_WAL_BYTES = 64 * GIB      # max_wal_size set by setup_instance.sh
OTHER_BYTES = 8 * GIB         # answer tables, ground truth, streams, results
OTHER_BYTES_TINY = GIB // 4
RAM_BASE_BYTES = 4 * GIB      # headroom on top of the HNSW graph
RAM_BASE_BYTES_TINY = GIB

# distinct official query vectors per dataset (the interpolation step draws from them); ESCI and SPHERE
# are the expected distinct text counts of harness/DECISIONS.md
OFFICIAL_QUERIES = {"esci": 130_193, "spacev": spacev.N_QUERY, "sift10m": sift10m.N_QUERY, "sphere": 96_005,
                    "tiny": 600}
EMBEDDED_TEXTS = {"esci": 1_814_924 + 130_193, "sphere": 96_005}


def need_bytes(dataset: str) -> dict:
    """Disk by component and the RAM needed for the HNSW build.

    The raw source, the float32 file and the pgvector table coexist. WAL is bounded by max_wal_size; wal_level=minimal
    should keep it far lower, but that is not verified at full size.
    """
    n, d = SIZING[dataset]
    tiny = dataset == "tiny"
    base_f32 = n * d * 4
    graph = hnsw_graph_bytes(n, d, BUILD_PARAMS.get(dataset, {"m": 16})["m"])
    parts = {"raw": n * d * RAW_ITEMSIZE[dataset], "f32": base_f32, "table": n * (d * 4 + 64), "index": graph,
             "wal": min(MAX_WAL_BYTES, base_f32), "other": OTHER_BYTES_TINY if tiny else OTHER_BYTES}
    ram_base = RAM_BASE_BYTES_TINY if tiny else RAM_BASE_BYTES
    return {**parts, "disk": int(sum(parts.values())), "ram": int(graph * 1.15 + ram_base)}


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


def check(datasets, data_dir: Path, require_linux: bool = True, repo: Path | None = None,
          shortfalls: list[str] | None = None) -> list[str]:
    """Return the list of problems; an empty list means the instance can run.

    Disk and RAM shortfalls are problems, unless a `shortfalls` list is passed: they are then appended to it
    (a dry run on a small machine) and the other problems are still returned.
    """
    problems = []
    if require_linux and (platform.system() != "Linux" or platform.machine() != "x86_64"):
        problems.append(f"need Linux x86_64, found {platform.system()} {platform.machine()}")
    status = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True)
    if status.returncode != 0:
        problems.append("not a git checkout")
    elif status.stdout.strip():
        problems.append("git tree is dirty: commit or stash before deploying")
    problems += check_data_volume(data_dir)
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    needs = [need_bytes(d) for d in datasets]
    if not needs:
        return problems
    disk_needed = sum(n["disk"] for n in needs)
    sink = problems if shortfalls is None else shortfalls
    free = shutil.disk_usage(data_dir).free
    if free < disk_needed:
        sink.append(f"disk: {free / GIB:.0f} GiB free in {data_dir}, need {disk_needed / GIB:.0f} GiB")
    ram_needed = max(n["ram"] for n in needs)
    total_ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") if hasattr(os, "sysconf") else 0
    if total_ram < ram_needed:
        sink.append(f"RAM: {total_ram / GIB:.0f} GiB, need {ram_needed / GIB:.0f} GiB for the HNSW build")
    return problems


def compute_notes(datasets) -> list[str]:
    """Where the compute stages will run and, on the CPU, how long each dataset takes (order of magnitude)."""
    info = device_mod.resolve("auto")
    if info.kind != "cpu":
        return [f"compute stages run on {info.kind} ({info.name})"]
    notes = [info.notice("ground truth and text embedding")]
    for name in datasets:
        if name not in OFFICIAL_QUERIES:
            continue
        n, d = SIZING[name]
        n_ids = universe.N_GROUPS * universe.GROUP_SIZE + universe.N_GROUPS + OFFICIAL_QUERIES[name]
        if name == "tiny":
            n_ids = 100 * 5 + 100 + OFFICIAL_QUERIES[name]
        embed = cost.encode_seconds(name, EMBEDDED_TEXTS.get(name, 0), info.threads)
        truth = cost.ground_truth_seconds(n_ids, n, d, info.threads)
        notes.append(f"{name} on the CPU ({info.threads} threads): text embedding "
                     f"{cost.describe(embed) if embed else 'none (released vectors)'}, ground truth "
                     f"{cost.describe(truth)} (order of magnitude)")
    return notes


def check_gpu() -> list[str]:
    """A CUDA device must be usable (only with --require-gpu)."""
    try:
        import torch
    except ImportError:
        return ["torch is not installed"]
    code, message = gpu_check.diagnose(torch)
    return [] if code == gpu_check.EXIT_OK else [message]


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
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS), choices=list(SIZING),
                    help="datasets to size disk and RAM for (tiny = smoke test); none skips the sizing")
    ap.add_argument("--allow-shortfall", action="store_true",
                    help="dry run on a small machine: disk and RAM shortfalls are warnings, other problems still fail")
    ap.add_argument("--data-dir", default="/data")
    ap.add_argument("--allow-non-linux", action="store_true", help="for dry runs on a dev machine")
    ap.add_argument("--require-gpu", action="store_true", help="fail when no GPU is usable (default: a note)")
    ap.add_argument("--dsn", default=None, help="also check PostgreSQL 16 and pgvector 0.8.0")
    args = ap.parse_args(argv)
    shortfalls: list[str] = []
    problems = check(args.datasets, Path(args.data_dir), not args.allow_non_linux,
                     shortfalls=shortfalls if args.allow_shortfall else None)
    if args.require_gpu:
        problems += check_gpu()
    if args.dsn:
        problems += check_postgres(args.dsn)
    for note in compute_notes(args.datasets):
        print(f"PREFLIGHT NOTE: {note}")
    for s in shortfalls:
        print(f"PREFLIGHT WARN: {s} (ignored: --allow-shortfall)")
    for p in problems:
        print(f"PREFLIGHT FAIL: {p}", file=sys.stderr)
    warned = f" with {len(shortfalls)} shortfall warning(s)" if shortfalls else ""
    print(f"preflight ok{warned}" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
