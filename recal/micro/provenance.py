"""What a result was computed from: a hash of the code each experiment uses, of the prepared inputs and of the libraries.

The hash replaces the commit in the config hash. A commit that changes no file an experiment uses keeps its units
valid, and an edit that is not committed (or a rebuilt core) changes it.
"""

import hashlib
import importlib.metadata
import platform
from pathlib import Path

import recal

ROOT = Path(recal.__file__).resolve().parents[1]
LIBRARIES = ("numpy", "scipy", "scikit-learn", "hnswlib", "threadpoolctl")
CORE_DIR = "recal/core"

COMMON = (
    "recal/accounting.py", "recal/backend.py", "recal/build.py", "recal/history.py", "recal/history_mode.py",
    "recal/ksets.py", "recal/params.py", "recal/policy.py", "recal/records.py", "recal/sim.py", "recal/stored.py",
    "recal/system.py", "recal/core/__init__.py", "recal/core/recal_core.cpp",
    "recal/micro/common.py", "recal/micro/fit.py", "recal/micro/world.py", "recal/micro/topk.py",
    "recal/micro/golden.py", "recal/micro/preflight.py", "recal/micro/units.py", "recal/micro/provenance.py",
    "scripts/micro/run.py",
)
V11 = ("recal/micro/e10.py", "recal/micro/intervals.py", "recal/micro/e4.py", "recal/micro/e3.py",
       "recal/micro/intruder_check.py", "recal/micro/quantize.py", "recal/micro/e1.py")
EXTRA = {
    "E1": ("recal/micro/e1.py",),
    "E3": ("recal/micro/e3.py", "recal/micro/quantize.py", "recal/micro/e1.py"),
    "E4": ("recal/micro/e4.py", "recal/micro/e3.py", "recal/micro/e1.py"),
    "E7": ("recal/micro/e7.py", "recal/micro/e4.py") + V11,
    "E8": ("recal/micro/e8.py", "recal/micro/e1.py"),
    "E9": ("recal/micro/e9.py", "recal/lru.py", "recal/lru_build.py", "recal/analysis/rates.py",
           "recal/analysis/guarantee.py"),
    "intruder_check": ("recal/micro/intruder_check.py", "recal/micro/e4.py", "recal/micro/e3.py",
                       "recal/micro/e1.py"),
    "E10": V11,
    "E11": V11 + ("recal/micro/e11.py",),
    "E12": V11 + ("recal/micro/e12.py",),
    "E13": V11 + ("recal/micro/e13.py",),
}


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def core_binaries(root: Path = ROOT) -> list[Path]:
    return sorted((root / CORE_DIR).glob("_recal_core*.so"))


def source_files(experiment: str) -> list[str]:
    return sorted(set(COMMON) | set(EXTRA.get(experiment, ())) | {f"recal/micro/{experiment.lower()}.py"})


def code_hash(experiment: str, root: Path = ROOT) -> str:
    """Hash of the sources the experiment uses, the core source and the compiled core."""
    h = hashlib.sha256()
    for rel in source_files(experiment):
        path = root / rel
        h.update(rel.encode())
        h.update(file_digest(path).encode() if path.exists() else b"missing")
    for binary in core_binaries(root):
        h.update(binary.name.encode())
        h.update(file_digest(binary).encode())
    return h.hexdigest()[:16]


def prep_manifest(prep: Path) -> dict:
    """sha256 of the prepared inputs the experiments read (streams, queries, exact answers)."""
    prep = Path(prep)
    return {name: file_digest(prep / name) for name in ("q.npy", "top500.npy", "streams.npz", "meta.json")
            if (prep / name).exists()}


def prep_hash(prep: Path) -> str:
    manifest = prep_manifest(prep)
    return hashlib.sha256("".join(f"{k}:{v}" for k, v in sorted(manifest.items())).encode()).hexdigest()[:16]


def library_versions() -> dict:
    out = {"python": platform.python_version()}
    for name in LIBRARIES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def runtime(workers: int = 1, jobs: int = 1, world_workers: int = 1) -> dict:
    """Recorded in every result file (not in the hash: results must not depend on the worker count)."""
    return dict(libraries=library_versions(), workers=workers, jobs=jobs, world_workers=world_workers)
