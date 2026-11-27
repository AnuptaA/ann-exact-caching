"""What a result was computed from: a hash of the code each experiment uses, of the prepared inputs and of the libraries.

The hash replaces the commit in the config hash. A commit that changes no file an experiment uses keeps its units
valid, and an edit that is not committed (or a rebuilt core) changes it. PROVENANCE_VERSION is in every hash.
"""

import ast
import hashlib
import importlib.metadata
import platform
from pathlib import Path

import recal

ROOT = Path(recal.__file__).resolve().parents[1]
LIBRARIES = ("numpy", "scipy", "scikit-learn", "hnswlib", "threadpoolctl")
CORE_DIR = "recal/core"

PROVENANCE_VERSION = 1      # bump by hand when the meaning of a result changes without a source change
CORE_SOURCE = "recal/core/recal_core.cpp"


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def core_binaries(root: Path = ROOT) -> list[Path]:
    return sorted((root / CORE_DIR).glob("_recal_core*.so"))


def _module_file(root: Path, module: str) -> str | None:
    base = Path(*module.split("."))
    for rel in (base.with_suffix(".py"), base / "__init__.py"):
        if (root / rel).is_file():
            return rel.as_posix()
    return None


def _imports(root: Path, rel: str) -> set[str]:
    """Modules under recal/ that the file imports (every import statement, including those inside functions)."""
    tree = ast.parse((root / rel).read_text(), filename=rel)
    package = rel.split("/")[:-1]
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = ".".join(package[:len(package) - node.level + 1]) if node.level else ""
            module = ".".join(part for part in (base, node.module) if part)
            names = [module] + [f"{module}.{alias.name}" for alias in node.names]
        else:
            continue
        for name in names:
            parts = name.split(".")
            found.update(".".join(parts[:i]) for i in range(1, len(parts) + 1) if parts[0] == "recal")
    return found


def source_files(experiment: str, root: Path = ROOT) -> list[str]:
    """The experiment's module and every recal module it imports, directly or not, plus the core source. The runner
    and this registry are not imported by experiments, so adding an experiment changes no other hash."""
    entry = f"recal/micro/{experiment.lower()}.py"
    files = {CORE_SOURCE}
    todo = [entry] if (root / entry).is_file() else []
    files.add(entry)
    while todo:
        rel = todo.pop()
        for module in _imports(root, rel):
            path = _module_file(root, module)
            if path is not None and path not in files:
                files.add(path)
                todo.append(path)
    return sorted(files)


def code_hash(experiment: str, root: Path = ROOT) -> str:
    """Hash of the experiment's import closure under recal/, the core source, the compiled core and
    PROVENANCE_VERSION."""
    h = hashlib.sha256(f"provenance_version:{PROVENANCE_VERSION}".encode())
    for rel in source_files(experiment, root):
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
