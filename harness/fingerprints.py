"""Content fingerprints of generated inputs and of the code that replays them.

Generation stores them in <dataset>/fingerprints.json; replay recomputes them and refuses to run
on a mismatch, and results are reused only when they still match.
"""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from workloads.q1 import atomic

FILE = "fingerprints.json"
BLOCK = 8 * 2 ** 20
SAMPLE_ROWS = 4096
CODE_PATHS = {"aker_tree": "third_party/aker", "driver_src_tree": "harness/cpp"}


class FingerprintMismatch(RuntimeError):
    """Generated inputs or code differ from what generation recorded."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(BLOCK):
            digest.update(block)
    return digest.hexdigest()


def sha256_or_none(path: Path) -> str | None:
    return sha256_file(path) if Path(path).exists() else None


def base_signature(path: Path, rows: int = SAMPLE_ROWS) -> dict:
    """Size, header and a hash of evenly spaced rows: cheap on a 30 GB file."""
    from harness import data
    base = data.read_f32bin(path)
    picks = np.unique(np.linspace(0, len(base) - 1, min(rows, len(base))).astype(np.int64))
    sample = hashlib.sha256(np.ascontiguousarray(base[picks]).tobytes()).hexdigest()
    return {"n": int(base.shape[0]), "d": int(base.shape[1]), "bytes": Path(path).stat().st_size,
            "sample_sha256": sample}


def tree_listing(directory: Path, pattern: str) -> str:
    """Hash of sorted (name, size) pairs, for directories of many large chunk files."""
    entries = [(p.name, p.stat().st_size) for p in sorted(Path(directory).glob(pattern))]
    return hashlib.sha256(json.dumps(entries).encode()).hexdigest()


def answers_fingerprint(root: Path) -> str | None:
    first10 = root / "answers_first10.npz"
    if first10.exists():
        return sha256_file(first10)
    answers = root / "answers"
    if not answers.exists():
        return None
    digest = hashlib.sha256()
    for path in sorted(answers.glob("block_*.npz")):
        digest.update(path.name.encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def dataset_fingerprints(root: Path, seed: int = 0) -> dict:
    root = Path(root)
    return {"base": base_signature(root / "base.f32bin"),
            "official": sha256_file(root / "official.f32bin"),
            "universe": sha256_file(root / "universe.npz"),
            "streams": sha256_file(root / f"streams_seed{seed}.npz"),
            "answers": answers_fingerprint(root),
            "answers_meta": sha256_or_none(root / "answers" / "meta.json"),
            "tune": sha256_or_none(root / "tune.json"),
            "ground_truth": {"meta": sha256_or_none(root / "gt" / "meta.json"),
                             "chunks": tree_listing(root / "gt", "chunk_*.npz")}}


def combine(fingerprints: dict, keys=("base", "official", "universe", "streams", "answers", "answers_meta")) -> str:
    """Short id of the inputs that decide a replay, for naming caches."""
    blob = json.dumps({k: fingerprints[k] for k in keys}, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def store(root: Path, seed: int = 0) -> dict:
    """Record fingerprints of everything generated so far (merging other seeds' streams)."""
    path = Path(root) / FILE
    stored = json.loads(path.read_text()) if path.exists() else {}
    stored[str(seed)] = dataset_fingerprints(root, seed)
    atomic.write_json(path, stored, indent=1)
    return stored[str(seed)]


def verify(root: Path, seed: int = 0) -> dict:
    """Stored fingerprints of this seed if the files on disk still match them."""
    path = Path(root) / FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: finish `make q1-generate` for this dataset")
    stored = json.loads(path.read_text()).get(str(seed))
    if stored is None:
        raise FingerprintMismatch(f"{path} has no entry for stream seed {seed}")
    current = dataset_fingerprints(root, seed)
    changed = sorted(k for k in current if current[k] != stored.get(k))
    if changed:
        raise FingerprintMismatch(f"{root}: {changed} changed since generation finished; "
                                  "regenerate (a later stage with --force) before replaying")
    return stored


def _git(repo: Path, *args: str) -> str | None:
    out = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


def code_fingerprints(repo: Path, driver_dir: Path | None = None) -> dict:
    """Git tree ids of the vendored library (with its patches) and of the driver sources.

    With a driver directory, also the binaries' checksums, and the check that a recorded
    BUILT_FROM_COMMIT has the same trees as HEAD.
    """
    out = {name: _git(repo, "rev-parse", f"HEAD:{path}") for name, path in CODE_PATHS.items()}
    if driver_dir is not None:
        driver_dir = Path(driver_dir)
        out["driver_sha256"] = {mode: sha256_or_none(driver_dir / f"ak_replay_{mode}")
                                for mode in ("standard", "proximity")}
        built = driver_dir / "BUILT_FROM_COMMIT"
        if built.exists():
            commit = built.read_text().strip()
            for name, path in CODE_PATHS.items():
                if _git(repo, "rev-parse", f"{commit}:{path}") != out[name]:
                    raise FingerprintMismatch(f"{driver_dir} was built from {commit[:8]}, whose {path} "
                                              "differs from this checkout: rebuild the drivers")
    return out


def reuse_key(fingerprints: dict, code: dict) -> dict:
    """What a stored result must match to be reused."""
    return {"inputs": combine(fingerprints), "tune": fingerprints["tune"],
            "aker_tree": code["aker_tree"], "driver_src_tree": code["driver_src_tree"]}
