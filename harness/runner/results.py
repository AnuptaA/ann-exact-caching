"""Results: one atomic .npz per cell with per-request fields and a JSON header."""

import json
import logging
import subprocess
from pathlib import Path

import numpy as np

from workloads.q1 import atomic

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def git_state(repo: Path | None = None) -> dict:
    run = lambda *a: subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def result_path(out_dir: Path, cell_id: str) -> Path:
    return Path(out_dir) / f"{cell_id}.npz"


def is_done(out_dir: Path, cell_id: str, commit: str | None = None, reuse_key: dict | None = None) -> bool:
    """A finished cell. A result from another commit, or made from other inputs or library
    sources (reuse_key, see harness.fingerprints.reuse_key), counts as not done."""
    path = result_path(out_dir, cell_id)
    if not path.exists():
        return False
    if commit is None and reuse_key is None:
        return True
    header = read_result(path)[0]
    if commit is not None and header.get("commit") != commit:
        logger.warning(f"{path.name} was produced by commit {str(header.get('commit'))[:8]}, "
                       f"not {commit[:8]}: redoing")
        return False
    if reuse_key is not None and header.get("reuse_key") != reuse_key:
        logger.warning(f"{path.name} was made from different inputs or library sources: redoing")
        return False
    return True


def write_result(out_dir: Path, cell_id: str, header: dict, **per_request) -> Path:
    """Header must carry config hash, commit, backend id, ef_search, library revision, reuse key."""
    required = {"cell", "config_hash", "commit", "backend", "ef_search", "aker_revision", "reuse_key"}
    missing = required - set(header)
    if missing:
        raise ValueError(f"result header is missing {sorted(missing)}")
    path = result_path(out_dir, cell_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic.save_npz_compressed(path, header=json.dumps({**header, "schema": SCHEMA_VERSION}), **per_request)
    return path


def read_result(path: Path) -> tuple[dict, dict]:
    z = np.load(path, allow_pickle=False)
    return json.loads(str(z["header"])), {k: z[k] for k in z.files if k != "header"}
