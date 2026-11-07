"""Results: one atomic .npz per cell with per-request fields and a JSON header."""

import json
import logging
import subprocess
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def git_state(repo: Path | None = None) -> dict:
    run = lambda *a: subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def result_path(out_dir: Path, cell_id: str) -> Path:
    return Path(out_dir) / f"{cell_id}.npz"


def is_done(out_dir: Path, cell_id: str, commit: str | None = None) -> bool:
    """A finished cell; with commit, a result from another commit counts as not done."""
    path = result_path(out_dir, cell_id)
    if not path.exists():
        return False
    if commit is None:
        return True
    stored = read_result(path)[0].get("commit")
    if stored != commit:
        logger.warning(f"{path.name} was produced by commit {str(stored)[:8]}, not {commit[:8]}: redoing")
        return False
    return True


def write_result(out_dir: Path, cell_id: str, header: dict, **per_request) -> Path:
    """Header must carry config hash, commit, backend id, ef_search, library revision."""
    required = {"cell", "config_hash", "commit", "backend", "ef_search", "aker_revision"}
    missing = required - set(header)
    if missing:
        raise ValueError(f"result header is missing {sorted(missing)}")
    path = result_path(out_dir, cell_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, header=json.dumps({**header, "schema": SCHEMA_VERSION}), **per_request)
    tmp.rename(path)
    return path


def read_result(path: Path) -> tuple[dict, dict]:
    z = np.load(path, allow_pickle=False)
    return json.loads(str(z["header"])), {k: z[k] for k in z.files if k != "header"}
