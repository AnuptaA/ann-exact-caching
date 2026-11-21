"""Write to a temporary name, then rename: readers never see a partial file."""

import json
import os
import re
import time
from pathlib import Path

import numpy as np


def _replace(path: Path, write) -> Path:
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("wb") as f:
            write(f)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def save_npy(path: Path, array) -> Path:
    return _replace(path, lambda f: np.save(f, array))


def save_npz(path: Path, **arrays) -> Path:
    return _replace(path, lambda f: np.savez(f, **arrays))


def save_npz_compressed(path: Path, **arrays) -> Path:
    return _replace(path, lambda f: np.savez_compressed(f, **arrays))


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def remove_stale_temporaries(directory: Path, legacy_age: float = 3600.0) -> list[Path]:
    """Delete temporaries left by killed writers; files of live processes are kept."""
    removed = []
    for path in Path(directory).glob("*.tmp"):
        match = re.search(r"\.(\d+)\.tmp$", path.name)
        if match and not _alive(int(match.group(1))):
            path.unlink(missing_ok=True)
            removed.append(path)
    for path in Path(directory).glob("*.tmp.npz"):
        if time.time() - path.stat().st_mtime > legacy_age:
            path.unlink(missing_ok=True)
            removed.append(path)
    return removed


def write_text(path: Path, text: str) -> Path:
    return _replace(path, lambda f: f.write(text.encode()))


def write_json(path: Path, obj, **kwargs) -> Path:
    return write_text(path, json.dumps(obj, **kwargs))
