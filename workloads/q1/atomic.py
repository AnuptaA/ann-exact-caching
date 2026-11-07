"""Write to a temporary name, then rename: readers never see a partial file."""

import json
import os
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


def write_text(path: Path, text: str) -> Path:
    return _replace(path, lambda f: f.write(text.encode()))


def write_json(path: Path, obj, **kwargs) -> Path:
    return write_text(path, json.dumps(obj, **kwargs))
