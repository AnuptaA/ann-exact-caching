"""Shared helpers for the SPHERE query investigation (read-only, no repo writes)."""

import time
import urllib.request
from pathlib import Path

import numpy as np

AKER = "https://huggingface.co/datasets/sjoon-oh/aker/resolve/main"
HERMES = "https://huggingface.co/datasets/mohdumar/SPHERE_100K/resolve/main/data"
NPY_HEADER = 128


def range_get(url: str, start: int, end: int, retries: int = 3) -> bytes:
    """Bytes [start, end] inclusive of url via an HTTP range request."""
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(2)
    raise RuntimeError("unreachable")


def read_npy_rows(url: str, start: int, count: int, dim: int, dtype: str,
                  header: int = NPY_HEADER) -> np.ndarray:
    """Rows [start, start + count) of a 2-d .npy at url, as float32."""
    itemsize = np.dtype(dtype).itemsize
    first = header + start * dim * itemsize
    last = first + count * dim * itemsize - 1
    raw = range_get(url, first, last)
    return np.frombuffer(raw, dtype=dtype).reshape(count, dim).astype(np.float32)


def npy_header(url: str) -> dict:
    """Parse the dtype, shape and header size of a remote .npy."""
    raw = range_get(url, 0, 255)
    hlen = int.from_bytes(raw[8:10], "little")
    text = raw[10:10 + hlen].decode("ascii")
    meta = eval(text, {"__builtins__": {}}, {})  # literal dict written by numpy
    return {"dtype": meta["descr"], "shape": meta["shape"], "header": 10 + hlen}


def stratified_starts(total: int, n_blocks: int, block: int) -> list[int]:
    """Evenly spaced block start rows."""
    return [int(i * (total - block) / max(n_blocks - 1, 1)) for i in range(n_blocks)]


def read_stratified(url: str, total: int, n_blocks: int, block: int, dim: int,
                    dtype: str, header: int = NPY_HEADER) -> tuple[np.ndarray, np.ndarray]:
    """Concatenated blocks and their global row ids."""
    starts = stratified_starts(total, n_blocks, block)
    parts = [read_npy_rows(url, s, block, dim, dtype, header) for s in starts]
    ids = np.concatenate([np.arange(s, s + block) for s in starts])
    return np.vstack(parts), ids


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def relative_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)


def scratch_dir() -> Path:
    path = Path("/private/tmp/claude-501/-Users-anupta-Work-thesis/"
                "5a04402a-0702-4832-8b75-01f7a8588141/scratchpad/p1")
    path.mkdir(parents=True, exist_ok=True)
    return path
