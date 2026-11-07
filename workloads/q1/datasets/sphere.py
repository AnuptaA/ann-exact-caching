"""SPHERE 10M: Aker's released sphere-10m.npy, used as is.

float64, 768-d, not normalized, on disk (about 61 GB); indexes and ground truth
use float32 copies made chunk by chunk.
"""

from pathlib import Path

import numpy as np

from workloads.q1.datasets import common

BASE_URL = "https://huggingface.co/datasets/sjoon-oh/aker/resolve/main/sphere-10m.npy"
N_BASE = 10_000_000
DIM = 768
EXPECTED_BYTES = 61_440_000_128


def fetch(out_dir: Path, base_url: str = BASE_URL) -> Path:
    out_dir = Path(out_dir)
    common.check_free_disk(out_dir, EXPECTED_BYTES)
    return common.download_file(base_url, out_dir / "sphere-10m.npy")


def verify(base: np.ndarray, n_base: int = N_BASE) -> None:
    if base.dtype != np.float64 or base.shape != (n_base, DIM):
        raise ValueError(f"base {base.dtype}{base.shape}, expected float64 {(n_base, DIM)}")


def load(out_dir: Path) -> np.ndarray:
    return np.load(Path(out_dir) / "sphere-10m.npy", mmap_mode="r")
