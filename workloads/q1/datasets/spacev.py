"""SPACEV 10M: Aker's released spacev-10m.npy plus the official 29,316 queries.

The base is int8, 100-d, used as is; indexes and ground truth cast to float32.
The official query set comes from the big-ann-benchmarks SPACEV1B release.
"""

from pathlib import Path

import numpy as np

from workloads.q1.datasets import common

BASE_URL = "https://huggingface.co/datasets/sjoon-oh/aker/resolve/main/spacev-10m.npy"
QUERY_URL = "https://comp21storage.z5.web.core.windows.net/comp21/spacev1b/query.i8bin"
N_BASE = 10_000_000
N_QUERY = 29_316
DIM = 100


def fetch(out_dir: Path, base_url: str = BASE_URL, query_url: str = QUERY_URL) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    base = common.download_file(base_url, out_dir / "spacev-10m.npy")
    query = common.download_file(query_url, out_dir / "spacev_query.i8bin")
    return base, query


def verify(base: np.ndarray, queries: np.ndarray, n_base: int = N_BASE) -> None:
    if base.dtype != np.int8 or base.shape != (n_base, DIM):
        raise ValueError(f"base {base.dtype}{base.shape}, expected int8 {(n_base, DIM)}")
    if queries.shape != (N_QUERY, DIM):
        raise ValueError(f"query shape {queries.shape}, expected {(N_QUERY, DIM)}")


def load(out_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    out_dir = Path(out_dir)
    return (np.load(out_dir / "spacev-10m.npy", mmap_mode="r"),
            common.memmap_bin(out_dir / "spacev_query.i8bin", np.int8))
