"""SIFT-10M: the first 10M vectors of SIFT-1B plus its 10K official queries.

Source: the big-ann-benchmarks mirror. The .u8bin base has an (n, d) int32 header,
so the first n rows are one byte range and the 1B file is never downloaded.
"""

from pathlib import Path

import numpy as np

from workloads.q1.datasets import common

BASE_URL = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin"
QUERY_URL = ("https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/"
             "query.public.10K.u8bin")
N_BASE = 10_000_000
N_QUERY = 10_000
DIM = 128


def fetch(out_dir: Path, n_base: int = N_BASE, base_url: str = BASE_URL,
          query_url: str = QUERY_URL) -> tuple[Path, Path]:
    """Download the base slice and the query file as .u8bin files."""
    out_dir = Path(out_dir)
    n_total, dim = np.frombuffer(common.fetch_range(base_url, 0, 8), dtype="<i4")
    if dim != DIM or n_base > n_total:
        raise ValueError(f"unexpected source header (n={n_total}, d={dim})")
    base = common.download_slice(base_url, out_dir / "sift10m_base.u8bin", 8, 8 + n_base * dim,
                                 header=common.bin_header(n_base, dim))
    common.ensure_verified(base, base_url, use_published=False)
    query = common.download_file(query_url, out_dir / "sift_query.u8bin")
    return base, query


def verify(base: np.ndarray, queries: np.ndarray, n_base: int = N_BASE) -> None:
    if base.shape != (n_base, DIM):
        raise ValueError(f"base shape {base.shape}, expected {(n_base, DIM)}")
    if queries.shape != (N_QUERY, DIM):
        raise ValueError(f"query shape {queries.shape}, expected {(N_QUERY, DIM)}")


def load(out_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """uint8 memory maps (base, queries); cast chunks to float32 at use."""
    out_dir = Path(out_dir)
    return (common.memmap_bin(out_dir / "sift10m_base.u8bin", np.uint8),
            common.memmap_bin(out_dir / "sift_query.u8bin", np.uint8))
