"""Float32 .f32bin files: int32 (n, d) header, then row-major float32 rows."""

from pathlib import Path

import numpy as np

from workloads.q1.datasets.common import bin_header

CHUNK_ROWS = 100_000


def write_f32bin(path: Path, rows, chunk_rows: int = CHUNK_ROWS) -> Path:
    """Convert any (n, d) array (uint8, int8, float64, memmap) to float32, chunk by chunk."""
    path = Path(path)
    tmp = path.with_name(path.name + ".part")
    n, d = rows.shape
    with tmp.open("wb") as f:
        f.write(bin_header(n, d))
        for start in range(0, n, chunk_rows):
            f.write(np.ascontiguousarray(rows[start:start + chunk_rows], dtype=np.float32).tobytes())
    tmp.rename(path)
    return path


def read_f32bin(path: Path) -> np.ndarray:
    path = Path(path)
    header = np.fromfile(path, dtype="<i4", count=2)
    if len(header) != 2:
        raise ValueError(f"{path}: shorter than the 8-byte header")
    n, d = int(header[0]), int(header[1])
    expected = 8 + n * d * 4
    if n < 0 or d < 0 or path.stat().st_size != expected:
        raise ValueError(f"{path}: {path.stat().st_size} bytes, header (n={n}, d={d}) says {expected}")
    return np.memmap(path, dtype=np.float32, mode="r", offset=8, shape=(n, d))
