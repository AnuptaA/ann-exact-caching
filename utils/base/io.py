#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Vector file readers."""

from typing import Optional

import numpy as np

#-------------------------------------------------------------------------------


def load_fvecs_fast(filepath: str, max_n: Optional[int] = None) -> np.ndarray:
    """Read an fvecs file (int32 dim, then dim float32 per vector); at most max_n vectors."""
    with open(filepath, "rb") as f:
        dim = int(np.fromfile(f, dtype=np.int32, count=1)[0])
    row = dim + 1
    count = -1 if max_n is None else max_n * row
    raw = np.fromfile(filepath, dtype=np.int32, count=count).reshape(-1, row)
    return np.ascontiguousarray(raw[:, 1:].view(np.float32))
