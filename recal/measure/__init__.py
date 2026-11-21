"""Helpers shared by the throughput and timing measurement scripts."""
import json
import os
from pathlib import Path

import numpy as np


def common_sample(n_total: int, n: int, seed: int) -> np.ndarray:
    """Sorted offsets in [0, n_total): n distinct values, the same for every caller with the same arguments.
    Samples for a smaller n are subsets of those for a larger n."""
    if n >= n_total:
        return np.arange(n_total, dtype=np.int64)
    order = np.random.default_rng(seed).permutation(n_total)
    return np.sort(order[:n]).astype(np.int64)


def write_json_atomic(path, obj, indent=1) -> None:
    """Write to a temp file in the same directory, then os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=indent))
    os.replace(tmp, path)
