"""The single distance function: float32 Euclidean, computed from differences.

Ground truth, tie comparison, recall scoring and the distances handed to Aker
all go through these functions. Aker receives the squared variant.
"""

import numpy as np

CHUNK = 65536


def sq_l2_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance between matching rows of a and b (float32)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    out = np.empty(len(a), dtype=np.float32)
    for start in range(0, len(a), CHUNK):
        diff = a[start:start + CHUNK] - b[start:start + CHUNK]
        out[start:start + CHUNK] = np.einsum("ij,ij->i", diff, diff)
    return out


def l2_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(sq_l2_rows(a, b))


def l2_to_ids(query: np.ndarray, base: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Euclidean distance from one query to base rows `ids`."""
    rows = np.asarray(base[np.asarray(ids)], dtype=np.float32)
    return l2_rows(np.broadcast_to(np.asarray(query, dtype=np.float32), rows.shape), rows)
