"""Scalar quantization: query vectors (per-vector minimum and scale) and the core's shell codes."""

import numpy as np

from recal.shell_quant import SHELL_LEVELS, shell_codes  # noqa: F401  (the core's shell rounding lives there)

def quantize(X: np.ndarray, bits: int) -> np.ndarray:
    """Round-trip each row through `bits` bits; returns float32 with the same shape."""
    X = np.asarray(X, dtype=np.float32)
    levels = (1 << bits) - 1
    lo = X.min(axis=1, keepdims=True)
    span = X.max(axis=1, keepdims=True) - lo
    scale = np.where(span > 0, span / levels, 1.0)
    codes = np.clip(np.rint((X - lo) / scale), 0, levels)
    return (codes * scale + lo).astype(np.float32)
