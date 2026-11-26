"""Scalar quantization: query vectors (per-vector minimum and scale) and the core's shell codes."""

import numpy as np

SHELL_LEVELS = 15


def shell_codes(X: np.ndarray, lo: np.ndarray, step: np.ndarray) -> np.ndarray:
    """4-bit codes as recal_core.cpp quantize_into computes them: float64, ties away from zero, clipped.

    Integer data (SPACEV int8) puts many coordinates exactly on a level midpoint, so float32 arithmetic or
    numpy's round-half-to-even give different codes there.
    """
    v = (np.asarray(X, dtype=np.float32).astype(np.float64) - np.asarray(lo, dtype=np.float32).astype(np.float64)) \
        / np.asarray(step, dtype=np.float32).astype(np.float64)
    whole = np.trunc(v)
    rounded = whole + np.where(np.abs(v - whole) >= 0.5, np.sign(v), 0.0)
    return np.clip(rounded, 0, SHELL_LEVELS).astype(np.uint8)


def quantize(X: np.ndarray, bits: int) -> np.ndarray:
    """Round-trip each row through `bits` bits; returns float32 with the same shape."""
    X = np.asarray(X, dtype=np.float32)
    levels = (1 << bits) - 1
    lo = X.min(axis=1, keepdims=True)
    span = X.max(axis=1, keepdims=True) - lo
    scale = np.where(span > 0, span / levels, 1.0)
    codes = np.clip(np.rint((X - lo) / scale), 0, levels)
    return (codes * scale + lo).astype(np.float32)
