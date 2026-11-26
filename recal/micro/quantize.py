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


NATIVE_SAMPLE = 20_000


def native_info(X: np.ndarray, sample: int = NATIVE_SAMPLE) -> dict:
    """What a vector file really holds: the stored dtype, whether the values are integers inside the int8 range
    (SPACEV's int8 vectors are saved as float32), and so how many bits per coordinate hold them exactly."""
    n = len(X)
    rows = np.asarray(X[np.linspace(0, n - 1, min(n, sample)).astype(np.int64)]) if n else np.zeros((0, 1))
    integer = bool(rows.size) and bool(np.all(rows == np.rint(rows)))
    lo, hi = (float(rows.min()), float(rows.max())) if rows.size else (float("nan"), float("nan"))
    fits_int8 = integer and lo >= -128 and hi <= 127
    return dict(dtype=str(np.asarray(X[:1]).dtype), integer_valued=integer, min=lo, max=hi, rows_checked=int(len(rows)),
                native_bits=8 if fits_int8 else 32)
