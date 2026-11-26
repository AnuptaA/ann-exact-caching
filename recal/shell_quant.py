"""The core's 4-bit shell quantization, in one place (the core, the LRU cache and the experiments must agree).

recal_core.cpp: step = (float) max(hi - lo, EPS) / LEVELS computed in float64; code = clamp(round((x - lo) / step),
0, LEVELS) in float64 with std::round (ties away from zero). Integer data (SPACEV int8) puts many coordinates exactly
on a level midpoint, so float32 arithmetic or numpy's round-half-to-even give different codes there.
"""

import numpy as np

SHELL_LEVELS = 15
EPS = 1e-12


def shell_step(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Quantizer step per coordinate, as Core.set_quantizer computes it (float32 result)."""
    span = np.maximum(np.asarray(hi, np.float32).astype(np.float64) - np.asarray(lo, np.float32).astype(np.float64),
                      EPS)
    return (span / SHELL_LEVELS).astype(np.float32)


def default_step(dim: int) -> np.ndarray:
    """The core's step before set_quantizer is called (range [0, 1])."""
    return np.full(dim, 1.0 / SHELL_LEVELS, np.float32)


def shell_codes(X: np.ndarray, lo: np.ndarray, step: np.ndarray) -> np.ndarray:
    """4-bit codes as recal_core.cpp quantize_into computes them: float64, ties away from zero, clipped."""
    v = (np.asarray(X, dtype=np.float32).astype(np.float64) - np.asarray(lo, dtype=np.float32).astype(np.float64)) \
        / np.asarray(step, dtype=np.float32).astype(np.float64)
    whole = np.trunc(v)
    rounded = whole + np.where(np.abs(v - whole) >= 0.5, np.sign(v), 0.0)
    return np.clip(rounded, 0, SHELL_LEVELS).astype(np.uint8)
