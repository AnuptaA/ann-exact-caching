#!/usr/bin/env python3
"""The SPHERE inner-product corpus (769-d float32) from Aker's sphere-10m.npy (768-d float64).

Every vector x gets sqrt(M^2 - |x|^2) appended, M the largest norm of the float32 vectors, so Euclidean
nearest neighbors equal inner-product nearest neighbors. M must match the one published with the TriviaQA
requests (sphere-ip-10m.json).
"""

import json
import os
from pathlib import Path

import numpy as np

BLOCK = 500_000
M_TOLERANCE = 1e-6


def build(src, out: Path, block: int = BLOCK) -> dict:
    """Write the 769-d corpus to out; returns the norm statistics (M included)."""
    n, d = src.shape
    sq = np.empty(n, dtype=np.float64)
    for b in range(0, n, block):
        x = np.asarray(src[b:b + block], dtype=np.float32).astype(np.float64)
        sq[b:b + len(x)] = np.einsum("ij,ij->i", x, x)
    m2 = float(sq.max())
    stats = {"n": n, "dim": d, "M": float(np.sqrt(m2)), "norm_min": float(np.sqrt(sq.min())),
             "norm_median": float(np.sqrt(np.median(sq))), "norm_max": float(np.sqrt(m2))}
    tmp = out.with_name(out.name + ".tmp")
    target = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n, d + 1))
    for b in range(0, n, block):
        target[b:b + block, :d] = np.asarray(src[b:b + block], dtype=np.float32)
        target[b:b + block, d] = np.sqrt(np.maximum(m2 - sq[b:b + block], 0.0)).astype(np.float32)
    target.flush()
    del target
    os.replace(tmp, out)
    return stats


def check_m(stats: dict, published: Path) -> None:
    expected = json.loads(published.read_text())["M"]
    if abs(stats["M"] - expected) > M_TOLERANCE * expected:
        raise SystemExit(f"M of the built corpus {stats['M']} differs from the published {expected}: "
                         "the base file is not the one the requests were encoded against")


def build_from_files(base_path: Path, out: Path, published_json: Path) -> dict:
    stats = build(np.load(base_path, mmap_mode="r"), out)
    check_m(stats, published_json)
    return stats
