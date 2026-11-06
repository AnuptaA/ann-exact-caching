#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Shell sample: uncached vectors just outside the cache balls, b-bit quantized."""

from pathlib import Path
import logging

import numpy as np

from utils.base.knn import exact_topk, fetch_rows, pairwise_dist, sq_norms

logger = logging.getLogger(__name__)

EPS = 1e-12
SAMPLE_J = 5
COUNT_SCALES = (1.0, 1.1, 1.25)

#-------------------------------------------------------------------------------


def quantize(x, bits):
    """b-bit uniform scalar quantization per dimension (reconstructed)."""
    if bits < 1:
        raise ValueError(f"bits must be at least 1, got {bits}")
    if bits >= 32:
        return x
    lo, hi = x.min(axis=0), x.max(axis=0)
    step = np.maximum(hi - lo, EPS) / (2 ** bits - 1)
    return lo + np.round((x - lo) / step) * step


def load_cand_ext(run, run_dir, R):
    """Exact top-R database ids per candidate query (for shells deeper than the
    cached top-500), computed once and saved to <run>/cand_top<R>.npy."""
    path = Path(run_dir) / f"cand_top{R}.npy"
    if path.exists():
        ids = np.load(path)
        if ids.shape == (len(run.cand_q), R):
            return ids
        logger.warning(f"{path} has shape {ids.shape}, expected "
                       f"{(len(run.cand_q), R)}; recomputing")
    logger.info(f"exact top-{R} for {len(run.cand_q)} candidate queries...")
    ids = exact_topk(run.cand_q, run.db, R)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, ids)
    tmp.rename(path)
    return ids


def shell_ids(run, cand, K, R):
    """Database ids at ranks K+1..R around one candidate query. Uses the saved
    deeper neighbor lists (run.cand_ext) when R exceeds the cached top-500."""
    n_kv = len(run.k_values)
    top = run.cached_ids[cand * n_kv + n_kv - 1]
    if R > len(top):
        if run.cand_ext is None or run.cand_ext.shape[1] < R:
            have = None if run.cand_ext is None else run.cand_ext.shape[1]
            raise ValueError(f"shell depth R={R} exceeds the cached top-{len(top)}; "
                             f"load_cand_ext(run, run_dir, {R}) first (cand_ext width {have})")
        return np.asarray(run.cand_ext[cand][K:R])
    return np.asarray(top[K:R])


def shell_sample(run, bl, in_cache, p, R, bits):
    shells = [shell_ids(run, i, k, R) for i, k in zip(bl["cand"], bl["K"]) if k < R]
    ids = np.unique(np.concatenate(shells)) if shells else np.empty(0, np.int64)
    ids = ids[~in_cache[ids]]
    ids = ids[run.draw[ids] < p]
    if len(ids) < SAMPLE_J:
        return None
    vecs = quantize(fetch_rows(run.db, ids), bits)
    return {"ids": ids, "vecs": vecs, "sq": sq_norms(vecs),
            "memory": len(ids) * min(bits, 32) / 32 / len(run.db)}


def shell_features(q, nr, samp):
    F = np.zeros((len(q), SAMPLE_J + len(COUNT_SCALES)))
    if samp is None:
        return F
    for s in range(0, len(q), 1024):
        d = pairwise_dist(q[s:s + 1024], samp["vecs"], samp["sq"])
        c = nr[s:s + 1024][:, None]
        near = np.sort(np.partition(d, SAMPLE_J - 1, axis=1)[:, :SAMPLE_J], axis=1)
        F[s:s + 1024, :SAMPLE_J] = np.log((near + EPS) / c)
        for i, t in enumerate(COUNT_SCALES):
            F[s:s + 1024, SAMPLE_J + i] = np.log1p((d < t * c).sum(axis=1))
    return F
