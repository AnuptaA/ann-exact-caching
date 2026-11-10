#!/usr/bin/env python3
"""
Inner-product -> Euclidean transform of the SPHERE / TriviaQA workload.

SPHERE's encoder (DPR_web) is trained for maximum inner product, but Aker (and
this repo) search by Euclidean distance, where passage norms enter the distance
and neighbors are poorly separated (median d10 / d1000 = .967). The standard
fix: append sqrt(M^2 - |x|^2) to every database vector x (M = the largest
database norm) and 0 to every query q. Then |q' - x'|^2 = |q|^2 + M^2 - 2 q.x,
so the Euclidean nearest neighbors of q' are exactly the inner-product
nearest neighbors of q, and every distance-based mechanism (caches, shells,
Aker's thresholds, DiskANN with L2) keeps working unchanged.

Writes (float32, 769-d):
  datasets/aker/sphere-ip-10m.npy        database (+ a symlink sphere-ip-10m-f32.npy so the
                                         simulators memory-map it instead of copying)
  datasets/aker/triviaqa-ip-sim-100k-<z>.npy   the DPR_web question vectors of
                                         triviaqa-dpr-sim-100k-<z>.npy with a 0 appended
  datasets/aker/sphere-ip-10m.json       M and norm statistics

  .venv/bin/python datasets/make_sphere_ip_transform.py
"""

import os
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
AKER = Path("datasets/aker")
BLOCK = 500_000


def main():
    src = np.load(AKER / "sphere-10m-f32.npy", mmap_mode="r")
    n, d = src.shape
    sq = np.empty(n, dtype=np.float64)
    for b in range(0, n, BLOCK):
        x = np.asarray(src[b:b + BLOCK], dtype=np.float64)
        sq[b:b + len(x)] = np.einsum("ij,ij->i", x, x)
    m2 = float(sq.max())
    stats = {"n": n, "dim": d, "M": float(np.sqrt(m2)), "norm_min": float(np.sqrt(sq.min())),
             "norm_median": float(np.sqrt(np.median(sq))), "norm_max": float(np.sqrt(m2))}
    logger.info(json.dumps(stats))
    tmp = AKER / "sphere-ip-10m.npy.tmp"
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n, d + 1))
    for b in range(0, n, BLOCK):
        out[b:b + BLOCK, :d] = src[b:b + BLOCK]
        out[b:b + BLOCK, d] = np.sqrt(np.maximum(m2 - sq[b:b + BLOCK], 0.0)).astype(np.float32)
        if (b // BLOCK) % 4 == 0:
            logger.info(f"  {b + BLOCK} / {n}")
    out.flush()
    del out
    os.replace(tmp, AKER / "sphere-ip-10m.npy")
    link = AKER / "sphere-ip-10m-f32.npy"
    if not link.exists():
        link.symlink_to("sphere-ip-10m.npy")
    (AKER / "sphere-ip-10m.json").write_text(json.dumps(stats, indent=1))
    for z in ("0.3", "0.6", "0.99"):
        q = np.load(AKER / f"triviaqa-dpr-sim-100k-{z}.npy")
        np.save(AKER / f"triviaqa-ip-sim-100k-{z}.npy",
                np.concatenate([q, np.zeros((len(q), 1), dtype=np.float32)], axis=1))
        logger.info(f"wrote triviaqa-ip-sim-100k-{z}.npy {q.shape[0]} x {d + 1}")
    # check on a few queries: Euclidean neighbors of q' = inner-product neighbors of q
    new = np.load(AKER / "sphere-ip-10m.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    base = np.ascontiguousarray(new[:300000])
    q = np.load(AKER / "triviaqa-ip-sim-100k-0.99.npy")[rng.choice(99000, 50, replace=False)]
    ip = q[:, :d] @ base[:, :d].T
    l2 = (q * q).sum(1)[:, None] + (base * base).sum(1)[None] - 2.0 * (q @ base.T)
    agree = np.mean([np.array_equal(np.argsort(-ip[i])[:10], np.argsort(l2[i])[:10]) for i in range(len(q))])
    logger.info(f"top-10 by inner product == top-10 by Euclidean (transformed), 300k rows: {agree:.2f} of queries")


if __name__ == "__main__":
    main()
