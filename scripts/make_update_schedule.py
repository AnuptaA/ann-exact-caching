#!/usr/bin/env python3
"""Write an update schedule (initial presence, inserts and deletes over the test window).

  python scripts/make_update_schedule.py --base base.npy --queries q.npy --stream-ids s.npy --fraction 0.05 \\
      --mode targeted --first 30000 --last 100000 --out data_targeted5.npz
"""

import argparse
from pathlib import Path

import numpy as np

from recal.update_schedule import build_schedule
from workloads.q1 import shifts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--stream-ids", type=Path, required=True, help="query index of every request")
    ap.add_argument("--fraction", type=float, required=True)
    ap.add_argument("--mode", choices=["uniform", "targeted"], default="uniform")
    ap.add_argument("--first", type=int, required=True)
    ap.add_argument("--last", type=int, required=True)
    ap.add_argument("--sample", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    base = np.load(a.base, mmap_mode="r")
    rng = np.random.default_rng(a.seed)
    sample = np.asarray(base[np.sort(rng.choice(len(base), min(a.sample, len(base)), replace=False))],
                        dtype=np.float32)
    centroids = shifts.kmeans_centroids(sample)
    vector_cluster = shifts.assign(base, centroids)
    queries = np.load(a.queries, mmap_mode="r")
    used = np.unique(np.load(a.stream_ids))
    counts = np.bincount(shifts.assign(np.asarray(queries[used], dtype=np.float32), centroids),
                         minlength=len(centroids))
    schedule = build_schedule(vector_cluster, counts, a.fraction, a.first, a.last, a.mode, seed=a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **schedule)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
