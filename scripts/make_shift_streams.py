#!/usr/bin/env python3
"""Abrupt, gradual and correlated shift streams over k-means clusters (64 clusters, seed 0).

  python scripts/make_shift_streams.py --official q.npy --base base.npy --n 100000 --start 30000 --span 35000 \\
      --seeds 0 1 2 3 4 --out-dir shifts/

Writes shift_<kind>_seed<k>.npz (request ids, phase) and, for the correlated shift, the insert schedule of
held-out vectors of the new clusters (the same layout as recal/update_schedule.py).
"""

import argparse
from pathlib import Path

import numpy as np

from workloads.q1 import shifts
from workloads.q1.universe import build_universe


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--official", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--start", type=int, default=30_000)
    ap.add_argument("--span", type=int, default=35_000)
    ap.add_argument("--groups", type=int, default=10_000)
    ap.add_argument("--group-size", type=int, default=50)
    ap.add_argument("--held-out", type=float, default=0.05, help="share of base vectors held out for inserts")
    ap.add_argument("--n-inserts", type=int, default=0, help="correlated inserts (default: all held-out new-cluster ones)")
    ap.add_argument("--sample", type=int, default=100_000)
    ap.add_argument("--allow-numpy-kmeans", action="store_true", help="tests only: numpy k-means without faiss")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    official = np.load(a.official, mmap_mode="r")
    base = np.load(a.base, mmap_mode="r")
    rng = np.random.default_rng(shifts.SEED)
    sample = np.asarray(base[np.sort(rng.choice(len(base), min(a.sample, len(base)), replace=False))],
                        dtype=np.float32)
    centroids = shifts.kmeans_centroids(sample, allow_fallback=a.allow_numpy_kmeans)
    n_clusters = len(centroids)
    vector_cluster = shifts.assign(base, centroids)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for seed in a.seeds:
        uni = build_universe(len(official), a.groups, a.group_size, seed)
        group_cluster = shifts.assign(np.asarray(official[uni.base_idx], dtype=np.float32), centroids)
        for kind in ("abrupt", "gradual", "correlated"):
            out = shifts.shift_stream(uni, group_cluster, kind, a.n, a.start, a.span, seed=seed,
                                      n_clusters=n_clusters)
            np.savez(a.out_dir / f"shift_{kind}_seed{seed}.npz", **out)
        held = np.random.default_rng([seed, 61]).choice(len(base), int(a.held_out * len(base)), replace=False)
        _, new = shifts.cluster_sets(n_clusters)
        available = int(np.isin(vector_cluster[held], new).sum())
        ins = shifts.correlated_inserts(vector_cluster, held, a.start, a.span, a.n_inserts or available,
                                        n_clusters, seed)
        initial = np.ones(len(base), dtype=bool)
        initial[held] = False
        np.savez(a.out_dir / f"shift_correlated_inserts_seed{seed}.npz", initial=initial, t=ins["t"],
                 insert=np.ones(len(ins["t"]), dtype=bool), id=ins["id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
