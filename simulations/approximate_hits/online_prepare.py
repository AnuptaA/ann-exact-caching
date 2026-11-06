#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Inputs for the online simulator (online_sim.py), ESCI 1M, fixed database.

Writes into --out:
  top500.npy    exact top-R database ids for every ESCI query (ground truth
                and the backend's answers; R = --R)
  clusters.npy  k-means cluster of every query (--n-clusters)
  streams.npz   query orders:
                  random  all queries in random order (no shift)
                  shift   clusters split into two groups of about equal query
                          counts; group A in random order, then group B
                  phase   0 / 1 per position of the shift stream
"""

import argparse
import logging
from pathlib import Path


import numpy as np

from utils.base.logs import setup_logging

from utils.base.knn import exact_topk

logger = logging.getLogger(__name__)

ESCI = Path("datasets/esci/data_vectors")

#-------------------------------------------------------------------------------


def split_clusters(labels, rng):
    """Two groups of clusters with about equal query counts (largest first,
    each cluster to the currently smaller group)."""
    sizes = np.bincount(labels)
    group = np.zeros(len(sizes), dtype=np.int64)
    tot = [0, 0]
    for c in np.argsort(-sizes, kind="stable"):
        g = int(tot[1] < tot[0])
        group[c] = g
        tot[g] += sizes[c]
    return group


def shift_stream(labels, group, rng):
    qg = group[labels]
    a, b = np.flatnonzero(qg == 0), np.flatnonzero(qg == 1)
    order = np.r_[rng.permutation(a), rng.permutation(b)]
    phase = np.r_[np.zeros(len(a), dtype=np.int8), np.ones(len(b), dtype=np.int8)]
    return order, phase


def main():
    setup_logging()
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out", type=str, default="simulations/approximate_hits/results/online/esci1m")
    ap.add_argument("--n-db", type=int, default=1000000)
    ap.add_argument("--R", type=int, default=500)
    ap.add_argument("--n-clusters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    q = np.load(ESCI / "esci_queries.npy").astype(np.float32)

    if not (out / "clusters.npy").exists():
        from sklearn.cluster import MiniBatchKMeans
        logger.info(f"k-means ({args.n_clusters} clusters) on {len(q)} queries...")
        km = MiniBatchKMeans(n_clusters=args.n_clusters, random_state=0, n_init=3,
                             batch_size=4096).fit(q)
        labels = km.labels_.astype(np.int64)
        np.save(out / "clusters.npy", labels)
        group = split_clusters(labels, rng)
        order, phase = shift_stream(labels, group, rng)
        np.savez(out / "streams.npz", random=rng.permutation(len(q)), shift=order, phase=phase,
                 group=group)
        logger.info(f"shift stream: phase A {int((phase == 0).sum())}, B {int((phase == 1).sum())}")

    if not (out / f"top{args.R}.npy").exists():
        db = np.load(ESCI / f"esci_db{args.n_db}_s{args.seed}.npy").astype(np.float32)
        logger.info(f"exact top-{args.R} for {len(q)} queries over {len(db)} vectors...")
        ids = exact_topk(q, db, args.R).astype(np.int32)
        np.save(out / f"top{args.R}.npy", ids)
    logger.info(f"wrote {out}")


if __name__ == "__main__":
    main()
