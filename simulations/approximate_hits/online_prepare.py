#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Inputs for the online simulator (online_sim.py), fixed database: ESCI 1M
(every real query once) or an Aker / SPACEV1B trace (run_e2e_oracle_trace
workloads; repeated requests).

Writes into --out:
  meta.json     dataset description (read by online_sim.load)
  q.npy         distinct query vectors (traces only; ESCI reads its own file)
  top500.npy    exact top-R database ids for every distinct query (ground
                truth and the backend's answers; R = --R)
  clusters.npy  k-means cluster of every distinct query (--n-clusters)
  streams.npz   request orders (indices into the distinct queries):
                  random  all requests in random order (no shift)
                  shift   clusters split into two groups of about equal
                          request counts; group A's requests, then group B's
                          (ESCI: random order within a group; traces: original
                          order); phase_shift = 0 / 1 per position
                  trace   original trace order (traces only); phase_trace for
                          traces made of two parts (spacev-real: log, then
                          official queries)
"""

import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from simulations.approximate_hits.run_e2e_oracle_sift import exact_topk

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
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


def write_streams(out, labels, requests, rng, natural_phase=None):
    """random / shift / trace (original order; its natural phase if given)
    streams over request sequences (indices into the distinct query
    vectors). natural_phase=False: no trace stream (ESCI)."""
    group = split_clusters(labels, rng)
    qg = group[labels[requests]]
    a, b = np.flatnonzero(qg == 0), np.flatnonzero(qg == 1)
    shift = np.r_[requests[a], requests[b]]                # original order within halves
    phase = np.r_[np.zeros(len(a), np.int8), np.ones(len(b), np.int8)]
    extra = {} if natural_phase is False else {"trace": requests}
    if natural_phase is not None and natural_phase is not False:
        extra["phase_trace"] = natural_phase
    np.savez(out / "streams.npz", random=rng.permutation(requests), shift=shift,
             phase_shift=phase, phase=phase, group=group, **extra)
    logger.info(f"shift stream: phase A {len(a)}, B {len(b)} requests")


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out", type=str, default="simulations/approximate_hits/results/online/esci1m")
    ap.add_argument("--dataset", choices=["esci", "trace"], default="esci")
    ap.add_argument("--trace", type=str, default=None,
                    help="run_e2e_oracle_trace workload name, e.g. spacev-aker-0.6")
    ap.add_argument("--n-db", type=int, default=1000000)
    ap.add_argument("--R", type=int, default=500)
    ap.add_argument("--n-clusters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    if args.dataset == "esci":
        q = np.load(ESCI / "esci_queries.npy").astype(np.float32)
        # each query once; random arrival order (also within the shift halves)
        requests, natural = rng.permutation(len(q)), False
        meta = {"kind": "esci", "n_db": args.n_db}
    else:
        from simulations.approximate_hits.run_e2e_oracle_trace import WORKLOADS, build_trace
        uniq, inv, bounds = build_trace(args.trace)
        q = uniq.astype(np.float32)
        np.save(out / "q.npy", q)
        requests = inv.astype(np.int64)
        parts = WORKLOADS[args.trace]["trace"]
        natural = None
        if len(parts) > 1:      # e.g. spacev-real: query log, then official queries
            n_first = len(_load_first(args.trace))
            natural = (np.arange(len(requests)) >= n_first).astype(np.int8)
        meta = {"kind": "trace", "trace": args.trace}
    (out / "meta.json").write_text(json.dumps(meta))

    if not (out / "streams.npz").exists():
        from sklearn.cluster import MiniBatchKMeans
        logger.info(f"k-means ({args.n_clusters} clusters) on {len(q)} distinct queries...")
        labels = MiniBatchKMeans(n_clusters=args.n_clusters, random_state=0, n_init=3,
                                 batch_size=4096).fit(q).labels_.astype(np.int64)
        np.save(out / "clusters.npy", labels)
        write_streams(out, labels, requests, rng, natural)

    if not (out / f"top{args.R}.npy").exists():
        if args.dataset == "esci":
            db = np.load(ESCI / f"esci_db{args.n_db}_s{args.seed}.npy").astype(np.float32)
            logger.info(f"exact top-{args.R} for {len(q)} queries over {len(db)} vectors...")
            ids = exact_topk(q, db, args.R)
        else:
            from simulations.approximate_hits.run_e2e_oracle_trace import load_base
            from simulations.approximate_hits.run_e2e_oracle_sift import exact_topk_blocked
            db = load_base(args.trace)
            logger.info(f"exact top-{args.R} for {len(q)} queries over {len(db)} vectors...")
            ids = exact_topk_blocked(uniq, db, args.R)
        np.save(out / f"top{args.R}.npy", ids.astype(np.int32))
    logger.info(f"wrote {out}")


def _load_first(trace):
    from simulations.approximate_hits.run_e2e_oracle_trace import AKER, WORKLOADS, _load_vectors
    return _load_vectors(AKER / WORKLOADS[trace]["trace"][0])


if __name__ == "__main__":
    main()
