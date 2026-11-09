#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Timing study (context doc 5ac): per-query latency of a disk-based backend
alone vs with our cache in front of it. Run with .venv-ann (diskannpy).

Everything is single-threaded and one query at a time (BLAS threads 1,
DiskANN 1 thread), on a cache state saved by online_sim.py --save-state
after warm-up. Per query, the hit decision (the online hot path):
  knn     squared distances to every cached vector, the k nearest
          (neighborhood radius = distance to the k-th)
  balls   distances to every entry center; fit = (radius + offset) / ball
          radius; relative offset and size for the 3 best-fitting balls
  shell   distances to every vector of the shell sample (b-bit, as stored);
          count of those inside the neighborhood radius
  model   logistic score from the 7 features; hit iff score >= lambda
On a hit the answer is the k nearest cached ids; on a miss the backend
searches (DiskANN, k results, search list --L). Arms, on the same queries:
  backend        DiskANN alone (node cache off)
  cache          our cache + DiskANN on misses (node cache off)
  backend_mem    DiskANN alone with its node cache given our cache's memory
                 (equal total memory; bytes per cached node = full vector +
                 neighbor list)
The per-query scores are checked against the simulator's own scores for
the same queries (saved in the state). Audits (a backend call for a
sampled share of hits) are off the critical path and not timed here.
Writes results/timing/<prep>/timing_<state>.json.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import time
import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

RES = Path("simulations/approximate_hits/results")
ESCI = Path("datasets/esci/data_vectors")
N_BALLS = 3
POOR_FIT = 10.0
EPS = 1e-12

#-------------------------------------------------------------------------------


class HitPath:
    """The hit decision for one query, stage by stage (timed)."""

    def __init__(self, db, q, st):
        self.k = int(st["k"])
        self.cv = np.ascontiguousarray(db[st["cached_ids"]], dtype=np.float32)
        self.cv_sq = (self.cv * self.cv).sum(axis=1)
        self.ids = st["cached_ids"]
        self.cq = np.ascontiguousarray(q[st["centers"]], dtype=np.float32)
        self.cq_sq = (self.cq * self.cq).sum(axis=1)
        self.radius = st["radius"].astype(np.float64)
        self.sv = np.ascontiguousarray(st["sv"], dtype=np.float32)
        self.sv_sq = (self.sv * self.sv).sum(axis=1)
        self.mu, self.sd = st["mu"], st["sd"]
        self.coef, self.b = st["coef"], float(st["intercept"])
        self.lam = float(st["lam"])
        # bytes held in memory for the hit decision and the answers
        self.bytes = {"cached_vectors": self.cv.nbytes + self.ids.nbytes,
                      "centers_and_radii": self.cq.nbytes + self.radius.nbytes,
                      "shell_sample_as_stored": int(len(self.sv) * self.sv.shape[1]
                                                    * int(st["bits"]) / 8),
                      "norms": self.cv_sq.nbytes + self.cq_sq.nbytes + self.sv_sq.nbytes}

    def decide(self, x, t):
        """Score and hit decision for query vector x; adds stage times
        (seconds) to dict t."""
        k = self.k
        x_sq = float(x @ x)
        a = time.perf_counter()
        d2 = self.cv_sq - 2.0 * (self.cv @ x) + x_sq
        part = np.argpartition(d2, k - 1)[:k]
        nr = float(np.sqrt(max(d2[part].max(), 0.0))) + EPS
        b = time.perf_counter()
        de = np.sqrt(np.maximum(self.cq_sq - 2.0 * (self.cq @ x) + x_sq, 0.0))
        fit = (nr + de) / self.radius
        m = min(N_BALLS, len(fit))
        best = np.argpartition(fit, m - 1)[:m] if m < len(fit) else np.arange(m)
        best = best[np.argsort(fit[best])]
        f = np.full(2 * N_BALLS + 1, POOR_FIT)
        f[0:2 * m:2] = de[best] / self.radius[best]
        f[1:2 * m:2] = nr / self.radius[best]
        c = time.perf_counter()
        if len(self.sv):
            cnt = int(((self.sv_sq - 2.0 * (self.sv @ x) + x_sq) < nr * nr).sum())
        else:
            cnt = 0
        f[-1] = np.log1p(cnt)
        d = time.perf_counter()
        score = float(((f - self.mu) / self.sd) @ self.coef + self.b)
        hit = score >= self.lam
        e = time.perf_counter()
        t["knn"] += b - a
        t["balls"] += c - b
        t["shell"] += d - c
        t["model"] += e - d
        return hit, score, self.ids[part]


def open_diskann(path, nodes, init_threads=32):
    """nodes = 0: node caching disabled. Otherwise DiskANN's standard warm-up
    picks the nodes to cache from its sample queries (run with
    init_threads). diskannpy hangs when an index is opened with 1 thread, so
    2 contexts are opened; each timed search is still one query, issued
    one at a time from one thread."""
    import diskannpy
    return diskannpy.StaticDiskIndex(index_directory=str(path),
                                     num_threads=init_threads if nodes else 2,
                                     num_nodes_to_cache=int(nodes),
                                     cache_mechanism=1 if nodes else 0, index_prefix="ann")


def time_backend(ix, xs, k, L):
    lat = np.empty(len(xs))
    for i, x in enumerate(xs):
        a = time.perf_counter()
        ix.search(x, k_neighbors=k, complexity=L, beam_width=2)
        lat[i] = time.perf_counter() - a
    return lat


def summary(lat):
    return {"mean_ms": float(lat.mean() * 1e3), "p50_ms": float(np.percentile(lat, 50) * 1e3),
            "p99_ms": float(np.percentile(lat, 99) * 1e3)}


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--prep", type=str, default=str(RES / "online" / "esci1m"))
    ap.add_argument("--state", type=str, required=True)
    ap.add_argument("--diskann", type=str, default=str(RES / "backends" / "esci1m" / "diskann"))
    ap.add_argument("--L", type=int, default=128, help="DiskANN search list (128: recall@10 .96)")
    ap.add_argument("--n", type=int, default=3000, help="queries timed (after warm-up)")
    ap.add_argument("--warm", type=int, default=200, help="untimed queries first, per arm")
    args = ap.parse_args()
    st = dict(np.load(args.state))
    db = np.load(ESCI / "esci_db1000000_s42.npy").astype(np.float32)
    q = np.load(ESCI / "esci_queries.npy").astype(np.float32)
    test_q = st["test_q"][:args.n]
    xs = np.ascontiguousarray(q[test_q])
    k = int(st["k"])
    hp = HitPath(db, q, st)
    cache_bytes = int(sum(hp.bytes.values()))
    node_bytes = db.shape[1] * 4 + 4 + 64 * 4
    logger.info(f"cache memory {cache_bytes / 1e6:.1f} MB ({hp.bytes}); "
                f"equal-memory DiskANN node cache: {cache_bytes // node_bytes} nodes")
    res = {"state": args.state, "L": args.L, "n": len(xs), "cache_bytes": hp.bytes,
           "cache_bytes_total": cache_bytes, "hit_threshold": hp.lam}

    # hit decisions (and validation against the simulator's scores)
    t = {"knn": 0.0, "balls": 0.0, "shell": 0.0, "model": 0.0}
    for x in xs[:args.warm]:
        hp.decide(x, dict(t))
    t = {"knn": 0.0, "balls": 0.0, "shell": 0.0, "model": 0.0}
    dec = np.empty(len(xs), dtype=np.float64)
    hits = np.zeros(len(xs), dtype=bool)
    per_q = np.empty(len(xs))
    for i, x in enumerate(xs):
        a = time.perf_counter()
        hits[i], dec[i], _ = hp.decide(x, t)
        per_q[i] = time.perf_counter() - a
    diff = np.abs(dec - st["test_scores"][:len(xs)])
    res["validation"] = {"max_score_diff": float(diff.max()),
                         "decisions_agree": float(np.mean(
                             hits == (st["test_scores"][:len(xs)] >= hp.lam)))}
    res["hit_rate"] = float(hits.mean())
    res["hit_path"] = dict(summary(per_q), stages_ms={s: v / len(xs) * 1e3 for s, v in t.items()})
    logger.info(f"hit rate {hits.mean():.3f}; hit decision {res['hit_path']}; "
                f"validation {res['validation']}")

    # arms
    for arm, nodes in (("backend", 0), ("backend_mem", cache_bytes // node_bytes)):
        ix = open_diskann(args.diskann, nodes)
        time_backend(ix, xs[:args.warm], k, args.L)
        lat = time_backend(ix, xs, k, args.L)
        res[arm] = dict(summary(lat), nodes_cached=int(nodes))
        logger.info(f"{arm}: {res[arm]}")
        if arm == "backend":
            # cache arm: hit decision for every query + backend on misses
            # (same backend latencies, measured on the same queries)
            total = per_q + np.where(hits, 0.0, lat)
            res["cache"] = dict(summary(total), hits_ms=summary(per_q[hits]) if hits.any() else None,
                                misses_ms=summary(total[~hits]) if (~hits).any() else None)
            logger.info(f"cache: {res['cache']}")
        del ix
    out = Path(args.state).with_name(f"timing_{Path(args.state).stem}_L{args.L}.json")
    out.write_text(json.dumps(res, indent=1))
    logger.info(f"wrote {out}")


if __name__ == "__main__":
    main()
