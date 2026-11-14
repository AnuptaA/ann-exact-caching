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

import sys
import json
import time
import argparse
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

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
        self.cv = np.ascontiguousarray(db[np.sort(st["cached_ids"])], dtype=np.float32)
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


class FastHitPath(HitPath):
    """Variants of the hit decision for timing:
      knn    "brute" (scan every cached vector, as above) or "hnsw<ef>":
             a FAISS HNSW graph over the cached vectors (M 32,
             efConstruction 200), search list ef; approximate 10 nearest
             cached vectors, so the neighborhood radius can be a bit larger
      shell  "fp32" (scan every shell vector at full precision, as above),
             "sq4" (scan the 4-bit codes themselves: FAISS 4-bit scalar
             quantizer, one list, radius search), or "balls<n>" (scan only
             the sampled shell vectors of the n best-fitting balls; "balls" = 3)
    """

    def __init__(self, db, q, st, knn, shell):
        super().__init__(db, q, st)
        import faiss
        faiss.omp_set_num_threads(32)       # building only; searches below use 1
        self.knn_mode, self.shell_mode = knn, shell
        self.extra_bytes = 0
        if knn.startswith("hnsw"):
            self.hnsw = faiss.IndexHNSWFlat(self.cv.shape[1], 32)
            self.hnsw.hnsw.efConstruction = 200
            self.hnsw.add(self.cv)
            self.hnsw.hnsw.efSearch = int(knn[4:])
            # graph links (FAISS stores M neighbors per level, 2 M at level 0)
            self.extra_bytes += int(self.hnsw.hnsw.neighbors.size() * 4)
        faiss.omp_set_num_threads(1)
        if shell == "sq4":
            quant = faiss.IndexFlatL2(self.sv.shape[1])
            self.sq = faiss.IndexIVFScalarQuantizer(quant, self.sv.shape[1], 1,
                                                    faiss.ScalarQuantizer.QT_4bit,
                                                    faiss.METRIC_L2)
            raw = np.ascontiguousarray(db[st["samp_ids"]], dtype=np.float32)
            self.sq.train(raw)
            self.sq.add(raw)
            self.sq.nprobe = 1
        if shell.startswith("balls"):
            self.es, self.eo = st["entry_samp"], st["entry_samp_off"]
            self.nb = int(shell[5:] or 3)

    def decide(self, x, t):
        k = self.k
        x_sq = float(x @ x)
        a = time.perf_counter()
        if self.knn_mode == "brute":
            d2 = self.cv_sq - 2.0 * (self.cv @ x) + x_sq
            part = np.argpartition(d2, k - 1)[:k]
            nr = float(np.sqrt(max(d2[part].max(), 0.0))) + EPS
            near = self.ids[part]
        else:
            dd, ii = self.hnsw.search(x[None], k)
            nr = float(np.sqrt(max(dd[0].max(), 0.0))) + EPS
            near = self.ids[ii[0]]
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
        if not len(self.sv):
            cnt = 0
        elif self.shell_mode == "fp32":
            cnt = int(((self.sv_sq - 2.0 * (self.sv @ x) + x_sq) < nr * nr).sum())
        elif self.shell_mode == "sq4":
            lims, _, _ = self.sq.range_search(x[None], nr * nr)
            cnt = int(lims[1])
        else:
            sel = best if self.nb <= m else (
                np.argpartition(fit, self.nb - 1)[:self.nb] if self.nb < len(fit)
                else np.arange(len(fit)))
            pos = np.unique(np.concatenate([self.es[self.eo[e]:self.eo[e + 1]] for e in sel]))
            cnt = int(((self.sv_sq[pos] - 2.0 * (self.sv[pos] @ x) + x_sq) < nr * nr).sum()) \
                if len(pos) else 0
        f[-1] = np.log1p(cnt)
        d = time.perf_counter()
        score = float(((f - self.mu) / self.sd) @ self.coef + self.b)
        hit = score >= self.lam
        e = time.perf_counter()
        t["knn"] += b - a
        t["balls"] += c - b
        t["shell"] += d - c
        t["model"] += e - d
        return hit, score, near


def run_variants(args, db, q, st, xs):
    """Time each hit-decision variant on the same queries; compare its
    decisions with the simulator's (scores saved in the state)."""
    ref_hit = st["test_scores"][:len(xs)] >= float(st["lam"])
    out = {}
    exact = HitPath(db, q, st)
    for v in args.variants:
        knn, shell = v.split("+")
        hp = FastHitPath(db, q, st, knn, shell)
        for x in xs[:args.warm]:
            hp.decide(x, {"knn": 0.0, "balls": 0.0, "shell": 0.0, "model": 0.0})
        t = {"knn": 0.0, "balls": 0.0, "shell": 0.0, "model": 0.0}
        per_q = np.empty(len(xs))
        hits = np.zeros(len(xs), dtype=bool)
        knn_recall = np.empty(len(xs))
        for i, x in enumerate(xs):
            a = time.perf_counter()
            hits[i], _, near = hp.decide(x, t)
            per_q[i] = time.perf_counter() - a
            if knn != "brute":
                _, _, near_ex = exact.decide(x, {"knn": 0.0, "balls": 0.0, "shell": 0.0,
                                                 "model": 0.0})
                knn_recall[i] = np.isin(near_ex, near).mean()
        out[v] = dict(summary(per_q), stages_ms={s_: val / len(xs) * 1e3 for s_, val in t.items()},
                      hit_rate=float(hits.mean()),
                      decisions_agree_with_simulator=float(np.mean(hits == ref_hit)),
                      knn_recall=None if knn == "brute" else float(knn_recall.mean()),
                      extra_index_bytes=hp.extra_bytes)
        logger.info(f"{v}: {out[v]}")
    return out


def open_diskann(path, nodes, init_threads=32):
    """nodes = 0: node caching disabled. Otherwise DiskANN's standard warm-up
    picks the nodes to cache from its sample queries (run with
    init_threads). diskannpy hangs when an index is opened with 1 thread, so
    2 contexts are opened; each timed search is still one query, issued
    one at a time from one thread."""
    from simulations.approximate_hits.diskann_aio import open_disk_index
    # timing runs: paused jobs keep their AIO contexts during a quiet batch, so wait at most 30 min
    return open_disk_index(max_wait=1800, index_directory=str(path),
                                     num_threads=init_threads if nodes else 2,
                                     num_nodes_to_cache=int(nodes),
                                     cache_mechanism=1 if nodes else 0, index_prefix="ann")


def time_backend(ix, xs, k, L, beam=2):
    """Latency of one DiskANN search per query (xs already in the index's dtype)."""
    lat = np.empty(len(xs))
    for i, x in enumerate(xs):
        a = time.perf_counter()
        ix.search(x, k_neighbors=k, complexity=L, beam_width=beam)
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
    ap.add_argument("--decision", type=str, default="hnsw32+balls10",
                    help="hit decision for the end-to-end arms (knn+shell, see FastHitPath)")
    ap.add_argument("--variants", type=str, nargs="*", default=None,
                    help="only time hit-decision variants knn+shell, e.g. brute+fp32 "
                         "brute+sq4 brute+balls hnsw32+balls (no backend arms)")
    from simulations.approximate_hits.timing_env import add_args, make_guard
    add_args(ap)
    args = ap.parse_args()
    guard = make_guard(args)       # refuses unless the machine is quiet (timing_env.py)
    guard.start()
    st = dict(np.load(args.state))
    prep = Path(args.prep)
    meta = json.loads((prep / "meta.json").read_text()) if (prep / "meta.json").exists() \
        else {"kind": "esci"}
    if meta["kind"] == "trace":
        # SPACEV: 8-bit integer vectors, memory-mapped (only cached rows are read);
        # DiskANN is an 8-bit index, so its queries are cast to 8 bits (exact:
        # the queries are integer-valued)
        from simulations.approximate_hits.run_e2e_oracle_trace import AKER, WORKLOADS
        db = np.load(AKER / WORKLOADS[meta["trace"]]["base"], mmap_mode="r")
        q = np.load(prep / "q.npy").astype(np.float32)
    else:
        db = np.load(ESCI / "esci_db1000000_s42.npy").astype(np.float32)
        q = np.load(ESCI / "esci_queries.npy").astype(np.float32)
    test_q = st["test_q"][:args.n]
    xs = np.ascontiguousarray(q[test_q])
    k = int(st["k"])
    idx_dtype = db.dtype if db.dtype in (np.int8, np.uint8) else np.float32
    xd = np.ascontiguousarray(xs.astype(idx_dtype))     # queries as the DiskANN index takes them
    if args.variants:
        res = {"state": args.state, "n": len(xs), "variants": run_variants(args, db, q, st, xs)}
        guard.stop()
        res["environment"] = guard.report()
        out = Path(args.state).with_name(f"variants_{Path(args.state).stem}.json")
        out.write_text(json.dumps(res, indent=1))
        logger.info(f"wrote {out}")
        return
    # the hit decision: exact scans ("brute+fp32", the simulator's default)
    # or a faster variant (e.g. "hnsw32+balls10"; the state must come from a
    # simulator run with the matching --shell-scope / --shell-balls)
    knn, shell = args.decision.split("+")
    hp = FastHitPath(db, q, st, knn, shell)
    cache_bytes = int(sum(hp.bytes.values())) + hp.extra_bytes
    node_bytes = db.shape[1] * np.dtype(idx_dtype).itemsize + 4 + 64 * 4   # vector + 64 neighbor ids
    n_nodes = cache_bytes // node_bytes
    logger.info(f"decision {args.decision}; cache memory {cache_bytes / 1e6:.1f} MB "
                f"({hp.bytes}, index {hp.extra_bytes}); equal-memory DiskANN node cache: "
                f"{n_nodes} nodes")
    res = {"state": args.state, "decision": args.decision, "L": args.L, "n": len(xs),
           "cache_bytes": dict(hp.bytes, index=hp.extra_bytes),
           "cache_bytes_total": cache_bytes, "hit_threshold": hp.lam}
    ref_hit = st["test_scores"][:len(xs)] >= hp.lam

    def zero():
        return {"knn": 0.0, "balls": 0.0, "shell": 0.0, "model": 0.0}

    # end to end, measured directly: each query runs the hit decision and,
    # on a miss, the DiskANN search, timed together in one loop
    ix = open_diskann(args.diskann, 0)
    for x, xq in zip(xs[:args.warm], xd[:args.warm]):
        if not hp.decide(x, zero())[0]:
            ix.search(xq, k_neighbors=k, complexity=args.L, beam_width=2)
    t = zero()
    hits = np.zeros(len(xs), dtype=bool)
    dec_t = np.empty(len(xs))
    total = np.empty(len(xs))
    for i, x in enumerate(xs):
        a = time.perf_counter()
        hits[i] = hp.decide(x, t)[0]
        b = time.perf_counter()
        if not hits[i]:
            ix.search(xd[i], k_neighbors=k, complexity=args.L, beam_width=2)
        total[i] = time.perf_counter() - a
        dec_t[i] = b - a
    res["hit_rate"] = float(hits.mean())
    res["decisions_agree_with_simulator"] = float(np.mean(hits == ref_hit))
    res["decision_ms"] = dict(summary(dec_t), stages_ms={s_: v / len(xs) * 1e3
                                                         for s_, v in t.items()})
    res["cache"] = dict(summary(total), hits_ms=summary(total[hits]) if hits.any() else None,
                        misses_ms=summary(total[~hits]) if (~hits).any() else None)
    logger.info(f"hit rate {hits.mean():.3f} (agree {res['decisions_agree_with_simulator']:.4f}); "
                f"decision {res['decision_ms']}; cache + DiskANN {res['cache']}")
    # the same queries against DiskANN alone, without and with our memory
    for arm, nodes in (("backend", 0), ("backend_mem", n_nodes)):
        if arm == "backend_mem":
            del ix
            ix = open_diskann(args.diskann, nodes)
        time_backend(ix, xd[:args.warm], k, args.L)
        lat = time_backend(ix, xd, k, args.L)
        res[arm] = dict(summary(lat), nodes_cached=int(nodes))
        logger.info(f"{arm}: {res[arm]}")
    m = res["backend"]["mean_ms"]
    res["speedup"] = {"overall": m / res["cache"]["mean_ms"],
                      "limit_1_over_1_minus_h": 1 / max(1 - res["hit_rate"], 1e-9),
                      "per_hit": m / res["cache"]["hits_ms"]["mean_ms"] if hits.any() else None,
                      "miss_overhead_ms": (res["cache"]["misses_ms"]["mean_ms"] - m)
                      if (~hits).any() else None,
                      "vs_equal_memory": res["backend_mem"]["mean_ms"] / res["cache"]["mean_ms"]}
    logger.info(f"speedup {res['speedup']}")
    guard.stop()
    res["environment"] = guard.report()
    logger.info(f"environment: {res['environment']['verdict']} "
                f"(other processes: mean {res['environment']['during']['other_cores_mean']} cores, "
                f"max {res['environment']['during']['other_cores_max']})")
    out = Path(args.state).with_name(
        f"e2e_{Path(args.state).stem}_{args.decision.replace('+', '_')}_L{args.L}.json")
    out.write_text(json.dumps(res, indent=1))
    logger.info(f"wrote {out}")


if __name__ == "__main__":
    main()
