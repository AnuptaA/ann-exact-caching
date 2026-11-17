#!/usr/bin/env python3
"""Sanity check for the cache core's graph library (doc 9ah): FAISS IndexHNSWFlat vs hnswlib on our own
cached-vector sets (warm-up states), same graph (M 32, efConstruction 200), single-query search on one
thread at the search lists the two-stage hit path uses; recall@10 vs brute force over the cached vectors.

  .venv-core/bin/python simulations/approximate_hits/bench_hnsw_libs.py --out /tmp/mx/bench_hnsw.json
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import numpy as np

RES = Path("simulations/approximate_hits/results")
CELLS = [("spacev-aker-0.6", "trace"), ("tripclick-ip", "trace"), ("triviaqa-ip-0.6", "trace")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", default="0.1")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--efs", type=int, nargs="+", default=[16, 64, 256, 1024])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import faiss
    import hnswlib
    from simulations.approximate_hits.prep_io import load_prep
    res = []
    for prep, stream in CELLS:
        db, q, _, _ = load_prep(RES / "online" / prep)
        st = np.load(RES / "timing_v4" / prep / f"state_b{a.budget}.npz")
        cv = np.ascontiguousarray(np.asarray(db[np.sort(st["cached_ids"])], dtype=np.float32))
        w = int(st["warmup"])
        req = np.load(RES / "online" / prep / "streams.npz")[stream][:w][-a.n:]
        xq = np.ascontiguousarray(np.asarray(q[req], dtype=np.float32))
        faiss.omp_set_num_threads(32)
        _, gt = faiss.knn(xq, cv, 10)
        t0 = time.time()
        fx = faiss.IndexHNSWFlat(cv.shape[1], 32)
        fx.hnsw.efConstruction = 200
        fx.add(cv)
        fb = time.time() - t0
        t0 = time.time()
        hx = hnswlib.Index(space="l2", dim=cv.shape[1])
        hx.init_index(max_elements=len(cv), M=32, ef_construction=200, random_seed=0)
        hx.add_items(cv, np.arange(len(cv)), num_threads=32)
        hb = time.time() - t0
        faiss.omp_set_num_threads(1)
        hx.set_num_threads(1)
        for ef in a.efs:
            fx.hnsw.efSearch = ef
            hx.set_ef(max(ef, 10))
            for lib in ("faiss", "hnswlib"):
                ids = np.empty((len(xq), 10), np.int64)
                t0 = time.perf_counter()
                for i, x in enumerate(xq):
                    if lib == "faiss":
                        _, ii = fx.search(x[None], 10)
                        ids[i] = ii[0]
                    else:
                        ii, _ = hx.knn_query(x[None], k=10)
                        ids[i] = ii[0]
                ms = (time.perf_counter() - t0) / len(xq) * 1e3
                rec = float(np.mean([len(set(r.tolist()) & set(g.tolist())) / 10 for r, g in zip(ids, gt)]))
                row = {"prep": prep, "n_cached": len(cv), "dim": cv.shape[1], "lib": lib, "ef": ef,
                       "ms": ms, "recall": rec, "build_s": fb if lib == "faiss" else hb}
                res.append(row)
                print(json.dumps(row), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
