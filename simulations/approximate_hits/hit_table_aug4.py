#!/usr/bin/env python3
"""Deeper unions for cheap serving (doc 9an; hit_table_aug4.py for m = 1, 3, 10): instead of the k nearest of the whole cache (a graph search at
stage 2), serve the k nearest among the members of the m best-fitting entries (their ball vectors, at most
m x K): exact, a few thousand distance computations. Per request and m in (1, 3, 10): the served answer's
retention vs the backend's top-k (y_union<m>) and vs the exact top-k, and the number of vectors scanned.
Writes <table>.aug4.npz (m = 30, 100; the fit order is recomputed over all entries).
  PYTHONPATH=. .venv-core/bin/python simulations/approximate_hits/hit_table_aug4.py <table.npz>"""
import sys, json, logging
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import numpy as np
from simulations.approximate_hits import hit_table as H
from simulations.approximate_hits.workloads import get
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
MS = (30, 100)


def main():
    path = Path(sys.argv[1])
    z = np.load(path)
    m = json.loads(str(z["meta"]))
    OS, args, D, stream, sim = H.build_cache(get(m["workload"]), m["budget"], m["variant"], m["backend"])
    cache = sim.cache
    cache.prepare()
    if abs(cache.p - m["p0"]) > 1e-12:
        raise SystemExit("cache differs from the table")
    U, k = z["U"], m["k"]
    cids = cache.cached_ids
    cv = np.ascontiguousarray(cache.cv, dtype=np.float32)
    cen = cache.centers
    mem = [np.searchsorted(cids, cache.members(int(c))) for c in cen]
    A = OS.topk_present(D, U, k, key="answer")
    Tt = OS.topk_present(D, U, k, key="truth")
    Q = D["q"]
    cq = np.ascontiguousarray(Q[cen], dtype=np.float32)
    cq_sq = (cq.astype(np.float64) ** 2).sum(axis=1).astype(np.float32)
    rad = cache.radius.astype(np.float64)
    nrz = z["nr"]
    fit = np.zeros((len(U), max(MS)), np.int32)
    for s0 in range(0, len(U), 1024):
        x = np.ascontiguousarray(Q[U[s0:s0 + 1024]], dtype=np.float32)
        x_sq = (x.astype(np.float64) ** 2).sum(axis=1).astype(np.float32)
        de = np.sqrt(H.sqd(x, x_sq, cq, cq_sq).astype(np.float64))
        f = (nrz[s0:s0 + 1024, None] + de) / rad[None, :]
        fit[s0:s0 + 1024, :min(max(MS), f.shape[1])] = H.topn(f, max(MS))
    out = {}
    for mm in MS:
        out[f"y_union{mm}"] = np.zeros(len(U)); out[f"ytr_union{mm}"] = np.zeros(len(U)); out[f"scan_union{mm}"] = np.zeros(len(U), np.int32)
    for i in range(len(U)):
        x = np.asarray(Q[U[i]], dtype=np.float32)
        for mm in MS:
            pos = np.unique(np.concatenate([mem[e] for e in fit[i, :mm]]))
            out[f"scan_union{mm}"][i] = len(pos)
            if len(pos) < k:
                continue
            d = ((cv[pos] - x) ** 2).sum(axis=1)
            top = pos[np.argpartition(d, k - 1)[:k]] if len(pos) > k else pos
            ids = cids[top]
            out[f"y_union{mm}"][i] = np.isin(A[i], ids).mean()
            out[f"ytr_union{mm}"][i] = np.isin(Tt[i], ids).mean()
        if i % 5000 == 0:
            logging.info(f"{i} / {len(U)}")
    np.savez(path.with_suffix(".aug4.npz"), **out)
    logging.info(f"wrote {path.with_suffix('.aug4.npz')}")


if __name__ == "__main__":
    main()
