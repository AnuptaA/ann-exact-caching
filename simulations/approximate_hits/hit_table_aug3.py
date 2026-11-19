#!/usr/bin/env python3
"""Cheap serving for a hit table (doc 9an): instead of the k nearest of the whole cache (a graph search at
stage 2), serve the k nearest among the members of the m best-fitting entries (their ball vectors, at most
m x K): exact, a few thousand distance computations. Per request and m in (1, 3, 10): the served answer's
retention vs the backend's top-k (y_union<m>) and vs the exact top-k, and the number of vectors scanned.
Writes <table>.aug3.npz (hit_ladder.py picks it up).
  PYTHONPATH=. .venv-core/bin/python simulations/approximate_hits/hit_table_aug3.py <table.npz>"""
import sys, json, logging
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import numpy as np
from simulations.approximate_hits import hit_table as H
from simulations.approximate_hits.workloads import get
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
MS = (1, 3, 10)


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
    fit = z["fit_idx"]
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
    np.savez(path.with_suffix(".aug3.npz"), **out)
    logging.info(f"wrote {path.with_suffix('.aug3.npz')}")


if __name__ == "__main__":
    main()
