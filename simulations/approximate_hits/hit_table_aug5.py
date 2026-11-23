#!/usr/bin/env python3
"""Intruder counts from the backend's own compressed codes (doc 9au): instead of our sampled 4-bit shell, count the
uncached shell vectors of the request's 10 best-fitting balls (their whole shells, no sampling) whose distance to the
request, from DiskANN's in-memory PQ codes, is below nr (the request's k-th cached distance). Also the same count with
full-precision distances (the full-shell ceiling: separates "no sampling" from "PQ noise"). Writes <table>.aug5.npz.

  PYTHONPATH=. .venv-core/bin/python simulations/approximate_hits/hit_table_aug5.py <table.npz> [--procs 16]
"""
import sys
import json
import logging
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

sys.path.append(str(Path(__file__).parent.parent.parent))
import numpy as np

from simulations.approximate_hits import hit_table as H
from simulations.approximate_hits.workloads import get

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
G = {}


def load_pq(index_dir):
    """DiskANN's PQ: pivots (256 x dim), centroid (dim), chunk offsets, and the codes (n x chunks, uint8)."""
    b = np.fromfile(Path(index_dir) / "ann_pq_pivots.bin", dtype=np.uint8)
    nr_, nc_ = np.frombuffer(b[:8], dtype=np.int32)
    offs = np.frombuffer(b[8:8 + 8 * nr_ * nc_], dtype=np.uint64).astype(np.int64)

    def mat(o, dt):
        r, c = np.frombuffer(b[o:o + 8], dtype=np.int32)
        return np.frombuffer(b[o + 8:o + 8 + r * c * np.dtype(dt).itemsize], dtype=dt).reshape(r, c)
    piv = mat(offs[0], np.float32)
    cen = mat(offs[1], np.float32)[:, 0]
    chunk = mat(offs[2], np.uint32)[:, 0].astype(np.int64)
    c = np.fromfile(Path(index_dir) / "ann_pq_compressed.bin", dtype=np.uint8)
    n, m = np.frombuffer(c[:8], dtype=np.int32)
    codes = c[8:8 + int(n) * int(m)].reshape(int(n), int(m))
    return piv, cen, chunk, codes


def work(lo, hi):
    U, nr, fidx = G["U"], G["nr"], G["fidx"]
    piv, cen, chunk, codes = G["pq"]
    db, Q, shells, cached = G["db"], G["Q"], G["shells"], G["cached"]
    M = len(chunk) - 1
    out_pq = np.zeros(hi - lo, np.int32)
    out_f = np.zeros(hi - lo, np.int32)
    out_n = np.zeros(hi - lo, np.int32)
    for i in range(lo, hi):
        ids = np.unique(np.concatenate([shells[e] for e in fidx[i]]))
        ids = ids[~cached[ids]]
        out_n[i - lo] = len(ids)
        if not len(ids):
            continue
        x = np.asarray(Q[U[i]], dtype=np.float32)
        r2 = nr[i] * nr[i]
        xc = x - cen
        tab = np.empty((M, 256), np.float32)
        for c in range(M):
            a, b_ = chunk[c], chunk[c + 1]
            tab[c] = ((piv[:, a:b_] - xc[a:b_]) ** 2).sum(axis=1)
        d_pq = tab[np.arange(M)[None, :], codes[ids]].sum(axis=1)
        out_pq[i - lo] = int((d_pq < r2).sum())
        V = np.asarray(db[ids], dtype=np.float32)
        out_f[i - lo] = int((((V - x) ** 2).sum(axis=1) < r2).sum())
    return lo, out_pq, out_f, out_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--balls", type=int, default=10)
    a = ap.parse_args()
    path = Path(a.table)
    z = np.load(path)
    m = json.loads(str(z["meta"]))
    w = get(m["workload"])
    OS, args, D, stream, sim = H.build_cache(w, m["budget"], m["variant"], m["backend"])
    cache = sim.cache
    cache.prepare()
    if abs(cache.p - m["p0"]) > 1e-12 or not np.allclose(cache.radius, z["rad"]):
        raise SystemExit("cache differs from the table")
    present = D["present"]
    shells = []
    for c in cache.centers:
        _, sh = cache._ball_shell(int(c))
        sh = sh[sh >= 0]
        shells.append(sh[present[sh]].astype(np.int64))
    G.update(U=z["U"], nr=z["nr"], fidx=z["fit_idx"][:, :a.balls], pq=load_pq(w.backend_info()["index_dir"]),
             db=D["db"], Q=D["q"], shells=shells, cached=cache._count > 0)
    n = len(z["U"])
    step = (n + a.procs * 4 - 1) // (a.procs * 4)
    pq = np.zeros(n, np.int32)
    fl = np.zeros(n, np.int32)
    nn = np.zeros(n, np.int32)
    with ProcessPoolExecutor(a.procs) as ex:        # fork: workers share G
        for lo, p_, f_, n_ in ex.map(work, range(0, n, step), [min(n, s + step) for s in range(0, n, step)]):
            pq[lo:lo + len(p_)], fl[lo:lo + len(f_)], nn[lo:lo + len(n_)] = p_, f_, n_
    np.savez(path.with_suffix(".aug5.npz"), pq_shell10=pq, f_shell10=fl, n_shell10=nn)
    logging.info(f"wrote {path.with_suffix('.aug5.npz')}: shell ids per request {nn.mean():.0f}, "
                 f"PQ count mean {pq.mean():.2f}, float count mean {fl.mean():.2f}")


if __name__ == "__main__":
    main()
