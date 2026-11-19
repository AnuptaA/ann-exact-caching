#!/usr/bin/env python3
"""Exact intruder count for a hit table (doc 9al; not deployable): the uncached database vectors closer to the
request than nr (its k-th cached distance), counted in the request's own exact top-500 list. Tells how much
the sampled count's noise costs: with this column the model sees the quantity the sample estimates.
count_trunc: the 500th neighbor is still inside nr (count is a lower bound). Writes <table>.aug2.npz.
  PYTHONPATH=. .venv-core/bin/python simulations/approximate_hits/hit_table_aug2.py <table.npz>"""
import sys, json, logging
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import numpy as np
from simulations.approximate_hits import hit_table as H
from simulations.approximate_hits.workloads import get
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")


def main():
    path = Path(sys.argv[1])
    z = np.load(path)
    m = json.loads(str(z["meta"]))
    OS, args, D, stream, sim = H.build_cache(get(m["workload"]), m["budget"], m["variant"], m["backend"])
    cache = sim.cache
    cache.prepare()
    if abs(cache.p - m["p0"]) > 1e-12:
        raise SystemExit("cache differs from the table")
    U, nr = z["U"], z["nr"]
    cached = cache._count > 0
    db, Q = D["db"], D["q"]
    cnt = np.zeros(len(U), np.int32)
    trunc = np.zeros(len(U), bool)
    for s in range(0, len(U), 256):
        qi = U[s:s + 256]
        rk = D["rank"][qi]
        ok = rk >= 0
        ids = np.where(ok, rk, 0)
        V = np.asarray(db[ids.ravel()], dtype=np.float32).reshape(len(qi), rk.shape[1], -1)
        x = np.asarray(Q[qi], dtype=np.float32)
        d2 = ((V - x[:, None, :]) ** 2).sum(axis=2)
        inside = ok & (d2 < (nr[s:s + 256] ** 2)[:, None])
        cnt[s:s + len(qi)] = (inside & ~cached[ids] & D["present"][ids]).sum(axis=1)
        trunc[s:s + len(qi)] = inside[:, -1]
    np.savez(path.with_suffix(".aug2.npz"), exact_cnt=cnt, exact_cnt_trunc=trunc)
    logging.info(f"wrote {path.with_suffix('.aug2.npz')} (truncated rows {trunc.mean():.3f})")


if __name__ == "__main__":
    main()
