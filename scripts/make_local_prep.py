#!/usr/bin/env python3
"""Prep folder for the ESCI 50K subsample in the layout of online_prepare.py (local checks).

Writes results/online/<name>/{meta.json, top500.npy, streams.npz} for the queries used as a stationary random
stream (distinct queries, no repeats) and a Zipf-like stream with repeats. Exact top-500 by the exact backend.

  PYTHONPATH=. .venv-core/bin/python scripts/make_local_prep.py --name esci50k --queries 60000
"""

import json
import argparse
from pathlib import Path

import numpy as np

from recal.backend import ExactBackend


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="esci50k")
    ap.add_argument("--queries", type=int, default=60000)
    ap.add_argument("--zipf", type=float, default=0.99)
    args = ap.parse_args()
    out = Path("simulations/approximate_hits/results/online") / args.name
    out.mkdir(parents=True, exist_ok=True)
    db = np.load("datasets/esci/data_vectors/esci_db50000_s42.npy")
    pool = np.load("datasets/esci/data_vectors/esci_queries.npy", mmap_mode="r")
    rng = np.random.default_rng(2026)
    chosen = rng.permutation(len(pool))
    # the first `queries` rows of `chosen` are the stream pool; top500 is stored for every query row of the file
    top = np.zeros((len(pool), 500), dtype=np.int32)
    backend = ExactBackend(db)
    used = np.sort(chosen[:args.queries])
    for s in range(0, len(used), 4096):
        rows = used[s:s + 4096]
        top[rows] = backend.search(np.asarray(pool[rows], dtype=np.float32), 500, kind="prep")[0]
    np.save(out / "top500.npy", top)
    random = chosen[:args.queries]
    ranks = rng.permutation(args.queries)
    p = (np.arange(1, args.queries + 1) ** -args.zipf)
    p /= p.sum()
    zipf = chosen[:args.queries][ranks[rng.choice(args.queries, size=args.queries, p=p)]]
    np.savez(out / "streams.npz", random=random, trace=zipf)
    (out / "meta.json").write_text(json.dumps({"kind": "esci", "n_db": 50000}))
    print(out)


if __name__ == "__main__":
    main()
