#!/usr/bin/env python3
"""End-to-end ReCal run over a request stream with the exact backend; per-window report.

  PYTHONPATH=. .venv-core/bin/python scripts/recal_stream.py --dataset esci --budget 0.05 --requests 40000
  PYTHONPATH=. .venv-core/bin/python scripts/recal_stream.py --dataset tripclick --budget 0.05 --requests 60000

esci: distinct queries of the ESCI pool in random order over the 50K subsample (stationary, no repeats).
tripclick: the first --requests searches of the 2018-01-01 log over the 50K document subsample.
Per window of --window requests: share of requests answered by the mini cache and by main-cache hits, mean
recall of the main-cache hits against the exact top-K (computed outside the system), lambda, the share of the
window with lambda undefined, constructions so far, and the rebuild events.
"""

import json
import time
import shutil
import argparse
import tempfile

import numpy as np

from recal.backend import ExactBackend
from recal.params import Params
from recal.system import ReCal


def load(dataset: str, n: int):
    if dataset == "esci":
        db = np.load("datasets/esci/data_vectors/esci_db50000_s42.npy")
        pool = np.load("datasets/esci/data_vectors/esci_queries.npy")
        perm = np.random.default_rng(2026).permutation(len(pool))
        return db, pool[perm[10000:10000 + n]]
    db = np.load("datasets/tripclick/tripclick_db50000_s0.npy")
    return db, np.load("datasets/aker/tripclick-ip-requests-2018-01-01.npy", mmap_mode="r")[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["esci", "tripclick"], default="esci")
    ap.add_argument("--budget", type=float, default=0.05)
    ap.add_argument("--requests", type=int, default=40000)
    ap.add_argument("--window", type=int, default=5000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--split-by-query", action="store_true")
    args = ap.parse_args()
    db, stream = load(args.dataset, args.requests)
    stream = np.ascontiguousarray(stream, dtype=np.float32)
    backend = ExactBackend(db)
    t0 = time.time()
    truth, _ = backend.search(stream, 10, kind="eval")
    print(f"truth {time.time() - t0:.0f}s", flush=True)
    M = args.budget * db.size * 4
    work = tempfile.mkdtemp(prefix="recal_stream_")
    system = ReCal(backend, Params(split_by_query=args.split_by_query), M, work)
    rows, win = [], {"mini": 0, "main": 0, "n": 0, "recall": [], "undef": 0}
    t0 = time.time()
    for i, q in enumerate(stream):
        r = system.serve(q, 10)
        win["n"] += 1
        win["mini"] += r.source == "mini"
        if r.source == "main":
            win["main"] += 1
            win["recall"].append(float(np.isin(truth[i], r.ids).mean()))
        win["undef"] += system.lam is None
        if win["n"] == args.window:
            rec = np.mean(win["recall"]) if win["recall"] else float("nan")
            row = {"requests": i + 1, "mini_share": win["mini"] / win["n"], "main_share": win["main"] / win["n"],
                   "main_recall": rec, "lambda": system.lam, "lambda_undefined": win["undef"] / win["n"],
                   "constructions": system.constructions, "entries": system.built.n_entries if system.built else 0,
                   "elapsed": time.time() - t0}
            rows.append(row)
            print(json.dumps(row), flush=True)
            win = {"mini": 0, "main": 0, "n": 0, "recall": [], "undef": 0}
    summary = {"rows": rows, "events": system.signals.events, "counts": system.counts,
               "bytes": {k: float(v) for k, v in system.bytes_by_structure().items()}, "M": M,
               "backend": backend.stats, "status": system.status_report()}
    print(json.dumps({k: summary[k] for k in ("events", "counts", "backend", "status")}, default=str), flush=True)
    if args.out:
        json.dump(summary, open(args.out, "w"), indent=1, default=str)
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
