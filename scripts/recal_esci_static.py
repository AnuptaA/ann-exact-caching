#!/usr/bin/env python3
"""Static check of one ReCal construction on ESCI 50K against the earlier implementation's numbers.

History of 30,000 distinct ESCI queries (roles 6,000 / 12,000 / 6,000 / 6,000), 10,000 test queries, K = 10,
T = 0.9, exact backend. Prints entries, m, the initial lambda, the hit rate and the recall of the hits on the
test queries, and the bytes. Earlier implementation: hit rate 8.0 / 14.4 / 20.0 percent at 1 / 5 / 10 percent,
m 27 / 83 / 116, entries 59 / 242 / 421.

  PYTHONPATH=. .venv-core/bin/python scripts/recal_esci_static.py --budgets 0.01 0.05 0.1
"""

import json
import time
import argparse

import numpy as np

from recal import build as builder
from recal import policy
from recal.backend import ExactBackend
from recal.params import Params


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    ap.add_argument("--split-by-query", action="store_true")
    ap.add_argument("--out", default="recal_esci_static.json")
    args = ap.parse_args()
    db = np.load("datasets/esci/data_vectors/esci_db50000_s42.npy")
    pool = np.load("datasets/esci/data_vectors/esci_queries.npy")
    perm = np.random.default_rng(2026).permutation(len(pool))
    test, hist = pool[perm[:10000]], pool[perm[10000:40000]]
    backend = ExactBackend(db)
    ids, dists = backend.search(hist, 10, kind="label")
    t_ids, _ = backend.search(test, 10, kind="eval")
    K = np.full(len(hist), 10, np.int32)
    snap = {"vec": hist, "k": K, "ids": ids, "dists": dists, "has": np.ones(len(hist), bool),
            "version": np.zeros(len(hist), np.int64), "index": np.arange(len(hist))}
    results = {}
    for frac in args.budgets:
        p = Params(split_by_query=args.split_by_query)
        M = frac * db.size * 4
        t0 = time.time()
        built = builder.build(snap, backend, p, M, np.random.default_rng(0), 30000, hashes=np.arange(30000, dtype=np.uint64))
        feat, cand, _ = built.core.candidates_batch(test, 10, built.m)
        ok = feat[:, 0] == 1
        scores = np.zeros(len(test))
        scores[ok] = policy.score_features(feat[ok][:, [2, 3, 4]], built.model)
        hit = ok & (built.lam is not None) & (scores >= (built.lam if built.lam is not None else 2.0))
        recall = np.array([np.isin(t_ids[i], cand[i]).mean() for i in np.flatnonzero(hit)])
        b = built.core.bytes_by_structure()
        results[frac] = {"entries": built.n_entries, "m": built.m, "lambda": built.lam, "hit_rate": float(hit.mean()) * 100,
                         "recall_of_hits": float(recall.mean()) * 100 if len(recall) else None,
                         "h0_calibration": built.h0 * 100, "W": built.W, "R": built.R, "bytes_pct_of_M": b["total"] / M * 100,
                         "by_structure_pct": {k: round(v / M * 100, 2) for k, v in b.items() if k not in ("total", "budget")},
                         "seconds": time.time() - t0}
        print(frac, json.dumps(results[frac]), flush=True)
    json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
