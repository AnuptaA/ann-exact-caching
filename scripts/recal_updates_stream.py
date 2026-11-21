#!/usr/bin/env python3
"""ReCal on ESCI 50K with an insert and delete stream (M3 acceptance).

The database starts with 45,000 of the 50,000 subsample vectors; every --every requests a batch of the
remaining vectors is inserted and a few cached vectors are deleted; once, a burst deletes a large share of
the pool. Checks after every update step that the bytes stay within M, and reports sweeps, frozen inserts,
rebuild signals and the recall of main-cache hits against the exact top-K of the current database.

  PYTHONPATH=. .venv-core/bin/python scripts/recal_updates_stream.py --requests 45000
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=0.05)
    ap.add_argument("--requests", type=int, default=45000)
    ap.add_argument("--every", type=int, default=500)
    ap.add_argument("--burst-at", type=int, default=30000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    full = np.load("datasets/esci/data_vectors/esci_db50000_s42.npy")
    pool = np.load("datasets/esci/data_vectors/esci_queries.npy")
    perm = np.random.default_rng(2026).permutation(len(pool))
    stream = np.ascontiguousarray(pool[perm[10000:10000 + args.requests]], dtype=np.float32)
    backend = ExactBackend(full[:45000])
    M = args.budget * full.size * 4
    work = tempfile.mkdtemp(prefix="recal_updates_")
    system = ReCal(backend, Params(), M, work)
    rng = np.random.default_rng(0)
    next_id, worst, rows, recalls, t0 = 45000, 0.0, [], [], time.time()
    for i, q in enumerate(stream):
        r = system.serve(q, 10)
        if r.source == "main":
            truth, _ = backend.search(q, 10, kind="eval")
            recalls.append(float(np.isin(truth, r.ids).mean()))
        if i % args.every == args.every - 1 and system.built is not None:
            take = min(50, len(full) - next_id)
            if take > 0:
                system.insert(np.arange(next_id, next_id + take), full[next_id:next_id + take])
                next_id += take
            live = [int(x) for e in range(0, system.core.n_entries(), 7) for x in system.core.entry(e)["response_ids"]]
            victims = rng.choice(live, size=min(20, len(live)), replace=False) if live else []
            if i + 1 >= args.burst_at and not any(e["burst"] for e in rows):
                pool_ids = sorted({x for e in range(system.core.n_entries()) for x in system.core.entry(e)["response_ids"]})
                victims = rng.choice(pool_ids, size=int(0.22 * len(pool_ids)), replace=False)
            system.delete(victims)
            total = system.bytes_by_structure()["total"]
            worst = max(worst, total / M)
            rows.append({"requests": i + 1, "burst": len(victims) > 100, "total_pct_of_M": total / M * 100,
                         "dead": system.core.n_dead_slots(), "pool": system.core.n_data_slots(),
                         "frozen": system.updater.frozen, "constructions": system.constructions})
    summary = {"worst_total_pct_of_M": worst * 100, "events": system.signals.events, "constructions": system.constructions,
               "main_hit_recall": float(np.mean(recalls)) if recalls else None, "n_main_hits": len(recalls),
               "updater": {"admitted": system.updater.admitted, "frozen": system.updater.frozen,
                           "ignored": system.updater.ignored},
               "rows_tail": rows[-5:], "seconds": time.time() - t0}
    print(json.dumps(summary, default=str, indent=1))
    if args.out:
        json.dump({"summary": summary, "rows": rows}, open(args.out, "w"), default=str)
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
