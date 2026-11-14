#!/usr/bin/env python3
"""
Single-query DiskANN latency vs search list and beam width, on a quiet machine (doc 9s).
DiskANN reads one 4 KB sector per expanded node; with beam width W it issues W reads per round
trip, so latency ~ (expanded nodes / W) x read latency. The default W = 2 was never tuned.
Recall@10 is measured on the same queries (exact truth from the prep).

  .venv-ann/bin/python simulations/approximate_hits/latency_probe.py --prep .../spacev-aker-0.6 \
      --index .../backends/spacev10m/diskann --out .../latency_probe/spacev.json
"""

import sys
import json
import time
import argparse
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--index", required=True, help="DiskANN index directory (prefix ann)")
    ap.add_argument("--raw-768", action="store_true", help="queries without the appended 0 (native mips index)")
    ap.add_argument("--efforts", type=int, nargs="+", default=[64, 128, 200, 320, 512, 800, 1250, 2000])
    ap.add_argument("--beams", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    ap.add_argument("--n-recall", type=int, default=0,
                    help="queries for a batched recall per (search list, beam width); default 0 = none (11-13: "
                         "it dominated the quiet window, ~2.5 min per cell on SPACEV and far more at TriviaQA's "
                         "search lists, while workload_recall.py measures recall over each workload's requests "
                         "without needing a quiet machine)")
    ap.add_argument("--n", type=int, default=100, help="queries per (search list, beam width)")
    ap.add_argument("--out", required=True)
    from simulations.approximate_hits.timing_env import add_args, make_guard
    add_args(ap)
    args = ap.parse_args()
    guard = make_guard(args)
    guard.start()
    from simulations.approximate_hits.prep_io import load_prep
    db, q, top, _ = load_prep(Path(args.prep), native=True)
    xs = q.astype(db.dtype) if db.dtype != np.float32 else q
    if args.raw_768:
        xs = np.ascontiguousarray(xs[:, :-1])
    rows = np.random.default_rng(2).choice(len(xs), args.n, replace=False)
    import diskannpy                      # 24 thread contexts for the batched recall; no node cache
    ix = diskannpy.StaticDiskIndex(index_directory=str(args.index), num_threads=24, num_nodes_to_cache=0,
                                   cache_mechanism=0, index_prefix="ann")
    for i in rows[:20]:                                         # warm-up, untimed
        ix.search(xs[i], k_neighbors=10, complexity=128, beam_width=2)
    res = {"prep": args.prep, "index": args.index, "n": args.n, "grid": []}
    for L in args.efforts:
        base = {}
        for W in sorted(set(args.beams) | {2}):
            lat, rec, same = [], [], []
            for i in rows:
                a = time.perf_counter()
                r = ix.search(xs[i], k_neighbors=10, complexity=L, beam_width=W)
                lat.append(time.perf_counter() - a)
                ids = np.asarray(r.identifiers if hasattr(r, "identifiers") else r[0])
                rec.append(len(np.intersect1d(ids, top[i, :10])) / 10)
                if W == 2:
                    base[int(i)] = ids
                same.append(bool(np.array_equal(ids, base[int(i)])))
            # same_as_beam2: share of queries whose top-10 equals the beam-2 result (the stored answers
            # that drive the timing replay are beam-2 searches)
            rec_b = None
            if args.n_recall > 0:
                rr = np.random.default_rng(1).choice(len(xs), min(args.n_recall, len(xs)), replace=False)
                rb = ix.batch_search(np.ascontiguousarray(xs[rr]), k_neighbors=10, complexity=L, beam_width=W,
                                     num_threads=24)
                ib = np.asarray(rb.identifiers if hasattr(rb, "identifiers") else rb[0])
                rec_b = float(np.mean([len(np.intersect1d(a, b)) / 10 for a, b in zip(ib, top[rr, :10])]))
            row = {"L": L, "W": W, "recall": float(np.mean(rec)), "recall_5000": rec_b,
                   "same_as_beam2": float(np.mean(same)),
                   "mean_ms": float(np.mean(lat) * 1e3),
                   "p50_ms": float(np.median(lat) * 1e3), "p99_ms": float(np.percentile(lat, 99) * 1e3)}
            res["grid"].append(row)
            logger.info(f"L {L} W {W}: recall {row['recall']:.3f} ({args.n} queries), {row['mean_ms']:.1f} ms "
                        f"(p99 {row['p99_ms']:.1f})")
    guard.stop()
    res["environment"] = guard.report()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
