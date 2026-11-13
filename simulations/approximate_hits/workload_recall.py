#!/usr/bin/env python3
"""
Per-workload DiskANN recall for backend selection (doc 9v): recall@10 vs exact over the workload's
REQUESTS (sampled from its stream after the warm-up, repeats included, i.e. as users see it), for
every (search list, beam width) of a probed index. Batched (latency comes from the quiet probe,
latency_probe.py, measured once per index). Writes results/backends/<prep>/recall_<probe>.json,
which select_backend.py prefers over the probe's own recall.

  .venv-ann/bin/python simulations/approximate_hits/workload_recall.py --prep .../spacev-aker-0.99 \
      --probe spacev-pq110
"""

import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
RES = Path("simulations/approximate_hits/results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--probe", required=True, help="probe name (results/latency_probe/<probe>.json)")
    ap.add_argument("--stream", default=None, help="default: trace (ESCI: random)")
    ap.add_argument("--warmup", type=int, default=20000)
    ap.add_argument("--n", type=int, default=5000, help="requests sampled from the stream after the warm-up")
    ap.add_argument("--threads", type=int, default=24)
    args = ap.parse_args()
    import diskannpy
    from simulations.approximate_hits.prep_io import load_prep
    prep = Path(args.prep)
    probe = json.loads((RES / "latency_probe" / f"{args.probe}.json").read_text())
    db, q, top, kind = load_prep(prep, native=True)
    xs = q.astype(db.dtype) if db.dtype != np.float32 else q
    stream = np.load(prep / "streams.npz")[args.stream or ("random" if kind == "esci" else "trace")]
    rng = np.random.default_rng(3)
    req = rng.choice(np.asarray(stream[args.warmup:], dtype=np.int64), args.n, replace=True)
    ix = diskannpy.StaticDiskIndex(index_directory=probe["index"], num_threads=args.threads, num_nodes_to_cache=0,
                                   cache_mechanism=0, index_prefix="ann")
    xq = np.ascontiguousarray(xs[req])
    grid = []
    for L, W in sorted({(g["L"], g["W"]) for g in probe["grid"]}):
        r = ix.batch_search(xq, k_neighbors=10, complexity=L, beam_width=W, num_threads=args.threads)
        ids = np.asarray(r.identifiers if hasattr(r, "identifiers") else r[0])
        rec = float(np.mean([len(np.intersect1d(a, b)) / 10 for a, b in zip(ids, top[req, :10])]))
        grid.append({"L": L, "W": W, "recall_requests": rec})
        logger.info(f"{prep.name} {args.probe} L {L} W {W}: recall over requests {rec:.4f}")
    out = RES / "backends" / prep.name
    out.mkdir(parents=True, exist_ok=True)
    (out / f"recall_{args.probe}.json").write_text(json.dumps(
        {"prep": prep.name, "probe": args.probe, "index": probe["index"], "n_requests": args.n,
         "distinct_queries": int(len(np.unique(req))), "grid": grid}, indent=1))


if __name__ == "__main__":
    main()
