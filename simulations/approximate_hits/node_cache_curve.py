#!/usr/bin/env python3
"""
DiskANN node-cache size per workload, chosen where the gain saturates (doc 9ae, decided with Jialin 11-16).

DiskANN's node cache holds the graph nodes most visited by its sample queries (the hub nodes every
search starts from), so it helps every request, hit or miss, whatever repeats. This job measures mean
single-query latency (10 ms gap between searches, as the other timing arms) at node counts growing by
x4 from --start (fraction of the index's points) and stops at the first size whose x4 successor
improves the mean by less than --tol; that size is the chosen node cache. Quiet machine required.

  .venv-ann/bin/python simulations/approximate_hits/node_cache_curve.py --prep .../spacev-aker-0.6 \
      --out .../backends/spacev-aker-0.6/node_cache.json
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
    ap.add_argument("--target-recall", default="0.95")
    ap.add_argument("--stream", default="trace")
    ap.add_argument("--n", type=int, default=400, help="requests timed per size")
    ap.add_argument("--start", type=float, default=0.0003, help="first node cache, as a fraction of the points")
    ap.add_argument("--factor", type=float, default=4.0)
    ap.add_argument("--tol", type=float, default=0.03)
    ap.add_argument("--max-frac", type=float, default=0.2)
    ap.add_argument("--gap-ms", type=float, default=10.0)
    ap.add_argument("--full", action="store_true",
                    help="measure every size up to --max-frac (no early stop); also report the smallest size within "
                         "--tol of the best measured latency (doc 9ag: the first-flat-step rule stops too early on "
                         "ESCI / SPACEV real)")
    ap.add_argument("--out", required=True)
    from simulations.approximate_hits.timing_env import add_args, make_guard
    add_args(ap)
    args = ap.parse_args()
    guard = make_guard(args)
    guard.start()
    from simulations.approximate_hits.eval_timing import open_diskann, time_backend
    from simulations.approximate_hits.prep_io import load_prep
    prep = Path(args.prep)
    db_native = load_prep(prep, native=True)[0]
    _, q, _, _ = load_prep(prep)
    stream = np.load(prep / "streams.npz")[args.stream]
    w = min(20000, len(stream) // 5)
    bk = json.loads((RES / "backends" / prep.name / "diskann.json").read_text())
    L = int(bk["targets"][args.target_recall]["effort"])
    beam = int(bk["targets"][args.target_recall].get("beam", bk.get("beam", 2)))
    index = bk.get("index_dir") or str(RES / "backends" / prep.name / "diskann")
    req = np.asarray(stream[w:], dtype=np.int64)[::50][:args.n]
    xs = q.astype(db_native.dtype) if db_native.dtype != np.float32 else q
    xq = np.ascontiguousarray(xs[req])
    warm = np.ascontiguousarray(xs[np.asarray(stream[max(0, w - 200):w], dtype=np.int64)])
    N = len(db_native)
    node_bytes = db_native.shape[1] * np.dtype(db_native.dtype).itemsize + 4 + 64 * 4
    logger.info(f"{prep.name}: {N} points, search list {L}, beam {beam}, index {index}")
    curve, nodes, chosen = [], 0, None
    ix = open_diskann(index, 0)
    time_backend(ix, warm, 10, L, beam, args.gap_ms)
    base = float(time_backend(ix, xq, 10, L, beam, args.gap_ms).mean() * 1e3)
    curve.append({"nodes": 0, "frac": 0.0, "mean_ms": base})
    logger.info(f"no node cache: {base:.2f} ms")
    del ix
    nodes = max(int(args.start * N), 100)
    while nodes <= args.max_frac * N:
        ix = open_diskann(index, nodes)
        time_backend(ix, warm, 10, L, beam, args.gap_ms)
        ms = float(time_backend(ix, xq, 10, L, beam, args.gap_ms).mean() * 1e3)
        curve.append({"nodes": nodes, "frac": nodes / N, "mean_ms": ms, "memory_mb": nodes * node_bytes / 1e6})
        logger.info(f"{nodes} nodes ({100 * nodes / N:.3f}%): {ms:.2f} ms")
        del ix
        if not args.full and len(curve) >= 3 and curve[-2]["mean_ms"] - ms < args.tol * curve[-2]["mean_ms"]:
            chosen = curve[-2]          # the previous size already got (almost) all the gain
            break
        nodes = int(nodes * args.factor)
    first_flat = next((curve[i - 1] for i in range(2, len(curve))
                       if curve[i - 1]["mean_ms"] - curve[i]["mean_ms"] < args.tol * curve[i - 1]["mean_ms"]), curve[-1])
    best = min(c["mean_ms"] for c in curve)
    near_best = next(c for c in curve if c["mean_ms"] <= best * (1 + args.tol))
    if chosen is None:
        chosen = first_flat if not args.full else first_flat
    guard.stop()
    res = {"prep": prep.name, "points": N, "search_list": L, "beam": beam, "gap_ms": args.gap_ms, "tol": args.tol,
           "curve": curve, "chosen": chosen, "first_flat_step": first_flat, "within_tol_of_best": near_best,
           "environment": guard.report()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    logger.info(f"chosen node cache: {chosen}")


if __name__ == "__main__":
    main()
