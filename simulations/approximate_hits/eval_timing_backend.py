#!/usr/bin/env python3
"""
DiskANN-only arms of the timing plot, measured ONCE per workload (doc 9ad) instead of inside every
technique's run (eval_timing_online.py --no-backend-arms): the arms do not depend on the technique.

  alone         DiskANN without node cache on the timed requests (every --sample-every-th request
                after the warm-up, by position: the same requests the technique runs time), at the
                workload's chosen search list and beam width (backends/<prep>/diskann.json);
  equal memory  per cache size: DiskANN with its node cache given the memory of the static ours
                cache at that size (FastHitPath bytes of the warm-up state; every technique has the
                same vector budget), same requests.
One query at a time, single thread, quiet-machine guard (timing_env.py), as eval_timing_online.py.

  .venv-ann/bin/python simulations/approximate_hits/eval_timing_backend.py --prep .../spacev-aker-0.6 \
      --states .../timing_v4/spacev-aker-0.6/state_b0.01.npz ... --sample-every 50 --out .../backend_s50.json
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
    ap.add_argument("--states", nargs="+", required=True, help="warm-up states, one per cache size")
    ap.add_argument("--target-recall", default="0.95")
    ap.add_argument("--stream", default="trace")
    ap.add_argument("--sample-every", type=int, default=50)
    ap.add_argument("--warm", type=int, default=200)
    ap.add_argument("--gap-ms", type=float, default=10.0,
                    help="pause before every timed search, as between a technique run's timed requests "
                         "(they are separated by ~50 untimed requests); 0 = back to back (doc 9ad)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", required=True)
    from simulations.approximate_hits.timing_env import add_args, make_guard
    add_args(ap)
    args = ap.parse_args()
    guard = make_guard(args)
    guard.start()
    from simulations.approximate_hits.eval_timing import open_diskann, time_backend, FastHitPath
    from simulations.approximate_hits.eval_timing_online import summary, KNN_EF
    from simulations.approximate_hits.prep_io import load_prep
    prep = Path(args.prep)
    db_native = load_prep(prep, native=True)[0]
    db, q, _, _ = load_prep(prep)
    stream = np.load(prep / "streams.npz")[args.stream]
    bk = json.loads((RES / "backends" / prep.name / "diskann.json").read_text())
    L = int(bk["targets"][args.target_recall]["effort"])
    beam = int(bk["targets"][args.target_recall].get("beam", bk.get("beam", 2)))
    index = bk.get("index_dir") or str(RES / "backends" / prep.name / "diskann")
    sts = {float(Path(s).stem.split("_b")[1]): dict(np.load(s, allow_pickle=False)) for s in args.states}
    warmups = {int(st["warmup"]) for st in sts.values()}
    if len(warmups) != 1:
        raise SystemExit(f"states with different warm-ups: {warmups}")
    w = warmups.pop()
    req = np.asarray(stream[w:], dtype=np.int64)
    timed = np.flatnonzero((np.arange(len(req)) % args.sample_every) == 0)
    xs = q.astype(db_native.dtype) if db_native.dtype != np.float32 else q
    xq = np.ascontiguousarray(xs[req[timed]])
    warm = np.ascontiguousarray(xs[np.asarray(stream[max(0, w - args.warm):w], dtype=np.int64)])
    logger.info(f"{prep.name}: {len(timed)} timed requests (1 in {args.sample_every}), search list {L}, beam {beam}")
    ix = open_diskann(index, 0)
    time_backend(ix, warm, args.k, L, beam, args.gap_ms)
    lat = time_backend(ix, xq, args.k, L, beam, args.gap_ms)
    res = {"prep": prep.name, "search_list": L, "beam": beam, "index": index, "sample_every": args.sample_every, "gap_ms": args.gap_ms,
           "warmup": w, "n_timed": int(len(timed)), "backend": dict(summary(lat)), "per_budget": {}}
    logger.info(f"DiskANN alone: {res['backend']['mean_ms']:.2f} ms")
    del ix
    node_bytes = db_native.shape[1] * np.dtype(db_native.dtype).itemsize + 4 + 64 * 4
    for b, st in sorted(sts.items()):
        hp = FastHitPath(db, q, st, f"hnsw{KNN_EF}", "balls10")
        mem0 = int(sum(hp.bytes.values()) + hp.extra_bytes)
        del hp
        nodes = int(mem0 // node_bytes)
        ix2 = open_diskann(index, nodes)
        time_backend(ix2, warm, args.k, L, beam, args.gap_ms)
        lat_m = time_backend(ix2, xq, args.k, L, beam, args.gap_ms)
        res["per_budget"][f"{b:g}"] = dict(summary(lat_m), nodes_cached=nodes, cache_memory_bytes=mem0)
        logger.info(f"budget {b:g}: equal memory ({nodes} nodes) {res['per_budget'][f'{b:g}']['mean_ms']:.2f} ms")
        del ix2
    guard.stop()
    res["environment"] = guard.report()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    np.save(str(Path(args.out).with_suffix("")) + "_lat.npy", lat)


if __name__ == "__main__":
    main()
