#!/usr/bin/env python3
"""
DiskANN-only arms of the timing plot, measured ONCE per workload (doc 9ad) instead of inside every
technique's run (eval_timing_online.py --no-backend-arms): the arms do not depend on the technique.

  alone         DiskANN with the workload's base node cache (--node-cache; 0 before doc 9af) on the timed requests (every --sample-every-th request
                after the warm-up, by position: the same requests the technique runs time), at the
                workload's chosen search list and beam width (backends/<prep>/diskann.json);
  equal memory  per cache size: DiskANN with the base node cache plus the memory of the static ours
                cache at that size (FastHitPath bytes of the warm-up state; every technique has the
                same vector budget), same requests.
One query at a time, single thread, quiet-machine guard (timing_env.py), as eval_timing_online.py.
The index files are evicted from the page cache before every pass (--no-evict disables), the order of the arms is
shuffled per repetition (--seed, --repetitions) and the page cache residency before and after each pass is stored.

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

from recal.measure import pagecache

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
    ap.add_argument("--node-cache", default="0",
                    help="base DiskANN node cache in both arms (doc 9af): a node count or 'auto' (the workload's "
                         "choice, eval_timing.base_node_cache); the equal-memory arm adds the cache's memory to it")
    ap.add_argument("--no-evict", action="store_true", help="old behaviour: no page cache eviction between passes")
    ap.add_argument("--repetitions", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0, help="arm order")
    ap.add_argument("--out", required=True)
    from simulations.approximate_hits.timing_env import add_args, make_guard
    add_args(ap)
    args = ap.parse_args()
    guard = make_guard(args)
    guard.start()
    from simulations.approximate_hits.eval_timing import open_diskann, time_backend, FastHitPath, base_node_cache
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
    nodes0, nodes0_from = base_node_cache(prep.name) if args.node_cache == "auto" else (int(args.node_cache), None)
    node_bytes = db_native.shape[1] * np.dtype(db_native.dtype).itemsize + 4 + 64 * 4
    arms = [("alone", nodes0, None)]
    mems = {}
    for b, st in sorted(sts.items()):
        hp = FastHitPath(db, q, st, f"hnsw{KNN_EF}", "balls10")
        mems[b] = int(sum(hp.bytes.values()) + hp.extra_bytes)
        del hp
        arms.append((f"{b:g}", int(mems[b] // node_bytes) + nodes0, b))
    lats = {name: [] for name, _, _ in arms}
    runs = []
    for rep in range(args.repetitions):
        for name, nodes, _ in (arms if args.no_evict else pagecache.shuffled_arms(arms, args.seed, rep)):
            state = {} if args.no_evict else {"evict": pagecache.evict([index])}
            before = dict(pagecache.residency([index]), **state)
            ix = open_diskann(index, nodes)
            time_backend(ix, warm, args.k, L, beam, args.gap_ms)
            lats[name].append(time_backend(ix, xq, args.k, L, beam, args.gap_ms))
            runs.append({"arm": name, "nodes": nodes, "repetition": rep, "order_seed": args.seed,
                         "page_cache": pagecache.record(before, pagecache.residency([index]))})
            logger.info(f"{name}: {nodes} nodes cached, rep {rep}: {np.mean(lats[name][-1]) * 1e3:.2f} ms")
            del ix
    lat = np.concatenate(lats["alone"])
    res["backend"] = dict(summary(lat))
    res["runs"] = runs
    for name, nodes, b in arms[1:]:
        res["per_budget"][name] = dict(summary(np.concatenate(lats[name])), nodes_cached=nodes, cache_memory_bytes=mems[b])
        logger.info(f"budget {name}: equal memory ({nodes} nodes) {res['per_budget'][name]['mean_ms']:.2f} ms")
    guard.stop()
    res["environment"] = guard.report()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    np.save(str(Path(args.out).with_suffix("")) + "_lat.npy", lat)


if __name__ == "__main__":
    main()
