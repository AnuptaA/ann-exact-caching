#!/usr/bin/env python3
"""
Per-workload DiskANN recall for backend selection (doc 9v): recall@10 vs exact over the workload's
REQUESTS (sampled from its stream after the warm-up, repeats included, i.e. as users see it), for
the (search list, beam width) cells of a probed index that the selection needs (two stages, see
main; checked to give the same choice as the full grid on SPACEV, 8-9 of 40 cells). Batched (latency comes from the quiet probe,
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
    ap.add_argument("--warmup", type=int, default=None, help="default: min(20000, a fifth of the stream)")
    ap.add_argument("--n", type=int, default=5000, help="requests sampled from the stream after the warm-up")
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--ref-beam", type=int, default=16)
    ap.add_argument("--targets", type=float, nargs="+", default=[0.95, 0.9], help="as select_backend.py")
    ap.add_argument("--margin", type=float, default=0.03,
                    help="beam widths change recall by at most this much at a search list (seen: 0.022)")
    args = ap.parse_args()
    from simulations.approximate_hits.prep_io import load_prep
    prep = Path(args.prep)
    probe = json.loads((RES / "latency_probe" / f"{args.probe}.json").read_text())
    db, q, top, kind = load_prep(prep, native=True)
    xs = q.astype(db.dtype) if db.dtype != np.float32 else q
    stream = np.load(prep / "streams.npz")[args.stream or ("random" if kind == "esci" else "trace")]
    rng = np.random.default_rng(3)
    if args.warmup is None:
        args.warmup = min(20000, len(stream) // 5)
    req = rng.choice(np.asarray(stream[args.warmup:], dtype=np.int64), args.n, replace=True)
    from simulations.approximate_hits.diskann_aio import open_disk_index
    ix = open_disk_index(index_directory=probe["index"], num_threads=args.threads, num_nodes_to_cache=0,
                                   cache_mechanism=0, index_prefix="ann")
    xq = np.ascontiguousarray(xs[req])
    def recall(L, W):
        r = ix.batch_search(xq, k_neighbors=10, complexity=L, beam_width=W, num_threads=args.threads)
        ids = np.asarray(r.identifiers if hasattr(r, "identifiers") else r[0])
        rec = float(np.mean([len(np.intersect1d(a, b)) / 10 for a, b in zip(ids, top[req, :10])]))
        logger.info(f"{prep.name} {args.probe} L {L} W {W}: recall over requests {rec:.4f}")
        return rec

    # two stages (11-14; every (L, W) took 2.5 h per SPACEV job and far longer for TriviaQA):
    # 1. every search list at the reference beam width, ascending, until recall exceeds the upper end
    #    of the band (recall grows with the search list: larger lists get that value as a lower bound);
    # 2. per target, cells in order of probe latency, lazily: a search list whose reference recall is
    #    below target - margin is skipped (no beam width reaches the target there; largest spread over
    #    beam widths seen: 0.022, SPACEV L 100), the first cell that reaches the target is the fastest
    #    one that does. Unmeasured slower cells are left out of the grid (select_backend.py only
    #    considers cells with a recall), so the selection equals the one from the full grid.
    lat = {(g["L"], g["W"]): g["mean_ms"] for g in probe["grid"]}
    Ls = sorted({L for L, _ in lat})
    Ws = sorted({W for _, W in lat})
    ref = args.ref_beam if args.ref_beam in Ws else max(Ws)
    hi = max(args.targets) + args.margin
    rec, src, bound = {}, {}, None
    for L in Ls:
        if bound is not None:
            rec[(L, ref)], src[(L, ref)] = bound, f"lower bound (W {ref} at a smaller search list)"
            continue
        rec[(L, ref)], src[(L, ref)] = recall(L, ref), "measured"
        if rec[(L, ref)] > hi:
            bound = rec[(L, ref)]
    for t in args.targets:
        for c in sorted(lat, key=lambda c: lat[c]):
            if rec[(c[0], ref)] < t - args.margin:
                continue                                    # no beam width reaches t at this L
            if c not in rec:
                if src[(c[0], ref)] != "measured":          # L beyond the bound: >= bound > t
                    rec[c], src[c] = rec[(c[0], ref)], src[(c[0], ref)]
                else:
                    rec[c], src[c] = recall(*c), "measured"
            if rec[c] >= t:
                break
    for L in Ls:                                            # pruned search lists: ineligible anyway
        if rec[(L, ref)] < min(args.targets) - args.margin:
            for W in Ws:
                if (L, W) not in rec:
                    rec[(L, W)], src[(L, W)] = rec[(L, ref)], f"W {ref} value, below every target - margin"
    grid = [{"L": L, "W": W, "recall_requests": rec[(L, W)], "source": src[(L, W)]} for L, W in sorted(rec)]
    logger.info(f"{len([1 for v in src.values() if v == 'measured'])} cells measured of {len(lat)}")
    out = RES / "backends" / prep.name
    out.mkdir(parents=True, exist_ok=True)
    (out / f"recall_{args.probe}.json").write_text(json.dumps(
        {"prep": prep.name, "probe": args.probe, "index": probe["index"], "n_requests": args.n,
         "distinct_queries": int(len(np.unique(req))), "ref_beam": ref, "margin": args.margin, "targets": args.targets, "grid": grid}, indent=1))


if __name__ == "__main__":
    main()
