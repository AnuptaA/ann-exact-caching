#!/usr/bin/env python3
"""
Why does DiskANN need search list ~2000 for recall@10 .95 on TriviaQA (SPACEV: 160)? (doc 9r)
Builds one index variant over the 10M SPHERE passages and measures recall@10 vs search list
(batched, 2,000 random trace queries, exact inner-product truth from the prep) and single-thread
latency per query at each search list (as in the timing harness: one query at a time, beam 2).

Variants (--variant):
  base     disk index, L2 on the inner-product transform (769-d), 256-byte PQ, degree 64, build L 100
           (the current index; measured from its directory, not rebuilt)
  pq768    as base, 768-byte PQ (1 byte per dimension)
  r128     as base, degree 128, build list 200
  mips     disk index, DiskANN's native inner product on the raw 768-d vectors, 256-byte PQ
  mem      in-memory Vamana (no PQ), L2 on the transform, degree 64, build L 100: graph quality alone

  .venv-ann/bin/python simulations/approximate_hits/index_study.py --variant pq768
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
RES = Path("simulations/approximate_hits/results")
DATA_IP = Path("datasets/aker/sphere-ip-10m-f32.npy")      # 769-d, inner-product transform
DATA_RAW = Path("datasets/aker/sphere-10m-f32.npy")        # 768-d raw DPR_web passages
DATA_SPACEV = Path("datasets/aker/spacev-10m.npy")         # 100-d int8
VARIANTS = {
    "base": dict(kind="disk", data="ip", metric="l2", pq=256, R=64, L=100),
    "pq768": dict(kind="disk", data="ip", metric="l2", pq=768, R=64, L=100),
    "r128": dict(kind="disk", data="ip", metric="l2", pq=256, R=128, L=200),
    "mips": dict(kind="disk", data="raw", metric="mips", pq=256, R=64, L=100),
    "mem": dict(kind="memory", data="ip", metric="l2", pq=0, R=64, L=100),
    # the Aker paper's DiskANN settings (doc 9u): TriviaQA degree 96, build list 100, 5.1 GB in-memory
    # PQ (~512 bytes per vector); SPACEV degree 64, build list 100, 1.1 GB (~110 bytes per vector)
    "aker_tqa": dict(kind="disk", data="ip", metric="l2", pq=512, R=96, L=100),
    "spacev_pq110": dict(kind="disk", data="spacev", metric="l2", pq=110, R=64, L=100),
}
EFFORTS = [32, 64, 100, 128, 160, 200, 256, 320, 400, 512, 640, 800, 1000, 1250, 1600, 2000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), required=True)
    ap.add_argument("--prep", default=None, help="queries / truth (default: triviaqa-ip-0.6; SPACEV: spacev-aker-0.6)")
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--n-recall", type=int, default=2000)
    ap.add_argument("--n-latency", type=int, default=0,
                    help="single-query latency samples per search list (0: recall only; on EBS latency "
                         "is only meaningful on a quiet machine, doc 9r)")
    args = ap.parse_args()
    import diskannpy
    v = VARIANTS[args.variant]
    out = RES / "backends" / "index_study" / args.variant
    out.mkdir(parents=True, exist_ok=True)
    d = (RES / "backends" / "sphereip10m-pq256" / "diskann") if args.variant == "base" else (out / "index")
    d.mkdir(parents=True, exist_ok=True)
    data_path = {"ip": DATA_IP, "raw": DATA_RAW, "spacev": DATA_SPACEV}[v["data"]]
    n_db = 10_000_000
    t0 = time.time()
    if not any(d.glob("ann*")):
        db = np.load(data_path, mmap_mode="r")
        logger.info(f"building {args.variant} on {data_path} {db.shape}")
        if v["kind"] == "disk":
            diskannpy.build_disk_index(data=db, distance_metric=v["metric"], index_directory=str(d),
                                       complexity=v["L"], graph_degree=v["R"],
                                       search_memory_maximum=v["pq"] * n_db / 1e9, build_memory_maximum=96,
                                       num_threads=args.threads, pq_disk_bytes=0, index_prefix="ann")
        else:
            diskannpy.build_memory_index(data=db, distance_metric=v["metric"], index_directory=str(d),
                                         complexity=v["L"], graph_degree=v["R"], num_threads=args.threads,
                                         index_prefix="ann")
        del db
    build_s = time.time() - t0
    prep = Path(args.prep or RES / "online" / ("spacev-aker-0.6" if v["data"] == "spacev" else "triviaqa-ip-0.6"))
    q = np.ascontiguousarray(np.load(prep / "q.npy"), dtype=np.float32)
    if v["data"] == "spacev":
        q = q.astype(np.int8)                      # the index stores int8 vectors (integer-valued queries)
    if v["data"] == "raw":
        assert np.allclose(q[:, -1], 0.0), "the ip queries carry a 0 in the appended dimension"
        q = np.ascontiguousarray(q[:, :-1])
    truth = np.load(prep / "top500.npy", mmap_mode="r")
    rng = np.random.default_rng(1)
    rows = rng.choice(len(q), args.n_recall, replace=False)
    if v["kind"] == "disk":
        from simulations.approximate_hits.diskann_aio import open_disk_index
        ix = open_disk_index(index_directory=str(d), num_threads=args.threads, num_nodes_to_cache=0,
                                       cache_mechanism=0, index_prefix="ann")
    else:
        ix = diskannpy.StaticMemoryIndex(index_directory=str(d), num_threads=args.threads,
                                         initial_search_complexity=max(EFFORTS), index_prefix="ann")

    def batch(qs, L):
        if v["kind"] == "disk":
            r = ix.batch_search(qs, k_neighbors=10, complexity=L, beam_width=2, num_threads=args.threads)
        else:
            r = ix.batch_search(qs, k_neighbors=10, complexity=L, num_threads=args.threads)
        return np.asarray(r.identifiers if hasattr(r, "identifiers") else r[0], dtype=np.int64)

    def one(x, L):
        if v["kind"] == "disk":
            return ix.search(x, k_neighbors=10, complexity=L, beam_width=2)
        return ix.search(x, k_neighbors=10, complexity=L)

    curve = {}
    for L in EFFORTS:
        ids = batch(q[rows], L)
        rec = float(np.mean([len(np.intersect1d(a, b)) / 10 for a, b in zip(ids, truth[rows, :10])]))
        lat = [np.nan]
        for i in rows[:args.n_latency]:
            a = time.perf_counter()
            one(q[i], L)
            lat.append(time.perf_counter() - a)
        lat = lat[1:] if len(lat) > 1 else lat
        curve[L] = {"recall": rec, "latency_ms_mean": float(np.mean(lat) * 1e3),
                    "latency_ms_p50": float(np.median(lat) * 1e3)}
        logger.info(f"{args.variant} L {L}: recall@10 {rec:.4f}")
        if rec >= 0.97:
            break
    res = {"variant": args.variant, "config": v, "build_s": build_s, "curve": curve,
           "index_bytes": int(sum(f.stat().st_size for f in d.glob("ann*")))}
    for tgt in (0.9, 0.95):
        ok = [L for L, c in curve.items() if c["recall"] >= tgt]
        res[f"at_{tgt}"] = {"L": min(ok), **curve[min(ok)]} if ok else None
    (out / "result.json").write_text(json.dumps(res, indent=1))
    logger.info(f"{args.variant}: {res.get('at_0.9')} | {res.get('at_0.95')}")


if __name__ == "__main__":
    main()
