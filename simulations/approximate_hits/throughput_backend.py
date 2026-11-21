#!/usr/bin/env python3
"""Throughput of the real DiskANN backend vs concurrency (doc 9ap): how many misses per second the backend can
serve when queries arrive from many threads (closed loop: every thread starts the next query as soon as its
last one finished), at the workload's search list and beam width, with the workload's base node cache.
Needs a quiet machine (the timing guard of the other timing scripts does not apply here: run it with the
untimed jobs paused). Also reports the per-query latency at each concurrency (wall time x threads / queries).

  PYTHONPATH=. .venv-ann/bin/python simulations/approximate_hits/throughput_backend.py --workload tripclick-ip \
      --threads 1 2 4 8 16 32 --n 4000
"""
import sys
import time
import argparse
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import numpy as np

from recal.measure import common_sample, write_json_atomic
from recal.measure import pagecache

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
RES = Path("simulations/approximate_hits/results")


def main():
    from simulations.approximate_hits.workloads import get
    from simulations.approximate_hits.diskann_aio import open_disk_index
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--n", type=int, default=4000, help="queries per measurement (post-warm-up requests, evenly spaced)")
    ap.add_argument("--nodes", type=int, default=None, help="node cache size (default: the workload's base node cache); "
                    "the equal-memory arm uses per_budget[budget].nodes_cached of timing_v6/<w>/backend_s50.json")
    ap.add_argument("--warmup", type=int, default=None,
                    help="stream warm-up; use the recorded runs' value (default: min(20000, len // 5))")
    ap.add_argument("--seed", type=int, default=0, help="query sample and arm order; same as throughput_replay.py")
    ap.add_argument("--legacy-sample", action="store_true", help="old behaviour: evenly spaced post-warm-up queries")
    ap.add_argument("--no-evict", action="store_true", help="old behaviour: no page cache eviction before the runs")
    ap.add_argument("--L", type=int, default=None, help="search list (default: the workload's)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    w = get(a.workload)
    info = w.backend_info()
    if a.L:
        info["L"] = a.L
    nodes = w.node_cache() if a.nodes is None else a.nodes
    z = np.load(w.prep_dir / "streams.npz")
    s = np.asarray(z[w.stream], dtype=np.int64)
    warm = min(20000, len(s) // 5) if a.warmup is None else a.warmup
    post = s[warm:]
    if a.legacy_sample:
        pick = post[np.linspace(0, len(post) - 1, a.n).astype(int)]
    else:
        pick = s[warm + common_sample(len(post), a.n, a.seed)]
    qf = w.prep_dir / "q.npy"
    q = np.load(qf if qf.exists() else Path("datasets/esci/data_vectors/esci_queries.npy"))
    xq = np.ascontiguousarray(q[pick].astype(np.int8 if w.name.startswith("spacev") else np.float32))
    tmax = max(a.threads)
    logger.info(f"{w.name}: L {info['L']} beam {info['W']} base node cache {nodes} nodes; {len(xq)} queries; open with {tmax} threads")
    ix = open_disk_index(max_wait=3600, index_directory=info["index_dir"], num_threads=tmax, num_nodes_to_cache=int(nodes),
                         cache_mechanism=1 if nodes else 0, index_prefix="ann")
    index_paths = [info["index_dir"]]
    state = {} if a.no_evict else {"evict": pagecache.evict(index_paths)}
    before = dict(pagecache.residency(index_paths), **state)
    ix.batch_search(xq[:200], 10, info["L"], num_threads=min(8, tmax), beam_width=info["W"])          # warm the page cache / node cache
    from simulations.approximate_hits.throughput_replay import node_cache_bytes
    res = {"workload": w.name, "L": info["L"], "beam": info["W"], "node_cache": nodes, "n": len(xq), "warmup": warm,
           "index_instances": 1, "node_cache_bytes_loaded": node_cache_bytes(info["index_dir"], int(nodes)),
           "sample_seed": a.seed, "legacy_sample": a.legacy_sample, "index_dir": info["index_dir"], "runs": []}
    res["page_cache"] = pagecache.record(before, None)
    for t in a.threads:
        best = None
        for rep in range(2):
            t0 = time.perf_counter()
            ix.batch_search(xq, 10, info["L"], num_threads=t, beam_width=info["W"])
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
        qps = len(xq) / best
        res["runs"].append({"threads": t, "wall_s": best, "qps": qps, "ms_per_query_per_thread": 1e3 * best * t / len(xq)})
        logger.info(f"threads {t:2d}: {qps:8.1f} queries/s  ({1e3 * best * t / len(xq):.2f} ms per query per thread)")
    out = Path(a.out) if a.out else RES / "throughput" / (f"backend_{w.name}" + ("" if a.nodes is None else f"_nodes{a.nodes}") + ".json")
    res["page_cache"]["after"] = pagecache.residency(index_paths)
    write_json_atomic(out, res)
    logger.info(f"wrote {out}")


if __name__ == "__main__":
    main()
