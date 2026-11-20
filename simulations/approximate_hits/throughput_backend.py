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


def main():
    from simulations.approximate_hits.workloads import get
    from simulations.approximate_hits.diskann_aio import open_disk_index
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--n", type=int, default=4000, help="queries per measurement (post-warm-up requests, evenly spaced)")
    ap.add_argument("--nodes", type=int, default=None, help="node cache size (default: the workload's base node cache); "
                    "the equal-memory arm uses per_budget[budget].nodes_cached of timing_v6/<w>/backend_s50.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    w = get(a.workload)
    info = w.backend_info()
    nodes = w.node_cache() if a.nodes is None else a.nodes
    z = np.load(w.prep_dir / "streams.npz")
    s = np.asarray(z[w.stream], dtype=np.int64)
    warm = min(20000, len(s) // 5)
    post = s[warm:]
    pick = post[np.linspace(0, len(post) - 1, a.n).astype(int)]
    qf = w.prep_dir / "q.npy"
    q = np.load(qf if qf.exists() else Path("datasets/esci/data_vectors/esci_queries.npy"))
    xq = np.ascontiguousarray(q[pick].astype(np.float32))
    tmax = max(a.threads)
    logger.info(f"{w.name}: L {info['L']} beam {info['W']} base node cache {nodes} nodes; {len(xq)} queries; open with {tmax} threads")
    ix = open_disk_index(max_wait=3600, index_directory=info["index_dir"], num_threads=tmax, num_nodes_to_cache=int(nodes),
                         cache_mechanism=1 if nodes else 0, index_prefix="ann")
    ix.batch_search(xq[:200], 10, info["L"], num_threads=min(8, tmax), beam_width=info["W"])          # warm the page cache / node cache
    res = {"workload": w.name, "L": info["L"], "beam": info["W"], "node_cache": nodes, "n": len(xq), "runs": []}
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
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    logger.info(f"wrote {out}")


if __name__ == "__main__":
    main()
