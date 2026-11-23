#!/usr/bin/env python3
"""Reads per search and latency against DiskANN node cache size (runs on the cluster).

  python scripts/node_cache_curve.py --index-dir <dir> --queries q.npy --nodes 0 10000 100000 \\
      --node-bytes 368 --k 10 --search-list 128 --out curve.json
"""

import argparse
from pathlib import Path

import numpy as np

from recal.system_benefit import curves


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--nodes", type=int, nargs="+", required=True)
    ap.add_argument("--node-bytes", type=float, required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--search-list", type=int, default=128)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    from simulations.approximate_hits.eval_timing import open_diskann
    queries = np.load(a.queries, mmap_mode="r")[:a.n]
    table = curves.measure_node_cache_curve(lambda n: open_diskann(a.index_dir, n), np.asarray(queries),
                                            a.nodes, a.node_bytes, a.k, a.search_list)
    curves.save(a.out, table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
