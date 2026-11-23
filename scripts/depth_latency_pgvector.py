#!/usr/bin/env python3
"""pgvector latency by search depth, in the table format of latency_probe.py --depths (cluster only).

  python scripts/depth_latency_pgvector.py --dsn <dsn> --queries q.npy --ks 10 100 --out pg.json
"""

import argparse
from pathlib import Path

import numpy as np

from recal.system_benefit import curves


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[10])
    ap.add_argument("--delta", type=int, default=0)
    ap.add_argument("--ef-search", type=int, default=1000)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    import psycopg
    from harness.backend.pgvector import PgvectorBackend, with_keepalives
    backend = PgvectorBackend(psycopg.connect(with_keepalives(a.dsn)))
    queries = np.asarray(np.load(a.queries, mmap_mode="r")[:a.n], dtype=np.float32)
    depths = curves.table_depths(a.ks, a.delta)
    table = curves.measure_depth_latency(lambda q, depth: backend.search(q[None], depth, a.ef_search), queries, depths)
    curves.save(a.out, table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
