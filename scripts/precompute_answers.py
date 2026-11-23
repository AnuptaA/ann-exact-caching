#!/usr/bin/env python3
"""Backend answers to depth 7 x K_max and exact ground truth to depth 200, per workload.

  python -m scripts.precompute_answers --backend pgvector --dsn <dsn> --queries q.npy \\
      --base base.npy --kmax 100 --out answers/spacev --ef-search 1000
  python -m scripts.precompute_answers --backend diskann --index-dir <dir> --dtype int8 ...
  python -m scripts.precompute_answers --backend exact ... --gt-out gt/spacev

Answer tables use harness/backend/answer_table.py (resumable blocks, distances recomputed in
float32). Ground truth uses workloads/q1/groundtruth.py. Runs per workload on the cluster.
"""

import argparse
from pathlib import Path

import numpy as np

ANSWER_DEPTH_FACTOR = 7
GT_DEPTH = 200


def answer_depth(k_max: int) -> int:
    return ANSWER_DEPTH_FACTOR * k_max


def make_backend(args, base):
    if args.backend == "exact":
        from harness.backend.exact_table import ExactTableBackend
        return ExactTableBackend(base)
    if args.backend == "diskann":
        from harness.backend.diskann_search import DiskannBackend
        return DiskannBackend.open(args.index_dir, args.dtype)
    import psycopg
    from harness.backend.pgvector import PgvectorBackend, with_keepalives
    return PgvectorBackend(psycopg.connect(with_keepalives(args.dsn)))


def precompute_answers(backend, queries, base, out_dir: Path, k_max: int, ef_search: int,
                       block: int = 2000) -> int:
    from harness.backend.answer_table import build_answer_table, validate_blocks
    depth = answer_depth(k_max)
    build_answer_table(backend, queries, base, Path(out_dir), depth, ef_search, block=block)
    validate_blocks(Path(out_dir), len(base), depth, len(queries))
    return depth


def precompute_ground_truth(queries, base, out_dir: Path, depth: int = GT_DEPTH,
                            device: str = "auto") -> None:
    from workloads.q1.groundtruth import build_ground_truth
    build_ground_truth(queries, base, Path(out_dir), k=depth, device=device)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["exact", "pgvector", "diskann"], default="exact")
    ap.add_argument("--dsn")
    ap.add_argument("--index-dir")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--kmax", type=int, default=100)
    ap.add_argument("--ef-search", type=int, default=1000)
    ap.add_argument("--block", type=int, default=2000)
    ap.add_argument("--out", type=Path, help="answer table directory")
    ap.add_argument("--gt-out", type=Path, help="exact ground truth directory (depth 200)")
    ap.add_argument("--gt-depth", type=int, default=GT_DEPTH)
    ap.add_argument("--device", default="auto")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    queries = np.load(args.queries, mmap_mode="r")
    base = np.load(args.base, mmap_mode="r")
    if args.out:
        precompute_answers(make_backend(args, base), queries, base, args.out, args.kmax,
                           args.ef_search, args.block)
    if args.gt_out:
        precompute_ground_truth(queries, base, args.gt_out, args.gt_depth, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
