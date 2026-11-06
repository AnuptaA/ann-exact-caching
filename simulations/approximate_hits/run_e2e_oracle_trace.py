#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Best-case approximate-hit experiment on request traces (Aker workloads).

Workloads (datasets/aker/, see download.sh):
  spacev-aker-{0.3,0.6,0.99}   SPACEV 10M (100-d int8) + Aker simZipf traces
  triviaqa-aker-{0.3,0.6,0.99} SPHERE 10M (768-d) + Aker TriviaQA paraphrase traces
  spacev-real                  SPACEV 10M + SPACEV1B historical query log
                               (train) followed by the official query set (test)

Each trace is split in time order into consecutive chunks:
  cand   candidate-generating requests (their top-K balls are cache candidates)
  score  scoring requests (greedy utility)
  fit / dev / cal   predictor fitting, model selection, calibration
  test   held-out evaluation (also the arrival order for online admission)
Repeated vectors within a chunk are deduplicated and weighted.

Exact neighbors are computed once (exact_topk_blocked) for the globally
distinct request vectors and saved to <out>/topk.npz, so every later
evaluation uses identical neighbor lists.
"""

import json
import argparse
import logging
from pathlib import Path


import numpy as np

from utils.base.logs import setup_logging

from utils.cache.workloads import WORKLOADS, build_trace, load_base, split_view
from simulations.approximate_hits import run_e2e_oracle_sift as base
from utils.cache.population import GreedyPopulator
from simulations.approximate_hits.run_e2e_oracle_sift import (
    DEFAULT_K_VALUES, DEFAULT_TARGETS, plot_results, _print_target,
)
from utils.base.knn import exact_topk_blocked

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINTS = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]

#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------


def main():
    setup_logging()
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--trace", choices=sorted(WORKLOADS), required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES)
    ap.add_argument("--targets", type=float, nargs="+", default=DEFAULT_TARGETS)
    ap.add_argument("--checkpoints", type=float, nargs="+", default=DEFAULT_CHECKPOINTS)
    ap.add_argument("--max-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {args.trace}...")
    db = load_base(args.trace)
    uniq, inv, bounds = build_trace(args.trace)
    logger.info(f"  db={db.shape} {db.dtype}; requests={len(inv)} distinct={len(uniq)}; "
                + ", ".join(f"{s}={hi - lo}" for s, (lo, hi) in bounds.items()))

    topk_path = out_dir / "topk.npz"
    cand_ids = split_view(inv, bounds, "cand")[0]
    k_max = max(args.k_values)
    if topk_path.exists():
        z = np.load(topk_path)
        topk_all, cand_top = z["topk"], z["cand_top"]
        logger.info(f"Loaded saved neighbors from {topk_path}")
    else:
        logger.info(f"Exact top-{k_max} for {len(cand_ids)} candidate vectors...")
        cand_top = exact_topk_blocked(uniq[cand_ids], db, k_max)
        logger.info(f"Exact top-{args.k} for {len(uniq)} distinct request vectors...")
        topk_all = exact_topk_blocked(uniq, db, args.k)
        np.savez(topk_path, uniq=uniq, inv=inv, topk=topk_all, cand_top=cand_top,
                 **{f"bounds_{s}": np.array(b) for s, b in bounds.items()})
        logger.info(f"Saved {topk_path}")

    cand_vectors = [cand_top[i, :K] for i in range(len(cand_ids)) for K in args.k_values]
    score_ids, score_w, _ = split_view(inv, bounds, "score")
    test_ids, test_w, test_seq = split_view(inv, bounds, "test")
    score_topk = topk_all[score_ids][:, :args.k]
    test_topk = topk_all[test_ids][:, :args.k]
    n_db = len(db)
    max_size = int(args.max_frac * n_db)
    checkpoints_abs = sorted(int(c * n_db) for c in args.checkpoints if c <= args.max_frac)
    logger.info(f"{len(cand_vectors)} candidates; scoring distinct {len(score_ids)}; "
                f"test distinct {len(test_ids)}")

    shared = dict(cand_vectors=cand_vectors, score_topk=score_topk, test_topk=test_topk,
                  n_db=n_db, k=args.k, max_size=max_size, checkpoints_abs=checkpoints_abs,
                  score_weights=score_w, test_weights=test_w, test_sequence=test_seq)
    logger.info("Building shared greedy index...")
    shared["template"] = GreedyPopulator(cand_vectors, score_topk, n_db, args.k, 1.0,
                                         eval_topk=test_topk, score_weights=score_w,
                                         eval_weights=test_w)
    del db  # not needed for greedy; free memory before forking workers
    import multiprocessing as mp
    base._SHARED.update(shared)
    logger.info(f"Greedy population for targets {args.targets} ({args.workers} workers)...")
    with mp.get_context("fork").Pool(min(args.workers, len(args.targets))) as pool:
        results = pool.map(base._run_target_worker, args.targets)
    for res in results:
        _print_target(res)

    config = dict(vars(args), n_db=n_db,
                  split_requests={s: hi - lo for s, (lo, hi) in bounds.items()})
    with open(out_dir / "results.json", "w") as f:
        json.dump({"config": config, "results": results}, f)
    plot_results(results, out_dir, n_db, title=args.trace)
    logger.info(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
