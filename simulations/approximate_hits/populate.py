#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Greedy cache population with oracle retention (best case), one subcommand per workload.

  populate sift       SIFT1M database, learn / query sets as the query workload
  populate esci       ESCI product subsample and the real ESCI queries
  populate synthetic  SIFT with simZipf request streams (similar-query groups, Zipf skew)
  populate trace      Aker request traces (SPACEV, TriviaQA on SPHERE) and the SPACEV1B log

Each writes results.json (config, per-target greedy curves, selected entries) and
hit-rate plots into --out. Candidate entries are the top-K balls of candidate-
generating queries for several K; greedy adds the entry with the largest gain in
scoring-query hits per added vector under the hits-only average-retention policy.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from simulations.approximate_hits.common import config_of
from simulations.approximate_hits.report.plots import plot_results, print_target
from utils.base.knn import exact_topk, exact_topk_blocked
from utils.cache.population import run_targets
from utils.cache.workloads import (
    PATHS, WORKLOADS, build_inputs, build_trace, load_base, load_inputs, split_view,
)

logger = logging.getLogger(__name__)

DEFAULT_TARGETS = [0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
DEFAULT_K_VALUES = [10, 20, 50, 100, 200, 500]
DEFAULT_CHECKPOINTS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
TRACE_CHECKPOINTS = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]

#-------------------------------------------------------------------------------


def _common_args(p, checkpoints, max_frac, workers, out_required=False):
    p.add_argument("--k", type=int, default=10, help="test query result count")
    p.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES,
                   help="candidate ball sizes K (strictly increasing)")
    p.add_argument("--targets", type=float, nargs="+", default=DEFAULT_TARGETS)
    p.add_argument("--checkpoints", type=float, nargs="+", default=checkpoints)
    p.add_argument("--max-frac", type=float, default=max_frac)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=workers,
                   help="parallel processes over targets")
    p.add_argument("--out", type=str, required=out_required, default=None,
                   help="output directory (default: results/<timestamp> next to this file)")


def add_args(parser):
    sub = parser.add_subparsers(dest="workload", required=True)
    for name in ("sift", "esci"):
        p = sub.add_parser(name, help=f"{name.upper()} database and real queries")
        p.set_defaults(dataset=name, run=run_vectors)
        if name == "esci":
            p.add_argument("--esci-dir", type=str, default=str(PATHS.esci))
        p.add_argument("--n-db", type=int, default=50000)
        p.add_argument("--n-cand-queries", type=int, default=5000)
        p.add_argument("--n-score-queries", type=int, default=5000)
        p.add_argument("--n-test", type=int, default=10000)
        _common_args(p, DEFAULT_CHECKPOINTS, 0.3, 1)
    p = sub.add_parser("synthetic", help="SIFT with simZipf request streams")
    p.set_defaults(run=run_synthetic)
    p.add_argument("--zipf", type=float, default=0.99, help="Zipf exponent s")
    p.add_argument("--method", choices=["interp", "gauss"], default="interp")
    p.add_argument("--lambda-max", type=float, default=0.5)
    p.add_argument("--noise-rel", type=float, default=0.1)
    p.add_argument("--group-size", type=int, default=50)
    p.add_argument("--n-db", type=int, default=50000)
    p.add_argument("--n-cand-requests", type=int, default=5000)
    p.add_argument("--n-score-requests", type=int, default=20000)
    p.add_argument("--n-test-requests", type=int, default=100000)
    _common_args(p, DEFAULT_CHECKPOINTS, 0.3, 6)
    p = sub.add_parser("trace", help="Aker request traces")
    p.set_defaults(run=run_trace)
    p.add_argument("--trace", choices=sorted(WORKLOADS), required=True)
    _common_args(p, TRACE_CHECKPOINTS, 0.05, 6, out_required=True)


def _check_k_values(args):
    kv = args.k_values
    if kv[0] < 1 or any(a >= b for a, b in zip(kv, kv[1:])):
        raise SystemExit(f"--k-values must be positive and strictly increasing, got {kv}")


def _schedule(args, n_db):
    max_size = int(args.max_frac * n_db)
    return max_size, sorted(int(c * n_db) for c in args.checkpoints if c <= args.max_frac)


def _config(args, **extra):
    return dict(config_of(args), **extra)


def _finish(args, results, n_db, title, config, **top):
    for res in results:
        print_target(res)
    out_dir = Path(args.out) if args.out else (
        Path(__file__).parent / "results" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump({"config": config, **top, "results": results}, f)
    plot_results(results, out_dir, n_db, title=title)
    logger.info(f"Wrote {out_dir}")


def _candidates(cand_topk, k_values):
    return [cand_topk[i, :K] for i in range(len(cand_topk)) for K in k_values]


def run_vectors(args):
    """sift and esci."""
    _check_k_values(args)
    db, cand_q, score_q, test = load_inputs(args)
    logger.info(f"  db={db.shape} cand_queries={cand_q.shape} "
                f"score_queries={score_q.shape} test={test.shape}")
    logger.info(f"Exact top-{max(args.k_values)} for candidate queries...")
    cand_topk = exact_topk(cand_q, db, max(args.k_values))
    logger.info(f"Exact top-{args.k} for scoring and test queries...")
    score_topk = exact_topk(score_q, db, args.k)
    test_topk = exact_topk(test, db, args.k)
    cand_vectors = _candidates(cand_topk, args.k_values)
    logger.info(f"{len(cand_vectors)} candidates ({len(cand_q)} queries x K in {args.k_values})")
    max_size, checkpoints_abs = _schedule(args, args.n_db)
    results = run_targets(args.targets, args.workers, cand_vectors=cand_vectors,
                          score_topk=score_topk, test_topk=test_topk, n_db=args.n_db, k=args.k,
                          max_size=max_size, checkpoints_abs=checkpoints_abs)
    _finish(args, results, args.n_db, args.dataset.upper(), _config(args))


def run_synthetic(args):
    _check_k_values(args)
    inp = build_inputs(args)
    db, wl, stats = inp["db"], inp["wl"], inp["stats"]
    logger.info(f"Workload stats: {stats}")
    k_max = max(args.k_values)
    logger.info(f"Exact top-{k_max} for {len(inp['cand_ids'])} candidate queries...")
    cand_topk = exact_topk(wl.vectors(inp["cand_ids"]), db, k_max)
    logger.info(f"Exact top-{args.k} for scoring and test queries...")
    score_topk = exact_topk(wl.vectors(inp["score_ids"]), db, args.k)
    test_topk = exact_topk(wl.vectors(inp["test_ids"]), db, args.k)
    cand_vectors = _candidates(cand_topk, args.k_values)
    logger.info(f"{len(cand_vectors)} candidates")
    max_size, checkpoints_abs = _schedule(args, args.n_db)
    results = run_targets(args.targets, args.workers, cand_vectors=cand_vectors,
                          score_topk=score_topk, test_topk=test_topk, n_db=args.n_db, k=args.k,
                          max_size=max_size, checkpoints_abs=checkpoints_abs,
                          score_weights=inp["score_w"], test_weights=inp["test_w"],
                          test_sequence=inp["test_seq"])
    title = f"SIFT simZipf s={args.zipf} ({args.method})"
    _finish(args, results, args.n_db, title, _config(args), workload_stats=stats)


def _saved_neighbors(path, uniq, k, k_max):
    z = np.load(path)
    topk_all, cand_top = z["topk"], z["cand_top"]
    if len(topk_all) != len(uniq) or topk_all.shape[1] < k or cand_top.shape[1] < k_max:
        raise SystemExit(f"{path} does not match this trace or --k/--k-values "
                         f"(top-k {topk_all.shape}, candidate table {cand_top.shape}); delete it "
                         f"to recompute")
    return topk_all, cand_top


def run_trace(args):
    _check_k_values(args)
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
        topk_all, cand_top = _saved_neighbors(topk_path, uniq, args.k, k_max)
        logger.info(f"Loaded saved neighbors from {topk_path}")
    else:
        logger.info(f"Exact top-{k_max} for {len(cand_ids)} candidate vectors...")
        cand_top = exact_topk_blocked(uniq[cand_ids], db, k_max)
        logger.info(f"Exact top-{args.k} for {len(uniq)} distinct request vectors...")
        topk_all = exact_topk_blocked(uniq, db, args.k)
        np.savez(topk_path, uniq=uniq, inv=inv, topk=topk_all, cand_top=cand_top,
                 **{f"bounds_{s}": np.array(b) for s, b in bounds.items()})
        logger.info(f"Saved {topk_path}")
    cand_vectors = _candidates(cand_top[:len(cand_ids)], args.k_values)
    score_ids, score_w, _ = split_view(inv, bounds, "score")
    test_ids, test_w, test_seq = split_view(inv, bounds, "test")
    n_db = len(db)
    max_size, checkpoints_abs = _schedule(args, n_db)
    logger.info(f"{len(cand_vectors)} candidates; scoring distinct {len(score_ids)}; "
                f"test distinct {len(test_ids)}")
    del db  # not needed for greedy; free memory before forking workers
    results = run_targets(args.targets, args.workers, cand_vectors=cand_vectors,
                          score_topk=topk_all[score_ids][:, :args.k],
                          test_topk=topk_all[test_ids][:, :args.k], n_db=n_db, k=args.k,
                          max_size=max_size, checkpoints_abs=checkpoints_abs,
                          score_weights=score_w, test_weights=test_w, test_sequence=test_seq)
    config = _config(args, n_db=n_db, split_requests={s: hi - lo for s, (lo, hi) in bounds.items()})
    _finish(args, results, n_db, args.trace, config)
