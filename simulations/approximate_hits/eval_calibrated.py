#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Calibrated admission (B5) on saved greedy caches.

For each cache state (checkpoint of a saved greedy run) and target T:
  1. Features: cheap per-query quantities computed from the cache only (no
     database access): distances to cached vectors, depth inside the nearest
     cache ball, certificate quantities, etc.
  2. Predictor: ridge regression (features + squares) -> retention score,
     fit on FIT queries (fresh draws from the workload).
  3. Threshold: admit a query iff score >= lambda. lambda is chosen on
     CALIBRATION queries (fresh, independent draws) so that, with probability
     ~>= 1 - delta, the true mean retention over admitted queries is >= T.
     Default procedure (utils/cache/calibration.py): normal-approximation lower
     bound per threshold, fixed-order testing over a grid of FIT-score
     quantiles from strict to lenient (stop at the first failure).

Fit and calibration queries never overlap with the scoring queries greedy used
to choose the cache, so they are not biased by the cache selection.
Also reported: "naive" lambda (empirical calibration mean >= T, no confidence
bound), and the realized mean retention over hits on the test queries.
"""

import json
import argparse
import logging
from pathlib import Path


import numpy as np

from utils.base.logs import setup_logging

from utils.cache.entries import state_tables
from utils.cache.legacy_features import FEATURE_NAMES, features
from utils.cache.models import fit_ridge
from utils.cache.parallel import worker_pool
from utils.cache.workloads import load_fit_cal_splits, load_trace_run, make_run_data
from utils.cache.calibration import (
    DEFAULT_METHOD, METHODS, select, threshold_grid,
)

logger = logging.getLogger(__name__)

#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------


def _evaluate(s, y, w, lam):
    total = float(w.sum())
    if lam is None:
        return {"hit_rate": 0.0, "hit_retention": None, "effective_retention": 1.0}
    adm = s >= lam
    wa = float(w[adm].sum())
    kept = float((w * y)[adm].sum())
    return {
        "hit_rate": wa / total,
        "hit_retention": kept / wa if wa > 0 else None,
        "effective_retention": 1.0 - wa / total + kept / total,
    }


def _spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def eval_target(res, run, args):
    target = res["target"]
    rows = []
    for p in res["points"]:
        entries = res["selected"][:p["n_entries"]]
        row = {"cache_frac": p["cache_frac"], "n_entries": p["n_entries"],
               "oracle_hit_rate_online": p["test_hit_rate_online"],
               "oracle_hit_rate": p["test_hit_rate"]}
        if len(entries) == 0:
            rows.append(row)
            continue
        st = state_tables(run, entries)
        if len(st["cached"]) <= run.k:
            rows.append(row)
            continue
        Fy = {}
        for name in ("fit", "cal", "test"):
            q, w = run.sets[name]
            y = st["in_cache"][run.topk[name]].mean(axis=1)
            Fy[name] = (features(run, q, st), y, w.astype(np.float64))
        predict = fit_ridge(*Fy["fit"])
        s_fit = predict(Fy["fit"][0])
        grid_desc = threshold_grid(s_fit)
        F_cal, y_cal, w_cal = Fy["cal"]
        rep = np.repeat(np.arange(len(y_cal)), w_cal.astype(np.int64))
        s_cal = predict(F_cal)[rep]
        y_cal = y_cal[rep]
        lam_g = select(args.method, s_cal, y_cal, grid_desc, target, args.delta)
        lam_n = select("naive", s_cal, y_cal, grid_desc, target, args.delta)
        s_test = predict(Fy["test"][0])
        y_test, w_test = Fy["test"][1], Fy["test"][2]
        for tag, lam in (("guaranteed", lam_g), ("naive", lam_n)):
            for key, val in _evaluate(s_test, y_test, w_test, lam).items():
                row[f"{tag}_{key}"] = val
        row["test_spearman"] = _spearman(s_test, y_test)
        rows.append(row)
    return {"target": target, "rows": rows}


def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--n-fit", type=int, default=20000)
    ap.add_argument("--n-calib", type=int, default=20000)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--method", choices=METHODS, default=DEFAULT_METHOD,
                    help="threshold-selection procedure for the guaranteed threshold")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--targets", type=float, nargs="+", default=None,
                    help="only evaluate these targets (default: all in the run)")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    run_cfg = json.loads((run_dir / "results.json").read_text())["config"]
    cand_top = pre_topk = None
    if "trace" in run_cfg:
        # trace runs: fixed time-ordered splits and saved exact neighbors
        t = load_trace_run(run_dir)
        data, cfg, db, cand_q = t["data"], t["cfg"], t["db"], t["cand_q"]
        sets = {s: t["sets"][s] for s in ("fit", "cal", "test")}
        pre_topk = {s: t["topk"][s] for s in sets}
        cand_top = t["cand_top"]
    else:
        data, cfg, db, cand_q, sets = load_fit_cal_splits(run_dir, args.n_fit, args.n_calib)
    if args.targets is not None:
        data["results"] = [r for r in data["results"]
                           if any(abs(r["target"] - t) < 1e-9 for t in args.targets)]
    logger.info(f"{run_dir.name}: " + ", ".join(
        f"{k}={len(v[0])} distinct ({int(v[1].sum())} total)" for k, v in sets.items()))
    run = make_run_data(cfg, db, cand_q, sets, cand_top, pre_topk)

    logger.info(f"Evaluating {len(data['results'])} targets...")
    with worker_pool(eval_target, args.workers, len(data["results"]), run=run, args=args) as pool:
        out = pool.map(data["results"])
    path = run_dir / "calibrated.json"
    path.write_text(json.dumps({"delta": args.delta, "method": args.method,
                                "n_fit": args.n_fit,
                                "n_calib": args.n_calib, "features": FEATURE_NAMES,
                                "results": out}))
    logger.info(f"Wrote {path}")


if __name__ == "__main__":
    main()
