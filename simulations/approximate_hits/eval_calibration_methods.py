#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Compare threshold-selection procedures for calibrated admission (B5).

Setting: admit a query iff predicted-retention score >= lambda. Choose lambda
on calibration queries so the true mean retention over admitted queries is
>= T with probability >= 1 - delta.

Two separate choices are compared:
  per-threshold lower bound   normal approximation (as in SUPG) or
                              empirical Bernstein (finite-sample)
  many thresholds             Bonferroni over the grid (SUPG Alg. 3),
                              fixed-order testing (BARGAIN Alg. 1: test from
                              strict to lenient at full level, stop at the
                              first failure), or a split (choose lambda on one
                              half, test it once on the other)

Validation by repeated trials: for each cache state, a large pool of fresh
queries is split many times into a calibration set and an evaluation set. The
evaluation set (tens of thousands of queries) stands in for the true
distribution; a trial fails if the admitted evaluation queries' mean
retention is below T. A valid method should fail in at most ~delta of trials.
"""

import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from simulations.approximate_hits import eval_calibrated as ecal
from simulations.approximate_hits.eval_certificates import build_entry_tables
from simulations.approximate_hits.run_e2e_oracle_sift import exact_topk, load_inputs
from simulations.approximate_hits.threshold_selection import (
    METHODS, select, threshold_grid,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

_W: dict = {}  # worker inputs

#-------------------------------------------------------------------------------


def load_pool(run_dir, n_fit):
    data = json.loads((run_dir / "results.json").read_text())
    cfg = argparse.Namespace(**data["config"])
    if "zipf" in data["config"]:
        from simulations.approximate_hits.run_e2e_oracle_synthetic import (
            build_inputs, _dedup,
        )
        inp = build_inputs(cfg)
        wl = inp["wl"]
        rng = np.random.default_rng(cfg.seed + 3)  # independent of other draws
        fit_ids, fit_w, _ = _dedup(wl.sample(n_fit, rng))
        pool_ids, pool_w, _ = _dedup(wl.sample(_W["n_pool_requests"], rng))
        sets = {"fit": (wl.vectors(fit_ids), fit_w),
                "pool": (wl.vectors(pool_ids), pool_w),
                "test": (wl.vectors(inp["test_ids"]), inp["test_w"])}
        return data, cfg, inp["db"], wl.vectors(inp["cand_ids"]), sets
    db, cand_q, _, test_q = load_inputs(cfg)
    pool = np.load(Path(cfg.esci_dir) / "esci_queries.npy")
    perm = np.random.default_rng(cfg.seed).permutation(len(pool))
    start = cfg.n_test + cfg.n_cand_queries + cfg.n_score_queries
    ones = lambda m: np.ones(m, dtype=np.int64)
    fit_q = pool[perm[start:start + n_fit]]
    pool_q = pool[perm[start + n_fit:]]
    sets = {"fit": (fit_q, ones(len(fit_q))), "pool": (pool_q, ones(len(pool_q))),
            "test": (test_q, ones(len(test_q)))}
    return data, cfg, db, cand_q, sets


def eval_state(task):
    target, point, entries = task
    G = ecal._G
    st = ecal.state_tables(entries)
    if len(st["cached"]) <= G["k"]:
        return None
    feats = {}
    for name in ("fit", "pool", "test"):
        q, w = G["sets"][name]
        y = st["in_cache"][G["topk"][name]].mean(axis=1)
        feats[name] = (ecal.features(q, st), y, w.astype(np.float64))
    predict = ecal.fit_ridge(*feats["fit"])
    s_fit = predict(feats["fit"][0])
    grid_desc = threshold_grid(s_fit)

    # expand pool and test to request level (weights = multiplicities)
    def expand(name):
        F, y, w = feats[name]
        idx = np.repeat(np.arange(len(y)), w.astype(np.int64))
        return predict(F)[idx], y[idx]

    s_pool, y_pool = expand("pool")
    s_test, y_test = expand("test")
    rng = np.random.default_rng(_W["seed"])
    stats = {m: {"fail": 0, "trials": 0, "eval_hit": [], "test_hit": [], "eval_ret": []}
             for m in METHODS}
    for _ in range(_W["trials"]):
        perm = rng.permutation(len(s_pool))
        cal, ev = perm[:_W["n_cal"]], perm[_W["n_cal"]:]
        for m in METHODS:
            lam = select(m, s_pool[cal], y_pool[cal], grid_desc, target,
                         _W["delta"], rng)
            rec = stats[m]
            rec["trials"] += 1
            if lam is None:
                rec["eval_hit"].append(0.0)
                rec["test_hit"].append(0.0)
                continue
            adm = s_pool[ev] >= lam
            ret = float(y_pool[ev][adm].mean()) if adm.any() else 1.0
            rec["eval_ret"].append(ret)
            rec["fail"] += int(ret < target)
            rec["eval_hit"].append(float(adm.mean()))
            rec["test_hit"].append(float((s_test >= lam).mean()))
    summary = {"target": target, "cache_frac": point["cache_frac"],
               "oracle_hit_rate_online": point["test_hit_rate_online"]}
    for m, rec in stats.items():
        summary[m] = {
            "fail_rate": rec["fail"] / rec["trials"],
            "mean_eval_hit_rate": float(np.mean(rec["eval_hit"])),
            "mean_test_hit_rate": float(np.mean(rec["test_hit"])),
            "mean_eval_retention": float(np.mean(rec["eval_ret"])) if rec["eval_ret"] else None,
        }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--targets", type=float, nargs="+", default=[0.9, 0.95])
    ap.add_argument("--sizes", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.30])
    ap.add_argument("--n-fit", type=int, default=20000)
    ap.add_argument("--n-cal", type=int, default=20000)
    ap.add_argument("--n-pool-requests", type=int, default=150000,
                    help="fresh workload requests for the pool (synthetic only)")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    _W.update(n_pool_requests=args.n_pool_requests, trials=args.trials,
              n_cal=args.n_cal, delta=args.delta, seed=args.seed)

    data, cfg, db, cand_q, sets = load_pool(run_dir, args.n_fit)
    logger.info(f"{run_dir.name}: " + ", ".join(
        f"{k}={len(v[0])} distinct ({int(v[1].sum())} requests)" for k, v in sets.items()))
    cached_ids, radius, _, _ = build_entry_tables(cand_q, db, cfg.k_values, 1)
    topk = {name: exact_topk(q, db, cfg.k) for name, (q, _) in sets.items()}
    ecal._G.update(k=cfg.k, k_values=cfg.k_values, db=db, cand_q=cand_q, sets=sets,
                   topk=topk, cached_ids=cached_ids, radius=radius)

    tasks = []
    for res in data["results"]:
        if not any(abs(res["target"] - t) < 1e-9 for t in args.targets):
            continue
        for size in args.sizes:
            p = min(res["points"], key=lambda p: abs(p["cache_frac"] - size))
            tasks.append((res["target"], p, res["selected"][:p["n_entries"]]))
    logger.info(f"{len(tasks)} states x {args.trials} trials")
    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(args.workers) as pool:
            out = pool.map(eval_state, tasks)
    else:
        out = [eval_state(t) for t in tasks]
    out = [o for o in out if o is not None]
    path = run_dir / "calibration_methods.json"
    path.write_text(json.dumps({"args": vars(args), "methods": METHODS, "states": out}))
    logger.info(f"Wrote {path}")


if __name__ == "__main__":
    main()
