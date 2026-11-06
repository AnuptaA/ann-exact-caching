#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Larger shell samples, and how many calibration queries lambda needs.

Per cache checkpoint of one saved run (greedy cache for --target), the
7-feature logistic model (3 balls + intruder count, eval_final) with shell
samples at every (p, R) in --ps x --Rs (R > 500 uses exact top-R neighbor
lists of the candidate queries, computed once and saved):
  test hit rate, realized retention, sample memory
Calibration-size study, default shell (p = 0.25, R = 500): the cal, dev and
test requests are pooled and re-split at random --draws times per n: n
requests calibrate lambda (each method in --methods), the rest evaluate it.
Reports mean hit rate and the fraction of draws whose admitted mean
retention on the evaluation part is below T and below T - 0.01.

Writes <run>/shell_size.json. Requires scikit-learn (.venv python).
"""

import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from simulations.approximate_hits import eval_calibrated as ecal
from simulations.approximate_hits import eval_core as ec
from simulations.approximate_hits import eval_predictors as epred
from simulations.approximate_hits import eval_simple as es
from simulations.approximate_hits.eval_radius_scores import cached_knn
from simulations.approximate_hits.threshold_selection import select, threshold_grid

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SPLITS = ("fit", "dev", "cal", "test")
DELTA = 0.05
BITS = 4
BASE = (0.25, 500)
_R: dict = {}

#-------------------------------------------------------------------------------


def _expand(y, w):
    return np.repeat(np.arange(len(y)), w.astype(np.int64))


def _admit(lam, s, y):
    adm = s >= lam if lam is not None else np.zeros(len(s), dtype=bool)
    return float(adm.mean()), (float(y[adm].mean()) if adm.any() else None)


def cal_study(s_pool, y_pool, grid, target, ns, draws, methods, rng):
    """Repeated random re-splits of the pooled cal + dev + test requests: n
    requests calibrate lambda, all the others evaluate it (fresh, independent
    calibration sets, so the below-T frequency estimates the failure rate)."""
    out = {m: {} for m in methods}
    for n in ns:
        if n >= len(s_pool):
            continue
        res = {m: ([], 0, 0) for m in methods}
        for _ in range(draws):
            perm = rng.permutation(len(s_pool))
            c, e = perm[:n], perm[n:]
            for m in methods:
                lam = select(m, s_pool[c], y_pool[c], grid, target, DELTA)
                h, r = _admit(lam, s_pool[e], y_pool[e])
                hits, below, below01 = res[m]
                hits.append(h)
                res[m] = (hits, below + (r is not None and r < target),
                          below01 + (r is not None and r < target - 0.01))
        for m in methods:
            hits, below, below01 = res[m]
            out[m][str(n)] = {"mean_hit_rate": float(np.mean(hits)),
                              "frac_below_T": below / draws, "frac_below_T_0.01": below01 / draws}
    return out


def eval_point(task):
    target, point, entries = task
    args = _R["args"]
    k = ecal._G["k"]
    st = ecal.state_tables(entries)
    if len(st["cached"]) <= k:
        return None
    bl = ec.balls_of(entries)
    sets, topk = ecal._G["sets"], ecal._G["topk"]
    y = {sp: st["in_cache"][topk[sp]].mean(axis=1) for sp in SPLITS}
    w = {sp: sets[sp][1].astype(np.float64) for sp in SPLITS}
    nr = {sp: np.maximum(cached_knn(sets[sp][0], st)[:, k - 1], 1e-12) for sp in SPLITS}
    B = {sp: np.nan_to_num(ec.ball_features(sets[sp][0], nr[sp], bl, es.N_BALLS),
                           nan=es.POOR_FIT) for sp in SPLITS}
    ex = {sp: _expand(y[sp], w[sp]) for sp in SPLITS}
    out = {"cache_frac": point["cache_frac"], "oracle_hit_rate": point["test_hit_rate"],
           "shells": {}}
    for R in args.Rs:
        for p in args.ps:
            samp = ec.shell_sample(bl, st["in_cache"], p, R, BITS)
            S = {sp: ec.shell_features(sets[sp][0], nr[sp], samp)[:, [ec.SAMPLE_J]]
                 for sp in SPLITS}
            X = {sp: np.hstack([B[sp], S[sp]]) for sp in SPLITS}
            s = es.logistic_scores(X, y["fit"], w["fit"])
            grid = threshold_grid(s["fit"])
            lam = select("normal_fixed", s["cal"][ex["cal"]], y["cal"][ex["cal"]], grid,
                         target, DELTA)
            h, r = _admit(lam, s["test"][ex["test"]], y["test"][ex["test"]])
            out["shells"][f"p={p:g},R={R}"] = {"hit_rate": h, "hit_retention": r,
                                               "sample_memory": samp["memory"] if samp else 0.0}
            if (p, R) == BASE:
                pool = ("cal", "dev", "test")
                s_pool = np.concatenate([s[sp][ex[sp]] for sp in pool])
                y_pool = np.concatenate([y[sp][ex[sp]] for sp in pool])
                out["cal_study"] = cal_study(s_pool, y_pool, grid, target, args.cal_sizes,
                                             args.draws, args.methods, np.random.default_rng(0))
    return out


#-------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--sizes", type=float, nargs="+", required=True)
    ap.add_argument("--ps", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--Rs", type=int, nargs="+", default=[500, 1000, 2000])
    ap.add_argument("--cal-sizes", type=int, nargs="+", default=[250, 500, 1000, 2000, 5000, 10000])
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--methods", nargs="+",
                    default=["normal_fixed", "normal_fixed_min30", "eb_fixed"])
    ap.add_argument("--n-each", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-name", type=str, default="shell_size.json")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    data, _ = epred.setup_run(run_dir, args.n_each)
    _R["args"] = args
    ec._R["draw"] = np.random.default_rng(1).random(len(ecal._G["db"]))  # as eval_core
    if max(args.Rs) > 500:
        ec._R["cand_ext"] = ec.load_cand_ext(run_dir, max(args.Rs))
    res = next(r for r in data["results"] if abs(r["target"] - args.target) < 1e-9)
    tasks = []
    for size in args.sizes:
        p = min(res["points"], key=lambda p: abs(p["cache_frac"] - size))
        tasks.append((args.target, p, res["selected"][:p["n_entries"]]))
    import multiprocessing as mp
    out = []
    with mp.get_context("fork").Pool(min(args.workers, len(tasks))) as pool:
        for s in pool.imap(eval_point, tasks):
            if s is None:
                continue
            out.append(s)
            logger.info(f"cache {100 * s['cache_frac']:.2f}% (oracle {s['oracle_hit_rate']:.3f}): "
                        + " ".join(f"{n} {v['hit_rate']:.3f}(+{100 * v['sample_memory']:.2f}%)"
                                   for n, v in s["shells"].items()))
            (run_dir / args.out_name).write_text(
                json.dumps({"args": vars(args), "states": out}, indent=1))


if __name__ == "__main__":
    main()
