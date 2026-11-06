#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
The 7-feature model on one saved run, at every cache checkpoint.

Model: logistic regression (eval_simple.logistic_scores) on relative offset
and relative size for the 3 best-fitting balls plus the intruder count
(sampled shell points inside the neighborhood; shell p, ranks up to R, b
bits). Baseline: the same model on the 6 ball features only (no sample).
Per checkpoint:
  - calibrated test hit rate and realized retention for both models, the
    oracle, and the shell sample's memory (fraction of the database)
  - calibration-size sweep (7-feature model): lambda chosen on random
    subsets of n calibration requests, --draws times per n; mean / min test
    hit rate and the fraction of draws with realized test retention < T
Memory-fair comparison (summarize with --summary): the 7-feature model at
total memory = cache + sample vs the balls-only model and the oracle
interpolated (in log size) at the same total memory from their checkpoints.

Writes <run>/final.json. Requires scikit-learn (.venv python).
"""

import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from utils.base.knn import knn_dists
from utils.cache.metrics import interp_log
from utils.cache.entries import state_tables
from utils.cache.features import N_BALLS, POOR_FIT, ball_features, balls_of
from utils.cache.metrics import expand_index
from utils.cache.models import logistic_scores
from utils.cache.parallel import worker_pool
from utils.cache.shell import SAMPLE_J, load_cand_ext, shell_features, shell_ids, shell_sample
from utils.cache.workloads import SCORE_SPLITS, load_run_data
from utils.cache.calibration import (
    DEFAULT_METHOD, select, threshold_grid,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

DELTA = 0.05

#-------------------------------------------------------------------------------


def admit(lam, s, y):
    adm = s >= lam if lam is not None else np.zeros(len(s), dtype=bool)
    return float(adm.mean()), (float(y[adm].mean()) if adm.any() else None)


def calibration_sweep(s_cal, y_cal, grid, s_test, y_test, target, ns, draws, rng):
    """lambda from random subsets of n calibration requests."""
    out = {}
    for n in ns:
        if n > len(s_cal):
            continue
        hits, low = [], 0
        for _ in range(draws):
            idx = rng.choice(len(s_cal), n, replace=False)
            lam = select(DEFAULT_METHOD, s_cal[idx], y_cal[idx], grid, target, DELTA)
            h, r = admit(lam, s_test, y_test)
            hits.append(h)
            low += r is not None and r < target
        out[str(n)] = {"mean_hit_rate": float(np.mean(hits)), "min_hit_rate": float(np.min(hits)),
                       "frac_retention_below_T": low / draws}
    return out


def eval_point(task, run, args):
    target, point, entries = task
    k = run.k
    st = state_tables(run, entries)
    if len(st["cached"]) <= k:
        return None
    bl = balls_of(run, entries)
    sets, topk = run.sets, run.topk
    y = {sp: st["in_cache"][topk[sp]].mean(axis=1) for sp in SCORE_SPLITS}
    w = {sp: sets[sp][1].astype(np.float64) for sp in SCORE_SPLITS}
    nr = {sp: np.maximum(knn_dists(sets[sp][0], st["db_c"], st["db_c_sq"], k)[:, k - 1], 1e-12) for sp in SCORE_SPLITS}
    B = {sp: np.nan_to_num(ball_features(sets[sp][0], nr[sp], bl, N_BALLS),
                           nan=POOR_FIT) for sp in SCORE_SPLITS}
    samp = shell_sample(run, bl, st["in_cache"], args.p, args.R, args.b)
    S = {sp: shell_features(sets[sp][0], nr[sp], samp)[:, [SAMPLE_J]] for sp in SCORE_SPLITS}
    ti, ci = expand_index(w["test"]), expand_index(w["cal"])
    out = {"cache_frac": point["cache_frac"], "oracle_hit_rate": point["test_hit_rate"],
           "sample_memory": samp["memory"] if samp else 0.0}
    for name, X in (("model7", {sp: np.hstack([B[sp], S[sp]]) for sp in SCORE_SPLITS}),
                    ("balls_only", B)):
        s = logistic_scores(X, y["fit"], w["fit"])
        grid = threshold_grid(s["fit"])
        lam = select(DEFAULT_METHOD, s["cal"][ci], y["cal"][ci], grid, target, DELTA)
        h, r = admit(lam, s["test"][ti], y["test"][ti])
        out[name] = {"hit_rate": h, "hit_retention": r}
        if name == "model7":
            out["calibration_sweep"] = calibration_sweep(
                s["cal"][ci], y["cal"][ci], grid, s["test"][ti], y["test"][ti], target,
                args.cal_sizes, args.draws, np.random.default_rng(0))
    return out


def summarize(path):
    d = json.loads(Path(path).read_text())
    S = sorted(d["states"], key=lambda s: s["cache_frac"])
    xs = [s["cache_frac"] for s in S]
    f = lambda v: "  -- " if v is None else f"{100 * v:5.1f}"
    print(f"{Path(path).parent.name}: 7-feature model vs balls-only and oracle")
    print(f"{'cache':>7s} {'+mem':>6s} | {'oracle':>6s} {'7feat':>6s} {'ret':>5s} {'balls':>6s} | "
          f"at total memory: {'balls':>6s} {'oracle':>6s} | cal n -> mean hit (frac ret < T)")
    for s in S:
        tot = s["cache_frac"] + s["sample_memory"]
        bm = interp_log(xs, [x["balls_only"]["hit_rate"] for x in S], tot)
        om = interp_log(xs, [x["oracle_hit_rate"] for x in S], tot)
        m7 = s["model7"]
        sweep = " ".join(f"{n}:{f(v['mean_hit_rate'])}({v['frac_retention_below_T']:.2f})"
                         for n, v in s["calibration_sweep"].items())
        print(f"{100 * s['cache_frac']:6.2f}% {100 * s['sample_memory']:5.2f}% | "
              f"{f(s['oracle_hit_rate'])} {f(m7['hit_rate'])} {m7['hit_retention'] or 0:5.3f} "
              f"{f(s['balls_only']['hit_rate'])} |                  {f(bm)} {f(om)} | {sweep}")


#-------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--p", type=float, default=0.25)
    ap.add_argument("--R", type=int, default=500)
    ap.add_argument("--b", type=int, default=4)
    ap.add_argument("--cal-sizes", type=int, nargs="+",
                    default=[500, 1000, 2000, 5000, 10000, 20000])
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--n-each", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--summary", action="store_true", help="only print run_dir/final.json")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    if args.summary:
        summarize(run_dir / "final.json")
        return
    data, _, run = load_run_data(run_dir, args.n_each)
    res = next(r for r in data["results"] if abs(r["target"] - args.target) < 1e-9)
    tasks = [(args.target, p, res["selected"][:p["n_entries"]])
             for p in sorted(res["points"], key=lambda p: p["cache_frac"]) if p["n_entries"]]
    out = []
    with worker_pool(eval_point, args.workers, len(tasks), run=run, args=args) as pool:
        for s in pool.imap(tasks):
            if s is None:
                continue
            out.append(s)
            logger.info(f"cache {100 * s['cache_frac']:.2f}%: oracle {s['oracle_hit_rate']:.3f} "
                        f"7feat {s['model7']['hit_rate']:.3f} balls {s['balls_only']['hit_rate']:.3f}"
                        f" (+{100 * s['sample_memory']:.2f}% memory)")
            (run_dir / "final.json").write_text(
                json.dumps({"args": vars(args), "states": out}, indent=1))


if __name__ == "__main__":
    main()
