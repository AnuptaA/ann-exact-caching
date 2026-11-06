#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Which shell-sample features matter? Logistic regression (eval_simple) on the
3-ball stats plus subsets of the 8 shell features (eval_core.shell_features):
  d1..d5     distances to the 5 nearest sampled points / neighborhood radius
  n1.0       sampled points within 1.0 x neighborhood radius (seen intruders)
  n1.1       ... within 1.1 x
  n1.25      ... within 1.25 x
Also the GBT on balls + all 8 as a reference. Per configuration: calibrated
and best-cutoff test hit rate, plus uncalibrated quality of the model's own
predicted retention (logistic: sigmoid of the decision value):
  naive      admit iff predicted retention >= T: hit rate and realized retention
  mae        mean |predicted - true retention| over all test queries
  spearman   rank correlation of predicted vs true retention
  top_bins   mean predicted vs mean true retention among the top 5 / 10 / 20%
             of test queries by predicted retention (honesty where admission
             happens) Writes <run>/shell_ablation.json.
Requires scikit-learn (.venv python).
"""

import json
import argparse
import logging
from pathlib import Path


import numpy as np

from utils.base.logs import setup_logging

from utils.cache.models import gbt_regressor
from utils.cache.metrics import best_cutoff_rate
from utils.cache.entries import state_tables
from utils.cache.features import N_BALLS, POOR_FIT, ball_features, balls_of, neighborhood_radius
from utils.cache.metrics import expand_index
from utils.cache.models import logistic_scores
from utils.cache.parallel import worker_pool
from utils.cache.shell import shell_features, shell_sample
from utils.cache.workloads import SCORE_SPLITS, load_run_data
from utils.cache.calibration import (
    DEFAULT_METHOD, select, threshold_grid,
)

logger = logging.getLogger(__name__)

SHELL_NAMES = ["d1", "d2", "d3", "d4", "d5", "n1.0", "n1.1", "n1.25"]
CONFIGS = {
    "balls only": [],
    "all 8": SHELL_NAMES,
    "n1.0": ["n1.0"],
    "d1": ["d1"],
    "n1.0 + d1": ["n1.0", "d1"],
    "n1.0 + d1 + n1.25": ["n1.0", "d1", "n1.25"],
    "counts (n1.0, n1.1, n1.25)": ["n1.0", "n1.1", "n1.25"],
    "distances (d1..d5)": ["d1", "d2", "d3", "d4", "d5"],
    "d1 + d2": ["d1", "d2"],
    "counts + d1": ["n1.0", "n1.1", "n1.25", "d1"],
}

#-------------------------------------------------------------------------------


def _rank(a):
    return np.argsort(np.argsort(a, kind="stable"), kind="stable")


def _evaluate(s, y, ti, ci, target, pred=None):
    """s: scores per split; pred: predicted retention on test (uncalibrated)."""
    lam = select(DEFAULT_METHOD, s["cal"][ci], y["cal"][ci], threshold_grid(s["fit"]),
                 target, 0.05)
    st, yt = s["test"][ti], y["test"][ti]
    adm = st >= lam if lam is not None else np.zeros(len(st), dtype=bool)
    out = {"calibrated_hit_rate": float(adm.mean()),
           "calibrated_hit_retention": float(yt[adm].mean()) if adm.any() else None,
           "best_cutoff_hit_rate": best_cutoff_rate(st, yt, target)}
    pt = pred[ti]
    nv = pt >= target
    order = np.argsort(-pt, kind="stable")
    out.update({
        "naive_hit_rate": float(nv.mean()),
        "naive_hit_retention": float(yt[nv].mean()) if nv.any() else None,
        "mae": float(np.abs(pt - yt).mean()),
        "spearman": float(np.corrcoef(_rank(pt), _rank(yt))[0, 1]),
        "top_bins": {f"{q:g}": [float(pt[order[:int(q * len(pt))]].mean()),
                                float(yt[order[:int(q * len(pt))]].mean())]
                     for q in (0.05, 0.1, 0.2)}})
    return out


def eval_state(task, run, args):
    target, point, entries = task
    k = run.k
    st = state_tables(run, entries)
    if len(st["cached"]) <= k:
        return None
    bl = balls_of(run, entries)
    sets, topk = run.sets, run.topk
    y = {sp: st["in_cache"][topk[sp]].mean(axis=1) for sp in SCORE_SPLITS}
    w = {sp: sets[sp][1].astype(np.float64) for sp in SCORE_SPLITS}
    nr = {sp: neighborhood_radius(sets[sp][0], st, k) for sp in SCORE_SPLITS}
    B = {sp: np.nan_to_num(ball_features(sets[sp][0], nr[sp], bl, N_BALLS),
                           nan=POOR_FIT) for sp in SCORE_SPLITS}
    samp = shell_sample(run, bl, st["in_cache"], args.p, args.R, args.b)
    S = {sp: shell_features(sets[sp][0], nr[sp], samp) for sp in SCORE_SPLITS}
    col = {n: i for i, n in enumerate(SHELL_NAMES)}
    ti, ci = expand_index(w["test"]), expand_index(w["cal"])
    out = {"cache_frac": point["cache_frac"], "oracle_hit_rate": point["test_hit_rate"],
           "sample_memory": samp["memory"] if samp else 0.0}
    for name, feats in CONFIGS.items():
        X = {sp: np.hstack([B[sp], S[sp][:, [col[f] for f in feats]]]) for sp in SCORE_SPLITS}
        sc = logistic_scores(X, y["fit"], w["fit"])
        out[name] = _evaluate(sc, y, ti, ci, target, pred=1 / (1 + np.exp(-sc["test"])))
    X = {sp: np.hstack([B[sp], S[sp]]) for sp in SCORE_SPLITS}
    g = gbt_regressor(0).fit(X["fit"], y["fit"], sample_weight=w["fit"])
    sc = {sp: g.predict(X[sp]) for sp in SCORE_SPLITS}
    out["GBT, all 8"] = _evaluate(sc, y, ti, ci, target, pred=sc["test"])
    return out


#-------------------------------------------------------------------------------


def main():
    setup_logging()
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--sizes", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.3])
    ap.add_argument("--p", type=float, default=0.25)
    ap.add_argument("--R", type=int, default=500)
    ap.add_argument("--b", type=int, default=4)
    ap.add_argument("--n-each", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    data, _, run = load_run_data(run_dir, args.n_each)
    res = next(r for r in data["results"] if abs(r["target"] - args.target) < 1e-9)
    tasks = []
    for size in args.sizes:
        p = min(res["points"], key=lambda p: abs(p["cache_frac"] - size))
        tasks.append((args.target, p, res["selected"][:p["n_entries"]]))
    out = []
    with worker_pool(eval_state, args.workers, len(tasks), run=run, args=args) as pool:
        for s in pool.imap(tasks):
            if s is None:
                continue
            out.append(s)
            names = [n for n in s if isinstance(s[n], dict)]
            logger.info(f"cache {100 * s['cache_frac']:.2f}% (oracle {s['oracle_hit_rate']:.3f}): "
                        + "; ".join(f"{n} {s[n]['calibrated_hit_rate']:.3f}" for n in names))
            (run_dir / "shell_ablation.json").write_text(
                json.dumps({"args": vars(args), "configs": CONFIGS, "states": out}, indent=1))


if __name__ == "__main__":
    main()
