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
import logging
from pathlib import Path


import numpy as np


from simulations.approximate_hits.common import config_of
from utils.cache.models import gbt_regressor
from utils.cache.metrics import admit, best_cutoff_rate, result_for_target, state_tasks
from utils.cache.features import shell_block, split_inputs
from utils.cache.metrics import expand_index
from utils.cache.models import logistic_scores
from utils.cache.parallel import worker_pool
from utils.cache.workloads import SCORE_SPLITS, load_run_data
from utils.cache.calibration import calibrated_lambda

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
    lam = calibrated_lambda(s, y, ci, target)
    st, yt = s["test"][ti], y["test"][ti]
    rate, ret = admit(lam, st, yt)
    out = {"calibrated_hit_rate": rate, "calibrated_hit_retention": ret,
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
    inp = split_inputs(run, entries, SCORE_SPLITS)
    if inp is None:
        return None
    sets, st, bl, y, w, nr, B = run.sets, inp.st, inp.bl, inp.y, inp.w, inp.nr, inp.B
    samp, S = shell_block(run, inp, args.p, args.R, args.b)
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


def add_args(ap):
    ap.add_argument("run_dir")
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--sizes", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.3])
    ap.add_argument("--p", type=float, default=0.25)
    ap.add_argument("--R", type=int, default=500)
    ap.add_argument("--b", type=int, default=4)
    ap.add_argument("--n-each", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=4)
    ap.set_defaults(run=run)


def run(args):
    run_dir = Path(args.run_dir)
    data, _, run = load_run_data(run_dir, args.n_each)
    res = result_for_target(data, args.target)
    tasks = state_tasks(res, args.sizes, args.target)
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
                json.dumps({"args": config_of(args), "configs": CONFIGS, "states": out}, indent=1))
