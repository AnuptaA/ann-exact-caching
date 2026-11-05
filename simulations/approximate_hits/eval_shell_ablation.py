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
from simulations.approximate_hits.eval_ablation import _gbt
from simulations.approximate_hits.eval_gap import best_cutoff_rate
from simulations.approximate_hits.eval_radius_scores import cached_knn
from simulations.approximate_hits.threshold_selection import (
    DEFAULT_METHOD, select, threshold_grid,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SPLITS = es.SPLITS
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
_R: dict = {}

#-------------------------------------------------------------------------------


def _expand(y, w):
    return np.repeat(np.arange(len(y)), w.astype(np.int64))


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


def eval_state(task):
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
    samp = ec.shell_sample(bl, st["in_cache"], args.p, args.R, args.b)
    S = {sp: ec.shell_features(sets[sp][0], nr[sp], samp) for sp in SPLITS}
    col = {n: i for i, n in enumerate(SHELL_NAMES)}
    ti, ci = _expand(y["test"], w["test"]), _expand(y["cal"], w["cal"])
    out = {"cache_frac": point["cache_frac"], "oracle_hit_rate": point["test_hit_rate"],
           "sample_memory": samp["memory"] if samp else 0.0}
    for name, feats in CONFIGS.items():
        X = {sp: np.hstack([B[sp], S[sp][:, [col[f] for f in feats]]]) for sp in SPLITS}
        sc = es.logistic_scores(X, y["fit"], w["fit"])
        out[name] = _evaluate(sc, y, ti, ci, target, pred=1 / (1 + np.exp(-sc["test"])))
    X = {sp: np.hstack([B[sp], S[sp]]) for sp in SPLITS}
    g = _gbt(0).fit(X["fit"], y["fit"], sample_weight=w["fit"])
    sc = {sp: g.predict(X[sp]) for sp in SPLITS}
    out["GBT, all 8"] = _evaluate(sc, y, ti, ci, target, pred=sc["test"])
    return out


#-------------------------------------------------------------------------------


def main():
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
    data, _ = epred.setup_run(run_dir, args.n_each)
    _R["args"] = args
    ec._R["draw"] = np.random.default_rng(1).random(len(ecal._G["db"]))  # as eval_core
    res = next(r for r in data["results"] if abs(r["target"] - args.target) < 1e-9)
    tasks = []
    for size in args.sizes:
        p = min(res["points"], key=lambda p: abs(p["cache_frac"] - size))
        tasks.append((args.target, p, res["selected"][:p["n_entries"]]))
    import multiprocessing as mp
    out = []
    with mp.get_context("fork").Pool(min(args.workers, len(tasks))) as pool:
        for s in pool.imap(eval_state, tasks):
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
