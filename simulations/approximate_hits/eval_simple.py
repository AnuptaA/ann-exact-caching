#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Simple models vs the GBT on the two feature families (eval_core features:
per-ball stats for the 3 best-fitting balls; shell sample with --p, --R, --b).

Scores (each turned into admissions by the same calibrated cutoff):
  gbt           gradient-boosted trees (as in eval_core)
  gbt_shallow   depth-2 trees with monotone constraints (worse fit / more
                sampled points nearby -> lower score)
  linear        ridge regression on standardized features
  logistic      logistic regression with retention as a soft label
  analytic      no fitting: rank by the number of sampled points inside the
                neighborhood (fewer is better), ties broken by how well the
                best ball fits (smaller sqrt(rel offset^2 + rel size^2))
  analytic_balls  no fitting, balls only: -min over balls of
                sqrt(rel offset^2 + rel size^2)
Missing ball slots (fewer balls than 3) are filled with a poor fit (10).

Writes <run>/simple.json. Requires scikit-learn (.venv python).
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

N_BALLS = 3
POOR_FIT = 10.0
SPLITS = ("fit", "cal", "test")
_R: dict = {}

#-------------------------------------------------------------------------------


def best_fit_l2(B):
    """min over balls of sqrt(rel offset^2 + rel size^2); B = [ro, rs] x n."""
    ro, rs = B[:, 0::2], B[:, 1::2]
    return np.sqrt(ro ** 2 + rs ** 2).min(axis=1)


def analytic(B, S):
    """Fewer sampled points inside the neighborhood first; ties by best fit."""
    count = np.expm1(S[:, ec.SAMPLE_J])                 # sampled within 1.0 x radius
    fit = best_fit_l2(B)
    return -np.round(count) - fit / (1.0 + fit)          # tie-break term in (-1, 0]


def monotone_cst(n_ball_cols, n_shell_cols):
    # balls: larger rel offset / size is worse; shell: farther nearest sample
    # vectors are better, larger counts are worse
    return ([-1] * n_ball_cols + [1] * ec.SAMPLE_J
            + [-1] * (n_shell_cols - ec.SAMPLE_J))


def _standardize(X, ref):
    mu, sd = ref.mean(axis=0), ref.std(axis=0) + 1e-9
    return (X - mu) / sd


def logistic_scores(X, y, w):
    """Logistic regression on standardized X with retention as a soft label:
    each row is a positive with weight y and a negative with weight 1 - y
    (exact cross-entropy on fractional labels). Returns {split: score}."""
    from sklearn.linear_model import LogisticRegression
    Z = {sp: _standardize(X[sp], X["fit"]) for sp in SPLITS}
    n = len(y)
    lr = LogisticRegression(C=1.0, max_iter=2000)
    lr.fit(np.vstack([Z["fit"], Z["fit"]]), np.r_[np.ones(n), np.zeros(n)],
           sample_weight=np.r_[w * y, w * (1 - y)] + 1e-12)
    return {sp: lr.decision_function(Z[sp]) for sp in SPLITS}


def scores(B, S, y, w):
    """name -> {split: score} for all models."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    X = {sp: np.hstack([B[sp], S[sp]]) for sp in SPLITS}
    out = {}
    g = _gbt(0).fit(X["fit"], y, sample_weight=w)
    out["gbt"] = {sp: g.predict(X[sp]) for sp in SPLITS}
    h = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=2,
                                      early_stopping=True, validation_fraction=0.1,
                                      random_state=0,
                                      monotonic_cst=monotone_cst(B["fit"].shape[1],
                                                                 S["fit"].shape[1]))
    h.fit(X["fit"], y, sample_weight=w)
    out["gbt_shallow"] = {sp: h.predict(X[sp]) for sp in SPLITS}
    Z = {sp: _standardize(X[sp], X["fit"]) for sp in SPLITS}
    r = Ridge(alpha=1.0).fit(Z["fit"], y, sample_weight=w)
    out["linear"] = {sp: r.predict(Z[sp]) for sp in SPLITS}
    out["logistic"] = logistic_scores(X, y, w)
    out["analytic"] = {sp: analytic(B[sp], S[sp]) for sp in SPLITS}
    out["analytic_balls"] = {sp: -best_fit_l2(B[sp]) for sp in SPLITS}
    return out


def _expand(y, w):
    return np.repeat(np.arange(len(y)), w.astype(np.int64))


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
    B = {sp: np.nan_to_num(ec.ball_features(sets[sp][0], nr[sp], bl, N_BALLS), nan=POOR_FIT)
         for sp in SPLITS}
    samp = ec.shell_sample(bl, st["in_cache"], args.p, args.R, args.b)
    S = {sp: ec.shell_features(sets[sp][0], nr[sp], samp) for sp in SPLITS}
    ti, ci = _expand(y["test"], w["test"]), _expand(y["cal"], w["cal"])
    out = {"cache_frac": point["cache_frac"], "oracle_hit_rate": point["test_hit_rate"],
           "sample_memory": samp["memory"] if samp else 0.0}
    for name, s in scores(B, S, y["fit"], w["fit"]).items():
        lam = select(DEFAULT_METHOD, s["cal"][ci], y["cal"][ci], threshold_grid(s["fit"]),
                     target, 0.05)
        st_, yt = s["test"][ti], y["test"][ti]
        adm = st_ >= lam if lam is not None else np.zeros(len(st_), dtype=bool)
        out[name] = {"calibrated_hit_rate": float(adm.mean()),
                     "calibrated_hit_retention": float(yt[adm].mean()) if adm.any() else None,
                     "best_cutoff_hit_rate": best_cutoff_rate(st_, yt, target)}
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
            logger.info(f"cache {100 * s['cache_frac']:.2f}% (oracle {s['oracle_hit_rate']:.3f},"
                        f" +{100 * s['sample_memory']:.2f}% memory): " + " ".join(
                            f"{n} {s[n]['calibrated_hit_rate']:.3f}/"
                            f"{s[n]['best_cutoff_hit_rate']:.3f}" for n in names))
            (run_dir / "simple.json").write_text(
                json.dumps({"args": vars(args), "states": out}, indent=1))


if __name__ == "__main__":
    main()
