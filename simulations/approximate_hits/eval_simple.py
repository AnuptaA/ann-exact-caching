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

from utils.cache.models import gbt_regressor, logistic_scores, standardize
from utils.cache.metrics import best_cutoff_rate
from utils.base.knn import knn_dists
from utils.cache.entries import state_tables
from utils.cache.features import N_BALLS, POOR_FIT, ball_features, balls_of
from utils.cache.metrics import expand_index
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


#-------------------------------------------------------------------------------


def best_fit_l2(B):
    """min over balls of sqrt(rel offset^2 + rel size^2); B = [ro, rs] x n."""
    ro, rs = B[:, 0::2], B[:, 1::2]
    return np.sqrt(ro ** 2 + rs ** 2).min(axis=1)


def analytic(B, S):
    """Fewer sampled points inside the neighborhood first; ties by best fit."""
    count = np.expm1(S[:, SAMPLE_J])                 # sampled within 1.0 x radius
    fit = best_fit_l2(B)
    return -np.round(count) - fit / (1.0 + fit)          # tie-break term in (-1, 0]


def monotone_cst(n_ball_cols, n_shell_cols):
    # balls: larger rel offset / size is worse; shell: farther nearest sample
    # vectors are better, larger counts are worse
    return ([-1] * n_ball_cols + [1] * SAMPLE_J
            + [-1] * (n_shell_cols - SAMPLE_J))


def scores(B, S, y, w):
    """name -> {split: score} for all models."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    X = {sp: np.hstack([B[sp], S[sp]]) for sp in SCORE_SPLITS}
    out = {}
    g = gbt_regressor(0).fit(X["fit"], y, sample_weight=w)
    out["gbt"] = {sp: g.predict(X[sp]) for sp in SCORE_SPLITS}
    h = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=2,
                                      early_stopping=True, validation_fraction=0.1,
                                      random_state=0,
                                      monotonic_cst=monotone_cst(B["fit"].shape[1],
                                                                 S["fit"].shape[1]))
    h.fit(X["fit"], y, sample_weight=w)
    out["gbt_shallow"] = {sp: h.predict(X[sp]) for sp in SCORE_SPLITS}
    Z = {sp: standardize(X[sp], X["fit"]) for sp in SCORE_SPLITS}
    r = Ridge(alpha=1.0).fit(Z["fit"], y, sample_weight=w)
    out["linear"] = {sp: r.predict(Z[sp]) for sp in SCORE_SPLITS}
    out["logistic"] = logistic_scores(X, y, w)
    out["analytic"] = {sp: analytic(B[sp], S[sp]) for sp in SCORE_SPLITS}
    out["analytic_balls"] = {sp: -best_fit_l2(B[sp]) for sp in SCORE_SPLITS}
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
    nr = {sp: np.maximum(knn_dists(sets[sp][0], st["db_c"], st["db_c_sq"], k)[:, k - 1], 1e-12) for sp in SCORE_SPLITS}
    B = {sp: np.nan_to_num(ball_features(sets[sp][0], nr[sp], bl, N_BALLS), nan=POOR_FIT)
         for sp in SCORE_SPLITS}
    samp = shell_sample(run, bl, st["in_cache"], args.p, args.R, args.b)
    S = {sp: shell_features(sets[sp][0], nr[sp], samp) for sp in SCORE_SPLITS}
    ti, ci = expand_index(w["test"]), expand_index(w["cal"])
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
            logger.info(f"cache {100 * s['cache_frac']:.2f}% (oracle {s['oracle_hit_rate']:.3f},"
                        f" +{100 * s['sample_memory']:.2f}% memory): " + " ".join(
                            f"{n} {s[n]['calibrated_hit_rate']:.3f}/"
                            f"{s[n]['best_cutoff_hit_rate']:.3f}" for n in names))
            (run_dir / "simple.json").write_text(
                json.dumps({"args": vars(args), "states": out}, indent=1))


if __name__ == "__main__":
    main()
