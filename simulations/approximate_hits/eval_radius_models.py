#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Predict the radius, then count, vs predicting retention directly.

The radius (distance from the query to its k-th nearest database vector) is
the only unknown: given it, retention from a cache is the number of cached
vectors within it, divided by k. The k-th nearest cached distance ("upper") is
always >= the radius. Models (gradient-boosted trees, same features as
eval_predictors unless noted), each turned into a retention score:
  ret      predict retention directly (reference, = eval_predictors gbt)
  ratio    predict log(radius / upper) (<= 0), radius = upper * exp(pred)
  radius   predict log(radius), capped at upper
  radius_q predict log(radius) from the raw query vector only (cache-free
           model), capped at upper
Radius-based scores count cached vectors within the predicted radius,
interpolated between consecutive cached distances so the score is continuous:
  score = (i + (r - d_i) / (d_{i+1} - d_i)) / k for d_i <= r < d_{i+1}
Each score: naive admission (score >= T) and calibrated admission
(normal_fixed lambda on CAL). Also radius accuracy (median |log error|) and
retention MAE of the implied score on test.

Writes <run>/radius_models.json. Requires scikit-learn (.venv python).
"""

import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from simulations.approximate_hits import eval_calibrated as ecal
from simulations.approximate_hits import eval_predictors as epred
from simulations.approximate_hits.eval_radius_scores import cached_knn, true_radius
from utils.cache.calibration import (
    DEFAULT_METHOD, select, threshold_grid,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

EPS = 1e-12
MODELS = ["ret", "ratio", "radius", "radius_q"]
_R: dict = {}

#-------------------------------------------------------------------------------


def soft_count(dc, r):
    """Continuous count of cached distances (rows of dc, sorted) within r, / k."""
    k = dc.shape[1]
    d0 = np.hstack([np.zeros((len(dc), 1)), dc])
    i = np.clip((d0 <= r[:, None]).sum(axis=1) - 1, 0, k)
    lo = d0[np.arange(len(dc)), i]
    hi = d0[np.arange(len(dc)), np.minimum(i + 1, k)]
    frac = np.where(i < k, (r - lo) / np.maximum(hi - lo, EPS), 0.0)
    return (i + np.clip(frac, 0.0, 1.0)) / k


def _expand(split):
    return np.repeat(np.arange(len(split["y"])), split["w"].astype(np.int64))


def _fit_radius_q():
    """Cache-free model: log radius from the raw query vector."""
    fq, fw = ecal._G["sets"]["fit"]
    g = epred._gbt("reg").fit(np.asarray(fq, dtype=np.float32),
                              np.log(_R["radius"]["fit"] + EPS), sample_weight=fw)
    return {sp: g.predict(np.asarray(ecal._G["sets"][sp][0], dtype=np.float32))
            for sp in epred.SPLITS}


def eval_state(task):
    target, point, entries = task
    state = epred.build_state(task)
    if state is None:
        return None
    st = ecal.state_tables(entries)
    X, dc, r = {}, {}, {}
    for sp in epred.SPLITS:
        q, _ = ecal._G["sets"][sp]
        X[sp] = np.hstack([state[sp]["base"], state[sp]["ext"]])
        dc[sp] = cached_knn(q, st)
        r[sp] = _R["radius"][sp]
    fit, w = state["fit"], state["fit"]["w"]
    upper = {sp: np.maximum(dc[sp][:, -1], EPS) for sp in epred.SPLITS}

    g_ret = epred._gbt("reg").fit(X["fit"], fit["y"], sample_weight=w)
    g_ratio = epred._gbt("reg").fit(
        X["fit"], np.minimum(np.log((r["fit"] + EPS) / upper["fit"]), 0.0), sample_weight=w)
    g_rad = epred._gbt("reg").fit(X["fit"], np.log(r["fit"] + EPS), sample_weight=w)

    scores, r_hat = {m: {} for m in MODELS}, {m: {} for m in MODELS[1:]}
    for sp in epred.SPLITS:
        scores["ret"][sp] = g_ret.predict(X[sp])
        r_hat["ratio"][sp] = upper[sp] * np.exp(np.minimum(g_ratio.predict(X[sp]), 0.0))
        r_hat["radius"][sp] = np.minimum(np.exp(g_rad.predict(X[sp])), upper[sp])
        r_hat["radius_q"][sp] = np.minimum(np.exp(_R["radius_q_pred"][sp]), upper[sp])
        for m in MODELS[1:]:
            scores[m][sp] = soft_count(dc[sp], r_hat[m][sp])

    out = {"cache_frac": point["cache_frac"], "oracle_hit_rate": point["test_hit_rate"]}
    test, cal = state["test"], state["cal"]
    ti, ci = _expand(test), _expand(cal)
    for m in MODELS:
        s_t, y_t = scores[m]["test"][ti], test["y"][ti]
        rec = {"retention_mae": float(np.abs(s_t - y_t).mean())}
        if m in r_hat:
            err = np.log(r_hat[m]["test"] + EPS) - np.log(r["test"] + EPS)
            rec["radius_median_abs_log_err"] = float(np.median(np.abs(err)))
            rec["radius_frac_over"] = float((err > 0).mean())
        adm = s_t >= target
        rec["naive_hit_rate"] = float(adm.mean())
        rec["naive_hit_retention"] = float(y_t[adm].mean()) if adm.any() else None
        lam = select(DEFAULT_METHOD, scores[m]["cal"][ci], cal["y"][ci],
                     threshold_grid(scores[m]["fit"]), target, 0.05)
        adm = s_t >= lam if lam is not None else np.zeros(len(s_t), dtype=bool)
        rec["calibrated_hit_rate"] = float(adm.mean())
        rec["calibrated_hit_retention"] = float(y_t[adm].mean()) if adm.any() else None
        out[m] = rec
    return out


#-------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--sizes", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.3])
    ap.add_argument("--n-each", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    data, _ = epred.setup_run(run_dir, args.n_each)

    _R["radius"] = {sp: true_radius(ecal._G["sets"][sp][0], ecal._G["topk"][sp])
                    for sp in epred.SPLITS}
    logger.info("Fitting cache-free radius model on raw query vectors...")
    import multiprocessing as mp
    # fit in a child: OpenMP initialized in the parent deadlocks forked workers
    with mp.get_context("fork").Pool(1) as pool:
        _R["radius_q_pred"] = pool.apply(_fit_radius_q)

    res = next(r for r in data["results"] if abs(r["target"] - args.target) < 1e-9)
    tasks = []
    for size in args.sizes:
        p = min(res["points"], key=lambda p: abs(p["cache_frac"] - size))
        tasks.append((args.target, p, res["selected"][:p["n_entries"]]))
    out = []
    f = lambda v: "  -- " if v is None else f"{v:.3f}"
    with mp.get_context("fork").Pool(min(args.workers, len(tasks))) as pool:
        for s in pool.imap(eval_state, tasks):
            if s is None:
                continue
            out.append(s)
            logger.info(f"cache {100 * s['cache_frac']:.2f}% (oracle {s['oracle_hit_rate']:.3f}):")
            for m in MODELS:
                x = s[m]
                rad = (f" | radius |log err| {x['radius_median_abs_log_err']:.3f}"
                       f" over {x['radius_frac_over']:.2f}" if m != "ret" else "")
                logger.info(f"  {m:9s} ret MAE {x['retention_mae']:.3f}{rad}"
                            f" | naive hit {x['naive_hit_rate']:.3f} ret {f(x['naive_hit_retention'])}"
                            f" | calibrated hit {x['calibrated_hit_rate']:.3f}"
                            f" ret {f(x['calibrated_hit_retention'])}")
            (run_dir / "radius_models.json").write_text(
                json.dumps({"args": vars(args), "states": out}, indent=1))


if __name__ == "__main__":
    main()
