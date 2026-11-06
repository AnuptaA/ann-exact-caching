#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Compare retention predictors for calibrated admission (B5).

Query splits (all fresh, disjoint from cache selection):
  fit   train the predictor (and per-entry history features)
  dev   model selection: guaranteed hit rate on dev
  cal   calibration: choose lambda (normal approximation + fixed order)
  test  final report only

Features: the 12 base features of eval_calibrated.py plus
  log(D_r / D_k) for r = ceil(k/4), ceil(k/2), ceil(3k/4)
  fraction of returned vectors lying in a ball that contains q
  log(d(q, nearest entry) / D_k)
  per-entry history: smoothed mean retention of FIT queries whose nearest
    entry is the same (out-of-fold on the fit set), and log1p of their count
  log cache size (used by the pooled model)

Models: ridge (base / extended features), gradient-boosted trees for
retention (squared loss), for 1{retention >= T} (classifier), for the 20th
percentile of retention (quantile loss), and one gradient-boosted model pooled
over all evaluated cache states of the run.

Requires scikit-learn (run with the repo's .venv python).
"""

import json
import argparse
import logging
from pathlib import Path


import numpy as np

from utils.base.logs import setup_logging

from utils.cache.legacy_features import EXT_NAMES, FEATURE_NAMES, SPLITS, build_state
from utils.cache.metrics import expand_index, state_tasks
from utils.cache.models import fit_ridge, gbt
from utils.cache.parallel import worker_pool
from utils.cache.workloads import load_run_data
from utils.cache.calibration import select, threshold_grid

logger = logging.getLogger(__name__)

MODELS = ["ridge_base", "ridge_ext", "gbt", "gbt_cls", "gbt_q20", "gbt_pooled"]

#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------


def train_models(state, pooled_model):
    """Return {model_name: score function over a split}."""
    fit = state["fit"]
    X_base, X_ext = fit["base"], np.hstack([fit["base"], fit["ext"]])
    y, w = fit["y"], fit["w"]
    scorers = {}
    rb = fit_ridge(X_base, y, w)
    scorers["ridge_base"] = lambda sp: rb(sp["base"])
    re = fit_ridge(X_ext, y, w)
    scorers["ridge_ext"] = lambda sp: re(np.hstack([sp["base"], sp["ext"]]))
    g = gbt("reg").fit(X_ext, y, sample_weight=w)
    scorers["gbt"] = lambda sp: g.predict(np.hstack([sp["base"], sp["ext"]]))
    labels = (y >= state["target"] - 1e-9).astype(int)
    if 0 < labels.sum() < len(labels):
        c = gbt("cls").fit(X_ext, labels, sample_weight=w)
        scorers["gbt_cls"] = lambda sp: c.predict_proba(np.hstack([sp["base"], sp["ext"]]))[:, 1]
    q = gbt("quantile", 0.2).fit(X_ext, y, sample_weight=w)
    scorers["gbt_q20"] = lambda sp: q.predict(np.hstack([sp["base"], sp["ext"]]))
    if pooled_model is not None:
        scorers["gbt_pooled"] = lambda sp: pooled_model.predict(np.hstack([sp["base"], sp["ext"]]))
    return scorers


def evaluate_state(state, scorers, delta):
    out = {k: state[k] for k in ("target", "cache_frac", "oracle_hit_rate_online",
                                 "oracle_hit_rate")}
    target = state["target"]
    for name, score in scorers.items():
        s = {sp: score(state[sp]) for sp in SPLITS}
        grid = threshold_grid(s["fit"])
        cal_idx = expand_index(state["cal"]["w"])
        rec = {}
        for tag, method in (("guaranteed", "normal_fixed"), ("naive", "naive")):
            lam = select(method, s["cal"][cal_idx], state["cal"]["y"][cal_idx],
                         grid, target, delta)
            for sp in ("dev", "test"):
                idx = expand_index(state[sp]["w"])
                ss, yy = s[sp][idx], state[sp]["y"][idx]
                adm = ss >= lam if lam is not None else np.zeros(len(ss), dtype=bool)
                rec[f"{tag}_{sp}_hit_rate"] = float(adm.mean())
                rec[f"{tag}_{sp}_hit_retention"] = float(yy[adm].mean()) if adm.any() else None
        dev = state["dev"]
        ra = np.argsort(np.argsort(s["dev"]))
        rb = np.argsort(np.argsort(dev["y"]))
        rec["dev_spearman"] = float(np.corrcoef(ra, rb)[0, 1])
        out[name] = rec
    return out


def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--targets", type=float, nargs="+", default=[0.9, 0.95])
    ap.add_argument("--sizes", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.30])
    ap.add_argument("--n-each", type=int, default=20000)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    data, cfg, run = load_run_data(run_dir, args.n_each)

    tasks = []
    for res in data["results"]:
        if not any(abs(res["target"] - t) < 1e-9 for t in args.targets):
            continue
        tasks += state_tasks(res, args.sizes)
    logger.info(f"Building features for {len(tasks)} cache states...")
    with worker_pool(build_state, args.workers, len(tasks), run=run) as pool:
        states = [s for s in pool.map(tasks) if s is not None]

    logger.info("Training pooled model...")
    Xp = np.vstack([np.hstack([s["fit"]["base"], s["fit"]["ext"]]) for s in states])
    yp = np.concatenate([s["fit"]["y"] for s in states])
    wp = np.concatenate([s["fit"]["w"] for s in states])
    pooled = gbt("reg").fit(Xp, yp, sample_weight=wp)

    logger.info("Training per-state models and evaluating...")
    out = []
    for st in states:
        out.append(evaluate_state(st, train_models(st, pooled), args.delta))
    path = run_dir / "predictors.json"
    path.write_text(json.dumps({"args": vars(args), "models": MODELS,
                                "base_features": FEATURE_NAMES,
                                "ext_features": EXT_NAMES, "states": out}))
    logger.info(f"Wrote {path}")


if __name__ == "__main__":
    main()
