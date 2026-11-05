#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Focus training on likely hits: plain vs reweighted vs ranking loss.

Three targets (gradient-boosted trees on the eval_predictors features):
  ret     retention
  ratio   true radius / k-th nearest cached distance (<= 1), then count
  radius  log true radius (capped at the k-th nearest cached distance), then count
Counting turns a predicted radius into a score: cached vectors within it / k
(continuous, eval_radius_models.soft_count).

Variants:
  plain   unweighted squared loss (as before)
  wtd     two rounds: fit plain, find the region a cutoff would admit on FIT
          (naive cutoff: top-scored FIT queries average >= T), give FIT queries
          scoring in the top 2x that region weight --boost, refit
  rank    LambdaRank (LightGBM) on retention, graded labels round(10 *
          retention), random groups of --group-size queries, NDCG truncated at
          the top --rank-top of each group; output used directly as the score

Every score: calibrated admission (normal_fixed lambda on CAL), test hit rate
and realized retention. Writes <run>/focused.json. Needs scikit-learn and
lightgbm (.venv python).
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
from simulations.approximate_hits.eval_radius_models import soft_count
from simulations.approximate_hits.eval_radius_scores import cached_knn, true_radius
from simulations.approximate_hits.threshold_selection import (
    DEFAULT_METHOD, select, threshold_grid,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

EPS = 1e-12
TARGETS = ["ret", "ratio", "radius"]
SCORES = [f"{t}_{v}" for t in TARGETS for v in ("plain", "wtd")] + ["rank"]
_R: dict = {}

#-------------------------------------------------------------------------------


def _expand(split):
    return np.repeat(np.arange(len(split["y"])), split["w"].astype(np.int64))


def focus_weights(score, y, w, target, boost):
    """Weight boost for FIT queries in (twice) the region a cutoff would admit."""
    idx = _expand({"y": y, "w": w})
    lam = select("naive", score[idx], y[idx], threshold_grid(score), target, 0.05)
    frac = float(np.average(score >= lam, weights=w)) if lam is not None else 0.0
    frac = min(max(2.0 * frac, 0.02), 1.0)
    cut = np.quantile(score, 1.0 - frac)
    return w * np.where(score >= cut, boost, 1.0)


def fit_rank(X, y, w, group_size, top, seed=0):
    import lightgbm as lgb
    perm = np.random.default_rng(seed).permutation(len(y))
    groups = [len(perm[i:i + group_size]) for i in range(0, len(perm), group_size)]
    model = lgb.LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05,
                           num_leaves=31, lambdarank_truncation_level=top,
                           eval_at=[top], verbose=-1, random_state=seed, n_jobs=8)
    model.fit(X[perm], np.round(10 * y[perm]).astype(int), sample_weight=w[perm],
              group=groups)
    return model


def eval_state(task):
    target, point, entries = task
    args = _R["args"]
    state = epred.build_state(task)
    if state is None:
        return None
    st = ecal.state_tables(entries)
    X, dc = {}, {}
    for sp in epred.SPLITS:
        X[sp] = np.hstack([state[sp]["base"], state[sp]["ext"]])
        dc[sp] = cached_knn(ecal._G["sets"][sp][0], st)
    upper = {sp: np.maximum(dc[sp][:, -1], EPS) for sp in epred.SPLITS}
    r = _R["radius"]
    fit = state["fit"]
    labels = {"ret": fit["y"],
              "ratio": np.minimum(np.log((r["fit"] + EPS) / upper["fit"]), 0.0),
              "radius": np.log(r["fit"] + EPS)}

    def to_score(t, pred, sp):
        if t == "ret":
            return pred
        rad = (upper[sp] * np.exp(np.minimum(pred, 0.0)) if t == "ratio"
               else np.minimum(np.exp(pred), upper[sp]))
        return soft_count(dc[sp], rad)

    scores = {}
    for t in TARGETS:
        g = epred._gbt("reg").fit(X["fit"], labels[t], sample_weight=fit["w"])
        scores[f"{t}_plain"] = {sp: to_score(t, g.predict(X[sp]), sp) for sp in epred.SPLITS}
        w2 = focus_weights(scores[f"{t}_plain"]["fit"], fit["y"], fit["w"], target, args.boost)
        g = epred._gbt("reg").fit(X["fit"], labels[t], sample_weight=w2)
        scores[f"{t}_wtd"] = {sp: to_score(t, g.predict(X[sp]), sp) for sp in epred.SPLITS}
    m = fit_rank(X["fit"], fit["y"], fit["w"], args.group_size, args.rank_top)
    scores["rank"] = {sp: m.predict(X[sp]) for sp in epred.SPLITS}

    out = {"cache_frac": point["cache_frac"], "oracle_hit_rate": point["test_hit_rate"]}
    test, cal = state["test"], state["cal"]
    ti, ci = _expand(test), _expand(cal)
    for name in SCORES:
        s = scores[name]
        lam = select(DEFAULT_METHOD, s["cal"][ci], cal["y"][ci], threshold_grid(s["fit"]),
                     target, 0.05)
        st_, yt = s["test"][ti], test["y"][ti]
        adm = st_ >= lam if lam is not None else np.zeros(len(st_), dtype=bool)
        out[name] = {"hit_rate": float(adm.mean()),
                     "hit_retention": float(yt[adm].mean()) if adm.any() else None}
    return out


#-------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--sizes", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.3])
    ap.add_argument("--n-each", type=int, default=20000)
    ap.add_argument("--boost", type=float, default=5.0)
    ap.add_argument("--group-size", type=int, default=200)
    ap.add_argument("--rank-top", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    data, _ = epred.setup_run(run_dir, args.n_each)
    _R["args"] = args
    _R["radius"] = {sp: true_radius(ecal._G["sets"][sp][0], ecal._G["topk"][sp])
                    for sp in epred.SPLITS}

    res = next(r for r in data["results"] if abs(r["target"] - args.target) < 1e-9)
    tasks = []
    for size in args.sizes:
        p = min(res["points"], key=lambda p: abs(p["cache_frac"] - size))
        tasks.append((args.target, p, res["selected"][:p["n_entries"]]))
    import multiprocessing as mp
    out = []
    f = lambda v: " -- " if v is None else f"{v:.3f}"
    with mp.get_context("fork").Pool(min(args.workers, len(tasks))) as pool:
        for s in pool.imap(eval_state, tasks):
            if s is None:
                continue
            out.append(s)
            logger.info(f"cache {100 * s['cache_frac']:.2f}% (oracle {s['oracle_hit_rate']:.3f}): "
                        + " ".join(f"{n} {s[n]['hit_rate']:.3f}/{f(s[n]['hit_retention'])}"
                                   for n in SCORES))
            (run_dir / "focused.json").write_text(
                json.dumps({"args": vars(args), "states": out}, indent=1))


if __name__ == "__main__":
    main()
