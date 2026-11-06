#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Feature ablation for the retention GBT (eval_predictors features).

Feature categories (see docs/approximate_cache_hits_context.md, 5m):
  answer    the cache's candidate answer: distances from the query to its
            nearest cached vectors (scale and shape)
  coverage  position relative to complete cache balls
  scale     a nearby exact answer (the best entry's radius) vs the query
  history   track record of the best entry
  context   log cache size (constant within one cache state)
Configurations: full; full with other GBT seeds (noise level); each category
dropped; each answer / coverage feature dropped; absolute-scale distances
dropped; a minimal ratio-only "story" model. Per configuration: calibrated
test hit rate and retention (normal_fixed lambda on CAL), and best-cutoff
test hit rate (cutoff chosen on test; less noisy).

Writes <run>/ablation.json. Requires scikit-learn (.venv python).
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
from simulations.approximate_hits.eval_gap import best_cutoff_rate
from simulations.approximate_hits.threshold_selection import (
    DEFAULT_METHOD, select, threshold_grid,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

NAMES = list(ecal.FEATURE_NAMES) + list(epred.EXT_NAMES)
CATEGORIES = {
    "answer": ["log_D1", "log_Dk", "log_Dk_over_D1", "log_gap_k",
               "log_Dq1_over_Dk", "log_Dq2_over_Dk", "log_Dq3_over_Dk"],
    "coverage": ["log_min_d_entry", "log_min_d_over_r", "log_overshoot", "a1_frac",
                 "frac_topk_from_entry", "log1p_inside", "covered_frac"],
    "scale": ["log_Dk_over_r_entry", "log_r_entry", "log_dentry_over_Dk"],
    "history": ["entry_hist_mean", "entry_hist_log_count"],
    "context": ["log_cache_frac"],
}
ABSOLUTE = ["log_D1", "log_Dk", "log_min_d_entry", "log_r_entry"]
STORY = ["covered_frac", "log_min_d_over_r", "log_overshoot", "log_Dk_over_D1",
         "log_Dk_over_r_entry"]
_R: dict = {}

#-------------------------------------------------------------------------------


def configs():
    full = [n for n in NAMES]
    out = {"full": (full, 0), "full_seed1": (full, 1), "full_seed2": (full, 2)}
    for cat, feats in CATEGORIES.items():
        out[f"-{cat}"] = ([n for n in full if n not in feats], 0)
    for cat in ("answer", "coverage"):
        for f in CATEGORIES[cat]:
            out[f"-{f}"] = ([n for n in full if n != f], 0)
    out["-absolute"] = ([n for n in full if n not in ABSOLUTE], 0)
    out["story"] = (STORY, 0)
    out["story+history"] = (STORY + CATEGORIES["history"], 0)
    return out


def _gbt(seed):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         early_stopping=True, validation_fraction=0.1,
                                         random_state=seed)


def _expand(split):
    return np.repeat(np.arange(len(split["y"])), split["w"].astype(np.int64))


def eval_state(task):
    target = task[0]
    state = epred.build_state(task)
    if state is None:
        return None
    X = {sp: np.hstack([state[sp]["base"], state[sp]["ext"]]) for sp in epred.SPLITS}
    col = {n: i for i, n in enumerate(NAMES)}
    test, cal, fit = state["test"], state["cal"], state["fit"]
    ti, ci = _expand(test), _expand(cal)
    out = {"cache_frac": state["cache_frac"], "oracle_hit_rate": state["oracle_hit_rate"]}
    for name, (feats, seed) in configs().items():
        idx = [col[f] for f in feats]
        g = _gbt(seed).fit(X["fit"][:, idx], fit["y"], sample_weight=fit["w"])
        s = {sp: g.predict(X[sp][:, idx]) for sp in ("fit", "cal", "test")}
        lam = select(DEFAULT_METHOD, s["cal"][ci], cal["y"][ci], threshold_grid(s["fit"]),
                     target, 0.05)
        st, yt = s["test"][ti], test["y"][ti]
        adm = st >= lam if lam is not None else np.zeros(len(st), dtype=bool)
        out[name] = {"calibrated_hit_rate": float(adm.mean()),
                     "calibrated_hit_retention": float(yt[adm].mean()) if adm.any() else None,
                     "best_cutoff_hit_rate": best_cutoff_rate(st, yt, target)}
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
            logger.info(f"cache {100 * s['cache_frac']:.2f}% (oracle {s['oracle_hit_rate']:.3f}): "
                        + " ".join(f"{n} {s[n]['calibrated_hit_rate']:.3f}/"
                                   f"{s[n]['best_cutoff_hit_rate']:.3f}"
                                   for n in s if isinstance(s[n], dict)))
            (run_dir / "ablation.json").write_text(
                json.dumps({"args": vars(args), "categories": CATEGORIES,
                            "story": STORY, "states": out}, indent=1))


if __name__ == "__main__":
    main()
