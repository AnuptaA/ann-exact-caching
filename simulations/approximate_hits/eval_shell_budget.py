#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Split a fixed memory budget between cache and shell sample, chosen on dev.

For each budget M (fraction of the database, counting cached vectors plus
the sample's vector-equivalents), options:
  cache only   greedy cache of size M, 6-feature balls-only model
  split f, R   greedy cache of size f * M (prefix of the greedy order) plus a
               shell sample at ranks up to R filling the rest: p = min(1,
               (1 - f) M / memory of the full shell), 7-feature model
Every option: logistic model fit on FIT, lambda on CAL (normal_fixed), hit
rate on DEV and TEST. The option with the best DEV hit rate is chosen (no
test data used); reported with its test hit rate and retention, next to every
option's test result and the oracle at M (interpolated on the greedy curve).

Writes <run>/shell_budget.json. Requires scikit-learn (.venv python).
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
from simulations.approximate_hits.eval_final import interp_log
from simulations.approximate_hits.eval_radius_scores import cached_knn
from utils.cache.calibration import select, threshold_grid

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SPLITS = ("fit", "dev", "cal", "test")
DELTA = 0.05
BITS = 4
_R: dict = {}

#-------------------------------------------------------------------------------


def cumulative_cache_size(selected):
    """Number of distinct cached vectors after each prefix of the greedy order."""
    seen = np.zeros(len(ecal._G["db"]), dtype=bool)
    out = np.empty(len(selected), dtype=np.int64)
    total = 0
    for i, j in enumerate(selected):
        ids = np.asarray(ecal._G["cached_ids"][j])
        new = ids[~seen[ids]]
        seen[new] = True
        total += len(new)
        out[i] = total
    return out


def prefix_for(frac):
    """Longest greedy prefix whose cache is at most frac of the database."""
    n = int(np.searchsorted(_R["cum"], frac * len(ecal._G["db"]), side="right"))
    return _R["selected"][:n]


def full_shell_memory(bl, in_cache, R):
    shells = [ec.shell_ids(i, k, R) for i, k in zip(bl["cand"], bl["K"]) if k < R]
    if not shells:
        return 0.0
    ids = np.unique(np.concatenate(shells))
    return (~in_cache[ids]).sum() * BITS / 32 / len(ecal._G["db"])


def _expand(y, w):
    return np.repeat(np.arange(len(y)), w.astype(np.int64))


def eval_option(task):
    budget, f, R = task
    target, k = _R["args"].target, ecal._G["k"]
    entries = prefix_for(f * budget)
    if not entries:
        return None
    st = ecal.state_tables(entries)
    if len(st["cached"]) <= k:
        return None
    bl = ec.balls_of(entries)
    sets, topk = ecal._G["sets"], ecal._G["topk"]
    y = {sp: st["in_cache"][topk[sp]].mean(axis=1) for sp in SPLITS}
    w = {sp: sets[sp][1].astype(np.float64) for sp in SPLITS}
    nr = {sp: np.maximum(cached_knn(sets[sp][0], st)[:, k - 1], 1e-12) for sp in SPLITS}
    X = {sp: np.nan_to_num(ec.ball_features(sets[sp][0], nr[sp], bl, es.N_BALLS),
                           nan=es.POOR_FIT) for sp in SPLITS}
    cache_frac = len(st["cached"]) / len(ecal._G["db"])
    p, mem = 0.0, 0.0
    if f < 1:
        full = full_shell_memory(bl, st["in_cache"], R)
        p = min(1.0, (budget - cache_frac) / full) if full > 0 else 0.0
        samp = ec.shell_sample(bl, st["in_cache"], p, R, BITS) if p > 0 else None
        mem = samp["memory"] if samp else 0.0
        X = {sp: np.hstack([X[sp], ec.shell_features(sets[sp][0], nr[sp], samp)[:, [ec.SAMPLE_J]]])
             for sp in SPLITS}
    s = es.logistic_scores(X, y["fit"], w["fit"])
    ex = {sp: _expand(y[sp], w[sp]) for sp in SPLITS}
    lam = select("normal_fixed", s["cal"][ex["cal"]], y["cal"][ex["cal"]],
                 threshold_grid(s["fit"]), target, DELTA)
    out = {"budget": budget, "f": f, "R": R if f < 1 else None, "p": p,
           "cache_frac": cache_frac, "sample_memory": mem}
    for sp in ("dev", "test"):
        st_, yt = s[sp][ex[sp]], y[sp][ex[sp]]
        adm = st_ >= lam if lam is not None else np.zeros(len(st_), dtype=bool)
        out[f"{sp}_hit_rate"] = float(adm.mean())
        out[f"{sp}_hit_retention"] = float(yt[adm].mean()) if adm.any() else None
    return out


#-------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--budgets", type=float, nargs="+", required=True)
    ap.add_argument("--fs", type=float, nargs="+", default=[0.85, 0.7])
    ap.add_argument("--Rs", type=int, nargs="+", default=[500, 2000])
    ap.add_argument("--n-each", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    data, _ = epred.setup_run(run_dir, args.n_each)
    _R["args"] = args
    ec._R["draw"] = np.random.default_rng(1).random(len(ecal._G["db"]))  # as eval_core
    if max(args.Rs) > 500:
        ec._R["cand_ext"] = ec.load_cand_ext(run_dir, max(args.Rs))
    res = next(r for r in data["results"] if abs(r["target"] - args.target) < 1e-9)
    _R["selected"] = res["selected"]
    _R["cum"] = cumulative_cache_size(res["selected"])
    tasks = [(b, 1.0, None) for b in args.budgets]
    tasks += [(b, f, R) for b in args.budgets for f in args.fs for R in args.Rs]
    import multiprocessing as mp
    with mp.get_context("fork").Pool(min(args.workers, len(tasks))) as pool:
        rows = [r for r in pool.map(eval_option, tasks) if r is not None]
    pts = sorted(res["points"], key=lambda p: p["cache_frac"])
    xs, ys = [p["cache_frac"] for p in pts], [p["test_hit_rate"] for p in pts]
    summary = []
    for b in args.budgets:
        opts = [r for r in rows if r["budget"] == b]
        if not opts:
            continue
        best = max(opts, key=lambda r: r["dev_hit_rate"])
        only = next((r for r in opts if r["f"] == 1.0), None)
        summary.append({"budget": b, "oracle": interp_log(xs, ys, b), "chosen": best,
                        "cache_only": only, "options": opts})
        logger.info(f"budget {100 * b:.2f}%: oracle {interp_log(xs, ys, b)} | cache only "
                    f"{only['test_hit_rate'] if only else None} | chosen f={best['f']} R={best['R']} "
                    f"p={best['p']:.2f}: test {best['test_hit_rate']:.3f} "
                    f"(ret {best['test_hit_retention']})")
    (run_dir / "shell_budget.json").write_text(
        json.dumps({"args": vars(args), "budgets": summary}, indent=1))


if __name__ == "__main__":
    main()
