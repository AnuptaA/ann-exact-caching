#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Does the 7-feature logistic model generalize across cache states and k?

Features (eval core): relative offset and relative size for the 3 best-fitting
balls, plus the intruder count (sampled shell points inside the neighborhood;
shell p = 25%, ranks up to 500, 4-bit). Neighborhood radius = distance to the
k-th closest cached vector, so the features are defined for any k.

--mode caches (k = the run's k): cache family = greedy caches at --sizes, the
  greedy cache selected for T = --other-target at --base-size, and the base
  cache with --perturb of its entries replaced by random candidates. For each
  ordered pair (source A, destination B):
    recal   A's model, lambda recalibrated on B's cal queries
    frozen  A's model and A's lambda
  plus pooled models fit on all caches together (with log cache size as an
  input), with per-cache lambda and with one shared lambda.
--mode k (one cache per --sizes, labels at every k in --ks; exact top-max(ks)
  computed for the query splits): per k
    native   model fit and calibrated at that k
    from_k   model fit at --source-k, lambda recalibrated at this k
    pooled   one model fit on all k together (log k as an input), lambda per k
    shared   pooled model with ONE lambda calibrated on all k together
Each: test hit rate, realized retention, and the oracle (optimal admission on
true retention). Writes <run>/generalize_<mode>.json. Needs scikit-learn.
"""

import json
import logging
from pathlib import Path


import numpy as np

from simulations.approximate_hits.common import config_of
from utils.cache.metrics import admit as metrics_admit, nearest_point, result_for_target

from utils.base.knn import pairwise_dist
from utils.cache.entries import perturbed
from utils.base.knn import exact_topk
from utils.cache.entries import state_tables
from utils.cache.features import N_BALLS, POOR_FIT, ball_features, balls_of
from utils.cache.models import fit_logistic
from utils.cache.shell import SAMPLE_J, shell_features, shell_sample
from utils.cache.workloads import SCORE_SPLITS, load_run_data
from utils.cache.calibration import (
    DEFAULT_METHOD, select, threshold_grid,
)

logger = logging.getLogger(__name__)

SHELL = (0.25, 500, 4)
DELTA = 0.05

#-------------------------------------------------------------------------------


def cached_kth(q, st, k):
    out = np.empty(len(q))
    for s in range(0, len(q), 2048):
        d = pairwise_dist(q[s:s + 2048], st["db_c"], st["db_c_sq"])
        out[s:s + 2048] = np.partition(d, k - 1, axis=1)[:, k - 1]
    return np.maximum(out, 1e-12)


def optimal_rate(y, target):
    ys = np.sort(y)[::-1]
    ok = np.cumsum(ys) / np.arange(1, len(ys) + 1) >= target - 1e-12
    return float((np.flatnonzero(ok).max() + 1) / len(ys)) if ok.any() else 0.0


def cache_data(run, entries, ks, topk):
    """Per k: {split: (X, y)} for one cache state (X = the 7 features)."""
    st = state_tables(run, entries)
    bl = balls_of(run, entries)
    samp = shell_sample(run, bl, st["in_cache"], *SHELL)
    out = {}
    for k in ks:
        out[k] = {}
        for sp in SCORE_SPLITS:
            q = run.sets[sp][0]
            nr = cached_kth(q, st, k)
            B = np.nan_to_num(ball_features(q, nr, bl, N_BALLS), nan=POOR_FIT)
            S = shell_features(q, nr, samp)[:, [SAMPLE_J]]
            y = st["in_cache"][topk[sp][:, :k]].mean(axis=1)
            out[k][sp] = (np.hstack([B, S]), y)
    return out


def choose_lambda(score, fit_X, cal_X, cal_y, target):
    return select(DEFAULT_METHOD, score(cal_X), cal_y, threshold_grid(score(fit_X)), target, DELTA)


def admit(score, lam, X, y):
    hit_rate, hit_retention = metrics_admit(lam, score(X), y)
    return {"hit_rate": hit_rate, "hit_retention": hit_retention}


def _stack(parts, extra=None):
    """Stack (X, y) parts; extra[i] is a constant column appended to part i."""
    Xs, ys = [], []
    for i, (X, y) in enumerate(parts):
        if extra is not None:
            X = np.hstack([X, np.full((len(X), 1), extra[i])])
        Xs.append(X)
        ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


def _with(X, v):
    return np.hstack([X, np.full((len(X), 1), v)])


#-------------------------------------------------------------------------------


def mode_caches(data, args, run, topk):
    res = {round(r["target"], 6): r for r in data["results"]}
    main = res[round(args.target, 6)]
    fam = {}
    for size in args.sizes:
        p = nearest_point(main, size)
        fam[f"{100 * size:g}%"] = (main["selected"][:p["n_entries"]], p["cache_frac"])
    r = res[round(args.other_target, 6)]
    p = nearest_point(r, args.base_size)
    fam[f"T={args.other_target:g} cache, {100 * args.base_size:g}%"] = (
        r["selected"][:p["n_entries"]], p["cache_frac"])
    p = nearest_point(main, args.base_size)
    n_kv = len(run.k_values)
    ent = perturbed(main["selected"][:p["n_entries"]], args.perturb, len(run.cand_q),
                    n_kv, np.random.default_rng(0))
    fam[f"{100 * args.base_size:g}%, {100 * args.perturb:g}% entries replaced"] = (
        ent, p["cache_frac"])
    k = run.k
    D = {}
    for name, (entries, frac) in fam.items():
        logger.info(f"features for cache {name}...")
        D[name] = (cache_data(run, entries, [k], topk)[k], frac)
    T = args.target
    models = {n: fit_logistic(*d["fit"]) for n, (d, _) in D.items()}
    lam_own = {n: choose_lambda(models[n], d["fit"][0], *d["cal"], T) for n, (d, _) in D.items()}
    pairs = []
    for a in D:
        for b, (db_, _) in D.items():
            rec = {"source": a, "dest": b}
            rec["recal"] = admit(models[a], choose_lambda(models[a], D[a][0]["fit"][0],
                                                          *db_["cal"], T), *db_["test"])
            rec["frozen"] = admit(models[a], lam_own[a], *db_["test"])
            pairs.append(rec)
    # pooled across caches (log cache size as an input)
    lf = [np.log(f) for _, (_, f) in D.items()]
    names = list(D)
    Xf, yf = _stack([D[n][0]["fit"] for n in names], lf)
    pooled = fit_logistic(Xf, yf)
    Xc, yc = _stack([D[n][0]["cal"] for n in names], lf)
    lam_shared = select(DEFAULT_METHOD, pooled(Xc), yc, threshold_grid(pooled(Xf)), T, DELTA)
    per_cache = {}
    for n, l in zip(names, lf):
        d = D[n][0]
        lam = choose_lambda(pooled, _with(d["fit"][0], l), _with(d["cal"][0], l), d["cal"][1], T)
        per_cache[n] = {"oracle": optimal_rate(d["test"][1], T),
                        "native": admit(models[n], lam_own[n], *d["test"]),
                        "pooled": admit(pooled, lam, _with(d["test"][0], l), d["test"][1]),
                        "pooled_shared_lambda": admit(pooled, lam_shared, _with(d["test"][0], l),
                                                      d["test"][1])}
    return {"caches": names, "pairs": pairs, "per_cache": per_cache}


def mode_k(data, args, run, topk):
    res = result_for_target(data, args.target)
    T, ks = args.target, args.ks
    out = {}
    for size in args.sizes:
        p = nearest_point(res, size)
        logger.info(f"features for cache {100 * size:g}% at k = {ks}...")
        D = cache_data(run, res["selected"][:p["n_entries"]], ks, topk)
        models = {k: fit_logistic(*D[k]["fit"]) for k in ks}
        lk = {k: np.log(k) for k in ks}
        Xf, yf = _stack([D[k]["fit"] for k in ks], [lk[k] for k in ks])
        pooled = fit_logistic(Xf, yf)
        Xc, yc = _stack([D[k]["cal"] for k in ks], [lk[k] for k in ks])
        lam_shared = select(DEFAULT_METHOD, pooled(Xc), yc, threshold_grid(pooled(Xf)), T, DELTA)
        rows = {}
        for k in ks:
            d = D[k]
            src = models[args.source_k]
            Xf_k, Xc_k, Xt_k = (_with(d[sp][0], lk[k]) for sp in SCORE_SPLITS)
            rows[str(k)] = {
                "oracle": optimal_rate(d["test"][1], T),
                "native": admit(models[k], choose_lambda(models[k], d["fit"][0], *d["cal"], T),
                                *d["test"]),
                "from_k": admit(src, choose_lambda(src, d["fit"][0], *d["cal"], T), *d["test"]),
                "pooled": admit(pooled, choose_lambda(pooled, Xf_k, Xc_k, d["cal"][1], T),
                                Xt_k, d["test"][1]),
                "shared": admit(pooled, lam_shared, Xt_k, d["test"][1])}
        out[f"{100 * p['cache_frac']:.2f}%"] = rows
    return out


def add_args(ap):
    ap.add_argument("run_dir")
    ap.add_argument("--mode", choices=["caches", "k"], required=True)
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--sizes", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.3])
    ap.add_argument("--base-size", type=float, default=0.1)
    ap.add_argument("--other-target", type=float, default=0.7)
    ap.add_argument("--perturb", type=float, default=0.3)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 10, 50, 100])
    ap.add_argument("--source-k", type=int, default=10)
    ap.add_argument("--n-each", type=int, default=20000)
    ap.set_defaults(run=run)


def run(args):
    if args.mode == "k" and args.source_k not in args.ks:
        raise SystemExit(f"--source-k {args.source_k} must be one of --ks {args.ks}")
    run_dir = Path(args.run_dir)
    data, _, run = load_run_data(run_dir, args.n_each)
    if args.mode == "k":
        logger.info(f"exact top-{max(args.ks)} for the query splits...")
        topk = {sp: exact_topk(run.sets[sp][0], run.db, max(args.ks)) for sp in SCORE_SPLITS}
        out = mode_k(data, args, run, topk)
    else:
        out = mode_caches(data, args, run, run.topk)
    path = run_dir / f"generalize_{args.mode}.json"
    path.write_text(json.dumps({"args": config_of(args), "results": out}, indent=1))
    logger.info(f"wrote {path}")
