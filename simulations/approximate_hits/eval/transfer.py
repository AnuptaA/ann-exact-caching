#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Does a calibrated admission threshold transfer across caches?

For one saved run, build a family of caches:
  size     greedy cache for --target at each of --sizes (nested prefixes)
  target   greedy caches selected for other targets (--other-targets) at --base-size
  perturb  the --target cache at --base-size with a fraction p of its entries
           replaced by random unselected candidates of the same K (--perturb)
For every ordered pair (source A, destination B), with admission target T:
  frozen  GBT and lambda from A (fit / cal on A), applied to B's test queries
  recal   GBT from A, lambda recalibrated on B's cal queries
  native  GBT and lambda from B (the diagonal of frozen)
Features on B are always computed against B (cache-relative features, and the
entry-history features from B's fit queries relabeled against B), as an online
system could do from stored backend results.

Reports realized test retention over hits and hit rate per pair. Writes
<run>/threshold_transfer.json. Requires scikit-learn (run with .venv python).
"""

import json
import logging
from pathlib import Path


import numpy as np


from simulations.approximate_hits.common import config_of
from utils.cache.entries import perturbed
from utils.cache.legacy_features import build_state
from utils.cache.metrics import expand_index, nearest_point
from utils.cache.models import gbt
from utils.cache.parallel import worker_pool
from utils.cache.workloads import load_run_data
from utils.cache.calibration import (
    DEFAULT_METHOD, select, threshold_grid,
)

logger = logging.getLogger(__name__)

#-------------------------------------------------------------------------------


def _cache_frac(run, entries):
    ids = [run.cached_ids[j] for j in entries]
    return len(np.unique(np.concatenate(ids))) / len(run.db) if ids else 0.0


def cache_family(data, args, run):
    """[(name, task for build_state)] for all caches of the experiment."""
    by_t = {round(r["target"], 6): r for r in data["results"]}
    main = by_t[round(args.target, 6)]
    fam = []
    for size in args.sizes:
        p = nearest_point(main, size)
        fam.append((f"size{100 * size:g}%", (args.target, p, main["selected"][:p["n_entries"]])))
    for t in args.other_targets:
        r = by_t[round(t, 6)]
        p = nearest_point(r, args.base_size)
        fam.append((f"T{t:g}@{100 * args.base_size:g}%",
                    (args.target, p, r["selected"][:p["n_entries"]])))
    p = nearest_point(main, args.base_size)
    base = main["selected"][:p["n_entries"]]
    n_kv = len(run.k_values)
    n_cand = len(run.cand_q)
    rng = np.random.default_rng(args.seed)
    for frac in args.perturb:
        ent = perturbed(base, frac, n_cand, n_kv, rng)
        q = {"cache_frac": _cache_frac(run, ent), "test_hit_rate": None,
             "test_hit_rate_online": None}
        fam.append((f"perturb{100 * frac:g}%@{100 * args.base_size:g}%", (args.target, q, ent)))
    return fam


def _X(split):
    return np.hstack([split["base"], split["ext"]])


def _admit(scores, split, lam):
    idx = expand_index(split["w"])
    s, y = scores[idx], split["y"][idx]
    adm = s >= lam if lam is not None else np.zeros(len(s), dtype=bool)
    return float(adm.mean()), (float(y[adm].mean()) if adm.any() else None)


def _lambda(model, src, cal, target, delta):
    grid = threshold_grid(model.predict(_X(src["fit"])))
    idx = expand_index(cal["w"])
    return select(DEFAULT_METHOD, model.predict(_X(cal))[idx], cal["y"][idx],
                  grid, target, delta)


def add_args(ap):
    ap.add_argument("run_dir")
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--sizes", type=float, nargs="+", default=[0.001, 0.005, 0.01, 0.05])
    ap.add_argument("--base-size", type=float, default=0.01)
    ap.add_argument("--other-targets", type=float, nargs="+", default=[0.7, 0.95])
    ap.add_argument("--perturb", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    ap.add_argument("--n-each", type=int, default=20000)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.set_defaults(run=run)


def run(args):
    run_dir = Path(args.run_dir)

    data, cfg, run = load_run_data(run_dir, args.n_each)
    fam = cache_family(data, args, run)
    logger.info(f"Building features for {len(fam)} caches...")
    with worker_pool(build_state, args.workers, len(fam), run=run) as pool:
        built = pool.map([task for _, task in fam])
    caches = [(name, st) for (name, _), st in zip(fam, built) if st is not None]

    logger.info("Fitting one GBT per cache...")
    models = {}
    for name, st in caches:
        fit = st["fit"]
        models[name] = gbt("reg").fit(_X(fit), fit["y"], sample_weight=fit["w"])
    lam_own = {name: _lambda(models[name], st, st["cal"], args.target, args.delta)
               for name, st in caches}

    pairs = []
    for a, sa in caches:
        for b, sb in caches:
            test_scores = models[a].predict(_X(sb["test"]))
            lam_rec = _lambda(models[a], sb, sb["cal"], args.target, args.delta)
            rec = {"source": a, "dest": b, "dest_cache_frac": sb["cache_frac"]}
            for tag, lam in (("frozen", lam_own[a]), ("recal", lam_rec)):
                rec[f"{tag}_hit_rate"], rec[f"{tag}_hit_retention"] = _admit(
                    test_scores, sb["test"], lam)
            pairs.append(rec)
    native = {p["dest"]: p for p in pairs if p["source"] == p["dest"]}
    for p in pairs:
        n = native[p["dest"]]
        p["native_hit_rate"] = n["frozen_hit_rate"]
        p["native_hit_retention"] = n["frozen_hit_retention"]

    for tag in ("frozen", "recal"):
        off = [p for p in pairs if p["source"] != p["dest"] and p[f"{tag}_hit_retention"]]
        gaps = np.array([p[f"{tag}_hit_retention"] - args.target for p in off])
        if len(gaps):
            logger.info(f"  {tag}: {len(off)} off-diagonal pairs with hits; worst retention - T "
                        f"{gaps.min():+.3f}; below T-0.01: {(gaps < -0.01).mean():.0%}")
    out = {"args": config_of(args), "caches": [{"name": n, "cache_frac": st["cache_frac"]}
                                          for n, st in caches], "pairs": pairs}
    (run_dir / "threshold_transfer.json").write_text(json.dumps(out, indent=1))
    logger.info(f"Wrote {run_dir / 'threshold_transfer.json'}")
