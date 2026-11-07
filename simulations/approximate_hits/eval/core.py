#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Oracle vs retention models built only from the two feature families we care
about. Terms (per query):
  neighborhood radius  distance from the query to its k-th closest cached vector
  ball                 one per cached past query (its largest cached K):
                       center = that past query, radius = distance from the
                       center to its K-th nearest database vector
  offset               distance from the query to a ball's center

Feature families:
  balls   per-ball stats for the n balls the neighborhood fits best (ranked by
          (offset + neighborhood radius) / ball radius, the worst-case fit
          test): relative offset = offset / ball radius and relative size =
          neighborhood radius / ball radius (2n numbers, missing if fewer balls)
  shell   targeted sample of the space around the balls: for every ball, the
          uncached database vectors at ranks K+1..R around its center (known
          from its backend answer), each kept iff a fixed per-vector uniform
          draw is < p; stored with b-bit uniform scalar quantization per
          dimension (b = 32: uncompressed). Features (distances to the
          reconstructed vectors): log(d_j / neighborhood radius) for the query's
          j = 1..5 nearest sample vectors, and log1p of the number of sample
          vectors within 1.0 / 1.1 / 1.25 x the neighborhood radius (inside
          1.0 x they are seen intruders, since sample vectors are uncached)
Knobs: n (balls), p (sample fraction), R (shell radius, as a rank), b (bits).

Per configuration: calibrated test hit rate and retention (normal_fixed
lambda on CAL), best-cutoff hit rate (cutoff chosen on test), sample memory
(vector-equivalents / database size = n_sample * b / 32 / N), and the oracle
at the same cache size and at the same TOTAL memory (cache + sample;
interpolated on the greedy oracle curve, None beyond its range).

Writes <run>/core.json. Requires scikit-learn (.venv python).
"""

import json
import logging
from pathlib import Path


import numpy as np


from simulations.approximate_hits.common import config_of
from utils.cache.models import gbt_regressor
from utils.cache.metrics import best_cutoff_rate, result_for_target, state_tasks
from utils.cache.features import ball_features, split_inputs
from utils.cache.metrics import expand_index
from utils.cache.parallel import worker_pool
from utils.cache.shell import ensure_shell_depth, shell_features, shell_sample
from utils.cache.workloads import SCORE_SPLITS, load_run_data
from utils.cache.calibration import (
    DEFAULT_METHOD, select, threshold_grid,
)

logger = logging.getLogger(__name__)

EPS = 1e-12
DEFAULT = {"n": 3, "p": 0.1, "R": 500, "b": 32}

#-------------------------------------------------------------------------------


def configs():
    """name -> (n balls or 0, shell (p, R, b) or None)."""
    d = DEFAULT
    shell = (d["p"], d["R"], d["b"])
    out = {f"balls{n}": (n, None) for n in (1, 3, 5, 10)}
    out["shell"] = (0, shell)
    for n in (1, 3, 5, 10):
        out[f"balls{n}+shell"] = (n, shell)
    for p in (0.05, 0.25, 0.5):
        out[f"balls3+shell(p={p:g})"] = (d["n"], (p, d["R"], d["b"]))
    for R in (100, 200):
        out[f"balls3+shell(R={R})"] = (d["n"], (d["p"], R, d["b"]))
    for b in (8, 4, 2):
        out[f"balls3+shell(b={b})"] = (d["n"], (d["p"], d["R"], b))
    for p in (0.25, 0.5):
        out[f"balls3+shell(p={p:g},b=4)"] = (d["n"], (p, d["R"], 4))
    return out


def oracle_at(points, frac):
    xs = np.array([p["cache_frac"] for p in points])
    ys = np.array([p["test_hit_rate"] for p in points])
    o = np.argsort(xs)
    xs, ys = xs[o], ys[o]
    if frac < xs[0] or frac > xs[-1] * 1.001:
        return None
    return float(np.interp(np.log(frac), np.log(xs), ys))


def eval_state(task, run, args, points):
    target, point, entries = task
    inp = split_inputs(run, entries, SCORE_SPLITS, n_balls=None)
    if inp is None:
        return None
    sets, st, bl, y, w, nr = run.sets, inp.st, inp.bl, inp.y, inp.w, inp.nr
    ball_cache, shell_cache = {}, {}
    out = {"cache_frac": point["cache_frac"], "n_balls": len(bl["r"]),
           "oracle_hit_rate": point["test_hit_rate"]}
    ti = expand_index(w["test"])
    ci = expand_index(w["cal"])
    for name, (n, sh) in configs().items():
        blocks = {sp: [] for sp in ("fit", "cal", "test")}
        if n:
            if n not in ball_cache:
                ball_cache[n] = {sp: ball_features(sets[sp][0], nr[sp], bl, n)
                                 for sp in blocks}
            for sp in blocks:
                blocks[sp].append(ball_cache[n][sp])
        mem = 0.0
        if sh is not None:
            if sh not in shell_cache:
                samp = shell_sample(run, bl, st["in_cache"], *sh)
                shell_cache[sh] = ({sp: shell_features(sets[sp][0], nr[sp], samp)
                                    for sp in blocks}, samp["memory"] if samp else 0.0)
            feats, mem = shell_cache[sh]
            for sp in blocks:
                blocks[sp].append(feats[sp])
        X = {sp: np.hstack(blocks[sp]) for sp in blocks}
        g = gbt_regressor(0).fit(X["fit"], y["fit"], sample_weight=w["fit"])
        s = {sp: g.predict(X[sp]) for sp in blocks}
        lam = select(DEFAULT_METHOD, s["cal"][ci], y["cal"][ci], threshold_grid(s["fit"]),
                     target, 0.05)
        st_, yt = s["test"][ti], y["test"][ti]
        adm = st_ >= lam if lam is not None else np.zeros(len(st_), dtype=bool)
        out[name] = {"n_features": X["fit"].shape[1], "sample_memory": mem,
                     "calibrated_hit_rate": float(adm.mean()),
                     "calibrated_hit_retention": float(yt[adm].mean()) if adm.any() else None,
                     "best_cutoff_hit_rate": best_cutoff_rate(st_, yt, target),
                     "oracle_same_total_memory": oracle_at(points, point["cache_frac"] + mem)}
    return out


#-------------------------------------------------------------------------------


def add_args(ap):
    ap.add_argument("run_dir")
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--sizes", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.3])
    ap.add_argument("--n-each", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=4)
    ap.set_defaults(run=run)


def run(args):
    run_dir = Path(args.run_dir)
    data, _, run = load_run_data(run_dir, args.n_each)
    ensure_shell_depth(run, run_dir, DEFAULT["R"])
    res = result_for_target(data, args.target)
    tasks = state_tasks(res, args.sizes, args.target)
    out = []
    with worker_pool(eval_state, args.workers, len(tasks), run=run, args=args,
                     points=res["points"]) as pool:
        for s in pool.imap(tasks):
            if s is None:
                continue
            out.append(s)
            logger.info(f"cache {100 * s['cache_frac']:.2f}% (oracle {s['oracle_hit_rate']:.3f},"
                        f" {s['n_balls']} balls): " + " ".join(
                            f"{n} {s[n]['calibrated_hit_rate']:.3f}"
                            f"(+{100 * s[n]['sample_memory']:.1f}%)" for n in configs()))
            (run_dir / "core.json").write_text(
                json.dumps({"args": config_of(args), "default": DEFAULT,
                            "configs": {n: [c[0], c[1]] for n, c in configs().items()},
                            "states": out}, indent=1))
