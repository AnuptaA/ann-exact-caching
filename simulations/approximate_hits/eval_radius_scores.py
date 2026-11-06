#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Simple radius-based retention scores vs the GBT, on saved greedy caches.

Fact used throughout: for a query, every database vector closer than its true
radius (distance to its k-th nearest database vector) is a true neighbor. So
if the radius were r, retention from the cache would be
    count_within(r) = min(k, #cached vectors within r of the query) / k,
which needs only the query's k nearest cached distances. Two hard bounds:
  upper  the k-th nearest cached distance (cache is a subset of the database)
  lip    for any past query p with known radius: |radius(q) - radius(p)| <=
         d(q, p) (the k-th nearest distance moves by at most the distance moved)
Past queries = FIT split (radii known). Scores per query:
  bucket       mean of count_within over FIT radii in the query's LSH bucket
  nn1          count_within(radius of the nearest past query, capped at upper)
  nn20         mean count_within over the 20 nearest past queries' radii
  bucket_trunc bucket radii restricted to the hard interval [lip low, upper]
  lip_cert     count_within(lip low): a PROVABLE lower bound on retention
  gbt          the GBT from eval_predictors (reference)
For each score: naive admission (score >= T: mean score vs realized retention
among admitted, i.e. honesty), calibrated admission (normal_fixed lambda on
CAL; test hit rate and retention). lip_cert is also used directly (admit if
bound >= T, guaranteed per query).

Writes <run>/radius_scores.json. Requires scikit-learn (.venv python).
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
from simulations.approximate_hits.eval_certificates import pairwise_dist, _sq_norms
from simulations.approximate_hits.threshold_selection import (
    DEFAULT_METHOD, select, threshold_grid,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CHUNK = 2048
N_NEAR = 20
LSH_BITS = 8
MIN_BUCKET = 20
SCORES = ["bucket", "nn1", "nn20", "bucket_trunc", "lip_cert", "gbt"]
_R: dict = {}

#-------------------------------------------------------------------------------


def true_radius(q, ids):
    db = ecal._G["db"]
    out = np.empty(len(q))
    for s in range(0, len(q), CHUNK):
        diff = (np.asarray(q[s:s + CHUNK], dtype=np.float64)[:, None, :]
                - np.asarray(db[ids[s:s + CHUNK]], dtype=np.float64))
        out[s:s + CHUNK] = np.sqrt((diff ** 2).sum(-1)).max(axis=1)
    return out


def near_past(q, exclude_self):
    """Distances to and indices of the N_NEAR nearest FIT queries."""
    fit_q, fit_sq = _R["fit_q"], _R["fit_sq"]
    idx = np.empty((len(q), N_NEAR), dtype=np.int64)
    dist = np.empty((len(q), N_NEAR))
    for s in range(0, len(q), CHUNK):
        d = pairwise_dist(q[s:s + CHUNK], fit_q, fit_sq)
        if exclude_self:
            d[np.arange(len(d)), np.arange(s, s + len(d))] = np.inf
        p = np.argpartition(d, N_NEAR, axis=1)[:, :N_NEAR]
        dp = np.take_along_axis(d, p, axis=1)
        o = np.argsort(dp, axis=1)
        idx[s:s + CHUNK] = np.take_along_axis(p, o, axis=1)
        dist[s:s + CHUNK] = np.take_along_axis(dp, o, axis=1)
    return idx, dist


def lsh_code(q):
    return (((np.asarray(q, dtype=np.float64) - _R["mu"]) @ _R["H"]) > 0) @ _R["pow"]


def cached_knn(q, st):
    k = ecal._G["k"]
    out = np.empty((len(q), k))
    for s in range(0, len(q), CHUNK):
        d = pairwise_dist(q[s:s + CHUNK], st["db_c"], st["db_c_sq"])
        out[s:s + CHUNK] = np.sort(np.partition(d, k - 1, axis=1)[:, :k], axis=1)
    return out


def _count(dc, radii):
    """Mean over radii of min(k, #cached within radius) / k, for one query."""
    return np.searchsorted(dc, radii, side="right").mean() / len(dc)


def radius_scores(q, st, exclude_self):
    """All radius-based scores for queries q against cache state st."""
    dc = cached_knn(q, st)
    upper = dc[:, -1]
    nidx, ndist = near_past(q, exclude_self)
    nr = _R["fit_r"][nidx]
    lip_low = np.maximum((nr - ndist).max(axis=1), 0.0)
    hi = np.minimum(upper, (nr + ndist).min(axis=1))
    codes = lsh_code(q)
    out = {s: np.empty(len(q)) for s in SCORES if s != "gbt"}
    for i in range(len(q)):
        c = dc[i]
        rows = _R["buckets"].get(int(codes[i]))
        if rows is not None and exclude_self:
            rows = rows[rows != i]
        b = _R["fit_r"][rows] if rows is not None and len(rows) >= MIN_BUCKET else _R["fit_r"]
        out["bucket"][i] = _count(c, np.minimum(b, upper[i]))
        out["nn1"][i] = _count(c, [min(nr[i, 0], upper[i])])
        out["nn20"][i] = _count(c, np.minimum(nr[i], upper[i]))
        lo_i, hi_i = lip_low[i], max(hi[i], lip_low[i])
        bt = b[(b >= lo_i) & (b <= hi_i)]
        out["bucket_trunc"][i] = _count(c, bt if len(bt) else np.clip(b, lo_i, hi_i))
        out["lip_cert"][i] = _count(c, [lip_low[i]])
    return out


def _expand(split):
    return np.repeat(np.arange(len(split["y"])), split["w"].astype(np.int64))


def eval_state(task):
    target, point, entries = task
    state = epred.build_state(task)
    if state is None:
        return None
    st = ecal.state_tables(entries)
    scores = {}
    for sp in epred.SPLITS:
        q, _ = ecal._G["sets"][sp]
        scores[sp] = radius_scores(q, st, exclude_self=(sp == "fit"))
    fit = state["fit"]
    X = lambda s: np.hstack([s["base"], s["ext"]])
    g = epred._gbt("reg").fit(X(fit), fit["y"], sample_weight=fit["w"])
    for sp in epred.SPLITS:
        scores[sp]["gbt"] = g.predict(X(state[sp]))

    out = {"cache_frac": point["cache_frac"], "oracle_hit_rate": point["test_hit_rate"]}
    for name in SCORES:
        rec = {}
        test, cal = state["test"], state["cal"]
        ti, ci = _expand(test), _expand(cal)
        s_t, y_t = scores["test"][name][ti], test["y"][ti]
        adm = s_t >= target
        rec["naive_hit_rate"] = float(adm.mean())
        rec["naive_mean_score"] = float(s_t[adm].mean()) if adm.any() else None
        rec["naive_hit_retention"] = float(y_t[adm].mean()) if adm.any() else None
        grid = threshold_grid(scores["fit"][name])
        lam = select(DEFAULT_METHOD, scores["cal"][name][ci], cal["y"][ci], grid,
                     target, 0.05)
        adm = s_t >= lam if lam is not None else np.zeros(len(s_t), dtype=bool)
        rec["calibrated_hit_rate"] = float(adm.mean())
        rec["calibrated_hit_retention"] = float(y_t[adm].mean()) if adm.any() else None
        if name == "lip_cert":
            rec["cert_violations"] = float((y_t < s_t - 1e-9).mean())
        out[name] = rec
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

    fit_q, _ = ecal._G["sets"]["fit"]
    _R["fit_q"] = np.asarray(fit_q, dtype=np.float64)
    _R["fit_sq"] = _sq_norms(_R["fit_q"])
    _R["fit_r"] = true_radius(fit_q, ecal._G["topk"]["fit"])
    _R["mu"] = _R["fit_q"].mean(axis=0)
    _R["H"] = np.random.default_rng(0).standard_normal((fit_q.shape[1], LSH_BITS))
    _R["pow"] = 1 << np.arange(LSH_BITS)
    _R["fit_codes"] = lsh_code(fit_q)
    _R["buckets"] = {int(c): np.flatnonzero(_R["fit_codes"] == c)
                     for c in np.unique(_R["fit_codes"])}

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
            logger.info(f"cache {100 * s['cache_frac']:.2f}% (oracle {s['oracle_hit_rate']:.3f}):")
            for name in SCORES:
                r = s[name]
                f = lambda v: "  -- " if v is None else f"{v:.3f}"
                extra = (f" cert violations {r['cert_violations']:.4f}"
                         if "cert_violations" in r else "")
                logger.info(f"  {name:12s} naive hit {r['naive_hit_rate']:.3f} "
                            f"score {f(r['naive_mean_score'])} real {f(r['naive_hit_retention'])}"
                            f" | calibrated hit {r['calibrated_hit_rate']:.3f} "
                            f"ret {f(r['calibrated_hit_retention'])}{extra}")
            (run_dir / "radius_scores.json").write_text(
                json.dumps({"args": vars(args), "states": out}, indent=1))


if __name__ == "__main__":
    main()
