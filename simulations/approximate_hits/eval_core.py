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

import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from simulations.approximate_hits import eval_calibrated as ecal
from simulations.approximate_hits import eval_predictors as epred
from simulations.approximate_hits.eval_ablation import _gbt
from simulations.approximate_hits.eval_aker_thresholds import fetch
from simulations.approximate_hits.eval_certificates import pairwise_dist, _sq_norms
from simulations.approximate_hits.eval_gap import best_cutoff_rate, entry_kmax
from simulations.approximate_hits.eval_radius_scores import cached_knn
from simulations.approximate_hits.threshold_selection import (
    DEFAULT_METHOD, select, threshold_grid,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

EPS = 1e-12
SAMPLE_J = 5
COUNT_SCALES = (1.0, 1.1, 1.25)
DEFAULT = {"n": 3, "p": 0.1, "R": 500, "b": 32}
_R: dict = {}

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


def balls_of(entries):
    """One ball per cached past query (its largest K): centers, radii."""
    cand, K = entry_kmax(entries)
    n_kv = len(ecal._G["k_values"])
    t = np.array([ecal._G["k_values"].index(int(k)) for k in K])
    rad = ecal._G["radius"][cand * n_kv + t]
    cq = np.asarray(ecal._G["cand_q"][cand], dtype=np.float64)
    return {"cand": cand, "K": K, "c": cq, "c_sq": _sq_norms(cq), "r": rad}


def ball_features(q, nr, bl, n):
    """Relative offset and size for the n best-fitting balls."""
    F = np.full((len(q), 2 * n), np.nan)
    m = min(n, len(bl["r"]))
    for s in range(0, len(q), 2048):
        d = pairwise_dist(q[s:s + 2048], bl["c"], bl["c_sq"])
        c = nr[s:s + 2048]
        fit = (c[:, None] + d) / bl["r"][None, :]
        part = (np.argpartition(fit, m - 1, axis=1)[:, :m] if m < fit.shape[1]
                else np.tile(np.arange(fit.shape[1]), (len(d), 1)))
        order = np.take_along_axis(part, np.argsort(np.take_along_axis(fit, part, axis=1),
                                                    axis=1), axis=1)
        rows = np.arange(len(d))
        for t in range(m):
            j = order[:, t]
            F[s:s + 2048, 2 * t] = d[rows, j] / bl["r"][j]
            F[s:s + 2048, 2 * t + 1] = c / bl["r"][j]
    return F


def quantize(x, bits):
    """b-bit uniform scalar quantization per dimension (reconstructed)."""
    if bits >= 32:
        return x
    lo, hi = x.min(axis=0), x.max(axis=0)
    step = np.maximum(hi - lo, EPS) / (2 ** bits - 1)
    return lo + np.round((x - lo) / step) * step


def load_cand_ext(run_dir, R):
    """Exact top-R database ids per candidate query (for shells deeper than the
    cached top-500), computed once and saved to <run>/cand_top<R>.npy."""
    from simulations.approximate_hits.run_e2e_oracle_sift import exact_topk
    path = Path(run_dir) / f"cand_top{R}.npy"
    if path.exists():
        return np.load(path)
    logger.info(f"exact top-{R} for {len(ecal._G['cand_q'])} candidate queries...")
    ids = exact_topk(ecal._G["cand_q"], ecal._G["db"], R)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, ids)
    tmp.rename(path)
    return ids


def shell_ids(cand, K, R):
    """Database ids at ranks K+1..R around one candidate query. Uses the saved
    deeper neighbor lists (_R["cand_ext"]) when R exceeds the cached top-500."""
    n_kv = len(ecal._G["k_values"])
    top = ecal._G["cached_ids"][cand * n_kv + n_kv - 1]
    if R > len(top):
        return np.asarray(_R["cand_ext"][cand][K:R])
    return np.asarray(top[K:R])


def shell_sample(bl, in_cache, p, R, bits):
    shells = [shell_ids(i, k, R) for i, k in zip(bl["cand"], bl["K"]) if k < R]
    ids = np.unique(np.concatenate(shells)) if shells else np.empty(0, np.int64)
    ids = ids[~in_cache[ids]]
    ids = ids[_R["draw"][ids] < p]
    if len(ids) < SAMPLE_J:
        return None
    vecs = quantize(fetch(ecal._G["db"], ids), bits)
    return {"ids": ids, "vecs": vecs, "sq": _sq_norms(vecs),
            "memory": len(ids) * min(bits, 32) / 32 / len(ecal._G["db"])}


def shell_features(q, nr, samp):
    F = np.zeros((len(q), SAMPLE_J + len(COUNT_SCALES)))
    if samp is None:
        return F
    for s in range(0, len(q), 1024):
        d = pairwise_dist(q[s:s + 1024], samp["vecs"], samp["sq"])
        c = nr[s:s + 1024][:, None]
        near = np.sort(np.partition(d, SAMPLE_J - 1, axis=1)[:, :SAMPLE_J], axis=1)
        F[s:s + 1024, :SAMPLE_J] = np.log((near + EPS) / c)
        for i, t in enumerate(COUNT_SCALES):
            F[s:s + 1024, SAMPLE_J + i] = np.log1p((d < t * c).sum(axis=1))
    return F


def oracle_at(points, frac):
    xs = np.array([p["cache_frac"] for p in points])
    ys = np.array([p["test_hit_rate"] for p in points])
    o = np.argsort(xs)
    xs, ys = xs[o], ys[o]
    if frac < xs[0] or frac > xs[-1] * 1.001:
        return None
    return float(np.interp(np.log(frac), np.log(xs), ys))


def _expand(split):
    return np.repeat(np.arange(len(split["y"])), split["w"].astype(np.int64))


def eval_state(task):
    target, point, entries = task
    k = ecal._G["k"]
    st = ecal.state_tables(entries)
    if len(st["cached"]) <= k:
        return None
    bl = balls_of(entries)
    sets, topk = ecal._G["sets"], ecal._G["topk"]
    y = {sp: st["in_cache"][topk[sp]].mean(axis=1) for sp in ("fit", "cal", "test")}
    w = {sp: sets[sp][1].astype(np.float64) for sp in ("fit", "cal", "test")}
    nr = {sp: np.maximum(cached_knn(sets[sp][0], st)[:, k - 1], EPS)
          for sp in ("fit", "cal", "test")}
    ball_cache, shell_cache = {}, {}
    out = {"cache_frac": point["cache_frac"], "n_balls": len(bl["r"]),
           "oracle_hit_rate": point["test_hit_rate"]}
    ti = _expand({"y": y["test"], "w": w["test"]})
    ci = _expand({"y": y["cal"], "w": w["cal"]})
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
                samp = shell_sample(bl, st["in_cache"], *sh)
                shell_cache[sh] = ({sp: shell_features(sets[sp][0], nr[sp], samp)
                                    for sp in blocks}, samp["memory"] if samp else 0.0)
            feats, mem = shell_cache[sh]
            for sp in blocks:
                blocks[sp].append(feats[sp])
        X = {sp: np.hstack(blocks[sp]) for sp in blocks}
        g = _gbt(0).fit(X["fit"], y["fit"], sample_weight=w["fit"])
        s = {sp: g.predict(X[sp]) for sp in blocks}
        lam = select(DEFAULT_METHOD, s["cal"][ci], y["cal"][ci], threshold_grid(s["fit"]),
                     target, 0.05)
        st_, yt = s["test"][ti], y["test"][ti]
        adm = st_ >= lam if lam is not None else np.zeros(len(st_), dtype=bool)
        out[name] = {"n_features": X["fit"].shape[1], "sample_memory": mem,
                     "calibrated_hit_rate": float(adm.mean()),
                     "calibrated_hit_retention": float(yt[adm].mean()) if adm.any() else None,
                     "best_cutoff_hit_rate": best_cutoff_rate(st_, yt, target),
                     "oracle_same_total_memory": oracle_at(_R["points"],
                                                           point["cache_frac"] + mem)}
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
    _R["draw"] = np.random.default_rng(1).random(len(ecal._G["db"]))
    res = next(r for r in data["results"] if abs(r["target"] - args.target) < 1e-9)
    _R["points"] = res["points"]
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
            logger.info(f"cache {100 * s['cache_frac']:.2f}% (oracle {s['oracle_hit_rate']:.3f},"
                        f" {s['n_balls']} balls): " + " ".join(
                            f"{n} {s[n]['calibrated_hit_rate']:.3f}"
                            f"(+{100 * s[n]['sample_memory']:.1f}%)" for n in configs()))
            (run_dir / "core.json").write_text(
                json.dumps({"args": vars(args), "default": DEFAULT,
                            "configs": {n: [c[0], c[1]] for n, c in configs().items()},
                            "states": out}, indent=1))


if __name__ == "__main__":
    main()
