#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Revised geometric features for the retention GBT, plus ablation.

Neighborhood of a query = ball around it out to its k-th nearest cached
vector (radius c_k). Every database vector inside it but outside every cache
ball is a hidden intruder candidate; inside a cache ball nothing is hidden.

New feature groups:
  top3    for the 3 cache balls that best fit the neighborhood (smallest
          (offset + c_k) / ball radius): depth = offset / radius and
          size = c_k / radius (6 numbers; missing if fewer balls)
  region  share of probe points on the neighborhood's boundary (radius c_k,
          and 0.9 c_k) that lie outside EVERY cache ball. Probe directions:
          N_PROBES fixed random directions shaped like the data (Gaussian with
          the database covariance), shared by all queries. Pools all balls.
  shell   probe features from the targeted shell sample (eval_gap, p = 10%)
Reference groups: base20 (eval_predictors features), story5 (eval_ablation).

Per configuration: calibrated test hit rate (normal_fixed lambda on CAL) and
best-cutoff test hit rate (cutoff chosen on test; less noisy).
Writes <run>/features_v2.json. Requires scikit-learn (.venv python).
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
from simulations.approximate_hits import eval_gap as egap
from simulations.approximate_hits.eval_ablation import NAMES, STORY, _gbt
from utils.base.knn import pairwise_dist
from simulations.approximate_hits.eval_radius_scores import cached_knn
from utils.cache.calibration import (
    DEFAULT_METHOD, select, threshold_grid,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

N_TOP = 3
N_PROBES = 32
PROBE_RADII = (1.0, 0.9)
SHELL_FRAC = 0.1
CONFIGS = {
    "base20": ["base20"],
    "story5": ["story5"],
    "top1+region": ["top1", "region"],
    "top3": ["top3"],
    "top3+region": ["top3", "region"],
    "top3+region+covered": ["top3", "region", "covered"],
    "top3+region+history": ["top3", "region", "history"],
    "base20+shell": ["base20", "shell"],
    "top3+shell": ["top3", "shell"],
    "top3+region+shell": ["top3", "region", "shell"],
    "top3+region+history+shell": ["top3", "region", "history", "shell"],
}
_R: dict = {}

#-------------------------------------------------------------------------------


def probe_directions(n, seed=0):
    """Unit directions drawn from a Gaussian with the database covariance."""
    db = ecal._G["db"]
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(len(db), min(20000, len(db)), replace=False))
    x = np.asarray(db[rows], dtype=np.float64)
    x -= x.mean(axis=0)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    z = rng.standard_normal((n, len(s))) * (s / np.sqrt(len(x)))
    u = z @ vt
    return u / np.linalg.norm(u, axis=1, keepdims=True)


def geometry_features(q, ck, st):
    """top3 depth / size and region (uncovered probe share) features."""
    U = _R["U"]
    cq, r = st["ent_q"], st["ent_r"]
    r2 = (r ** 2)[None, None, :]
    uc = U @ cq.T                                        # probes x entries
    n, E = len(q), len(r)
    top = np.full((n, 2 * N_TOP), np.nan)
    region = np.empty((n, len(PROBE_RADII)))
    chunk = max(8, int(1.5e8 / (N_PROBES * max(E, 1))))
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        qs = np.asarray(q[s:e], dtype=np.float64)
        d = pairwise_dist(qs, cq, st["ent_q_sq"])
        c = ck[s:e]
        over = (c[:, None] + d) / r[None, :]
        m = min(N_TOP, E)
        part = np.argpartition(over, m - 1, axis=1)[:, :m] if m < E else \
            np.tile(np.arange(E), (e - s, 1))
        order = np.take_along_axis(part, np.argsort(np.take_along_axis(over, part, axis=1),
                                                    axis=1), axis=1)
        for t in range(m):
            j = order[:, t]
            top[s:e, 2 * t] = d[np.arange(e - s), j] / r[j]
            top[s:e, 2 * t + 1] = c / r[j]
        uq = qs @ U.T                                    # queries x probes
        d2 = (d ** 2)[:, None, :]
        for i, t in enumerate(PROBE_RADII):
            rad = (t * c)[:, None, None]
            # |q + rad u - c|^2 = |q - c|^2 + 2 rad u.(q - c) + rad^2
            p2 = d2 + 2.0 * rad * (uq[:, :, None] - uc[None, :, :]) + rad ** 2
            region[s:e, i] = 1.0 - (p2 < r2).any(axis=2).mean(axis=1)
    return top, region


def _expand(split):
    return np.repeat(np.arange(len(split["y"])), split["w"].astype(np.int64))


def eval_state(task):
    target, point, entries = task
    state = epred.build_state(task)
    if state is None:
        return None
    st = ecal.state_tables(entries)
    k = ecal._G["k"]
    col = {n: i for i, n in enumerate(NAMES)}
    shell = egap.shell_sample(entries, st["in_cache"], SHELL_FRAC)
    G = {}
    for sp in epred.SPLITS:
        q = ecal._G["sets"][sp][0]
        ck = np.maximum(cached_knn(q, st)[:, k - 1], 1e-12)
        X0 = np.hstack([state[sp]["base"], state[sp]["ext"]])
        top, region = geometry_features(q, ck, st)
        G[sp] = {"base20": X0,
                 "story5": X0[:, [col[f] for f in STORY]],
                 "covered": X0[:, [col["covered_frac"]]],
                 "history": X0[:, [col["entry_hist_mean"], col["entry_hist_log_count"]]],
                 "top1": top[:, :2], "top3": top, "region": region,
                 "shell": (egap.sample_features(q, ck, st["in_cache"], shell)
                           if shell is not None else np.zeros((len(q), egap.SAMPLE_J + 3)))}
    test, cal, fit = state["test"], state["cal"], state["fit"]
    ti, ci = _expand(test), _expand(cal)
    out = {"cache_frac": point["cache_frac"], "oracle_hit_rate": point["test_hit_rate"],
           "shell_frac": len(shell["ids"]) / len(ecal._G["db"]) if shell else 0.0}
    y = test["y"]
    reg = G["test"]["region"][:, 0]
    out["region_by_retention"] = {
        "ret1": float(np.median(reg[y == 1])) if (y == 1).any() else None,
        "ret_0.5_1": float(np.median(reg[(y >= 0.5) & (y < 1)])) if ((y >= 0.5) & (y < 1)).any()
        else None,
        "ret_lt_0.5": float(np.median(reg[y < 0.5])) if (y < 0.5).any() else None}
    for name, groups in CONFIGS.items():
        X = {sp: np.hstack([G[sp][g] for g in groups]) for sp in epred.SPLITS}
        g = _gbt(0).fit(X["fit"], fit["y"], sample_weight=fit["w"])
        s = {sp: g.predict(X[sp]) for sp in ("fit", "cal", "test")}
        lam = select(DEFAULT_METHOD, s["cal"][ci], cal["y"][ci], threshold_grid(s["fit"]),
                     target, 0.05)
        st_, yt = s["test"][ti], test["y"][ti]
        adm = st_ >= lam if lam is not None else np.zeros(len(st_), dtype=bool)
        out[name] = {"n_features": X["fit"].shape[1],
                     "calibrated_hit_rate": float(adm.mean()),
                     "calibrated_hit_retention": float(yt[adm].mean()) if adm.any() else None,
                     "best_cutoff_hit_rate": egap.best_cutoff_rate(st_, yt, target)}
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
    _R["U"] = probe_directions(N_PROBES)
    egap._R["draw"] = np.random.default_rng(1).random(len(ecal._G["db"]))  # as eval_gap
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
            logger.info(f"cache {100 * s['cache_frac']:.2f}% (oracle {s['oracle_hit_rate']:.3f},"
                        f" shell {100 * s['shell_frac']:.2f}%): region uncovered median by "
                        f"retention {s['region_by_retention']} | "
                        + " ".join(f"{n} {s[n]['calibrated_hit_rate']:.3f}/"
                                   f"{s[n]['best_cutoff_hit_rate']:.3f}"
                                   for n in CONFIGS))
            (run_dir / "features_v2.json").write_text(
                json.dumps({"args": vars(args), "configs": CONFIGS, "states": out}, indent=1))


if __name__ == "__main__":
    main()
