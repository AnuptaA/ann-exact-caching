#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Detector study: replay drift tests on graded workload shifts (context doc 5v).

A detector only needs, per query, its score and retention against the cache
in use and whether it is served. So the state at the moment a shift
arrives (warm-up cache, model, lambda; no response yet) is fixed, scores and
retentions are computed once for every query (--prepare), and detectors are
replayed on many sampled streams (--run):
  stream = P pre-shift queries from group A (the warm-up mix), then H
           post-shift queries from (1 - eps) A + eps B (B = the other
           clusters; eps = 0: no shift, for false alarms), drawn with
           replacement (scores do not change with repeats: cache fixed)
  served = score >= lambda (frozen warm-up lambda)
Because the cache and lambda are fixed, the true served retention of each
post-shift mix is known exactly: harmful iff < T.

Detectors (checked every B queries; level schedules below):
  score      two-sample KS test, the last W scores vs a reference: "cal"
             (the warm-up calibration queries' scores) or "grow" (every
             score since warm-up before the recent window)
  hitrate    two-proportion z test, share of scores >= lambda, same samples
  retention  upper confidence bound on the mean retention of audited served
             queries is < T; data: the last n audits ("count") or the audits
             of the last W queries ("time"), or several time windows at once
             ("scan", level split over them); bounds:
               hoeffding  range only
               kl         Chernoff / KL form of Hoeffding (same assumptions,
                          tighter near the edge of [0, 1])
               eb         empirical Bernstein (same assumptions, uses the
                          observed variance)
               normal     CLT (stronger assumption; as in calibration)
             audit policy: fixed rate a, or adaptive: rate for the next
             batch = clip(A / hits in the last batch, a0, 1) (aims at A
             audits per batch; label-free decision, so valid); audited
             retentions weighted 1 / rate (effective n = (sum w)^2 / sum w^2)
Level schedules: jj = delta / (j (j + 1)) for the j-th check (anytime);
fixed = delta / J for a known number of checks J; none = delta (no
correction, to show it is needed).

Metrics per (detector, eps), over R replications: false alarms (share of
runs alarming before the shift), detection rate within H, median delay
(queries from the shift to the alarm), audits per 1000 queries.
"""

import sys
import json
import argparse
import logging
import itertools
from pathlib import Path
from statistics import NormalDist

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

RES = Path("simulations/approximate_hits/results")
T, DELTA = 0.9, 0.05
EPS_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
SCAN = (1000, 2000, 5000, 10000, 20000)
_G: dict = {}

#-------------------------------------------------------------------------------
# bounds


def hoeffding_ucb(m, n, level):
    return m + np.sqrt(np.log(1 / level) / (2 * n))


def kl_ucb(m, n, level, iters=40):
    """Largest mu >= m with n kl(m, mu) <= log(1 / level) (Bernoulli KL;
    valid for [0, 1] variables, Hoeffding 1963)."""
    m = float(np.clip(m, 0.0, 1.0))
    c = np.log(1 / level) / n
    lo, hi = m, 1.0

    def kl(p, q):
        q = min(max(q, 1e-12), 1 - 1e-12)
        a = p * np.log(p / q) if p > 0 else 0.0
        b = (1 - p) * np.log((1 - p) / (1 - q)) if p < 1 else 0.0
        return a + b
    if kl(m, 1.0 - 1e-12) <= c:
        return 1.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if kl(m, mid) <= c:
            lo = mid
        else:
            hi = mid
    return hi


def eb_ucb(m, sd, n, level):
    """Empirical Bernstein (Maurer-Pontil), values in [0, 1]."""
    lt = np.log(2 / level)
    return m + np.sqrt(2 * sd ** 2 * lt / n) + 7 * lt / (3 * (n - 1))


def normal_ucb(m, sd, n, level):
    return m + NormalDist().inv_cdf(1 - level) * sd / np.sqrt(n)


def ucb(kind, y, w, level):
    """Upper bound on the (weighted) mean of y; effective n from weights."""
    W = w.sum()
    m = float((w * y).sum() / W)
    n = float(W ** 2 / (w ** 2).sum())
    if n < 2:
        return np.inf
    sd = float(np.sqrt(max((w * (y - m) ** 2).sum() / W * n / (n - 1), 0.0)))
    if kind == "hoeffding":
        return hoeffding_ucb(m, n, level)
    if kind == "kl":
        return kl_ucb(m, n, level)
    if kind == "eb":
        return eb_ucb(m, sd, n, level)
    return normal_ucb(m, sd, n, level)


def level_at(sched, j, J):
    if sched == "jj":
        return DELTA / (j * (j + 1))
    if sched == "fixed":
        return DELTA / J
    return DELTA


#-------------------------------------------------------------------------------
# prepare: scores and retentions against the warm-up cache


def prepare(prep, out, n_warm=20000):
    import argparse as ap_
    from simulations.approximate_hits import online_sim as osim
    a = ap_.Namespace(k=10, target=T, delta=0.05, budget=0.1 if "esci" in prep else 0.01,
                      f=0.85, R=500, bits=4, warmup=n_warm, seed=0, cal_span=20000,
                      n_db=1000000)
    D, streams = osim.load(RES / "online" / prep, a)
    labels = np.load(RES / "online" / prep / "clusters.npy")
    group = streams["group"]
    trace = "trace" in streams.files
    requests = streams["trace"] if trace else np.random.default_rng(0).permutation(len(labels))
    qg = group[labels[requests]]
    pos_a = np.flatnonzero(qg == 0)
    warm = requests[pos_a[:n_warm]]
    sim = osim.Sim(D, warm, a)
    sim.warmup()
    lam = sim.lam
    cal_scores = sim.cal[0][1]
    # pools of requests (positions after warm-up); ESCI: each query once
    rest_a = requests[pos_a[n_warm:]]
    rest_b = requests[qg == 1]
    if not trace:
        rest_a = np.setdiff1d(rest_a, warm)
    uq = np.unique(np.r_[rest_a, rest_b])
    s = np.empty(len(D["q"]))
    y = np.empty(len(D["q"]))
    for i in range(0, len(uq), 8192):
        idx = uq[i:i + 8192]
        X, _ = sim.cache.features(idx)
        s[idx] = sim.model(X)
        y[idx] = sim.cache.retention(idx)
        logger.info(f"{prep}: features {min(i + 8192, len(uq))} / {len(uq)}")
    np.savez(out / f"{prep}.npz", s=s, y=y, pool_a=rest_a, pool_b=rest_b,
             cal_scores=cal_scores, lam=lam)
    logger.info(f"{prep}: lambda {lam}, pool A {len(rest_a)}, pool B {len(rest_b)}")


def served_retention(z, eps):
    """Exact served retention and hit rate of the post-shift mix."""
    s, y, lam = z["s"], z["y"], float(z["lam"])
    out = []
    for pool, wt in ((z["pool_a"], 1 - eps), (z["pool_b"], eps)):
        hit = s[pool] >= lam
        out.append((wt * hit.mean(), wt * (y[pool] * hit).mean()))
    h = out[0][0] + out[1][0]
    return (out[0][1] + out[1][1]) / h if h > 0 else None, h


#-------------------------------------------------------------------------------
# detectors


def configs():
    out = []
    for kind in ("score", "hitrate"):
        for ref, W, sched in itertools.product(("cal", "grow"), (500, 2000, 5000), ("jj", "fixed")):
            out.append({"kind": kind, "ref": ref, "W": W, "sched": sched, "B": 1000})
        out.append({"kind": kind, "ref": "grow", "W": 2000, "sched": "jj", "B": 250})
        out.append({"kind": kind, "ref": "grow", "W": 2000, "sched": "none", "B": 1000})
    for win, bound, audit in itertools.product(
            (("count", 100), ("count", 500), ("time", 2000), ("time", 5000), ("time", 20000),
             ("scan", 0)),
            ("hoeffding", "kl", "eb", "normal"),
            (("fixed", 0.1), ("fixed", 0.3), ("adapt", 10), ("adapt", 30))):
        out.append({"kind": "retention", "win": win[0], "n": win[1], "bound": bound,
                    "audit": audit[0], "a": audit[1], "sched": "jj", "B": 1000})
    for bound in ("kl", "normal"):
        out.append({"kind": "retention", "win": "scan", "n": 0, "bound": bound,
                    "audit": "adapt", "a": 30, "sched": "fixed", "B": 1000})
        out.append({"kind": "retention", "win": "scan", "n": 0, "bound": bound,
                    "audit": "adapt", "a": 30, "sched": "none", "B": 1000})
    return out


def name_of(c):
    if c["kind"] == "retention":
        return (f"ret_{c['win']}{c['n'] or ''}_{c['bound']}_{c['audit']}{c['a']:g}"
                f"_{c['sched']}")
    return f"{c['kind']}_{c['ref']}_W{c['W']}_{c['sched']}_B{c['B']}"


def run_one(c, s, y, u, lam, cal_scores, P):
    """Index (query position) of the first alarm, or None; audits made."""
    from scipy.stats import ks_2samp
    B = c["B"]
    n = len(s)
    J = n // B
    hit = s >= lam
    audits = 0
    a_rate = c.get("a", 0.1) if c.get("audit") == "fixed" else 0.1
    aud_t, aud_y, aud_w = [], [], []
    for j in range(1, J + 1):
        lo, hi = (j - 1) * B, j * B
        level = level_at(c["sched"], j, J)
        if c["kind"] == "retention":
            h = hit[lo:hi]
            au = h & (u[lo:hi] < a_rate)
            audits += int(au.sum())
            idx = np.flatnonzero(au) + lo
            aud_t.append(idx)
            aud_y.append(y[idx])
            aud_w.append(np.full(len(idx), 1 / a_rate))
            if c["audit"] == "adapt":
                a_rate = float(np.clip(c["a"] / max(h.sum(), 1), 0.1, 1.0))
            t_all = np.concatenate(aud_t)
            y_all, w_all = np.concatenate(aud_y), np.concatenate(aud_w)
            if c["win"] == "count":
                sel = [slice(max(len(y_all) - c["n"], 0), None)]
                lv = level
            elif c["win"] == "time":
                sel = [t_all >= hi - c["n"]]
                lv = level
            else:
                sel = [t_all >= hi - w for w in SCAN]
                lv = level / len(SCAN)
            for m in sel:
                yy, ww = y_all[m], w_all[m]
                n_min = 30 if c["bound"] == "normal" else 10
                if len(yy) >= n_min and ucb(c["bound"], yy, ww, lv) < T:
                    return hi, audits
        else:
            if hi < c["W"]:
                continue
            rec = s[hi - c["W"]:hi]
            ref = cal_scores if c["ref"] == "cal" else s[:hi - c["W"]]
            if len(ref) < 1000:
                continue
            if c["kind"] == "score":
                p = ks_2samp(rec, ref, method="asymp").pvalue
            else:
                x1, x0 = (rec >= lam).sum(), (ref >= lam).sum()
                from simulations.approximate_hits.online_sim import two_prop_pvalue
                p = two_prop_pvalue(x1, len(rec), x0, len(ref))
            if p < level:
                return hi, audits
    return None, audits


def task(args):
    prep, eps, rep, P, H = args
    z = _G[prep]
    rng = np.random.default_rng([rep, int(eps * 1000), P])
    pre = rng.choice(z["pool_a"], P)
    n_b = rng.binomial(H, eps)
    post = np.r_[rng.choice(z["pool_a"], H - n_b), rng.choice(z["pool_b"], n_b)]
    q = np.r_[pre, rng.permutation(post)]
    s, y = z["s"][q], z["y"][q]
    u = rng.random(len(q))
    lam = float(z["lam"])
    out = []
    for c in configs():
        t, audits = run_one(c, s, y, u, lam, z["cal_scores"], P)
        out.append((t, audits))
    return eps, rep, out


def run(preps, out, R, P, H, workers):
    import multiprocessing as mp
    for prep in preps:
        z = dict(np.load(out / f"{prep}.npz"))
        _G[prep] = z
        harm = {eps: served_retention(z, eps) for eps in EPS_GRID}
        logger.info(f"{prep}: served retention / hit rate by eps " +
                    " ".join(f"{e}: {r:.3f}/{h:.3f}" for e, (r, h) in harm.items()))
        tasks = [(prep, eps, rep, P, H) for eps in EPS_GRID for rep in range(R)]
        res = {}
        with mp.get_context("fork").Pool(workers) as pool:
            for i, (eps, rep, o) in enumerate(pool.imap_unordered(task, tasks, chunksize=1)):
                res[(eps, rep)] = o
                if (i + 1) % 50 == 0:
                    logger.info(f"{prep}: {i + 1} / {len(tasks)} streams")
        cs = configs()
        table = {}
        for ci, c in enumerate(cs):
            row = {}
            for eps in EPS_GRID:
                ts = [res[(eps, r)][ci][0] for r in range(R)]
                au = [res[(eps, r)][ci][1] for r in range(R)]
                false = np.mean([t is not None and t <= P for t in ts])
                det = [t - P for t in ts if t is not None and t > P]
                row[str(eps)] = {"false_alarm": float(false),
                                 "detect": len(det) / R,
                                 "median_delay": float(np.median(det)) if det else None,
                                 "audits_per_1000": float(np.mean(au)) / (P + H) * 1000}
            table[name_of(c)] = {"config": c, "by_eps": row}
        (out / f"{prep}_P{P}_detectors.json").write_text(json.dumps(
            {"P": P, "H": H, "R": R, "harm": {str(e): v for e, v in harm.items()},
             "table": table}, indent=1))
        logger.info(f"wrote {out / prep}_P{P}_detectors.json")


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--prepare", type=str, nargs="*", default=[])
    ap.add_argument("--run", type=str, nargs="*", default=[])
    ap.add_argument("--R", type=int, default=100, help="replications per eps")
    ap.add_argument("--P", type=int, default=20000, help="pre-shift queries after warm-up")
    ap.add_argument("--H", type=int, default=40000, help="post-shift queries")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", type=str, default=str(RES / "detectors"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in args.prepare:
        prepare(p, out)
    if args.run:
        run(args.run, out, args.R, args.P, args.H, args.workers)


if __name__ == "__main__":
    main()
