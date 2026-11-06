#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Do the features keep their meaning across cache states? (context doc 5t)

A feature generalizes if P(retention | features) does not depend on hidden
properties of the cache state. Cache states (ESCI 1M, online_sim.Cache,
greedy cache built like the online warm-up) differ from a base state in one
property at a time:
  base     budget 10%, cache share f = 0.85 (p set to fill the budget),
           R = 500, 4 bits, K in {10, ..., 500}
  f*       cache share 0.95 / 0.7 / 0.5 (p fills the rest: other p)
  p*       the base cache with p forced to 0.1 / 0.25 / 0.5 / 1.0
  budget*  2% / 5% / 20%
  R*       shell to rank 200 / 300
  bits*    8 / 2 bits
  Ksmall   K in {10, 20, 50};  Klarge  K in {200, 500} (K = R: no shell)
  seedB    base settings, other candidate / scoring queries (another cache)
Labels and features at k = 10, 20, 50 (one cache, built for k = 10).

--state NAME: compute and save raw ingredients per query split (fit / cal /
  test, disjoint from the queries that built the cache) to
  <out>/<NAME>.npz: per k, relative offset and size for the 3 best-fitting
  balls, raw intruder count, shell radius / ball radius of those balls,
  retention; plus p, R, bits.
--analyze: feature variants
  base7   current: offsets, sizes, log1p(count)
  est7    log1p(count / p)
  ratio7  k / (k + count / p): a direct retention estimate from the
          neighborhood's k cached vs estimated uncached vectors
  cov8    ratio7 + coverage of the best ball's shell, (offset + size) /
          (shell radius / ball radius); <= 1: neighborhood inside the shell
For each variant: model fit on the base state at k = 10, applied to every
(state, k): hit rate with lambda recalibrated on that state's cal split,
relative to a model fit natively on it (ranking transfer), and retention /
hit rate on test at the base lambda (frozen, calibration transfer); also a
pooled model fit on the budget / f / p states, tested on the others.
"""

import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PREP = Path("simulations/approximate_hits/results/online/esci1m")
KS = (10, 20, 50)
SPLITS = {"cand": 4000, "score": 8000, "fit": 15000, "cal": 15000, "test": 30000}
BASE = {"budget": 0.1, "f": 0.85, "R": 500, "bits": 4, "Ks": None, "p": None, "seed": 0}
STATES = {"base": {},
          "f95": {"f": 0.95}, "f70": {"f": 0.7}, "f50": {"f": 0.5},
          "p0.1": {"p": 0.1}, "p0.25": {"p": 0.25}, "p0.5": {"p": 0.5}, "p1.0": {"p": 1.0},
          "budget2": {"budget": 0.02}, "budget5": {"budget": 0.05}, "budget20": {"budget": 0.2},
          "R200": {"R": 200}, "R300": {"R": 300}, "bits8": {"bits": 8}, "bits2": {"bits": 2},
          "Ksmall": {"Ks": (10, 20, 50)}, "Klarge": {"Ks": (200, 500)}, "seedB": {"seed": 1}}
POOL_TRAIN = ("base", "f95", "f70", "f50", "budget2", "budget5", "budget20", "p0.25", "p1.0")
T, DELTA = 0.9, 0.05
EPS = 1e-12
CHUNK = 1024

#-------------------------------------------------------------------------------


def splits(n_q, seed):
    """Disjoint query splits; the evaluation splits (fit / cal / test) are
    the same for every state, only cand / score change with the seed."""
    perm = np.random.default_rng(0).permutation(n_q)
    out, s = {}, 0
    for name in ("fit", "cal", "test"):
        out[name] = perm[s:s + SPLITS[name]]
        s += SPLITS[name]
    rest = np.random.default_rng(100 + seed).permutation(perm[s:])
    out["cand"], out["score"] = rest[:SPLITS["cand"]], rest[SPLITS["cand"]:][:SPLITS["score"]]
    return out


def raw_features(cache, qidx, k):
    """Per query: offsets / sizes / shell-to-ball radius ratio for the 3
    best-fitting balls, raw intruder count, retention."""
    from simulations.approximate_hits.online_sim import sq_dist
    cache.prepare()
    D = cache.D
    Q, top = D["q"], D["top"]
    shell_r = np.sqrt(((cache.cq - D["db"][top[cache.centers, cache.R - 1]]) ** 2).sum(axis=1))
    n = len(qidx)
    off, size, srel = (np.full((n, 3), np.nan) for _ in range(3))
    count = np.zeros(n)
    for s in range(0, n, CHUNK):
        q = Q[qidx[s:s + CHUNK]]
        m = len(q)
        d2 = sq_dist(q, cache.cv, cache.cv_sq)
        part = np.argpartition(d2, k - 1, axis=1)[:, :k]
        nr = np.sqrt(np.take_along_axis(d2, part, axis=1).max(axis=1)) + EPS
        de = np.sqrt(sq_dist(q, cache.cq, cache.cq_sq))
        fit = (nr[:, None] + de) / cache.radius[None, :]
        cand = np.argpartition(fit, 2, axis=1)[:, :3]
        order = np.take_along_axis(cand, np.argsort(np.take_along_axis(fit, cand, axis=1),
                                                    axis=1), axis=1)
        rows = np.arange(m)
        for j in range(3):
            e = order[:, j]
            off[s:s + m, j] = de[rows, e] / cache.radius[e]
            size[s:s + m, j] = nr / cache.radius[e]
            srel[s:s + m, j] = shell_r[e] / cache.radius[e]
        if len(cache.sv):
            count[s:s + m] = (sq_dist(q, cache.sv, cache.sv_sq) < (nr ** 2)[:, None]).sum(axis=1)
    y = cache.cached[top[qidx, :k]].mean(axis=1)
    return off, size, srel, count, y


def compute_state(name, out):
    from simulations.approximate_hits import online_sim as osim
    cfg = dict(BASE, **STATES[name])
    meta = {"kind": "esci"}
    db = np.load(osim.ESCI / "esci_db1000000_s42.npy").astype(np.float32)
    q = np.load(osim.ESCI / "esci_queries.npy").astype(np.float32)
    top = np.load(PREP / "top500.npy")
    D = {"db": db, "q": q, "top": top, "draw": np.random.default_rng(1).random(len(db))}
    sp = splits(len(q), cfg["seed"])
    M = cfg["budget"] * len(db)
    cache = osim.Cache(D, 10, cfg["R"], cfg["bits"], M)
    if cfg["Ks"] is not None:
        osim.K_VALUES = tuple(cfg["Ks"])
    entries, _ = osim.greedy_entries(D, [(int(c), 500) for c in sp["cand"]], sp["score"],
                                     np.ones(len(sp["score"])), cfg["f"] * M, 10, T)
    cache.replace(entries)
    cache.set_p_to_fill()
    if cfg["p"] is not None:
        cache.p = cfg["p"]
    logger.info(f"{name}: {len(entries)} entries, {cache.n_cached()} cached, p {cache.p:.3f}, "
                f"memory {cache.memory():.0f} / {M:.0f}")
    res = {"p": cache.p, "R": cfg["R"], "bits": cfg["bits"], "n_cached": cache.n_cached(),
           "n_entries": len(entries), "memory": cache.memory(), "budget": M, "meta": meta}
    arrays = {}
    for k in KS:
        for s in ("fit", "cal", "test"):
            for key, v in zip(("off", "size", "srel", "count", "y"),
                              raw_features(cache, sp[s], k)):
                arrays[f"{key}_{s}_k{k}"] = v
    np.savez(out / f"{name}.npz", **arrays)
    (out / f"{name}.json").write_text(json.dumps(res))
    logger.info(f"wrote {out / name}.npz")


#-------------------------------------------------------------------------------


VARIANTS = ("base7", "est7", "ratio7", "cov8")


def design(z, meta, s, k, variant):
    off, size, srel = z[f"off_{s}_k{k}"], z[f"size_{s}_k{k}"], z[f"srel_{s}_k{k}"]
    cnt = z[f"count_{s}_k{k}"]
    off, size = np.nan_to_num(off, nan=10.0), np.nan_to_num(size, nan=10.0)
    cols = [off[:, j // 2] if j % 2 == 0 else size[:, j // 2] for j in range(6)]
    est = cnt / max(meta["p"], EPS)
    if variant == "base7":
        cols.append(np.log1p(cnt))
    elif variant == "est7":
        cols.append(np.log1p(est))
    else:
        cols.append(k / (k + est))
        if variant == "cov8":
            cols.append(np.minimum((off[:, 0] + size[:, 0]) / srel[:, 0], 10.0))
    return np.column_stack(cols)


def evaluate(model, z, meta, k, variant):
    """Hit rate and retention on test at lambda recalibrated on this state's
    cal split; and the scores (for frozen-lambda evaluation)."""
    from simulations.approximate_hits.threshold_selection import select, threshold_grid
    sc = model(design(z, meta, "cal", k, variant))
    yc = z[f"y_cal_k{k}"]
    lam = select("normal_fixed", sc, yc, threshold_grid(sc), T, DELTA)
    st = model(design(z, meta, "test", k, variant))
    yt = z[f"y_test_k{k}"]
    adm = st >= lam if lam is not None else np.zeros(len(st), bool)
    return {"hit": float(adm.mean()), "ret": float(yt[adm].mean()) if adm.any() else None,
            "lam": lam}, st, yt


def analyze(out):
    from simulations.approximate_hits.eval_generalize import fit_logistic, optimal_rate
    data = {n: (np.load(out / f"{n}.npz"), json.loads((out / f"{n}.json").read_text()))
            for n in STATES if (out / f"{n}.npz").exists()}
    res = {}
    for v in VARIANTS:
        zb, mb = data["base"]
        src = fit_logistic(design(zb, mb, "fit", 10, v), zb["y_fit_k10"])
        base_lam = evaluate(src, zb, mb, 10, v)[0]["lam"]
        Xp = np.vstack([design(*data[n], "fit", 10, v) for n in POOL_TRAIN if n in data])
        yp = np.concatenate([data[n][0]["y_fit_k10"] for n in POOL_TRAIN if n in data])
        pooled = fit_logistic(Xp, yp)
        rows = {}
        for n, (z, m) in data.items():
            for k in KS:
                nat = fit_logistic(design(z, m, "fit", k, v), z[f"y_fit_k{k}"])
                rn = evaluate(nat, z, m, k, v)[0]
                rt, st, yt = evaluate(src, z, m, k, v)
                rp = evaluate(pooled, z, m, k, v)[0]
                adm = st >= base_lam if base_lam is not None else np.zeros(len(st), bool)
                rows[f"{n}/k{k}"] = {
                    "p": m["p"], "oracle": float(optimal_rate(yt, T)),
                    "native": rn, "transfer": rt, "pooled": rp,
                    "frozen": {"hit": float(adm.mean()),
                               "ret": float(yt[adm].mean()) if adm.any() else None}}
        res[v] = rows
    (out / "invariance.json").write_text(json.dumps(res, indent=1))
    f = lambda x: "  -- " if x is None else f"{x:.3f}"
    for v in VARIANTS:
        print(f"\n== {v}: hit at recalibrated lambda (native / base model / pooled), "
              f"oracle; frozen base lambda: hit, retention")
        for key, r in res[v].items():
            pooled_tag = "*" if key.split("/")[0] in POOL_TRAIN and key.endswith("k10") else " "
            print(f"{key:14s} p {r['p']:.2f}  {f(r['native']['hit'])} {f(r['transfer']['hit'])} "
                  f"{f(r['pooled']['hit'])}{pooled_tag} orc {f(r['oracle'])} | frozen "
                  f"{f(r['frozen']['hit'])} {f(r['frozen']['ret'])}")


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--state", choices=list(STATES))
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--out", type=str, default="simulations/approximate_hits/results/invariance")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.state:
        compute_state(args.state, out)
    if args.analyze:
        analyze(out)


if __name__ == "__main__":
    main()
