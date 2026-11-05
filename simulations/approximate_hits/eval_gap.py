#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Where does the gap between our guaranteed hit rate and the oracle come from?

Per cache state (greedy cache for --target), retention GBT (eval_predictors
features) and variants with extra inputs:
  base       eval_predictors features (= gbt)
  depth_hist + track record of past (FIT) queries with the same best entry AND
             similar depth in it (4 depth bins, smoothed)  [legitimate]
  profile    + stored distance profiles of the N_NEAR nearest cached entries:
             per entry log(offset / c_k) and log(stored d_j / c_k) for ranks
             j = 10, 20, 50 (missing if the entry's K < j)  [legitimate]
  sample_p   + probe of a uniform random sample of a fraction p of the
             database (kept alongside the cache, --sample-fracs): log(s_j /
             c_k) for the query's j = 1..5 nearest sample vectors, and counts
             of UNCACHED sample vectors within c_k (seen intruders), 1.1 c_k,
             and of all sample vectors within 1.25 c_k  [legitimate; the
             sample's memory is extra, reported as sample_frac]
  shell_p    + same probe features, but from a TARGETED sample: the vectors
             ranked K+1..500 around each cached entry (known from the backend
             answer when the entry was inserted; K = the entry's cached K),
             each kept iff a fixed per-vector uniform draw is < p (so every
             sampled vector had inclusion probability p, outside the shells 0)
             [legitimate; sample size reported as shell_frac_p]
  density    + true database distance profile: log(d_j / c_k) for the query's
             j = 20th and 50th nearest database vectors, c_k = k-th nearest
             cached distance  [cheat: needs the database]
  ratio_s    + true radius / c_k with multiplicative noise exp(N(0, s)), for
             s in --noise  [cheat: radius known to within ~s]
For each: calibrated test hit rate (normal_fixed lambda on CAL) and the
best-cutoff hit rate (largest test set admitted by a cutoff on the score with
realized test retention >= T; cheats on the cutoff only). The gap splits into
  oracle - best_cutoff(base)        ordering mistakes (missing information)
  best_cutoff(base) - calibrated    safety margin of calibration

Also a diagnostic of how well the nearest entry's stored 20th / 50th
neighbor distance estimates the query's true one (entries with K >= 50).

Database top-50 distances are computed once and saved to <run>/profile50.npz.
Writes <run>/gap.json. Requires scikit-learn (.venv python).
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
from simulations.approximate_hits.eval_aker_thresholds import fetch
from simulations.approximate_hits.eval_certificates import pairwise_dist, _sq_norms
from simulations.approximate_hits.eval_radius_scores import cached_knn, true_radius
from simulations.approximate_hits.run_e2e_oracle_sift import exact_topk_blocked
from simulations.approximate_hits.threshold_selection import (
    DEFAULT_METHOD, select, threshold_grid,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

EPS = 1e-12
CHUNK = 2048
PROFILE_K = 50
PROFILE_RANKS = (10, 20, 50)
N_NEAR = 3
N_DEPTH_BINS = 4
SAMPLE_J = 5
_R: dict = {}

#-------------------------------------------------------------------------------


def _expand(split):
    return np.repeat(np.arange(len(split["y"])), split["w"].astype(np.int64))


def db_profile(run_dir):
    """Distances to the 20th and 50th nearest database vectors per split."""
    path = run_dir / "profile50.npz"
    if path.exists():
        z = np.load(path)
        return {sp: z[sp] for sp in epred.SPLITS}
    db = ecal._G["db"]
    out = {}
    for sp in epred.SPLITS:
        q = ecal._G["sets"][sp][0]
        logger.info(f"Exact top-{PROFILE_K} for {sp} ({len(q)} queries)...")
        ids = exact_topk_blocked(q, db, PROFILE_K)
        d = np.empty((len(q), 2))
        for s in range(0, len(q), CHUNK):
            sel = ids[s:s + CHUNK][:, [19, PROFILE_K - 1]]
            rows = fetch(db, sel.ravel()).reshape(len(sel), 2, -1)
            diff = np.asarray(q[s:s + CHUNK], dtype=np.float64)[:, None, :] - rows
            d[s:s + CHUNK] = np.sqrt((diff ** 2).sum(-1))
        out[sp] = np.sort(d, axis=1)
    np.savez(path, **out)
    return out


def center_profiles():
    """Per candidate query: distances to its 50 nearest database vectors."""
    db, cand_q = ecal._G["db"], ecal._G["cand_q"]
    n_kv = len(ecal._G["k_values"])
    out = np.empty((len(cand_q), PROFILE_K))
    for s in range(0, len(cand_q), CHUNK):
        e = min(s + CHUNK, len(cand_q))
        ids = np.stack([np.asarray(ecal._G["cached_ids"][i * n_kv + n_kv - 1][:PROFILE_K])
                        for i in range(s, e)])
        rows = fetch(db, ids.ravel()).reshape(e - s, PROFILE_K, -1)
        diff = np.asarray(cand_q[s:e], dtype=np.float64)[:, None, :] - rows
        out[s:e] = np.sort(np.sqrt((diff ** 2).sum(-1)), axis=1)
    return out


def entry_kmax(entries):
    """Cached candidate indices and the largest K cached for each."""
    n_kv = len(ecal._G["k_values"])
    kmax = {}
    for j in entries:
        i, K = j // n_kv, ecal._G["k_values"][j % n_kv]
        kmax[i] = max(kmax.get(i, 0), K)
    cand = np.array(sorted(kmax), dtype=np.int64)
    return cand, np.array([kmax[i] for i in cand], dtype=np.int64)


def shell_sample(entries, in_cache, frac):
    """Targeted sample: uncached vectors in the stored rank K+1..500 shells of
    cached entries, kept iff their fixed uniform draw is < frac."""
    n_kv = len(ecal._G["k_values"])
    cand, K = entry_kmax(entries)
    shells = [np.asarray(ecal._G["cached_ids"][i * n_kv + n_kv - 1][k:]) for i, k in zip(cand, K)]
    ids = np.unique(np.concatenate(shells)) if shells else np.empty(0, np.int64)
    ids = ids[~in_cache[ids]]
    ids = ids[_R["draw"][ids] < frac]
    if len(ids) < SAMPLE_J:
        return None
    vecs = fetch(ecal._G["db"], ids)
    return {"ids": ids, "vecs": vecs, "sq": _sq_norms(vecs)}


def profile_features(q, entries, ck):
    """Stored-profile features of the N_NEAR nearest cached entry centers.

    Returns (features, [nearest center's candidate index, its K, offset])."""
    cand, K = entry_kmax(entries)
    cq = np.asarray(ecal._G["cand_q"][cand], dtype=np.float64)
    prof = _R["center_prof"][cand]
    F = np.full((len(q), N_NEAR * (1 + len(PROFILE_RANKS))), np.nan)
    first = np.empty((len(q), 3))
    for s in range(0, len(q), CHUNK):
        d = pairwise_dist(q[s:s + CHUNK], cq, _sq_norms(cq))
        m = min(N_NEAR, d.shape[1])
        part = (np.argpartition(d, m - 1, axis=1)[:, :m] if m < d.shape[1]
                else np.tile(np.arange(m), (len(d), 1)))
        nn = np.take_along_axis(part, np.argsort(np.take_along_axis(d, part, axis=1), axis=1),
                                axis=1)
        off = np.take_along_axis(d, nn, axis=1)
        c = ck[s:s + CHUNK]
        for t in range(m):
            col = t * (1 + len(PROFILE_RANKS))
            F[s:s + CHUNK, col] = np.log((off[:, t] + EPS) / c)
            for u, rank in enumerate(PROFILE_RANKS):
                ok = K[nn[:, t]] >= rank
                v = np.log(prof[nn[:, t], rank - 1] / c)
                F[s:s + CHUNK, col + 1 + u] = np.where(ok, v, np.nan)
        first[s:s + CHUNK] = np.column_stack([cand[nn[:, 0]], K[nn[:, 0]], off[:, 0]])
    return F, first


def draw_sample(frac, seed=0):
    """Uniform random sample of database ids and their vectors (float64)."""
    db = ecal._G["db"]
    n = max(SAMPLE_J, int(round(frac * len(db))))
    ids = np.sort(np.random.default_rng(seed).choice(len(db), n, replace=False))
    vecs = fetch(db, ids)
    return {"ids": ids, "vecs": vecs, "sq": _sq_norms(vecs)}


def sample_features(q, ck, in_cache, samp):
    """Probe features from the random sample; also whether an intruder was seen."""
    unc = ~in_cache[samp["ids"]]
    F = np.empty((len(q), SAMPLE_J + 3))
    for s in range(0, len(q), 1024):
        d = pairwise_dist(q[s:s + 1024], samp["vecs"], samp["sq"])
        c = ck[s:s + 1024][:, None]
        near = np.sort(np.partition(d, SAMPLE_J - 1, axis=1)[:, :SAMPLE_J], axis=1)
        F[s:s + 1024, :SAMPLE_J] = np.log((near + EPS) / c)
        F[s:s + 1024, SAMPLE_J] = np.log1p(((d < c) & unc).sum(axis=1))
        F[s:s + 1024, SAMPLE_J + 1] = np.log1p(((d < 1.1 * c) & unc).sum(axis=1))
        F[s:s + 1024, SAMPLE_J + 2] = np.log1p((d < 1.25 * c).sum(axis=1))
    return F


def profile_diagnostic(first, true_d, ck):
    """How well the nearest entry's stored 20th / 50th distance estimates the
    query's true one, over test queries whose nearest entry has K >= 50."""
    cand_idx = first[:, 0].astype(np.int64)
    ok = first[:, 1] >= PROFILE_K
    out = {"frac_with_k50": float(ok.mean())}
    if ok.sum() < 10:
        return out
    off = first[ok, 2]
    for col, rank in ((0, 20), (1, PROFILE_K)):
        est = _R["center_prof"][cand_idx[ok], rank - 1]
        tru = true_d[ok, col]
        target = np.log(tru / ck[ok])
        err = np.log(est / tru)
        rk = lambda a: np.argsort(np.argsort(a))
        out[f"d{rank}"] = {
            "median_abs_log_err": float(np.median(np.abs(err))),
            "median_offset_over_true": float(np.median(off / tru)),
            "spread_of_target": float(np.std(target)),
            "spearman_est_vs_true": float(np.corrcoef(rk(np.log(est / ck[ok])), rk(target))[0, 1]),
        }
    return out


def depth_history(state, jstar, n_entries):
    """Smoothed track record per (best entry, depth bin); FIT out-of-fold."""
    depth = {sp: state[sp]["base"][:, 5] for sp in epred.SPLITS}  # log dist / radius
    edges = np.quantile(depth["fit"], np.linspace(0, 1, N_DEPTH_BINS + 1)[1:-1])
    key = {sp: jstar[sp] * N_DEPTH_BINS + np.searchsorted(edges, depth[sp])
           for sp in epred.SPLITS}
    fit = state["fit"]
    (oof_m, oof_c), full_m, full_c = epred.entry_history(
        key["fit"], fit["y"], fit["w"], n_entries * N_DEPTH_BINS)
    out = {}
    for sp in epred.SPLITS:
        if sp == "fit":
            out[sp] = np.column_stack([oof_m, np.log1p(oof_c)])
        else:
            out[sp] = np.column_stack([full_m[key[sp]], np.log1p(full_c[key[sp]])])
    return out


def best_cutoff_rate(s, y, target):
    """Largest fraction admitted by a cutoff on s with mean y >= target."""
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    mean = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    # only cut between distinct scores
    ok = (mean >= target) & np.append(ss[1:] < ss[:-1], True)
    return float((np.flatnonzero(ok).max() + 1) / len(ys)) if ok.any() else 0.0


def eval_state(task):
    target, point, entries = task
    args = _R["args"]
    state = epred.build_state(task)
    if state is None:
        return None
    st = ecal.state_tables(entries)
    k = ecal._G["k"]
    jstar, ck = {}, {}
    for sp in epred.SPLITS:
        q = ecal._G["sets"][sp][0]
        jstar[sp] = epred.full_features(q, st, point["cache_frac"])[2]
        ck[sp] = np.maximum(cached_knn(q, st)[:, k - 1], EPS)
    X0 = {sp: np.hstack([state[sp]["base"], state[sp]["ext"]]) for sp in epred.SPLITS}
    extra = {"base": {sp: np.empty((len(X0[sp]), 0)) for sp in epred.SPLITS},
             "depth_hist": depth_history(state, jstar, len(entries)),
             "density": {sp: np.log(_R["profile"][sp] / ck[sp][:, None]) for sp in epred.SPLITS}}
    pf, first = {}, None
    for sp in epred.SPLITS:
        pf[sp], f = profile_features(ecal._G["sets"][sp][0], entries, ck[sp])
        if sp == "test":
            first = f
    extra["profile"] = pf
    diag = {}
    samples = {f"sample_{f:g}": v for f, v in _R["samples"].items()}
    for f in args.shell_fracs:
        samp = shell_sample(entries, st["in_cache"], f)
        if samp is not None:
            samples[f"shell_{f:g}"] = samp
    for name, samp in samples.items():
        extra[name] = {sp: sample_features(ecal._G["sets"][sp][0], ck[sp], st["in_cache"], samp)
                       for sp in epred.SPLITS}
        seen = extra[name]["test"][:, SAMPLE_J] > 0
        y = state["test"]["y"]
        diag[name] = {"seen_given_ret_lt1": float(seen[y < 1].mean()) if (y < 1).any() else None,
                      "seen_given_ret_lt05": float(seen[y < 0.5].mean()) if (y < 0.5).any() else None,
                      "seen_given_ret1": float(seen[y == 1].mean()) if (y == 1).any() else None,
                      "sample_frac": len(samp["ids"]) / len(ecal._G["db"])}
    rng = np.random.default_rng(0)
    for s in args.noise:
        extra[f"ratio_{s:g}"] = {
            sp: (np.log(_R["radius"][sp] / ck[sp]) + s * rng.standard_normal(len(ck[sp])))[:, None]
            for sp in epred.SPLITS}

    extra = {n: v for n, v in extra.items() if n in args.variants or n == "base"}
    out = {"cache_frac": point["cache_frac"], "oracle_hit_rate": point["test_hit_rate"],
           "profile_diagnostic": profile_diagnostic(first, _R["profile"]["test"], ck["test"]),
           "sample_diagnostic": diag}
    test, cal, fit = state["test"], state["cal"], state["fit"]
    ti, ci = _expand(test), _expand(cal)
    for name, ex in extra.items():
        X = {sp: np.hstack([X0[sp], ex[sp]]) for sp in epred.SPLITS}
        g = epred._gbt("reg").fit(X["fit"], fit["y"], sample_weight=fit["w"])
        s = {sp: g.predict(X[sp]) for sp in epred.SPLITS}
        lam = select(DEFAULT_METHOD, s["cal"][ci], cal["y"][ci], threshold_grid(s["fit"]),
                     target, 0.05)
        st_, yt = s["test"][ti], test["y"][ti]
        adm = st_ >= lam if lam is not None else np.zeros(len(st_), dtype=bool)
        out[name] = {"calibrated_hit_rate": float(adm.mean()),
                     "calibrated_hit_retention": float(yt[adm].mean()) if adm.any() else None,
                     "best_cutoff_hit_rate": best_cutoff_rate(st_, yt, target)}
    # oracle ordering with the same request weighting (sanity: ~ oracle_hit_rate)
    out["perfect_score_hit_rate"] = best_cutoff_rate(test["y"][ti], test["y"][ti], target)
    return out


#-------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--sizes", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.3])
    ap.add_argument("--noise", type=float, nargs="+", default=[0.002, 0.005, 0.01, 0.02])
    ap.add_argument("--variants", nargs="+",
                    default=["depth_hist", "profile", "density", "ratio_0.002",
                             "ratio_0.005", "ratio_0.01", "ratio_0.02"],
                    help="variants besides base to evaluate")
    ap.add_argument("--sample-fracs", type=float, nargs="+", default=[0.01])
    ap.add_argument("--shell-fracs", type=float, nargs="+", default=[])
    ap.add_argument("--n-each", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-name", type=str, default="gap.json")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    data, _ = epred.setup_run(run_dir, args.n_each)
    _R["args"] = args
    logger.info("Stored distance profiles of candidate entries...")
    _R["center_prof"] = center_profiles()
    _R["samples"] = {f: draw_sample(f) for f in args.sample_fracs}
    _R["draw"] = np.random.default_rng(1).random(len(ecal._G["db"]))
    _R["radius"] = {sp: true_radius(ecal._G["sets"][sp][0], ecal._G["topk"][sp])
                    for sp in epred.SPLITS}
    _R["profile"] = db_profile(run_dir)

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
            names = [n for n in s if isinstance(s[n], dict) and not n.endswith("_diagnostic")]
            logger.info(f"cache {100 * s['cache_frac']:.2f}%: oracle {s['oracle_hit_rate']:.3f}"
                        f" perfect-score {s['perfect_score_hit_rate']:.3f} | "
                        + " ".join(f"{n} cal {s[n]['calibrated_hit_rate']:.3f}"
                                   f" best {s[n]['best_cutoff_hit_rate']:.3f};" for n in names))
            logger.info(f"  profile diagnostic: {json.dumps(s['profile_diagnostic'])}")
            logger.info(f"  sample diagnostic: {json.dumps(s['sample_diagnostic'])}")
            (run_dir / args.out_name).write_text(
                json.dumps({"args": vars(args), "states": out}, indent=1))


if __name__ == "__main__":
    main()
