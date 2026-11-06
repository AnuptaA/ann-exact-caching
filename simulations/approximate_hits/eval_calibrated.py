#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Calibrated admission (B5) on saved greedy caches.

For each cache state (checkpoint of a saved greedy run) and target T:
  1. Features: cheap per-query quantities computed from the cache only (no
     database access): distances to cached vectors, depth inside the nearest
     cache ball, certificate quantities, etc.
  2. Predictor: ridge regression (features + squares) -> retention score,
     fit on FIT queries (fresh draws from the workload).
  3. Threshold: admit a query iff score >= lambda. lambda is chosen on
     CALIBRATION queries (fresh, independent draws) so that, with probability
     ~>= 1 - delta, the true mean retention over admitted queries is >= T.
     Default procedure (threshold_selection.py): normal-approximation lower
     bound per threshold, fixed-order testing over a grid of FIT-score
     quantiles from strict to lenient (stop at the first failure).

Fit and calibration queries never overlap with the scoring queries greedy used
to choose the cache, so they are not biased by the cache selection.
Also reported: "naive" lambda (empirical calibration mean >= T, no confidence
bound), and the realized mean retention over hits on the test queries.
"""

import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from simulations.approximate_hits.eval_certificates import (
    build_entry_tables, pairwise_dist, _sq_norms,
)
from simulations.approximate_hits.run_e2e_oracle_sift import exact_topk, load_inputs
from simulations.approximate_hits.threshold_selection import (
    DEFAULT_METHOD, METHODS, select, threshold_grid,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CHUNK = 2048
EPS = 1e-12
_G: dict = {}

FEATURE_NAMES = [
    "log_D1", "log_Dk", "log_Dk_over_D1", "log_gap_k", "log_min_d_entry",
    "log_min_d_over_r", "log_overshoot", "a1_frac", "frac_topk_from_entry",
    "log1p_inside", "log_Dk_over_r_entry", "log_r_entry",
]

#-------------------------------------------------------------------------------


def load_all(run_dir: Path, n_fit: int, n_calib: int):
    """Rebuild a run's inputs plus fresh fit and calibration queries."""
    data = json.loads((run_dir / "results.json").read_text())
    cfg = argparse.Namespace(**data["config"])
    if "zipf" in data["config"]:
        from simulations.approximate_hits.run_e2e_oracle_synthetic import (
            build_inputs, _dedup,
        )
        inp = build_inputs(cfg)
        wl = inp["wl"]
        rng = np.random.default_rng(cfg.seed + 2)  # independent of all run draws
        fit_ids, fit_w, _ = _dedup(wl.sample(n_fit, rng))
        cal_ids, cal_w, _ = _dedup(wl.sample(n_calib, rng))
        sets = {
            "fit": (wl.vectors(fit_ids), fit_w),
            "cal": (wl.vectors(cal_ids), cal_w),
            "test": (wl.vectors(inp["test_ids"]), inp["test_w"]),
        }
        db = inp["db"]
        cand_q = wl.vectors(inp["cand_ids"])
    else:
        db, cand_q, _, test_q = load_inputs(cfg)
        pool = np.load(Path(cfg.esci_dir) / "esci_queries.npy")
        perm = np.random.default_rng(cfg.seed).permutation(len(pool))
        start = cfg.n_test + cfg.n_cand_queries + cfg.n_score_queries
        fit_q = pool[perm[start:start + n_fit]]
        cal_q = pool[perm[start + n_fit:start + n_fit + n_calib]]
        sets = {
            "fit": (fit_q, np.ones(len(fit_q), dtype=np.int64)),
            "cal": (cal_q, np.ones(len(cal_q), dtype=np.int64)),
            "test": (test_q, np.ones(len(test_q), dtype=np.int64)),
        }
    return data, cfg, db, cand_q, sets


#-------------------------------------------------------------------------------


def state_tables(entries):
    db = _G["db"]
    n_db = len(db)
    in_cache = np.zeros(n_db, dtype=bool)
    keys = []
    for e, j in enumerate(entries):
        ids = np.asarray(_G["cached_ids"][j], dtype=np.int64)
        in_cache[ids] = True
        keys.append(e * n_db + ids)
    cached = np.flatnonzero(in_cache)
    ent = np.asarray(entries, dtype=np.int64)
    ent_q = _G["cand_q"][ent // len(_G["k_values"])]
    db_c = np.asarray(db[cached], dtype=np.float64)
    return {
        "in_cache": in_cache, "cached": cached, "n_db": n_db,
        # sorted (entry, vector) membership keys: entry e holds vector v iff
        # e * n_db + v is present (avoids an entries x database table)
        "member_keys": np.unique(np.concatenate(keys)) if keys else np.empty(0, np.int64),
        "db_c": db_c, "db_c_sq": _sq_norms(db_c),
        "ent_q": ent_q, "ent_q_sq": _sq_norms(np.asarray(ent_q, dtype=np.float64)),
        "ent_r": _G["radius"][ent],
    }


def is_member(st, entry_idx, vector_ids):
    """Elementwise: does entry entry_idx (index into the state's entries) hold
    vector vector_ids? Arguments broadcast against each other."""
    q = (np.asarray(entry_idx, dtype=np.int64) * st["n_db"]
         + np.asarray(vector_ids, dtype=np.int64))
    keys = st["member_keys"]
    if len(keys) == 0:
        return np.zeros(q.shape, dtype=bool)
    pos = np.minimum(np.searchsorted(keys, q), len(keys) - 1)
    return keys[pos] == q


def features(qvecs, st):
    k = _G["k"]
    n = len(qvecs)
    F = np.empty((n, len(FEATURE_NAMES)))
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        q = qvecs[s:e]
        d_qc = pairwise_dist(q, st["db_c"], st["db_c_sq"])
        part = np.argpartition(d_qc, k, axis=1)[:, :k + 1]
        dpart = np.take_along_axis(d_qc, part, axis=1)
        order = np.argsort(dpart, axis=1)
        part = np.take_along_axis(part, order, axis=1)
        dpart = np.take_along_axis(dpart, order, axis=1)
        d1, dk, dk1 = dpart[:, 0], dpart[:, k - 1], dpart[:, k]
        topk_ids = st["cached"][part[:, :k]]

        d_qe = pairwise_dist(q, st["ent_q"], st["ent_q_sq"])
        r = st["ent_r"][None, :]
        norm = d_qe / np.maximum(r, EPS)
        jstar = norm.argmin(axis=1)
        overshoot = ((dk[:, None] + d_qe) / np.maximum(r, EPS)).min(axis=1)
        rho = (r - d_qe).max(axis=1)
        a1 = np.minimum((d_qc <= rho[:, None]).sum(axis=1), k) / k
        inside = (d_qe < r).sum(axis=1)
        frac_from = is_member(st, jstar[:, None], topk_ids).mean(axis=1)
        r_js = st["ent_r"][jstar]

        F[s:e] = np.column_stack([
            np.log(d1 + EPS), np.log(dk + EPS), np.log((dk + EPS) / (d1 + EPS)),
            np.log((dk1 - dk) / (dk + EPS) + 1e-6), np.log(d_qe.min(axis=1) + EPS),
            np.log(norm.min(axis=1) + EPS), np.log(overshoot + EPS), a1, frac_from,
            np.log1p(inside), np.log((dk + EPS) / (r_js + EPS)), np.log(r_js + EPS),
        ])
    return F


def fit_ridge(F, y, w, alpha=1.0):
    mu = np.average(F, axis=0, weights=w)
    sd = np.sqrt(np.average((F - mu) ** 2, axis=0, weights=w)) + 1e-9

    def design(Fx):
        Z = (Fx - mu) / sd
        return np.column_stack([np.ones(len(Z)), Z, Z ** 2])

    X = design(F)
    reg = alpha * np.eye(X.shape[1])
    reg[0, 0] = 0.0
    beta = np.linalg.solve(X.T @ (X * w[:, None]) + reg, X.T @ (w * y))
    return lambda Fx: design(Fx) @ beta


def _evaluate(s, y, w, lam):
    total = float(w.sum())
    if lam is None:
        return {"hit_rate": 0.0, "hit_retention": None, "effective_retention": 1.0}
    adm = s >= lam
    wa = float(w[adm].sum())
    kept = float((w * y)[adm].sum())
    return {
        "hit_rate": wa / total,
        "hit_retention": kept / wa if wa > 0 else None,
        "effective_retention": 1.0 - wa / total + kept / total,
    }


def _spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def eval_target(res):
    target = res["target"]
    rows = []
    for p in res["points"]:
        entries = res["selected"][:p["n_entries"]]
        row = {"cache_frac": p["cache_frac"], "n_entries": p["n_entries"],
               "oracle_hit_rate_online": p["test_hit_rate_online"],
               "oracle_hit_rate": p["test_hit_rate"]}
        if len(entries) == 0:
            rows.append(row)
            continue
        st = state_tables(entries)
        if len(st["cached"]) <= _G["k"]:
            rows.append(row)
            continue
        Fy = {}
        for name in ("fit", "cal", "test"):
            q, w = _G["sets"][name]
            y = st["in_cache"][_G["topk"][name]].mean(axis=1)
            Fy[name] = (features(q, st), y, w.astype(np.float64))
        predict = fit_ridge(*Fy["fit"])
        s_fit = predict(Fy["fit"][0])
        grid_desc = threshold_grid(s_fit)
        F_cal, y_cal, w_cal = Fy["cal"]
        rep = np.repeat(np.arange(len(y_cal)), w_cal.astype(np.int64))
        s_cal = predict(F_cal)[rep]
        y_cal = y_cal[rep]
        lam_g = select(_G["method"], s_cal, y_cal, grid_desc, target, _G["delta"])
        lam_n = select("naive", s_cal, y_cal, grid_desc, target, _G["delta"])
        s_test = predict(Fy["test"][0])
        y_test, w_test = Fy["test"][1], Fy["test"][2]
        for tag, lam in (("guaranteed", lam_g), ("naive", lam_n)):
            for key, val in _evaluate(s_test, y_test, w_test, lam).items():
                row[f"{tag}_{key}"] = val
        row["test_spearman"] = _spearman(s_test, y_test)
        rows.append(row)
    return {"target": target, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--n-fit", type=int, default=20000)
    ap.add_argument("--n-calib", type=int, default=20000)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--method", choices=METHODS, default=DEFAULT_METHOD,
                    help="threshold-selection procedure for the guaranteed threshold")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--targets", type=float, nargs="+", default=None,
                    help="only evaluate these targets (default: all in the run)")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    run_cfg = json.loads((run_dir / "results.json").read_text())["config"]
    cand_top = pre_topk = None
    if "trace" in run_cfg:
        # trace runs: fixed time-ordered splits and saved exact neighbors
        from simulations.approximate_hits.run_e2e_oracle_trace import load_trace_run
        t = load_trace_run(run_dir)
        data, cfg, db, cand_q = t["data"], t["cfg"], t["db"], t["cand_q"]
        sets = {s: t["sets"][s] for s in ("fit", "cal", "test")}
        pre_topk = {s: t["topk"][s] for s in sets}
        cand_top = t["cand_top"]
    else:
        data, cfg, db, cand_q, sets = load_all(run_dir, args.n_fit, args.n_calib)
    if args.targets is not None:
        data["results"] = [r for r in data["results"]
                           if any(abs(r["target"] - t) < 1e-9 for t in args.targets)]
    logger.info(f"{run_dir.name}: " + ", ".join(
        f"{k}={len(v[0])} distinct ({int(v[1].sum())} total)" for k, v in sets.items()))
    cached_ids, radius, _, _ = build_entry_tables(cand_q, db, cfg.k_values, 1,
                                                  cached_top=cand_top)
    topk = pre_topk if pre_topk is not None else {
        name: exact_topk(q, db, cfg.k) for name, (q, _) in sets.items()}
    _G.update(k=cfg.k, k_values=cfg.k_values, db=db, cand_q=cand_q, sets=sets,
              topk=topk, cached_ids=cached_ids, radius=radius, delta=args.delta,
              method=args.method)

    logger.info(f"Evaluating {len(data['results'])} targets...")
    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(args.workers, len(data["results"]))) as pool:
            out = pool.map(eval_target, data["results"])
    else:
        out = [eval_target(r) for r in data["results"]]
    path = run_dir / "calibrated.json"
    path.write_text(json.dumps({"delta": args.delta, "method": args.method,
                                "n_fit": args.n_fit,
                                "n_calib": args.n_calib, "features": FEATURE_NAMES,
                                "results": out}))
    logger.info(f"Wrote {path}")


if __name__ == "__main__":
    main()
