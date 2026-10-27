#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Compare retention predictors for calibrated admission (B5).

Query splits (all fresh, disjoint from cache selection):
  fit   train the predictor (and per-entry history features)
  dev   model selection: guaranteed hit rate on dev
  cal   calibration: choose lambda (normal approximation + fixed order)
  test  final report only

Features: the 12 base features of eval_calibrated.py plus
  log(D_r / D_k) for r = ceil(k/4), ceil(k/2), ceil(3k/4)
  fraction of returned vectors lying in a ball that contains q
  log(d(q, nearest entry) / D_k)
  per-entry history: smoothed mean retention of FIT queries whose nearest
    entry is the same (out-of-fold on the fit set), and log1p of their count
  log cache size (used by the pooled model)

Models: ridge (base / extended features), gradient-boosted trees for
retention (squared loss), for 1{retention >= T} (classifier), for the 20th
percentile of retention (quantile loss), and one gradient-boosted model pooled
over all evaluated cache states of the run.

Requires scikit-learn (run with the repo's .venv python).
"""

import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from simulations.approximate_hits import eval_calibrated as ecal
from simulations.approximate_hits.eval_certificates import build_entry_tables, pairwise_dist
from simulations.approximate_hits.run_e2e_oracle_sift import exact_topk, load_inputs
from simulations.approximate_hits.threshold_selection import select, threshold_grid

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

EPS = 1e-12
CHUNK = 2048
SPLITS = ("fit", "dev", "cal", "test")
EXT_NAMES = ["log_Dq1_over_Dk", "log_Dq2_over_Dk", "log_Dq3_over_Dk",
             "covered_frac", "log_dentry_over_Dk", "entry_hist_mean",
             "entry_hist_log_count", "log_cache_frac"]
MODELS = ["ridge_base", "ridge_ext", "gbt", "gbt_cls", "gbt_q20", "gbt_pooled"]

#-------------------------------------------------------------------------------


def load_splits(run_dir: Path, n_each: int):
    data = json.loads((run_dir / "results.json").read_text())
    cfg = argparse.Namespace(**data["config"])
    if "zipf" in data["config"]:
        from simulations.approximate_hits.run_e2e_oracle_synthetic import (
            build_inputs, _dedup,
        )
        inp = build_inputs(cfg)
        wl = inp["wl"]
        rng = np.random.default_rng(cfg.seed + 4)  # independent of all other draws
        sets = {}
        for name in ("fit", "dev", "cal"):
            ids, w, _ = _dedup(wl.sample(n_each, rng))
            sets[name] = (wl.vectors(ids), w)
        sets["test"] = (wl.vectors(inp["test_ids"]), inp["test_w"])
        return data, cfg, inp["db"], wl.vectors(inp["cand_ids"]), sets
    db, cand_q, _, test_q = load_inputs(cfg)
    pool = np.load(Path(cfg.esci_dir) / "esci_queries.npy")
    perm = np.random.default_rng(cfg.seed).permutation(len(pool))
    start = cfg.n_test + cfg.n_cand_queries + cfg.n_score_queries
    sets = {}
    for i, name in enumerate(("fit", "dev", "cal")):
        q = pool[perm[start + i * n_each:start + (i + 1) * n_each]]
        sets[name] = (q, np.ones(len(q), dtype=np.int64))
    sets["test"] = (test_q, np.ones(len(test_q), dtype=np.int64))
    return data, cfg, db, cand_q, sets


def full_features(qvecs, st, cache_frac):
    """Base features (as eval_calibrated.features), extended features, j*."""
    k = ecal._G["k"]
    n = len(qvecs)
    base = ecal.features(qvecs, st)
    ext = np.zeros((n, len(EXT_NAMES)))
    jstar_all = np.empty(n, dtype=np.int64)
    ranks = [max(1, int(np.ceil(k * f))) - 1 for f in (0.25, 0.5, 0.75)]
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        q = qvecs[s:e]
        d_qc = pairwise_dist(q, st["db_c"], st["db_c_sq"])
        part = np.argpartition(d_qc, k - 1, axis=1)[:, :k]
        dpart = np.take_along_axis(d_qc, part, axis=1)
        order = np.argsort(dpart, axis=1)
        part = np.take_along_axis(part, order, axis=1)
        dpart = np.take_along_axis(dpart, order, axis=1)
        dk = dpart[:, k - 1]
        topk_ids = st["cached"][part]

        d_qe = pairwise_dist(q, st["ent_q"], st["ent_q_sq"])
        r = st["ent_r"][None, :]
        jstar = (d_qe / np.maximum(r, EPS)).argmin(axis=1)
        inside = d_qe < r
        covered = np.zeros(topk_ids.shape, dtype=bool)
        for j in np.flatnonzero(inside.any(axis=0)):
            rows = inside[:, j]
            covered[rows] |= ecal.is_member(st, j, topk_ids[rows])
        ext[s:e, 0:3] = np.log((dpart[:, ranks] + EPS) / (dk[:, None] + EPS))
        ext[s:e, 3] = covered.mean(axis=1)
        ext[s:e, 4] = np.log((d_qe[np.arange(e - s), jstar] + EPS) / (dk + EPS))
        jstar_all[s:e] = jstar
    ext[:, 7] = np.log(cache_frac + EPS)
    return base, ext, jstar_all


def entry_history(jstar_fit, y_fit, w_fit, n_entries, smooth=20.0, folds=5, seed=0):
    """Smoothed per-entry mean retention from FIT queries.

    Returns (fit_values out-of-fold, full_table_mean, full_table_count).
    """
    prior = float(np.average(y_fit, weights=w_fit))

    def table(mask):
        cnt = np.bincount(jstar_fit[mask], weights=w_fit[mask], minlength=n_entries)
        tot = np.bincount(jstar_fit[mask], weights=(w_fit * y_fit)[mask], minlength=n_entries)
        return (tot + smooth * prior) / (cnt + smooth), cnt

    fold = np.random.default_rng(seed).integers(0, folds, size=len(y_fit))
    oof_mean = np.empty(len(y_fit))
    oof_cnt = np.empty(len(y_fit))
    for f in range(folds):
        m, c = table(fold != f)
        rows = fold == f
        oof_mean[rows] = m[jstar_fit[rows]]
        oof_cnt[rows] = c[jstar_fit[rows]]
    full_mean, full_cnt = table(np.ones(len(y_fit), dtype=bool))
    return (oof_mean, oof_cnt), full_mean, full_cnt


def build_state(task):
    """Features and labels for all splits of one cache state."""
    target, point, entries = task
    if len(entries) == 0:
        return None
    st = ecal.state_tables(entries)
    if len(st["cached"]) <= ecal._G["k"]:
        return None
    out = {"target": target, "cache_frac": point["cache_frac"],
           "oracle_hit_rate_online": point["test_hit_rate_online"],
           "oracle_hit_rate": point["test_hit_rate"]}
    raw = {}
    for name in SPLITS:
        q, w = ecal._G["sets"][name]
        y = st["in_cache"][ecal._G["topk"][name]].mean(axis=1)
        base, ext, jstar = full_features(q, st, point["cache_frac"])
        raw[name] = (base, ext, jstar, y, w.astype(np.float64))
    n_ent = len(entries)
    (oof_m, oof_c), full_m, full_c = entry_history(raw["fit"][2], raw["fit"][3],
                                                   raw["fit"][4], n_ent)
    for name in SPLITS:
        base, ext, jstar, y, w = raw[name]
        if name == "fit":
            ext[:, 5], ext[:, 6] = oof_m, np.log1p(oof_c)
        else:
            ext[:, 5], ext[:, 6] = full_m[jstar], np.log1p(full_c[jstar])
        out[name] = {"base": base, "ext": ext, "y": y, "w": w}
    return out


#-------------------------------------------------------------------------------


def _expand(split):
    idx = np.repeat(np.arange(len(split["y"])), split["w"].astype(np.int64))
    return idx


def _gbt(kind, quantile=None):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    common = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                  early_stopping=True, validation_fraction=0.1, random_state=0)
    if kind == "cls":
        return HistGradientBoostingClassifier(**common)
    if kind == "quantile":
        return HistGradientBoostingRegressor(loss="quantile", quantile=quantile, **common)
    return HistGradientBoostingRegressor(**common)


def train_models(state, pooled_model):
    """Return {model_name: score function over a split}."""
    fit = state["fit"]
    X_base, X_ext = fit["base"], np.hstack([fit["base"], fit["ext"]])
    y, w = fit["y"], fit["w"]
    scorers = {}
    rb = ecal.fit_ridge(X_base, y, w)
    scorers["ridge_base"] = lambda sp: rb(sp["base"])
    re = ecal.fit_ridge(X_ext, y, w)
    scorers["ridge_ext"] = lambda sp: re(np.hstack([sp["base"], sp["ext"]]))
    g = _gbt("reg").fit(X_ext, y, sample_weight=w)
    scorers["gbt"] = lambda sp: g.predict(np.hstack([sp["base"], sp["ext"]]))
    labels = (y >= state["target"] - 1e-9).astype(int)
    if 0 < labels.sum() < len(labels):
        c = _gbt("cls").fit(X_ext, labels, sample_weight=w)
        scorers["gbt_cls"] = lambda sp: c.predict_proba(np.hstack([sp["base"], sp["ext"]]))[:, 1]
    q = _gbt("quantile", 0.2).fit(X_ext, y, sample_weight=w)
    scorers["gbt_q20"] = lambda sp: q.predict(np.hstack([sp["base"], sp["ext"]]))
    if pooled_model is not None:
        scorers["gbt_pooled"] = lambda sp: pooled_model.predict(np.hstack([sp["base"], sp["ext"]]))
    return scorers


def evaluate_state(state, scorers, delta):
    out = {k: state[k] for k in ("target", "cache_frac", "oracle_hit_rate_online",
                                 "oracle_hit_rate")}
    target = state["target"]
    for name, score in scorers.items():
        s = {sp: score(state[sp]) for sp in SPLITS}
        grid = threshold_grid(s["fit"])
        cal_idx = _expand(state["cal"])
        rec = {}
        for tag, method in (("guaranteed", "normal_fixed"), ("naive", "naive")):
            lam = select(method, s["cal"][cal_idx], state["cal"]["y"][cal_idx],
                         grid, target, delta)
            for sp in ("dev", "test"):
                idx = _expand(state[sp])
                ss, yy = s[sp][idx], state[sp]["y"][idx]
                adm = ss >= lam if lam is not None else np.zeros(len(ss), dtype=bool)
                rec[f"{tag}_{sp}_hit_rate"] = float(adm.mean())
                rec[f"{tag}_{sp}_hit_retention"] = float(yy[adm].mean()) if adm.any() else None
        dev = state["dev"]
        ra = np.argsort(np.argsort(s["dev"]))
        rb = np.argsort(np.argsort(dev["y"]))
        rec["dev_spearman"] = float(np.corrcoef(ra, rb)[0, 1])
        out[name] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--targets", type=float, nargs="+", default=[0.9, 0.95])
    ap.add_argument("--sizes", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.30])
    ap.add_argument("--n-each", type=int, default=20000)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    run_cfg = json.loads((run_dir / "results.json").read_text())["config"]
    cand_top = pre_topk = None
    if "trace" in run_cfg:
        # trace runs: fixed time-ordered splits and saved exact neighbors
        from simulations.approximate_hits.run_e2e_oracle_trace import load_trace_run
        t = load_trace_run(run_dir)
        data, cfg, db, cand_q = t["data"], t["cfg"], t["db"], t["cand_q"]
        sets = {s: t["sets"][s] for s in SPLITS}
        pre_topk = {s: t["topk"][s] for s in SPLITS}
        cand_top = t["cand_top"]
    else:
        data, cfg, db, cand_q, sets = load_splits(run_dir, args.n_each)
    logger.info(f"{run_dir.name}: " + ", ".join(
        f"{k}={len(v[0])} distinct ({int(v[1].sum())} requests)" for k, v in sets.items()))
    cached_ids, radius, _, _ = build_entry_tables(cand_q, db, cfg.k_values, 1,
                                                  cached_top=cand_top)
    topk = pre_topk if pre_topk is not None else {
        name: exact_topk(q, db, cfg.k) for name, (q, _) in sets.items()}
    ecal._G.update(k=cfg.k, k_values=cfg.k_values, db=db, cand_q=cand_q, sets=sets,
                   topk=topk, cached_ids=cached_ids, radius=radius)

    tasks = []
    for res in data["results"]:
        if not any(abs(res["target"] - t) < 1e-9 for t in args.targets):
            continue
        for size in args.sizes:
            p = min(res["points"], key=lambda p: abs(p["cache_frac"] - size))
            tasks.append((res["target"], p, res["selected"][:p["n_entries"]]))
    logger.info(f"Building features for {len(tasks)} cache states...")
    import multiprocessing as mp
    with mp.get_context("fork").Pool(min(args.workers, len(tasks))) as pool:
        states = [s for s in pool.map(build_state, tasks) if s is not None]

    logger.info("Training pooled model...")
    Xp = np.vstack([np.hstack([s["fit"]["base"], s["fit"]["ext"]]) for s in states])
    yp = np.concatenate([s["fit"]["y"] for s in states])
    wp = np.concatenate([s["fit"]["w"] for s in states])
    pooled = _gbt("reg").fit(Xp, yp, sample_weight=wp)

    logger.info("Training per-state models and evaluating...")
    out = []
    for st in states:
        out.append(evaluate_state(st, train_models(st, pooled), args.delta))
    path = run_dir / "predictors.json"
    path.write_text(json.dumps({"args": vars(args), "models": MODELS,
                                "base_features": ecal.FEATURE_NAMES,
                                "ext_features": EXT_NAMES, "states": out}))
    logger.info(f"Wrote {path}")


if __name__ == "__main__":
    main()
