#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Radius-estimation baselines vs direct retention prediction, end to end.

For saved greedy caches (selected states), every estimator produces a
per-query retention estimate:
  radius-based   estimated retention = min(1, #cached within r_hat / k)
                 with r_hat from: LSH buckets over FIT-query radii (the
                 original approach; config chosen on DEV by radius RMSE),
                 the global mean FIT radius, or the true radius (sanity check)
  direct         ridge (base features) and gradient-boosted trees (GBT)
                 predicting retention from cache features

Reported per estimator: radius RMSE (radius-based only), retention MAE vs
true retention, realized retention when the estimate is trusted as-is
(admit while the estimated mean over hits >= T), and the guaranteed test hit
rate when the estimate is used as the score for calibrated admission
(normal approximation + fixed order, delta = 0.05).

Requires scikit-learn (run with the repo's .venv python).
"""

import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from datasets.dataloaders import load_fvecs_fast
from simulations.approximate_hits import eval_calibrated as ecal
from simulations.approximate_hits import eval_predictors as epred
from simulations.approximate_hits.eval_certificates import build_entry_tables, pairwise_dist
from simulations.approximate_hits.run_e2e_oracle_sift import (
    SIFT1M_DIR, exact_topk, load_inputs,
)
from simulations.approximate_hits.threshold_selection import select, threshold_grid
from simulations.radius_estimation.run_granularity_sweep import MultiTableLSH

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SPLITS = ("fit", "dev", "cal", "test")
LSH_CONFIGS = [(12, 1), (16, 1), (20, 1), (12, 4), (16, 4), (20, 4)]  # (bits, tables)
CHUNK = 2048

#-------------------------------------------------------------------------------


def load_splits(run_dir: Path, n_each: int):
    data = json.loads((run_dir / "results.json").read_text())
    cfg = argparse.Namespace(**data["config"])
    if not hasattr(cfg, "dataset"):
        cfg.dataset = "sift"  # runs from before --dataset existed were SIFT
    if cfg.dataset == "sift" and "zipf" not in data["config"]:
        db, cand_q, _, test_q = load_inputs(cfg)
        learn = load_fvecs_fast(str(SIFT1M_DIR / "sift_learn.fvecs"))
        rng = np.random.default_rng(cfg.seed)
        rng.choice(1_000_000, size=cfg.n_db, replace=False)  # same draws as load_inputs
        used = rng.choice(len(learn), size=cfg.n_cand_queries + cfg.n_score_queries,
                          replace=False)
        free = np.setdiff1d(np.arange(len(learn)), used)
        free = np.random.default_rng(cfg.seed + 5).permutation(free)
        sets = {}
        for i, name in enumerate(("fit", "dev", "cal")):
            q = learn[free[i * n_each:(i + 1) * n_each]]
            sets[name] = (q, np.ones(len(q), dtype=np.int64))
        sets["test"] = (test_q, np.ones(len(test_q), dtype=np.int64))
        return data, cfg, db, cand_q, sets
    return epred.load_splits(run_dir, n_each)


def kth_distance(q, db, ids):
    """Distance from each query to its k-th true neighbor (last column of ids)."""
    kth = ids[:, -1]
    diff = q.astype(np.float64) - db[kth].astype(np.float64)
    return np.sqrt(np.einsum("ij,ij->i", diff, diff))


def cached_counts_within(q, st, radii):
    """#cached vectors within radii[i] of q[i]."""
    out = np.empty(len(q), dtype=np.int64)
    for s in range(0, len(q), CHUNK):
        e = min(s + CHUNK, len(q))
        d = pairwise_dist(q[s:e], st["db_c"], st["db_c_sq"])
        out[s:e] = (d <= radii[s:e, None]).sum(axis=1)
    return out


def trusted_realized(est, true, target, k):
    """Admit using the estimate as if it were exact (best-case admission on
    estimates); return (hit rate, realized true mean retention over hits)."""
    order = np.argsort(-est)
    est_s, true_s = est[order], true[order]
    cum = np.cumsum(est_s) / np.arange(1, len(est_s) + 1)
    # cum is non-increasing, so the admissible set is a prefix
    n = int((cum >= target - 1e-12).sum())
    if n == 0:
        return 0.0, None
    return n / len(est), float(true_s[:n].mean())


def evaluate(run_dir, data, cfg, db, cand_q, sets, targets, sizes, delta):
    k = cfg.k
    topk = {name: exact_topk(q, db, k) for name, (q, _) in sets.items()}
    radius_true = {name: kth_distance(sets[name][0], db, topk[name]) for name in SPLITS}
    cached_ids, radius, _, _ = build_entry_tables(cand_q, db, cfg.k_values, 1)
    ecal._G.update(k=k, k_values=cfg.k_values, db=db, cand_q=cand_q, sets=sets,
                   topk=topk, cached_ids=cached_ids, radius=radius)

    # radius estimators (independent of the cache): LSH tuned on dev, global mean
    w_fit = sets["fit"][1].astype(np.float64)
    gmean = float(np.average(radius_true["fit"], weights=w_fit))
    lsh_rmse = {}
    for bits, L in LSH_CONFIGS:
        lsh = MultiTableLSH(L, bits, db.shape[1], cfg.seed)
        lsh.fit(sets["fit"][0], radius_true["fit"])
        est, _ = lsh.estimate(sets["dev"][0], 0, gmean)
        lsh_rmse[(bits, L)] = float(np.sqrt(np.mean((est - radius_true["dev"]) ** 2)))
    best_cfg = min(lsh_rmse, key=lsh_rmse.get)
    lsh = MultiTableLSH(best_cfg[1], best_cfg[0], db.shape[1], cfg.seed)
    lsh.fit(sets["fit"][0], radius_true["fit"])
    r_hat = {"lsh": {}, "global_mean": {}, "true_radius": {}}
    for name in ("cal", "test"):
        r_hat["lsh"][name] = lsh.estimate(sets[name][0], 0, gmean)[0]
        r_hat["global_mean"][name] = np.full(len(sets[name][0]), gmean)
        r_hat["true_radius"][name] = radius_true[name]
    rt = radius_true["test"]
    radius_stats = {
        "true_mean": float(rt.mean()), "true_std": float(rt.std()),
        "lsh_config": {"bits": best_cfg[0], "tables": best_cfg[1]},
        "lsh_dev_rmse_by_config": {f"{b}b_{t}t": v for (b, t), v in lsh_rmse.items()},
    }
    for est in ("lsh", "global_mean"):
        err = r_hat[est]["test"] - rt
        radius_stats[est] = {"rmse": float(np.sqrt(np.mean(err ** 2))),
                             "mean_error": float(err.mean())}
    logger.info(f"  radius: true mean {rt.mean():.3f} std {rt.std():.3f}; "
                f"LSH {best_cfg} RMSE {radius_stats['lsh']['rmse']:.3f}; "
                f"global RMSE {radius_stats['global_mean']['rmse']:.3f}")

    states = []
    for res in data["results"]:
        if not any(abs(res["target"] - t) < 1e-9 for t in targets):
            continue
        target = res["target"]
        for size in sizes:
            p = min(res["points"], key=lambda p: abs(p["cache_frac"] - size))
            entries = res["selected"][:p["n_entries"]]
            if not entries:
                continue
            st = ecal.state_tables(entries)
            true_ret = {n: st["in_cache"][topk[n]].mean(axis=1) for n in SPLITS}
            scores = {}
            for est in r_hat:
                scores[est] = {n: np.minimum(1.0, cached_counts_within(
                    sets[n][0], st, r_hat[est][n]) / k) for n in ("cal", "test")}
            feats = {n: epred.full_features(sets[n][0], st, p["cache_frac"]) for n in SPLITS}
            Xb = {n: feats[n][0] for n in SPLITS}
            Xe = {n: np.hstack([feats[n][0], feats[n][1]]) for n in SPLITS}
            ridge = ecal.fit_ridge(Xb["fit"], true_ret["fit"], w_fit)
            gbt = epred._gbt("reg").fit(Xe["fit"], true_ret["fit"], sample_weight=w_fit)
            scores["ridge"] = {n: ridge(Xb[n]) for n in ("fit", "cal", "test")}
            scores["gbt"] = {n: gbt.predict(Xe[n]) for n in ("fit", "cal", "test")}

            row = {"target": target, "cache_frac": p["cache_frac"],
                   "oracle_best_case_hit_rate": p["test_hit_rate"],
                   "oracle_online_hit_rate": p["test_hit_rate_online"]}
            yt = true_ret["test"]
            for est, sc in scores.items():
                # grid from an independent split: fit for learned models; for
                # radius-based scores (no fitting) the calibration split's
                # score quantiles are replaced by fixed retention levels
                if "fit" in sc:
                    grid = threshold_grid(sc["fit"])
                else:
                    grid = np.arange(k, -1, -1) / k
                lam = select("normal_fixed", sc["cal"], true_ret["cal"], grid, target, delta)
                adm = sc["test"] >= lam if lam is not None else np.zeros(len(yt), bool)
                trust_hit, trust_ret = trusted_realized(sc["test"], yt, target, k)
                row[est] = {
                    "retention_mae": float(np.mean(np.abs(np.clip(sc["test"], 0, 1) - yt))),
                    "retention_mean_error": float(np.mean(np.clip(sc["test"], 0, 1) - yt)),
                    "guaranteed_hit_rate": float(adm.mean()),
                    "guaranteed_hit_retention": float(yt[adm].mean()) if adm.any() else None,
                    "trusted_hit_rate": trust_hit,
                    "trusted_realized_retention": trust_ret,
                }
            states.append(row)
            logger.info(f"  T={target} {100 * p['cache_frac']:.1f}%: " + ", ".join(
                f"{e} {100 * row[e]['guaranteed_hit_rate']:.1f}" for e in scores))
    return {"radius": radius_stats, "states": states}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--targets", type=float, nargs="+", default=[0.9, 0.95])
    ap.add_argument("--sizes", type=float, nargs="+", default=[0.01, 0.10, 0.30])
    ap.add_argument("--n-each", type=int, default=20000)
    ap.add_argument("--delta", type=float, default=0.05)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    data, cfg, db, cand_q, sets = load_splits(run_dir, args.n_each)
    logger.info(f"{run_dir.name}: " + ", ".join(f"{k}={len(v[0])}" for k, v in sets.items()))
    out = evaluate(run_dir, data, cfg, db, cand_q, sets, args.targets, args.sizes, args.delta)
    (run_dir / "radius_baselines.json").write_text(json.dumps(out))
    logger.info(f"Wrote {run_dir / 'radius_baselines.json'}")


if __name__ == "__main__":
    main()
