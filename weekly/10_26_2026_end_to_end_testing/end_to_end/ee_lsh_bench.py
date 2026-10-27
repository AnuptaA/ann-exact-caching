#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Stage E: LSH mini-benchmark. Grid: (L, m) in the 7 configurations, nu in {1, 2, 4, 8, 16, 32} per table, sampled database
points S in {none, 2,500, 10,000} plus S = full (reference only, never chosen). All three lambda regimes, both datasets,
seeds 100 to 102 (never used for reported runs). Each configuration is evaluated on two 5% reference pools: Popularity
and Group Union SEL2 (T = 0.9). Metrics per test query: estimated minus true recall (mean signed error, MAE, share with
an overestimate above 0.05) and the radius relative error (mean, sd, slope of r_hat on theta). The chosen configuration per
dataset and pool has the lowest MAE averaged over regimes and seeds; configurations within 0.005 of that MAE are ranked by
the lowest overestimation rate. One resumable file per (dataset, regime, seed).

Run from the thesis root (prefer ee_supervise.py):  python3 weekly/10_26_2026_end_to_end_testing/end_to_end/ee_lsh_bench.py"""

import argparse
import json
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

import ee_baselines as eb
import ee_config as cfg
import ee_estimators as est
import ee_incidence as ei
import ee_proposals as ep
import ee_workload as ew
from data_prep import load_base_pool
from rt_chunks import RunAbort, current_rss_mb, free_disk_gb, write_table_atomic
from rt_engine import make_hyperplanes
from utils.base.distance_metrics import pairwise_angular_distance

OUT_DIR = cfg.RESULTS_DIR / "stage_e"
UNIT_DIR = OUT_DIR / "units"
for _d in (OUT_DIR, UNIT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

NUS = list(cfg.NU_GRID)
S_OPTIONS = (0, 2500, 10_000, cfg.N_DB)  # 0 = none, N_DB = full (reference only)
S_LABEL = {0: "none", 2500: "2500", 10_000: "10000", cfg.N_DB: "full"}
POOL_NAMES = ("popularity", "group_union_sel2")
OVER_TOL = 0.05
TIE_TOL = 0.005
REF_BUDGET = cfg.BUDGETS[1]  # 5%
REF_T = 0.9

#-------------------------------------------------------------------------------
# Metrics

def recall_metrics(est_counts: np.ndarray, true_counts: np.ndarray, K: int = cfg.K) -> dict:
    """Estimated minus true recall (unclipped estimate): mean signed error, MAE, share above OVER_TOL."""
    diff = (np.asarray(est_counts) - np.asarray(true_counts)) / K
    return dict(mean_signed_error=float(diff.mean()), mae=float(np.abs(diff).mean()),
                over_rate=float((diff > OVER_TOL + 1e-12).mean()), mean_est_recall=float(np.mean(est_counts) / K),
                mean_true_recall=float(np.mean(true_counts) / K))

def radius_metrics(r_hat: np.ndarray, theta: np.ndarray) -> dict:
    rel = (r_hat - theta) / theta
    return dict(radius_mean_rel_err=float(rel.mean()), radius_sd_rel_err=float(rel.std()),
                radius_slope=float(np.polyfit(theta, r_hat, 1)[0]))

def config_bytes(L: int, m: int, d: int, n_labeled: int) -> int:
    """Codes (L tables of m bits per labeled point) + float64 labels + the +/-1 hyperplanes as bits."""
    return int(n_labeled * (L * m // 8 + 8) + L * m * d // 8)

#-------------------------------------------------------------------------------

def bench_unit(w, base: np.ndarray, db_theta: np.ndarray, seed: int, pools: dict, configs=cfg.CONFIGS, nus=NUS,
               s_options=S_OPTIONS, extra: dict = None) -> pd.DataFrame:
    """Rows for one workload: every (pool, config, nu, S). `pools` maps a name to a boolean pool mask."""
    d = base.shape[1]
    W = make_hyperplanes(seed, d)
    perm = np.random.default_rng([seed, 21]).permutation(len(base))
    true_counts = {name: est.true_recall_counts(mask, w.test_top_ids) for name, mask in pools.items()}
    pool_dist = {name: pairwise_angular_distance(w.test, base[np.flatnonzero(mask)]) for name, mask in pools.items()}
    rows = []
    for S in s_options:
        ids = perm[:S]
        labeled = np.vstack([w.train, base[ids]]) if S else w.train
        labels = np.concatenate([w.theta_train, db_theta[ids]]) if S else w.theta_train
        for L, m in configs:
            r_all = est.LSHStore(W, L, m, labeled, labels).estimate_multi(w.test, nus)
            for j, nu in enumerate(nus):
                r_hat = r_all[j]
                rad = radius_metrics(r_hat, w.theta_test)
                for name in pools:
                    counts = (pool_dist[name] <= r_hat[:, None]).sum(axis=1)
                    rows.append(dict(extra or {}, seed=seed, pool=name, L=L, m=m, total_bits=L * m, nu=nu,
                                     S=S_LABEL.get(S, str(S)), S_num=S, n_labeled=len(labeled),
                                     bytes=config_bytes(L, m, d, len(labeled)),
                                     **recall_metrics(counts, true_counts[name], w.K), **rad))
    return pd.DataFrame(rows)

#-------------------------------------------------------------------------------
# Selection

def choose_config(rows: pd.DataFrame, dataset: str, pool: str, tie_tol: float = TIE_TOL) -> dict:
    """Lowest MAE averaged over regimes and seeds among S != full; ties within tie_tol go to the lowest overestimation
    rate (then the lowest MAE, then the first configuration in grid order)."""
    r = rows[(rows.dataset == dataset) & (rows.pool == pool) & (rows.S != "full")]
    g = r.groupby(["L", "m", "nu", "S"], sort=False).agg(mae=("mae", "mean"), over_rate=("over_rate", "mean"),
                                                         mean_signed_error=("mean_signed_error", "mean"),
                                                         bytes=("bytes", "first"), n=("mae", "size")).reset_index()
    best = g.mae.min()
    cand = g[g.mae <= best + tie_tol + 1e-15].sort_values(["over_rate", "mae"], kind="stable")
    pick = cand.iloc[0]
    out = {k: (v.item() if hasattr(v, "item") else v) for k, v in pick.to_dict().items()}
    out.update(dataset=dataset, pool=pool, best_mae=float(best), n_within_tie=int(len(cand)), n_averaged=int(pick.n))
    return out

#-------------------------------------------------------------------------------

def reference_pools(w, T: float = REF_T, B: int = REF_BUDGET) -> dict:
    inc, _ = ep.make_incidence(w, "group_union")
    return dict(popularity=eb.popularity(w, B).pool_mask,
                group_union_sel2=ei.greedy(inc, ei.build_index(inc), "SEL2", T, B).pool_mask)

def unit_name(ds: str, regime: str, seed: int) -> str:
    return f"{ds}_{regime}_seed{seed}"

ROWS_PER_UNIT = len(POOL_NAMES) * len(cfg.CONFIGS) * len(NUS) * len(S_OPTIONS)

def main() -> int:
    from logging_utils import setup_logging
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default="sift,esci")
    ap.add_argument("--seeds", default=",".join(str(s) for s in cfg.BENCH_SEEDS))
    ap.add_argument("--select-only", action="store_true")
    a = ap.parse_args()
    logger = setup_logging("ee_lsh_bench", cfg.LOGS_DIR)
    regimes = ["lam0", "lam0.5", "lam1"]
    if not a.select_only:
        try:
            for ds in a.datasets.split(","):
                base = load_base_pool(ds)
                db_theta = ew.db_theta(ds, base)
                for seed in (int(s) for s in a.seeds.split(",")):
                    for rg in regimes:
                        name = unit_name(ds, rg, seed)
                        path = UNIT_DIR / f"{name}.csv.gz"
                        if path.exists() and len(pd.read_csv(path, comment="#")) == ROWS_PER_UNIT:
                            logger.info(f"{name}: complete, skipping")
                            continue
                        if free_disk_gb(cfg.RESULTS_DIR) < cfg.MIN_FREE_DISK_GB:
                            raise RunAbort(f"free disk below {cfg.MIN_FREE_DISK_GB} GiB before {name}")
                        t0 = time.time()
                        lam, pa = cfg.REGIME_SPECS[rg]
                        w = ew.get_workload(ds, seed, lam, base, per_anchor=pa)
                        df = bench_unit(w, base, db_theta, seed, reference_pools(w), extra=dict(dataset=ds, regime=rg))
                        if df.isna().any().any() or len(df) != ROWS_PER_UNIT:
                            raise RunAbort(f"{name}: NaN or wrong row count ({len(df)})")
                        write_table_atomic(df, path)
                        logger.info(f"{name}: {len(df)} rows in {time.time() - t0:.0f}s, rss {current_rss_mb():.0f} MB")
                        if current_rss_mb() > cfg.RSS_LIMIT_MB:
                            raise RunAbort(f"RSS above {cfg.RSS_LIMIT_MB} MB after {name}")
        except RunAbort as exc:
            (OUT_DIR / "ABORT_stage_e.txt").write_text(f"{datetime.now().isoformat()}\n{exc.reason}\n")
            logger.error(f"ABORT: {exc.reason}")
            return cfg.EXIT_ABORT
    rows = pd.concat([pd.read_csv(p, comment="#", low_memory=False) for p in sorted(UNIT_DIR.glob("*.csv.gz"))],
                     ignore_index=True)
    chosen = {}
    for pool in POOL_NAMES:
        chosen[pool] = {ds: choose_config(rows, ds, pool) for ds in sorted(rows.dataset.unique())}
        (OUT_DIR / f"chosen_config_{pool}.json").write_text(json.dumps(chosen[pool], indent=2))
    primary = {ds: dict(chosen["popularity"][ds]) for ds in chosen["popularity"]}
    differs = {ds: {k: chosen["popularity"][ds][k] for k in ("L", "m", "nu", "S")} !=
               {k: chosen["group_union_sel2"][ds][k] for k in ("L", "m", "nu", "S")} for ds in primary}
    (OUT_DIR / "chosen_config.json").write_text(json.dumps(dict(primary_pool="popularity", config=primary,
                                                                differs_between_pools=differs), indent=2))
    logger.info(f"chosen configs written; differs between pools: {differs}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
