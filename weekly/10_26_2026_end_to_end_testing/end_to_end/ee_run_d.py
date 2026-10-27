#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Stage D (oracle arm): all proposal x score pools plus the baselines, policies HP1, HP2, HP3 and HP3f, seeds 0 to 4,
all regimes. One resumable unit per (dataset, regime, seed): 3 budgets x 3 targets x 25 pools x 4 policies = 900 rows,
written atomically; a unit with a complete file is skipped on restart. Per-step gains of the SEL2 and SEL5 runs are
saved for the stall diagnostics. Abort rules: RSS above the limit, free disk below the limit, NaN in a result column,
Y_t below T for an oracle policy (cannot happen by construction), or a hit rate above the hindsight ceiling.

Run from the thesis root (prefer ee_supervise.py for the full run):
    python3 weekly/10_26_2026_end_to_end_testing/end_to_end/ee_run_d.py [--dry-run]"""

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
import ee_hitpolicy as hp
import ee_incidence as ei
import ee_proposals as ep
import ee_workload as ew
from data_prep import load_base_pool
from rt_chunks import RunAbort, current_rss_mb, free_disk_gb, write_table_atomic

OUT_DIR = cfg.RESULTS_DIR / "stage_d"
UNIT_DIR = OUT_DIR / "units"
STEP_DIR = OUT_DIR / "steps"
for _d in (OUT_DIR, UNIT_DIR, STEP_DIR):
    _d.mkdir(parents=True, exist_ok=True)

M_GRID = (0.5, 1.0, 1.5)
ROWS_PER_UNIT = len(cfg.BUDGETS) * len(cfg.TARGETS) * (3 * 7 + 4) * len(hp.POLICIES)
# Per dataset regime list: the diffuse regime is included only where the 10% budget binds (SIFT, not ESCI).
REGIMES = {"sift": ["lam0", "lam0.5", "lam1", "diffuse"], "esci": ["lam0", "lam0.5", "lam1"]}
STEP_SCORES = ("SEL2", "SEL5")

def unit_name(ds: str, regime: str, seed: int) -> str:
    return f"{ds}_{regime}_seed{seed}"

def evaluate(w, pool: eb.Pool, T: float, seed: int, meta: dict) -> list:
    true_test = est.true_recall_counts(pool.pool_mask, w.test_top_ids)
    rows = hp.evaluate_pool(seed, T, true_test, true_test, pool.train_true_counts, pool.train_true_counts, w.K,
                            oracle_arm=True)
    return [dict(meta, pool=pool.name, pool_size=int(pool.pool_mask.sum()), n_selected=pool.n_selected, **r)
            for r in rows]

def sel_meta(r: ei.Result, seconds: float = None) -> dict:
    return dict(stalled=r.stalled, unused_budget=r.unused_budget, h_star_train=r.h_star,
                n_sources=r.n_distinct_sources, select_seconds=round(r.seconds if seconds is None else seconds, 3))

def check_unit(df: pd.DataFrame, unit: str) -> None:
    cols = ["hit_rate", "achieved_recall", "y_min", "hindsight_hit_rate_oracle"]
    if df[cols].isna().any().any():
        raise RunAbort(f"{unit}: NaN in {df[cols].columns[df[cols].isna().any()].tolist()}")
    if (df["frac_below_T"] > 0).any():
        raise RunAbort(f"{unit}: running Y_t fell below T for an oracle policy")
    if (df["hit_rate"] > df["hindsight_hit_rate_oracle"] + 1e-9).any():
        raise RunAbort(f"{unit}: a hit rate exceeds the hindsight ceiling")
    if len(df) != ROWS_PER_UNIT:
        raise RunAbort(f"{unit}: {len(df)} rows, expected {ROWS_PER_UNIT}")

def run_unit(ds: str, regime: str, seed: int, base: np.ndarray, logger) -> dict:
    lam, per_anchor = cfg.REGIME_SPECS[regime]
    name = unit_name(ds, regime, seed)
    t_unit = time.time()
    w = ew.get_workload(ds, seed, lam, base, per_anchor=per_anchor)
    incs = {}
    for kind in ep.KINDS:
        inc, _ = ep.make_incidence(w, kind)
        incs[kind] = (inc, ei.build_index(inc))
    rows, steps = [], {}

    def record_steps(kind, score, B, T, r):
        if score in STEP_SCORES and len(r.steps["gain"]):
            key = f"{kind}|{score}|{B}|{T}"
            for field in ("gain", "cost", "n_positive", "h_before", "h_after", "pool_size"):
                steps[f"{key}|{field}"] = r.steps[field].astype(np.int32)

    for B in cfg.BUDGETS:
        base_pools = [eb.popularity(w, B), eb.random_own(w, B, seed, *incs["own"]), eb.threshold_baseline(w, B),
                      eb.no_cache(w)]
        sel3 = {kind: ei.greedy(*incs[kind], "SEL3", 0.9, B) for kind in ep.KINDS}  # T-independent
        for T in cfg.TARGETS:
            meta = dict(dataset=ds, regime=regime, lam=lam, per_anchor=per_anchor, seed=seed, budget=B, T=T)
            ref = ei.greedy(*incs["group_union"], "SEL2", T, B)
            h0s = {m: min(w.n, max(1, int(round(m * ref.h_star)))) for m in M_GRID}
            for kind in ep.KINDS:
                inc_idx = incs[kind]
                results = {"SEL1": ei.greedy(*inc_idx, "SEL1", T, B),
                           "SEL2": ref if kind == "group_union" else ei.greedy(*inc_idx, "SEL2", T, B),
                           "SEL3": sel3[kind], "SEL5": ei.greedy(*inc_idx, "SEL5", T, B)}
                for m in M_GRID:
                    results[f"SEL4(m={m})"] = ei.greedy(*inc_idx, "SEL4", T, B, h0s[m])
                for label, r in results.items():
                    pool = eb.pool_from_result(f"{kind}+{label}", r)
                    for row in evaluate(w, pool, T, seed, meta):
                        row.update(sel_meta(r))
                        rows.append(row)
                    record_steps(kind, label.split("(")[0], B, T, r)
            for pool in base_pools:
                for row in evaluate(w, pool, T, seed, meta):
                    rows.append(row)
        assert current_rss_mb() < cfg.RSS_LIMIT_MB, f"RSS {current_rss_mb():.0f} MB"
    df = pd.DataFrame(rows)
    check_unit(df, name)
    write_table_atomic(df, UNIT_DIR / f"{name}.csv.gz")
    if steps:
        tmp = STEP_DIR / f"{name}.tmp.npz"
        np.savez_compressed(tmp, **steps)
        tmp.replace(STEP_DIR / f"{name}.npz")
    return dict(unit=name, rows=len(df), seconds=round(time.time() - t_unit, 1), rss_mb=round(current_rss_mb()))

def unit_done(name: str) -> bool:
    p = UNIT_DIR / f"{name}.csv.gz"
    if not p.exists():
        return False
    try:
        return len(pd.read_csv(p, comment="#")) == ROWS_PER_UNIT
    except Exception:
        return False

def main() -> int:
    from logging_utils import setup_logging
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default="sift,esci")
    ap.add_argument("--regimes", default=None, help="comma list; default is the per-dataset list")
    ap.add_argument("--seeds", default=",".join(str(s) for s in cfg.SEEDS))
    ap.add_argument("--dry-run", action="store_true", help="one unit: sift lam1 seed 0 (no skipping of done units)")
    a = ap.parse_args()
    logger = setup_logging("ee_run_d", cfg.LOGS_DIR)
    datasets = a.datasets.split(",")
    seeds = [int(s) for s in a.seeds.split(",")]
    units = [(ds, rg, s) for s in seeds for ds in datasets for rg in
             (a.regimes.split(",") if a.regimes else REGIMES[ds]) if rg in REGIMES[ds] or a.regimes]
    if a.dry_run:
        units = [("sift", "lam1", 0)]
    logger.info(f"EE_RUN_D STARTING {len(units)} units, {ROWS_PER_UNIT} rows each")
    bases = {}
    done = []
    try:
        for ds, rg, seed in units:
            name = unit_name(ds, rg, seed)
            if not a.dry_run and unit_done(name):
                logger.info(f"{name}: complete, skipping")
                continue
            free = free_disk_gb(cfg.RESULTS_DIR)
            if free < cfg.MIN_FREE_DISK_GB:
                raise RunAbort(f"free disk {free:.2f} GiB below {cfg.MIN_FREE_DISK_GB} GiB before {name}")
            if ds not in bases:
                bases[ds] = load_base_pool(ds)
            info = run_unit(ds, rg, seed, bases[ds], logger)
            done.append(info)
            logger.info(f"{name}: {info['rows']} rows in {info['seconds']}s, rss {info['rss_mb']} MB")
            with open(cfg.LOGS_DIR / "stage_d_units.csv", "a") as f:
                f.write(f"{datetime.now().isoformat()},{name},{info['rows']},{info['seconds']},{info['rss_mb']}\n")
            if current_rss_mb() > cfg.RSS_LIMIT_MB:
                raise RunAbort(f"RSS {current_rss_mb():.0f} MB above {cfg.RSS_LIMIT_MB:.0f} MB after {name}")
    except RunAbort as exc:
        path = OUT_DIR / "ABORT_stage_d.txt"
        path.write_text(f"{datetime.now().isoformat()}\n{exc.reason}\n")
        logger.error(f"ABORT: {exc.reason} (report: {path})")
        return cfg.EXIT_ABORT
    logger.info(f"EE_RUN_D COMPLETE, {len(done)} units run")
    (OUT_DIR / "stage_d_summary.json").write_text(json.dumps(dict(units_run=len(done), finished=datetime.now().isoformat(),
                                                                   peak_rss_mb=current_rss_mb()), indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
