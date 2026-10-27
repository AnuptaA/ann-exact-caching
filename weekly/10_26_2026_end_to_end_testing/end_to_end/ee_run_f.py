#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Stage F (LSH arm): the stage D grid (all pools and baselines, HP1, HP2, HP3, HP3f, seeds 0 to 4, all regimes) with
the radius estimated by the chosen LSH configuration (results/stage_e/chosen_config.json). Pool construction is unchanged
(exact deficits). Stage D did not store pool masks, so each pool is rebuilt deterministically and checked against the
stage D rows (pool size and oracle-arm hit rate must match). Every row carries the LSH metrics, the oracle-arm metrics of
the same pool (columns or_*) and the structure and cache bytes.

Train estimates for the HP3 cap use the group-aware path: a pool vector within r_hat counts for train query q only if a
selected candidate containing it has a group excluding q (Popularity: pool rebuilt without q's own top-K set). Train r_hat
is leave-one-out.

Run from the thesis root (prefer ee_supervise.py):
    python3 weekly/10_26_2026_end_to_end_testing/end_to_end/ee_run_f.py [--dry-run]"""

import argparse
import hashlib
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
import ee_lsh_bench as lb
import ee_proposals as ep
import ee_run_d as rd
import ee_workload as ew
from data_prep import load_base_pool
from rt_chunks import RunAbort, current_rss_mb, free_disk_gb, write_table_atomic

OUT_DIR = cfg.RESULTS_DIR / "stage_f"
UNIT_DIR = OUT_DIR / "units"
for _d in (OUT_DIR, UNIT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

ROWS_PER_UNIT = rd.ROWS_PER_UNIT
OR_FIELDS = ("hit_rate", "hit_rate_recall_ge_half", "achieved_recall", "mean_true_recall_of_hits", "y_min",
             "frac_below_T", "hindsight_hit_rate_oracle")

def load_chosen(path=lb.OUT_DIR / "chosen_config.json") -> dict:
    chosen = json.loads(path.read_text())["config"]
    return {ds: dict(L=int(c["L"]), m=int(c["m"]), nu=int(c["nu"]), S=0 if c["S"] == "none" else int(c["S"]))
            for ds, c in chosen.items()}

def digest(*arrays) -> str:
    h = hashlib.blake2b(digest_size=16)
    for a in arrays:
        if a is not None:
            h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()

class PoolEstimates:
    """LSH estimated recall counts of one workload, memoized per pool (T independent pools recur across T)."""

    def __init__(self, w, base: np.ndarray, arm: est.LSHArm):
        self.w, self.base = w, base
        self.r_train, self.r_test = arm.radius("train"), arm.radius("test")
        self.memo = {}

    def counts(self, pool: eb.Pool, budget: int) -> tuple:
        key = (id(pool.inc), digest(pool.pool_mask, pool.selected), pool.name == "popularity", budget)
        if key not in self.memo:
            self.memo[key] = self._compute(pool, budget)
        return self.memo[key]

    def _compute(self, pool: eb.Pool, budget: int) -> tuple:
        w, base = self.w, self.base
        ids = np.flatnonzero(pool.pool_mask)
        if len(ids) == 0:
            return np.zeros(w.n, np.int64), np.zeros(w.n, np.int64)
        test = est.vectors_within_counts(w.test, self.r_test, base[ids])
        if pool.name == "popularity":
            train = eb.popularity_lsh_train_counts(w, budget, self.r_train, base)
        else:
            train = ei.group_aware_counts(pool.inc, pool.selected, ids, self.r_train, w.train, base)
        return test, train

def evaluate_both(w, pe: PoolEstimates, pool: eb.Pool, budget: int, T: float, seed: int, meta: dict) -> list:
    true_test = est.true_recall_counts(pool.pool_mask, w.test_top_ids)
    test_est, train_est = pe.counts(pool, budget)
    lsh = hp.evaluate_pool(seed, T, test_est, true_test, train_est, pool.train_true_counts, w.K, oracle_arm=False)
    orc = hp.evaluate_pool(seed, T, true_test, true_test, pool.train_true_counts, pool.train_true_counts, w.K,
                           oracle_arm=True)
    rows = []
    for a, b in zip(lsh, orc):
        assert a["policy"] == b["policy"]
        rows.append(dict(meta, pool=pool.name, pool_size=int(pool.pool_mask.sum()), n_selected=pool.n_selected, **a,
                         **{f"or_{k}": b[k] for k in OR_FIELDS}))
    return rows

def check_against_stage_d(df: pd.DataFrame, name: str) -> None:
    """Rebuilt pools must reproduce stage D: same pool size and oracle-arm hit rate for every (budget, T, pool, policy)."""
    path = rd.UNIT_DIR / f"{name}.csv.gz"
    if not path.exists():
        raise RunAbort(f"{name}: stage D unit missing, cannot verify the rebuilt pools")
    d = pd.read_csv(path, comment="#", low_memory=False)
    key = ["budget", "T", "pool", "policy"]
    m = df[key + ["pool_size", "or_hit_rate"]].merge(d[key + ["pool_size", "hit_rate"]], on=key, suffixes=("", "_d"))
    if len(m) != len(df) or (m.pool_size != m.pool_size_d).any() or not np.allclose(m.or_hit_rate, m.hit_rate, atol=1e-12, rtol=0):
        raise RunAbort(f"{name}: rebuilt pools differ from stage D")

def check_unit(df: pd.DataFrame, unit: str) -> None:
    cols = ["hit_rate", "achieved_recall", "y_min", "est_signed_error", "over_rate", "structure_bytes"]
    if df[cols].isna().any().any():
        raise RunAbort(f"{unit}: NaN in {df[cols].columns[df[cols].isna().any()].tolist()}")
    if len(df) != ROWS_PER_UNIT:
        raise RunAbort(f"{unit}: {len(df)} rows, expected {ROWS_PER_UNIT}")

def run_unit(ds: str, regime: str, seed: int, base: np.ndarray, db_theta: np.ndarray, chosen: dict) -> dict:
    lam, per_anchor = cfg.REGIME_SPECS[regime]
    name = rd.unit_name(ds, regime, seed)
    t_unit = time.time()
    w = ew.get_workload(ds, seed, lam, base, per_anchor=per_anchor)
    c = chosen[ds]
    arm = est.LSHArm(w, base, seed, c["L"], c["m"], c["nu"], c["S"], db_theta)
    pe = PoolEstimates(w, base, arm)
    n_labeled = w.n + c["S"]
    sbytes = lb.config_bytes(c["L"], c["m"], base.shape[1], n_labeled)
    incs = {kind: (lambda inc: (inc, ei.build_index(inc)))(ep.make_incidence(w, kind)[0]) for kind in ep.KINDS}
    rows = []

    def add(B, T, pool, meta, sel=None):
        for row in evaluate_both(w, pe, pool, B, T, seed, meta):
            row.update(sel or {}, structure_bytes=sbytes, cache_bytes=lb.cache_bytes(row["pool_size"], base.shape[1]))
            rows.append(row)

    for B in cfg.BUDGETS:
        base_pools = [eb.popularity(w, B), eb.random_own(w, B, seed, *incs["own"]), eb.threshold_baseline(w, B),
                      eb.no_cache(w)]
        sel3 = {kind: ei.greedy(*incs[kind], "SEL3", 0.9, B) for kind in ep.KINDS}
        for T in cfg.TARGETS:
            meta = dict(dataset=ds, regime=regime, lam=lam, per_anchor=per_anchor, seed=seed, budget=B, T=T,
                        L=c["L"], m=c["m"], nu=c["nu"], S=c["S"])
            ref = ei.greedy(*incs["group_union"], "SEL2", T, B)
            h0s = {m: min(w.n, max(1, int(round(m * ref.h_star)))) for m in rd.M_GRID}
            for kind in ep.KINDS:
                inc_idx = incs[kind]
                results = {"SEL1": ei.greedy(*inc_idx, "SEL1", T, B),
                           "SEL2": ref if kind == "group_union" else ei.greedy(*inc_idx, "SEL2", T, B),
                           "SEL3": sel3[kind], "SEL5": ei.greedy(*inc_idx, "SEL5", T, B)}
                for m in rd.M_GRID:
                    results[f"SEL4(m={m})"] = ei.greedy(*inc_idx, "SEL4", T, B, h0s[m])
                for label, r in results.items():
                    add(B, T, eb.pool_from_result(f"{kind}+{label}", r, inc_idx[0]), meta, rd.sel_meta(r))
            for pool in base_pools:
                add(B, T, pool, meta)
        assert current_rss_mb() < cfg.RSS_LIMIT_MB, f"RSS {current_rss_mb():.0f} MB"
        pe.memo.clear()
    df = pd.DataFrame(rows)
    check_unit(df, name)
    check_against_stage_d(df, name)
    write_table_atomic(df, UNIT_DIR / f"{name}.csv.gz")
    return dict(unit=name, rows=len(df), seconds=round(time.time() - t_unit, 1), rss_mb=round(current_rss_mb()))

def unit_done(name: str) -> bool:
    p = UNIT_DIR / f"{name}.csv.gz"
    try:
        return p.exists() and len(pd.read_csv(p, comment="#")) == ROWS_PER_UNIT
    except Exception:
        return False

def main() -> int:
    from logging_utils import setup_logging
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default="sift,esci")
    ap.add_argument("--seeds", default=",".join(str(s) for s in cfg.SEEDS))
    ap.add_argument("--dry-run", action="store_true", help="one unit: sift lam1 seed 0 (a normal unit file, kept)")
    a = ap.parse_args()
    logger = setup_logging("ee_run_f", cfg.LOGS_DIR)
    chosen = load_chosen()
    logger.info(f"EE_RUN_F chosen config {chosen}")
    units = [(ds, rg, s) for s in (int(x) for x in a.seeds.split(",")) for ds in a.datasets.split(",")
             for rg in rd.REGIMES[ds]]
    if a.dry_run:
        units = [("sift", "lam1", 0)]
    bases, thetas, done = {}, {}, []
    try:
        for ds, rg, seed in units:
            name = rd.unit_name(ds, rg, seed)
            if not a.dry_run and unit_done(name):
                logger.info(f"{name}: complete, skipping")
                continue
            if free_disk_gb(cfg.RESULTS_DIR) < cfg.MIN_FREE_DISK_GB:
                raise RunAbort(f"free disk below {cfg.MIN_FREE_DISK_GB} GiB before {name}")
            if ds not in bases:
                bases[ds] = load_base_pool(ds)
                thetas[ds] = ew.db_theta(ds, bases[ds])
            info = run_unit(ds, rg, seed, bases[ds], thetas[ds], chosen)
            done.append(info)
            logger.info(f"{name}: {info['rows']} rows in {info['seconds']}s, rss {info['rss_mb']} MB")
            with open(cfg.LOGS_DIR / "stage_f_units.csv", "a") as f:
                f.write(f"{datetime.now().isoformat()},{name},{info['rows']},{info['seconds']},{info['rss_mb']}\n")
            if current_rss_mb() > cfg.RSS_LIMIT_MB:
                raise RunAbort(f"RSS {current_rss_mb():.0f} MB above {cfg.RSS_LIMIT_MB:.0f} MB after {name}")
    except RunAbort as exc:
        (OUT_DIR / "ABORT_stage_f.txt").write_text(f"{datetime.now().isoformat()}\n{exc.reason}\n")
        logger.error(f"ABORT: {exc.reason}")
        return cfg.EXIT_ABORT
    logger.info(f"EE_RUN_F COMPLETE, {len(done)} units run, {sum(i['seconds'] for i in done):.0f}s")
    (OUT_DIR / "stage_f_summary.json").write_text(json.dumps(dict(units_run=len(done), finished=datetime.now().isoformat(),
                                                                   peak_rss_mb=current_rss_mb()), indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
