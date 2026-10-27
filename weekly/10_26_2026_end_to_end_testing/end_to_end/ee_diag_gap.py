#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Diagnostic: HP3 hit rate against hindsight on the test stream for the cells where the train-minus-test hit fraction
gap exceeds 0.05 (from the saved stage D tables), and a cap analysis that recomputes the Popularity pools of SIFT diffuse:
the learned cap against the best cap chosen on the test stream, HP2 (no cap), the train and test hindsight ceilings.

Run from the thesis root:  python3 weekly/10_26_2026_end_to_end_testing/end_to_end/ee_diag_gap.py"""

import sys

import numpy as np
import pandas as pd

import ee_baselines as eb
import ee_config as cfg
import ee_estimators as est
import ee_hitpolicy as hp
import ee_workload as ew
from data_prep import load_base_pool

OUT_DIR = cfg.RESULTS_DIR / "gap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def large_gap_cells(cells: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    p = cells[cells.policy == "HP3"].copy()
    p["gap"] = p.train_hit_fraction - p.test_hit_fraction
    p = p[p.gap > threshold].copy()
    p["hindsight"] = p.hindsight_hit_rate_oracle
    p["hp3_over_hindsight"] = p.hit_rate / p.hindsight
    p["hindsight_minus_hp3"] = p.hindsight - p.hit_rate
    return p

def cap_analysis(ds: str, regime: str, seed: int, B: int, T: float, base: np.ndarray) -> list:
    lam, pa = cfg.REGIME_SPECS[regime]
    w = ew.get_workload(ds, seed, lam, base, per_anchor=pa)
    pool = eb.popularity(w, B)
    true_test = est.true_recall_counts(pool.pool_mask, w.test_top_ids)
    est_u = hp.counts_to_units(true_test)[hp.stream_order(seed, len(true_test))]
    true_c = true_test[hp.stream_order(seed, len(true_test))]
    train_u = hp.counts_to_units(pool.train_true_counts)
    cap = hp.learn_cap(train_u, T)
    train_hits = hp.run_policy("HP3", train_u[hp.stream_order(seed, len(train_u))], T, cap).mean()
    best = max(((c, hp.run_policy("HP3", est_u, T, c).mean()) for c in range(0, 21)), key=lambda x: (x[1], -x[0]))
    return [dict(dataset=ds, regime=regime, seed=seed, budget=B, T=T, learned_cap_units=cap,
                 train_mean_recall=float(pool.train_true_counts.mean() / w.K), test_mean_recall=float(true_test.mean() / w.K),
                 train_hit_fraction=float(train_hits), test_hit_hp3=float(hp.run_policy("HP3", est_u, T, cap).mean()),
                 test_hit_best_cap=float(best[1]), best_cap_units=best[0], test_hit_hp2=float(hp.run_policy("HP2", est_u, T).mean()),
                 hindsight_test=hp.hindsight_count(hp.counts_to_units(true_test), T) / len(true_test),
                 hindsight_train=hp.hindsight_count(train_u, T) / len(train_u))]

def main() -> int:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 300)
    cells = pd.read_csv(cfg.RESULTS_DIR / "stage_d" / "stage_d_cell_summary.csv.gz")
    big = large_gap_cells(cells)
    big.to_csv(OUT_DIR / "large_gap_cells.csv", index=False)
    print(f"cells with train - test > 0.05: {len(big)} of {int((cells.policy == 'HP3').sum())} HP3 cells")
    summary = big.groupby(["dataset", "regime", "pool"]).agg(n=("gap", "size"), mean_gap=("gap", "mean"), hp3=("hit_rate", "mean"),
                                                             hindsight=("hindsight", "mean"), shortfall=("hindsight_minus_hp3", "mean")).round(3)
    print(summary.to_string())
    rows = []
    for seed in cfg.SEEDS:
        base = load_base_pool("sift")
        for B in cfg.BUDGETS:
            for T in cfg.TARGETS:
                rows += cap_analysis("sift", "diffuse", seed, B, T, base)
    t = pd.DataFrame(rows)
    t.to_csv(OUT_DIR / "popularity_sift_diffuse_cap_analysis.csv", index=False)
    print(t.groupby(["budget", "T"]).mean(numeric_only=True).drop(columns=["seed"]).round(3).to_string())
    return 0

if __name__ == "__main__":
    sys.exit(main())
