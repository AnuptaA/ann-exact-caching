#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Stage F tables from the saved unit files: per-cell mean and sd over seeds of the LSH-arm metrics, the worst
y_min - T over seeds, and the gap to the oracle arm on the same pool (LSH minus oracle, per seed, then averaged).

Run from the thesis root:  python3 weekly/10_26_2026_end_to_end_testing/end_to_end/ee_tables_f.py"""

import sys

import pandas as pd

import ee_config as cfg
import ee_tables as et

UNIT_DIR = cfg.RESULTS_DIR / "stage_f" / "units"
OUT_DIR = cfg.RESULTS_DIR / "stage_f"
METRICS = ["hit_rate", "hit_rate_recall_ge_half", "mean_true_recall_of_hits", "achieved_recall", "frac_below_T",
           "frac_below_T_after_burn_in", "y_min", "y_min_minus_T", "est_signed_error", "est_mae", "over_rate",
           "frac_hits_below_half", "frac_hits_zero_recall", "hindsight_hit_rate_lsh", "hindsight_hit_rate_oracle",
           "learned_cap_units", "train_hit_fraction", "test_hit_fraction", "pool_size", "structure_bytes", "cache_bytes",
           "or_hit_rate", "or_hit_rate_recall_ge_half", "or_mean_true_recall_of_hits", "or_achieved_recall",
           "gap_hit_rate", "gap_hit_rate_recall_ge_half", "gap_mean_true_recall_of_hits", "gap_achieved_recall"]

def add_gaps(rows: pd.DataFrame) -> pd.DataFrame:
    """Per-row LSH minus oracle differences on the same pool, policy and seed."""
    out = rows.copy()
    for m in ("hit_rate", "hit_rate_recall_ge_half", "mean_true_recall_of_hits", "achieved_recall"):
        out[f"gap_{m}"] = out[m] - out[f"or_{m}"]
    return out

def cell_summary_f(rows: pd.DataFrame) -> pd.DataFrame:
    """Mean and sd over seeds per (dataset, regime, budget, T, pool, policy), plus the worst y_min - T over seeds."""
    rows = add_gaps(rows)
    cells = et.cell_summary(rows, METRICS)
    worst = rows.groupby(et.KEYS, sort=False)["y_min_minus_T"].min().rename("y_min_minus_T_worst").reset_index()
    return cells.merge(worst, on=et.KEYS)

def main() -> int:
    rows = et.load_units(UNIT_DIR)
    cells = cell_summary_f(rows)
    cells.to_csv(OUT_DIR / "stage_f_cell_summary.csv.gz", index=False)
    print(f"rows {len(rows)}, cells {len(cells)}, seeds per cell {cells.n_seeds.min()} to {cells.n_seeds.max()}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
