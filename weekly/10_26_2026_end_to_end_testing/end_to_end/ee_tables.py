#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Stage D tables from the saved unit files: per-cell mean and sd over seeds, and paired differences across seeds for
pool comparisons, policy comparisons (same pool) and the five named triplets.

Run from the thesis root:  python3 weekly/10_26_2026_end_to_end_testing/end_to_end/ee_tables.py"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import ee_config as cfg

UNIT_DIR = cfg.RESULTS_DIR / "stage_d" / "units"
OUT_DIR = cfg.RESULTS_DIR / "stage_d"
KEYS = ["dataset", "regime", "budget", "T", "pool", "policy"]
METRICS = ["hit_rate", "hit_rate_recall_ge_half", "achieved_recall", "y_min", "mean_true_recall_of_hits",
           "p10_hit_recall", "frac_hits_below_half", "frac_hits_zero_recall", "hindsight_hit_rate_oracle",
           "pool_size", "learned_cap_units", "train_hit_fraction", "test_hit_fraction"]
PAIR_METRICS = ["hit_rate", "hit_rate_recall_ge_half", "achieved_recall", "mean_true_recall_of_hits"]
# The five named triplets: (label, policy, pool). SEL4 uses m = 1.0 here; m = 0.5 and 1.5 are in the full tables.
TRIPLETS = [("T1 HP1 Ball SEL1", "HP1", "ball+SEL1"), ("T2 HP3 GroupUnion SEL2", "HP3", "group_union+SEL2"),
            ("T3 HP3 Own SEL5", "HP3", "own+SEL5"), ("T4 HP3 Ball SEL3", "HP3", "ball+SEL3"),
            ("T5 HP3 GroupUnion SEL4(m=1.0)", "HP3", "group_union+SEL4(m=1.0)")]
BASELINES = ["popularity", "random_own", "threshold_baseline", "no_cache"]

def load_units(directory: Path = UNIT_DIR) -> pd.DataFrame:
    parts = [pd.read_csv(p, comment="#", low_memory=False) for p in sorted(Path(directory).glob("*.csv.gz"))]
    return pd.concat(parts, ignore_index=True)

def cell_summary(rows: pd.DataFrame, metrics: list = METRICS) -> pd.DataFrame:
    """Mean and sd (ddof = 1) over seeds of every metric per (dataset, regime, budget, T, pool, policy)."""
    metrics = [m for m in metrics if m in rows.columns]
    g = rows.groupby(KEYS, sort=False)
    out = g[metrics].mean()
    for m in metrics:
        out[f"{m}_sd"] = g[m].std(ddof=1)
    out["n_seeds"] = g["seed"].nunique()
    return out.reset_index()

def _paired(a: pd.DataFrame, b: pd.DataFrame, join: list, metrics: list, label_a: str, label_b: str) -> pd.DataFrame:
    m = a.merge(b, on=join + ["seed"], suffixes=("_a", "_b"))
    recs = []
    for key, g in m.groupby(join, sort=False):
        rec = dict(zip(join, key if isinstance(key, tuple) else (key,)), a=label_a, b=label_b, n_seeds=len(g))
        for metric in metrics:
            d = (g[f"{metric}_a"] - g[f"{metric}_b"]).to_numpy(dtype=float)
            sd = d.std(ddof=1) if len(d) > 1 else np.nan
            rec[f"{metric}_diff"] = d.mean()
            rec[f"{metric}_diff_sd"] = sd
            rec[f"{metric}_n_pos"] = int((d > 0).sum())
            rec[f"{metric}_n_neg"] = int((d < 0).sum())
            rec[f"{metric}_t"] = d.mean() / (sd / np.sqrt(len(d))) if sd and sd > 0 else np.nan
        recs.append(rec)
    return pd.DataFrame(recs)

def pool_pairs(rows: pd.DataFrame, pairs: list, metrics: list = PAIR_METRICS) -> pd.DataFrame:
    """Per-seed differences (pool a minus pool b) under every policy, paired by (dataset, regime, budget, T, seed)."""
    join = ["dataset", "regime", "budget", "T", "policy"]
    cols = join + ["seed"] + metrics
    parts = []
    for pa, pb in pairs:
        a = rows.loc[rows.pool == pa, cols]
        b = rows.loc[rows.pool == pb, cols]
        if len(a) and len(b):
            parts.append(_paired(a, b, join, metrics, pa, pb))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

def default_pool_pairs(pools: list) -> list:
    selection = [p for p in pools if p not in BASELINES]
    pairs = [(p, b) for p in selection for b in ("popularity", "random_own", "threshold_baseline")]
    pairs += [("popularity", "random_own"), ("popularity", "threshold_baseline"), ("popularity", "no_cache")]
    for kind in ("own", "group_union", "ball"):  # same proposal, different score
        pairs += [(f"{kind}+SEL5", f"{kind}+SEL3"), (f"{kind}+SEL2", f"{kind}+SEL3"), (f"{kind}+SEL5", f"{kind}+SEL2")]
    for score in ("SEL1", "SEL2", "SEL3", "SEL5"):  # same score, different proposal
        pairs += [(f"group_union+{score}", f"own+{score}"), (f"ball+{score}", f"own+{score}"),
                  (f"ball+{score}", f"group_union+{score}")]
    return pairs

def policy_pairs(rows: pd.DataFrame, metrics: list = PAIR_METRICS) -> pd.DataFrame:
    """HP3 minus HP2, HP3f minus HP3 and HP2 minus HP1 on the same pool, paired across seeds."""
    join = ["dataset", "regime", "budget", "T", "pool"]
    cols = join + ["seed"] + metrics
    parts = []
    for pa, pb in (("HP3", "HP2"), ("HP3f", "HP3"), ("HP2", "HP1"), ("HP3f", "HP2")):
        parts.append(_paired(rows.loc[rows.policy == pa, cols], rows.loc[rows.policy == pb, cols], join, metrics, pa, pb))
    return pd.concat(parts, ignore_index=True)

def triplet_pairs(rows: pd.DataFrame, metrics: list = PAIR_METRICS) -> pd.DataFrame:
    """Pairwise differences among the five named triplets (each with its own policy), and each versus
    Popularity + HP3, paired across seeds."""
    join = ["dataset", "regime", "budget", "T"]
    cols = join + ["seed"] + metrics
    sel = {label: rows.loc[(rows.policy == pol) & (rows.pool == pool), cols] for label, pol, pool in TRIPLETS}
    sel["Popularity HP3"] = rows.loc[(rows.policy == "HP3") & (rows.pool == "popularity"), cols]
    parts = [_paired(sel[a], sel[b], join, metrics, a, b) for a, b in combinations([t[0] for t in TRIPLETS], 2)]
    parts += [_paired(sel[a], sel["Popularity HP3"], join, metrics, a, "Popularity HP3") for a, _, _ in TRIPLETS]
    return pd.concat(parts, ignore_index=True)

def main() -> int:
    rows = load_units()
    cells = cell_summary(rows)
    cells.to_csv(OUT_DIR / "stage_d_cell_summary.csv.gz", index=False)
    pools = sorted(rows.pool.unique())
    pool_p = pool_pairs(rows, default_pool_pairs(pools))
    pool_p.to_csv(OUT_DIR / "stage_d_paired_pools.csv.gz", index=False)
    pol_p = policy_pairs(rows)
    pol_p.to_csv(OUT_DIR / "stage_d_paired_policies.csv.gz", index=False)
    trip_p = triplet_pairs(rows)
    trip_p.to_csv(OUT_DIR / "stage_d_paired_triplets.csv.gz", index=False)
    print(f"rows {len(rows)}, cells {len(cells)}, paired pools {len(pool_p)}, policies {len(pol_p)}, triplets {len(trip_p)}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
