#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Follow-up A: calibration versus compression, from saved per-query rows.

Per (dataset, stage, config, nu, estimate, seed): forward slope/intercept (r_hat on theta), reverse
slope/intercept (theta on r_hat), R squared, mean error (%), sd of error (%). Plus a calibration table of 10
equal-count bins of r_hat (formed per seed).

Run from the thesis root:  python3 .../rt_followup_a.py"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import rt_config
import rt_tables

FOLLOWUP_DIR = rt_config.RESULTS_DIR / "followup"
FOLLOWUP_DIR.mkdir(parents=True, exist_ok=True)
(rt_config.PLOTS_DIR / "followup").mkdir(parents=True, exist_ok=True)

STAGES = [3, 4]
NUS = [1, 8, 32]
ESTIMATES = {"sample_train": "r_hat", "sample_only": "r_hat_sample_only"}
GROUP = ["dataset", "stage", "config", "nu", "estimate", "seed"]
N_BINS = 10

def load_long(results_dir: Path, dataset: str, stage: int) -> pd.DataFrame:
    """Per-query rows of the 7 configs at the reference S with theta attached, one row per estimate."""
    d = rt_tables.load_stage(results_dir, dataset, stage)
    rows = d["rows"]
    rows = rows[~rows["config"].isin(["EXACT", "CONSTANT"]) & rows["nu"].isin(NUS)]
    theta = d["cov"][d["cov"]["role"] == "test"].set_index("query_id")["theta"]
    parts = []
    for name, col in ESTIMATES.items():
        p = rows[["query_id", "seed", "config", "nu", col]].rename(columns={col: "r"})
        p["estimate"] = name
        parts.append(p)
    out = pd.concat(parts, ignore_index=True)
    out["theta"] = theta.reindex(out["query_id"]).to_numpy()
    assert out["theta"].notna().all() and out["r"].notna().all()
    out["dataset"], out["stage"] = dataset, stage
    return out

def fit_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-group statistics from centered sums (values are shifted by global means to avoid cancellation)."""
    x0, y0 = df["theta"].mean(), df["r"].mean()
    x, y = df["theta"].to_numpy() - x0, df["r"].to_numpy() - y0
    rel = (df["r"].to_numpy() - df["theta"].to_numpy()) / df["theta"].to_numpy()
    w = df[GROUP].copy()
    w["x"], w["y"], w["xx"], w["yy"], w["xy"] = x, y, x * x, y * y, x * y
    w["rel"], w["rel2"] = rel, rel * rel
    g = w.groupby(GROUP, sort=False)
    s = g[["x", "y", "xx", "yy", "xy", "rel", "rel2"]].sum()
    n = g.size()
    mx, my = s["x"] / n, s["y"] / n
    cxx, cyy, cxy = s["xx"] / n - mx ** 2, s["yy"] / n - my ** 2, s["xy"] / n - mx * my
    out = pd.DataFrame(index=s.index)
    out["n"] = n
    out["forward_slope"] = cxy / cxx
    out["forward_intercept"] = (my + y0) - out["forward_slope"] * (mx + x0)
    out["reverse_slope"] = cxy / cyy
    out["reverse_intercept"] = (mx + x0) - out["reverse_slope"] * (my + y0)
    out["r_squared"] = cxy ** 2 / (cxx * cyy)
    out["mean_error_pct"] = 100 * s["rel"] / n
    out["sd_error_pct"] = 100 * np.sqrt(np.maximum(s["rel2"] / n - (s["rel"] / n) ** 2, 0))
    return out.reset_index()

def summarize_over_seeds(per_seed: pd.DataFrame) -> pd.DataFrame:
    cols = ["forward_slope", "forward_intercept", "reverse_slope", "reverse_intercept", "r_squared",
            "mean_error_pct", "sd_error_pct"]
    keys = [c for c in GROUP if c != "seed"]
    g = per_seed.groupby(keys, sort=False)
    out = g[cols].mean()
    for c in cols:
        out[f"{c}_sd"] = g[c].std(ddof=1)
    out["n_seeds"] = g.size()
    return out.reset_index()

def calibration_table(df: pd.DataFrame, n_bins: int = N_BINS) -> pd.DataFrame:
    """Per seed: bins of equal count by rank of r (ties broken by query_id); mean r and mean theta per bin;
    then averaged over seeds."""
    d = df.sort_values(GROUP + ["r", "query_id"], kind="stable").reset_index(drop=True)
    pos = d.groupby(GROUP, sort=False).cumcount().to_numpy()
    size = d.groupby(GROUP, sort=False)["r"].transform("size").to_numpy()
    d["bin"] = pos * n_bins // size
    per_seed = d.groupby(GROUP + ["bin"], sort=False).agg(mean_r_hat=("r", "mean"), mean_theta=("theta", "mean"),
                                                         n=("r", "size")).reset_index()
    keys = [c for c in GROUP if c != "seed"] + ["bin"]
    out = per_seed.groupby(keys, sort=False).agg(mean_r_hat=("mean_r_hat", "mean"), mean_theta=("mean_theta", "mean"),
                                                 n_per_bin=("n", "first"), n_seeds=("n", "size")).reset_index()
    return out

def nu8_table(summary: pd.DataFrame) -> pd.DataFrame:
    s = summary[summary["nu"] == 8]
    rows = []
    for (stage, dataset, config), g in s.groupby(["stage", "dataset", "config"], sort=False):
        row = dict(stage=stage, dataset=dataset, config=config)
        for est, tag in (("sample_train", "ST"), ("sample_only", "SO")):
            r = g[g["estimate"] == est].iloc[0]
            row.update({f"{tag} fwd": r["forward_slope"], f"{tag} rev": r["reverse_slope"], f"{tag} R2": r["r_squared"],
                        f"{tag} err%": r["mean_error_pct"]})
        rows.append(row)
    out = pd.DataFrame(rows)
    out["config"] = pd.Categorical(out["config"], rt_config.CONFIG_NAMES, ordered=True)
    return out.sort_values(["stage", "dataset", "config"], kind="stable").reset_index(drop=True)

def main() -> int:
    results_dir = rt_config.RESULTS_DIR
    per_seed, calib = [], []
    for stage in STAGES:
        for ds in rt_config.DATASETS:
            df = load_long(results_dir, ds, stage)
            per_seed.append(fit_stats(df))
            calib.append(calibration_table(df))
            print(f"{ds} stage {stage}: {len(df)} rows", file=sys.stderr)
    per_seed = pd.concat(per_seed, ignore_index=True)
    summary = summarize_over_seeds(per_seed)
    calib = pd.concat(calib, ignore_index=True)
    per_seed.to_csv(FOLLOWUP_DIR / "A_fit_per_seed.csv", index=False)
    summary.to_csv(FOLLOWUP_DIR / "A_fit_summary.csv", index=False)
    calib.to_csv(FOLLOWUP_DIR / "A_calibration.csv", index=False)
    t = nu8_table(summary)
    t.to_csv(FOLLOWUP_DIR / "A_table_nu8.csv", index=False)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0

if __name__ == "__main__":
    sys.exit(main())
