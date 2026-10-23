#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Data layer for the report PDFs: loaders and table builders from saved data only.

error of one query = 100 * (r_hat - theta) / theta (percent); mean error is its mean over the test queries."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import rt_config
import rt_tables

REF_S = rt_config.REF_S
NU_PLOT = rt_config.NU_GRID
CONSTANT_LABEL = "constant radius"
DEGENERATE_CV = 0.01

#-------------------------------------------------------------------------------

def pooled_by_nearest_pct(x: np.ndarray, y: np.ndarray, frac: float = 0.01) -> tuple:
    """Sort by x; at every position average x and y over the frac of points nearest in x
    (half-width ceil(frac/2 * n) on each side, truncated at the ends). Returns (mean x, mean y)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    n = len(xs)
    h = math.ceil(frac / 2.0 * n)
    idx = np.arange(n)
    lo, hi = np.maximum(idx - h, 0), np.minimum(idx + h, n - 1) + 1
    cx = np.concatenate([[0.0], np.cumsum(xs)])
    cy = np.concatenate([[0.0], np.cumsum(ys)])
    cnt = hi - lo
    return (cx[hi] - cx[lo]) / cnt, (cy[hi] - cy[lo]) / cnt

def pooled_error_profile(theta: np.ndarray, err: np.ndarray, query_id: np.ndarray) -> tuple:
    """Pooled-by-nearest-1% profile with a deterministic order (ties in theta broken by query id)."""
    order = np.lexsort((query_id, theta))
    return pooled_by_nearest_pct(theta[order], err[order])

#-------------------------------------------------------------------------------

def load_stage_data(results_dir: Path, dataset: str, stage: int) -> dict:
    p = Path(results_dir) / f"{dataset}_stage{stage}"
    agg = rt_tables.read_table(Path(f"{p}_aggregates.csv.gz"))
    rows = rt_tables.read_table(Path(f"{p}_per_query_ref.csv.gz"))
    cov = rt_tables.read_table(Path(f"{p}_covariates.csv.gz"))
    summary = json.loads(Path(f"{p}_summary.json").read_text())
    return dict(dataset=dataset, stage=stage, agg=agg, rows=rows, cov=cov, summary=summary,
                metrics=rt_tables.per_seed_metrics(agg), queries=query_frame(rows, cov),
                constant=constant_levels(rows))

def query_frame(rows: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    """Per-query rows of the 7 configurations with theta and both errors (percent)."""
    r = rows[~rows["config"].isin(["EXACT", "CONSTANT"])].copy()
    theta = cov[cov["role"] == "test"].set_index("query_id")["theta"]
    r["theta"] = theta.reindex(r["query_id"]).to_numpy()
    assert r["theta"].notna().all()
    r["err_st"] = 100 * (r["r_hat"] - r["theta"]) / r["theta"]
    r["err_so"] = 100 * (r["r_hat_sample_only"] - r["theta"]) / r["theta"]
    return r

def constant_levels(rows: pd.DataFrame) -> dict:
    """seed -> constant radius level (median theta of the sample points) at the reference S."""
    c = rows[rows["config"] == "CONSTANT"].groupby("seed")["r_hat"].first()
    return {int(k): float(v) for k, v in c.items()}

def theta_relative_sd(metrics: pd.DataFrame) -> float:
    return float(metrics[metrics["config"] == "EXACT"]["theta_rel_sd"].iloc[0])

def slope_degenerate(metrics: pd.DataFrame) -> bool:
    return theta_relative_sd(metrics) < DEGENERATE_CV

#-------------------------------------------------------------------------------
# Tables

def _seed_mean(df: pd.DataFrame, col: str) -> float:
    return float(df[col].mean())

def table_main(metrics: pd.DataFrame, nu: int, S: int = REF_S) -> list:
    """T1 rows: dict(label, mean_err, sd_seeds, sd_err, slope, distinct, so_mean_err); constant row last."""
    out = []
    m = metrics
    for cname in rt_config.CONFIG_NAMES:
        st = m[(m["config"] == cname) & (m["S"] == S) & (m["pop"] == "sample_train") & (m["nu"] == nu)]
        so = m[(m["config"] == cname) & (m["S"] == S) & (m["pop"] == "sample_only") & (m["nu"] == nu)]
        out.append(dict(label=cname, mean_err=_seed_mean(st, "mean_error_pct"),
                        sd_seeds=float(st["mean_error_pct"].std(ddof=1)), sd_err=_seed_mean(st, "sd_error_pct"),
                        slope=_seed_mean(st, "slope"), distinct=_seed_mean(st, "mean_distinct"),
                        so_mean_err=_seed_mean(so, "mean_error_pct")))
    c = m[(m["config"] == "CONSTANT") & (m["S"] == S)]
    out.append(dict(label=CONSTANT_LABEL, mean_err=_seed_mean(c, "mean_error_pct"),
                    sd_seeds=float(c["mean_error_pct"].std(ddof=1)), sd_err=_seed_mean(c, "sd_error_pct"),
                    slope=_seed_mean(c, "slope"), distinct=None, so_mean_err=None))
    return out

def table_sample_size(metrics: pd.DataFrame, nu: int = 8) -> list:
    """T2 rows: dict(label, mean_err=[per S], sd_err=[per S]) for sample+train."""
    out = []
    for cname in rt_config.CONFIG_NAMES:
        me, sd = [], []
        for S in rt_config.S_GRID:
            g = metrics[(metrics["config"] == cname) & (metrics["S"] == S) & (metrics["pop"] == "sample_train") &
                        (metrics["nu"] == nu)]
            me.append(_seed_mean(g, "mean_error_pct"))
            sd.append(_seed_mean(g, "sd_error_pct"))
        out.append(dict(label=cname, mean_err=me, sd_err=sd))
    return out

def table_train_points(queries: pd.DataFrame, nu: int = 8) -> list:
    """T3 rows: dict(label, train_share, pct_smaller, mean_change_pp), per seed then averaged over seeds.
    mean_change_pp = mean(|error sample-only| - |error sample+train|), positive when train points helped."""
    q = queries[queries["nu"] == nu].copy()
    q["share"] = q["n_distinct_train"] / q["n_distinct_total"]
    q["smaller"] = (q["err_st"].abs() < q["err_so"].abs()).astype(float)
    q["change"] = q["err_so"].abs() - q["err_st"].abs()
    per_seed = q.groupby(["config", "seed"]).agg(share=("share", "mean"), smaller=("smaller", "mean"),
                                                 change=("change", "mean")).reset_index()
    g = per_seed.groupby("config").mean(numeric_only=True)
    return [dict(label=c, train_share=float(g.loc[c, "share"]), pct_smaller=100 * float(g.loc[c, "smaller"]),
                 mean_change_pp=float(g.loc[c, "change"])) for c in rt_config.CONFIG_NAMES]

#-------------------------------------------------------------------------------
# Line data for the mean-error-vs-nu figure

def mean_error_lines(metrics: pd.DataFrame, pop: str, S: int) -> dict:
    """config -> (mean over seeds, sd over seeds) arrays over NU_PLOT of the mean error (percent)."""
    out = {}
    for cname in rt_config.CONFIG_NAMES:
        g = metrics[(metrics["config"] == cname) & (metrics["S"] == S) & (metrics["pop"] == pop) &
                    metrics["nu"].isin(NU_PLOT)]
        p = g.pivot(index="nu", columns="seed", values="mean_error_pct").loc[NU_PLOT]
        out[cname] = (p.mean(axis=1).to_numpy(), p.std(axis=1, ddof=1).to_numpy())
    return out
