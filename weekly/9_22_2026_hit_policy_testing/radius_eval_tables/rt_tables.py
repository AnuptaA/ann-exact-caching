#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Summary tables from saved data (aggregates, per-query rows, covariates)."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import rt_config

NU_SPEARMAN = 8

def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#", low_memory=False, float_precision="round_trip")

def load_stage(results_dir: Path, dataset: str, stage: int) -> dict:
    p = Path(results_dir) / f"{dataset}_stage{stage}"
    out = dict(agg=read_table(Path(f"{p}_aggregates.csv.gz")), rows=read_table(Path(f"{p}_per_query_ref.csv.gz")),
               cov=read_table(Path(f"{p}_covariates.csv.gz")))
    dump = Path(f"{p}_neighbor_dump.csv.gz")
    out["dump"] = read_table(dump) if dump.exists() else None
    return out

#-------------------------------------------------------------------------------

def per_seed_metrics(agg: pd.DataFrame) -> pd.DataFrame:
    """Slope, intercept, mean/sd error (percent), mean distinct count per aggregate row."""
    a = agg.copy()
    n = a["n"]
    var_t = n * a["sum_theta2"] - a["sum_theta"] ** 2
    slope = (n * a["sum_theta_rhat"] - a["sum_theta"] * a["sum_rhat"]) / var_t
    a["slope"] = slope
    a["intercept"] = (a["sum_rhat"] - slope * a["sum_theta"]) / n
    a["mean_error_pct"] = 100 * a["sum_relerr"] / n
    a["sd_error_pct"] = 100 * np.sqrt(np.maximum(a["sum_relerr2"] / n - (a["sum_relerr"] / n) ** 2, 0))
    a["mean_distinct"] = a["sum_distinct_total"] / a["n_distinct"]
    a["mean_train_share"] = a["sum_train_share"] / a["n_distinct"]
    a["theta_mean"] = a["sum_theta"] / n
    a["theta_rel_sd"] = np.sqrt(np.maximum(a["sum_theta2"] / n - a["theta_mean"] ** 2, 0)) / a["theta_mean"]
    return a

def main_table(agg: pd.DataFrame) -> pd.DataFrame:
    """Per (config, S, pop, nu): mean and sd over seeds of each metric, plus the constant and EXACT rows."""
    m = per_seed_metrics(agg)
    theta_rel_sd = float(m[m["config"] == "EXACT"]["theta_rel_sd"].iloc[0])
    m["slope_degenerate"] = m["theta_rel_sd"] < 0.01
    cols = ["slope", "intercept", "mean_error_pct", "sd_error_pct", "mean_distinct", "mean_train_share"]
    keys = ["config", "L", "m", "S", "pop", "nu", "distinct_scope"]
    g = m.groupby(keys, sort=False, dropna=False)
    out = g[cols].mean()
    for c in cols:
        out[f"{c}_sd"] = g[c].std(ddof=1)
    out["n_seeds"] = g.size()
    out["slope_degenerate"] = g["slope_degenerate"].first()
    out = out.reset_index()
    out["theta_rel_sd"] = theta_rel_sd
    return out

#-------------------------------------------------------------------------------

def paired_comparison(rows: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    """Sample+train vs sample-only |relative error| at the reference S, per config and nu."""
    test = cov[cov["role"] == "test"].set_index("query_id")["theta"]
    r = rows[~rows["config"].isin(["EXACT", "CONSTANT"])].copy()
    theta = test.reindex(r["query_id"]).to_numpy()
    r["err_st"] = np.abs(r["r_hat"] - theta) / theta
    r["err_so"] = np.abs(r["r_hat_sample_only"] - theta) / theta
    r["st_better"] = (r["err_st"] < r["err_so"]).astype(float)
    r["diff_pct"] = 100 * (r["err_st"] - r["err_so"])
    per_seed = r.groupby(["config", "nu", "seed"]).agg(frac=("st_better", "mean"), diff=("diff_pct", "mean")).reset_index()
    g = per_seed.groupby(["config", "nu"], sort=False)
    out = g.agg(frac_st_better=("frac", "mean"), frac_sd=("frac", "std"), mean_diff_abs_err_pct=("diff", "mean"),
                diff_sd=("diff", "std")).reset_index()
    out["config"] = pd.Categorical(out["config"], rt_config.CONFIG_NAMES, ordered=True)
    return out.sort_values(["config", "nu"]).reset_index(drop=True)

def spearman_table(rows: pd.DataFrame, cov: pd.DataFrame, nu: int = NU_SPEARMAN) -> pd.DataFrame:
    """Spearman correlation of |relative error| (sample+train) with table_spread and geometry covariates."""
    test = cov[cov["role"] == "test"].set_index("query_id")
    r = rows[(rows["nu"] == nu) & ~rows["config"].isin(["EXACT", "CONSTANT"])].copy()
    c = test.reindex(r["query_id"])
    theta = c["theta"].to_numpy()
    r["abs_err"] = np.abs(r["r_hat"].to_numpy() - theta) / theta
    r["nn1_over_theta"] = (c["nn1"] / c["theta"]).to_numpy()
    r["d50_over_d10"] = (c["d50"] / c["d10"]).to_numpy()
    r["theta"], r["nn1"], r["d10"], r["d50"] = theta, c["nn1"].to_numpy(), c["d10"].to_numpy(), c["d50"].to_numpy()
    recs = []
    for cname in rt_config.CONFIG_NAMES:
        g = r[r["config"] == cname]
        for var in ["table_spread", "theta", "nn1", "d10", "d50", "nn1_over_theta", "d50_over_d10"]:
            ok = g[var].notna() & g["abs_err"].notna()
            if ok.sum() < 3 or g.loc[ok, var].nunique() < 2:
                rho = np.nan
            else:
                rho = spearmanr(g.loc[ok, var], g.loc[ok, "abs_err"])[0]
            recs.append(dict(config=cname, nu=nu, variable=var, spearman=rho, n=int(ok.sum())))
    return pd.DataFrame(recs)

def train_share_table(agg: pd.DataFrame) -> pd.DataFrame:
    """Mean train share of the neighborhood (n_distinct_train / n_distinct_total) per config, nu and S."""
    m = per_seed_metrics(agg)
    m = m[(m["pop"] == "sample_train") & m["mean_train_share"].notna()]
    out = m.groupby(["config", "S", "nu", "distinct_scope"], sort=False).agg(
        train_share=("mean_train_share", "mean"), train_share_sd=("mean_train_share", "std"),
        n_seeds=("seed", "nunique")).reset_index()
    return out

#-------------------------------------------------------------------------------

def md(df: pd.DataFrame, fmt: str = "{:.4g}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else
                                       (fmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
                                       for v in row) + " |")
    return "\n".join(lines)

def build_tables(results_dir: Path, dataset: str, stage: int) -> dict:
    d = load_stage(results_dir, dataset, stage)
    tables = dict(main=main_table(d["agg"]), paired=paired_comparison(d["rows"], d["cov"]),
                  spearman=spearman_table(d["rows"], d["cov"]), train_share=train_share_table(d["agg"]))
    prefix = Path(results_dir) / f"{dataset}_stage{stage}_table"
    for name, df in tables.items():
        df.to_csv(f"{prefix}_{name}.csv", index=False)
    return tables

def headline_text(tables: dict, dataset: str, stage: int, nus=(1, 8, 32), S: int = rt_config.REF_S) -> str:
    m = tables["main"]
    sub = m[(m["S"] == S) & (m["nu"].isin(nus)) & (m["pop"] == "sample_train")]
    cols = ["config", "nu", "mean_error_pct", "mean_error_pct_sd", "sd_error_pct", "slope", "mean_distinct"]
    t = sub[cols].copy()
    t["config"] = pd.Categorical(t["config"], rt_config.CONFIG_NAMES, ordered=True)
    t = t.sort_values(["nu", "config"])
    so = m[(m["S"] == S) & (m["nu"].isin(nus)) & (m["pop"] == "sample_only") & (m["config"].isin(rt_config.CONFIG_NAMES))]
    so = so[["config", "nu", "mean_error_pct"]].rename(columns={"mean_error_pct": "mean_error_pct_sample_only"})
    t = t.merge(so, on=["config", "nu"], how="left")
    const = m[(m["config"] == "CONSTANT") & (m["S"] == S)]
    note = (f"constant radius mean error {const['mean_error_pct'].iloc[0]:.3f}% (sd over seeds "
            f"{const['mean_error_pct_sd'].iloc[0]:.3f}); theta relative sd {100 * m['theta_rel_sd'].iloc[0]:.2f}%")
    return f"**{dataset} stage {stage}, S = {S}**  ({note})\n\n{md(t)}"

#-------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=str(rt_config.RESULTS_DIR))
    ap.add_argument("--stages", default="0,1,2,3,4")
    ap.add_argument("--datasets", default="sift,esci")
    args = ap.parse_args()
    for stage in [int(x) for x in args.stages.split(",")]:
        for ds in args.datasets.split(","):
            if not (Path(args.results_dir) / f"{ds}_stage{stage}.ok").exists():
                print(f"skip {ds} stage {stage}: not complete")
                continue
            print(headline_text(build_tables(Path(args.results_dir), ds, stage), ds, stage), "\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
