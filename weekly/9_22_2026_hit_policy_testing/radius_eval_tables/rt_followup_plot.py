#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Follow-up B plot: mean error and sd of error vs maximum perturbation angle (stage 4 construction),
with the stage 3 (uniform real queries) values as an extra category.

Run from the thesis root:  python3 .../rt_followup_plot.py"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import rt_chunks
import rt_config
import rt_followup_b as fb
from plot_style import MUTED_TEXT, PRIMARY_TEXT, style_axes

NU = 8
EXTRA_X = 80.0  # position of the "uniform real queries" category on the log axis
TICKS = [2, 5, 10, 20, 40]

def stage3_values(dataset: str) -> pd.DataFrame:
    """Stage 3 per (config, estimate): mean and sd over seeds 0 to 2 at S = 10,000, nu = 8."""
    agg = rt_chunks.read_table(rt_config.RESULTS_DIR / f"{dataset}_stage3_aggregates.csv.gz")
    agg = agg[(agg["S"] == fb.REF_S) & (agg["nu"] == NU) & agg["seed"].isin(fb.SEEDS) &
              agg["config"].isin(rt_config.CONFIG_NAMES)]
    m = fb.metrics_from_sums(agg)
    g = m.groupby(["config", "estimate"])
    out = g[["mean_error_pct", "sd_error_pct"]].mean()
    for c in ("mean_error_pct", "sd_error_pct"):
        out[f"{c}_sd"] = g[c].std(ddof=1)
    return out.reset_index()

def plot_dataset(dataset: str, summary: pd.DataFrame, out_dir: Path) -> Path:
    s = summary[(summary["dataset"] == dataset) & (summary["nu"] == NU)]
    s3 = stage3_values(dataset)
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    for row, (col, ylabel) in enumerate((("mean_error_pct", "mean error (%)"), ("sd_error_pct", "sd of error (%)"))):
        ax = axes[row]
        for cname in rt_config.CONFIG_NAMES:
            color = rt_config.CONFIG_COLORS[cname]
            for est, ls in (("sample+train", "-"), ("sample-only", "--")):
                g = s[(s["config"] == cname) & (s["estimate"] == est)].sort_values("X")
                ax.plot(g["X"], g[col], color=color, ls=ls, lw=1.6, marker="o", ms=3)
                ax.fill_between(g["X"], g[col] - g[f"{col}_sd"].fillna(0), g[col] + g[f"{col}_sd"].fillna(0),
                                color=color, alpha=0.12, linewidth=0)
                p = s3[(s3["config"] == cname) & (s3["estimate"] == est)].iloc[0]
                ax.errorbar([EXTRA_X], [p[col]], yerr=[p[f"{col}_sd"]], color=color, marker="s" if ls == "-" else "D",
                            ms=4, mfc=color if ls == "-" else "white", lw=1.0, capsize=2, ls="none")
        if row == 0:
            ax.axhline(0, color="black", lw=1.0)
        ax.axvline(EXTRA_X / 1.45, color=MUTED_TEXT, lw=0.9, ls=":")
        ax.set_xscale("log")
        ax.set_ylabel(ylabel, color=PRIMARY_TEXT)
        style_axes(ax)
    axes[1].set_xticks(TICKS + [EXTRA_X])
    axes[1].set_xticklabels([str(t) for t in TICKS] + ["uniform\nreal queries"])
    axes[1].minorticks_off()
    axes[1].set_xlim(1.7, EXTRA_X * 1.25)
    axes[1].set_xlabel("maximum perturbation angle (degrees)", color=PRIMARY_TEXT)
    handles = [plt.Line2D([], [], color=rt_config.CONFIG_COLORS[c], lw=2, label=c.replace("_", " "))
               for c in rt_config.CONFIG_NAMES]
    handles += [plt.Line2D([], [], color="#555555", ls="-", label="sample+train"),
                plt.Line2D([], [], color="#555555", ls="--", label="sample-only")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    fig.suptitle(f"{dataset} (d = {rt_config.DIM[dataset]}), S = 10,000, nu = {NU}, 50 anchors, 3 seeds "
                 "(bands: +/-1 sd over seeds; right-most point: stage 3 real queries)", color=PRIMARY_TEXT, fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    out = Path(out_dir) / f"{dataset}_stage4_perturbation_sweep.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out

def main() -> int:
    summary = pd.read_csv(fb.FOLLOWUP_DIR / "B_summary.csv")
    out_dir = rt_config.PLOTS_DIR / "followup"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds in rt_config.DATASETS:
        print(plot_dataset(ds, summary, out_dir))
    return 0

if __name__ == "__main__":
    sys.exit(main())
