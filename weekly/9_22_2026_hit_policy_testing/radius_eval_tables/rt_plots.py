#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""The five plots, drawn from saved tables.

Mean error always means the mean of (r_hat - theta)/theta in percent. The y = x line
and the zero-error line are solid black baselines."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import rt_config
import rt_tables
from plot_style import MUTED_TEXT, PRIMARY_TEXT, style_axes

NU_COLORS = {1: "#1b4f9c", 2: "#2a78d6", 4: "#2a9d5c", 8: "#c9a227", 16: "#eb6834", 32: "#b02a30"}
SAMPLE_COLOR, TRAIN_COLOR = "#2a78d6", "#eb6834"
WINDOW_FRACTION = 0.02
GROUP_512 = ["L1_m512", "L4_m128", "L8_m64"]
GROUP_2048 = ["L1_m2048", "L4_m512", "L8_m256", "L16_m128"]

def stage_title(stage: int, dataset: str, extra: str) -> str:
    kinds = {0: "uniform DB, uniform queries", 1: "uniform DB, real queries", 2: "uniform DB, locality queries",
             3: "real DB, real queries", 4: "real DB, locality queries"}
    return f"stage {stage} ({kinds[stage]}), {dataset} (d = {rt_config.DIM[dataset]}); {extra}"

def s_label(S) -> str:
    return "full (50,000)" if S == rt_config.S_GRID[-1] else f"{S:,}"

def sliding_window(x: np.ndarray, y: np.ndarray, fraction: float = WINDOW_FRACTION) -> tuple:
    """Centered-by-construction sliding window over points sorted by x: window mean of x and of y."""
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    w = max(2, int(round(fraction * len(xs))))
    kernel = np.ones(w) / w
    return np.convolve(xs, kernel, mode="valid"), np.convolve(ys, kernel, mode="valid")

def ref_rows(rows: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    """Per-query rows of the 7 configurations with theta and relative error (percent) attached."""
    r = rows[~rows["config"].isin(["EXACT", "CONSTANT"])].copy()
    test = cov[cov["role"] == "test"].set_index("query_id")
    c = test.reindex(r["query_id"])
    r["theta"] = c["theta"].to_numpy()
    r["rel_err_pct"] = 100 * (r["r_hat"].to_numpy() - r["theta"].to_numpy()) / r["theta"].to_numpy()
    r["nn1"], r["d10"], r["d50"] = c["nn1"].to_numpy(), c["d10"].to_numpy(), c["d50"].to_numpy()
    return r

def constant_level(rows: pd.DataFrame) -> float:
    return float(rows[(rows["config"] == "CONSTANT") & (rows["seed"] == 0)]["r_hat"].iloc[0])

def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path

#-------------------------------------------------------------------------------
# Plot 1

def plot_scatter(rows: pd.DataFrame, cov: pd.DataFrame, dataset: str, stage: int, out_dir: Path) -> Path:
    r = ref_rows(rows, cov)
    r = r[r["seed"] == 0]
    const = constant_level(rows)
    rel_sd = float(r.drop_duplicates("query_id")["theta"].std() / r.drop_duplicates("query_id")["theta"].mean())
    lo = min(r["theta"].min(), r["r_hat"].min())
    hi = max(r["theta"].max(), r["r_hat"].max())
    pad = 0.05 * (hi - lo)
    lim = (lo - pad, hi + pad)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), sharex=True, sharey=True)
    for ax, nu in zip(axes.ravel(), rt_config.NU_GRID):
        g = r[r["nu"] == nu]
        ax.scatter(g["theta"], g["r_hat"], s=3, color="#9a9a9a", alpha=0.12, linewidths=0, rasterized=True)
        wx, wy = sliding_window(g["theta"].to_numpy(), g["r_hat"].to_numpy())
        ax.plot(wx, wy, color="black", lw=3.0, zorder=5)
        ax.plot(lim, lim, color="black", lw=1.0, zorder=4)
        ax.axhline(const, color="black", lw=1.2, ls=":", zorder=4)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"nu = {nu}", color=PRIMARY_TEXT)
        style_axes(ax)
    for ax in axes[1]:
        ax.set_xlabel("true theta", color=PRIMARY_TEXT)
    for ax in axes[:, 0]:
        ax.set_ylabel("r_hat", color=PRIMARY_TEXT)
    handles = [plt.Line2D([], [], marker="o", ls="", color="#9a9a9a", alpha=0.6, label="queries (all 7 configs pooled)"),
               plt.Line2D([], [], color="black", lw=3, label="pooled mean (2% window)"),
               plt.Line2D([], [], color="black", lw=1, label="y = x"),
               plt.Line2D([], [], color="black", lw=1.2, ls=":", label="constant radius")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    note = f"theta relative sd = {100 * rel_sd:.2f}%: read the level only. " if stage <= 2 else ""
    fig.suptitle(stage_title(stage, dataset, f"scatter, S = 10,000, sample+train, seed 0. {note}"), color=PRIMARY_TEXT)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    return _save(fig, Path(out_dir) / f"{dataset}_stage{stage}_plot1_scatter.png")

#-------------------------------------------------------------------------------
# Plot 2

def plot_mean_error_vs_nu(agg: pd.DataFrame, dataset: str, stage: int, out_dir: Path) -> Path:
    m = rt_tables.main_table(agg)
    m = m[m["config"].isin(rt_config.CONFIG_NAMES)]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey="row")
    for j, S in enumerate(rt_config.S_GRID):
        for i in range(2):
            ax = axes[i, j]
            for cname in rt_config.CONFIG_NAMES:
                color = rt_config.CONFIG_COLORS[cname]
                for pop, ls in (("sample_train", "-"), ("sample_only", "--")):
                    g = m[(m["config"] == cname) & (m["S"] == S) & (m["pop"] == pop)].sort_values("nu")
                    x = g["nu"] if i == 0 else g["mean_distinct"]
                    ax.plot(x, g["mean_error_pct"], color=color, ls=ls, lw=1.6)
                    if pop == "sample_train" and g["n_seeds"].min() > 1:
                        ax.fill_between(x, g["mean_error_pct"] - g["mean_error_pct_sd"].fillna(0),
                                        g["mean_error_pct"] + g["mean_error_pct_sd"].fillna(0), color=color, alpha=0.15,
                                        linewidth=0)
            ax.axhline(0, color="black", lw=1.0)
            ax.set_xscale("log")
            style_axes(ax)
            if i == 0:
                ax.set_title(f"S = {s_label(S)}", color=PRIMARY_TEXT)
                ax.set_xlabel("nu (nearest labeled points per table)", color=PRIMARY_TEXT)
            else:
                ax.set_xlabel("mean distinct labeled points in the neighborhood", color=PRIMARY_TEXT)
                if S != rt_config.REF_S:
                    ax.text(0.02, 0.03, "distinct counts: seed 0 only", transform=ax.transAxes, fontsize=8,
                            color=MUTED_TEXT)
        axes[0, 0].set_ylabel("mean error (%)", color=PRIMARY_TEXT)
        axes[1, 0].set_ylabel("mean error (%)", color=PRIMARY_TEXT)
    handles = [plt.Line2D([], [], color=rt_config.CONFIG_COLORS[c], lw=2, label=c.replace("_", " ")) for c in
               rt_config.CONFIG_NAMES]
    handles += [plt.Line2D([], [], color="#555555", ls="-", label="sample+train (band: +/-1 sd over seeds)"),
                plt.Line2D([], [], color="#555555", ls="--", label="sample-only")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    rel = m[m["config"] == m["config"].iloc[0]]["theta_rel_sd"].iloc[0]
    note = f" theta relative sd = {100 * rel:.2f}%." if stage <= 2 else ""
    fig.suptitle(stage_title(stage, dataset, "mean error of r_hat vs nu (top) and vs neighborhood size (bottom)." + note),
                 color=PRIMARY_TEXT)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    return _save(fig, Path(out_dir) / f"{dataset}_stage{stage}_plot2_mean_error_vs_nu.png")

#-------------------------------------------------------------------------------
# Plot 3

def plot_bias_profile(rows: pd.DataFrame, cov: pd.DataFrame, dataset: str, stage: int, out_dir: Path) -> Path:
    r = ref_rows(rows, cov)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    for ax, nu in zip(axes.ravel(), rt_config.NU_GRID):
        for cname in rt_config.CONFIG_NAMES:
            g = r[(r["config"] == cname) & (r["nu"] == nu)]
            ys, x = [], None
            for _, gs in g.groupby("seed"):
                gs = gs.sort_values(["theta", "query_id"])
                x, y = sliding_window(gs["theta"].to_numpy(), gs["rel_err_pct"].to_numpy())
                ys.append(y)
            ys = np.array(ys)
            mean, sd = ys.mean(axis=0), (ys.std(axis=0, ddof=1) if len(ys) > 1 else np.zeros(ys.shape[1]))
            color = rt_config.CONFIG_COLORS[cname]
            ax.plot(x, mean, color=color, lw=1.6)
            ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.15, linewidth=0)
        ax.axhline(0, color="black", lw=1.0)
        ax.set_title(f"nu = {nu}", color=PRIMARY_TEXT)
        style_axes(ax)
    for ax in axes[1]:
        ax.set_xlabel("true theta (2% sliding window)", color=PRIMARY_TEXT)
    for ax in axes[:, 0]:
        ax.set_ylabel("mean error (%)", color=PRIMARY_TEXT)
    handles = [plt.Line2D([], [], color=rt_config.CONFIG_COLORS[c], lw=2, label=c.replace("_", " "))
               for c in rt_config.CONFIG_NAMES]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False)
    extra = "bias profile, S = 10,000, sample+train, band +/-1 sd over seeds."
    if stage == 4:
        extra += " Queries form 50 radius clumps (one per anchor)."
    fig.suptitle(stage_title(stage, dataset, extra), color=PRIMARY_TEXT)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    return _save(fig, Path(out_dir) / f"{dataset}_stage{stage}_plot3_bias_profile.png")

#-------------------------------------------------------------------------------
# Plot 4

def equal_count_edges(x: np.ndarray, n_bins: int) -> np.ndarray:
    return np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))

def plot_neighbor_error(dump: pd.DataFrame, cov: pd.DataFrame, dataset: str, stage: int, out_dir: Path) -> Path:
    test_theta = cov[cov["role"] == "test"].set_index("query_id")["theta"]
    d = dump.copy()
    d["theta_q"] = test_theta.reindex(d["query_id"]).to_numpy()
    d = d[(d["theta_neighbor"] > 0) & (d["theta_q"] > 0)]
    d["log_ratio_pct"] = 100 * np.log(d["theta_neighbor"].to_numpy() / d["theta_q"].to_numpy())
    fig, axes = plt.subplots(2, 7, figsize=(26, 8), sharex=False)
    for k, (L, m) in enumerate(rt_config.CONFIGS):
        cname = rt_config.config_name(L, m)
        g = d[d["config"] == cname].copy()
        g["x"] = g["hamming"] / m
        edges = equal_count_edges(g["x"].to_numpy(), 20)
        g["bin"] = np.clip(np.digitize(g["x"], edges[1:-1], right=True), 0, len(edges) - 2)
        centers = g.groupby("bin")["x"].mean()
        top, bot = axes[0, k], axes[1, k]
        counts = {}
        for name, flag, color in (("sample neighbors", 0, SAMPLE_COLOR), ("train neighbors", 1, TRAIN_COLOR)):
            s = g[g["from_train"] == flag].groupby("bin")["log_ratio_pct"].agg(["mean", "std", "count"])
            s = s[s["count"] >= 2]
            x = centers.reindex(s.index).to_numpy()
            se_mean = s["std"] / np.sqrt(s["count"])
            se_sd = s["std"] / np.sqrt(2 * (s["count"] - 1))
            top.plot(x, s["mean"], color=color, lw=1.6, label=name)
            top.fill_between(x, s["mean"] - se_mean, s["mean"] + se_mean, color=color, alpha=0.2, linewidth=0)
            bot.plot(x, s["std"], color=color, lw=1.6)
            bot.fill_between(x, s["std"] - se_sd, s["std"] + se_sd, color=color, alpha=0.2, linewidth=0)
            counts[name] = (centers.index.map(lambda b: int(((g["bin"] == b) & (g["from_train"] == flag)).sum())).to_numpy(),
                            color)
        top.axhline(0, color="black", lw=1.0)
        bot.axhline(0, color="black", lw=1.0)
        top.set_title(cname.replace("_", " "), color=PRIMARY_TEXT)
        bot.set_xlabel("neighbor Hamming distance / m", color=PRIMARY_TEXT)
        style_axes(top)
        style_axes(bot)
        inset = top.inset_axes([0.58, 0.60, 0.38, 0.28])
        for name, (cnt, color) in counts.items():
            inset.step(centers.to_numpy(), np.maximum(cnt, 0.5), where="mid", color=color, lw=1.0)
        inset.set_yscale("log")
        inset.tick_params(labelsize=6, colors=MUTED_TEXT)
        inset.set_title("neighbors per bin", fontsize=6, color=MUTED_TEXT)
    axes[0, 0].set_ylabel("mean of log(theta_nb / theta_q) (%)", color=PRIMARY_TEXT)
    axes[1, 0].set_ylabel("sd of log(theta_nb / theta_q) (%)", color=PRIMARY_TEXT)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(stage_title(stage, dataset, "neighbor error vs Hamming distance, seed 0, S = 10,000, ranks <= 32, "
                                              "20 equal-count bins, bands +/-1 standard error"), color=PRIMARY_TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, Path(out_dir) / f"{dataset}_stage{stage}_plot4_neighbor_error_vs_hamming.png")

#-------------------------------------------------------------------------------
# Plot 5

def plot_error_vs_geometry(rows: pd.DataFrame, cov: pd.DataFrame, dataset: str, stage: int, out_dir: Path) -> Path:
    r = ref_rows(rows, cov)
    r["nn1_over_theta"] = r["nn1"] / r["theta"]
    r["d50_over_d10"] = r["d50"] / r["d10"]
    xs = [("nn1_over_theta", "nearest database distance / theta"),
          ("d50_over_d10", "(50th nearest distance) / (10th nearest distance)")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for i, (group, label) in enumerate(((GROUP_512, "512 total bits"), (GROUP_2048, "2048 total bits"))):
        g = r[r["config"].isin(group)]
        for j, (var, xlabel) in enumerate(xs):
            ax = axes[i, j]
            edges = equal_count_edges(g.drop_duplicates("query_id")[var].to_numpy(), 10)
            bins = np.clip(np.digitize(g[var], edges[1:-1], right=True), 0, len(edges) - 2)
            g = g.assign(bin=bins)
            centers = g.groupby("bin")[var].mean()
            for nu in rt_config.NU_GRID:
                y = g[g["nu"] == nu].groupby("bin")["rel_err_pct"].mean()
                ax.plot(centers.reindex(y.index), y, color=NU_COLORS[nu], lw=1.6, marker="o", ms=3, label=f"nu = {nu}")
            ax.axhline(0, color="black", lw=1.0)
            ax.set_xlabel(xlabel, color=PRIMARY_TEXT)
            ax.set_title(label + " (" + ", ".join(c.replace("_", " ") for c in group) + " pooled)", fontsize=9,
                         color=PRIMARY_TEXT)
            style_axes(ax)
        axes[i, 0].set_ylabel("mean error (%)", color=PRIMARY_TEXT)
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=8)
    fig.suptitle(stage_title(stage, dataset, "error vs geometry, S = 10,000, sample+train, 10 equal-count bins"),
                 color=PRIMARY_TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, Path(out_dir) / f"{dataset}_stage{stage}_plot5_error_vs_geometry.png")
