#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Plots from the saved stage D (oracle) and stage F (LSH) cell tables, the stage E unit files and the two replayed
paths of ee_plot_paths.py. Nothing is recomputed here. PNGs go to plots/ and PDFs to pdf/; figures that show lambda = 1.0
go to the sanity subfolder of each. Oracle lines are solid, LSH lines dashed, bands are +/- 1 sd over seeds.

Run from the thesis root:  python3 weekly/10_26_2026_end_to_end_testing/end_to_end/ee_plots.py [--only A,B,...]"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import ee_config as cfg  # noqa: E402
import ee_lsh_bench as lb  # noqa: E402
from plot_style import MUTED_TEXT, PRIMARY_TEXT, style_axes  # noqa: E402

STAGE_D = cfg.RESULTS_DIR / "stage_d" / "stage_d_cell_summary.csv.gz"
STAGE_F = cfg.RESULTS_DIR / "stage_f" / "stage_f_cell_summary.csv.gz"
PATHS = cfg.RESULTS_DIR / "stage_f" / "paths_seed0.npz"

BUDGET_COLORS = {500: "#E69F00", 2500: "#0072B2", 5000: "#009E73"}
BUDGET_LABELS = {500: "1% (500)", 2500: "5% (2,500)", 5000: "10% (5,000)"}
T_MARKERS = {0.8: "o", 0.9: "s", 0.95: "^"}
TS = (0.8, 0.9, 0.95)
GREY = "#8A8A8A"
POLICY_COLORS = {"HP1": "#CC79A7", "HP2": "#D55E00", "HP3": "#332288", "HP3f": "#44AA99"}
TRIPLETS = (("T1", "HP1", "ball+SEL1", "Ball SEL1"), ("T2", "HP3", "group_union+SEL2", "Group Union SEL2"),
            ("T3", "HP3", "own+SEL5", "Own SEL5"), ("T4", "HP3", "ball+SEL3", "Ball SEL3"),
            ("T5", "HP3", "group_union+SEL4(m=1.0)", "Group Union SEL4 (m = 1.0)"))
POOLS_B = (("group_union+SEL2", "Group Union SEL2", "#0072B2"), ("own+SEL5", "Own SEL5", "#D55E00"),
           ("group_union+SEL4(m=1.0)", "Group Union SEL4 (m = 1.0)", "#009E73"), ("ball+SEL3", "Ball SEL3", "#CC79A7"),
           ("popularity", "Popularity", "#E69F00"), ("random_own", "Random Own", "#56B4E9"),
           ("threshold_baseline", "threshold_baseline", "#8C564B"), ("no_cache", "No cache", "#000000"))
DATASET_LABEL = {"sift": "SIFT", "esci": "ESCI"}
REGIME_LABEL = {"lam0": "lambda = 0", "lam0.5": "lambda = 0.5", "lam1": "lambda = 1.0",
                "diffuse": "diffuse (500 anchors x 5 copies)"}
MAIN_REGIMES = ("lam0", "lam0.5", "diffuse")
SANITY_REGIMES = ("lam1",)
HIT_METRICS = (("hit_rate", "hit rate", "hit"), ("hit_rate_recall_ge_half", "quality hit rate (true recall >= 0.5)", "quality"))

#-------------------------------------------------------------------------------
# Data and output helpers

class Data:
    def __init__(self):
        self.d = pd.read_csv(STAGE_D).set_index(["dataset", "regime", "pool", "policy", "budget", "T"]).sort_index()
        self.f = pd.read_csv(STAGE_F).set_index(["dataset", "regime", "pool", "policy", "budget", "T"]).sort_index()

    def series(self, arm: str, ds: str, regime: str, pool: str, policy: str, budget: int, metric: str) -> tuple:
        """(mean, sd) arrays over TS."""
        t = self.d if arm == "oracle" else self.f
        rows = [t.loc[(ds, regime, pool, policy, budget, T)] for T in TS]
        return (np.array([r[metric] for r in rows]), np.array([r[f"{metric}_sd"] for r in rows]))

    def value(self, arm: str, ds: str, regime: str, pool: str, policy: str, budget: int, T: float, metric: str) -> float:
        t = self.d if arm == "oracle" else self.f
        return float(t.loc[(ds, regime, pool, policy, budget, T)][metric])

def out_paths(group: str, name: str, sanity: bool) -> tuple:
    sub = Path("sanity") / group if sanity else Path(group)
    png, pdf = cfg.PLOTS_DIR / sub / f"{name}.png", cfg.PDF_DIR / sub / f"{name}.pdf"
    png.parent.mkdir(parents=True, exist_ok=True)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    return png, pdf

SAVED = []

def save(fig, group: str, name: str, note: str, sanity: bool, axes_desc: str, handles=None, ncol: int = 6,
         tight: bool = True, extra_bottom: float = 0.0) -> None:
    """Wrap the note under the figure, place the legend above it, lay out the axes and write PNG and PDF."""
    w, h = fig.get_size_inches()
    lines = textwrap.wrap(note, width=int(w * 17))
    note_h = 0.14 * len(lines) + 0.1
    leg_h = 0.0
    if handles:
        leg_h = 0.21 * (-(-len(handles) // ncol)) + 0.12
        fig.legend(handles=handles, loc="lower center", ncol=ncol, fontsize=7, frameon=False, bbox_to_anchor=(0.5, note_h / h))
    fig.text(0.005, 0.06 / h, "\n".join(lines), fontsize=7, color=MUTED_TEXT, ha="left", va="bottom")
    bottom = (note_h + leg_h + extra_bottom) / h
    if tight:
        fig.tight_layout(rect=(0, bottom, 1, 1))
    else:
        fig.subplots_adjust(bottom=bottom + 0.04)
    png, pdf = out_paths(group, name, sanity)
    fig.savefig(png, dpi=150)
    fig.savefig(pdf)
    plt.close(fig)
    SAVED.append((str(png.relative_to(cfg.PLOTS_DIR.parent)), axes_desc))

def regimes_for(sanity: bool) -> tuple:
    return SANITY_REGIMES if sanity else MAIN_REGIMES

def band(ax, x, mean, sd, color, linestyle) -> None:
    ax.plot(x, mean, color=color, linestyle=linestyle, linewidth=1.5)
    ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.13, linewidth=0)

def t_markers(ax, x, y, color) -> None:
    for xi, yi, T in zip(x, y, TS):
        ax.scatter([xi], [yi], marker=T_MARKERS[T], s=22, color=color, zorder=4, edgecolor="white", linewidth=0.4)

def set_t_axis(ax) -> None:
    ax.set_xticks(TS)
    ax.set_xticklabels(["0.8", "0.9", "0.95"])
    ax.set_xlim(0.785, 0.965)

def arm_handles(budgets=True, floor=None) -> list:
    handles = []
    if budgets:
        handles += [Line2D([], [], color=BUDGET_COLORS[b], label=BUDGET_LABELS[b]) for b in BUDGET_COLORS]
    handles += [Line2D([], [], color="black", linestyle="-", label="oracle arm"),
                Line2D([], [], color="black", linestyle="--", label="LSH arm")]
    handles += [Line2D([], [], color=GREY, marker=m, linestyle="", markersize=5, label=f"T = {T}") for T, m in T_MARKERS.items()]
    if floor:
        handles.append(Line2D([], [], color=GREY, linestyle="-.", label=floor))
    return handles

#-------------------------------------------------------------------------------
# A. Main figures

def plot_a(data: Data, ds: str, regime: str, kind: str, metric: str, metric_label: str, tag: str, sanity: bool) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.3), sharey=True)
    tt = np.linspace(0.785, 0.965, 50)
    for ax, (tid, policy, pool, pool_label) in zip(axes, TRIPLETS):
        style_axes(ax)
        for b in cfg.BUDGETS:
            for arm, ls in (("oracle", "-"), ("lsh", "--")):
                hit, hit_sd = data.series(arm, ds, regime, pool, policy, b, metric)
                y, y_sd = data.series(arm, ds, regime, pool, policy, b, "achieved_recall")
                c = BUDGET_COLORS[b]
                if kind == "hit_vs_T":
                    band(ax, np.array(TS), hit, hit_sd, c, ls)
                    t_markers(ax, TS, hit, c)
                elif kind == "Y_vs_T":
                    band(ax, np.array(TS), y, y_sd, c, ls)
                    t_markers(ax, TS, y, c)
                else:
                    ax.errorbar(hit, y, xerr=hit_sd, yerr=y_sd, color=c, linestyle=ls, linewidth=1.5, elinewidth=0.6,
                                alpha=0.9)
                    for xi, yi, T in zip(hit, y, TS):
                        ax.scatter([xi], [yi], marker=T_MARKERS[T], s=22, color=c, zorder=4, edgecolor="white", linewidth=0.4)
        if kind == "hit_vs_T":
            ax.plot(tt, 1 - tt, color=GREY, linestyle="-.", linewidth=1)
            set_t_axis(ax)
            ax.set_xlabel("target recall T")
            ax.set_ylabel(metric_label)
        elif kind == "Y_vs_T":
            ax.plot(tt, tt, color=GREY, linestyle="-.", linewidth=1)
            set_t_axis(ax)
            ax.set_xlabel("target recall T")
            ax.set_ylabel("achieved recall Y")
        else:
            if metric == "hit_rate":
                ax.plot(1 - tt, tt, color=GREY, linestyle="-.", linewidth=1)
            ax.set_xlabel(metric_label)
            ax.set_ylabel("achieved recall Y")
        ax.set_title(f"{tid} {policy}, {pool_label}", fontsize=9, color=PRIMARY_TEXT)
    floor = {"hit_vs_T": "hit rate = 1 - T (floor)", "Y_vs_T": "Y = T",
             "Y_vs_hit": "(1 - T, T): no-cache point" if metric == "hit_rate" else None}[kind]
    note = (f"Dataset: {DATASET_LABEL[ds]} | Regime: {REGIME_LABEL[regime]} | Metric: {metric_label} | policy and pool per panel "
            f"(T1 HP1 Ball SEL1; T2 HP3 Group Union SEL2; T3 HP3 Own SEL5; T4 HP3 Ball SEL3; T5 HP3 Group Union SEL4 m = 1.0) | "
            f"mean +/- 1 sd over seeds 0-4" + (" (errorbars)" if kind == "Y_vs_hit" else ""))
    desc = {"hit_vs_T": f"x: T, y: {metric_label}", "Y_vs_T": "x: T, y: achieved recall Y",
            "Y_vs_hit": f"x: {metric_label}, y: achieved recall Y"}[kind]
    save(fig, "A_main", f"A_{ds}_{regime}_{tag}_{kind}", note, sanity, f"5 panels T1-T5; {desc}; series: budget x arm",
         handles=arm_handles(floor=floor), ncol=10)

def make_a(data: Data) -> None:
    for sanity in (False, True):
        for ds in cfg.DATASETS:
            for regime in regimes_for(sanity):
                for metric, label, tag in HIT_METRICS:
                    for kind in ("hit_vs_T", "Y_vs_T", "Y_vs_hit"):
                        plot_a(data, ds, regime, kind, metric, label, tag, sanity)

#-------------------------------------------------------------------------------
# B. Population comparison

def does_not_bind(data: Data, ds: str, regime: str) -> bool:
    """Oracle Group Union SEL2 at 10%: the pool uses less than the full budget or adds under 0.01 hit rate over 5%."""
    key = (ds, regime, "group_union+SEL2", "HP3")
    unused = data.d.loc[key + (5000, 0.9)]["pool_size"] < 0.99 * 5000
    gain = data.value("oracle", ds, regime, "group_union+SEL2", "HP3", 5000, 0.9, "hit_rate") - \
        data.value("oracle", ds, regime, "group_union+SEL2", "HP3", 2500, 0.9, "hit_rate")
    return bool(unused or gain < 0.01)

def plot_b(data: Data, ds: str, regime: str, sanity: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    x = np.array([1, 5, 10])
    binds = not does_not_bind(data, ds, regime)
    for ax, arm, hs_col, title in ((axes[0], "oracle", "hindsight_hit_rate_oracle", "oracle arm"),
                                   (axes[1], "lsh", "hindsight_hit_rate_lsh", "LSH arm")):
        style_axes(ax)
        for pool, label, color in POOLS_B:
            y = [data.value(arm, ds, regime, pool, "HP3", b, 0.9, "hit_rate") for b in cfg.BUDGETS]
            ax.plot(x, y, color=color, marker="o", markersize=4, linewidth=1.5)
            if pool != "no_cache":
                hs = [data.value(arm, ds, regime, pool, "HP3", b, 0.9, hs_col) for b in cfg.BUDGETS]
                ax.plot(x, hs, color=color, linestyle=":", linewidth=1.1)
        if not binds:
            ax.axvspan(7.5, 11, color="#d9d9d9", alpha=0.5, linewidth=0)
            ax.text(9.25, 0.02, "10% budget\ndoes not bind", ha="center", va="bottom", fontsize=7, color=MUTED_TEXT)
        ax.set_xticks(x)
        ax.set_xticklabels(["1%", "5%", "10%"])
        ax.set_xlim(0.5, 11)
        ax.set_ylim(-0.02, 1.03)
        ax.set_xlabel("cache budget (share of the 50,000 database vectors)")
        ax.set_title(title, fontsize=9, color=PRIMARY_TEXT)
    axes[0].set_ylabel("HP3 hit rate at T = 0.9")
    handles = [Line2D([], [], color=c, marker="o", markersize=4, label=l) for _, l, c in POOLS_B]
    handles.append(Line2D([], [], color=GREY, linestyle=":", label="hindsight of the same pool (dotted)"))
    note = (f"Dataset: {DATASET_LABEL[ds]} | Regime: {REGIME_LABEL[regime]} | Policy: HP3, T = 0.9 | Pools: as listed | "
            f"hindsight in the LSH panel uses the LSH estimates | mean over seeds 0-4 | grey band: oracle Group Union SEL2 at 10% "
            f"uses less than the full budget or adds under 0.01 over 5%")
    save(fig, "B_population", f"B_{ds}_{regime}_population", note, sanity,
         "2 panels (oracle, LSH); x: budget; y: HP3 hit rate at T = 0.9; series: 8 pools plus dotted hindsight",
         handles=handles, ncol=5)

def make_b(data: Data) -> None:
    for sanity in (False, True):
        for ds in cfg.DATASETS:
            for regime in regimes_for(sanity):
                plot_b(data, ds, regime, sanity)

#-------------------------------------------------------------------------------
# C. Hit policy comparison on Group Union SEL2 at 5%

def plot_c(data: Data, ds: str, sanity: bool) -> None:
    regimes = regimes_for(sanity)
    fig, axes = plt.subplots(2, len(regimes), figsize=(5.2 * len(regimes), 7.6), squeeze=False)
    pool, B = "group_union+SEL2", cfg.BUDGETS[1]
    tt = np.array(TS)
    for j, regime in enumerate(regimes):
        for i, (metric, label) in enumerate((("hit_rate", "hit rate"), ("mean_true_recall_of_hits", "mean true recall of hits"))):
            ax = axes[i, j]
            style_axes(ax)
            for policy, color in POLICY_COLORS.items():
                for arm, ls in (("oracle", "-"), ("lsh", "--")):
                    m, sd = data.series(arm, ds, regime, pool, policy, B, metric)
                    band(ax, tt, m, sd, color, ls)
                    t_markers(ax, TS, m, color)
            if metric == "hit_rate":
                for arm, col, ls in (("oracle", "hindsight_hit_rate_oracle", ":"), ("lsh", "hindsight_hit_rate_lsh", (0, (1, 3)))):
                    ax.plot(tt, [data.value(arm, ds, regime, pool, "HP3", B, T, col) for T in TS], color="black", linestyle=ls,
                            linewidth=1.1)
            set_t_axis(ax)
            ax.set_xlabel("target recall T")
            ax.set_ylabel(label)
            if i == 0:
                ax.set_title(REGIME_LABEL[regime], fontsize=9, color=PRIMARY_TEXT)
    handles = [Line2D([], [], color=c, label=p) for p, c in POLICY_COLORS.items()]
    handles += [Line2D([], [], color="black", linestyle="-", label="oracle arm"), Line2D([], [], color="black", linestyle="--", label="LSH arm"),
                Line2D([], [], color="black", linestyle=":", label="hindsight, oracle estimates"),
                Line2D([], [], color="black", linestyle=(0, (1, 3)), label="hindsight, LSH estimates")]
    handles += [Line2D([], [], color=GREY, marker=m, linestyle="", markersize=5, label=f"T = {T}") for T, m in T_MARKERS.items()]
    note = (f"Dataset: {DATASET_LABEL[ds]} | Regimes: {', '.join(REGIME_LABEL[r] for r in regimes)} | Policies: HP1, HP2, HP3, HP3f | "
            f"Pool: Group Union SEL2 at the 5% budget (2,500) | mean +/- 1 sd over seeds 0-4")
    save(fig, "C_policy", f"C_{ds}_{'lam1' if sanity else 'main'}_policies", note, sanity,
         "columns: regimes; row 1 y: hit rate, row 2 y: mean true recall of hits; x: T; series: 4 policies x arm, hindsight dotted",
         handles=handles, ncol=6)

def make_c(data: Data) -> None:
    for sanity in (False, True):
        for ds in cfg.DATASETS:
            plot_c(data, ds, sanity)

#-------------------------------------------------------------------------------
# D. LSH minus oracle gap heatmaps

def plot_d(data: Data, pool: str, pool_label: str, metric: str, metric_label: str, tag: str, sanity: bool) -> None:
    regimes = regimes_for(sanity)
    rows = [(ds, r) for ds in cfg.DATASETS for r in regimes]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 0.55 * len(rows) + 2.6), sharey=True)
    for ax, T in zip(axes, TS):
        m = np.array([[data.value("lsh", ds, r, pool, "HP3", b, T, metric) for b in cfg.BUDGETS] for ds, r in rows])
        im = ax.imshow(m, cmap="RdBu", vmin=-0.3, vmax=0.3, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m[i, j]:+.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(m[i, j]) > 0.18 else PRIMARY_TEXT)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["1%", "5%", "10%"])
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{DATASET_LABEL[ds]} {r}" for ds, r in rows], fontsize=8)
        ax.set_xlabel("cache budget")
        ax.set_title(f"T = {T}", fontsize=9, color=PRIMARY_TEXT)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.subplots_adjust(left=0.12, right=0.9, top=0.92)
    cax = fig.add_axes([0.92, 0.25, 0.012, 0.55])
    fig.colorbar(im, cax=cax, label="LSH minus oracle")
    note = (f"Pool: {pool_label} | Policy: HP3 | Metric: gap in {metric_label} (LSH arm minus oracle arm, same pool) | rows: dataset and "
            f"regime | Datasets: SIFT, ESCI | Regimes: {', '.join(regimes)} | mean over seeds 0-4")
    save(fig, "D_gap", f"D_gap_{pool.split('+')[0]}_{tag}{'_lam1' if sanity else ''}", note, sanity,
         "3 panels (T); rows: dataset x regime; columns: budget; color and text: gap in " + metric_label, tight=False,
         extra_bottom=0.5)

def make_d(data: Data) -> None:
    for sanity in (False, True):
        for pool, label in (("group_union+SEL2", "Group Union SEL2"), ("popularity", "Popularity")):
            for metric, mlabel, tag in (("gap_hit_rate", "hit rate", "hit"),
                                         ("gap_hit_rate_recall_ge_half", "quality hit rate (true recall >= 0.5)", "quality")):
                plot_d(data, pool, label, metric, mlabel, tag, sanity)

#-------------------------------------------------------------------------------
# E. LSH safety

SAFETY_ROWS = (("final_gap", "final achieved Y minus T", None), ("frac_below_T", "fraction of steps with Y_t < T", "frac_below_T"),
               ("frac_below_T_after_burn_in", "same, after the first 100 queries", "frac_below_T_after_burn_in"),
               ("ymin", "min running Y_t - T (bar: worst seed; dot: mean)", None))
T_SHADES = {0.8: "#9ecae1", 0.9: "#4292c6", 0.95: "#08519c"}

def plot_e(data: Data, ds: str, pool: str, pool_label: str, sanity: bool) -> None:
    regimes = regimes_for(sanity)
    fig, axes = plt.subplots(4, len(regimes), figsize=(5.0 * len(regimes), 11), squeeze=False)
    policies = list(POLICY_COLORS)
    B = cfg.BUDGETS[1]
    w = 0.26
    for j, regime in enumerate(regimes):
        for i, (key, label, col) in enumerate(SAFETY_ROWS):
            ax = axes[i, j]
            style_axes(ax)
            for k, T in enumerate(TS):
                xs = np.arange(len(policies)) + (k - 1) * w
                if key == "final_gap":
                    vals = [data.value("lsh", ds, regime, pool, p, B, T, "achieved_recall") - T for p in policies]
                    errs = [data.value("lsh", ds, regime, pool, p, B, T, "achieved_recall_sd") for p in policies]
                elif key == "ymin":
                    vals = [data.value("lsh", ds, regime, pool, p, B, T, "y_min_minus_T_worst") for p in policies]
                    errs = None
                else:
                    vals = [data.value("lsh", ds, regime, pool, p, B, T, col) for p in policies]
                    sds = [data.value("lsh", ds, regime, pool, p, B, T, f"{col}_sd") for p in policies]
                    errs = [[min(e, v) for e, v in zip(sds, vals)], [min(e, 1 - v) for e, v in zip(sds, vals)]]
                ax.bar(xs, vals, width=w, color=T_SHADES[T], yerr=errs, error_kw=dict(elinewidth=0.7, ecolor=GREY), zorder=3)
                if key == "ymin":
                    means = [data.value("lsh", ds, regime, pool, p, B, T, "y_min_minus_T") for p in policies]
                    ax.scatter(xs, means, color="black", s=8, zorder=5)
            ax.axhline(0, color=GREY, linewidth=0.8)
            ax.set_xticks(range(len(policies)))
            ax.set_xticklabels(policies)
            if j == 0:
                ax.set_ylabel(label, fontsize=7.5)
            if i == 0:
                ax.set_title(REGIME_LABEL[regime], fontsize=9, color=PRIMARY_TEXT)
    handles = [Patch(color=T_SHADES[T], label=f"T = {T}") for T in TS] + [Line2D([], [], color="black", marker="o", linestyle="",
               markersize=3, label="mean over seeds (last row)")]
    note = (f"Dataset: {DATASET_LABEL[ds]} | Regimes: {', '.join(REGIME_LABEL[r] for r in regimes)} | Policies: HP1, HP2, HP3, HP3f | "
            f"Pool: {pool_label} at the 5% budget | LSH arm | bars: mean over seeds 0-4 with sd (rows 1-3), worst of 5 seeds (row 4)")
    save(fig, "E_safety", f"E_{ds}_{pool.split('+')[0]}{'_lam1' if sanity else ''}", note, sanity,
         "columns: regimes; rows: final Y - T, frac below T, frac below T after burn-in, min Y_t - T; x: policy; bars: T",
         handles=handles, ncol=4)

def make_e(data: Data) -> None:
    for sanity in (False, True):
        for ds in cfg.DATASETS:
            for pool, label in (("group_union+SEL2", "Group Union SEL2"), ("popularity", "Popularity")):
                plot_e(data, ds, pool, label, sanity)

#-------------------------------------------------------------------------------
# F. Running Y_t paths

def make_f(_data: Data) -> None:
    z = np.load(PATHS)
    cells = (("sift", "lam0", "popularity", "Popularity"), ("sift", "lam1", "group_union+SEL2", "Group Union SEL2"))
    for ds, regime, pool, label in cells:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        style_axes(ax)
        t = np.arange(1, len(z[f"{ds}|{regime}|{pool}|oracle"]) + 1)
        for arm, ls, name in (("oracle", "-", "oracle arm"), ("lsh", "--", "LSH arm")):
            y = z[f"{ds}|{regime}|{pool}|{arm}"]
            ax.plot(t, y, color=BUDGET_COLORS[2500], linestyle=ls, linewidth=1.4, label=name)
            ax.scatter([y.argmin() + 1], [y.min()], color=BUDGET_COLORS[2500], s=22, zorder=5, edgecolor="black", linewidth=0.6)
        ax.axhline(0.9, color=GREY, linestyle="-.", linewidth=1, label="T = 0.9")
        ax.axvline(cfg.BURN_IN, color=GREY, linestyle=":", linewidth=1, label="end of burn-in (100)")
        ax.set_xscale("log")
        ax.set_xlabel("stream position t")
        ax.set_ylabel("running Y_t")
        ax.legend(fontsize=7, frameon=False, loc="upper right")
        note = (f"Dataset: {DATASET_LABEL[ds]} | Regime: {REGIME_LABEL[regime]} | Policy: HP3 | Pool: {label} at the 5% budget | "
                f"T = 0.9 | seed 0 | replayed for this cell (marker: minimum of Y_t)")
        save(fig, "F_paths", f"F_{ds}_{regime}_{pool.split('+')[0]}_paths", note, regime == "lam1",
             "x: stream position t (log), y: running Y_t; series: oracle, LSH, T line, burn-in line")

#-------------------------------------------------------------------------------
# G. Estimator

def stage_e_rows() -> pd.DataFrame:
    chosen = json.loads((lb.OUT_DIR / "chosen_config.json").read_text())["config"]
    rows = pd.concat([pd.read_csv(p, comment="#", low_memory=False) for p in sorted(lb.UNIT_DIR.glob("*.csv.gz"))])
    keep = []
    for ds, c in chosen.items():
        keep.append(rows[(rows.dataset == ds) & (rows.L == c["L"]) & (rows.m == c["m"]) & (rows.nu == c["nu"]) &
                         (rows.S == c["S"])])
    return pd.concat(keep)

def plot_g_errors(rows: pd.DataFrame, sanity: bool) -> None:
    regimes = regimes_for(sanity)
    regimes = tuple(r for r in regimes if r in set(rows.regime))
    metrics = (("mean_signed_error", "signed error (estimated minus true recall)"), ("mae", "absolute error"),
               ("over_rate", "share overestimated by more than 0.05"))
    pool_colors = {"popularity": "#E69F00", "group_union_sel2": "#0072B2"}
    pool_labels = {"popularity": "Popularity 5% pool", "group_union_sel2": "Group Union SEL2 5% pool"}
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), squeeze=False)
    for i, ds in enumerate(cfg.DATASETS):
        for j, (metric, label) in enumerate(metrics):
            ax = axes[i, j]
            style_axes(ax)
            for k, pool in enumerate(lb.POOL_NAMES):
                g = rows[(rows.dataset == ds) & (rows.pool == pool)].groupby("regime")[metric]
                mean, sd = g.mean(), g.std(ddof=1)
                xs = np.arange(len(regimes)) + (k - 0.5) * 0.36
                ax.bar(xs, [mean[r] for r in regimes], width=0.34, color=pool_colors[pool], yerr=[sd[r] for r in regimes],
                       error_kw=dict(elinewidth=0.7, ecolor=GREY), zorder=3)
            ax.axhline(0, color=GREY, linewidth=0.8)
            ax.set_xticks(range(len(regimes)))
            ax.set_xticklabels([REGIME_LABEL[r].split(" (")[0] for r in regimes])
            ax.set_title(f"{DATASET_LABEL[ds]}: {label}", fontsize=8.5, color=PRIMARY_TEXT)
    handles = [Patch(color=c, label=pool_labels[p]) for p, c in pool_colors.items()]
    note = ("Stage E benchmark, chosen configuration per dataset (SIFT L=16 m=128 nu=2 S=10,000; ESCI L=8 m=64 nu=4 S=10,000) | "
            f"Regimes: {', '.join(REGIME_LABEL[r] for r in regimes)} | Datasets: SIFT, ESCI | Pools: Popularity and Group Union SEL2 "
            f"(5%, T = 0.9) | mean +/- 1 sd over seeds 100-102 (ESCI and SIFT)")
    save(fig, "G_estimator", f"G_estimator_errors{'_lam1' if sanity else ''}", note, sanity,
         "2 rows (datasets) x 3 columns (signed error, MAE, over rate); x: regime; bars: reference pool", handles=handles, ncol=2)

def plot_g_bytes() -> None:
    chosen = json.loads((lb.OUT_DIR / "chosen_config.json").read_text())["config"]
    dims = {"sift": 128, "esci": 384}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=False)
    for ax, ds in zip(axes, cfg.DATASETS):
        style_axes(ax)
        c = chosen[ds]
        s = lb.config_bytes(c["L"], c["m"], dims[ds], cfg.N_QUERIES + (0 if c["S"] == "none" else int(c["S"])))
        vals = [s] + [lb.cache_bytes(b, dims[ds]) for b in cfg.BUDGETS]
        labels = ["LSH structure", "cache 1%", "cache 5%", "cache 10%"]
        colors = ["#7F7F7F"] + [BUDGET_COLORS[b] for b in cfg.BUDGETS]
        ax.bar(range(4), vals, color=colors, zorder=3)
        for k, v in enumerate(vals):
            ax.text(k, v * 1.04, f"{v / 1e6:.2f} MB", ha="center", fontsize=8, color=PRIMARY_TEXT)
        ax.set_yscale("log")
        ax.set_ylim(min(vals) * 0.5, max(vals) * 2)
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("bytes (log scale)")
        ax.set_title(f"{DATASET_LABEL[ds]} (d = {dims[ds]})", fontsize=9, color=PRIMARY_TEXT)
    note = ("Structure: codes (L x m bits per labeled point) + float64 labels + hyperplane bits, labeled points = 2,500 train + "
            "10,000 sampled database vectors | Cache: pool vectors as float32 | chosen configuration per dataset")
    save(fig, "G_estimator", "G_estimator_bytes", note, False, "2 panels (datasets); x: structure and cache sizes; y: bytes (log)")

def make_g(_data: Data) -> None:
    rows = stage_e_rows()
    plot_g_errors(rows, False)
    plot_g_errors(rows, True)
    plot_g_bytes()

#-------------------------------------------------------------------------------

MAKERS = {"A": make_a, "B": make_b, "C": make_c, "D": make_d, "E": make_e, "F": make_f, "G": make_g}

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=",".join(MAKERS))
    a = ap.parse_args()
    plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5, "savefig.facecolor": "white", "figure.facecolor": "white"})
    data = Data()
    for k in a.only.split(","):
        MAKERS[k](data)
        print(f"{k}: {len(SAVED)} figures so far", flush=True)
    pd.DataFrame(SAVED, columns=["file", "axes_and_series"]).to_csv(cfg.PLOTS_DIR / f"figure_list_{a.only.replace(',', '')}.csv", index=False)
    return 0

if __name__ == "__main__":
    sys.exit(main())
