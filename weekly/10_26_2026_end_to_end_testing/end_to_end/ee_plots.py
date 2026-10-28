#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Slide builders for the stage D (oracle) and stage F (LSH) plots, from the saved cell tables, the stage E unit files and
the two replayed paths of ee_plot_paths.py. Nothing is recomputed here. Every builder yields Slide records (letter
landscape matplotlib figures, 11 pt minimum font, header with title and subtitle). Run as a script it writes one PNG per
slide to plots/ (figures that show lambda = 1.0 go to plots/sanity/); ee_report.py assembles the same slides into one PDF.
Oracle lines are solid, LSH lines dashed, bands are +/- 1 sd over seeds.

Run from the thesis root:  python3 weekly/10_26_2026_end_to_end_testing/end_to_end/ee_plots.py [--only A,B,...]"""

import argparse
import json
import sys
import textwrap
from dataclasses import dataclass
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
from plot_style import PRIMARY_TEXT, style_axes  # noqa: E402
from simulations.pdf.report import PAGE_SIZE  # noqa: E402

PAGE_W, PAGE_H = PAGE_SIZE
MIN_FONT = 11
SUBTITLE_WIDTH = 104
plt.rcParams.update({"font.size": MIN_FONT, "axes.titlesize": MIN_FONT, "axes.labelsize": MIN_FONT, "xtick.labelsize": MIN_FONT,
                     "ytick.labelsize": MIN_FONT, "legend.fontsize": MIN_FONT, "figure.titlesize": 15,
                     "savefig.facecolor": "white", "figure.facecolor": "white"})

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
REGIME_SHORT = {"lam0": "lambda = 0", "lam0.5": "lambda = 0.5", "lam1": "lambda = 1.0", "diffuse": "diffuse"}
MAIN_REGIMES = ("lam0", "lam0.5", "diffuse")
SANITY_REGIMES = ("lam1",)
HIT_METRICS = (("hit_rate", "Hit rate", "hit"), ("hit_rate_recall_ge_half", "Quality hit rate (true recall >= 0.5)", "quality"))
SEEDS_NOTE = "mean +/- 1 sd over seeds 0-4"

@dataclass
class Slide:
    fig: object
    group: str
    name: str
    title: str
    sanity: bool
    desc: str

#-------------------------------------------------------------------------------
# Data

class Data:
    def __init__(self):
        key = ["dataset", "regime", "pool", "policy", "budget", "T"]
        self.d = pd.read_csv(STAGE_D).set_index(key).sort_index()
        self.f = pd.read_csv(STAGE_F).set_index(key).sort_index()

    def table(self, arm: str) -> pd.DataFrame:
        return self.d if arm == "oracle" else self.f

    def series(self, arm: str, ds: str, regime: str, pool: str, policy: str, budget: int, metric: str) -> tuple:
        """(mean, sd) arrays over TS."""
        t = self.table(arm)
        rows = [t.loc[(ds, regime, pool, policy, budget, T)] for T in TS]
        return np.array([r[metric] for r in rows]), np.array([r[f"{metric}_sd"] for r in rows])

    def value(self, arm: str, ds: str, regime: str, pool: str, policy: str, budget: int, T: float, metric: str) -> float:
        return float(self.table(arm).loc[(ds, regime, pool, policy, budget, T)][metric])

def regimes_for(sanity: bool) -> tuple:
    return SANITY_REGIMES if sanity else MAIN_REGIMES

#-------------------------------------------------------------------------------
# Slide frame

def wrap_lines(lines) -> list:
    lines = [lines] if isinstance(lines, str) else lines
    out = []
    for line in lines:
        out.extend(textwrap.wrap(line, width=SUBTITLE_WIDTH, break_on_hyphens=False) or [""])
    return out

def draw_header(fig, title: str, sub_lines: list) -> float:
    """Title and subtitle at the top; returns the top of the content area as a figure fraction."""
    fig.text(0.04, 1 - 0.30 / PAGE_H, title, fontsize=15, fontweight="bold", color=PRIMARY_TEXT, va="center", ha="left")
    y = 1 - 0.62 / PAGE_H
    for line in sub_lines:
        fig.text(0.04, y, line, fontsize=MIN_FONT, color="#333333", va="center", ha="left")
        y -= 0.215 / PAGE_H
    return y - 0.55 / PAGE_H

def new_figure(title: str, sub_lines, nrows: int, ncols: int, legend_rows: int = 0, left: float = 0.09, right: float = 0.985,
               hspace: float = 0.5, wspace: float = 0.28, **kw) -> tuple:
    fig = plt.figure(figsize=PAGE_SIZE)
    top = draw_header(fig, title, wrap_lines(sub_lines))
    axes = fig.subplots(nrows, ncols, squeeze=False, **kw)
    bottom = (0.7 + 0.24 * legend_rows) / PAGE_H
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, hspace=hspace, wspace=wspace)
    for ax in axes.ravel():
        style_axes(ax)
    return fig, axes

def bottom_legend(fig, handles: list, ncol: int) -> None:
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False, fontsize=MIN_FONT, bbox_to_anchor=(0.5, 0.005))

def band(ax, x, mean, sd, color, linestyle) -> None:
    ax.plot(x, mean, color=color, linestyle=linestyle, linewidth=1.6)
    ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.13, linewidth=0)

def t_markers(ax, x, y, color) -> None:
    for xi, yi, T in zip(x, y, TS):
        ax.scatter([xi], [yi], marker=T_MARKERS[T], s=28, color=color, zorder=4, edgecolor="white", linewidth=0.4)

def set_t_axis(ax) -> None:
    ax.set_xticks(TS)
    ax.set_xticklabels(["0.8", "0.9", "0.95"])
    ax.set_xlim(0.785, 0.965)

def arm_handles(budgets: bool = True, floor: str = None) -> list:
    handles = []
    if budgets:
        handles += [Line2D([], [], color=BUDGET_COLORS[b], label=BUDGET_LABELS[b]) for b in BUDGET_COLORS]
    handles += [Line2D([], [], color="black", linestyle="-", label="oracle arm"),
                Line2D([], [], color="black", linestyle="--", label="LSH arm")]
    handles += [Line2D([], [], color=GREY, marker=m, linestyle="", markersize=6, label=f"T = {T}") for T, m in T_MARKERS.items()]
    if floor:
        handles.append(Line2D([], [], color=GREY, linestyle="-.", label=floor))
    return handles

#-------------------------------------------------------------------------------
# A. Main figures: five triplet panels (2 x 3 grid, the sixth cell holds the legend)

TRIPLET_TEXT = "T1 HP1 Ball SEL1; T2 HP3 Group Union SEL2; T3 HP3 Own SEL5; T4 HP3 Ball SEL3; T5 HP3 Group Union SEL4 (m = 1.0)"

def slide_a(data: Data, ds: str, regime: str, kind: str, metric: str, metric_label: str, tag: str, sanity: bool) -> Slide:
    hit_name = "quality hit rate (true recall >= 0.5)" if tag == "quality" else "hit rate"
    titles = {"hit_vs_T": f"{metric_label} against target recall T", "Y_vs_T": "Achieved recall Y against target recall T",
              "Y_vs_hit": f"Achieved recall Y against {hit_name}"}
    sub = [f"{DATASET_LABEL[ds]}, {REGIME_LABEL[regime]}. Panels: {TRIPLET_TEXT}.",
           f"Oracle arm solid, LSH arm dashed, one color per budget, {SEEDS_NOTE}"
           + (" (error bars)." if kind == "Y_vs_hit" else ".")]
    fig, axes = new_figure(titles[kind], sub, 2, 3, hspace=0.75, wspace=0.22, sharey=True)
    tt = np.linspace(0.785, 0.965, 50)
    for ax, (tid, policy, pool, pool_label) in zip(axes.ravel(), TRIPLETS):
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
                    ax.errorbar(hit, y, xerr=hit_sd, yerr=y_sd, color=c, linestyle=ls, linewidth=1.6, elinewidth=0.6, alpha=0.9)
                    for xi, yi, T in zip(hit, y, TS):
                        ax.scatter([xi], [yi], marker=T_MARKERS[T], s=28, color=c, zorder=4, edgecolor="white", linewidth=0.4)
        if kind == "hit_vs_T":
            ax.plot(tt, 1 - tt, color=GREY, linestyle="-.", linewidth=1)
            set_t_axis(ax)
            ax.set_xlabel("target recall T")
            ax.set_ylabel("quality hit rate" if tag == "quality" else "hit rate")
        elif kind == "Y_vs_T":
            ax.plot(tt, tt, color=GREY, linestyle="-.", linewidth=1)
            set_t_axis(ax)
            ax.set_xlabel("target recall T")
            ax.set_ylabel("achieved recall Y")
        else:
            if metric == "hit_rate":
                ax.plot(1 - tt, tt, color=GREY, linestyle="-.", linewidth=1)
            ax.set_xlabel("quality hit rate" if tag == "quality" else "hit rate")
            ax.set_ylabel("achieved recall Y")
        ax.set_title(f"{tid} {policy}\n{pool_label}", fontsize=MIN_FONT, color=PRIMARY_TEXT)
    floor = {"hit_vs_T": "hit rate = 1 - T (floor)", "Y_vs_T": "Y = T",
             "Y_vs_hit": "(1 - T, T): no-cache point" if metric == "hit_rate" else None}[kind]
    legend_ax = axes.ravel()[5]
    legend_ax.axis("off")
    legend_ax.legend(handles=arm_handles(floor=floor), loc="center", frameon=False, fontsize=MIN_FONT)
    name = f"A_{ds}_{regime}_Y_vs_T" if kind == "Y_vs_T" else f"A_{ds}_{regime}_{tag}_{kind}"
    desc = {"hit_vs_T": f"x: T, y: {hit_name}", "Y_vs_T": "x: T, y: achieved recall Y",
            "Y_vs_hit": f"x: {hit_name}, y: achieved recall Y"}[kind]
    return Slide(fig, "A_main", name, f"{DATASET_LABEL[ds]}, {REGIME_SHORT[regime]}: {titles[kind]}", sanity,
                 f"5 panels T1-T5; {desc}; series: budget x arm")

def slides_a(data: Data, sanity: bool):
    for ds in cfg.DATASETS:
        for regime in regimes_for(sanity):
            for metric, label, tag in HIT_METRICS:
                for kind in ("hit_vs_T", "Y_vs_T", "Y_vs_hit"):
                    if kind == "Y_vs_T" and tag == "quality":
                        continue  # Y does not depend on the hit metric
                    yield slide_a(data, ds, regime, kind, metric, label, tag, sanity)

#-------------------------------------------------------------------------------
# B. Population comparison

def does_not_bind(data: Data, ds: str, regime: str) -> bool:
    """Oracle Group Union SEL2 at 10%: the pool uses less than the full budget or adds under 0.01 hit rate over 5%."""
    unused = data.d.loc[(ds, regime, "group_union+SEL2", "HP3", 5000, 0.9)]["pool_size"] < 0.99 * 5000
    gain = data.value("oracle", ds, regime, "group_union+SEL2", "HP3", 5000, 0.9, "hit_rate") - \
        data.value("oracle", ds, regime, "group_union+SEL2", "HP3", 2500, 0.9, "hit_rate")
    return bool(unused or gain < 0.01)

def slide_b(data: Data, ds: str, regime: str, sanity: bool) -> Slide:
    binds = not does_not_bind(data, ds, regime)
    sub = [f"{DATASET_LABEL[ds]}, {REGIME_LABEL[regime]}. Policy HP3, T = 0.9, mean over seeds 0-4. Dotted lines: hindsight of the "
           "same pool (in the LSH panel it uses the LSH estimates)."]
    if not binds:
        sub.append("Grey band: the 10% budget does not bind (oracle Group Union SEL2 leaves budget unused or gains under 0.01 "
                   "over 5%).")
    fig, axes = new_figure("Hit rate against cache budget", sub, 1, 2, legend_rows=3, wspace=0.12, sharey=True)
    x = np.array([1, 5, 10])
    for ax, arm, hs_col, title in ((axes[0, 0], "oracle", "hindsight_hit_rate_oracle", "oracle arm"),
                                   (axes[0, 1], "lsh", "hindsight_hit_rate_lsh", "LSH arm")):
        for pool, label, color in POOLS_B:
            y = [data.value(arm, ds, regime, pool, "HP3", b, 0.9, "hit_rate") for b in cfg.BUDGETS]
            ax.plot(x, y, color=color, marker="o", markersize=5, linewidth=1.6)
            if pool != "no_cache":
                hs = [data.value(arm, ds, regime, pool, "HP3", b, 0.9, hs_col) for b in cfg.BUDGETS]
                ax.plot(x, hs, color=color, linestyle=":", linewidth=1.2)
        if not binds:
            ax.axvspan(7.5, 11, color="#d9d9d9", alpha=0.5, linewidth=0)
        ax.set_xticks(x)
        ax.set_xticklabels(["1%", "5%", "10%"])
        ax.set_xlim(0.5, 11)
        ax.set_ylim(-0.02, 1.03)
        ax.set_xlabel("cache budget")
        ax.set_title(title, fontsize=MIN_FONT, color=PRIMARY_TEXT)
    axes[0, 0].set_ylabel("HP3 hit rate at T = 0.9")
    handles = [Line2D([], [], color=c, marker="o", markersize=5, label=l) for _, l, c in POOLS_B]
    handles.append(Line2D([], [], color=GREY, linestyle=":", label="hindsight (dotted)"))
    bottom_legend(fig, handles, 3)
    return Slide(fig, "B_population", f"B_{ds}_{regime}_population", f"{DATASET_LABEL[ds]}, {REGIME_SHORT[regime]}: population comparison",
                 sanity, "2 panels (oracle, LSH); x: budget; y: HP3 hit rate at T = 0.9; series: 8 pools plus dotted hindsight")

def slides_b(data: Data, sanity: bool):
    for ds in cfg.DATASETS:
        for regime in regimes_for(sanity):
            yield slide_b(data, ds, regime, sanity)

#-------------------------------------------------------------------------------
# C. Hit policy comparison on Group Union SEL2 at 5%

def slide_c(data: Data, ds: str, sanity: bool) -> Slide:
    regimes = regimes_for(sanity)
    sub = [f"{DATASET_LABEL[ds]}. Pool Group Union SEL2 at the 5% budget (2,500), policies HP1, HP2, HP3, HP3f, {SEEDS_NOTE}.",
           "Oracle arm solid, LSH arm dashed. Black dotted lines: hindsight under the oracle and the LSH estimates."]
    fig, axes = new_figure("Hit policy comparison", sub, 2, len(regimes), legend_rows=3, hspace=0.5, wspace=0.3)
    pool, B, tt = "group_union+SEL2", cfg.BUDGETS[1], np.array(TS)
    for j, regime in enumerate(regimes):
        for i, (metric, label) in enumerate((("hit_rate", "hit rate"), ("mean_true_recall_of_hits", "mean true recall of hits"))):
            ax = axes[i, j]
            for policy, color in POLICY_COLORS.items():
                for arm, ls in (("oracle", "-"), ("lsh", "--")):
                    m, sd = data.series(arm, ds, regime, pool, policy, B, metric)
                    band(ax, tt, m, sd, color, ls)
                    t_markers(ax, TS, m, color)
            if metric == "hit_rate":
                for arm, col, ls in (("oracle", "hindsight_hit_rate_oracle", ":"), ("lsh", "hindsight_hit_rate_lsh", (0, (1, 3)))):
                    ax.plot(tt, [data.value(arm, ds, regime, pool, "HP3", B, T, col) for T in TS], color="black", linestyle=ls,
                            linewidth=1.2)
            set_t_axis(ax)
            ax.set_xlabel("target recall T")
            ax.set_ylabel(label)
            if i == 0:
                ax.set_title(REGIME_SHORT[regime], fontsize=MIN_FONT, color=PRIMARY_TEXT)
    handles = [Line2D([], [], color=c, label=p) for p, c in POLICY_COLORS.items()]
    handles += [Line2D([], [], color="black", linestyle="-", label="oracle arm"), Line2D([], [], color="black", linestyle="--", label="LSH arm"),
                Line2D([], [], color="black", linestyle=":", label="hindsight, oracle"),
                Line2D([], [], color="black", linestyle=(0, (1, 3)), label="hindsight, LSH")]
    handles += [Line2D([], [], color=GREY, marker=m, linestyle="", markersize=6, label=f"T = {T}") for T, m in T_MARKERS.items()]
    bottom_legend(fig, handles, 5)
    return Slide(fig, "C_policy", f"C_{ds}_{'lam1' if sanity else 'main'}_policies",
                 f"{DATASET_LABEL[ds]}{', lambda = 1.0' if sanity else ''}: hit policy comparison", sanity,
                 "columns: regimes; row 1 y: hit rate, row 2 y: mean true recall of hits; x: T; series: 4 policies x arm, hindsight dotted")

def slides_c(data: Data, sanity: bool):
    for ds in cfg.DATASETS:
        yield slide_c(data, ds, sanity)

#-------------------------------------------------------------------------------
# D. LSH minus oracle gap heatmaps

def slide_d(data: Data, pool: str, pool_label: str, metric: str, metric_label: str, tag: str, sanity: bool) -> Slide:
    regimes = regimes_for(sanity)
    rows = [(ds, r) for ds in cfg.DATASETS for r in regimes]
    sub = [f"Pool {pool_label}, policy HP3, mean over seeds 0-4. Rows: dataset and regime. Columns: cache budget (1%, 5%, 10%).",
           f"Value: {metric_label} of the LSH arm minus the oracle arm on the same pool."]
    fig = plt.figure(figsize=PAGE_SIZE)
    top = draw_header(fig, f"LSH minus oracle gap in {metric_label}", wrap_lines(sub))
    axes = fig.subplots(1, 3, sharey=True)
    fig.subplots_adjust(left=0.13, right=0.9, top=top, bottom=0.12, wspace=0.12)
    for ax, T in zip(axes, TS):
        m = np.array([[data.value("lsh", ds, r, pool, "HP3", b, T, metric) for b in cfg.BUDGETS] for ds, r in rows])
        im = ax.imshow(m, cmap="RdBu", vmin=-0.3, vmax=0.3, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m[i, j]:+.2f}", ha="center", va="center", fontsize=MIN_FONT,
                        color="white" if abs(m[i, j]) > 0.18 else PRIMARY_TEXT)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["1%", "5%", "10%"])
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{DATASET_LABEL[ds]} {REGIME_SHORT[r]}" for ds, r in rows])
        ax.set_xlabel("cache budget")
        ax.set_title(f"T = {T}", fontsize=MIN_FONT, color=PRIMARY_TEXT)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.colorbar(im, cax=fig.add_axes([0.92, 0.2, 0.012, 0.5]), label="LSH minus oracle")
    return Slide(fig, "D_gap", f"D_gap_{pool.split('+')[0]}_{tag}{'_lam1' if sanity else ''}",
                 f"Gap heatmap, {pool_label}, {metric_label}{', lambda = 1.0' if sanity else ''}", sanity,
                 f"3 panels (T); rows: dataset x regime; columns: budget; color and text: gap in {metric_label}")

def slides_d(data: Data, sanity: bool):
    for pool, label in (("group_union+SEL2", "Group Union SEL2"), ("popularity", "Popularity")):
        for metric, mlabel, tag in (("gap_hit_rate", "hit rate", "hit"), ("gap_hit_rate_recall_ge_half", "quality hit rate", "quality")):
            yield slide_d(data, pool, label, metric, mlabel, tag, sanity)

#-------------------------------------------------------------------------------
# E. LSH safety (two slides per dataset and pool: rows 1-2 and rows 3-4)

T_SHADES = {0.8: "#9ecae1", 0.9: "#4292c6", 0.95: "#08519c"}
SAFETY_ROWS = (("final_gap", "final achieved Y minus T"), ("frac_below_T", "fraction of steps with Y_t < T"),
               ("frac_below_T_after_burn_in", "same, after the first 100 queries"),
               ("ymin", "min running Y_t minus T\n(bar: worst seed, dot: mean)"))

def safety_values(data: Data, ds: str, regime: str, pool: str, policies: list, B: int, T: float, key: str) -> tuple:
    def get(p, col):
        return data.value("lsh", ds, regime, pool, p, B, T, col)
    if key == "final_gap":
        return [get(p, "achieved_recall") - T for p in policies], [get(p, "achieved_recall_sd") for p in policies]
    if key == "ymin":
        return [get(p, "y_min_minus_T_worst") for p in policies], None
    vals, sds = [get(p, key) for p in policies], [get(p, f"{key}_sd") for p in policies]
    return vals, [[min(e, v) for e, v in zip(sds, vals)], [min(e, 1 - v) for e, v in zip(sds, vals)]]

def slide_e(data: Data, ds: str, pool: str, pool_label: str, part: int, sanity: bool) -> Slide:
    regimes = regimes_for(sanity)
    rows = SAFETY_ROWS[:2] if part == 1 else SAFETY_ROWS[2:]
    sub = [f"{DATASET_LABEL[ds]}, LSH arm, pool {pool_label} at the 5% budget. Bars: mean over seeds 0-4 with sd "
           "(worst of the 5 seeds for the minimum).", "Columns: regimes. x: policy. One bar per T."]
    fig, axes = new_figure(f"LSH safety ({part} of 2)", sub, 2, len(regimes), legend_rows=1, hspace=0.45, wspace=0.28)
    policies, B, w = list(POLICY_COLORS), cfg.BUDGETS[1], 0.26
    for j, regime in enumerate(regimes):
        for i, (key, label) in enumerate(rows):
            ax = axes[i, j]
            for k, T in enumerate(TS):
                xs = np.arange(len(policies)) + (k - 1) * w
                vals, errs = safety_values(data, ds, regime, pool, policies, B, T, key)
                ax.bar(xs, vals, width=w, color=T_SHADES[T], yerr=errs, error_kw=dict(elinewidth=0.7, ecolor=GREY), zorder=3)
                if key == "ymin":
                    ax.scatter(xs, [data.value("lsh", ds, regime, pool, p, B, T, "y_min_minus_T") for p in policies], color="black",
                               s=10, zorder=5)
            ax.axhline(0, color=GREY, linewidth=0.8)
            ax.set_xticks(range(len(policies)))
            ax.set_xticklabels(policies)
            if j == 0:
                ax.set_ylabel(label)
            if i == 0:
                ax.set_title(REGIME_SHORT[regime], fontsize=MIN_FONT, color=PRIMARY_TEXT)
    bottom_legend(fig, [Patch(color=T_SHADES[T], label=f"T = {T}") for T in TS] +
                  [Line2D([], [], color="black", marker="o", linestyle="", markersize=4, label="mean over seeds (minimum row)")], 4)
    return Slide(fig, "E_safety", f"E_{ds}_{pool.split('+')[0]}_part{part}{'_lam1' if sanity else ''}",
                 f"{DATASET_LABEL[ds]}, {pool_label}: LSH safety, part {part}{', lambda = 1.0' if sanity else ''}", sanity,
                 "columns: regimes; rows: " + ", ".join(r[1].replace("\n", " ") for r in rows) + "; x: policy; bars: T")

def slides_e(data: Data, sanity: bool):
    for ds in cfg.DATASETS:
        for pool, label in (("group_union+SEL2", "Group Union SEL2"), ("popularity", "Popularity")):
            for part in (1, 2):
                yield slide_e(data, ds, pool, label, part, sanity)

#-------------------------------------------------------------------------------
# F. Running Y_t paths

def slides_f(_data: Data, sanity: bool):
    z = np.load(PATHS)
    cells = (("sift", "lam0", "popularity", "Popularity"), ("sift", "lam1", "group_union+SEL2", "Group Union SEL2"))
    for ds, regime, pool, label in cells:
        if (regime == "lam1") != sanity:
            continue
        sub = [f"{DATASET_LABEL[ds]}, {REGIME_LABEL[regime]}, pool {label} at the 5% budget, policy HP3, T = 0.9, seed 0.",
               "Replayed for this cell. Marker: minimum of Y_t. Dotted vertical line: end of the 100 query burn-in."]
        fig, axes = new_figure("Running achieved recall Y_t", sub, 1, 1, left=0.1)
        ax = axes[0, 0]
        t = np.arange(1, len(z[f"{ds}|{regime}|{pool}|oracle"]) + 1)
        for arm, ls, name in (("oracle", "-", "oracle arm"), ("lsh", "--", "LSH arm")):
            y = z[f"{ds}|{regime}|{pool}|{arm}"]
            ax.plot(t, y, color=BUDGET_COLORS[2500], linestyle=ls, linewidth=1.6, label=name)
            ax.scatter([y.argmin() + 1], [y.min()], color=BUDGET_COLORS[2500], s=34, zorder=5, edgecolor="black", linewidth=0.7)
        ax.axhline(0.9, color=GREY, linestyle="-.", linewidth=1, label="T = 0.9")
        ax.axvline(cfg.BURN_IN, color=GREY, linestyle=":", linewidth=1, label="end of burn-in (100)")
        ax.set_xscale("log")
        ax.set_xlabel("stream position t")
        ax.set_ylabel("running Y_t")
        ax.legend(frameon=False, loc="upper right")
        yield Slide(fig, "F_paths", f"F_{ds}_{regime}_{pool.split('+')[0]}_paths",
                    f"{DATASET_LABEL[ds]}, {REGIME_SHORT[regime]}, {label}: running Y_t", sanity,
                    "x: stream position t (log), y: running Y_t; series: oracle, LSH, T line, burn-in line")

#-------------------------------------------------------------------------------
# G. Estimator

def stage_e_rows() -> pd.DataFrame:
    chosen = json.loads((lb.OUT_DIR / "chosen_config.json").read_text())["config"]
    rows = pd.concat([pd.read_csv(p, comment="#", low_memory=False) for p in sorted(lb.UNIT_DIR.glob("*.csv.gz"))])
    keep = [rows[(rows.dataset == ds) & (rows.L == c["L"]) & (rows.m == c["m"]) & (rows.nu == c["nu"]) & (rows.S == c["S"])]
            for ds, c in chosen.items()]
    return pd.concat(keep)

def slide_g_errors(rows: pd.DataFrame, sanity: bool) -> Slide:
    regimes = tuple(r for r in regimes_for(sanity) if r in set(rows.regime))
    metrics = (("mean_signed_error", "signed error"), ("mae", "absolute error"), ("over_rate", "over rate"))
    pool_colors = {"popularity": "#E69F00", "group_union_sel2": "#0072B2"}
    pool_labels = {"popularity": "Popularity 5% pool", "group_union_sel2": "Group Union SEL2 5% pool"}
    sub = ["Stage E benchmark, chosen configuration (SIFT L=16 m=128 nu=2 S=10,000; ESCI L=8 m=64 nu=4 S=10,000).",
           "Signed error: estimated minus true recall. Over rate: share of test queries overestimated by more than 0.05. "
           "Mean +/- 1 sd over seeds 100-102."]
    fig, axes = new_figure("Recall estimation error of the LSH arm", sub, 2, 3, legend_rows=1, hspace=0.5, wspace=0.35)
    for i, ds in enumerate(cfg.DATASETS):
        for j, (metric, label) in enumerate(metrics):
            ax = axes[i, j]
            for k, pool in enumerate(lb.POOL_NAMES):
                g = rows[(rows.dataset == ds) & (rows.pool == pool)].groupby("regime")[metric]
                mean, sd = g.mean(), g.std(ddof=1)
                ax.bar(np.arange(len(regimes)) + (k - 0.5) * 0.36, [mean[r] for r in regimes], width=0.34, color=pool_colors[pool],
                       yerr=[sd[r] for r in regimes], error_kw=dict(elinewidth=0.7, ecolor=GREY), zorder=3)
            ax.axhline(0, color=GREY, linewidth=0.8)
            ax.set_xticks(range(len(regimes)))
            ax.set_xticklabels([REGIME_SHORT[r] for r in regimes])
            ax.set_title(f"{DATASET_LABEL[ds]}: {label}", fontsize=MIN_FONT, color=PRIMARY_TEXT)
    bottom_legend(fig, [Patch(color=c, label=pool_labels[p]) for p, c in pool_colors.items()], 2)
    return Slide(fig, "G_estimator", f"G_estimator_errors{'_lam1' if sanity else ''}",
                 f"Estimator error by regime{', lambda = 1.0' if sanity else ''}", sanity,
                 "2 rows (datasets) x 3 columns (signed error, absolute error, over rate); x: regime; bars: reference pool")

def slide_g_bytes() -> Slide:
    chosen = json.loads((lb.OUT_DIR / "chosen_config.json").read_text())["config"]
    dims = {"sift": 128, "esci": 384}
    sub = ["Structure: codes (L x m bits per labeled point) + float64 labels + hyperplane bits, labeled points = 2,500 train + "
           "10,000 sampled database vectors.", "Cache: pool vectors stored as float32. Chosen configuration per dataset."]
    fig, axes = new_figure("LSH structure bytes against cache bytes", sub, 1, 2, wspace=0.28, left=0.1)
    for ax, ds in zip(axes[0], cfg.DATASETS):
        c = chosen[ds]
        s = lb.config_bytes(c["L"], c["m"], dims[ds], cfg.N_QUERIES + (0 if c["S"] == "none" else int(c["S"])))
        vals = [s] + [lb.cache_bytes(b, dims[ds]) for b in cfg.BUDGETS]
        ax.bar(range(4), vals, color=["#7F7F7F"] + [BUDGET_COLORS[b] for b in cfg.BUDGETS], zorder=3)
        for k, v in enumerate(vals):
            ax.text(k, v * 1.06, f"{v / 1e6:.2f} MB", ha="center", fontsize=MIN_FONT, color=PRIMARY_TEXT)
        ax.set_yscale("log")
        ax.set_ylim(min(vals) * 0.5, max(vals) * 3)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["LSH\nstructure", "cache\n1%", "cache\n5%", "cache\n10%"])
        ax.set_ylabel("bytes (log scale)")
        ax.set_title(f"{DATASET_LABEL[ds]} (d = {dims[ds]})", fontsize=MIN_FONT, color=PRIMARY_TEXT)
    return Slide(fig, "G_estimator", "G_estimator_bytes", "LSH structure bytes against cache bytes", False,
                 "2 panels (datasets); x: structure and cache sizes; y: bytes (log)")

def slides_g(_data: Data, sanity: bool):
    yield slide_g_errors(stage_e_rows(), sanity)
    if not sanity:
        yield slide_g_bytes()

#-------------------------------------------------------------------------------

SECTIONS = (("A", "Main figures: five triplets", slides_a), ("B", "Population comparison", slides_b),
            ("C", "Hit policy comparison", slides_c), ("D", "LSH minus oracle gap", slides_d),
            ("E", "LSH safety", slides_e), ("F", "Running Y_t paths", slides_f), ("G", "Estimator error and bytes", slides_g))

def save_png(slide: Slide) -> Path:
    sub = Path("sanity") / slide.group if slide.sanity else Path(slide.group)
    path = cfg.PLOTS_DIR / sub / f"{slide.name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    slide.fig.savefig(path, dpi=130)
    return path

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=",".join(k for k, _, _ in SECTIONS))
    a = ap.parse_args()
    data = Data()
    listed = []
    for key, _, maker in SECTIONS:
        if key not in a.only.split(","):
            continue
        for sanity in (False, True):
            for slide in maker(data, sanity):
                path = save_png(slide)
                plt.close(slide.fig)
                listed.append((str(path.relative_to(cfg.PLOTS_DIR.parent)), slide.desc))
        print(f"{key}: {len(listed)} figures so far", flush=True)
    pd.DataFrame(listed, columns=["file", "axes_and_series"]).to_csv(
        cfg.PLOTS_DIR / f"figure_list_{a.only.replace(',', '')}.csv", index=False)
    return 0

if __name__ == "__main__":
    sys.exit(main())
