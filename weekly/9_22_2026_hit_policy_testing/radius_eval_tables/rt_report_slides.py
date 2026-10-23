#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Slide builders for the report PDFs (letter landscape, vector pages, 11 pt minimum font).

Each builder returns a matplotlib figure; rt_report.py adds it to the PDF.
Wording: "mean error" is the mean of (r_hat - theta)/theta in percent; lines on scatters are
"Pooled by nearest 1%"."""

import textwrap
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

import rt_config
import rt_report_data as rd
from plot_style import MUTED_TEXT, PRIMARY_TEXT, style_axes
from rt_plots import NU_COLORS
from simulations.pdf.report import PAGE_SIZE

PAGE_W, PAGE_H = PAGE_SIZE
MIN_FONT = 11
SUBTITLE_WIDTH = 80
MONO = "DejaVu Sans Mono"
POOLED = "Pooled by nearest 1%"
DATASET_NAME = {"sift": "SIFT", "esci": "ESCI"}
STAGE_TEXT = {0: ("uniform database", "uniform queries"), 1: ("uniform database", "real queries"),
              2: ("uniform database", "locality queries"), 3: ("real database", "real queries"),
              4: ("real database", "locality queries")}
Y_MEAN_ERR = "mean error (%)\n= mean of (r_hat - theta)/theta"
Y_ERR = "error (%)\n= (r_hat - theta)/theta"
X_THETA = "true theta"
RC = {"font.size": MIN_FONT, "axes.titlesize": MIN_FONT, "axes.labelsize": MIN_FONT, "xtick.labelsize": MIN_FONT,
      "ytick.labelsize": MIN_FONT, "legend.fontsize": MIN_FONT, "figure.titlesize": 15}
plt.rcParams.update(RC)

GLOSSARY = [
    "theta (true radius): distance from a query to its 100th nearest database vector",
    "r_hat (estimate): the method's prediction of theta",
    "error: (r_hat - theta)/theta in percent, negative means too small",
    "mean error: average error over the test queries",
    "sd of error: how spread out the errors are",
    "slope: slope of a line fitted to r_hat against theta, 1 means r_hat follows theta one-to-one",
    "table, L, m: a table codes each vector with m random bits, L tables are used",
    "Hamming distance: number of differing bits between two codes",
    "nu: number of nearest labeled points taken from each table",
    "labeled point: a stored code together with that vector's theta",
    "sample point, train point: a labeled point that is a database vector, a labeled point that is a training query",
    "distinct labeled points: number of different labeled points one estimate averaged over, across all tables",
    "cutoff Hamming distance: Hamming distance to the nu-th nearest labeled point in a table, divided by m, "
    "averaged over tables",
    "sample-only: the estimate computed without train points",
    "constant radius: the median theta of the sample points, used as a baseline estimate",
    "seed: one random draw of the hash bits and the sample",
    "pooled by nearest 1%: average over the 1% of queries with the closest true theta",
]

#-------------------------------------------------------------------------------
# Text helpers

def wrap_subtitle(lines, width: int = SUBTITLE_WIDTH) -> list:
    """Wrap each given line to at most `width` characters; returns the flat list of lines."""
    if isinstance(lines, str):
        lines = [lines]
    out = []
    for line in lines:
        out.extend(textwrap.wrap(line, width=width, break_long_words=True, break_on_hyphens=False) or [""])
    assert all(len(x) <= width for x in out)
    return out

def stage_label(stage: int, dataset: str) -> str:
    db, q = STAGE_TEXT[stage]
    return f"Stage {stage} ({db}, {q}), {DATASET_NAME[dataset]} (d = {rt_config.DIM[dataset]})"

def subtitle_for(data: dict, parts: list) -> list:
    """Subtitle lines: stage and dataset first, the given parts, and the stage 0 to 2 caveat."""
    lines = [stage_label(data["stage"], data["dataset"])] + list(parts)
    if data["stage"] <= 2:
        lines.append(f"theta relative sd = {100 * rd.theta_relative_sd(data['metrics']):.2f}%: read the mean error only")
    return wrap_subtitle(lines)

def fmt_num(v, decimals: int) -> str:
    if v is None:
        return "-"
    if isinstance(v, str):
        return v
    if isinstance(v, float) and np.isnan(v):
        return "-"
    return f"{v:.{decimals}f}"

#-------------------------------------------------------------------------------
# Figure frame

def draw_header(fig, title: str, sub_lines: list) -> float:
    """Title and subtitle at the top; returns the top of the content area (figure fraction)."""
    fig.text(0.04, 1 - 0.30 / PAGE_H, title, fontsize=15, fontweight="bold", color=PRIMARY_TEXT, va="center", ha="left")
    y = 1 - 0.62 / PAGE_H
    for line in sub_lines:
        fig.text(0.04, y, line, fontsize=MIN_FONT, color="#333333", va="center", ha="left")
        y -= 0.215 / PAGE_H
    return y - 0.18 / PAGE_H

def new_grid(title: str, sub_lines: list, nrows: int, ncols: int, bottom: float = 0.13, left: float = 0.10,
             right: float = 0.985, hspace: float = 0.42, wspace: float = 0.30, **subplot_kw):
    fig = plt.figure(figsize=PAGE_SIZE)
    top = draw_header(fig, title, sub_lines)
    axes = fig.subplots(nrows, ncols, squeeze=False, **subplot_kw)
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, hspace=hspace, wspace=wspace)
    for ax in axes.ravel():
        style_axes(ax)
    return fig, axes

def bottom_legend(fig, handles: list, ncol: int) -> None:
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False, fontsize=MIN_FONT,
               bbox_to_anchor=(0.5, 0.005))

def assert_fits(fig, texts: list, margin_in: float = 0.25) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for t in texts:
        bb = t.get_tightbbox(renderer)
        if bb.x1 / fig.dpi > PAGE_W - margin_in or bb.y0 / fig.dpi < margin_in * 0.5:
            raise ValueError(f"text does not fit on the slide: {t.get_text()[:60]!r}")

#-------------------------------------------------------------------------------
# Front slides

def title_slide(dataset: str):
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.text(0.5, 0.58, f"Multi-table LSH radius estimation: {DATASET_NAME[dataset]}", fontsize=30,
             fontweight="bold", ha="center", va="center", color=PRIMARY_TEXT)
    fig.text(0.5, 0.47, "L tables per estimate, m-bit hashcodes, nu neighbors per table", fontsize=17, ha="center",
             va="center", color="#333333")
    fig.text(0.5, 0.38, datetime.now().strftime("%B %d, %Y"), fontsize=14, ha="center", va="center", color=MUTED_TEXT)
    return fig

def glossary_lines(width: int) -> list:
    out = []
    for entry in GLOSSARY:
        out.extend(textwrap.wrap(entry, width=width, subsequent_indent="      ", break_long_words=False,
                                 break_on_hyphens=False))
    return out

def glossary_slide():
    for width in (118, 110, 100, 90, 80, 70):
        fig = plt.figure(figsize=PAGE_SIZE)
        fig.text(0.04, 1 - 0.40 / PAGE_H, "Glossary", fontsize=20, fontweight="bold", color=PRIMARY_TEXT, va="center")
        lines = glossary_lines(width)
        t = fig.text(0.04, 1 - 0.85 / PAGE_H, "\n".join(lines), fontsize=MIN_FONT, va="top", ha="left",
                     linespacing=1.55, color=PRIMARY_TEXT)
        try:
            assert_fits(fig, [t])
            return fig
        except ValueError:
            plt.close(fig)
    raise ValueError("glossary does not fit on the slide")

def config_slide():
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.text(0.04, 1 - 0.40 / PAGE_H, "Configurations", fontsize=20, fontweight="bold", color=PRIMARY_TEXT, va="center")
    fig.text(0.04, 1 - 0.80 / PAGE_H, "Table t of a configuration uses rows [t*m, (t+1)*m) of the same 2048 x d hash matrix",
             fontsize=MIN_FONT, color="#333333", va="center")
    rows = [(rt_config.config_name(L, m), L, m, L * m) for L, m in rt_config.CONFIGS]
    header = f"{'configuration':<16}{'L':>4}{'m':>7}{'total bits':>12}"
    x_text, y0, dy = 0.14, 0.72, 0.065
    texts = [fig.text(x_text, y0, header, family=MONO, fontsize=MIN_FONT, va="center", fontweight="bold")]
    fig.add_artist(Line2D([x_text, x_text + 0.46], [y0 - dy * 0.5] * 2, color="#888888", lw=0.8))
    for i, (name, L, m, bits) in enumerate(rows):
        y = y0 - dy * (i + 1)
        fig.add_artist(Rectangle((0.075, y - 0.018), 0.045, 0.036, transform=fig.transFigure,
                                 facecolor=rt_config.CONFIG_COLORS[name], edgecolor="black", lw=0.6))
        texts.append(fig.text(x_text, y, f"{name:<16}{L:>4}{m:>7}{bits:>12}", family=MONO, fontsize=MIN_FONT, va="center"))
    assert_fits(fig, texts)
    return fig

def stage_header_slide(data: dict):
    s = data["summary"]
    stage = data["stage"]
    db, q = STAGE_TEXT[stage]
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.text(0.5, 0.60, f"Stage {stage}: {db}, {q}", fontsize=26, fontweight="bold", ha="center", va="center",
             color=PRIMARY_TEXT)
    seeds = s["seeds"]
    exact = "0" if s["exact_max_abs_error"] == 0 else f"{s['exact_max_abs_error']:g}"
    facts = [f"n_train {s['n_train']}, n_test {s['n_test']}, seeds {min(seeds)} to {max(seeds)}",
             f"theta mean {s['test_theta_mean']:.4f}, theta relative sd {100 * s['test_theta_relative_sd']:.2f}%",
             f"EXACT error {exact}, peak memory {s['peak_rss_mb']:.0f} MB, wall time {s['wall_seconds']:.0f} s"]
    y = 0.48
    for line in wrap_subtitle(facts):
        fig.text(0.5, y, line, fontsize=14, ha="center", va="center", color="#333333")
        y -= 0.045
    return fig

#-------------------------------------------------------------------------------
# Tables

def format_table(label_header: str, header: list, rows: list, groups: list = None, separators: set = None) -> list:
    """Fixed-width grid: rows are (label, [cells]); numbers right-aligned, labels left-aligned. Every line has the
    same width. `groups` is an optional list of (text, ncols) printed centered over column groups; `separators` is a set
    of row indexes that get a separator line before them."""
    ncol = len(header)
    cells = [[str(c) for c in cs] for _, cs in rows]
    labels = [str(l) for l, _ in rows]
    wl = max([len(label_header)] + [len(l) for l in labels])
    widths = [max([len(header[j])] + [len(cs[j]) for cs in cells]) for j in range(ncol)]
    gap = "  "
    if groups:
        j = 0
        for text, k in groups:
            span = sum(widths[j:j + k]) + len(gap) * (k - 1)
            if len(text) > span:
                extra = len(text) - span
                widths[j + k - 1] += extra
            j += k
    total = wl + sum(len(gap) + w for w in widths)
    lines = []
    if groups:
        s, j = " " * wl, 0
        for text, k in groups:
            span = sum(widths[j:j + k]) + len(gap) * (k - 1)
            s += gap + text.center(span)
            j += k
        lines.append(s)
    lines.append(label_header.ljust(wl) + "".join(gap + header[j].rjust(widths[j]) for j in range(ncol)))
    lines.append("-" * total)
    for i, (label, cs) in enumerate(zip(labels, cells)):
        if separators and i in separators:
            lines.append("-" * total)
        lines.append(label.ljust(wl) + "".join(gap + cs[j].rjust(widths[j]) for j in range(ncol)))
    lines.append("-" * total)
    assert len({len(l) for l in lines}) == 1
    return lines

def table_slides(title: str, sub_lines: list, lines: list, head_len: int, max_body: int = 24) -> list:
    """One slide per chunk of body lines (header lines repeated); each slide is a monospace text block."""
    head, body = lines[:head_len], lines[head_len:]
    chunks = [body[i:i + max_body] for i in range(0, len(body), max_body)] or [[]]
    figs = []
    for k, chunk in enumerate(chunks):
        fig = plt.figure(figsize=PAGE_SIZE)
        suffix = f" (part {k + 1} of {len(chunks)})" if len(chunks) > 1 else ""
        top = draw_header(fig, title + suffix, sub_lines)
        t = fig.text(0.04, top - 0.1 / PAGE_H, "\n".join(head + chunk), family=MONO, fontsize=MIN_FONT, va="top",
                     ha="left", linespacing=1.45, color=PRIMARY_TEXT)
        assert_fits(fig, [t])
        figs.append(fig)
    return figs

def slide_table_main(data: dict, nu: int, decimals: int) -> list:
    rows_raw = rd.table_main(data["metrics"], nu)
    na_slope = rd.slope_degenerate(data["metrics"])
    rows = []
    for r in rows_raw:
        slope = "n/a" if na_slope else fmt_num(r["slope"], decimals)
        rows.append((r["label"], [fmt_num(r["mean_err"], decimals), fmt_num(r["sd_seeds"], decimals),
                                  fmt_num(r["sd_err"], decimals), slope, fmt_num(r["distinct"], decimals),
                                  fmt_num(r["so_mean_err"], decimals)]))
    header = ["mean err (%)", "+/- sd seeds", "sd err (%)", "slope", "distinct pts", "sample-only (%)"]
    lines = format_table("configuration", header, rows, separators={len(rows) - 1})
    sub = subtitle_for(data, [f"Main table, nu = {nu}, S = 10,000, sample+train, mean over seeds 0 to 4",
                              "mean err: mean error (%); +/- sd seeds: sd of the mean error over seeds; sd err: sd of "
                              "error (%) over test queries; distinct pts: distinct labeled points; sample-only: mean "
                              "error (%) of the sample-only estimate"])
    return table_slides(f"Main table, nu = {nu}", sub, lines, head_len=2)

def slide_table_sample_size(data: dict, decimals: int) -> list:
    rows = []
    for r in rd.table_sample_size(data["metrics"]):
        cells = []
        for me, sd in zip(r["mean_err"], r["sd_err"]):
            cells += [fmt_num(me, decimals), fmt_num(sd, decimals)]
        rows.append((r["label"], cells))
    groups = [("S = 2,500", 2), ("S = 10,000", 2), ("S = full (50,000)", 2)]
    header = ["mean err (%)", "sd err (%)"] * 3
    lines = format_table("configuration", header, rows, groups=groups)
    sub = subtitle_for(data, ["Sample size table, nu = 8, sample+train, mean over seeds 0 to 4",
                              "mean err: mean error (%); sd err: sd of error (%) over test queries"])
    return table_slides("Sample size table, nu = 8", sub, lines, head_len=3)

def slide_table_train_points(data: dict, decimals: int) -> list:
    rows = []
    for r in rd.table_train_points(data["queries"]):
        rows.append((r["label"], [fmt_num(r["train_share"], decimals), fmt_num(r["pct_smaller"], decimals),
                                  fmt_num(r["mean_change_pp"], decimals)]))
    header = ["train share", "% sample+train better", "mean |err| change (pp)"]
    lines = format_table("configuration", header, rows)
    sub = subtitle_for(data, ["Train points table, nu = 8, S = 10,000, mean over seeds 0 to 4",
                              "train share: n_distinct_train / n_distinct_total; % sample+train better: test queries "
                              "where sample+train has the smaller |error|; mean |err| change: |error| of sample-only "
                              "minus |error| of sample+train, in percentage points, positive means train points helped"])
    return table_slides("Train points table, nu = 8", sub, lines, head_len=2)

#-------------------------------------------------------------------------------
# Scatter against theta (P1, P2)

def scatter_limits(queries, const: float) -> tuple:
    vals = np.concatenate([queries["theta"].to_numpy(), queries["r_hat"].to_numpy(), [const]])
    lo, hi = float(vals.min()), float(vals.max())
    pad = 0.04 * (hi - lo if hi > lo else 1.0)
    return lo - pad, hi + pad

def fig_scatter_theta(data: dict, config: str = None):
    q = data["queries"]
    q0 = q[q["seed"] == 0]
    const = data["constant"][0]
    lim = scatter_limits(q0, const)
    name = "all 7 configurations pooled" if config is None else config
    title = "Scatter: r_hat against theta, all configurations" if config is None else f"Scatter: r_hat against theta, {config}"
    sub = subtitle_for(data, ["S = 10,000, sample+train, seed 0, one panel per nu", f"Points: {name}"])
    fig, axes = new_grid(title, sub, 2, 3, hspace=0.50, wspace=0.42, left=0.09)
    color = "#9a9a9a" if config is None else rt_config.CONFIG_COLORS[config]
    for ax, nu in zip(axes.ravel(), rt_config.NU_GRID):
        g = q0[q0["nu"] == nu]
        if config is not None:
            g = g[g["config"] == config]
        ax.scatter(g["theta"], g["r_hat"], s=4, color=color, alpha=0.30 if config is None else 0.45, linewidths=0,
                   rasterized=True)
        px, py = rd.pooled_error_profile(g["theta"].to_numpy(), g["r_hat"].to_numpy(), g["query_id"].to_numpy())
        ax.plot(px, py, color="black", lw=3.0)
        ax.plot(lim, lim, color="black", lw=1.0)
        ax.axhline(const, color="black", lw=1.4, ls=":")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"nu = {nu}", color=PRIMARY_TEXT)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.ticklabel_format(useOffset=False, style="plain")
    for ax in axes[1]:
        ax.set_xlabel(X_THETA)
    for ax in axes[:, 0]:
        ax.set_ylabel("r_hat (estimate)")
    handles = [Line2D([], [], marker="o", ls="", color=color, alpha=0.8, label=f"queries ({name})"),
               Line2D([], [], color="black", lw=3, label=POOLED), Line2D([], [], color="black", lw=1, label="y = x"),
               Line2D([], [], color="black", lw=1.4, ls=":", label="constant radius")]
    bottom_legend(fig, handles, 4)
    return fig

#-------------------------------------------------------------------------------
# Mean error against nu (P3)

def fig_mean_error_vs_nu(data: dict):
    m = data["metrics"]
    sub = subtitle_for(data, ["Mean error against nu, one line per configuration",
                              "Band: +/-1 sd over seeds 0 to 4. Top row sample+train, bottom row sample-only"])
    fig, axes = new_grid("Mean error against nu", sub, 2, 3, sharey="row", hspace=0.50, wspace=0.12, left=0.11)
    pos = np.arange(len(rd.NU_PLOT))
    for i, (pop, pname) in enumerate((("sample_train", "sample+train"), ("sample_only", "sample-only"))):
        for j, S in enumerate(rt_config.S_GRID):
            ax = axes[i, j]
            lines = rd.mean_error_lines(m, pop, S)
            for cname, (mu, sd) in lines.items():
                color = rt_config.CONFIG_COLORS[cname]
                ax.plot(pos, mu, color=color, lw=1.6, marker="o", ms=5)
                ax.fill_between(pos, mu - sd, mu + sd, color=color, alpha=0.15, linewidth=0)
            ax.axhline(0, color="black", lw=1.0)
            ax.set_xticks(pos)
            ax.set_xticklabels([str(n) for n in rd.NU_PLOT])
            ax.set_xlim(-0.4, len(pos) - 0.6)
            sname = "full (50,000)" if S == rt_config.S_GRID[-1] else f"{S:,}"
            ax.set_title(f"S = {sname}, {pname}")
            if i == 1:
                ax.set_xlabel("nu (nearest labeled points per table)")
    if data["stage"] <= 2:
        for ax in axes.ravel():
            ax.set_ylim(-0.1, 0.1)
    for ax in axes[:, 0]:
        ax.set_ylabel(Y_MEAN_ERR)
    handles = [Line2D([], [], color=rt_config.CONFIG_COLORS[c], lw=2, marker="o", ms=5, label=c.replace("_", " "))
               for c in rt_config.CONFIG_NAMES]
    bottom_legend(fig, handles, 7)
    return fig

#-------------------------------------------------------------------------------
# Per-query plots for stages 3 and 4 (P4, P5, P6)

def _config_grid(data: dict, title: str, sub_parts: list, **kw):
    sub = subtitle_for(data, sub_parts)
    fig, axes = new_grid(title, sub, 2, 4, **kw)
    flat = list(axes.ravel())
    for ax, cname in zip(flat[:7], rt_config.CONFIG_NAMES):
        ax.set_title(cname.replace("_", " "), color=rt_config.CONFIG_COLORS[cname], fontweight="bold")
    legend_ax = flat[7]
    legend_ax.axis("off")
    return fig, axes, flat[:7], legend_ax

DISTINCT_TICKS = [1, 2, 5, 10, 20, 50, 100, 200, 500]

def fig_error_vs_distinct(data: dict):
    q = data["queries"]
    q0 = q[q["seed"] == 0]
    fig, axes, panels, lax = _config_grid(
        data, "Error against distinct labeled points",
        ["S = 10,000, sample+train, seed 0, one point per test query and nu, colored by nu",
         "Large marker: mean over test queries for each nu"], sharey=True, left=0.10, hspace=0.60, wspace=0.10, bottom=0.10)
    xmax = max(600.0, float(q0["n_distinct_total"].max()) * 1.15)
    for ax, cname in zip(panels, rt_config.CONFIG_NAMES):
        g = q0[q0["config"] == cname]
        for nu in rt_config.NU_GRID:
            h = g[g["nu"] == nu]
            ax.scatter(h["n_distinct_total"], h["err_st"], s=4, color=NU_COLORS[nu], alpha=0.30, linewidths=0,
                       rasterized=True)
        for nu in rt_config.NU_GRID:
            h = g[g["nu"] == nu]
            ax.scatter([h["n_distinct_total"].mean()], [h["err_st"].mean()], s=110, color=NU_COLORS[nu],
                       edgecolor="black", linewidths=1.6, zorder=5)
        ax.axhline(0, color="black", lw=1.0)
        ax.set_xscale("log")
        ax.set_xlim(0.85, xmax)
        ax.set_xticks([t for t in DISTINCT_TICKS if t <= xmax])
        ax.set_xticklabels([str(t) for t in DISTINCT_TICKS if t <= xmax], rotation=90)
        ax.minorticks_off()
    for ax in axes[1]:
        ax.set_xlabel("distinct labeled points")
    for ax in axes[:, 0]:
        ax.set_ylabel(Y_ERR)
    handles = [Line2D([], [], marker="o", ls="", color=NU_COLORS[nu], mec="black", ms=9, label=f"nu = {nu}")
               for nu in rt_config.NU_GRID] + [Line2D([], [], color="black", lw=1.0, label="zero error")]
    lax.legend(handles=handles, loc="center", frameon=False, title="small points: test queries\nlarge: mean per nu")
    return fig

def fig_bias_profile_a(data: dict):
    q = data["queries"]
    q8 = q[(q["seed"] == 0) & (q["nu"] == 8)]
    fig, axes, panels, lax = _config_grid(
        data, "Error against theta, nu = 8",
        ["S = 10,000, sample+train, seed 0, nu = 8, one point per test query",
         f"Color is the configuration. Line: {POOLED} of the error"] + (["queries form 50 radius clumps, one per anchor"] if data["stage"] == 4 else []),
        sharey=True, left=0.10, hspace=0.45, wspace=0.10, bottom=0.10)
    for ax, cname in zip(panels, rt_config.CONFIG_NAMES):
        g = q8[q8["config"] == cname]
        ax.scatter(g["theta"], g["err_st"], s=4, color=rt_config.CONFIG_COLORS[cname], alpha=0.40, linewidths=0,
                   rasterized=True)
        px, py = rd.pooled_error_profile(g["theta"].to_numpy(), g["err_st"].to_numpy(), g["query_id"].to_numpy())
        ax.plot(px, py, color="black", lw=2.6)
        ax.axhline(0, color="black", lw=1.0)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.ticklabel_format(useOffset=False, style="plain", axis="x")
    for ax in axes[1]:
        ax.set_xlabel(X_THETA)
    for ax in axes[:, 0]:
        ax.set_ylabel(Y_ERR)
    handles = [Line2D([], [], marker="o", ls="", color="#777777", label="test queries"),
               Line2D([], [], color="black", lw=2.6, label=POOLED), Line2D([], [], color="black", lw=1.0, label="zero error")]
    lax.legend(handles=handles, loc="center", frameon=False)
    return fig

def fig_bias_profile_b(data: dict):
    q = data["queries"]
    extra = ["queries form 50 radius clumps, one per anchor"] if data["stage"] == 4 else []
    sub = subtitle_for(data, ["S = 10,000, sample+train, seeds 0 to 4, one panel per nu",
                              f"Line: {POOLED} of the error, mean over seeds", "Band: +/-1 sd over seeds"] + extra)
    fig, axes = new_grid("Error profile against theta, all configurations", sub, 2, 3, sharey=True, hspace=0.50,
                         wspace=0.10, left=0.11)
    for ax, nu in zip(axes.ravel(), rt_config.NU_GRID):
        for cname in rt_config.CONFIG_NAMES:
            g = q[(q["config"] == cname) & (q["nu"] == nu)]
            ys, x = [], None
            for _, gs in g.groupby("seed"):
                x, y = rd.pooled_error_profile(gs["theta"].to_numpy(), gs["err_st"].to_numpy(), gs["query_id"].to_numpy())
                ys.append(y)
            ys = np.array(ys)
            mu, sd = ys.mean(axis=0), ys.std(axis=0, ddof=1)
            color = rt_config.CONFIG_COLORS[cname]
            ax.plot(x, mu, color=color, lw=1.5)
            ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.15, linewidth=0)
        ax.axhline(0, color="black", lw=1.0)
        ax.set_title(f"nu = {nu}")
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.ticklabel_format(useOffset=False, style="plain", axis="x")
    for ax in axes[1]:
        ax.set_xlabel(X_THETA)
    for ax in axes[:, 0]:
        ax.set_ylabel(Y_MEAN_ERR)
    handles = [Line2D([], [], color=rt_config.CONFIG_COLORS[c], lw=2, label=c.replace("_", " ")) for c in rt_config.CONFIG_NAMES]
    bottom_legend(fig, handles, 7)
    return fig

def fig_error_vs_cutoff(data: dict):
    q = data["queries"]
    q8 = q[(q["seed"] == 0) & (q["nu"] == 8)]
    fig, axes, panels, lax = _config_grid(
        data, "Error against cutoff Hamming distance, nu = 8",
        ["S = 10,000, sample+train, seed 0, nu = 8, one point per test query",
         "Color is the configuration. x: cutoff Hamming distance / m, averaged over tables",
         f"Line: {POOLED} of the error (the 1% of queries nearest in x)"], sharey=True, left=0.10, hspace=0.45, wspace=0.10, bottom=0.10)
    for ax, cname in zip(panels, rt_config.CONFIG_NAMES):
        g = q8[q8["config"] == cname]
        ax.scatter(g["mean_cutoff_h_over_m"], g["err_st"], s=4, color=rt_config.CONFIG_COLORS[cname], alpha=0.40,
                   linewidths=0, rasterized=True)
        px, py = rd.pooled_error_profile(g["mean_cutoff_h_over_m"].to_numpy(), g["err_st"].to_numpy(), g["query_id"].to_numpy())
        ax.plot(px, py, color="black", lw=2.6)
        ax.axhline(0, color="black", lw=1.0)
        ax.xaxis.set_major_locator(MaxNLocator(3))
    for ax in axes[1]:
        ax.set_xlabel("cutoff Hamming\ndistance / m")
    for ax in axes[:, 0]:
        ax.set_ylabel(Y_ERR)
    handles = [Line2D([], [], marker="o", ls="", color="#777777", label="test queries"),
               Line2D([], [], color="black", lw=2.6, label=POOLED), Line2D([], [], color="black", lw=1.0, label="zero error")]
    lax.legend(handles=handles, loc="center", frameon=False)
    return fig
