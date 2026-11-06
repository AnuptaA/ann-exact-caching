#!/usr/bin/env python3
"""2-D illustration of the 7 model inputs (k = 5 for readability).

Inputs: relative offset and relative size for each of the 3 best-fitting
cache balls, plus the intruder count (sampled uncached points inside the
query's neighborhood). A query between balls whose 5 closest cached points
come from two balls. Every quantity is labeled where it is measured, and the
inputs are listed (highlighted) with the arithmetic.
Left: per-ball stats. Right (zoom on the query): the shell sample.
Writes results/summary/feature_diagram.png.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge

from simulations.approximate_hits.report import common

K = 5
SHELL_RANK = 120
P = 0.4
rng = np.random.default_rng(3)
db = rng.uniform(-10, 10, size=(700, 2))
draw = np.random.default_rng(7).random(len(db))


def ball(center, size):
    d = np.linalg.norm(db - center, axis=1)
    order = np.argsort(d)
    return {"c": np.asarray(center, float), "r": d[order[size - 1]],
            "m": set(order[:size].tolist()), "shell": order[size:SHELL_RANK],
            "R": d[order[SHELL_RANK - 1]]}


balls = {"A": ball((-1.5, 0.0), 70), "B": ball((4.2, 1.6), 35), "C": ball((2.6, -4.6), 30)}
cached = np.zeros(len(db), dtype=bool)
in_shell = np.zeros(len(db), dtype=bool)
for b in balls.values():
    cached[list(b["m"])] = True
    in_shell[b["shell"]] = True
in_shell &= ~cached
sampled = in_shell & (draw < P)


def analyze(q):
    d = np.linalg.norm(db - q, axis=1)
    cand = np.flatnonzero(cached)[np.argsort(d[cached])[:K]]
    nr = d[cand].max()
    intr = np.flatnonzero(~cached & (d < nr))
    rows = []
    for n, b in balls.items():
        off = np.linalg.norm(q - b["c"])
        rows.append((off / b["r"] + nr / b["r"], n, off, b["r"]))
    return {"q": q, "d": d, "cand": cand, "nr": nr, "true_r": np.sort(d)[K - 1],
            "intr": intr, "seen": intr[sampled[intr]], "unseen": intr[~sampled[intr]],
            "rows": sorted(rows), "retention": np.isin(np.argsort(d)[:K], cand).mean()}


def source(i):
    return "+".join(n for n, b in balls.items() if i in b["m"])


def find_query():
    best = None
    for x in np.linspace(-1, 4, 101):
        for y in np.linspace(-2.5, 3.5, 121):
            q = np.array([x, y])
            s = analyze(q)
            src = [source(i) for i in s["cand"]]
            if min(sum(t == "A" for t in src), sum(t == "B" for t in src)) < 2:
                continue
            if len(s["seen"]) < 1 or len(s["unseen"]) < 1:
                continue
            gap = np.min(np.linalg.norm(db[s["seen"]][:, None] - db[s["unseen"]][None], axis=2))
            if gap < 0.3 * s["nr"]:
                continue
            score = abs(len(s["intr"]) - 2)
            if best is None or score < best[0]:
                best = (score, q)
    return best[1]


def _frame(ax, center, half_x, half_y, title):
    ax.set_xlim(center[0] - half_x, center[0] + half_x)
    ax.set_ylim(center[1] - half_y, center[1] + half_y)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11, loc="left")


def _balls(ax, label=True):
    for n, b in balls.items():
        ax.add_patch(Circle(b["c"], b["r"], fc="tab:blue", alpha=0.08, ec="tab:blue", lw=1.5))
        ax.plot(*b["c"], "+", color="tab:blue", ms=11, mew=2)
        if label:
            ax.annotate(f"ball {n}", b["c"] + np.array([0, b["r"]]), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=10, color="tab:blue")


def _circles(ax, s, labels=True):
    q = s["q"]
    ax.add_patch(Circle(q, s["nr"], fc="none", ec="tab:red", lw=2, ls="--"))
    ax.add_patch(Circle(q, s["true_r"], fc="none", ec="k", lw=1.3))
    ax.plot(*q, "*", ms=17, c="tab:red", mec="k", zorder=7)
    ax.plot([], [], "--", c="tab:red", lw=2, label="neighborhood: out to the 5th-closest CACHED point")
    ax.plot([], [], "-", c="k", lw=1.3, label="true top-5: out to the 5th-closest point of the DATABASE")


def _arrow(ax, a, b, text, color, offset=(0, 6), fs=9, at=0.5, ha="center"):
    ax.annotate("", b, a, arrowprops=dict(arrowstyle="->", color=color, lw=1.8))
    a, b = np.asarray(a), np.asarray(b)
    ax.annotate(text, a + at * (b - a), xytext=offset, textcoords="offset points",
                fontsize=fs, color=color, ha=ha,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))


def _candidates(ax, s, size=150):
    colors = {"A": "tab:blue", "B": "tab:green", "C": "tab:olive"}
    dd = s["d"][s["cand"]]
    for rank, j in enumerate(np.argsort(dd), 1):
        i = s["cand"][j]
        src = source(i)
        ax.scatter(*db[i], s=size, fc="none", ec=colors.get(src, "tab:cyan"), lw=2.2, zorder=5,
                   label=f"5 closest cached: from ball {src}")
        ax.annotate(str(rank), db[i], xytext=(6, 3), textcoords="offset points",
                    fontsize=9, weight="bold", zorder=6)


def draw_balls(ax, s):
    lo = np.min([b["c"] - b["r"] for b in balls.values()], axis=0)
    hi = np.max([b["c"] + b["r"] for b in balls.values()], axis=0)
    mid = (lo + hi) / 2
    _frame(ax, mid, (hi - lo)[0] / 2 + 0.4, (hi - lo)[1] / 2 + 0.9, "1. Per-ball stats")
    _balls(ax)
    ax.scatter(*db[cached].T, s=7, c="tab:blue", alpha=0.45, label="cached point")
    _candidates(ax, s)
    _circles(ax, s)
    q = s["q"]
    for n, b in balls.items():
        off = np.linalg.norm(q - b["c"])
        _arrow(ax, b["c"], q, f"offset {n} = {off:.2f}", "tab:purple")
        away = b["c"] - q
        ang = np.arctan2(away[1], away[0]) + 0.6
        _arrow(ax, b["c"], b["c"] + b["r"] * np.array([np.cos(ang), np.sin(ang)]),
               f"radius {n} = {b['r']:.2f}", "tab:blue")
    _arrow(ax, q, q + s["nr"] * np.array([0, 1]), "", "tab:red")
    ax.annotate(f"neighborhood radius = {s['nr']:.2f}", q + s["nr"] * np.array([0, 1]),
                xytext=(0, 5), textcoords="offset points", fontsize=9, color="tab:red",
                ha="center", bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.5))


def draw_shell(ax, s):
    q, nr = s["q"], s["nr"]
    half = 1.75 * nr
    _frame(ax, q, half * 1.35, half, f"2. Shell sample, zoom on the query (p = {int(100 * P)}%"
           " here, for visibility)")
    for b in balls.values():
        ax.add_patch(Wedge(b["c"], b["R"], 0, 360, width=b["R"] - b["r"], fc="tab:orange",
                           alpha=0.10, ec="tab:orange", lw=1, ls=":"))
    ax.plot([], [], "s", c="tab:orange", alpha=0.35, ms=10,
            label="shell: just outside a ball (sampled at rate p)")
    _balls(ax, label=False)
    ax.scatter(*db[cached].T, s=14, c="tab:blue", alpha=0.6)
    other = ~cached & ~sampled
    ax.scatter(*db[other].T, s=14, c="0.7", label="uncached point (not sampled)")
    ax.scatter(*db[sampled].T, s=55, marker="^", c="tab:orange", ec="k", lw=0.5,
               label="sampled uncached point")
    ax.scatter(*db[s["seen"]].T, s=260, marker="o", fc="none", ec="k", lw=2,
               label="intruder the sample sees", zorder=5)
    ax.scatter(*db[s["unseen"]].T, s=170, marker="X", c="0.45", ec="k", lw=0.5,
               label="intruder the sample misses", zorder=5)
    _circles(ax, s)


def feature_box(s):
    nr = s["nr"]
    L = ["THE 7 MODEL INPUTS for this query", "",
         f"neighborhood radius = {nr:.2f}   (red arrow / red dashed circle: distance to"
         " cached point #5)", "",
         "Per-ball stats (panel 1; offset = purple arrow, radius = blue arrow), 3 best-fitting"
         " balls (fit = rel. offset + rel. size):"]
    for rank, (fit, n, off, r) in enumerate(s["rows"], 1):
        L.append(f"  {rank}. ball {n}:  relative offset = {off:.2f} / {r:.2f} = [{off / r:.2f}]"
                 f"   relative size = {nr:.2f} / {r:.2f} = [{nr / r:.2f}]   (fit {fit:.2f})")
    L += ["", f"Shell sample (panel 2): intruder count = sampled points (orange triangles) inside"
          f" the red dashed circle = [{len(s['seen'])}]", "",
          "Bracketed numbers are the inputs. Outcome to predict: "
          f"{len(s['intr'])} intruders inside the neighborhood -> true retention = "
          f"{s['retention']:.1f} (the sample saw {len(s['seen'])} of them)"]
    return "\n".join(L)


def add_args(ap):
    ap.set_defaults(run=run)


def run(args):
    out = common.RESULTS / "summary" / "feature_diagram.png"
    s = analyze(find_query())
    fig = plt.figure(figsize=(17, 10.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.1, 0.95])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    draw_balls(axes[0], s)
    draw_shell(axes[1], s)
    t = fig.add_subplot(gs[1, :])
    t.axis("off")
    t.text(0.01, 0.95, feature_box(s), family="monospace", fontsize=9.5, va="top",
           transform=t.transAxes, bbox=dict(fc="#fff6cc", ec="#c9a227", lw=1.5, pad=0.6))
    seen = {}
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    t.legend(list(seen.values()), list(seen.keys()), loc="upper right", fontsize=8.5,
             frameon=False, ncol=1)
    fig.suptitle("Balls are cached past queries with their exact top-K: every database point "
                 "inside a ball is cached. Uncached points inside\nthe neighborhood (red dashed) "
                 "are intruders: each pushes a cached point out of the true top-5.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=0.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")
    print(feature_box(s))
