#!/usr/bin/env python3
"""
DRAFT of the headline figure (context doc 9e): hit rate against cache size, one panel per
workload, one line per technique, every technique compared at the SAME promised recall.

Per panel (workload) and budget, each technique contributes one point:
  - the best hit rate among its settings that reach retention >= T - TOL (T = 0.9, TOL = 0.01
    so noise does not flip a 0.899); if no setting reaches it, the setting with the highest
    retention, drawn as a HOLLOW marker with its retention printed;
  - ours and LRU: mean of the seeds that exist (bars = min / max); the baselines' settings are
    tuned and measured on seed 0 (fresh-seed re-runs of the best settings are still pending).
A thin strip under each panel shows retention - T for the plotted points (how far off).
The right axis converts hit rate into the end-to-end speedup bound 1 / (1 - hit rate).
Real Aker code where we have it (SPACEV, TriviaQA, real query log); the reimplementation of
Aker on ESCI; QVCache is always our reimplementation (no public code).

  .venv/bin/python simulations/approximate_hits/plot_headline.py [--out FILE]
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec

RES = Path("simulations/approximate_hits/results")
T, TOL = 0.9, 0.01
BUDGETS = [0.01, 0.1]
STYLE = {"ours": ("Ours (calibrated)", "#D55E00", "o", 2.6),
         "lru": ("LRU", "#6B6B6B", "^", 1.6),
         "aker": ("Aker", "#0072B2", "D", 1.6),
         "qvcache": ("QVCache", "#009E73", "s", 1.6)}
JITTER = {"ours": 1.0, "lru": 0.93, "aker": 1.07, "qvcache": 1.14}     # x offsets (log axis)
# (title, prep, stream, uses real Aker?)
LAYOUT = [
    [("SPACEV, Zipf 0.3", "spacev-aker-0.3", "trace", True),
     ("SPACEV, Zipf 0.6", "spacev-aker-0.6", "trace", True),
     ("SPACEV, Zipf 0.99", "spacev-aker-0.99", "trace", True)],
    [("TriviaQA (dpr), Zipf 0.3", "triviaqa-dpr-0.3", "trace", True),
     ("TriviaQA (dpr), Zipf 0.6", "triviaqa-dpr-0.6", "trace", True),
     ("TriviaQA (dpr), Zipf 0.99", "triviaqa-dpr-0.99", "trace", True)],
    [("ESCI (no repeated queries)", "esci1m", "random", False),
     ("SPACEV real query log", "spacev-real", "trace", True),
     None],
]
OURS_NAMES = ["greedy_score_rebuild", "greedy_none", "static_default"]
LRU_NAMES = ["lru_k10_none", "lru_k20_none", "lru_k20_all"]     # configurations kept at every budget


def sim_stats(path):
    """(hit rate %, mean retention of hits) of a simulator result, or None."""
    if not path.exists():
        return None
    log = json.loads(path.read_text())["log"]
    n = sum(b["n"] for b in log)
    s = sum(b["served"] for b in log)
    if n == 0 or s == 0:
        return None
    return 100.0 * s / n, sum(b["served_retention_sum"] for b in log) / s


def sim_dir(prep, stream, budget, seed):
    default = 0.1 if prep.startswith("esci") else 0.01
    suffix = "" if budget == default else f"_b{budget:g}"
    return RES / "online" / prep / f"r2_{stream}{suffix}_s{seed}"


def candidate(name, runs):
    """runs: list of (hit, ret) over seeds -> a candidate setting."""
    h = [r[0] for r in runs]
    r = [r[1] for r in runs]
    return {"name": name, "hit": float(np.mean(h)), "hit_lo": min(h), "hit_hi": max(h),
            "ret": float(np.mean(r)), "n": len(runs)}


def pick(cands):
    """Best hit rate among settings reaching T - TOL; else the highest retention (hollow)."""
    if not cands:
        return None
    ok = [c for c in cands if c["ret"] >= T - TOL]
    best = max(ok, key=lambda c: c["hit"]) if ok else max(cands, key=lambda c: c["ret"])
    return dict(best, ok=bool(ok), n_settings=len(cands))


def collect(prep, stream, budget, real_aker):
    out = {}
    # ours and LRU: seeds 0 and 1
    for tech, names in (("ours", OURS_NAMES), ("lru", LRU_NAMES)):
        cands = []
        for name in names:
            runs = [sim_stats(sim_dir(prep, stream, budget, s) / f"{name}.json") for s in (0, 1)]
            runs = [r for r in runs if r]
            if runs:
                cands.append(candidate(name, runs))
        if tech == "ours" and cands:
            cands = cands[:1]       # the first available form of our method (they agree on stable streams)
        out[tech] = pick(cands)
    d0 = sim_dir(prep, stream, budget, 0)
    # QVCache (reimplementation); Aker: real code if available, else the reimplementation
    cands = []
    for f in sorted(d0.glob("qvcache_*.json")) if d0.exists() else []:
        r = sim_stats(f)
        if r:
            cands.append(candidate(f.stem, [r]))
    out["qvcache"] = pick(cands)
    cands = []
    if real_aker:
        for f in sorted((RES / "aker_real" / prep / f"trace_b{budget:g}").glob("at*_d*.json")):
            s = json.loads(f.read_text())["summary"]
            cands.append(candidate(f.stem, [(100 * s["hit_rate"], s["retention_vs_backend"])]))
    else:
        for f in sorted(d0.glob("aker_*.json")) if d0.exists() else []:
            r = sim_stats(f)
            if r:
                cands.append(candidate(f.stem, [r]))
    out["aker"] = pick(cands)
    return out


def draw_panel(fig, outer, title, data, ymax_pad=1.18, speedup_axis=False, first_col=False):
    inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer, height_ratios=[3.3, 1], hspace=0.07)
    ax = fig.add_subplot(inner[0])
    st = fig.add_subplot(inner[1], sharex=ax)
    for tech in ("lru", "qvcache", "aker", "ours"):
        label, col, mk, lw = STYLE[tech]
        pts = [(b * JITTER[tech], data[b][tech]) for b in BUDGETS if data[b].get(tech)]
        if not pts:
            continue
        for (x0, p0), (x1, p1) in zip(pts[:-1], pts[1:]):
            both = p0["ok"] and p1["ok"]
            ax.plot([x0, x1], [p0["hit"], p1["hit"]], color=col, lw=lw if both else 1.1,
                    ls="-" if both else (0, (1.5, 2.5)), alpha=0.9 if both else 0.55,
                    zorder=3 if tech == "ours" else 2)
            st.plot([x0, x1], [p0["ret"] - T, p1["ret"] - T], color=col, lw=0.8, alpha=0.5)
        for x, p in pts:
            face = col if p["ok"] else "white"
            ms = 8 if tech == "ours" else 6.5
            if p["n"] > 1:
                ax.errorbar([x], [p["hit"]], yerr=[[p["hit"] - p["hit_lo"]], [p["hit_hi"] - p["hit"]]],
                            color=col, capsize=2, lw=1)
            ax.plot([x], [p["hit"]], marker=mk, ms=ms, mfc=face, mec=col, mew=1.6, ls="none",
                    zorder=5, alpha=1.0 if p["ok"] else 0.6)
            st.plot([x], [p["ret"] - T], marker=mk, ms=5, mfc=face, mec=col, mew=1.2, ls="none")
            if not p["ok"]:
                ax.annotate(f"recall {p['ret']:.2f}".replace("0.", "."), (x, p["hit"]), xytext=(8, -2),
                            textcoords="offset points", ha="left", fontsize=6.5, color=col)
    ax.set_xscale("log")
    ax.set_xlim(0.0066, 0.17)
    allhit = [data[b][t]["hit_hi"] for b in BUDGETS for t in STYLE if data[b].get(t)]
    ax.set_ylim(0, max(allhit + [1.0]) * ymax_pad)
    ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
    ax.tick_params(labelbottom=False, labelsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    if first_col:
        ax.set_ylabel("hit rate (%)", fontsize=9)
    if speedup_axis:
        sec = ax.secondary_yaxis("right", functions=(lambda h: 1.0 / (1.0 - np.minimum(h, 99.0) / 100.0),
                                                      lambda s: 100.0 * (1.0 - 1.0 / np.maximum(s, 1.0))))
        ymax = ax.get_ylim()[1]
        ticks = [v for v in (1.1, 1.25, 1.5, 2, 3, 5, 10) if 100 * (1 - 1 / v) <= min(ymax, 98)]
        sec.set_yticks(ticks)
        sec.set_yticklabels([f"{v:g}x" for v in ticks])
        sec.set_ylabel("speedup bound 1/(1-h)", fontsize=8)
        sec.tick_params(labelsize=7)
    st.axhspan(-TOL, 0.2, color="#2ca02c", alpha=0.08, lw=0)
    st.axhline(0, color="black", lw=0.8)
    st.set_ylim(-0.36, 0.09)
    st.set_yticks([-0.3, -0.15, 0.0])
    st.set_yticklabels(["-.30", "-.15", "0"])
    st.tick_params(labelsize=7)
    st.set_xticks(BUDGETS)
    st.set_xticklabels([f"{b * 100:g}%" for b in BUDGETS], fontsize=8)
    st.minorticks_off()
    st.grid(alpha=0.25, lw=0.5)
    if first_col:
        st.set_ylabel("recall - 0.9", fontsize=8)
    return ax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RES / "summary" / "headline_draft.png"))
    args = ap.parse_args()
    fig = plt.figure(figsize=(14.5, 12.2))
    outer = fig.add_gridspec(3, 3, hspace=0.34, wspace=0.30, left=0.055, right=0.955, top=0.915,
                             bottom=0.065)
    for r, row in enumerate(LAYOUT):
        for c, cell in enumerate(row):
            if cell is None:
                continue
            title, prep, stream, real = cell
            data = {b: collect(prep, stream, b, real) for b in BUDGETS}
            last = (c == 2) or (r == 2 and c == 1)
            draw_panel(fig, outer[r, c], title, data, speedup_axis=last, first_col=(c == 0))
    # legend and notes in the free cell
    ax = fig.add_subplot(outer[2, 2])
    ax.axis("off")
    handles = [plt.Line2D([], [], color=col, marker=mk, lw=lw, ms=7, label=lab)
               for lab, col, mk, lw in STYLE.values()]
    handles += [plt.Line2D([], [], color="k", marker="o", ls="none", mfc="k", label="reaches recall 0.9 (within .01)"),
                plt.Line2D([], [], color="k", marker="o", ls="none", mfc="white", mew=1.6,
                           label="no setting reaches it: faded, dotted line,\nbest-recall setting, its recall printed")]
    ax.legend(handles=handles, loc="upper left", fontsize=9, frameon=False, title="technique", title_fontsize=9)
    ax.text(0.0, 0.0,
            "How to read: each point is the best hit rate a technique gets at the\n"
            "promised recall 0.9 (average over hits of the share of the exact top-10\n"
            "returned). Strips: recall - 0.9; green band = reached.\n"
            "Ours / LRU: mean of 2 seeds (bars: min-max); LRU: best of K 10/20.\n"
            "Aker / QVCache: best of the tuned grid on seed 0 (selection and\n"
            "measurement share the run; fresh-seed re-runs pending).\n"
            "Aker = real code (ESCI: reimplementation); QVCache = reimplementation.",
            fontsize=7.8, va="bottom", transform=ax.transAxes)
    fig.suptitle("DRAFT: hit rate at the same promised recall (0.9), exact backend, k = 10\n"
                 "only two cache sizes exist so far (1% and 10% of the dataset); missing points are runs not finished yet",
                 fontsize=11, y=0.985)
    fig.text(0.5, 0.025, "cache size (fraction of the dataset, cached vectors)", ha="center", fontsize=10)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
