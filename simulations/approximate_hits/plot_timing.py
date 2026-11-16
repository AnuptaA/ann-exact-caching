#!/usr/bin/env python3
"""
Timing plot, draft (context doc 9k): end-to-end speedup over DiskANN alone of each technique in
front of the same DiskANN (recall@10 0.95), SPACEV Zipf traces, cache budgets 1% and 10%.

Per panel (one trace, one budget), one bar per technique = mean latency of DiskANN alone / mean
latency of cache + DiskANN, over the same 2,000 requests (each run measures its own DiskANN-alone
arm, right after its cache arm). Marks: the limit 1 / (1 - h + audited share) a free cache would
reach (black tick), and DiskANN with its node cache given the technique's memory (grey diamond).
Under each bar: hit rate, recall vs exact over all requests; a hollow bar = the technique missed
the target (mean retention of hits < T - 0.01, the simulator's definition where available).
Also writes a table (results/latest/timing_table.md) with per-hit / per-miss cost and p99.

  PYTHONPATH=. .venv/bin/python simulations/approximate_hits/plot_timing.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = Path("simulations/approximate_hits/results")
T, TOL = 0.9, 0.01
TRACES = [("Zipf 0.3", "spacev-aker-0.3"), ("Zipf 0.6", "spacev-aker-0.6"), ("Zipf 0.99", "spacev-aker-0.99"),
          ("ESCI (no repeated queries)", "esci1m"),
          ("TriviaQA, Zipf 0.3", "triviaqa-ip-0.3"), ("TriviaQA, Zipf 0.6", "triviaqa-ip-0.6"),
          ("TriviaQA, Zipf 0.99", "triviaqa-ip-0.99"), ("SPACEV real query log", "spacev-real"),
          ("TripClick real health-search log", "tripclick-ip"), ("MedRAG-Zipf (Proximity)", "medrag-ip")]
BUDGETS = [0.003, 0.01, 0.03, 0.1]
SIZES = {"medrag-ip": [0.0001, 0.0003, 0.001, 0.003]}   # 0.3% already holds MedRAG-Zipf's working set
# LRU's cache update (DiskANN top-500 search for the new entry, insert, evictions) is charged as
# background work (decided with Jialin, 11-11); the sync runs stay in the table only
TECHS = [("ours", "Ours", "C0", lambda n: n.startswith("ours_")),
         ("ours_drift", "Ours (drift handling)", "C0", lambda n: n.startswith("oursdrift_")),
         ("ours_online", "Ours (continuous greedy)", "C0", lambda n: n.startswith("oursonline_")),
         ("lru_bg", "LRU", "C1", lambda n: n.startswith(("lru_bg_", "lru_s"))),
         ("aker", "Aker", "C2", lambda n: n.startswith("aker_")),
         ("aker_paper", "Aker (paper default)", "C2", lambda n: n.startswith("akerpaper_")),
         ("qvcache_paper", "QVCache (paper's D)", "C3", lambda n: n.startswith("qvpaper_")),
         ("qvcache", "QVCache", "C3", lambda n: n.startswith("qvcache_"))]


TABLE_ONLY = [("lru_sync", "LRU (sync update)", lambda n: n.startswith("lru_sync_"))]
HEADLINE_KEY = {"ours": "ours", "ours_drift": "ours_drift", "ours_online": "ours_online", "lru_bg": "lru", "lru_sync": "lru", "aker": "aker", "qvcache": "qvcache",
                "aker_paper": "aker_paper", "qvcache_paper": "qvcache_paper"}


VERSION, EVERY = "timing_v4", 50     # full-stream runs, 1 request in EVERY timed; tuned DiskANN (doc 9u)


def load(prep, budget, every=None):
    every = EVERY if every is None else every
    d = RES / VERSION / prep
    out = {}
    for f in sorted(d.glob(f"*_s{every}_b{budget:g}.json")) if d.exists() else []:
        if f.name.startswith(("greedy_state", "state_")):
            continue
        for key, match in [(t[0], t[3]) for t in TECHS] + [(t[0], t[2]) for t in TABLE_ONLY]:
            if match(f.stem):
                r = json.loads(f.read_text())
                r["file"] = f.stem
                if r.get("speedup") is None:      # run with --no-backend-arms (doc 9ad)
                    arm = d / f"backend_s{every}.json"
                    if not arm.exists():
                        continue
                    a = json.loads(arm.read_text())
                    r["backend"] = a["backend"]
                    if f"{budget:g}" in a["per_budget"]:
                        r["backend_mem"] = a["per_budget"][f"{budget:g}"]
                    b0 = r["backend"]["mean_ms"]
                    r["speedup"] = {"overall": b0 / r["cache"]["mean_ms"],
                                    "limit_1_over_1_minus_h": 1 / max(1 - r["hit_rate"] + r["audited_share"], 1e-9),
                                    "vs_equal_memory": (r["backend_mem"]["mean_ms"] / r["cache"]["mean_ms"])
                                    if "backend_mem" in r else None}
                out[key] = r
    return out


def ret_of(r):
    """Mean retention of hits: the simulator's definition if recorded, else vs DiskANN's answer."""
    # the served answer's overlap with DiskANN's top-k (doc 9o), the same measure for every technique
    return r.get("retention_vs_backend_hits")


def main():
    """Line plot in the headline's style: per trace, speedup over DiskANN alone vs cache size, one
    line per technique (filled marker: the setting meets the target in the headline run; hollow:
    it does not); grey dashed: DiskANN given the memory of the cache (mean over techniques);
    strip below: recall vs exact over all requests (grey: DiskANN alone)."""
    from matplotlib.gridspec import GridSpec
    from simulations.approximate_hits.plot_headline import STYLE, JITTER
    ncol = 4
    nrow = (len(TRACES) + ncol - 1) // ncol
    fig = plt.figure(figsize=(5.0 * ncol, 4.4 * nrow + 0.8))
    gs = GridSpec(2 * nrow, ncol, height_ratios=[3.2, 1.2] * nrow, hspace=0.12, wspace=0.28, figure=fig,
                  top=0.9, bottom=0.1)
    rows = []
    for j, (title, prep) in enumerate(TRACES):
        ax = fig.add_subplot(gs[2 * (j // ncol), j % ncol])
        st = fig.add_subplot(gs[2 * (j // ncol) + 1, j % ncol], sharex=ax)
        bs = SIZES.get(prep, BUDGETS)
        per_b = {b: load(prep, b) for b in bs}
        eq_pts = []
        for key, label, _, _ in TECHS:
            hk = HEADLINE_KEY[key]
            name, color, marker, lw = STYLE[hk]
            pts = []
            for b in bs:
                r = per_b[b].get(key)
                if r is None:
                    continue
                # target met: this run's own mean retention of hits (whole stream, real DiskANN)
                rt = ret_of(r)
                ok = rt is not None and rt >= T - TOL
                pts.append((b * JITTER[hk], r["speedup"]["overall"], rt if rt is not None else np.nan, ok))
                if r.get("backend_mem"):
                    eq_pts.append((b, r["backend"]["mean_ms"] / r["backend_mem"]["mean_ms"]))
                c = r["cache"]
                lost = sum(1 for x in r.get("trace", []) if x.get("labels") is not None and x.get("lam") is None)
                if lost:
                    label = f"{label} (cutoff lost {lost}k requests)"
                rows.append((title, f"{100 * b:g}%", label, 100 * r["hit_rate"], ret_of(r),
                             r["recall_vs_exact_all"], r["backend"]["mean_ms"], c["mean_ms"],
                             (c.get("hits_ms") or {}).get("mean_ms"), (c.get("misses_ms") or {}).get("mean_ms"),
                             c["p99_ms"], r["backend"]["p99_ms"], r["speedup"]["overall"],
                             r["speedup"]["limit_1_over_1_minus_h"], r.get("environment", {}).get("verdict")))
            if not pts:
                continue
            xs, ys, rs, oks = zip(*pts)
            ax.plot(xs, ys, color=color, lw=lw, zorder=3, label=name if j == 0 else None,
                    ls=(0, (5, 2.5)) if hk.endswith("_paper") else "-")
            st.plot(xs, rs, color=color, lw=1, alpha=0.7)
            for x, y, rr, ok in pts:
                kw = dict(marker=marker, ms=8 if hk == "ours" else 6.5, color=color, zorder=4,
                          mfc=color if ok else "white", mec=color, mew=1.5)
                ax.plot([x], [y], **kw)
                st.plot([x], [rr], **dict(kw, ms=4.5))
        if eq_pts:
            eb = sorted(set(b for b, _ in eq_pts))
            ax.plot(eb, [np.mean([v for b2, v in eq_pts if b2 == b]) for b in eb], color="grey", ls="--",
                    lw=1.2, marker="x", ms=5, label="DiskANN, cache's memory as node cache" if j == 0 else None)
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(title if not prep.startswith("spacev-aker") else f"SPACEV, {title}", fontsize=10,
                     loc="left", fontweight="bold")
        if j % ncol == 0:
            ax.set_ylabel("speedup over DiskANN alone\n(mean latency, all requests)")
            st.set_ylabel("retention\nof hits", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelbottom=False)
        st.axhline(0.9, color="grey", lw=1, ls="--")       # the target
        st.set_ylim(0.30, 1.0)
        st.grid(alpha=0.3)
        st.set_xticks(bs)
        st.set_xticklabels([f"{100 * b:g}%" for b in bs], fontsize=8)
        st.minorticks_off()
        st.set_xlim(bs[0] * 0.75, bs[-1] * 1.35)
        st.set_xlabel("cache size (fraction of the dataset)", fontsize=9)
    fig.legend(loc="lower center", ncol=5, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.text(0.5, 0.955, "DiskANN tuned per workload (search list x beam width, recall@10 0.95 or 0.9 if 0.95 > 100 ms), same index for every technique; whole stream after "
             "warm-up, 1 in 50 requests timed end to end (DiskANN baselines with a 10 ms gap between searches; technique runs composed: backend calls charged the request's measured DiskANN time, hit path with a per-cache search list); cache updates are background work.\n"
             "Filled: mean retention of hits (served answer vs DiskANN's top-10) >= 0.89 in this run; hollow: below. "
             "Strip: mean retention of hits (served answer vs DiskANN's top-10; grey dashed: the target 0.9). Recall over all requests is in timing_table.md.",
             ha="center", fontsize=7.5)
    out = RES / "latest" / "timing.png"          # latest plots only (11-14)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    hdr = ("| trace | budget | technique | hit % | ret. hits | recall all | DiskANN ms | cache+DiskANN ms | "
           "hit ms | miss ms | p99 ms | DiskANN p99 | speedup | 1/(1-h) | env |")
    lines = [hdr, "|" + "---|" * 15]
    fmt = lambda v: "-" if v is None else (f"{v:.3g}" if isinstance(v, float) else str(v))  # noqa: E731
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows]
    # sample-rate check: the same runs timed 1 in 20 / 10 / 5
    lines += ["", "Sample-rate check (mean ms, cache + DiskANN / DiskANN alone / speedup; whole-run hit rate)", ""]
    for z, b in (("0.6", 0.01), ("0.99", 0.1)):
        for key, label in (("ours", "Ours"), ("lru_bg", "LRU")):
            cells = []
            for every in (20, 10, 5):
                r = load(f"spacev-aker-{z}", b, every).get(key)
                cells.append(f"1/{every}: " + ("pending" if r is None else
                             f"{r['cache']['mean_ms']:.2f} / {r['backend']['mean_ms']:.2f} / "
                             f"{r['speedup']['overall']:.3f} (h {100 * r['hit_rate']:.1f}%, n {r['n_timed']})"))
            lines.append(f"- Zipf {z}, {100 * b:g}%, {label}: " + "; ".join(cells))
    (RES / "latest" / "timing_table.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out} and timing_table.md ({len(rows)} runs)")


if __name__ == "__main__":
    main()
