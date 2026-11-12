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
Also writes a table (results/summary/timing_table.md) with per-hit / per-miss cost and p99.

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
TRACES = [("Zipf 0.99", "spacev-aker-0.99"), ("Zipf 0.6", "spacev-aker-0.6"), ("Zipf 0.3", "spacev-aker-0.3")]
BUDGETS = [0.01, 0.1]
# LRU's cache update (DiskANN top-500 search for the new entry, insert, evictions) is charged as
# background work (decided with Jialin, 11-11); the sync runs stay in the table only
TECHS = [("ours", "Ours", "C0", lambda n: n.startswith("ours_")),
         ("lru_bg", "LRU", "C1", lambda n: n.startswith(("lru_bg_", "lru_s"))),
         ("aker", "Aker", "C2", lambda n: n.startswith("aker_")),
         ("qvcache", "QVCache", "C3", lambda n: n.startswith("qvcache_"))]


TABLE_ONLY = [("lru_sync", "LRU (sync update)", lambda n: n.startswith("lru_sync_"))]
HEADLINE_KEY = {"ours": "ours", "lru_bg": "lru", "lru_sync": "lru", "aker": "aker", "qvcache": "qvcache"}


VERSION, EVERY = "timing_v3", 20     # full-stream runs, 1 request in EVERY timed (doc 9m)


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
                out[key] = r
    return out


def ret_of(r):
    """Mean retention of hits: the simulator's definition if recorded, else vs DiskANN's answer."""
    v = r.get("retention_sim_hits")
    return v if v is not None else r.get("retention_vs_backend_hits")


def main():
    fig, axes = plt.subplots(len(BUDGETS), len(TRACES), figsize=(4.2 * len(TRACES), 3.6 * len(BUDGETS)),
                             squeeze=False)
    rows = []
    for i, b in enumerate(BUDGETS):
        for j, (title, prep) in enumerate(TRACES):
            ax = axes[i][j]
            data = load(prep, b)
            from simulations.approximate_hits.plot_headline import collect
            head = collect(prep, "trace", b, True)
            xs = np.arange(len(TECHS))
            for x, (key, label, color, _) in zip(xs, TECHS):
                r = data.get(key)
                if r is None:
                    ax.text(x, 0.05, "pending", rotation=90, ha="center", va="bottom", fontsize=7, color="grey")
                    continue
                sp = r["speedup"]["overall"]
                ret = ret_of(r)
                # target met = the chosen setting meets it over the whole headline run (the timing
                # window has few hits: its retention is noisy)
                h = head.get(HEADLINE_KEY[key])
                ok = bool(h and h["ok"])
                ax.bar(x, sp, color=color if ok else "white", edgecolor=color, linewidth=1.5)
                lim = r["speedup"]["limit_1_over_1_minus_h"]
                ax.plot([x - 0.35, x + 0.35], [lim, lim], color="k", lw=1)
                if r["speedup"].get("vs_equal_memory"):
                    eq = r["backend"]["mean_ms"] / r["backend_mem"]["mean_ms"]
                    ax.plot(x, eq, marker="D", color="grey", ms=4)
                ax.text(x, sp, f"{100 * r['hit_rate']:.0f}%\n{r['recall_vs_exact_all']:.2f}",
                        ha="center", va="bottom", fontsize=7)
                c = r["cache"]
                rows.append((title, f"{100 * b:g}%", label, 100 * r["hit_rate"], ret,
                             r["recall_vs_exact_all"], r["backend"]["mean_ms"], c["mean_ms"],
                             (c.get("hits_ms") or {}).get("mean_ms"), (c.get("misses_ms") or {}).get("mean_ms"),
                             c["p99_ms"], r["backend"]["p99_ms"], sp, lim,
                             r.get("environment", {}).get("verdict")))
            for key, label, _ in TABLE_ONLY:
                r = data.get(key)
                if r:
                    c = r["cache"]
                    rows.append((title, f"{100 * b:g}%", label, 100 * r["hit_rate"], ret_of(r),
                                 r["recall_vs_exact_all"], r["backend"]["mean_ms"], c["mean_ms"],
                                 (c.get("hits_ms") or {}).get("mean_ms"), (c.get("misses_ms") or {}).get("mean_ms"),
                                 c["p99_ms"], r["backend"]["p99_ms"], r["speedup"]["overall"],
                                 r["speedup"]["limit_1_over_1_minus_h"], r.get("environment", {}).get("verdict")))
            ax.axhline(1.0, color="grey", lw=0.8, ls="--")
            ax.set_xticks(xs)
            ax.set_xticklabels([t[1] for t in TECHS], rotation=30, ha="right", fontsize=8)
            ax.set_title(f"SPACEV {title}, cache {100 * b:g}%", fontsize=9)
            if j == 0:
                ax.set_ylabel("speedup over DiskANN alone")
            ax.set_ylim(0, max([ax.get_ylim()[1], 1.2]) * 1.15)
    fig.suptitle("Cache + DiskANN (recall@10 0.95), mean latency over all requests. Bar labels: hit rate, "
                 "recall vs exact.\nTick: 1/(1-h) limit; diamond: DiskANN with equal-memory node cache; "
                 "hollow: setting misses the target (headline run); cache updates counted as background work. Whole stream after warm-up, 1 in 20 requests timed", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = RES / "summary" / "timing_draft_v2.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
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
    (RES / "summary" / "timing_table.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out} and timing_table.md ({len(rows)} runs)")


if __name__ == "__main__":
    main()
