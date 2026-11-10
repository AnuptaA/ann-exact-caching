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
TECHS = [("ours", "Ours", "C0", lambda n: n.startswith("ours_")),
         ("lru_sync", "LRU", "C1", lambda n: n.startswith("lru_sync_")),
         ("lru_bg", "LRU (bg. upd.)", "C1", lambda n: n.startswith("lru_bg_")),
         ("aker", "Aker", "C2", lambda n: n.startswith("aker_")),
         ("qvcache", "QVCache", "C3", lambda n: n.startswith("qvcache_"))]


def load(prep, budget):
    d = RES / "timing_v2" / prep
    out = {}
    for f in sorted(d.glob(f"*_b{budget:g}.json")) if d.exists() else []:
        if f.name.startswith(("greedy_state", "state_")):
            continue
        for key, _, _, match in TECHS:
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
            xs = np.arange(len(TECHS))
            for x, (key, label, color, _) in zip(xs, TECHS):
                r = data.get(key)
                if r is None:
                    ax.text(x, 0.05, "pending", rotation=90, ha="center", va="bottom", fontsize=7, color="grey")
                    continue
                sp = r["speedup"]["overall"]
                ret = ret_of(r)
                ok = ret is not None and ret >= T - TOL
                quiet = r.get("environment", {}).get("verdict") == "quiet"
                ax.bar(x, sp, color=color if ok else "white", edgecolor=color if quiet else "k",
                       linewidth=1.5, hatch=None if quiet else "//")
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
            ax.axhline(1.0, color="grey", lw=0.8, ls="--")
            ax.set_xticks(xs)
            ax.set_xticklabels([t[1] for t in TECHS], rotation=30, ha="right", fontsize=8)
            ax.set_title(f"SPACEV {title}, cache {100 * b:g}%", fontsize=9)
            if j == 0:
                ax.set_ylabel("speedup over DiskANN alone")
            ax.set_ylim(0, max([ax.get_ylim()[1], 1.2]) * 1.15)
    fig.suptitle("Cache + DiskANN (recall@10 0.95), mean latency over all requests. Bar labels: hit rate, "
                 "recall vs exact.\nTick: 1/(1-h) limit; diamond: DiskANN with equal-memory node cache; "
                 "hollow: retention of hits < 0.89; hatched: machine not quiet", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = RES / "summary" / "timing_draft_v1.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    hdr = ("| trace | budget | technique | hit % | ret. hits | recall all | DiskANN ms | cache+DiskANN ms | "
           "hit ms | miss ms | p99 ms | DiskANN p99 | speedup | 1/(1-h) | env |")
    lines = [hdr, "|" + "---|" * 15]
    fmt = lambda v: "-" if v is None else (f"{v:.3g}" if isinstance(v, float) else str(v))  # noqa: E731
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows]
    (RES / "summary" / "timing_table.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out} and timing_table.md ({len(rows)} runs)")


if __name__ == "__main__":
    main()
