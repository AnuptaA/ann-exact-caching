#!/usr/bin/env python3
"""Throughput plot (doc 9as / 9at): closed-loop QPS with 16 clients over real DiskANN (8 processes), relative to DiskANN
alone in the same replay, per workload and cache size. Ours: continuous greedy with the mini cache, I/O-complete (misses
return the deep list, audited hits are real background searches, chosen candidates' vectors re-read from disk). Best
alternative: Aker or QVCache (best hit rate at the target in the timing runs). Node cache: DiskANN with the cache's memory as
extra node cache (its ratio from a separate replay; it does not depend on any cache's hit rate).

  PYTHONPATH=. .venv/bin/python simulations/approximate_hits/plot_throughput.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = Path("simulations/approximate_hits/results")
T = RES / "throughput"
PANELS = [("SPACEV, Zipf 0.6", "spacev-aker-0.6"), ("SPACEV, Zipf 0.99", "spacev-aker-0.99"),
          ("TriviaQA, Zipf 0.6", "triviaqa-ip-0.6"), ("TriviaQA, Zipf 0.99", "triviaqa-ip-0.99"),
          ("MedRAG-Zipf (Proximity)", "medrag-ip"), ("TripClick real health-search log", "tripclick-ip")]


def arms(path):
    if not path.exists():
        return {}
    return {r["arm"]: r for r in json.loads(path.read_text())}


def cell(w, b):
    """{"ours", "alt", "alt_name", "node", "hit_ours", "hit_alt"} as ratios to DiskANN alone, or None."""
    out = {}
    main = arms(T / f"replay_final_{w}_b{b:g}.json")          # the final sweep (with re-reads)
    if main:
        none = main["none"]["qps"]
        out["ours"], out["hit_ours"] = main["ours"]["qps"] / none, main["ours"]["hit_rate"]
        alt = [k for k in main if k not in ("none", "ours")]
        if alt:
            out["alt"], out["alt_name"], out["hit_alt"] = main[alt[0]]["qps"] / none, alt[0], main[alt[0]]["hit_rate"]
    else:                                                      # the 10% cells: whole-stream run, ours re-run with re-reads
        full = arms(T / f"replay_io_full_{w}_b{b:g}.json") or arms(T / f"replay_io_{w}_b{b:g}.json")
        rr = arms(T / f"replay_io_reread_{w}_b{b:g}.json")
        if not full:
            return None
        none = full["none"]["qps"]
        o = rr.get("ours") or full.get("ours")
        out["ours"], out["hit_ours"], out["reread"] = o["qps"] / none, o["hit_rate"], "ours" in rr
        alt = [k for k in full if k not in ("none", "ours")]
        if alt:
            out["alt"], out["alt_name"], out["hit_alt"] = full[alt[0]]["qps"] / none, alt[0], full[alt[0]]["hit_rate"]
    for f in (T / f"replay_final_{w}_node_b{b:g}.json", T / f"replay_io_{w}_node_int8_b{b:g}.json",
              T / f"replay_io_{w}_node_b{b:g}.json"):
        nd = arms(f)
        if nd:
            base = none
            out["node"] = nd["none"]["qps"] / base
            break
    return out


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), squeeze=False)
    rows = []
    for ax, (title, w) in zip(axes.ravel(), PANELS):
        sizes = [0.0001, 0.0003, 0.001, 0.003] if w == "medrag-ip" else [0.003, 0.01, 0.03, 0.1, 0.3]
        pts = {k: [] for k in ("ours", "alt", "node")}
        for b in sizes:
            c = cell(w, b)
            if not c:
                continue
            for k in pts:
                if k in c:
                    pts[k].append((b, c[k]))
            rows.append((title, f"{100 * b:g}%", c.get("ours"), c.get("hit_ours"), c.get("alt_name"), c.get("alt"),
                         c.get("hit_alt"), c.get("node")))
        for k, lab, col, mk in (("ours", "Ours (mini cache, I/O-complete)", "#D55E00", "o"),
                                ("alt", "best of Aker / QVCache", "#0072B2", "D"),
                                ("node", "DiskANN, cache's memory as node cache", "grey", "x")):
            if pts[k]:
                xs, ys = zip(*pts[k])
                ax.plot(xs, ys, marker=mk, color=col, lw=2 if k == "ours" else 1.4, ls="--" if k == "node" else "-",
                        label=lab)
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{100 * b:g}%" for b in sizes])
        ax.minorticks_off()
        ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
        ax.set_ylabel("throughput / DiskANN alone")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Throughput, 16 clients, real DiskANN (disk-bound), relative to DiskANN alone", fontsize=11)
    fig.tight_layout()
    out = RES / "latest" / "throughput.png"
    fig.savefig(out, dpi=140)
    fmt = lambda v: "-" if v is None else (f"{v:.3g}" if isinstance(v, float) else str(v))  # noqa: E731
    lines = ["| workload | size | ours | ours hit | alternative | alt | alt hit | node cache |", "|" + "---|" * 8]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows]
    (RES / "latest" / "throughput_table.md").write_text("\n".join(lines) + "\n")
    print("wrote", out, f"({len(rows)} cells)")


if __name__ == "__main__":
    main()
