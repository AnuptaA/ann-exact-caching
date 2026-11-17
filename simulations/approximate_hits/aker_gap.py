#!/usr/bin/env python3
"""Why ours loses to Aker (doc 9ag): per technique, hit rate, mean latency split into the every-request
lookup (kNN + balls + shell + model, per request) and the rest, the speedup if that lookup were free,
and 1 / (1 - hit rate) (the bound for a free lookup and free hits).

  python3 simulations/approximate_hits/aker_gap.py tripclick-ip spacev-real95
"""
import sys
import json
from pathlib import Path

RES = Path("simulations/approximate_hits/results/timing_v5")
NAMES = [("ours_s50", "ours static"), ("oursdrift_s50", "ours drift"), ("oursonline_s50", "ours cont."),
         ("lru_s50", "LRU (our rule)"), ("aker_", "Aker"), ("qvcache_", "QVCache")]
LOOKUP = ("knn", "balls", "shell", "model")

for prep in sys.argv[1:]:
    d = RES / prep
    base = json.loads((d / "backend_s50.json").read_text())["backend"]["mean_ms"]
    print(f"\n== {prep}: DiskANN {base:.2f} ms")
    for b in sorted({f.stem.split("_b")[-1] for f in d.glob("*_s50_b*.json") if not f.name.startswith("backend")}, key=float):
        print(f"  cache {float(b) * 100:g}%   {'technique':15s} hit%  speedup | lookup ms/req (knn balls shell) serve/hit | "
              f"hit ms  miss ms | free-lookup speedup  1/(1-h)")
        for pre, name in NAMES:
            fs = [f for f in d.glob(f"{pre}*_b{b}.json")]
            if not fs:
                continue
            r = json.loads(fs[0].read_text())
            st = r.get("stage_ms", {})
            lk = sum(st.get(k, 0.0) for k in LOOKUP)
            h = r["hit_rate"]
            c = r["cache"]
            sp = base / c["mean_ms"]
            free = base / max(c["mean_ms"] - lk, 1e-9)
            serve = st.get("serve", 0.0) / h if h else 0.0
            print(f"           {name:15s} {100 * h:5.1f} {sp:5.2f}x | {lk:5.2f} ({st.get('knn', 0):.2f} {st.get('balls', 0):.2f} "
                  f"{st.get('shell', 0):.2f}) {serve:5.2f} | {c['hits_ms']['mean_ms'] if c.get('hits_ms') else float('nan'):6.2f} "
                  f"{c['misses_ms']['mean_ms']:7.2f} | {free:5.2f}x  {1 / (1 - h):5.2f}x")
