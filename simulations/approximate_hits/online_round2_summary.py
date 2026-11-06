#!/usr/bin/env python3
"""Round-2 summary: our configurations and LRU (mean of seeds 0, 1), best
Aker / QVCache setting meeting T (seed 0), per workload.

  python online_round2_summary.py [prep:stream ...]
"""
import sys, json, glob
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from online_summary import summarize
R = "simulations/approximate_hits/results/online"
W = [("esci1m","shift"),("esci1m","random"),("spacev-aker-0.6","shift"),("spacev-aker-0.6","trace"),
     ("spacev-aker-0.3","trace"),("spacev-aker-0.99","trace"),("spacev-real","trace")]
if len(sys.argv) > 1: W = [tuple(a.split(":")) for a in sys.argv[1:]]
OURS = ["static_default","greedy_none"] + [f"greedy_{d}_{r}" for d in ["score","retention","hitrate","all"] for r in ["recal","rebuild"]] + ["greedy_none_oraclecal","greedy_all_rebuild_oraclecal"]
def load(prep, st, name, seeds):
    rows = []
    for s in seeds:
        p = Path(f"{R}/{prep}/r2_{st}_s{s}/{name}.json")
        if p.exists(): rows.append(summarize(json.loads(p.read_text()), 5000))
    return rows
def avg(rows, k):
    v = [r[k] for r in rows if r.get(k) is not None]
    return sum(v)/len(v) if v else None
f = lambda v: "  -- " if v is None else f"{v:.3f}"
for prep, st in W:
    print(f"\n== {prep} {st}   (ours / LRU: mean of seeds 0,1; baselines: best on seed 0 meeting T)")
    print(f"{'config':28s}   hit  orac   ret | B hit B orc B ret | below  worst | alarms rebuilds")
    def show(name, rows):
        if not rows: return
        g = lambda k: avg(rows, k)
        print(f"{name:28s} {f(g('hit'))} {f(g('oracle'))} {f(g('ret'))} | {f(g('B_hit'))} {f(g('B_oracle'))} {f(g('B_ret'))} | {g('below'):4.1f}/{g('n_win'):2.0f} {f(g('worst'))} | {g('alarms'):5.1f} {g('rebuilds'):5.1f}")
    for n in OURS: show(n, load(prep, st, n, [0,1]))
    for fam, seeds in (("lru", [0,1]), ("aker", [0]), ("qvcache", [0])):
        names = sorted({Path(p).stem for p in glob.glob(f"{R}/{prep}/r2_{st}_s0/{fam}_*.json")})
        cand = [(n, load(prep, st, n, seeds)) for n in names]
        ok = [c for c in cand if c[1] and avg(c[1], "ret") and avg(c[1], "ret") >= 0.9]
        if fam == "lru":
            for n, rows in cand: show(n, rows)
        elif cand:
            best = max(ok or cand, key=lambda c: avg(c[1], "hit"))
            show(best[0] + ("" if ok else " (none meet T)"), best[1])
