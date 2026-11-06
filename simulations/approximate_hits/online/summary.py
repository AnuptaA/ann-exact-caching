#!/usr/bin/env python3
"""Summarize online_sim.py results.

One row per result file name (<method><tag>.json), averaged over all given
directories (e.g. one per seed): hit rate and mean retention over served
queries after warm-up (overall and per workload phase), share of
--window-query windows whose served retention is below T and the worst such
window, backend work, rebuilds and alarms.

  python online_summary.py DIR [DIR ...] [--window 5000]
"""

import json
from pathlib import Path

ORDER = ["static", "periodic", "lru", "aker", "qvcache", "triggered"]


def windows(log, size):
    """Aggregate batches into consecutive windows of about `size` queries."""
    out, cur = [], {"n": 0, "served": 0, "ret": 0.0}
    for b in log:
        cur["n"] += b["n"]
        cur["served"] += b["served"]
        cur["ret"] += b["served_retention_sum"]
        if cur["n"] >= size:
            out.append(cur)
            cur = {"n": 0, "served": 0, "ret": 0.0}
    if cur["n"]:
        out.append(cur)
    return out


def _rates(log):
    n = sum(b["n"] for b in log)
    srv = sum(b["served"] for b in log)
    ret = sum(b["served_retention_sum"] for b in log)
    return srv / max(n, 1), (ret / srv if srv else None)


def summarize(d, size):
    log, T = d["log"], d["args"]["target"]
    r = {}
    r["hit"], r["ret"] = _rates(log)
    if "phase_start" in d:
        ps = d["phase_start"]
        r["A_hit"], r["A_ret"] = _rates([b for b in log if b["t"] <= ps])
        r["B_hit"], r["B_ret"] = _rates([b for b in log if b["t"] > ps])
    rets = [w["ret"] / w["served"] for w in windows(log, size) if w["served"]]
    r["below"] = sum(x < T for x in rets)
    r["n_win"] = len(rets)
    r["worst"] = min(rets) if rets else None
    c = d["cost"]
    r["topk"], r["topR"], r["fetched"] = c["topk"], c["topR"], c["vectors_fetched"]
    r["rebuilds"] = sum(e["event"] == "rebuild" for e in d["events"])
    r["alarms"] = sum(e["event"] == "alarm" for e in d["events"])
    return r


def _mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def add_args(ap):
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--window", type=int, default=5000)
    ap.set_defaults(run=run)


def run(args):
    names = sorted({p.stem for d in args.dirs for p in Path(d).glob("*.json")},
                   key=lambda n: (next((i for i, m in enumerate(ORDER) if n.startswith(m)), 9), n))
    f = lambda v: "  -- " if v is None else f"{v:.3f}"
    print(f"averaged over {len(args.dirs)} director{'y' if len(args.dirs) == 1 else 'ies'}")
    print(f"{'config':30s} {'hit':>6s} {'ret':>6s} | {'A hit':>6s} {'A ret':>6s} {'B hit':>6s} "
          f"{'B ret':>6s} | {'win<T':>6s} {'worst':>6s} | {'top-k':>6s} {'top-R':>6s} "
          f"{'fetched':>8s} | rebuilds alarms")
    for n in names:
        rows = [summarize(json.loads((Path(d) / f"{n}.json").read_text()), args.window)
                for d in args.dirs if (Path(d) / f"{n}.json").exists()]
        g = lambda k: _mean(rows, k)
        print(f"{n:30s} {f(g('hit'))} {f(g('ret'))} | {f(g('A_hit'))} {f(g('A_ret'))} "
              f"{f(g('B_hit'))} {f(g('B_ret'))} | {g('below'):3.1f}/{g('n_win'):2.0f} "
              f"{f(g('worst'))} | {g('topk'):6.0f} {g('topR'):6.0f} {g('fetched'):8.0f} | "
              f"{g('rebuilds'):8.1f} {g('alarms'):6.1f}")
