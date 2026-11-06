#!/usr/bin/env python3
"""Summarize online_sim.py results.

One row per result file name (<method><tag>.json), averaged over all given
directories (e.g. one per seed): hit rate and mean retention over served
queries after warm-up (overall and per workload phase), share of
--window-query windows whose served retention is below T and the worst such
window, backend work, rebuilds and alarms; "orac" = hit rate of the oracle
serve decision (perfect knowledge of each query's retention) for the cache
the method actually had, from per-batch retention histograms.

  python online_summary.py DIR [DIR ...] [--window 5000]
"""

import sys
import json
import argparse
from pathlib import Path

ORDER = ["static", "greedy", "lru", "aker", "qvcache"]


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


def oracle_rate(log, target):
    """Hit rate of the oracle serve decision (perfect knowledge of each
    query's retention) for the cache in use, from per-batch histograms."""
    import numpy as np
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from simulations.approximate_hits.greedy_population import hits_from_histograms
    hs = [b["ret_hist"] for b in log if "ret_hist" in b]
    if not hs:
        return None
    hist = np.sum(hs, axis=0)
    k = len(hist) - 1
    return float(hits_from_histograms(hist, target, k)[0] / hist.sum())


def _rates(log):
    n = sum(b["n"] for b in log)
    srv = sum(b["served"] for b in log)
    ret = sum(b["served_retention_sum"] for b in log)
    return srv / max(n, 1), (ret / srv if srv else None)


def summarize(d, size):
    log, T = d["log"], d["args"]["target"]
    r = {}
    r["hit"], r["ret"] = _rates(log)
    r["oracle"] = oracle_rate(log, T)
    if "phase_start" in d:
        ps = d["phase_start"]
        A, B = [b for b in log if b["t"] <= ps], [b for b in log if b["t"] > ps]
        r["A_hit"], r["A_ret"] = _rates(A)
        r["B_hit"], r["B_ret"] = _rates(B)
        r["A_oracle"], r["B_oracle"] = oracle_rate(A, T), oracle_rate(B, T)
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


def phase_rows(d):
    """Per phase (from phase_starts): hit rate, retention, oracle hit rate,
    queries from the phase start to the first alarm in the phase (None =
    no alarm), and number of rebuilds in the phase."""
    log, T, ev = d["log"], d["args"]["target"], d["events"]
    starts = d.get("phase_starts", [0]) + [float("inf")]
    out = []
    for lo, hi in zip(starts[:-1], starts[1:]):
        b = [x for x in log if lo < x["t"] <= hi]
        if not b:
            continue
        hit, ret = _rates(b)
        al = [e["t"] for e in ev if e["event"] == "alarm" and lo < e["t"] <= hi]
        out.append({"hit": hit, "ret": ret, "oracle": oracle_rate(b, T),
                    "delay": (min(al) - lo) if al else None,
                    "alarms": len(al),
                    "rebuilds": sum(e["event"] == "rebuild" and lo < e["t"] <= hi for e in ev)})
    return out


def print_phases(names, dirs):
    f = lambda v: "  -- " if v is None else f"{v:.3f}"
    print("per phase: hit / oracle / retention, delay = queries from the phase start "
          "to the first alarm (mean over runs with an alarm; n = runs with one), "
          "alarms and rebuilds in the phase (means)")
    for n in names:
        runs = [phase_rows(json.loads((Path(d) / f"{n}.json").read_text()))
                for d in dirs if (Path(d) / f"{n}.json").exists()]
        if not runs:
            continue
        cells = []
        for i in range(min(len(r) for r in runs)):
            g = lambda k: _mean([r[i] for r in runs], k)
            dl = [r[i]["delay"] for r in runs if r[i]["delay"] is not None]
            cells.append(f"{f(g('hit'))} {f(g('oracle'))} {f(g('ret'))} "
                         f"d{(sum(dl) / len(dl)) if dl else float('nan'):6.0f} n{len(dl)} "
                         f"a{g('alarms'):.1f} r{g('rebuilds'):.1f}")
        print(f"{n:30s} " + " | ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--window", type=int, default=5000)
    ap.add_argument("--phases", action="store_true",
                    help="per-phase table (e.g. multishift streams)")
    args = ap.parse_args()
    names = sorted({p.stem for d in args.dirs for p in Path(d).glob("*.json")},
                   key=lambda n: (next((i for i, m in enumerate(ORDER) if n.startswith(m)), 9), n))
    if args.phases:
        return print_phases(names, args.dirs)
    f = lambda v: "  -- " if v is None else f"{v:.3f}"
    print(f"averaged over {len(args.dirs)} director{'y' if len(args.dirs) == 1 else 'ies'}")
    print(f"{'config':30s} {'hit':>6s} {'orac':>6s} {'ret':>6s} | {'A hit':>6s} {'A orc':>6s} "
          f"{'A ret':>6s} {'B hit':>6s} {'B orc':>6s} {'B ret':>6s} | {'win<T':>6s} {'worst':>6s} | "
          f"{'top-k':>6s} {'top-R':>6s} {'fetched':>8s} | rebuilds alarms")
    for n in names:
        rows = [summarize(json.loads((Path(d) / f"{n}.json").read_text()), args.window)
                for d in args.dirs if (Path(d) / f"{n}.json").exists()]
        g = lambda k: _mean(rows, k)
        print(f"{n:30s} {f(g('hit'))} {f(g('oracle'))} {f(g('ret'))} | {f(g('A_hit'))} "
              f"{f(g('A_oracle'))} {f(g('A_ret'))} {f(g('B_hit'))} {f(g('B_oracle'))} "
              f"{f(g('B_ret'))} | {g('below'):3.1f}/{g('n_win'):2.0f} "
              f"{f(g('worst'))} | {g('topk'):6.0f} {g('topR'):6.0f} {g('fetched'):8.0f} | "
              f"{g('rebuilds'):8.1f} {g('alarms'):6.1f}")


if __name__ == "__main__":
    sys.exit(main())
