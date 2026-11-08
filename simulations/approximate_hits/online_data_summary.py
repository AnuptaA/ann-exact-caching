#!/usr/bin/env python3
"""Summarize data-change runs (online_data_jobs.py): per configuration, mean
over seeds of hit rate and served retention after warm-up, 5000-query
windows whose served retention is below T, the same for "affected" queries
(current exact top-10 contains inserted vectors), alarms, rebuilds, backend
searches and upkeep cost. Burst runs: also before / after the burst.

  python online_data_summary.py [--kinds churn drift] [--schedules gradual burst]
"""

import sys
import json
import argparse
from pathlib import Path

RES = Path("simulations/approximate_hits/results/online/esci1m")
T = 0.9
ORDER = ["static", "slide_none", "slide_score", "slide_ret", "slide_both", "slide_ret_loose",
         "slide_both_loose", "maint_none", "maint_ret", "maint_both", "noupkeep_both",
         "oracle", "known_reset"]


def stats(log, T, lo=0, hi=float("inf")):
    b = [x for x in log if lo < x["t"] <= hi]
    n = sum(x["n"] for x in b)
    srv = sum(x["served"] for x in b)
    ret = sum(x["served_retention_sum"] for x in b)
    an = sum(x.get("aff_n", 0) for x in b)
    asrv = sum(x.get("aff_served", 0) for x in b)
    aret = sum(x.get("aff_ret_sum", 0.0) for x in b)
    wins, cur = [], [0, 0.0]
    for x in b:
        cur[0] += x["served"]
        cur[1] += x["served_retention_sum"]
        if x["t"] % 5000 == 0:
            if cur[0]:
                wins.append(cur[1] / cur[0])
            cur = [0, 0.0]
    return {"hit": srv / max(n, 1), "ret": ret / srv if srv else None,
            "aff_share": an / max(n, 1), "aff_hit": asrv / an if an else None,
            "aff_ret": aret / asrv if asrv else None,
            "below": sum(w < T for w in wins), "n_win": len(wins),
            "worst": min(wins) if wins else None}


def row(d):
    T = d["args"]["target"]
    r = stats(d["log"], T)
    if d["args"].get("schedule") == "burst":
        bt = 40000
        r.update({f"pre_{k}": v for k, v in stats(d["log"], T, hi=bt).items()})
        r.update({f"post_{k}": v for k, v in stats(d["log"], T, lo=bt).items()})
    ev = d["events"]
    r["alarms"] = sum(e["event"] == "alarm" for e in ev)
    r["alarm_kinds"] = ",".join(sorted({e["kind"] for e in ev if e["event"] == "alarm"}))
    r["rebuilds"] = sum(e["event"] == "rebuild" for e in ev)
    c = d["cost"]
    r["topk"] = c["topk"]
    r["checks"] = c.get("center_checks", c.get("refresh_dist", 0))
    r["labels"] = c.get("label_updates", 0)
    r["deleted_returned"] = c.get("deleted_returned", 0)
    return r


def mean(rows, k):
    v = [r[k] for r in rows if r.get(k) is not None]
    return sum(v) / len(v) if v else None


def load_rows(d, name, seeds):
    rows = []
    for s in seeds:
        p = next(iter(sorted((d.parent / f"{d.name}_s{s}").glob(f"*_{name}.json"))), None) \
            if not (d.parent / f"{d.name}_s{s}" / f"{name}.json").exists() \
            else d.parent / f"{d.name}_s{s}" / f"{name}.json"
        if p is not None:
            rows.append(row(json.loads(p.read_text())))
    return rows


def best_of(d, pattern, seeds):
    """Setting matching pattern with the highest mean hit rate among those
    whose mean served retention >= T; if none meets T, the highest retention."""
    names = sorted({p.stem for s in seeds for p in
                    (d.parent / f"{d.name}_s{s}").glob(pattern)})
    cand = [(n, load_rows(d, n, seeds)) for n in names]
    cand = [(n, r) for n, r in cand if r]
    if not cand:
        return None
    ok = [c for c in cand if mean(c[1], "ret") is not None and mean(c[1], "ret") >= T]
    pick = (max(ok, key=lambda c: mean(c[1], "hit")) if ok else
            max(cand, key=lambda c: mean(c[1], "ret") or 0.0))
    return pick[0], pick[1], bool(ok), len(ok), len(cand)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["churn", "drift"])
    ap.add_argument("--schedules", nargs="+", default=["gradual", "burst"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    args = ap.parse_args()
    f = lambda v: "  -- " if v is None else f"{v:.3f}"
    for kind in args.kinds:
        for sched in args.schedules:
            print(f"\n== {kind} {sched} (mean over seeds {args.seeds})")
            burst = sched == "burst"
            head = (f"{'config':18s}   hit   ret below worst | aff: share  hit   ret"
                    + (" | pre hit  ret | post hit  ret below" if burst else "")
                    + " | alarms rebuilds  topk/q")
            print(head)
            for name in ORDER:
                rows = []
                for s in args.seeds:
                    p = next(iter(sorted((RES / f"data_{kind}_{sched}_s{s}").glob(
                        f"*_{name}.json"))), None)
                    if p is not None:
                        rows.append(row(json.loads(p.read_text())))
                if not rows:
                    continue
                g = lambda k: mean(rows, k)
                line = (f"{name:18s} {f(g('hit'))} {f(g('ret'))} {g('below'):3.1f}/{g('n_win'):2.0f}"
                        f" {f(g('worst'))} |      {f(g('aff_share'))} {f(g('aff_hit'))} "
                        f"{f(g('aff_ret'))}")
                if burst:
                    line += (f" |     {f(g('pre_hit'))} {f(g('pre_ret'))} |     "
                             f"{f(g('post_hit'))} {f(g('post_ret'))} {g('post_below'):3.1f}")
                kinds = "/".join(sorted({r["alarm_kinds"] for r in rows if r["alarm_kinds"]}))
                line += (f" | {g('alarms'):5.1f} {kinds:15s} {g('rebuilds'):4.1f} "
                         f"{g('topk') / 110000:6.3f}")
                print(line)
            d = RES / f"data_{kind}_{sched}"
            for label, pat, seeds in (("lru (no tests)", "lru_k*_none.json", args.seeds),
                                      ("lru (tests)", "lru_k*_both.json", args.seeds),
                                      ("aker", "aker_*.json", [0]),
                                      ("qvcache keep", "qvcache_*_keep.json", [0]),
                                      ("qvcache remove", "qvcache_*_remove.json", [0])):
                b = best_of(d, pat, seeds)
                if b is None:
                    continue
                name, rows, ok, n_ok, n = b
                g = lambda k: mean(rows, k)
                line = (f"{label:18s} {f(g('hit'))} {f(g('ret'))} {g('below'):3.1f}/{g('n_win'):2.0f}"
                        f" {f(g('worst'))} |      {f(g('aff_share'))} {f(g('aff_hit'))} "
                        f"{f(g('aff_ret'))}")
                if burst:
                    line += (f" |     {f(g('pre_hit'))} {f(g('pre_ret'))} |     "
                             f"{f(g('post_hit'))} {f(g('post_ret'))} {g('post_below'):3.1f}")
                line += f" | {name}" + ("" if ok else " (none meets T)") + f" [{n_ok}/{n} meet T]"
                print(line)


if __name__ == "__main__":
    sys.exit(main())
