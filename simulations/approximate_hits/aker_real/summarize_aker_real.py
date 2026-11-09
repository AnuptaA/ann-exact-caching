#!/usr/bin/env python3
"""Summary of the real-Aker runs (results/aker_real/<prep>/<setting>_b<B>/at<a>_d<d>.json).
Per cell: the authors' default (alpha_tighten 0.9, delta 5) and the best grid
setting (highest hit rate with retention >= T against the backend's answer, else
the highest retention, marked '!'). Hit % / retention vs backend / vs exact;
data changes also: retention of queries whose answers include inserted vectors
is not tracked here (Aker's harness logs deleted vectors returned instead);
shifts: per phase hit / retention of the best setting.

  python summarize_aker_real.py [--preps ...] [--T 0.9]
"""

import json
import argparse
from pathlib import Path

ROOT = Path("simulations/approximate_hits/results/aker_real")


def summ(path, phase_starts=None):
    r = json.loads(path.read_text())
    s = r["summary"]
    out = {"hit": 100 * s["hit_rate"], "ret_b": s["retention_vs_backend"], "ret_x": s["retention"],
           "del": s.get("deleted_returned", 0), "ex": 100 * s["exact_hit_share"],
           "ap": 100 * s["approx_hit_share"], "ln": 100 * s.get("linked_hit_share", 0)}
    ps = r.get("phase_starts")
    if ps and len(ps) > 1:
        ends = ps[1:] + [10 ** 9]
        ph = []
        for a, b in zip(ps, ends):
            L = [x for x in r["log"] if a < x["t"] <= b]
            n = sum(x["n"] for x in L)
            sv = sum(x["served"] for x in L)
            if n:
                ph.append((100 * sv / n, sum(x["served_retention_backend_sum"] for x in L) / max(sv, 1)))
        out["phases"] = ph
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preps", nargs="*", default=None)
    ap.add_argument("--T", type=float, default=0.9)
    args = ap.parse_args()
    preps = args.preps or sorted(p.name for p in ROOT.iterdir() if p.is_dir())
    print("prep | setting | budget | default (at .9, d 5): hit % (exact+approx+linked) / ret vs backend / vs exact"
          " | best setting: same")
    for prep in preps:
        for d in sorted((ROOT / prep).glob("*_b*")):
            runs = {f.stem: summ(f) for f in d.glob("*.json")}
            if not runs:
                continue
            setting, b = d.name.rsplit("_b", 1)
            ok = {k: v for k, v in runs.items() if v["ret_b"] >= args.T}
            best = max(ok, key=lambda k: ok[k]["hit"]) if ok else \
                max(runs, key=lambda k: runs[k]["ret_b"])

            def fmt(k):
                v = runs[k]
                t = (f"{v['hit']:.1f} ({v['ex']:.1f}+{v['ap']:.1f}+{v['ln']:.1f}) / "
                     f"{v['ret_b']:.3f} / {v['ret_x']:.3f}")
                if v["del"]:
                    t += f", deleted returned {v['del']}"
                if "phases" in v:
                    t += " | phases " + ", ".join(f"{h:.1f}/{r:.3f}" for h, r in v["phases"])
                return t
            dflt = "at0.9_d5"
            print(f"{prep} | {setting} | {b} | {fmt(dflt) if dflt in runs else '-'} | "
                  f"{best}{'' if ok else ' !'}: {fmt(best)}")


if __name__ == "__main__":
    main()
