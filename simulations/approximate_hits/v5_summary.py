#!/usr/bin/env python3
"""Quick table of the v5 timing runs (doc 9af): per workload and cache size, each technique's speedup over
DiskANN with the base node cache, next to DiskANN given the cache's memory as extra node cache.

  python3 simulations/approximate_hits/v5_summary.py [--version timing_v5]
"""
import json
import argparse
from pathlib import Path

RES = Path("simulations/approximate_hits/results")
TECH = [("ours_s50", "ours static"), ("oursdrift_s50", "ours drift"), ("oursonline_s50", "ours cont."),
        ("lru_s50", "LRU (our rule)"), ("aker_", "Aker"), ("akerpaper_", "Aker paper"), ("qvcache_", "QVCache"),
        ("qvpaper_", "QVCache paper")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="timing_v5")
    a = ap.parse_args()
    print("* = run made before the audit-accounting fix, corrected here (rerun pending)")
    for d in sorted((RES / a.version).iterdir()):
        bj = d / "backend_s50.json"
        if not bj.exists():
            continue
        bk = json.loads(bj.read_text())
        base = bk["backend"]["mean_ms"]
        rows = {}
        for f in d.glob("*_s50_b*.json"):
            if f.name.startswith("backend"):
                continue
            b = f.stem.split("_b")[-1]
            name = next((lab for pre, lab in TECH if f.name.startswith(pre)), f.stem)
            r = json.loads(f.read_text())
            rh = r.get("retention_vs_backend_hits")
            au = r.get("audits_timed", {})
            ms = r["cache"]["mean_ms"]
            if au.get("n", 0) and au.get("accounting") != 2:
                # before the fix (doc 9af) each timed audit's DiskANN charge was subtracted from its request
                # twice: add it back once (total_s = charges + microseconds of label work)
                ms += au["total_s"] / r["n_timed"] * 1e3
                name += "*"
            rows.setdefault(b, []).append((name, base / ms, r["hit_rate"], rh, au.get("ms_per_timed_request")))
        if not rows:
            continue
        print(f"\n== {d.name}: DiskANN + base node cache ({bk['node_cache']['nodes']} nodes) {base:.2f} ms")
        for b in sorted(rows, key=float):
            eq = bk["per_budget"].get(b)
            eqs = base / eq["mean_ms"] if eq else float("nan")
            print(f"  cache {float(b) * 100:g}%: DiskANN + cache memory as node cache {eqs:.2f}x")
            for name, sp, h, rh, au in sorted(rows[b], key=lambda x: -x[1]):
                flag = "" if rh is None or rh >= 0.9 else "  (retention below 0.9)"
                print(f"    {name:16s} {sp:5.2f}x  hit rate {h:.3f}  retention of hits "
                      f"{'-' if rh is None else f'{rh:.3f}'}{flag}"
                      + (f"  audits {au:.2f} ms/request apart" if au else ""))


if __name__ == "__main__":
    main()
