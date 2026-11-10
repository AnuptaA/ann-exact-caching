#!/usr/bin/env python3
"""Summary of the probes of probe_jobs.py on the common window (requests 60,000-100,000):
hit rate, mean retention of hits, entries and cached vectors at the start of the window,
vectors per entry. Runs with a longer warm-up are compared on the same requests."""

import json
from pathlib import Path

RES = Path("simulations/approximate_hits/results/online")
FROM_T = 60000
PREPS = ["spacev-aker-0.6", "triviaqa-dpr-0.6", "triviaqa-dpr-0.99"]
DESC = {"greedy_P0": "ours, baseline (warm-up 20k)", "greedy_P1": "ours, 10k candidates / 5k scoring",
        "greedy_P2": "ours, balls K <= 20", "greedy_P2b": "ours, balls K <= 50",
        "greedy_P3a": "ours, warm-up 40k", "greedy_P3b": "ours, warm-up 60k",
        "greedy_P4": "ours, 10k candidates + K <= 20",
        "greedy_X1": "ORACLE X1: scored on 20k-100k (in-sample)", "greedy_X2": "ORACLE X2: scored on 20k-60k",
        "lru_L20": "LRU K=20, warm-up 20k", "lru_L40": "LRU K=20, warm-up 40k",
        "lru_L60": "LRU K=20, warm-up 60k"}


def main():
    for prep in PREPS:
        print(f"== {prep}, 10% budget, requests {FROM_T}-100000")
        print(f"   {'run':42s} {'hit % 60k-100k':>14s} {'ret':>6s} {'hit % 20k-100k':>14s} {'entries':>8s} {'vec/entry':>9s}")
        for tag, desc in DESC.items():
            p = RES / prep / "probe_s0" / f"{tag}.json"
            if not p.exists():
                print(f"   {desc:42s} (not finished)")
                continue
            allb = json.loads(p.read_text())["log"]
            L = [b for b in allb if b["t"] > FROM_T]
            n = sum(b["n"] for b in L)
            s = sum(b["served"] for b in L)
            r = sum(b["served_retention_sum"] for b in L) / max(s, 1)
            full = "-"
            if allb[0]["t"] <= 21000:           # warm-up 20k: the whole post-warm-up stream is measured
                full = f"{100 * sum(b['served'] for b in allb) / sum(b['n'] for b in allb):.1f}"
            x = allb[0]
            print(f"   {desc:42s} {100 * s / n:14.1f} {r:6.3f} {full:>14s} {x['entries']:8d} "
                  f"{x['cached'] / max(x['entries'], 1):9.0f}")


if __name__ == "__main__":
    main()
