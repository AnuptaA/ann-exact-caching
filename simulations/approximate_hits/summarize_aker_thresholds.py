#!/usr/bin/env python3
"""Table: Aker's hit rule vs our calibrated admission on the same caches.

Reads results/batch_aker/<trace>/{results,predictors,aker_thresholds*}.json.
Per cache size: hit rate (and mean union retention over hits) for the oracle,
our guaranteed best predictor (T = 0.9), Aker cold / warm, and the Aker
ceiling (static d < minD).
"""

import json
from pathlib import Path

import numpy as np

BATCH = Path(__file__).parent / "results" / "batch_aker"
TRACES = ["spacev-aker-0.3", "spacev-aker-0.6", "spacev-aker-0.99", "spacev-real",
          "triviaqa-aker-0.3", "triviaqa-aker-0.6", "triviaqa-aker-0.99"]
SIZES = [0.001, 0.01, 0.05]
TARGET = 0.9


def _load(path):
    return json.loads(path.read_text()) if path.exists() else None


def _near(rows, size):
    r = min(rows, key=lambda r: abs(r["cache_frac"] - size))
    return r if abs(r["cache_frac"] - size) < 0.2 * size else None


def _fmt(h, ret):
    return f"{100 * h:5.1f} ({ret:.2f})" if ret is not None else f"{100 * h:5.1f} ( -- )"


def ours(run_dir):
    d = _load(run_dir / "predictors.json")
    states = d["states"]
    models = [m for m in d["models"] if all(m in s for s in states)]
    best = max(models, key=lambda m: np.mean([s[m]["guaranteed_dev_hit_rate"] for s in states]))
    rows = [dict(cache_frac=s["cache_frac"], h=s[best]["guaranteed_test_hit_rate"],
                 ret=s[best]["guaranteed_test_hit_retention"])
            for s in states if abs(s["target"] - TARGET) < 1e-9]
    return best, rows


def main():
    head = (f"{'trace':20s} {'cache':>6s} | {'oracle':>6s} | {'ours (ret)':>12s} | "
            f"{'Aker cold':>12s} | {'Aker warm':>12s} | {'ceiling':>12s} | eq hits")
    print(f"hit % (mean union retention over hits), T = {TARGET} cache\n{head}")
    for t in TRACES:
        run = BATCH / t
        aker = _load(run / "aker_thresholds.json")
        ceil = _load(run / "aker_thresholds_ceiling.json")
        if aker is None or ceil is None:
            continue
        best, our_rows = ours(run)
        for size in SIZES:
            a, c, o = _near(aker["states"], size), _near(ceil["states"], size), _near(our_rows, size)
            if a is None:
                continue
            ours_s = _fmt(o["h"], o["ret"]) if o else f"{'n/a':>12s}"
            print(f"{t:20s} {100 * size:5.1f}% | {100 * a['oracle_hit_rate']:6.1f} | {ours_s} | "
                  f"{_fmt(a['cold']['hit_rate'], a['cold']['hit_retention_union'])} | "
                  f"{_fmt(a['warm']['hit_rate'], a['warm']['hit_retention_union'])} | "
                  f"{_fmt(c['cold']['hit_rate'], c['cold']['hit_retention_union'])} | "
                  f"{100 * a['cold']['equality_hit_rate']:.1f}")
        print(f"{'':20s} (ours = {best})")


if __name__ == "__main__":
    main()
