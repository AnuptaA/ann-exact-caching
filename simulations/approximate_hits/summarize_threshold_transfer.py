#!/usr/bin/env python3
"""Summarize threshold_transfer.json files (eval_threshold_transfer.py).

Per run: realized test retention matrix for frozen (source GBT + source
lambda), and hit rate relative to native for frozen and recal. Then a compact
table of off-diagonal pairs grouped by the kind of cache change.

  python summarize_threshold_transfer.py [--matrices] RUN_DIR...
"""

import sys
import json
from pathlib import Path

import numpy as np

TOL = 0.01


def kind(a, b):
    """Kind of change from source cache a to destination cache b."""
    if a.startswith("size") and b.startswith("size"):
        fa, fb = (float(x[4:-1]) for x in (a, b))
        return "larger cache" if fb > fa else "smaller cache"
    if "perturb" in a or "perturb" in b:
        return "entries replaced"
    return "other-target cache"


def _f(v):
    return "  -- " if v is None else f"{v:.3f}"


def matrices(d):
    names = [c["name"] for c in d["caches"]]
    P = {(p["source"], p["dest"]): p for p in d["pairs"]}
    head = " " * 18 + " ".join(f"{n[:11]:>11s}" for n in names)
    print("frozen: realized retention over hits (rows = source, cols = destination)")
    print(head)
    for a in names:
        print(f"{a:18s}" + " ".join(f"{_f(P[a, b]['frozen_hit_retention']):>11s}" for b in names))
    for tag in ("frozen", "recal"):
        print(f"{tag}: hit rate / native hit rate")
        print(head)
        for a in names:
            print(f"{a:18s}" + " ".join(
                f"{P[a, b][f'{tag}_hit_rate'] / max(P[a, b]['native_hit_rate'], 1e-9):11.2f}"
                for b in names))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show = "--matrices" in sys.argv
    rows = []
    for run in map(Path, args):
        path = run / "threshold_transfer.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        target = d["args"]["target"]
        if show:
            print(f"\n=== {run.name} (T = {target}) ===")
            matrices(d)
        for p in d["pairs"]:
            if p["source"] != p["dest"]:
                rows.append((run.name, kind(p["source"], p["dest"]), p, target))
    print(f"\noff-diagonal pairs; viol = realized retention < T - {TOL}; "
          "hit ratios vs native (median)")
    print(f"{'run':20s} {'change':20s} {'n':>3s} | {'frozen viol':>11s} {'worst':>7s} "
          f"{'hit':>5s} | {'recal viol':>10s} {'worst':>7s} {'hit':>5s}")
    groups = {}
    for run, k, p, t in rows:
        groups.setdefault((run, k), []).append((p, t))
    for (run, k), ps in groups.items():
        cells = []
        for tag in ("frozen", "recal"):
            gaps = [p[f"{tag}_hit_retention"] - t for p, t in ps
                    if p[f"{tag}_hit_retention"] is not None]
            ratio = [p[f"{tag}_hit_rate"] / p["native_hit_rate"] for p, _ in ps
                     if p["native_hit_rate"] > 0]
            viol = np.mean([g < -TOL for g in gaps]) if gaps else 0.0
            worst = min(gaps) if gaps else float("nan")
            cells.append(f"{viol:11.0%} {worst:+7.3f} {np.median(ratio):5.2f}")
        print(f"{run:20s} {k:20s} {len(ps):3d} | {cells[0]} | {cells[1][1:]}")


if __name__ == "__main__":
    main()
