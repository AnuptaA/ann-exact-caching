#!/usr/bin/env python3
"""Print compact hit-rate tables from approximate-hit results.json files.

Each cell: best-case hit% / effective retention% (online in a second table).
Values are taken at the recorded checkpoint closest to each target size.
"""

import sys
import json
import argparse
from pathlib import Path

SIZES = [0.01, 0.05, 0.10, 0.20, 0.30]


def _point_near(points, size):
    # checkpoint points plus the final state (the max-size point)
    return min(points, key=lambda p: abs(p["cache_frac"] - size))


def summarize(path: Path, online: bool) -> None:
    data = json.loads(path.read_text())
    hk = "test_hit_rate_online" if online else "test_hit_rate"
    ek = "test_effective_retention_online" if online else "test_effective_retention"
    label = "online" if online else "best-case"
    print(f"\n{path.parent.name} ({label}): hit% / effective retention%")
    print(f"  {'T':>5} " + "".join(f"{f'~{int(s*100)}%':>18}" for s in SIZES))
    for res in data["results"]:
        cells = []
        for s in SIZES:
            p = _point_near(res["points"], s)
            note = "" if abs(p["cache_frac"] - s) < 0.1 * s else f"@{100*p['cache_frac']:.2g}%"
            cells.append(f"{100*p[hk]:.1f}/{100*p[ek]:.2f}{note}")
        print(f"  {res['target']:>5} " + "".join(f"{c:>18}" for c in cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--online", action="store_true")
    args = ap.parse_args()
    for p in args.paths:
        summarize(Path(p), args.online)


if __name__ == "__main__":
    sys.exit(main())
