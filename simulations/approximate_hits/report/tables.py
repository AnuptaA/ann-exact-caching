#!/usr/bin/env python3
"""Print compact hit-rate tables from approximate-hit results.json files.

Each cell: best-case hit% / effective retention% (online in a second table).
Values are taken at the recorded checkpoint closest to each target size.
"""

from pathlib import Path

from simulations.approximate_hits.report import common

SIZES = [0.01, 0.05, 0.10, 0.20, 0.30]


def summarize(path: Path, online: bool) -> None:
    data = common.load_json(path)
    hk = "test_hit_rate_online" if online else "test_hit_rate"
    ek = "test_effective_retention_online" if online else "test_effective_retention"
    label = "online" if online else "best-case"
    print(f"\n{path.parent.name} ({label}): hit% / effective retention%")
    print(f"  {'T':>5} " + "".join(f"{f'~{int(s*100)}%':>18}" for s in SIZES))
    for res in data["results"]:
        cells = []
        for s in SIZES:
            p = common.nearest(res["points"], s)
            note = "" if abs(p["cache_frac"] - s) < 0.1 * s else f"@{100*p['cache_frac']:.2g}%"
            cells.append(f"{100*p[hk]:.1f}/{100*p[ek]:.2f}{note}")
        print(f"  {res['target']:>5} " + "".join(f"{c:>18}" for c in cells))


def add_args(ap):
    ap.add_argument("paths", nargs="+", help="results.json files")
    ap.add_argument("--online", action="store_true")
    ap.set_defaults(run=run)


def run(args):
    for p in args.paths:
        summarize(Path(p), args.online)

