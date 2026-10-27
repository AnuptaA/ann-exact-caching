#!/usr/bin/env python3
"""Trends over k and database size: oracle and guaranteed (B5) hit rates.

Reads results.json (oracle, online admission) and calibrated.json (guaranteed
calibrated admission) for each workload x setting; prints tables and writes
results/trends_T<T>.png (rows: workloads; columns: k sweep, database sweep).
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).parent / "results"
WORKLOADS = [
    ("esci", "ESCI, real queries"),
    ("syn_interp_zipf0.3", "simZipf 0.3"),
    ("syn_interp_zipf0.6", "simZipf 0.6"),
    ("syn_interp_zipf0.99", "simZipf 0.99"),
    ("syn_gauss0.1_zipf0.99", "gauss 0.99"),
]
SIZES = [0.01, 0.10, 0.30]


def run_dir(wl, k, db):
    if k == 10 and db == "50k":
        return RESULTS / ("esci_score20k" if wl == "esci" else wl)
    if db == "50k":
        return RESULTS / "batch_k_db" / f"{wl}_k{k}"
    if db == "200k":
        return RESULTS / "batch_k_db" / f"{wl}_db200k_k10"
    return RESULTS / "batch_db1m" / f"{wl}_db1m_k10"


K_SETTINGS = [(1, "50k"), (10, "50k"), (50, "50k"), (100, "50k")]
DB_SETTINGS = [(10, "50k"), (10, "200k"), (10, "1m")]


def curves(d, target):
    """(xs, oracle_online, guaranteed) for one run dir; None parts if missing."""
    oracle = guar = None
    if (d / "results.json").exists():
        for r in json.loads((d / "results.json").read_text())["results"]:
            if abs(r["target"] - target) < 1e-9:
                pts = sorted(r["points"], key=lambda p: p["cache_frac"])
                oracle = ([p["cache_frac"] for p in pts],
                          [p["test_hit_rate_online"] for p in pts])
    if (d / "calibrated.json").exists():
        for r in json.loads((d / "calibrated.json").read_text())["results"]:
            if abs(r["target"] - target) < 1e-9:
                rows = sorted((x for x in r["rows"] if "guaranteed_hit_rate" in x),
                              key=lambda x: x["cache_frac"])
                guar = ([x["cache_frac"] for x in rows],
                        [x["guaranteed_hit_rate"] for x in rows])
    return oracle, guar


def _at(curve, size):
    if curve is None or not curve[0]:
        return None
    i = int(np.argmin(np.abs(np.array(curve[0]) - size)))
    return curve[1][i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=float, nargs="+", default=[0.9, 0.95])
    args = ap.parse_args()
    for target in args.targets:
        print(f"\n=== T={target}: hit % at ~1% / 10% / 30% cache "
              "(oracle online -> guaranteed calibrated; '-' = not yet available) ===")
        fig, axes = plt.subplots(len(WORKLOADS), 2, figsize=(12, 3.6 * len(WORKLOADS)),
                                 squeeze=False)
        for row, (wl, title) in enumerate(WORKLOADS):
            print(f"  {title}")
            for col, (settings, label) in enumerate(((K_SETTINGS, "k"), (DB_SETTINGS, "db"))):
                ax = axes[row, col]
                colors = plt.cm.plasma(np.linspace(0, 0.85, len(settings)))
                for (k, db), color in zip(settings, colors):
                    tag = f"k={k}" if label == "k" else f"db={db}"
                    oracle, guar = curves(run_dir(wl, k, db), target)
                    cells = []
                    for s in SIZES:
                        o, g = _at(oracle, s), _at(guar, s)
                        fo = f"{100 * o:.1f}" if o is not None else "-"
                        fg = f"{100 * g:.1f}" if g is not None else "-"
                        cells.append(f"{fo}->{fg}")
                    print(f"    {tag:<8} (k={k}, db={db})".ljust(26) +
                          "".join(f"{c:>16}" for c in cells))
                    if oracle is not None:
                        ax.plot(*oracle, "-", color=color, lw=1, alpha=0.5)
                    if guar is not None:
                        xs, ys = np.array(guar[0]), np.array(guar[1])
                        keep = (xs > 0) & (ys > 0)
                        ax.plot(xs[keep], ys[keep], "o-", ms=3, color=color, label=tag)
                ref = np.logspace(-4, 0, 50)
                ax.plot(ref, ref, "k:", lw=1)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlim(5e-4, 0.35)
                ax.set_ylim(1e-3, 1.05)
                ax.grid(True, alpha=0.3)
                sweep = "k sweep (db=50k)" if label == "k" else "database sweep (k=10)"
                ax.set_title(f"{title}: {sweep}", fontsize=9)
                ax.legend(fontsize=7, loc="lower right")
                if row == len(WORKLOADS) - 1:
                    ax.set_xlabel("cache size (fraction of database)")
                if col == 0:
                    ax.set_ylabel("hit rate")
        fig.suptitle(f"T={target}: guaranteed calibrated hit rate (dots) vs oracle online "
                     "(thin lines)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        out = RESULTS / f"trends_T{target}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
