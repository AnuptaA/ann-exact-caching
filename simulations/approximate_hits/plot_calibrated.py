#!/usr/bin/env python3
"""Hit rate vs cache size: oracle vs calibrated admission (B5) vs certificates.

One panel per workload at a fixed target T. Reads results/<run>/calibrated.json
(and certificates.json if present); writes results/calibrated_T<T>.png and
prints a table including the realized test retention over hits.
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).parent / "results"
RUNS = [
    ("esci_score20k", "ESCI, real queries"),
    ("syn_interp_zipf0.3", "SIFT, simZipf s=0.3 (interp)"),
    ("syn_interp_zipf0.6", "SIFT, simZipf s=0.6 (interp)"),
    ("syn_interp_zipf0.99", "SIFT, simZipf s=0.99 (interp)"),
    ("syn_gauss0.1_zipf0.99", "SIFT, simZipf s=0.99 (gauss 0.1)"),
]
SIZES = [0.01, 0.05, 0.10, 0.30]


def _rows(run, name, target):
    path = RESULTS / run / name
    if not path.exists():
        return None
    for t in json.loads(path.read_text())["results"]:
        if abs(t["target"] - target) < 1e-9:
            return t["rows"]
    return None


def _near(rows, size):
    return min(rows, key=lambda r: abs(r["cache_frac"] - size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=float, nargs="+", default=[0.9, 0.95])
    args = ap.parse_args()
    for target in args.targets:
        ncols = 3
        nrows = int(np.ceil(len(RUNS) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.6 * nrows),
                                 sharex=True, sharey=True, squeeze=False)
        print(f"\n=== T={target}: hit % at ~1% / 5% / 10% / 30% cache "
              "(guaranteed: realized retention over hits in brackets) ===")
        for ax, (run, title) in zip(axes.flat, RUNS):
            cal = _rows(run, "calibrated.json", target)
            cert = _rows(run, "certificates.json", target)
            if cal is None:
                ax.set_title(f"{title}\n(missing)")
                continue
            series = [
                ("oracle (online)", cal, "oracle_hit_rate_online", "k", "-"),
                ("calibrated, guaranteed", cal, "guaranteed_hit_rate", "tab:purple", "-"),
                ("calibrated, naive", cal, "naive_hit_rate", "tab:purple", ":"),
            ]
            if cert is not None:
                series.append(("best certificate (A1+A3)", cert,
                               "a1+a3_1000_hit_rate_online", "tab:green", "--"))
            print(f"  {title}")
            for label, rows, key, color, ls in series:
                rows = [r for r in rows if key in r]
                xs = np.array([r["cache_frac"] for r in rows])
                ys = np.array([r[key] for r in rows])
                keep = (xs > 0) & (ys > 0)
                order = np.argsort(xs[keep])
                ax.plot(xs[keep][order], ys[keep][order], "o", ls=ls, ms=3.5,
                        color=color, label=label)
                cells = []
                for s in SIZES:
                    r = _near(rows, s)
                    cell = f"{100 * r[key]:6.2f}"
                    if key == "guaranteed_hit_rate":
                        ret = r.get("guaranteed_hit_retention")
                        cell += f" [{ret:.3f}]" if ret is not None else " [  -  ]"
                    cells.append(cell)
                print(f"    {label:<28}" + "".join(f"{c:>16}" for c in cells))
            ref = np.logspace(-4, 0, 50)
            ax.plot(ref, ref, "k:", lw=1, alpha=0.6, label="y = x")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(5e-4, 0.35)
            ax.set_ylim(1e-4, 1.05)
            ax.grid(True, alpha=0.3)
            ax.set_title(title, fontsize=10)
        for ax in axes.flat[len(RUNS):]:
            ax.axis("off")
        for ax in axes[-1]:
            ax.set_xlabel("cache size (fraction of database vectors)")
        for ax in axes[:, 0]:
            ax.set_ylabel("hit rate (fraction of test queries)")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9)
        fig.suptitle(f"T={target}: oracle vs calibrated admission (95% guarantee) "
                     "vs certificates (caches selected for oracle)", fontsize=12)
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        out = RESULTS / f"calibrated_T{target}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
