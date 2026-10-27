#!/usr/bin/env python3
"""Online hit rate vs cache size: oracle retention vs provable certificates.

One panel per workload at a fixed target T. Reads results/<run>/certificates.json
and writes results/certificates_T<T>.png; also prints a table.
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
METHODS = [
    ("oracle", "oracle (true retention)", "k", "-"),
    ("cig", "exact CIG", "tab:red", "-"),
    ("a1", "A1 partial CIG", "tab:orange", "-"),
    ("a3_1000", "A3 profile M=1000", "tab:blue", "-"),
    ("a1+a3_1000", "A1+A3 M=1000", "tab:green", "--"),
]
SIZES = [0.01, 0.05, 0.10, 0.30]


def load(run, target):
    path = RESULTS / run / "certificates.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    for t in data["results"]:
        if abs(t["target"] - target) < 1e-9:
            return t["rows"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=float, nargs="+", default=[0.9, 0.95])
    args = ap.parse_args()
    for target in args.targets:
        ncols = 3
        nrows = int(np.ceil(len(RUNS) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.6 * nrows),
                                 sharex=True, sharey=True, squeeze=False)
        print(f"\n=== T={target}: online hit % at ~1% / 5% / 10% / 30% cache ===")
        for ax, (run, title) in zip(axes.flat, RUNS):
            rows = load(run, target)
            if rows is None:
                ax.set_title(f"{title}\n(missing)")
                continue
            print(f"  {title}")
            for key, label, color, ls in METHODS:
                xs = np.array([r["cache_frac"] for r in rows])
                ys = np.array([r[f"{key}_hit_rate_online"] for r in rows])
                keep = (xs > 0) & (ys > 0)
                order = np.argsort(xs[keep])
                ax.plot(xs[keep][order], ys[keep][order], "o", ls=ls, ms=3.5,
                        color=color, label=label)
                cells = []
                for s in SIZES:
                    r = min(rows, key=lambda r: abs(r["cache_frac"] - s))
                    cells.append(f"{100 * r[f'{key}_hit_rate_online']:6.2f}")
                print(f"    {label:<26}" + "".join(f"{c:>9}" for c in cells))
            ref = np.logspace(-4, 0, 50)
            ax.plot(ref, ref, "k:", lw=1, label="y = x")
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
            ax.set_ylabel("online hit rate")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9)
        fig.suptitle(f"Online hit rate at T={target}: oracle retention vs provable "
                     "lower bounds (caches selected for oracle)", fontsize=12)
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        out = RESULTS / f"certificates_T{target}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
