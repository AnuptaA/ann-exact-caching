#!/usr/bin/env python3
"""One figure with a panel per dataset/workload: online hit rate vs cache size.

Reads results/<run>/results.json and writes results/master_online_<scale>.png.
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).parent / "results"

# (run folder, panel title)
RUNS = [
    ("sift_score20k", "SIFT, real queries"),
    ("esci_score20k", "ESCI, real queries"),
    ("syn_interp_zipf0.3", "SIFT, simZipf s=0.3 (interp)"),
    ("syn_interp_zipf0.6", "SIFT, simZipf s=0.6 (interp)"),
    ("syn_interp_zipf0.99", "SIFT, simZipf s=0.99 (interp)"),
    ("syn_gauss0.1_zipf0.99", "SIFT, simZipf s=0.99 (gauss 0.1)"),
]
KEY = "test_hit_rate_online"


def plot(scale: str, out: Path) -> None:
    log = scale == "loglog"
    ncols = 3
    nrows = int(np.ceil(len(RUNS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.6 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    handles, labels = None, None
    for ax, (run, title) in zip(axes.flat, RUNS):
        data = json.loads((RESULTS / run / "results.json").read_text())
        results = data["results"]
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(results)))
        for res, color in zip(results, colors):
            xs = np.array([p["cache_frac"] for p in res["points"]])
            ys = np.array([p[KEY] for p in res["points"]])
            keep = (xs > 0) & (ys > 0) if log else np.ones(len(xs), dtype=bool)
            order = np.argsort(xs[keep])
            ax.plot(xs[keep][order], ys[keep][order], "o-", ms=3.5,
                    color=color, label=f"T={res['target']}")
        ref = np.logspace(-4, 0, 50) if log else np.array([0.0, 1.0])
        ax.plot(ref, ref, "k--", lw=1, label="y = x")
        n_test = data["config"].get("n_test", data["config"].get("n_test_requests"))
        ax.set_title(f"{title}\n(n_db={data['config']['n_db']}, test={n_test})",
                     fontsize=10)
        if log:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(5e-4, 0.35)
            ax.set_ylim(1e-4, 1.05)
        else:
            ax.set_xlim(0.0, 0.33)
            ax.set_ylim(0.0, 1.0)
        ax.grid(True, which="major", alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
    for ax in axes.flat[len(RUNS):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("cache size (fraction of database vectors)")
    for ax in axes[:, 0]:
        ax.set_ylabel("online hit rate (fraction of test queries)")
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Online hit rate vs cache size, oracle radius, greedy population "
                 "(hits-only average policy)", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", nargs="+", default=["loglog", "linear"])
    args = ap.parse_args()
    for scale in args.scales:
        out = RESULTS / f"master_online_{scale}.png"
        plot(scale, out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
