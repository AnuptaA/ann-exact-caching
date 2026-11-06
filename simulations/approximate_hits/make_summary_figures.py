#!/usr/bin/env python3
"""Summary figures: radius-estimation baselines vs direct retention prediction,
and guaranteed (calibrated) vs oracle hit rates.

Writes into results/summary/:
  fig1_radius_baselines.png   radius error, retention error, trusted-estimate
                              realized retention, guaranteed hit rate
  fig2_esci_hit_rate.png      ESCI 50k / 1M, T = 0.9 / 0.95
  fig3_workloads_1m.png       all workloads at 1M, T = 0.9
  fig4_aker.png               Aker traces (10M databases), T = 0.9
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).parent / "results"
VIOL_TOL = 0.01
OUT = RESULTS / "summary"
EST_LABELS = {"true_radius": "true radius\n(sanity check)", "lsh": "LSH radius",
              "global_mean": "global-mean\nradius", "ridge": "ridge\n(retention)",
              "gbt": "GBT\n(retention)"}
EST_COLORS = {"true_radius": "0.6", "lsh": "tab:red", "global_mean": "tab:orange",
              "ridge": "tab:blue", "gbt": "tab:purple"}


def _load(path):
    return json.loads(path.read_text()) if path.exists() else None


def _state(d, target, size):
    rows = [s for s in d["states"] if abs(s["target"] - target) < 1e-9]
    return min(rows, key=lambda s: abs(s["cache_frac"] - size))


def fig1():
    runs = [("esci_score20k", "ESCI (real queries)"), ("sift_score20k", "SIFT (real queries)")]
    fig, axes = plt.subplots(len(runs), 3, figsize=(16, 5.0 * len(runs)), squeeze=False)
    for row, (run, title) in enumerate(runs):
        d = _load(RESULTS / run / "radius_baselines.json")
        if d is None:
            continue
        r = d["radius"]
        s = _state(d, 0.9, 0.10)
        ests = ["true_radius", "lsh", "global_mean", "ridge", "gbt"]

        ax = axes[row, 0]
        vals = [100 * r["lsh"]["rmse"] / r["true_mean"],
                100 * r["global_mean"]["rmse"] / r["true_mean"]]
        bars = ax.bar(["LSH buckets", "global average"], vals,
                      color=[EST_COLORS["lsh"], EST_COLORS["global_mean"]])
        ax.bar_label(bars, fmt="%.1f%%")
        skill = 1 - r["lsh"]["rmse"] / r["global_mean"]["rmse"]
        ax.set_ylabel("radius RMSE (% of mean radius)")
        ax.set_title(f"{title}\nradius error (LSH only {100 * skill:.0f}% better)", fontsize=10)

        ax = axes[row, 1]
        mae = [s[e]["retention_mae"] for e in ests]
        bars = ax.bar([EST_LABELS[e] for e in ests], mae, color=[EST_COLORS[e] for e in ests])
        ax.bar_label(bars, fmt="%.3f")
        ax.set_ylabel("mean |estimated - true retention|")
        ax.set_title("retention error (T=0.9, 10% cache)", fontsize=10)
        ax.tick_params(axis="x", labelsize=8)

        ax = axes[row, 2]
        x = np.arange(len(ests))
        trust = [s[e]["trusted_realized_retention"] or 0 for e in ests]
        guar = [100 * s[e]["guaranteed_hit_rate"] for e in ests]
        b1 = ax.bar(x - 0.2, guar, 0.4, color=[EST_COLORS[e] for e in ests],
                    label="guaranteed hit rate (%)")
        ax.bar_label(b1, fmt="%.1f", fontsize=8)
        ax.axhline(100 * s["oracle_best_case_hit_rate"], color="k", ls="--", lw=1,
                   label=f"oracle best case ({100 * s['oracle_best_case_hit_rate']:.1f}%)")
        ax.set_xticks(x)
        ax.set_xticklabels([EST_LABELS[e] for e in ests], fontsize=8)
        ax.set_ylabel("guaranteed test hit rate (%)")
        ax2 = ax.twinx()
        ax2.plot(x + 0.2, trust, "kD", ms=6, label="realized retention if estimate trusted")
        ax2.axhline(0.9, color="tab:green", lw=1, label="target T = 0.9")
        ax2.set_ylim(0.3, 1.0)
        ax2.set_ylabel("realized retention when trusting estimate")
        ax.set_title("end to end (T=0.9, 10% cache)", fontsize=10)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper center",
                  bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_radius_baselines.png", dpi=120)
    plt.close(fig)


def _oracle_curves(run_dir, target):
    d = _load(run_dir / "results.json")
    for r in d["results"]:
        if abs(r["target"] - target) < 1e-9:
            pts = sorted(r["points"], key=lambda p: p["cache_frac"])
            return ([p["cache_frac"] for p in pts], [p["test_hit_rate"] for p in pts],
                    [p["test_hit_rate_online"] for p in pts])
    return None


def _calibrated_curve(run_dir, target):
    d = _load(run_dir / "calibrated.json")
    if d is None:
        return None
    for r in d["results"]:
        if abs(r["target"] - target) < 1e-9:
            rows = sorted((x for x in r["rows"] if "guaranteed_hit_rate" in x),
                          key=lambda x: x["cache_frac"])
            return ([x["cache_frac"] for x in rows], [x["guaranteed_hit_rate"] for x in rows],
                    [x["guaranteed_hit_retention"] for x in rows])
    return None


def _best_predictor_points(run_dir, target):
    d = _load(run_dir / "predictors.json")
    if d is None:
        return None
    states = d["states"]
    models = [m for m in d["models"] if all(m in s for s in states)]
    best = max(models, key=lambda m: np.mean([s[m]["guaranteed_dev_hit_rate"] for s in states]))
    rows = sorted((s for s in states if abs(s["target"] - target) < 1e-9),
                  key=lambda s: s["cache_frac"])
    return (best, [s["cache_frac"] for s in rows],
            [s[best]["guaranteed_test_hit_rate"] for s in rows],
            [s[best]["guaranteed_test_hit_retention"] for s in rows])


def _panel(ax, run_dir, target, title, xlim=(5e-4, 0.35)):
    oc = _oracle_curves(run_dir, target)
    pos = lambda xs, ys: [(x, y) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if oc:
        xy = pos(oc[0], oc[1])
        ax.plot(*zip(*xy), "-", color="k", lw=1.5, label="oracle, best case")
        xy = pos(oc[0], oc[2])
        ax.plot(*zip(*xy), "--", color="k", lw=1, label="oracle, online")
    # Points whose realized test retention misses the target (beyond noise).
    bad = []
    cc = _calibrated_curve(run_dir, target)
    if cc:
        xy = pos(cc[0], cc[1])
        if xy:
            ax.plot(*zip(*xy), "o-", ms=3, color="tab:blue", label="guaranteed, ridge")
        bad += [(x, y) for x, y, t in zip(*cc) if x > 0 and y > 0 and t < target - VIOL_TOL]
    bp = _best_predictor_points(run_dir, target)
    if bp:
        xy = pos(bp[1], bp[2])
        if xy:
            ax.plot(*zip(*xy), "s-", ms=5, color="tab:purple",
                    label=f"guaranteed, best predictor ({bp[0]})")
        bad += [(x, y) for x, y, t in zip(*bp[1:]) if x > 0 and y > 0 and t < target - VIOL_TOL]
    if bad:
        ax.plot(*zip(*bad), "x", ms=9, mew=2, color="tab:red", ls="none",
                label="test retention below target")
        ax.legend(fontsize=7, loc="lower right")
    ref = np.logspace(-4, 0, 50)
    ax.plot(ref, ref, ":", color="0.4", lw=1, label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(1e-3, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("cache size (fraction of database)")
    ax.set_ylabel("hit rate (fraction of test queries)")


def fig2():
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
    runs = [(RESULTS / "esci_score20k", "ESCI, 50k database"),
            (RESULTS / "batch_db1m" / "esci_db1m_k10", "ESCI, 1M database")]
    for col, (run_dir, name) in enumerate(runs):
        for row, target in enumerate((0.9, 0.95)):
            _panel(axes[row, col], run_dir, target, f"{name}, T={target}")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle("ESCI real queries, k = 10: guaranteed (95% confidence) vs oracle hit rate",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "fig2_esci_hit_rate.png", dpi=120)
    plt.close(fig)


def fig3():
    runs = [("esci_db1m_k10", "ESCI, real queries"),
            ("syn_interp_zipf0.3_db1m_k10", "simZipf 0.3"),
            ("syn_interp_zipf0.6_db1m_k10", "simZipf 0.6"),
            ("syn_interp_zipf0.99_db1m_k10", "simZipf 0.99"),
            ("syn_gauss0.1_zipf0.99_db1m_k10", "simZipf 0.99, small Gaussian")]
    fig, axes = plt.subplots(1, len(runs), figsize=(4.4 * len(runs), 4.4), sharey=True)
    for ax, (run, title) in zip(axes, runs):
        _panel(ax, RESULTS / "batch_db1m" / run, 0.9, title)
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("All workloads, 1M database, k = 10, T = 0.9", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig3_workloads_1m.png", dpi=120)
    plt.close(fig)


def fig4():
    rows = [[("spacev-aker-0.3", "SPACEV, Zipf 0.3"),
             ("spacev-aker-0.6", "SPACEV, Zipf 0.6"),
             ("spacev-aker-0.99", "SPACEV, Zipf 0.99"),
             ("spacev-real", "SPACEV, query log -> official queries")],
            [("triviaqa-aker-0.3", "TriviaQA / SPHERE, Zipf 0.3"),
             ("triviaqa-aker-0.6", "TriviaQA / SPHERE, Zipf 0.6"),
             ("triviaqa-aker-0.99", "TriviaQA / SPHERE, Zipf 0.99"),
             None]]
    fig, axes = plt.subplots(2, 4, figsize=(17.6, 8.8), sharey=True)
    for r, row in enumerate(rows):
        for c, item in enumerate(row):
            ax = axes[r, c]
            run_dir = RESULTS / "batch_aker" / item[0] if item else None
            if run_dir is None or not (run_dir / "results.json").exists():
                ax.axis("off")
                continue
            _panel(ax, run_dir, 0.9, item[1], xlim=(3e-4, 0.07))
    axes[0, 0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Aker workloads, 10M databases, k = 10, T = 0.9", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "fig4_aker.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig1()
    fig2()
    fig3()
    fig4()
    print(f"wrote figures to {OUT}")
