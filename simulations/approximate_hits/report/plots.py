#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Hit rate versus cache size plots and the per-target console table."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

#-------------------------------------------------------------------------------


def plot_results(results, out_dir, n_db, title="SIFT"):
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(results)))
    panels = [
        ("test_hit_rate", "hit rate (best-case admission)"),
        ("test_hit_rate_online", "hit rate (online admission)"),
    ]
    for scale in ["linear", "loglog"]:
        log = scale == "loglog"
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
        for ax, (key, panel_title) in zip(axes, panels):
            for res, color in zip(results, colors):
                xs = np.array([p["cache_frac"] for p in res["points"]])
                ys = np.array([p[key] for p in res["points"]])
                keep = (xs > 0) & (ys > 0) if log else np.ones(len(xs), dtype=bool)
                order = np.argsort(xs[keep])
                ax.plot(xs[keep][order], ys[keep][order], "o-",
                        color=color, label=f"T={res['target']}")
            ref = np.logspace(-4, 0, 50) if log else np.array([0.0, 1.0])
            ax.plot(ref, ref, "k--", lw=1, label="y = x")
            x_max = max(p["cache_frac"] for r in results for p in r["points"]) * 1.1
            if log:
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlim(5e-4, x_max)
                ax.set_ylim(1e-4, 1.05)
            else:
                ax.set_xlim(0.0, x_max)
                ax.set_ylim(0, 1)
            ax.set_xlabel("cache size (fraction of database vectors)")
            ax.set_ylabel("fraction of test queries")
            ax.set_title(panel_title, fontsize=10)
            ax.legend(fontsize=8)
        fig.suptitle(f"{title}, oracle radius, greedy population (hits-only average policy)")
        plt.tight_layout()
        fig.savefig(out_dir / f"hit_rate_vs_cache_size_{scale}.png", dpi=120)
        plt.close(fig)


def print_target(res):
    print(f"\n  T={res['target']}")
    print(f"  {'cache%':>8} {'hit%':>7} {'eff%':>7} {'onl hit%':>9} {'onl eff%':>9} "
          f"{'train%':>7} {'entries':>8}")
    for p in res["points"]:
        print(f"  {100*p['cache_frac']:>8.2f} {100*p['test_hit_rate']:>7.2f} "
              f"{100*p['test_effective_retention']:>7.2f} "
              f"{100*p['test_hit_rate_online']:>9.2f} "
              f"{100*p['test_effective_retention_online']:>9.2f} "
              f"{100*p['train_hit_rate']:>7.2f} {p['n_entries']:>8}")
