#!/usr/bin/env python3
"""Summarize predictors.json files: model comparison on dev, report on test.

For each run: mean guaranteed dev hit rate per model across evaluated cache
states (model selection), and test hit rate / realized retention for the
dev-selected model vs the baseline ridge.
"""

from pathlib import Path

from simulations.approximate_hits.report import common


def add_args(ap):
    ap.add_argument("paths", nargs="*",
                    help="predictors.json files (default: all under the results directory)")
    ap.set_defaults(run=run)


def run(args):
    paths = args.paths or sorted(str(p) for p in common.RESULTS.rglob("predictors.json"))
    for path in paths:
        d = common.load_json(path)
        states = d["states"]
        models = [m for m in d["models"] if all(m in s for s in states)]
        if not states or not models:
            continue
        run_dir = Path(path).resolve().parent
        name = (run_dir.relative_to(common.RESULTS.resolve())
                if run_dir.is_relative_to(common.RESULTS.resolve()) else run_dir.name)
        print(f"\n== {name}  ({len(states)} states)")
        print(f"  {'model':<12}{'dev guar':>10}{'dev naive':>11}{'dev spear':>11}"
              f"{'test guar':>11}{'min test ret / T':>18}")
        mean = lambda xs: sum(xs) / len(xs)
        dev_scores = {}
        for m in models:
            dg = mean([s[m]["guaranteed_dev_hit_rate"] for s in states])
            dn = mean([s[m]["naive_dev_hit_rate"] for s in states])
            sp = mean([s[m]["dev_spearman"] for s in states])
            tg = mean([s[m]["guaranteed_test_hit_rate"] for s in states])
            margins = [s[m]["guaranteed_test_hit_retention"] - s["target"] for s in states
                       if s[m]["guaranteed_test_hit_retention"] is not None]
            worst = f"{min(margins):+.3f}" if margins else "-"
            dev_scores[m] = dg
            print(f"  {m:<12}{100 * dg:>10.2f}{100 * dn:>11.2f}{sp:>11.2f}"
                  f"{100 * tg:>11.2f}{worst:>18}")
        best = max(dev_scores, key=dev_scores.get)
        oracle = mean([s["oracle_hit_rate_online"] for s in states])
        print(f"  dev-selected: {best}; oracle online (mean over states) {100 * oracle:.2f}")
        print(f"  per state (T, cache%): oracle online | ridge_base -> {best} (test guaranteed %)")
        for s in sorted(states, key=lambda s: (s["target"], s["cache_frac"])):
            print(f"    T={s['target']:<5} {100 * s['cache_frac']:>5.1f}%: "
                  f"{100 * s['oracle_hit_rate_online']:6.2f} | "
                  f"{100 * s['ridge_base']['guaranteed_test_hit_rate']:6.2f} -> "
                  f"{100 * s[best]['guaranteed_test_hit_rate']:6.2f}")

