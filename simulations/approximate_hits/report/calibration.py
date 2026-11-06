#!/usr/bin/env python3
"""Summarize calibration_methods.json: failure rates and hit rates per method."""

from simulations.approximate_hits.report import common
RUNS = [
    ("esci_score20k", "ESCI"),
    ("syn_interp_zipf0.3", "simZipf 0.3"),
    ("syn_interp_zipf0.6", "simZipf 0.6"),
    ("syn_interp_zipf0.99", "simZipf 0.99"),
    ("syn_gauss0.1_zipf0.99", "gauss 0.99"),
]
LABELS = {
    "naive": "naive (no bound)",
    "eb_bonferroni": "EB + Bonferroni",
    "normal_bonferroni": "Normal + Bonferroni",
    "normal_fixed": "Normal + fixed order",
    "eb_fixed": "EB + fixed order",
    "normal_split": "Normal + split",
}


def add_args(ap):
    ap.set_defaults(run=run)


def run(args):
    loaded = []
    for run_name, label in RUNS:
        data = common.load_json(common.RESULTS / run_name / "calibration_methods.json")
        if data is not None:
            loaded.append((label, data))
    if not loaded:
        print("no results")
        return
    methods = loaded[0][1]["methods"]

    print("\nFailure rate (fraction of trials where admitted queries' mean retention < T),")
    print("averaged over states; worst single state in brackets. Target: <= ~0.05")
    print(f"  {'method':<22}" + "".join(f"{lab:>18}" for lab, _ in loaded))
    for m in methods:
        cells = []
        for _, d in loaded:
            rates = [s[m]["fail_rate"] for s in d["states"]]
            cells.append(f"{sum(rates) / len(rates):.3f} [{max(rates):.2f}]")
        print(f"  {LABELS[m]:<22}" + "".join(f"{c:>18}" for c in cells))

    for target in sorted({s["target"] for _, d in loaded for s in d["states"]}):
        print(f"\nMean hit rate % over trials at T={target} "
              "(evaluation pool), by cache size; oracle online first")
        for lab, d in loaded:
            states = sorted((s for s in d["states"] if abs(s["target"] - target) < 1e-9),
                            key=lambda s: s["cache_frac"])
            sizes = "".join(f"{100 * s['cache_frac']:>9.1f}%" for s in states)
            print(f"  {lab}: cache {sizes}")
            print(f"    {'oracle (online)':<22}" + "".join(
                f"{100 * s['oracle_hit_rate_online']:>10.2f}" for s in states))
            for m in methods:
                print(f"    {LABELS[m]:<22}" + "".join(
                    f"{100 * s[m]['mean_eval_hit_rate']:>10.2f}" for s in states))

