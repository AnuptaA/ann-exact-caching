#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Run the LSH-bucket radius-estimation analysis across all scenarios.

For each scenario we:
  1. compute the true radius field (k-th NN distance) for data/train/test
  2. fit the LSH estimator on train radii, estimate test radii
  3. measure error = estimate - true, as a function of the true radius
  4. compute the analytical smoothing vs count-weighting decomposition over the
     SAME regions used empirically
  5. emit per-query CSV, per-scenario summary JSON, and diagnostic plots

Empirical and analytical paths share the exact bucket regions, so the two can
be compared directly.
"""

import sys
import json
import csv
import argparse
import logging
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simulations.radius_estimation.lsh_radius_estimator import (
    LSHRadiusEstimator, true_radius,
)
from simulations.radius_estimation.scenarios import build_scenarios
from simulations.radius_estimation.analytical import (
    decompose_region_bias, predict_sign_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

#-------------------------------------------------------------------------------


def _region_codes(estimator, code, hamming):
    """Set of bucket codes within hamming distance of code."""
    return set(estimator._neighbors_within_hamming(code, hamming))


def _uniform_region_radii(
    estimator, region_codes, data, k, n_samples, rng, lo, hi,
):
    """Sample points uniformly in the cube, keep those whose bucket is in
    region_codes, and return their true radii. Approximates the unweighted
    regional mean r_bar_R. Falls back to empty if nothing lands in region.
    """
    samples = rng.uniform(lo, hi, size=(n_samples, data.shape[1]))
    codes = estimator._codes(samples)
    keep = np.array([c in region_codes for c in codes.tolist()])
    if not keep.any():
        return np.empty(0, dtype=np.float64)
    return true_radius(samples[keep], data, k)


def run_scenario(
    scenario, k, n_bits, hamming, lsh_seed, out_dir, uniform_probe=3000,
):
    logger.info(f"Scenario: {scenario.name}")
    dim = scenario.data.shape[1]

    r_train = true_radius(scenario.train_queries, scenario.data, k)
    r_test = true_radius(scenario.test_queries, scenario.data, k)
    global_mean = float(np.mean(r_train))

    est = LSHRadiusEstimator(n_bits=n_bits, dim=dim, seed=lsh_seed)
    est.fit(scenario.train_queries, r_train)
    out = est.estimate(
        scenario.test_queries, hamming=hamming,
        expand_fallback=True, global_mean=global_mean,
    )

    estimate = out["estimate"]
    error = estimate - r_test

    # analytical decomposition over the SAME regions (infinite-train proxy:
    # train radii in region = count-weighted; uniform probe = unweighted mean)
    rng = np.random.default_rng(lsh_seed + 1)
    lo = float(min(scenario.data.min(), scenario.test_queries.min()))
    hi = float(max(scenario.data.max(), scenario.test_queries.max()))
    test_codes = est._codes(scenario.test_queries)

    smoothing_terms = np.full(len(r_test), np.nan)
    weighting_terms = np.full(len(r_test), np.nan)
    # analytical decomposition is expensive; sample a subset of test queries
    probe_idx = np.linspace(0, len(r_test) - 1,
                            min(200, len(r_test))).astype(int)
    for i in probe_idx.tolist():
        h_used = int(out["hamming_used"][i])
        if h_used < 0 or h_used > n_bits:
            continue
        region = _region_codes(est, int(test_codes[i]), h_used)
        # per-train-query radii in the region (count-weighted mean)
        train_codes = est._codes(scenario.train_queries)
        in_region = np.array([c in region for c in train_codes.tolist()])
        region_train_all = r_train[in_region]
        uni = _uniform_region_radii(
            est, region, scenario.data, k, uniform_probe, rng, lo, hi,
        )
        dec = decompose_region_bias(r_test[i], region_train_all, uni)
        smoothing_terms[i] = dec["smoothing_term"]
        weighting_terms[i] = dec["weighting_term"]

    summary = {
        "scenario": scenario.name,
        "description": scenario.description,
        "k": k,
        "n_bits": n_bits,
        "hamming": hamming,
        "dim": dim,
        "n_data": int(len(scenario.data)),
        "n_train": int(len(scenario.train_queries)),
        "n_test": int(len(scenario.test_queries)),
        "true_radius_mean": float(np.nanmean(r_test)),
        "true_radius_std": float(np.nanstd(r_test)),
        "estimate_mean": float(np.nanmean(estimate)),
        "mean_error": float(np.nanmean(error)),
        "mean_abs_error": float(np.nanmean(np.abs(error))),
        "median_hamming_used": float(np.nanmedian(out["hamming_used"])),
        "mean_n_train_avg": float(np.nanmean(out["n_train_avg"])),
        "global_fallback_frac": float(np.mean(out["hamming_used"] > n_bits)),
        "sign_summary": predict_sign_summary(error, r_test, estimate),
        "mean_smoothing_term": float(np.nanmean(smoothing_terms)),
        "mean_weighting_term": float(np.nanmean(weighting_terms)),
    }

    _write_scenario_outputs(
        out_dir, scenario, r_test, estimate, error, out,
        smoothing_terms, weighting_terms, summary,
    )
    return summary


#-------------------------------------------------------------------------------


def _write_scenario_outputs(
    out_dir, scenario, r_test, estimate, error, out,
    smoothing_terms, weighting_terms, summary,
):
    sdir = out_dir / scenario.name
    sdir.mkdir(parents=True, exist_ok=True)

    with open(sdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(sdir / "per_query.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "true_radius", "estimate", "error",
                    "hamming_used", "n_train_avg", "n_buckets",
                    "smoothing_term", "weighting_term"])
        for i in range(len(r_test)):
            w.writerow([
                i, round(float(r_test[i]), 6),
                round(float(estimate[i]), 6) if np.isfinite(estimate[i]) else "",
                round(float(error[i]), 6) if np.isfinite(error[i]) else "",
                int(out["hamming_used"][i]),
                int(out["n_train_avg"][i]),
                int(out["n_buckets"][i]),
                round(float(smoothing_terms[i]), 6) if np.isfinite(smoothing_terms[i]) else "",
                round(float(weighting_terms[i]), 6) if np.isfinite(weighting_terms[i]) else "",
            ])

    _plot_scenario(sdir, scenario, r_test, estimate, error, summary)


def _plot_scenario(sdir, scenario, r_test, estimate, error, summary):
    mask = np.isfinite(estimate) & np.isfinite(r_test)
    r = r_test[mask]
    e = error[mask]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # estimate vs true, with y=x
    ax = axes[0]
    ax.scatter(r, estimate[mask], s=8, alpha=0.3, color="steelblue")
    lim = [min(r.min(), estimate[mask].min()), max(r.max(), estimate[mask].max())]
    ax.plot(lim, lim, "k--", lw=1, label="y = x (unbiased)")
    ss = summary["sign_summary"]
    ax.set_xlabel("true radius")
    ax.set_ylabel("estimated radius")
    ax.set_title(f"{scenario.name}\nestimate vs true "
                 f"(std compression={ss['std_compression']:.2f})")
    ax.legend()

    # error vs true radius, with trend line
    ax = axes[1]
    ax.axhline(0, color="k", lw=1)
    ax.scatter(r, e, s=8, alpha=0.3, color="darkorange")
    if np.isfinite(ss["slope"]):
        xs = np.linspace(r.min(), r.max(), 50)
        ax.plot(xs, ss["slope"] * xs + ss["intercept"], "r-", lw=2,
                label=f"slope={ss['slope']:.3f}, corr={ss['corr']:.2f}")
    ax.set_xlabel("true radius")
    ax.set_ylabel("error (estimate - true)")
    ax.set_title("error vs true radius")
    ax.legend()

    # binned bias (the honest regression-to-the-mean signature)
    ax = axes[2]
    ax.axhline(0, color="k", lw=1)
    if ss["bin_centers"]:
        ax.plot(ss["bin_centers"], ss["binned_bias"], "o-", color="forestgreen")
    ax.set_xlabel("true radius (bin center)")
    ax.set_ylabel("mean error in bin")
    ax.set_title("binned bias\n(+ at low radius, - at high => regression to mean)")

    plt.tight_layout()
    fig.savefig(sdir / "error_analysis.png", dpi=120)
    plt.close(fig)


def _plot_cross_scenario(out_dir, summaries):
    names = [s["scenario"] for s in summaries]
    mean_err = [s["mean_error"] for s in summaries]
    smoothing = [s["mean_smoothing_term"] for s in summaries]
    weighting = [s["mean_weighting_term"] for s in summaries]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    y = np.arange(len(names))

    ax = axes[0]
    ax.barh(y, mean_err, color="darkorange")
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("mean error over test distribution")
    ax.set_title("Aggregate bias by scenario")

    ax = axes[1]
    width = 0.35
    ax.barh(y - width / 2, smoothing, height=width, label="smoothing term", color="steelblue")
    ax.barh(y + width / 2, weighting, height=width, label="weighting term", color="forestgreen")
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("analytical term (mean over probed test queries)")
    ax.set_title("Bias decomposition by scenario")
    ax.legend()

    plt.tight_layout()
    fig.savefig(out_dir / "cross_scenario_summary.png", dpi=120)
    plt.close(fig)


#-------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="LSH-bucket radius-estimation bias analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--k", type=int, default=10, help="k-th NN radius")
    parser.add_argument("--n-bits", type=int, default=16, help="LSH bits")
    parser.add_argument("--hamming", type=int, default=0, help="Hamming threshold")
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--n-data", type=int, default=5000)
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else (
        Path(__file__).parent / "results" / run_ts
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dir: {out_dir}")

    scenarios = build_scenarios(
        dim=args.dim, n_data=args.n_data,
        n_train=args.n_train, n_test=args.n_test, seed=args.seed,
    )

    summaries = []
    for sc in scenarios:
        summ = run_scenario(
            sc, k=args.k, n_bits=args.n_bits, hamming=args.hamming,
            lsh_seed=args.seed, out_dir=out_dir,
        )
        summaries.append(summ)
        ss = summ["sign_summary"]
        bias_lo = ss["binned_bias"][0] if ss["binned_bias"] else float("nan")
        bias_hi = ss["binned_bias"][-1] if ss["binned_bias"] else float("nan")
        logger.info(
            f"  mean_err={summ['mean_error']:+.4f}  "
            f"stdComp={ss['std_compression']:.2f}  "
            f"bias_lo={bias_lo:+.4f}  bias_hi={bias_hi:+.4f}  "
            f"smooth={summ['mean_smoothing_term']:+.4f}  "
            f"weight={summ['mean_weighting_term']:+.4f}  "
            f"fallback={summ['global_fallback_frac']:.2%}"
        )

    with open(out_dir / "all_summaries.json", "w") as f:
        json.dump(summaries, f, indent=2)
    _plot_cross_scenario(out_dir, summaries)

    _print_report(summaries)


def _print_report(summaries):
    print(f"\n{'='*104}")
    print("RADIUS-ESTIMATION BIAS ANALYSIS")
    print(f"{'='*104}")
    print(f"{'scenario':<36}{'mean_err':>10}{'stdComp':>9}"
          f"{'bias_lo':>9}{'bias_hi':>9}{'smooth':>9}{'weight':>9}{'fallbk':>8}")
    print(f"{'-'*104}")
    for s in summaries:
        ss = s["sign_summary"]
        bias_lo = ss["binned_bias"][0] if ss["binned_bias"] else float("nan")
        bias_hi = ss["binned_bias"][-1] if ss["binned_bias"] else float("nan")
        print(f"{s['scenario']:<36}{s['mean_error']:>+10.4f}"
              f"{ss['std_compression']:>9.2f}{bias_lo:>+9.4f}{bias_hi:>+9.4f}"
              f"{s['mean_smoothing_term']:>+9.4f}{s['mean_weighting_term']:>+9.4f}"
              f"{s['global_fallback_frac']:>7.1%}")
    print(f"{'='*104}")
    print("stdComp = std(estimate)/std(true); << 1 means heavy regression-to-the-mean")
    print("bias_lo/bias_hi = mean error in lowest/highest true-radius bin")
    print("  (+ at low, - at high => estimator compresses radius toward the regional mean)")
    print("smooth = regression-to-the-mean term; weight = count-weighting term")
    print(f"{'='*104}\n")


if __name__ == "__main__":
    main()
