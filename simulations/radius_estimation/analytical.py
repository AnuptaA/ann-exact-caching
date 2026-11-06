#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Analytical bias decomposition for the LSH-bucket radius estimator.

We work in the infinite-train limit so finite-sample variance drops out and
only the structural bias remains. For a fixed test query q landing in a region
R (the union of buckets within the Hamming threshold), the estimator converges
to the train-density-weighted average of the true radius field over R:

    r_hat(q) -> E_{x ~ p_train | x in R} [ r_true(x) ]

where p_train is the train-query density. The structural error is

    err(q) = E_{x ~ p_train | R}[ r_true(x) ] - r_true(q).

This splits into two interpretable terms. Let r_bar_R be the UNWEIGHTED
(uniform-over-R) mean of r_true, and let the weighted mean be
E_w[r_true] = E_{p_train|R}[r_true]. Then

    err(q) = ( r_bar_R - r_true(q) )            # smoothing / regression-to-mean
           + ( E_w[r_true] - r_bar_R )          # count-weighting

Term 1 depends only on how r_true varies across R (non-uniform data). For q at
an extreme of the field it has a definite sign: negative (underestimate) where
r_true(q) is above the regional mean, positive where below -> regression to the
mean, and hence error that depends on the true radius.

Term 2 is the covariance between the train density and r_true inside R,
normalized by the mean density. It vanishes when train queries are uniform over
R (flat weighting) OR when r_true is constant over R (flat field). It is
negative when train density is higher where r_true is smaller (queries track
dense/small-radius data), which is the realistic clustered case.

This module provides finite-sample estimators of these two terms given the same
data/queries used empirically, so theory and simulation use identical regions.
"""

import numpy as np

#-------------------------------------------------------------------------------


def decompose_region_bias(
    r_true_test: float,
    region_train_radii: np.ndarray,
    region_uniform_radii: np.ndarray,
) -> dict:
    """Decompose structural error for one test query into the two terms.

    Args:
        r_true_test: true radius of the test query.
        region_train_radii: r_true of train queries falling in the region R
            (their empirical distribution approximates p_train | R, so a plain
            mean is the count-weighted estimate).
        region_uniform_radii: r_true sampled UNIFORMLY over the same region R
            (approximates the unweighted regional mean r_bar_R).

    Returns dict with weighted_mean, uniform_mean, smoothing_term,
    weighting_term, total_error.
    """
    weighted_mean = float(np.mean(region_train_radii)) if len(region_train_radii) else np.nan
    uniform_mean = float(np.mean(region_uniform_radii)) if len(region_uniform_radii) else np.nan
    smoothing = uniform_mean - r_true_test
    weighting = weighted_mean - uniform_mean
    total = weighted_mean - r_true_test
    return {
        "weighted_mean": weighted_mean,
        "uniform_mean": uniform_mean,
        "smoothing_term": smoothing,
        "weighting_term": weighting,
        "total_error": total,
    }


#-------------------------------------------------------------------------------


def predict_sign_summary(errors: np.ndarray, r_true: np.ndarray,
                         estimate: np.ndarray = None, n_bins: int = 8) -> dict:
    """Summarize the error-vs-true-radius pattern.

    The global linear slope of error vs true radius is reported for continuity,
    but it is NOT a reliable summary when the estimator is near-constant: a
    constant estimate forces slope = -1 and corr = -1 trivially. The honest
    signals are:

      - std_compression: std(estimate) / std(r_true). A heavy low-pass /
        regression-to-the-mean estimator has ratio << 1. This is the primary
        diagnostic of how much true-radius signal survives.
      - binned_bias: mean error within equal-count bins of true radius. A
        regression-to-the-mean estimator shows positive bias (over-estimate) in
        the low-radius bins and negative bias (under-estimate) in the high ones,
        regardless of how near-constant the estimate is.
    """
    mask = np.isfinite(errors) & np.isfinite(r_true)
    e = errors[mask]
    r = r_true[mask]
    out = {"slope": np.nan, "intercept": np.nan,
           "mean_error": float(np.mean(e)) if len(e) else np.nan,
           "crossover_radius": np.nan, "corr": np.nan,
           "std_compression": np.nan, "binned_bias": [], "bin_centers": []}
    if len(e) < 2:
        return out
    slope, intercept = np.polyfit(r, e, 1)
    out["slope"] = float(slope)
    out["intercept"] = float(intercept)
    out["crossover_radius"] = float(-intercept / slope) if slope != 0 else np.nan
    out["corr"] = float(np.corrcoef(r, e)[0, 1])
    if estimate is not None:
        est = estimate[mask]
        rt_std = float(np.std(r))
        out["std_compression"] = float(np.std(est) / rt_std) if rt_std > 0 else np.nan

    # equal-count bins of true radius
    order = np.argsort(r)
    r_sorted = r[order]
    e_sorted = e[order]
    edges = np.linspace(0, len(r), n_bins + 1).astype(int)
    centers, biases = [], []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if hi <= lo:
            continue
        centers.append(float(np.mean(r_sorted[lo:hi])))
        biases.append(float(np.mean(e_sorted[lo:hi])))
    out["bin_centers"] = centers
    out["binned_bias"] = biases
    return out
