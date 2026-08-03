#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Does geometric overlap between a cached candidate's K-sphere and a test
query's K-sphere predict the oracle hit outcome? Two overlap metrics are computed
and compared: an angular (spherical-cap) fraction and a solid Euclidean ball fraction.

Distances are in normalized angular units and converted where necessary
"""

import sys
import json
import math
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats
from scipy.stats import beta as beta_dist
from scipy.special import betainc
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent.parent))

from simulations.population import debug_common as dc
from simulations.population.debug_common import parse_benchmark_sizes
from simulations.population.run_population import BUDGET_FRACTIONS
from utils.base.selection import select_candidates

#-------------------------------------------------------------------------------

K_VALUES = [1, 5, 10]
N_BINS = 20
OUT_DIR = Path('simulations/debug/population_intersection_calibration')

_RUN_TIMESTAMP = None


def _run_dir() -> Path:
    """Output directory for the current process's plots, timestamped once
    per process (not per plot) so all outputs from one run land together."""
    global _RUN_TIMESTAMP
    if _RUN_TIMESTAMP is None:
        _RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
    return OUT_DIR / _RUN_TIMESTAMP

#-------------------------------------------------------------------------------


_CAP_SAMPLER_CHUNK_SIZE = 32


def sample_cap_points(
    poles: np.ndarray, radii: np.ndarray, n_samples: int, rng: np.random.Generator,
    chunk_size: int = _CAP_SAMPLER_CHUNK_SIZE,
) -> np.ndarray:
    """Uniformly sample n_samples points within each of M spherical caps on
    S^{d-1}, one cap per pole/radius pair.

    poles: (M, d) unit vectors. radii: (M,) angular radii, normalized
    units (r = theta/pi). Returns: (M, n_samples, d) float32 unit
    vectors, each within angular distance radii[i] of poles[i].

    Samples cos(theta) from a truncated Beta((d-1)/2, (d-1)/2)
    distribution (marginal of one coordinate of a uniform S^{d-1} point,
    restricted to the cap), then a uniform direction orthogonal to the
    pole. Uses sf/isf, not cdf/ppf: at real embedding dimensions with
    small cap radii, `1-cdf` underflows to 0.0, collapsing every sample
    to the pole; sf/isf stay accurate.

    Processes poles in chunks (chunk_size, capped by _MAX_CHUNK_ELEMENTS)
    to bound peak memory because a single un-chunked call can materialize
    tens of GB of float64 intermediates at real n_samples/d.
    """
    M, d = poles.shape
    a = (d - 1) / 2.0
    out = np.empty((M, n_samples, d), dtype=np.float32)

    # Cap: bound peak per-chunk float64 intermediate size
    # (~a few hundred MB) regardless of the chunk_size argument
    _MAX_CHUNK_ELEMENTS = 20_000_000
    max_chunk_by_samples = max(1, _MAX_CHUNK_ELEMENTS // max(1, n_samples * d))
    chunk_size = max(1, min(chunk_size, max_chunk_by_samples))

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        chunk_poles = poles[start:end]
        chunk_radii = radii[start:end]
        m = end - start

        theta = chunk_radii * np.pi
        lower_x = np.clip((1.0 + np.cos(theta)) / 2.0, 0.0, 1.0)
        S_lower = beta_dist.sf(lower_x, a, a)

        u = rng.random((m, n_samples))
        s_scaled = np.clip(u * S_lower[:, None], 0.0, S_lower[:, None])
        X = beta_dist.isf(s_scaled, a, a)
        t = 2.0 * X - 1.0

        g = rng.standard_normal((m, n_samples, d))
        poles_exp = chunk_poles[:, None, :]
        proj = np.sum(g * poles_exp, axis=-1, keepdims=True)
        g -= proj * poles_exp
        norm = np.linalg.norm(g, axis=-1, keepdims=True)
        norm = np.where(norm < 1e-12, 1.0, norm)
        g /= norm

        sqrt_term = np.sqrt(np.maximum(1.0 - t ** 2, 0.0))[:, :, None]
        g *= sqrt_term
        g += t[:, :, None] * poles_exp
        out[start:end] = g.astype(np.float32)

    return out


def single_cap_volume_fraction(r: np.ndarray, d: int) -> np.ndarray:
    """Closed-form fractional volume of a cap of angular radius r (normalized
    units) on S^{d-1}, via the regularized incomplete beta function (Li 2011).
    Used only to validate the sampler against a known-exact reference."""
    theta = r * np.pi
    x = np.sin(theta) ** 2
    a = (d - 1) / 2.0
    b = 0.5
    ib = betainc(a, b, x)
    return np.where(theta <= np.pi / 2, 0.5 * ib, 1.0 - 0.5 * ib)


def _require_float64(**kwargs) -> None:
    """Reject non-float64 numeric input (e.g. np.float32). This module's
    computations lose enough precision at float32 to produce spurious wrong
    results, so callers must pass float64 explicitly."""
    for name, value in kwargs.items():
        if isinstance(value, float):
            continue
        if isinstance(value, np.floating) and np.dtype(type(value)) == np.float64:
            continue
        raise TypeError(
            f"{name}={value!r} (dtype {type(value).__name__}) is not float64"
        )


def conditional_cap_fraction(theta: float, D: float, r_c: float, d: int) -> float:
    """g(theta): fraction of the cross-sectional S^{d-2} at co-latitude
    theta (from pole T, along the great circle through T and candidate
    pole Q) lying within candidate Q's r_c-cap. theta, D, r_c are true
    radians (converted from normalized units by the caller).

    Caller must handle theta=0 and D=0 separately (division by zero here).
    """
    c = (math.cos(r_c) - math.cos(theta) * math.cos(D)) / (math.sin(theta) * math.sin(D))
    c = min(1.0, max(-1.0, c))
    theta_cross = math.acos(c)  # == (arccos(c)/pi) * pi, skips the redundant round trip
    x = math.sin(theta_cross) ** 2
    a = (d - 2) / 2.0
    ib = betainc(a, 0.5, x)
    return 0.5 * ib if theta_cross <= math.pi / 2 else 1.0 - 0.5 * ib


def exact_intersection_fraction(
    D: float, r_c: float, r_t: float, d: int, verify: bool = False,
) -> float:
    """Exact intersection_fraction(c, t): the fraction of test query t's
    r_t-cap volume lying within candidate c's r_c-cap.

    Raises TypeError if any input is not float64

    Raises ValueError if both num and den underflow to exactly 0.0
    (instead of returning NaN from 0/0), happens for small r_t
    (roughly <=0.03-0.05) at high d (>=~384)
    """
    from scipy.integrate import quad

    _require_float64(D=D, r_c=r_c, r_t=r_t)
    D_a, rc_a, rt_a = D * np.pi, r_c * np.pi, r_t * np.pi

    # full-containment checked first: at r_t=0, "no overlap" and "fully
    # contained" both reduce to D==r_c, and containment is closed (<=).
    if D_a + rt_a <= rc_a:
        return 1.0
    if D_a >= rc_a + rt_a:
        return 0.0
    if D_a < 1e-12:
        if rc_a >= rt_a:
            return 1.0
        return float(
            single_cap_volume_fraction(np.array([r_c]), d)[0]
            / single_cap_volume_fraction(np.array([r_t]), d)[0]
        )

    def integrand_num(theta):
        return conditional_cap_fraction(theta, D_a, rc_a, d) * math.sin(theta) ** (d - 2)

    def integrand_den(theta):
        return math.sin(theta) ** (d - 2)

    # eps, not exactly 0: sin(theta)=0 at theta=0 divides by zero in
    # conditional_cap_fraction. quad's nodes are strictly interior anyway.
    eps = 1e-10
    num, _ = quad(integrand_num, eps, rt_a, limit=200)
    den, _ = quad(integrand_den, eps, rt_a, limit=200)
    if den == 0.0:
        raise ValueError(
            f"exact_intersection_fraction: both integrals underflowed to exactly "
            f"0.0 (D={D}, r_c={r_c}, r_t={r_t}, d={d}) -- small r_t at high d."
        )
    ratio = num / den

    if verify:
        mid = (eps + rt_a) / 2.0
        num_lo, _ = quad(integrand_num, eps, mid, limit=200)
        num_hi, _ = quad(integrand_num, mid, rt_a, limit=200)
        den_lo, _ = quad(integrand_den, eps, mid, limit=200)
        den_hi, _ = quad(integrand_den, mid, rt_a, limit=200)
        den_split = den_lo + den_hi
        if den_split != 0.0:
            ratio_split = (num_lo + num_hi) / den_split
            if abs(ratio - ratio_split) > 1e-6:
                import warnings
                warnings.warn(
                    f"exact_intersection_fraction: split vs. single ratio mismatch: "
                    f"primary={ratio}, split={ratio_split}",
                    stacklevel=2,
                )
    return ratio

#-------------------------------------------------------------------------------
# Second predictor: solid Euclidean ball intersection
#-------------------------------------------------------------------------------


def chord_distance_from_angular(r: np.ndarray) -> np.ndarray:
    """Euclidean chord distance for angular distance r (normalized
    units): chord = sqrt(2 - 2*cos(r*pi)) = 2*sin(r*pi/2)."""
    return 2.0 * np.sin(r * np.pi / 2.0)


def solid_ball_intersection_fraction(dist: float, R1: float, R2: float, n: int) -> float:
    """Fraction of solid ball 2's (radius R2) volume lying within solid
    ball 1 (radius R1), in ordinary Euclidean R^n.

    Raises TypeError if any input is not float64.
    """
    _require_float64(dist=dist, R1=R1, R2=R2)

    if dist >= R1 + R2:
        return 0.0
    if dist + R2 <= R1:
        return 1.0
    if dist + R1 <= R2:
        return (R1 / R2) ** n
    if dist < 1e-12:
        return 1.0 if R1 >= R2 else (R1 / R2) ** n

    d1 = (dist ** 2 + R1 ** 2 - R2 ** 2) / (2.0 * dist)
    d2 = dist - d1

    def J(R, di):
        from scipy.special import betainc
        x = np.clip(1.0 - (di / R) ** 2, 0.0, 1.0)
        a = (n + 1) / 2.0
        ib = betainc(a, 0.5, x)
        return 0.5 * ib if di >= 0 else 1.0 - 0.5 * ib

    return (R1 / R2) ** n * J(R1, d1) + J(R2, d2)


def solid_ball_intersection_fraction_literal(dist: float, R1: float, R2: float, n: int) -> float:
    """Same result as solid_ball_intersection_fraction, computed directly."""
    from scipy.special import gammaln, betainc

    _require_float64(dist=dist, R1=R1, R2=R2)

    if dist >= R1 + R2:
        return 0.0
    if dist + R2 <= R1:
        return 1.0
    if dist + R1 <= R2:
        return (R1 / R2) ** n
    if dist < 1e-12:
        return 1.0 if R1 >= R2 else (R1 / R2) ** n

    d1 = (dist ** 2 + R1 ** 2 - R2 ** 2) / (2.0 * dist)
    d2 = dist - d1

    def J(R, di):
        x = np.clip(1.0 - (di / R) ** 2, 0.0, 1.0)
        a = (n + 1) / 2.0
        ib = betainc(a, 0.5, x)
        return 0.5 * ib if di >= 0 else 1.0 - 0.5 * ib

    def V_n(R):
        log_vol = (n / 2.0) * np.log(np.pi) - gammaln(n / 2.0 + 1.0) + n * np.log(R)
        return np.exp(log_vol)

    V_int = V_n(R1) * J(R1, d1) + V_n(R2) * J(R2, d2)
    return V_int / V_n(R2)


def _prepare_combo_context(dataset: str, benchmark_name: str, K: int) -> dict:
    """Budget-independent setup shared across a whole budget sweep for one
    (benchmark, K). propose_full_pool is budget-independent so re-running it per budget
    is pure waste."""
    base, cache_q, test_q, bdir = dc.load_benchmark(dataset, benchmark_name)
    n_cache, n_test = len(cache_q), len(test_q)
    train_r_t, test_r_t = dc.load_gt_r_t(bdir, K, n_cache, n_test)
    n_base = len(base)
    d = base.shape[1]
    candidates = dc.propose_full_pool(base, cache_q, test_q, train_r_t, test_r_t, K)
    test_gt_indices = dc.load_gt_indices(bdir, K, n_test)
    gt_matrix = np.array(test_gt_indices, dtype=np.int32)
    return {
        'test_q': test_q, 'n_cache': n_cache, 'n_test': n_test,
        'test_r_t': test_r_t, 'n_base': n_base, 'd': d,
        'candidates': candidates, 'gt_matrix': gt_matrix,
    }


def compare_angular_vs_solid_ball(
    dataset: str, benchmark_name: str, K: int, budget_fraction: float,
    n_test_queries: int = None, max_candidates_per_query: int = None, seed: int = 0,
    context: dict = None,
) -> dict:
    """For test queries, compute both the angular and solid-ball
    intersection fraction against selected candidates, alongside the
    exact oracle_hit label. n_test_queries/max_candidates_per_query."""
    if context is None:
        context = _prepare_combo_context(dataset, benchmark_name, K)
    test_q, n_cache, n_test = context['test_q'], context['n_cache'], context['n_test']
    test_r_t, n_base, d = context['test_r_t'], context['n_base'], context['d']
    candidates, gt_matrix = context['candidates'], context['gt_matrix']

    if not candidates:
        return None
    budget_B = max(1, int(budget_fraction * n_base))
    selected, _traj = select_candidates(
        candidates, budget_B, strategy=dc.STRATEGY,
        n_train_queries=n_cache, n_test_queries=n_test, seed=dc.SEED,
    )
    if not selected:
        return None

    # use float64 to guard against precision loss and resulting errors
    test_q_hat = test_q.astype(np.float64)
    test_q_hat /= np.linalg.norm(test_q_hat, axis=1, keepdims=True)
    cand_centers = np.array([c.query for c in selected], dtype=np.float64)
    cand_centers_hat = cand_centers / np.linalg.norm(cand_centers, axis=1, keepdims=True)
    cand_rc = np.array([c.r_c for c in selected])
    n_selected = len(selected)

    rng = np.random.default_rng(seed)
    n_test_queries = n_test if n_test_queries is None else min(n_test_queries, n_test)
    test_idx_sample = rng.choice(n_test, size=n_test_queries, replace=False)
    n_cand_sample = n_selected if max_candidates_per_query is None else min(max_candidates_per_query, n_selected)
    cand_idx_sample = rng.choice(n_selected, size=n_cand_sample, replace=False)

    # batch the per-pair computations for speed, but still return per-pair
    # results for calibration plots
    sample_cand_hat = cand_centers_hat[cand_idx_sample]
    sample_test_hat = test_q_hat[test_idx_sample]
    sample_rc = cand_rc[cand_idx_sample]
    sample_rt = test_r_t[test_idx_sample]

    dots = np.clip(sample_cand_hat @ sample_test_hat.T, -1.0, 1.0)
    D_matrix = np.arccos(dots) / np.pi
    R1_all = chord_distance_from_angular(sample_rc)
    R2_all = chord_distance_from_angular(sample_rt)
    dist_chord_matrix = chord_distance_from_angular(D_matrix.ravel()).reshape(D_matrix.shape)
    dist_direct_matrix = np.sqrt(np.clip(2.0 - 2.0 * dots, 0.0, None))
    assert np.all(np.abs(dist_chord_matrix - dist_direct_matrix) < 1e-4)

    all_angular, all_solid_ball, all_oracle_hit, all_recall = [], [], [], []
    best_angular = np.empty(n_test_queries)
    best_solid_ball = np.empty(n_test_queries)
    union_covered = np.empty(n_test_queries, dtype=bool)
    union_recall = np.empty(n_test_queries)

    cached_mask_selected = np.zeros(n_base, dtype=bool)
    for c in selected:
        cached_mask_selected |= c.R_c

    for qi, ti in enumerate(test_idx_sample):
        R2 = float(R2_all[qi])
        angular_this_query = np.empty(n_cand_sample)
        solid_ball_this_query = np.empty(n_cand_sample)

        for j, ci in enumerate(cand_idx_sample):
            D = float(D_matrix[j, qi])
            r_c = float(sample_rc[j])
            r_t = float(sample_rt[qi])

            angular_this_query[j] = exact_intersection_fraction(D, r_c, r_t, d)
            assert -1e-9 <= angular_this_query[j] <= 1.0 + 1e-9, angular_this_query[j]

            R1 = float(R1_all[j])
            dist_direct = float(dist_direct_matrix[j, qi])
            solid_ball_this_query[j] = solid_ball_intersection_fraction(dist_direct, R1, R2, d)
            assert -1e-9 <= solid_ball_this_query[j] <= 1.0 + 1e-9, solid_ball_this_query[j]

            gt_hits = selected[ci].R_c[gt_matrix[ti]]
            oracle_hit_pair = bool(gt_hits.all())
            recall_pair = float(gt_hits.sum()) / K
            all_angular.append(angular_this_query[j])
            all_solid_ball.append(solid_ball_this_query[j])
            all_oracle_hit.append(oracle_hit_pair)
            all_recall.append(recall_pair)

        best_angular[qi] = angular_this_query.max()
        best_solid_ball[qi] = solid_ball_this_query.max()
        union_hits = cached_mask_selected[gt_matrix[ti]]
        union_covered[qi] = bool(union_hits.all())
        union_recall[qi] = float(union_hits.sum()) / K

    return {
        'angular_fraction': np.array(all_angular),
        'solid_ball_fraction': np.array(all_solid_ball),
        'oracle_hit': np.array(all_oracle_hit, dtype=bool),
        'recall': np.array(all_recall),
        'best_angular_fraction': best_angular,
        'best_solid_ball_fraction': best_solid_ball,
        'union_covered': union_covered,
        'union_recall': union_recall,
        'd': d,
        'K': K,
        'n_pairs': len(all_angular),
        'n_test_queries': n_test_queries,
        'n_selected': n_selected,
        'n_cand_sample': n_cand_sample,
    }

#-------------------------------------------------------------------------------


def bin_calibration(
    intersection_fraction: np.ndarray, oracle_hit: np.ndarray,
    n_bins: int = N_BINS, min_count: int = 1,
) -> dict:
    """Empirical reliability-diagram data: per-bin (center, mean oracle_hit,
    count) over n_bins buckets spanning [0, 1]. Bins with fewer than
    min_count pairs get NaN means (excluded from the curve)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    bin_idx = np.clip(np.digitize(intersection_fraction, edges[1:-1], right=True), 0, n_bins - 1)
    means = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        mask = bin_idx == b
        counts[b] = mask.sum()
        if counts[b] >= min_count:
            means[b] = oracle_hit[mask].mean()
    return {'bin_centers': centers, 'bin_means': means, 'bin_counts': counts}


def auc_score(intersection_fraction: np.ndarray, oracle_hit: np.ndarray) -> float:
    """Manual rank-based AUC (Mann-Whitney U statistic). AUC =
    P(a randomly chosen hit's score > a randomly chosen miss's score),
    with ties contributing 0.5."""
    oracle_hit = oracle_hit.astype(bool)
    n_pos = oracle_hit.sum()
    n_neg = len(oracle_hit) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    ranks = _rankdata_avg(intersection_fraction)
    sum_ranks_pos = ranks[oracle_hit].sum()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _rankdata_avg(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-indexed), ties get the mean rank."""
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def fit_binomial_glm(x: np.ndarray, successes: np.ndarray, n_trials) -> dict:
    """Logistic-link binomial GLM: P(success) = sigmoid(b0 + b1*x), fit
    via statsmodels."""
    n_trials = np.broadcast_to(n_trials, x.shape).astype(float)
    endog = np.column_stack([successes, n_trials - successes])
    exog = sm.add_constant(x)
    fit = sm.GLM(endog, exog, family=sm.families.Binomial()).fit()
    null_fit = sm.GLM(endog, np.ones((len(x), 1)), family=sm.families.Binomial()).fit()
    return {
        'intercept': float(fit.params[0]),
        'slope': float(fit.params[1]),
        'slope_pvalue': float(fit.pvalues[1]),
        'pseudo_r2': float(1.0 - fit.llf / null_fit.llf),
    }


def mannwhitney_test(x: np.ndarray, oracle_hit: np.ndarray) -> dict:
    """Mann-Whitney U test: is x's distribution different between the
    hit and miss groups?."""
    oracle_hit = oracle_hit.astype(bool)
    hit_x, miss_x = x[oracle_hit], x[~oracle_hit]
    if len(hit_x) == 0 or len(miss_x) == 0:
        return {'auc': float('nan'), 'pvalue': float('nan')}
    stat, pvalue = stats.mannwhitneyu(hit_x, miss_x, alternative='two-sided')
    return {'auc': float(stat / (len(hit_x) * len(miss_x))), 'pvalue': float(pvalue)}


def point_biserial_test(x: np.ndarray, oracle_hit: np.ndarray) -> dict:
    """Point-biserial correlation (= Pearson r between continuous x and
    binary y) with its significance test."""
    oracle_hit = oracle_hit.astype(bool)
    corr, pvalue = stats.pointbiserialr(oracle_hit, x)
    return {'correlation': float(corr), 'pvalue': float(pvalue)}

#-------------------------------------------------------------------------------
# Dual-predictor comparison: angular vs. solid-ball, same pairs/labels as
# compare_angular_vs_solid_ball, swept across the full combo grid.
#-------------------------------------------------------------------------------


def _actual_budgets(data_by_bm_k: dict, k_values: list, exclude_full: bool = False) -> list:
    """Sorted union of budget fractions actually present in the data to avoid budgets
    that were never swept. Probably should be deprecated in favor of better code."""
    budgets = set()
    for bm_data in data_by_bm_k.values():
        for K in k_values:
            per_budget = bm_data.get(K)
            if per_budget:
                budgets.update(per_budget.keys())
    if exclude_full:
        budgets.discard(1.0)
    return sorted(budgets)


def run_dual_predictor_sweep(
    dataset: str, k_values: list = None, budget_fractions: list = None,
    n_test_queries: int = None, max_candidates_per_query: int = None,
    benchmarks: list = None,
) -> dict:
    """Runs compare_angular_vs_solid_ball across the full (benchmark, K,
    budget) grid, returning data_by_bm_k[benchmark][K][budget] = result
    dict or None."""
    k_values = k_values if k_values is not None else K_VALUES
    budget_fractions = budget_fractions if budget_fractions is not None else BUDGET_FRACTIONS
    cfg = dc.DATASET_CONFIG[dataset]
    benchmarks = benchmarks or cfg['benchmarks']

    data_by_bm_k = {}
    for benchmark_name in benchmarks:
        bdir = cfg['base_dir'] / benchmark_name
        if not (bdir / 'cache_queries.npy').exists():
            print(f"  {benchmark_name}: not prepared, skipping")
            continue
        print(f"  {benchmark_name}")
        data_by_bm_k[benchmark_name] = {}
        for K in k_values:
            data_by_bm_k[benchmark_name][K] = {}
            context = _prepare_combo_context(dataset, benchmark_name, K)
            for frac in budget_fractions:
                result = compare_angular_vs_solid_ball(
                    dataset, benchmark_name, K, frac,
                    n_test_queries=n_test_queries,
                    max_candidates_per_query=max_candidates_per_query,
                    context=context,
                )
                data_by_bm_k[benchmark_name][K][frac] = result
                if result is not None:
                    auc_ang = auc_score(result['angular_fraction'], result['oracle_hit'])
                    auc_sb = auc_score(result['solid_ball_fraction'], result['oracle_hit'])
                    print(f"    K={K} budget={frac*100:.0f}%: n_selected={result['n_selected']} "
                          f"n_pairs={result['n_pairs']} AUC_angular={auc_ang:.3f} "
                          f"AUC_solid_ball={auc_sb:.3f}")
    return data_by_bm_k

#-------------------------------------------------------------------------------

_SCALAR_RESULT_KEYS = ('d', 'K', 'n_pairs', 'n_test_queries', 'n_selected', 'n_cand_sample')


def _combo_result_path(run_dir: Path, dataset: str, benchmark_name: str, K: int, budget_fraction: float) -> Path:
    return run_dir / dataset / 'raw' / f'{benchmark_name}_K{K}_budget{budget_fraction*100:.0f}pct.npz'


def save_sweep(dataset: str, data_by_bm_k: dict, run_dir: Path) -> None:
    """Writes one .npz per (benchmark, K, budget) cell for persistence."""
    for benchmark_name, per_k in data_by_bm_k.items():
        for K, per_budget in per_k.items():
            for frac, result in per_budget.items():
                if result is None:
                    continue
                path = _combo_result_path(run_dir, dataset, benchmark_name, K, frac)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **result)


def load_combo_result(path: Path) -> Optional[dict]:
    """Reads one cell's .npz back into the same dict shape
    compare_angular_vs_solid_ball returns."""
    if not path.exists():
        return None
    with np.load(path) as data:
        result = {key: data[key] for key in data.files}
    for key in _SCALAR_RESULT_KEYS:
        if key in result:
            result[key] = int(result[key])
    return result


def load_sweep(
    dataset: str, run_dir: Path, k_values: list, budget_fractions: list,
    benchmarks: list = None,
) -> dict:
    """Reconstructs data_by_bm_k from a prior run_dual_predictor_sweep's
    persisted .npz files."""
    cfg = dc.DATASET_CONFIG[dataset]
    benchmarks = benchmarks or cfg['benchmarks']
    data_by_bm_k = {}
    for benchmark_name in benchmarks:
        data_by_bm_k[benchmark_name] = {}
        for K in k_values:
            data_by_bm_k[benchmark_name][K] = {}
            for frac in budget_fractions:
                path = _combo_result_path(run_dir, dataset, benchmark_name, K, frac)
                data_by_bm_k[benchmark_name][K][frac] = load_combo_result(path)
    return data_by_bm_k

#-------------------------------------------------------------------------------

DUAL_RELIABILITY_N_BINS = 12
DUAL_RELIABILITY_MIN_BIN_COUNT = 20

def _draw_dual_reliability(
    dataset: str, data_by_bm_k: dict, benchmarks: list, k_values: list,
    budget_fractions: list, title_suffix: str, out_path: Path,
) -> None:
    """Shared body for plot_dataset_dual_reliability: draws one
    benchmark x K grid for whatever budget_fractions list it's given."""
    n_rows, n_cols = len(benchmarks), len(k_values)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 4.2 * n_rows), squeeze=False)
    series = [('angular_fraction', 'angular', 'tab:green', 'o', '-'),
              ('solid_ball_fraction', 'solid-ball', 'tab:orange', '^', '--')]

    for r, bm in enumerate(benchmarks):
        train_n, _ = parse_benchmark_sizes(bm)
        for c, K in enumerate(k_values):
            ax = axes[r][c]
            per_budget = data_by_bm_k[bm].get(K)
            if per_budget is None:
                ax.axis('off')
                continue
            pooled = {key: [] for key, *_ in series}
            pooled_hit = []
            for frac in budget_fractions:
                d = per_budget.get(frac)
                if d is None:
                    continue
                for key, *_ in series:
                    pooled[key].append(d[key])
                pooled_hit.append(d['oracle_hit'])
            if not pooled_hit:
                ax.axis('off')
                continue
            oracle_hit = np.concatenate(pooled_hit)

            for key, label, color, marker, linestyle in series:
                fraction = np.concatenate(pooled[key])
                binned = bin_calibration(
                    fraction, oracle_hit,
                    n_bins=DUAL_RELIABILITY_N_BINS, min_count=DUAL_RELIABILITY_MIN_BIN_COUNT,
                )
                valid = ~np.isnan(binned['bin_means'])
                ax.plot(binned['bin_centers'][valid], binned['bin_means'][valid],
                         marker=marker, markersize=4, linestyle=linestyle, color=color,
                         label=label)

                if oracle_hit.any() and not oracle_hit.all():
                    glm = fit_binomial_glm(fraction, oracle_hit.astype(float), 1)
                    xs = np.linspace(0.0, 1.0, 100)
                    fitted = 1.0 / (1.0 + np.exp(-(glm['intercept'] + glm['slope'] * xs)))
                    ax.plot(xs, fitted, linestyle=':', linewidth=1, color=color, alpha=0.6,
                             label=f'{label} logistic fit')
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlim(-0.02, 1.02)

            if r == 0:
                ax.set_title(f'$K={K}$', fontsize=10)
            if c == 0:
                ax.set_ylabel(f'{train_n} Train Queries\nP(oracle hit)')
            if r == n_rows - 1:
                ax.set_xlabel('intersection fraction')
            ax.legend(fontsize=6, loc='best')

    fig.suptitle(
        f'{dataset.upper()}: Oracle Hit Probability vs. Intersection Fraction '
        f'({title_suffix})', fontsize=13,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")


def plot_dataset_dual_reliability(
    dataset: str, data_by_bm_k: dict, k_values: list = None, budget_fractions: list = None,
    per_budget: bool = True,
) -> None:
    """Plots the dual reliability of the dataset."""
    k_values = k_values if k_values is not None else K_VALUES
    benchmarks = [bm for bm in data_by_bm_k if any(data_by_bm_k[bm].get(k) for k in k_values)]
    if not benchmarks:
        return
    benchmarks = sorted(benchmarks, key=lambda bm: parse_benchmark_sizes(bm)[0])
    pooled_budgets = budget_fractions if budget_fractions is not None else _actual_budgets(data_by_bm_k, k_values, exclude_full=True)

    outdir = _run_dir() / dataset
    _draw_dual_reliability(
        dataset, data_by_bm_k, benchmarks, k_values, pooled_budgets,
        'Pooled Budgets', outdir / 'dual_calibration_vs_intersection_fraction.png',
    )
    if per_budget:
        for frac in _actual_budgets(data_by_bm_k, k_values):
            _draw_dual_reliability(
                dataset, data_by_bm_k, benchmarks, k_values, [frac],
                f'Budget = {frac*100:.0f}%',
                outdir / f'dual_calibration_vs_intersection_fraction_budget{frac*100:.0f}pct.png',
            )


def _draw_recall_reliability(
    dataset: str, data_by_bm_k: dict, benchmarks: list, k_values: list,
    budget_fractions: list, title_suffix: str, out_path: Path,
) -> None:
    """Plots the recall comparison between angular and solid-ball methods."""
    n_rows, n_cols = len(benchmarks), len(k_values)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 4.2 * n_rows), squeeze=False)
    series = [('angular_fraction', 'angular', 'tab:green', 'o', '-'),
              ('solid_ball_fraction', 'solid-ball', 'tab:orange', '^', '--')]

    for r, bm in enumerate(benchmarks):
        train_n, _ = parse_benchmark_sizes(bm)
        for c, K in enumerate(k_values):
            ax = axes[r][c]
            per_budget = data_by_bm_k[bm].get(K)
            if per_budget is None:
                ax.axis('off')
                continue
            pooled = {key: [] for key, *_ in series}
            pooled_recall = []
            for frac in budget_fractions:
                d = per_budget.get(frac)
                if d is None:
                    continue
                for key, *_ in series:
                    pooled[key].append(d[key])
                pooled_recall.append(d['recall'])
            if not pooled_recall:
                ax.axis('off')
                continue
            recall = np.concatenate(pooled_recall)

            for key, label, color, marker, linestyle in series:
                fraction = np.concatenate(pooled[key])
                binned = bin_calibration(
                    fraction, recall,
                    n_bins=DUAL_RELIABILITY_N_BINS, min_count=DUAL_RELIABILITY_MIN_BIN_COUNT,
                )
                valid = ~np.isnan(binned['bin_means'])
                ax.plot(binned['bin_centers'][valid], binned['bin_means'][valid],
                         marker=marker, markersize=4, linestyle=linestyle, color=color,
                         label=label)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlim(-0.02, 1.02)

            if r == 0:
                ax.set_title(f'$K={K}$', fontsize=10)
            if c == 0:
                ax.set_ylabel(f'{train_n} Train Queries\nmean recall (found/K)')
            if r == n_rows - 1:
                ax.set_xlabel('intersection fraction')
            ax.legend(fontsize=6, loc='best')

    fig.suptitle(
        f'{dataset.upper()}: Recall vs. Intersection Fraction '
        f'({title_suffix})', fontsize=13,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")


def plot_dataset_recall_reliability(
    dataset: str, data_by_bm_k: dict, k_values: list = None, budget_fractions: list = None,
    per_budget: bool = True,
) -> None:
    """Continuous-recall counterpart to plot_dataset_dual_reliability."""
    k_values = [k for k in (k_values if k_values is not None else K_VALUES) if k > 1]
    if not k_values:
        return
    benchmarks = [bm for bm in data_by_bm_k if any(data_by_bm_k[bm].get(k) for k in k_values)]
    if not benchmarks:
        return
    benchmarks = sorted(benchmarks, key=lambda bm: parse_benchmark_sizes(bm)[0])
    pooled_budgets = budget_fractions if budget_fractions is not None else _actual_budgets(data_by_bm_k, k_values, exclude_full=True)

    outdir = _run_dir() / dataset
    _draw_recall_reliability(
        dataset, data_by_bm_k, benchmarks, k_values, pooled_budgets,
        'Pooled Budgets', outdir / 'recall_vs_intersection_fraction.png',
    )
    if per_budget:
        for frac in _actual_budgets(data_by_bm_k, k_values):
            _draw_recall_reliability(
                dataset, data_by_bm_k, benchmarks, k_values, [frac],
                f'Budget = {frac*100:.0f}%',
                outdir / f'recall_vs_intersection_fraction_budget{frac*100:.0f}pct.png',
            )


_BREAK_FALLBACK = (0.1, 0.9)
_BREAK_MIN_MARGIN = 0.05
_BREAK_MIN_GAP = 0.1
_BREAK_PAD = 0.03
_BREAK_MARK = 0.02


def _find_axis_break(x: np.ndarray) -> tuple:
    """Largest gap in x's sorted unique values, as (left_edge, right_edge), 
    the widest x-range with no data points at all, used to split a
    panel into two zoomed sub-axes. Falls back to a fixed 0.1/0.9 split
    if there's no data, only one distinct value, or no gap wide enough to be
    worth breaking on."""
    if len(x) == 0:
        return _BREAK_FALLBACK
    xs = np.unique(x)
    if len(xs) < 2:
        return _BREAK_FALLBACK
    gaps = np.diff(xs)
    i = int(np.argmax(gaps))
    left_edge, right_edge = float(xs[i]), float(xs[i + 1])
    if (gaps[i] < _BREAK_MIN_GAP or left_edge < _BREAK_MIN_MARGIN
            or right_edge > 1.0 - _BREAK_MIN_MARGIN):
        return _BREAK_FALLBACK
    return left_edge, right_edge


def _draw_break_marks(ax_left, ax_right) -> None:
    d = _BREAK_MARK
    kwargs = dict(transform=ax_left.transAxes, color='k', clip_on=False, linewidth=1)
    ax_left.plot((1 - d, 1 + d), (-d, d), **kwargs)
    ax_left.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs['transform'] = ax_right.transAxes
    ax_right.plot((-d, d), (-d, d), **kwargs)
    ax_right.plot((-d, d), (1 - d, 1 + d), **kwargs)


def plot_dataset_dual_best_candidate_scatter(
    dataset: str, data_by_bm_k: dict, budget_fraction: float, k_values: list = None,
) -> None:
    """Angular and solid-ball best-candidate fractions overlaid in the
    same panel per K, distinguished by color."""
    k_values = k_values if k_values is not None else K_VALUES
    benchmarks = [bm for bm in data_by_bm_k if any(data_by_bm_k[bm].get(k) for k in k_values)]
    if not benchmarks:
        return
    benchmarks = sorted(benchmarks, key=lambda bm: parse_benchmark_sizes(bm)[0])

    if budget_fraction not in _actual_budgets(data_by_bm_k, k_values):
        return

    n_rows, n_cols = len(benchmarks), len(k_values)
    fig, axes = plt.subplots(
        n_rows, n_cols * 2, figsize=(2.4 * n_cols * 2, 3.8 * n_rows), squeeze=False,
        gridspec_kw={'width_ratios': [1, 1] * n_cols, 'wspace': 0.08},
    )
    jitter_rng = np.random.default_rng(0)

    series = [('best_angular_fraction', 'angular', 'tab:green'),
              ('best_solid_ball_fraction', 'solid-ball', 'tab:orange')]

    for r, bm in enumerate(benchmarks):
        train_n, _ = parse_benchmark_sizes(bm)
        for c, K in enumerate(k_values):
            ax_left, ax_right = axes[r][2 * c], axes[r][2 * c + 1]
            ax_left.sharey(ax_right)
            per_budget = data_by_bm_k[bm].get(K)
            d = per_budget.get(budget_fraction) if per_budget is not None else None

            if d is not None:
                y = d['union_covered'].astype(np.float64)
                x_all = np.concatenate([d[field] for field, *_ in series])
                left_edge, right_edge = _find_axis_break(x_all)
                for field, label, color in series:
                    x = d[field]
                    assert np.all((x >= -1e-9) & (x <= 1.0 + 1e-9)), x
                    y_jitter = jitter_rng.uniform(-0.35, 0.35, size=len(y))
                    for ax in (ax_left, ax_right):
                        ax.scatter(x, y + y_jitter, s=20, alpha=0.6, color=color,
                                   edgecolors='none', label=label)
            else:
                left_edge, right_edge = _BREAK_FALLBACK

            ax_left.set_xlim(-0.02, left_edge + _BREAK_PAD)
            ax_right.set_xlim(right_edge - _BREAK_PAD, 1.02)
            ax_left.spines['right'].set_visible(False)
            ax_right.spines['left'].set_visible(False)
            ax_right.tick_params(left=False, labelleft=False)
            _draw_break_marks(ax_left, ax_right)

            ax_left.set_ylim(-0.55, 1.55)
            ax_left.set_yticks([0, 1])
            ax_left.set_yticklabels(['miss', 'hit'])
            if r == 0:
                ax_left.set_title(f'$K={K}$', fontsize=10, loc='left', x=0.0)
            if c == 0:
                ax_left.set_ylabel(f'{train_n} Train Queries\nunion coverage')
            if r == n_rows - 1:
                ax_left.set_xlabel('best-candidate\nintersection fraction', fontsize=8)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if not handles:
        for row in axes:
            for ax in row:
                if ax.has_data():
                    handles, labels = ax.get_legend_handles_labels()
                    break
            if handles:
                break
    if handles:
        fig.legend(handles, labels, loc='upper right', markerscale=1.5, fontsize=10)

    fig.suptitle(
        f'{dataset.upper()}: Union Coverage vs. Best-Candidate Fraction '
        f'(budget={budget_fraction*100:.0f}%)', fontsize=13,
    )
    plt.tight_layout()

    outdir = _run_dir() / dataset
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f'dual_best_candidate_scatter_budget{budget_fraction*100:.0f}pct.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")

#-------------------------------------------------------------------------------

CONFUSION_THRESHOLD = 0.8


def _draw_confusion_matrices(
    dataset: str, data_by_bm_k: dict, benchmarks: list, k_values: list,
    budget_fractions: list, title_suffix: str, out_path: Path,
    threshold: float = CONFUSION_THRESHOLD,
) -> None:
    """Standard 2x2 confusion matrix per (benchmark, K, method): predicted
    hit = best_{method}_fraction >= threshold, actual = union_covered."""
    series = [('best_angular_fraction', 'angular'), ('best_solid_ball_fraction', 'solid-ball')]
    n_rows, n_cols = len(benchmarks), len(k_values)
    fig, axes = plt.subplots(
        n_rows, n_cols * 2, figsize=(2.5 * n_cols * 2, 2.9 * n_rows), squeeze=False,
        gridspec_kw={'width_ratios': [1, 1] * n_cols, 'wspace': 0.7, 'hspace': 0.9},
    )

    for r, bm in enumerate(benchmarks):
        train_n, _ = parse_benchmark_sizes(bm)
        for c, K in enumerate(k_values):
            per_budget = data_by_bm_k[bm].get(K)
            pooled_best = {key: [] for key, _ in series}
            pooled_union = []
            if per_budget is not None:
                for frac in budget_fractions:
                    d = per_budget.get(frac)
                    if d is None:
                        continue
                    for key, _ in series:
                        pooled_best[key].append(d[key])
                    pooled_union.append(d['union_covered'])

            for j, (key, label) in enumerate(series):
                ax = axes[r][2 * c + j]
                if not pooled_union:
                    ax.axis('off')
                    continue
                y_true = np.concatenate(pooled_union).astype(bool)
                y_pred = np.concatenate(pooled_best[key]) >= threshold
                cm = confusion_matrix(y_true, y_pred, labels=[False, True])
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                sns.heatmap(
                    cm, annot=True, fmt='d', cmap='YlOrRd', cbar=False, ax=ax,
                    xticklabels=['miss', 'hit'], yticklabels=['miss', 'hit'],
                    annot_kws={'fontsize': 7},
                )
                ax.set_xlabel('predicted', fontsize=7)
                ax.set_ylabel('actual' if j == 0 else '', fontsize=7)
                ax.tick_params(labelsize=6)
                title = f'{label}\nPrec={prec:.2f} Rec={rec:.2f} F1={f1:.2f}'
                if r == 0:
                    title = f'$K={K}$\n' + title
                ax.set_title(title, fontsize=7)
            if c == 0:
                axes[r][0].set_ylabel(f'{train_n} Train Queries\nactual', fontsize=7)

    fig.suptitle(
        f'{dataset.upper()}: Confusion Matrix, Best-Candidate Fraction >= {threshold} '
        f'vs. Union Coverage ({title_suffix})', fontsize=12, y=1.0 + 0.35 / n_rows,
    )
    fig.subplots_adjust(top=1.0 - 0.05 / n_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")


def plot_dataset_confusion_matrices(
    dataset: str, data_by_bm_k: dict, k_values: list = None,
    threshold: float = CONFUSION_THRESHOLD,
) -> None:
    """Writes one confusion-matrix grid per individual budget (including
    100%, matching plot_dataset_dual_best_candidate_scatter's per-budget
    convention) plus one pooled version (excluding 100%)."""
    k_values = k_values if k_values is not None else K_VALUES
    benchmarks = [bm for bm in data_by_bm_k if any(data_by_bm_k[bm].get(k) for k in k_values)]
    if not benchmarks:
        return
    benchmarks = sorted(benchmarks, key=lambda bm: parse_benchmark_sizes(bm)[0])
    outdir = _run_dir() / dataset

    for frac in _actual_budgets(data_by_bm_k, k_values):
        _draw_confusion_matrices(
            dataset, data_by_bm_k, benchmarks, k_values, [frac],
            f'Budget = {frac*100:.0f}%', outdir / f'confusion_matrix_budget{frac*100:.0f}pct.png',
            threshold=threshold,
        )
    pooled_budgets = _actual_budgets(data_by_bm_k, k_values, exclude_full=True)
    _draw_confusion_matrices(
        dataset, data_by_bm_k, benchmarks, k_values, pooled_budgets,
        'Pooled Budgets', outdir / 'confusion_matrix_pooled.png', threshold=threshold,
    )

def plot_dataset_angular_vs_solid_ball_scatter(
    dataset: str, data_by_bm_k: dict, k_values: list = None, budget_fractions: list = None,
) -> None:
    """Scatter of angular_fraction vs. solid_ball_fraction per pair, pooled
    across budgets, colored by oracle_hit, with a y=x reference line."""
    k_values = k_values if k_values is not None else K_VALUES
    benchmarks = [bm for bm in data_by_bm_k if any(data_by_bm_k[bm].get(k) for k in k_values)]
    if not benchmarks:
        return
    benchmarks = sorted(benchmarks, key=lambda bm: parse_benchmark_sizes(bm)[0])
    budget_fractions = budget_fractions if budget_fractions is not None else _actual_budgets(data_by_bm_k, k_values, exclude_full=True)

    n_rows, n_cols = len(benchmarks), len(k_values)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.0 * n_rows), squeeze=False)

    for r, bm in enumerate(benchmarks):
        train_n, _ = parse_benchmark_sizes(bm)
        for c, K in enumerate(k_values):
            ax = axes[r][c]
            per_budget = data_by_bm_k[bm].get(K)
            if per_budget is None:
                ax.axis('off')
                continue
            ang_all, sb_all, hit_all = [], [], []
            for frac in budget_fractions:
                d = per_budget.get(frac)
                if d is None:
                    continue
                ang_all.append(d['angular_fraction'])
                sb_all.append(d['solid_ball_fraction'])
                hit_all.append(d['oracle_hit'])
            if not ang_all:
                ax.axis('off')
                continue
            ang_all = np.concatenate(ang_all)
            sb_all = np.concatenate(sb_all)
            hit_all = np.concatenate(hit_all)

            ax.plot([0, 1], [0, 1], linestyle=':', color='gray', linewidth=1, alpha=0.6)
            ax.scatter(ang_all[~hit_all], sb_all[~hit_all], s=4, alpha=0.25,
                       color='tab:red', edgecolors='none', label='miss')
            ax.scatter(ang_all[hit_all], sb_all[hit_all], s=4, alpha=0.25,
                       color='tab:blue', edgecolors='none', label='hit')
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)

            if r == 0:
                ax.set_title(f'$K={K}$', fontsize=10)
            if c == 0:
                ax.set_ylabel(f'{train_n} Train Queries\nsolid-ball fraction')
            if r == n_rows - 1:
                ax.set_xlabel('angular fraction')

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, -0.04), markerscale=3)
    fig.suptitle(
        f'{dataset.upper()}: Angular vs. Solid-Ball Intersection Fraction, per Pair '
        f'(pooled across budgets)', fontsize=13,
    )
    plt.tight_layout()

    outdir = _run_dir() / dataset
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / 'angular_vs_solid_ball_scatter.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")


def plot_auc_comparison_grid(
    dataset: str, data_by_bm_k: dict, k_values: list = None, budget_fractions: list = None,
) -> None:
    """Angular AUC vs. solid-ball AUC per budget, as connected point-pairs
    across the full benchmark/K grid."""
    k_values = k_values if k_values is not None else K_VALUES
    benchmarks = [bm for bm in data_by_bm_k if any(data_by_bm_k[bm].get(k) for k in k_values)]
    if not benchmarks:
        return
    benchmarks = sorted(benchmarks, key=lambda bm: parse_benchmark_sizes(bm)[0])
    budget_fractions = budget_fractions if budget_fractions is not None else _actual_budgets(data_by_bm_k, k_values, exclude_full=True)

    n_rows, n_cols = len(benchmarks), len(k_values)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.8 * n_rows), squeeze=False)

    for r, bm in enumerate(benchmarks):
        train_n, _ = parse_benchmark_sizes(bm)
        for c, K in enumerate(k_values):
            ax = axes[r][c]
            per_budget = data_by_bm_k[bm].get(K)
            if per_budget is None:
                ax.axis('off')
                continue
            xs = list(range(len(budget_fractions)))
            auc_ang_list, auc_sb_list = [], []
            for frac in budget_fractions:
                d = per_budget.get(frac)
                if d is None:
                    auc_ang_list.append(np.nan)
                    auc_sb_list.append(np.nan)
                    continue
                auc_ang_list.append(auc_score(d['angular_fraction'], d['oracle_hit']))
                auc_sb_list.append(auc_score(d['solid_ball_fraction'], d['oracle_hit']))
            ax.plot(xs, auc_ang_list, marker='o', markersize=4, color='tab:green',
                     label='angular')
            ax.plot(xs, auc_sb_list, marker='^', markersize=4, color='tab:orange',
                     label='solid-ball')
            ax.axhline(0.5, linestyle=':', color='gray', linewidth=0.8, alpha=0.5)
            ax.set_xticks(xs)
            ax.set_xticklabels([f'{f*100:.0f}%' for f in budget_fractions], fontsize=7, rotation=45)
            ax.set_ylim(0.45, 1.02)

            if r == 0:
                ax.set_title(f'$K={K}$', fontsize=10)
            if c == 0:
                ax.set_ylabel(f'{train_n} Train Queries\nAUC')
            if r == n_rows - 1:
                ax.set_xlabel('budget')

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        f'{dataset.upper()}: AUC Comparison', fontsize=13,
    )
    plt.tight_layout()

    outdir = _run_dir() / dataset
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / 'auc_comparison_grid.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {out_path}")

#-------------------------------------------------------------------------------


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['sift', 'esci'], default=None,
                         help='Default: both')
    parser.add_argument('--k-values', type=int, nargs='+', default=None, dest='k_values')
    parser.add_argument('--budget-fractions', type=float, nargs='+', default=None,
                         dest='budget_fractions')
    parser.add_argument('--n-test-queries', type=int, default=None, dest='n_test_queries',
                         help='Default: every test query (uncapped)')
    parser.add_argument('--max-candidates-per-query', type=int, default=None,
                         dest='max_candidates_per_query',
                         help='Default: every selected candidate (uncapped)')
    parser.add_argument('--benchmarks', nargs='+', default=None)
    parser.add_argument('--from-run', type=str, default=None, dest='from_run',
                         help='Replot from a prior run\'s persisted raw data '
                              '(run_dir passed to save_sweep) instead of re-sweeping.')
    return parser.parse_args()


def _compute_cell_stats(per_budget: dict, K: int) -> dict:
    """Pooled statistics (Mann-Whitney, point-biserial, logistic/binomial
    GLM) for one (benchmark, K) cell, for the angular predictor, pooled
    across all budgets present except budget=100%."""
    pooled_frac, pooled_hit, pooled_recall = [], [], []
    for frac, d in per_budget.items():
        if frac == 1.0 or d is None:
            continue
        pooled_frac.append(d['angular_fraction'])
        pooled_hit.append(d['oracle_hit'])
        pooled_recall.append(d['recall'])
    if not pooled_hit:
        return {}
    x = np.concatenate(pooled_frac)
    hit = np.concatenate(pooled_hit)
    recall = np.concatenate(pooled_recall)

    out = {
        'mannwhitney': mannwhitney_test(x, hit),
        'point_biserial': point_biserial_test(x, hit),
    }
    if hit.any() and not hit.all():
        out['logistic_glm'] = fit_binomial_glm(x, hit.astype(float), 1)
    if K > 1:
        successes = np.round(recall * K)
        if successes.min() < K or successes.max() > 0:
            out['recall_glm'] = fit_binomial_glm(x, successes, K)
    return out


def main() -> None:
    args = _parse_args()
    k_values = args.k_values or K_VALUES
    budget_fractions = args.budget_fractions or BUDGET_FRACTIONS
    datasets = [args.dataset] if args.dataset else list(dc.DATASET_CONFIG.keys())

    summary = {}
    for dataset in datasets:
        print(f"\n=== {dataset.upper()} ===")
        if args.from_run:
            data_by_bm_k = load_sweep(
                dataset, Path(args.from_run), k_values=k_values,
                budget_fractions=budget_fractions, benchmarks=args.benchmarks,
            )
        else:
            data_by_bm_k = run_dual_predictor_sweep(
                dataset, k_values=k_values, budget_fractions=budget_fractions,
                n_test_queries=args.n_test_queries,
                max_candidates_per_query=args.max_candidates_per_query,
                benchmarks=args.benchmarks,
            )
            save_sweep(dataset, data_by_bm_k, _run_dir())

        summary[dataset] = {}
        for benchmark_name, per_k in data_by_bm_k.items():
            summary[dataset][benchmark_name] = {}
            for K, per_budget in per_k.items():
                for frac, result in per_budget.items():
                    if result is None:
                        continue
                    auc_ang = auc_score(result['angular_fraction'], result['oracle_hit'])
                    auc_sb = auc_score(result['solid_ball_fraction'], result['oracle_hit'])
                    summary[dataset][benchmark_name].setdefault(f'K={K}', {})[f'budget={frac*100:.0f}%'] = {
                        'auc_angular': auc_ang, 'auc_solid_ball': auc_sb,
                        'n_pairs': result['n_pairs'], 'n_selected': result['n_selected'],
                    }
                summary[dataset][benchmark_name].setdefault(f'K={K}', {})['pooled_stats_excl_100pct'] = (
                    _compute_cell_stats(per_budget, K)
                )

        plot_dataset_dual_reliability(dataset, data_by_bm_k, k_values=k_values, budget_fractions=budget_fractions)
        plot_dataset_recall_reliability(dataset, data_by_bm_k, k_values=k_values, budget_fractions=budget_fractions)
        for frac in _actual_budgets(data_by_bm_k, k_values):
            plot_dataset_dual_best_candidate_scatter(dataset, data_by_bm_k, frac, k_values=k_values)
        plot_dataset_confusion_matrices(dataset, data_by_bm_k, k_values=k_values)

    _run_dir().mkdir(parents=True, exist_ok=True)
    with open(_run_dir() / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. Summary: {_run_dir() / 'summary.json'}")


if __name__ == '__main__':
    main()
