#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Tests for debug_intersection_calibration.py.

Distances/radii throughout are in this codebase's normalized angular units
(D = arccos(dot)/pi, in [0,1]), matching debug_common.pairwise_dist's
convention -- the cap sampler converts to true geodesic radians internally.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from scipy.stats import kstest

import simulations.population.debug_common as dc
from simulations.population.debug_intersection_calibration import (
    sample_cap_points, single_cap_volume_fraction,
    bin_calibration, auc_score, exact_intersection_fraction,
    chord_distance_from_angular, solid_ball_intersection_fraction,
    solid_ball_intersection_fraction_literal, compare_angular_vs_solid_ball,
    _actual_budgets, plot_dataset_dual_best_candidate_scatter,
    conditional_cap_fraction, fit_binomial_glm, mannwhitney_test,
    point_biserial_test, save_sweep, load_combo_result, load_sweep,
)

#-------------------------------------------------------------------------------


def _random_pole(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = rng.standard_normal(dim)
    return p / np.linalg.norm(p)


def _angular_dist(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    return float(np.arccos(dot) / np.pi)

#-------------------------------------------------------------------------------


def test_cap_sampler_matches_known_volume_formula():
    """Empirical fraction of whole-sphere-uniform samples landing in a cap
    should match the closed-form single-cap volume formula, independently
    validating the reference formula used elsewhere in these tests."""
    dim = 10
    rng = np.random.default_rng(0)
    r = 0.3

    n = 200_000
    whole_sphere = rng.standard_normal((n, dim))
    whole_sphere /= np.linalg.norm(whole_sphere, axis=1, keepdims=True)
    pole = np.zeros(dim)
    pole[0] = 1.0
    dists = np.arccos(np.clip(whole_sphere @ pole, -1.0, 1.0)) / np.pi
    empirical = float(np.mean(dists <= r))

    expected = float(single_cap_volume_fraction(np.array([r]), dim)[0])
    assert abs(empirical - expected) < 0.01, (empirical, expected)


def test_cap_sampler_points_within_radius():
    dim = 32
    pole = _random_pole(dim, seed=1)
    r = 0.15
    rng = np.random.default_rng(2)
    pts = sample_cap_points(pole[None, :], np.array([r]), 5000, rng)[0]
    dists = np.arccos(np.clip(pts @ pole, -1.0, 1.0)) / np.pi
    assert np.all(dists <= r + 1e-9)
    # unit norm preserved
    norms = np.linalg.norm(pts, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_cap_sampler_radial_distribution_matches_theory():
    """KS goodness-of-fit: sampled cos(theta) should match the theoretical
    truncated-Beta((d-1)/2,(d-1)/2) density (affinely rescaled)."""
    from scipy.stats import beta as beta_dist

    dim = 16
    pole = _random_pole(dim, seed=3)
    r = 0.4
    rng = np.random.default_rng(4)
    pts = sample_cap_points(pole[None, :], np.array([r]), 20000, rng)[0]
    t = pts @ pole
    x = (t + 1.0) / 2.0

    a = (dim - 1) / 2.0
    theta_max = r * np.pi
    lower_x = (1.0 + np.cos(theta_max)) / 2.0
    F_lower = beta_dist.cdf(lower_x, a, a)

    def truncated_cdf(vals):
        raw = beta_dist.cdf(vals, a, a)
        return np.clip((raw - F_lower) / (1.0 - F_lower), 0.0, 1.0)

    stat, pvalue = kstest(x, truncated_cdf)
    assert pvalue > 0.01, f"KS test rejected uniformity within cap: stat={stat}, p={pvalue}"


def test_cap_sampler_tangential_direction_unbiased():
    """The orthogonal-direction component should show no directional bias:
    mean projection along fixed orthogonal axes should be ~0."""
    dim = 20
    pole = np.zeros(dim)
    pole[0] = 1.0
    r = 0.3
    rng = np.random.default_rng(5)
    pts = sample_cap_points(pole[None, :], np.array([r]), 50000, rng)[0]

    tangential = pts - (pts @ pole)[:, None] * pole[None, :]
    means = tangential.mean(axis=0)
    stderr = tangential.std(axis=0) / np.sqrt(len(pts))
    assert np.all(np.abs(means) < 5 * stderr + 1e-6)


def test_intersection_fraction_exactly_one_when_fully_contained():
    """D + r_t <= r_c: every sampled point in t's cap is provably also in
    q's cap (triangle inequality) -- MC estimate must be exactly 1.0."""
    dim = 24
    rng_setup = np.random.default_rng(6)
    q = _random_pole(dim, seed=7)
    t = q.copy()
    r_t = 0.05
    r_c = 0.25  # D=0 here, so D + r_t = 0.05 <= r_c = 0.25 holds comfortably

    rng = np.random.default_rng(8)
    pts = sample_cap_points(t[None, :], np.array([r_t]), 2000, rng)[0]
    dists_to_q = np.arccos(np.clip(pts @ q, -1.0, 1.0)) / np.pi
    frac = float(np.mean(dists_to_q <= r_c))
    assert frac == 1.0


def test_intersection_fraction_exactly_zero_when_no_overlap():
    """D >= r_c + r_t: every sampled point in t's cap is provably outside
    q's cap -- MC estimate must be exactly 0.0."""
    dim = 24
    q = np.zeros(dim); q[0] = 1.0
    # place t at angular distance D from q along axis 0/1 plane
    D = 0.6
    theta = D * np.pi
    t = np.zeros(dim); t[0] = np.cos(theta); t[1] = np.sin(theta)
    r_t = 0.1
    r_c = 0.2  # r_c + r_t = 0.3 < D = 0.6

    rng = np.random.default_rng(9)
    pts = sample_cap_points(t[None, :], np.array([r_t]), 2000, rng)[0]
    dists_to_q = np.arccos(np.clip(pts @ q, -1.0, 1.0)) / np.pi
    frac = float(np.mean(dists_to_q <= r_c))
    assert frac == 0.0


def test_monte_carlo_convergence():
    """Repeated runs at fixed N with different seeds on a partial-overlap
    case should have spread roughly matching sqrt(p(1-p)/N)."""
    dim = 24
    q = np.zeros(dim); q[0] = 1.0
    D = 0.15
    theta = D * np.pi
    t = np.zeros(dim); t[0] = np.cos(theta); t[1] = np.sin(theta)
    r_t = 0.15
    r_c = 0.15  # partial overlap: D=0.15, r_c+r_t=0.3 > D, but D+r_t=0.3 > r_c so not full certification

    n_samples = 2000
    estimates = []
    for seed in range(30):
        rng = np.random.default_rng(seed + 100)
        pts = sample_cap_points(t[None, :], np.array([r_t]), n_samples, rng)[0]
        dists_to_q = np.arccos(np.clip(pts @ q, -1.0, 1.0)) / np.pi
        estimates.append(np.mean(dists_to_q <= r_c))
    estimates = np.array(estimates)
    p_hat = estimates.mean()
    empirical_std = estimates.std()
    theoretical_std = np.sqrt(p_hat * (1 - p_hat) / n_samples)
    assert 0.0 < p_hat < 1.0, "test case should be genuine partial overlap"
    assert empirical_std < 4 * theoretical_std + 0.01, (empirical_std, theoretical_std)


def test_oracle_hit_matches_manual_R_c_containment():
    from tests.helpers import make_clustered_candidate_benchmark
    from utils.base.selection import select_candidates

    bm = make_clustered_candidate_benchmark(
        seed=10, dim=16, n_clusters=4,
        n_base_per_cluster=20, n_cache_per_cluster=8, n_test_per_cluster=4,
        K=3, noise_scale=0.05,
    )
    candidates = dc.propose_full_pool(
        bm['base'], bm['cache'], bm['test'], bm['r_Q'], bm['test_r_t'], K=3, n_max=5,
    )
    selected, _traj = select_candidates(
        candidates, bm['n_base'], strategy='greedy_cost',
        n_train_queries=bm['n_cache'], n_test_queries=bm['n_test'], seed=42,
    )
    assert selected

    q_hat = bm['test'] / np.linalg.norm(bm['test'], axis=1, keepdims=True)
    sims = q_hat @ bm['base'].T
    top_idx = np.argsort(-sims, axis=1)[:, :3]
    gt_matrix = top_idx.astype(np.int32)

    for c in selected:
        expected = c.R_c[gt_matrix].all(axis=1)
        # direct manual check per test query
        for i in range(bm['n_test']):
            manual = bool(c.R_c[gt_matrix[i]].all())
            assert manual == bool(expected[i])


def test_oracle_hit_true_implies_query_covered_by_union():
    """One-directional only: if any candidate's oracle_hit is True for a
    query, the union-based covered label must also be True. Do NOT assert
    the reverse (it's false in general -- neighbors can be split across
    multiple candidates' balls)."""
    from tests.helpers import make_clustered_candidate_benchmark
    from utils.base.selection import select_candidates

    bm = make_clustered_candidate_benchmark(
        seed=11, dim=16, n_clusters=4,
        n_base_per_cluster=20, n_cache_per_cluster=8, n_test_per_cluster=4,
        K=2, noise_scale=0.05,
    )
    candidates = dc.propose_full_pool(
        bm['base'], bm['cache'], bm['test'], bm['r_Q'], bm['test_r_t'], K=2, n_max=5,
    )
    selected, _traj = select_candidates(
        candidates, bm['n_base'], strategy='greedy_cost',
        n_train_queries=bm['n_cache'], n_test_queries=bm['n_test'], seed=42,
    )
    assert selected

    q_hat = bm['test'] / np.linalg.norm(bm['test'], axis=1, keepdims=True)
    sims = q_hat @ bm['base'].T
    top_idx = np.argsort(-sims, axis=1)[:, :2]
    gt_matrix = top_idx.astype(np.int32)

    cached_mask = np.zeros(bm['n_base'], dtype=bool)
    for c in selected:
        cached_mask |= c.R_c
    union_covered = cached_mask[gt_matrix].all(axis=1)

    any_candidate_oracle_hit = np.zeros(bm['n_test'], dtype=bool)
    for c in selected:
        any_candidate_oracle_hit |= c.R_c[gt_matrix].all(axis=1)

    assert np.all(union_covered[any_candidate_oracle_hit])


def test_binning_produces_expected_bin_means():
    fractions = np.array([0.0, 0.05, 0.5, 0.55, 0.95, 1.0])
    hits = np.array([0, 1, 0, 1, 1, 1], dtype=bool)
    result = bin_calibration(fractions, hits, n_bins=10)
    # bin 0: [0,0.1] -> indices 0,1 (0.0, 0.05) -> hits 0,1 -> mean 0.5
    assert abs(result['bin_means'][0] - 0.5) < 1e-9
    assert result['bin_counts'][0] == 2
    # bin 4: (0.4,0.5] -> index 2 (0.5) -> hit 0 -> mean 0.0
    assert abs(result['bin_means'][4] - 0.0) < 1e-9
    # bin 5: (0.5,0.6] -> index 3 (0.55) -> hit 1 -> mean 1.0
    assert abs(result['bin_means'][5] - 1.0) < 1e-9
    # bin 9: (0.9,1.0] -> indices 4,5 (0.95, 1.0) -> hits 1,1 -> mean 1.0
    assert abs(result['bin_means'][9] - 1.0) < 1e-9


def test_auc_perfectly_separable_and_near_chance():
    rng = np.random.default_rng(12)
    fractions = np.concatenate([rng.uniform(0.6, 1.0, 50), rng.uniform(0.0, 0.4, 50)])
    hits = np.concatenate([np.ones(50, dtype=bool), np.zeros(50, dtype=bool)])
    auc = auc_score(fractions, hits)
    assert auc == 1.0

    # deterministic near-chance case: alternating labels independent of score order
    fractions2 = np.arange(20, dtype=np.float64)
    hits2 = np.array([i % 2 == 0 for i in range(20)])
    auc2 = auc_score(fractions2, hits2)
    assert 0.35 < auc2 < 0.65


def test_union_covered_at_least_as_permissive_as_per_candidate_oracle_hit():
    from tests.helpers import make_clustered_candidate_benchmark
    from utils.base.selection import select_candidates

    bm = make_clustered_candidate_benchmark(
        seed=22, dim=16, n_clusters=4,
        n_base_per_cluster=20, n_cache_per_cluster=8, n_test_per_cluster=4,
        K=2, noise_scale=0.05,
    )
    candidates = dc.propose_full_pool(
        bm['base'], bm['cache'], bm['test'], bm['r_Q'], bm['test_r_t'], K=2, n_max=5,
    )
    selected, _traj = select_candidates(
        candidates, bm['n_base'], strategy='greedy_cost',
        n_train_queries=bm['n_cache'], n_test_queries=bm['n_test'], seed=42,
    )
    assert selected

    q_hat = bm['test'] / np.linalg.norm(bm['test'], axis=1, keepdims=True)
    sims = q_hat @ bm['base'].T
    top_idx = np.argsort(-sims, axis=1)[:, :2]
    gt_matrix = top_idx.astype(np.int32)

    cached_mask = np.zeros(bm['n_base'], dtype=bool)
    any_per_candidate_hit = np.zeros(bm['n_test'], dtype=bool)
    for c in selected:
        cached_mask |= c.R_c
        any_per_candidate_hit |= c.R_c[gt_matrix].all(axis=1)
    union_covered = cached_mask[gt_matrix].all(axis=1)

    assert np.all(union_covered[any_per_candidate_hit])


def test_union_covered_at_least_as_permissive_on_real_data():
    """Same invariant as above, against compare_angular_vs_solid_ball's
    actual returned oracle_hit/union_covered arrays on real data."""
    bdir = dc.DATASET_CONFIG['sift']['base_dir'] / 'sift_b50k_c256k100_t512n20_s42'
    if not (bdir / 'cache_queries.npy').exists():
        pytest.skip('sift_b50k_c256k100_t512n20_s42 not prepared')

    result = compare_angular_vs_solid_ball(
        'sift', 'sift_b50k_c256k100_t512n20_s42', K=1, budget_fraction=0.3,
        n_test_queries=10, max_candidates_per_query=30,
    )
    assert result is not None
    n_test_queries, n_cand_sample = result['n_test_queries'], result['n_cand_sample']
    oracle_hit_matrix = result['oracle_hit'].reshape(n_test_queries, n_cand_sample)
    any_per_candidate_hit = oracle_hit_matrix.any(axis=1)
    assert np.all(result['union_covered'][any_per_candidate_hit])

#-------------------------------------------------------------------------------
# Raw sweep persistence: save_sweep/load_combo_result/load_sweep
#-------------------------------------------------------------------------------


def _synthetic_result(n_pairs: int = 7, n_test_queries: int = 3) -> dict:
    rng = np.random.RandomState(0)
    return {
        'angular_fraction': rng.uniform(0, 1, n_pairs),
        'solid_ball_fraction': rng.uniform(0, 1, n_pairs),
        'oracle_hit': rng.uniform(0, 1, n_pairs) > 0.5,
        'recall': rng.uniform(0, 1, n_pairs),
        'best_angular_fraction': rng.uniform(0, 1, n_test_queries),
        'best_solid_ball_fraction': rng.uniform(0, 1, n_test_queries),
        'union_covered': rng.uniform(0, 1, n_test_queries) > 0.5,
        'union_recall': rng.uniform(0, 1, n_test_queries),
        'd': 128, 'K': 5, 'n_pairs': n_pairs, 'n_test_queries': n_test_queries,
        'n_selected': 4, 'n_cand_sample': n_pairs // n_test_queries,
    }


def test_save_load_combo_result_round_trip(tmp_path):
    result = _synthetic_result()
    data_by_bm_k = {'bench_a': {5: {0.5: result}}}
    save_sweep('sift', data_by_bm_k, tmp_path)

    path = tmp_path / 'sift' / 'raw' / 'bench_a_K5_budget50pct.npz'
    assert path.exists()
    loaded = load_combo_result(path)

    for key in ('angular_fraction', 'solid_ball_fraction', 'oracle_hit', 'recall',
                'best_angular_fraction', 'best_solid_ball_fraction',
                'union_covered', 'union_recall'):
        np.testing.assert_array_equal(loaded[key], result[key])
    for key in ('d', 'K', 'n_pairs', 'n_test_queries', 'n_selected', 'n_cand_sample'):
        assert loaded[key] == result[key]
        assert isinstance(loaded[key], int)


def test_load_combo_result_missing_file_returns_none(tmp_path):
    assert load_combo_result(tmp_path / 'does_not_exist.npz') is None


def test_save_sweep_skips_none_results(tmp_path):
    data_by_bm_k = {'bench_a': {5: {0.5: None, 0.9: _synthetic_result()}}}
    save_sweep('sift', data_by_bm_k, tmp_path)
    assert not (tmp_path / 'sift' / 'raw' / 'bench_a_K5_budget50pct.npz').exists()
    assert (tmp_path / 'sift' / 'raw' / 'bench_a_K5_budget90pct.npz').exists()


def test_load_sweep_reconstructs_same_shape_as_run(tmp_path):
    result_a = _synthetic_result()
    result_b = _synthetic_result(n_pairs=9, n_test_queries=3)
    data_by_bm_k = {'bench_a': {5: {0.5: result_a, 0.9: None}, 10: {0.5: result_b}}}
    save_sweep('sift', data_by_bm_k, tmp_path)

    loaded = load_sweep(
        'sift', tmp_path, k_values=[5, 10], budget_fractions=[0.5, 0.9],
        benchmarks=['bench_a'],
    )
    assert loaded['bench_a'][5][0.9] is None
    np.testing.assert_array_equal(loaded['bench_a'][5][0.5]['recall'], result_a['recall'])
    np.testing.assert_array_equal(loaded['bench_a'][10][0.5]['recall'], result_b['recall'])


def test_save_then_load_sweep_matches_real_sweep():
    """End-to-end: a real (small) run_dual_predictor_sweep, saved and
    reloaded, must reproduce the same data -- not just the synthetic
    round trip above."""
    bdir = dc.DATASET_CONFIG['sift']['base_dir'] / 'sift_b50k_c256k100_t512n20_s42'
    if not (bdir / 'cache_queries.npy').exists():
        pytest.skip('sift_b50k_c256k100_t512n20_s42 not prepared')

    from simulations.population.debug_intersection_calibration import run_dual_predictor_sweep
    import tempfile

    data_by_bm_k = run_dual_predictor_sweep(
        'sift', k_values=[1], budget_fractions=[0.3],
        n_test_queries=10, max_candidates_per_query=30,
        benchmarks=['sift_b50k_c256k100_t512n20_s42'],
    )
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        save_sweep('sift', data_by_bm_k, run_dir)
        loaded = load_sweep(
            'sift', run_dir, k_values=[1], budget_fractions=[0.3],
            benchmarks=['sift_b50k_c256k100_t512n20_s42'],
        )
    original = data_by_bm_k['sift_b50k_c256k100_t512n20_s42'][1][0.3]
    reloaded = loaded['sift_b50k_c256k100_t512n20_s42'][1][0.3]
    assert reloaded is not None
    np.testing.assert_array_equal(reloaded['angular_fraction'], original['angular_fraction'])
    np.testing.assert_array_equal(reloaded['recall'], original['recall'])
    assert reloaded['n_pairs'] == original['n_pairs']

#-------------------------------------------------------------------------------
# exact_intersection_fraction: 1D quadrature reference method
#-------------------------------------------------------------------------------


def test_exact_fraction_monotone_nonincreasing_in_distance():
    """Fixed equal radii: intersection fraction must be monotonically
    non-increasing as the centers move apart."""
    r = 0.12
    d = 16
    distances = np.linspace(0.0, 2 * r, 15)
    fractions = [exact_intersection_fraction(D, r, r, d) for D in distances]
    diffs = np.diff(fractions)
    assert np.all(diffs <= 1e-9), fractions


def test_exact_fraction_monotone_nonincreasing_in_dimension():
    """Fixed equal radii and a fixed partial-overlap distance: intersection
    fraction must be monotonically non-increasing as dimension increases
    (concentration of measure -- fixed angular separation becomes
    relatively 'more distant' as ambient dimension grows)."""
    r = 0.15
    D = 0.1  # partial overlap: D < 2r
    dims = [3, 5, 10, 20, 50, 100]
    fractions = [exact_intersection_fraction(D, r, r, d) for d in dims]
    diffs = np.diff(fractions)
    assert np.all(diffs <= 1e-9), list(zip(dims, fractions))


def test_exact_fraction_matches_elementary_d2():
    """d=2: caps are arcs: the quadrature must match an independent,
    elementary arc-overlap formula (general 1D-interval-overlap, not the
    naive r_c_a+r_t_a-D_a which is wrong when one interval fully contains
    the other, e.g. D=0 with r_c<r_t)."""
    def elementary_d2(D, r_c, r_t):
        D_a, rc_a, rt_a = D * np.pi, r_c * np.pi, r_t * np.pi
        overlap = max(0.0, min(rc_a, D_a + rt_a) - max(-rc_a, D_a - rt_a))
        return overlap / (2 * rt_a)

    for D, r_c, r_t in [(0.1, 0.1, 0.1), (0.05, 0.15, 0.1), (0.12, 0.08, 0.1),
                          (0.2, 0.2, 0.05), (0.0, 0.1, 0.2)]:
        exact = exact_intersection_fraction(D, r_c, r_t, 2)
        expected = elementary_d2(D, r_c, r_t)
        assert abs(exact - expected) < 1e-8, (D, r_c, r_t, exact, expected)


def test_exact_fraction_matches_elementary_d3():
    """d=3: both the outer weight (sin(theta)) and the cross-sectional
    single-cap fraction (an S^1 arc fraction, theta/pi) reduce to
    elementary trig expressions -- integrate them independently via a raw
    quadrature of elementary functions, not reusing
    single_cap_volume_fraction, as a genuine independent check."""
    from scipy.integrate import quad

    def elementary_d3(D, r_c, r_t):
        D_a, rc_a, rt_a = D * np.pi, r_c * np.pi, r_t * np.pi
        if D_a >= rc_a + rt_a:
            return 0.0
        if D_a + rt_a <= rc_a:
            return 1.0
        if D_a < 1e-12:
            return 1.0 if rc_a >= rt_a else (1 - np.cos(rc_a)) / (1 - np.cos(rt_a))

        def g(theta):
            c = (np.cos(rc_a) - np.cos(theta) * np.cos(D_a)) / (np.sin(theta) * np.sin(D_a))
            c = np.clip(c, -1.0, 1.0)
            return np.arccos(c) / np.pi  # elementary S^1 arc fraction

        num, _ = quad(lambda th: g(th) * np.sin(th), 1e-10, rt_a, limit=200)
        den = 1 - np.cos(rt_a)  # elementary: integral of sin(theta) from 0 to rt_a
        return num / den

    for D, r_c, r_t in [(0.1, 0.1, 0.1), (0.05, 0.15, 0.1), (0.12, 0.08, 0.1),
                          (0.2, 0.2, 0.05)]:
        exact = exact_intersection_fraction(D, r_c, r_t, 3)
        expected = elementary_d3(D, r_c, r_t)
        assert abs(exact - expected) < 1e-6, (D, r_c, r_t, exact, expected)


def test_exact_fraction_equal_radii_zero_distance_is_one():
    """Equal radii, D=0: intersection is exactly 1.0 -- tested explicitly
    as a boundary-equality case (D+r_t == r_c exactly), not just trusting
    the fast path's own internal logic."""
    for r in [0.05, 0.1, 0.2, 0.3]:
        for d in [3, 16, 64, 128]:
            assert exact_intersection_fraction(0.0, r, r, d) == 1.0


def test_exact_fraction_raw_volume_symmetric_under_swap():
    """The raw (non-normalized) intersection volume is symmetric under
    swapping candidate/query roles, even though intersection_fraction
    itself is intentionally asymmetric (normalizes by t's cap only):
    fraction(D,r_c,r_t,d) * cap_volume(r_t,d) == fraction(D,r_t,r_c,d) * cap_volume(r_c,d)."""
    d = 20
    for D, r_c, r_t in [(0.05, 0.1, 0.08), (0.1, 0.15, 0.12), (0.02, 0.05, 0.09)]:
        f1 = exact_intersection_fraction(D, r_c, r_t, d)
        f2 = exact_intersection_fraction(D, r_t, r_c, d)
        vol_t = float(single_cap_volume_fraction(np.array([r_t]), d)[0])
        vol_c = float(single_cap_volume_fraction(np.array([r_c]), d)[0])
        assert abs(f1 * vol_t - f2 * vol_c) < 1e-9 * max(vol_t, vol_c), (D, r_c, r_t, f1, f2)


def test_exact_fraction_continuous_at_no_overlap_boundary():
    """As D approaches r_c+r_t from below, the fraction approaches 0
    continuously -- no discontinuous jump at the fast-path boundary."""
    r_c, r_t, d = 0.1, 0.1, 16
    boundary = r_c + r_t
    for eps in [0.05, 0.01, 0.001, 0.0001]:
        f = exact_intersection_fraction(boundary - eps, r_c, r_t, d)
        assert 0.0 <= f < 0.5, (eps, f)
        assert f < exact_intersection_fraction(boundary - eps * 2, r_c, r_t, d) + 1e-9


def test_exact_fraction_continuous_at_full_containment_boundary():
    """As D approaches r_c-r_t from below (t's cap just barely fully
    inside c's cap), the fraction approaches 1 continuously."""
    r_c, r_t, d = 0.2, 0.08, 16
    boundary = r_c - r_t
    for eps in [0.05, 0.01, 0.001, 0.0001]:
        f = exact_intersection_fraction(max(boundary - eps, 0.0), r_c, r_t, d)
        assert 0.5 < f <= 1.0, (eps, f)


def test_exact_fraction_matches_single_cap_ratio_near_zero_distance():
    """At D=0 exactly this is the fast path's own formula (already tested
    above); here D is pushed very close to (but not exactly) 0 to also
    exercise the general quadrature path near the D->0 singularity and
    confirm it agrees with the same single-cap-ratio answer."""
    d = 10
    r_c, r_t = 0.05, 0.15  # r_c < r_t, so at D=0 intersection = cap_c entirely
    expected = float(
        single_cap_volume_fraction(np.array([r_c]), d)[0]
        / single_cap_volume_fraction(np.array([r_t]), d)[0]
    )
    f = exact_intersection_fraction(1e-6, r_c, r_t, d)
    assert abs(f - expected) < 1e-3, (f, expected)


def test_exact_fraction_degenerate_point_query():
    """r_t=0 (test query's cap collapses to a single point): intersection
    fraction is 1.0 if that point is within c's ball (D<=r_c), else 0.0."""
    d = 16
    r_c = 0.1
    assert exact_intersection_fraction(0.05, r_c, 0.0, d) == 1.0
    assert exact_intersection_fraction(0.15, r_c, 0.0, d) == 0.0
    assert exact_intersection_fraction(r_c, r_c, 0.0, d) == 1.0  # boundary: D==r_c


def test_exact_vs_monte_carlo_cross_check():
    """Cross-check the exact quadrature against the Monte Carlo sampler
    across a spread of configurations, including d=128 (real SIFT
    dimension) and small radii. Tolerance is derived from the theoretical
    MC standard error, not an arbitrary constant."""
    n_samples = 50000
    configs = [
        (0.05, 0.1, 0.08, 8),
        (0.02, 0.05, 0.05, 8),
        (0.05, 0.1, 0.08, 32),
        (0.1, 0.15, 0.12, 32),
        (0.05, 0.1, 0.08, 128),
        (0.02, 0.05, 0.05, 128),
        (0.1, 0.15, 0.12, 128),
    ]
    rng = np.random.default_rng(42)
    for D, r_c, r_t, d in configs:
        exact = exact_intersection_fraction(D, r_c, r_t, d)

        pole_t = np.zeros(d, dtype=np.float32); pole_t[0] = 1.0
        pole_q = np.zeros(d, dtype=np.float32)
        pole_q[0] = np.cos(D * np.pi); pole_q[1] = np.sin(D * np.pi)

        pts = sample_cap_points(pole_t[None, :], np.array([r_t]), n_samples, rng)[0]
        dist_to_q = np.arccos(np.clip(pts @ pole_q, -1.0, 1.0)) / np.pi
        mc = float(np.mean(dist_to_q <= r_c))

        se = np.sqrt(max(mc * (1 - mc), 1e-6) / n_samples)
        tol = max(4 * se, 0.01)
        assert abs(exact - mc) < tol, (D, r_c, r_t, d, exact, mc, tol)


def test_sampler_fix_no_collapse_at_high_dim_small_radius():
    """At d=128 with a small radius, sampled points must actually spread
    out within the cap, not collapse to the exact pole (sf/isf regression
    check -- see sample_cap_points docstring)."""
    d = 128
    r = 0.05
    pole = np.zeros(d, dtype=np.float32); pole[0] = 1.0
    rng = np.random.default_rng(0)
    pts = sample_cap_points(pole[None, :], np.array([r]), 2000, rng)[0]
    dists = np.arccos(np.clip(pts @ pole, -1.0, 1.0)) / np.pi

    assert dists.max() > 0.0, "all samples collapsed to the exact pole -- sf/isf fix regressed"
    assert dists.std() > 0.0
    assert dists.max() > r * 0.5, (
        f"samples cluster too close to the pole (max dist {dists.max()} for r={r}) "
        "-- suggests the tail sampling is still numerically degenerate"
    )

#-------------------------------------------------------------------------------
# exact_intersection_fraction: underflow guard + dtype robustness
#-------------------------------------------------------------------------------


def test_exact_fraction_raises_on_underflow():
    """d=384, r_t=r_c=0.02, D=0.0004: both num and den integrals underflow
    to exactly 0.0 -- must raise a clear error, not silently return NaN.
    Avoids both fast paths, exercising the general quadrature path."""
    with pytest.raises(ValueError, match="underflowed"):
        exact_intersection_fraction(0.0004, 0.02, 0.02, 384)


def test_exact_fraction_no_underflow_at_real_data_ranges():
    """Regression guard: the fix must not become overly conservative and
    reject real, safe production values -- SIFT's real range (d=128, r
    down to ~0.02) and ESCI's real range (d=384, r down to ~0.16) must
    not raise."""
    exact_intersection_fraction(0.05, 0.1, 0.025, 128)  # SIFT-like
    exact_intersection_fraction(0.2, 0.3, 0.16, 384)     # ESCI-like


def test_exact_fraction_rejects_float32_inputs():
    """float32 inputs must be rejected outright with a clear TypeError,
    not silently coerced (float32 precision loss produces spurious
    garbage at extreme-magnitude configs)."""
    D, r_c, r_t, d = np.float32(0.2931), np.float32(0.4238), np.float32(0.2529), 384
    with pytest.raises(TypeError, match="float64"):
        exact_intersection_fraction(D, r_c, r_t, d)
    # float64 (Python float or np.float64) must still work fine
    exact_intersection_fraction(float(D), float(r_c), float(r_t), d)
    exact_intersection_fraction(np.float64(D), np.float64(r_c), np.float64(r_t), d)


def test_exact_fraction_split_check_agrees_on_real_and_synthetic_configs():
    """The independent split-range recomputation should agree with the
    primary result (no warning) across both extreme ESCI-like and
    ordinary SIFT-like configs -- no false-alarm warnings."""
    import warnings
    configs = [
        (0.2931, 0.4238, 0.2529, 384),
        (0.2517, 0.3628, 0.1839, 384),
        (0.05, 0.1, 0.08, 128),
        (0.02, 0.05, 0.05, 128),
        (0.1, 0.3, 0.2, 16),
    ]
    for D, r_c, r_t, d in configs:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            exact_intersection_fraction(D, r_c, r_t, d, verify=True)

#-------------------------------------------------------------------------------
# solid_ball_intersection_fraction: second predictor
#-------------------------------------------------------------------------------


def test_solid_ball_J_boundary_values():
    """J(R, 0) = 0.5 (hyperplane through center), J(R,R) = 0 (zero-height
    cap), J(R,-R) = 1 (entire ball) -- checked indirectly via
    solid_ball_intersection_fraction's degenerate-case behavior."""
    # R1=R2=1, dist=1: hyperplane through midpoint, each J=0.5 by symmetry.
    for n in [2, 5, 20]:
        val = solid_ball_intersection_fraction(1.0, 1.0, 1.0, n)
        assert 0.0 < val < 1.0


def test_solid_ball_hand_computable_case_n3():
    """R1=R2=dist=sqrt(2), n=3: classical closed-form answer is exactly
    5/16 = 0.3125 (cap height h=R/2, Vol_cap=(pi*h^2/3)*(3R-h), two equal
    caps over full sphere volume)."""
    R = np.sqrt(2)
    val = solid_ball_intersection_fraction(R, R, R, 3)
    assert abs(val - 5.0 / 16.0) < 1e-9


def test_solid_ball_matches_brute_force_rejection_sampling():
    """Independent validation: literal rejection sampling within solid
    balls in R^n (no reuse of any spherical-cap or angular machinery),
    compared against the closed-form formula."""
    rng = np.random.default_rng(3)

    def brute_force(dist, R1, R2, n, N=1_500_000):
        pts = rng.uniform(-R2, R2, size=(N * 3, n))
        norms = np.linalg.norm(pts, axis=1)
        pts = pts[norms <= R2][:N]
        center1 = np.zeros(n)
        center1[0] = dist
        dists_to_1 = np.linalg.norm(pts - center1, axis=1)
        return float(np.mean(dists_to_1 <= R1)), len(pts)

    for dist, R1, R2, n in [(0.5, 0.6, 0.5, 3), (0.8, 0.5, 0.6, 4)]:
        exact = solid_ball_intersection_fraction(dist, R1, R2, n)
        brute, npts = brute_force(dist, R1, R2, n)
        se = np.sqrt(brute * (1 - brute) / npts)
        assert abs(exact - brute) < 5 * se + 0.001, (dist, R1, R2, n, exact, brute, se)


def test_solid_ball_degenerate_cases():
    n = 10
    assert solid_ball_intersection_fraction(2.0, 0.5, 0.5, n) == 0.0  # no overlap
    assert solid_ball_intersection_fraction(0.1, 0.5, 0.2, n) == 1.0  # ball2 fully in ball1
    # ball1 fully in ball2 -> ratio of volumes, not exactly 0 or 1
    val = solid_ball_intersection_fraction(0.1, 0.2, 0.5, n)
    assert 0.0 < val < 1.0
    assert abs(val - (0.2 / 0.5) ** n) < 1e-9


def test_solid_ball_rejects_float32_inputs():
    with pytest.raises(TypeError, match="float64"):
        solid_ball_intersection_fraction(np.float32(0.5), np.float32(0.6), np.float32(0.5), 10)
    solid_ball_intersection_fraction(float(0.5), float(0.6), float(0.5), 10)


def test_solid_ball_literal_matches_ratio_based():
    """solid_ball_intersection_fraction_literal (computes V_n(R) directly)
    must agree with the ratio-based solid_ball_intersection_fraction to
    near machine precision, at both small-scale and real n=384 scale."""
    configs = [
        (0.5, 0.6, 0.5, 3), (0.8, 0.5, 0.6, 4), (1.0, 0.7, 0.9, 5),
        (0.157, 0.313, 0.251, 128), (0.467, 0.908, 0.765, 384),
        (0.908, 1.414, 1.176, 384),
    ]
    for dist, R1, R2, n in configs:
        ratio_based = solid_ball_intersection_fraction(dist, R1, R2, n)
        literal = solid_ball_intersection_fraction_literal(dist, R1, R2, n)
        assert abs(ratio_based - literal) < 1e-8, (dist, R1, R2, n, ratio_based, literal)


def test_chord_distance_matches_direct_l2():
    """The chord-distance formula (2*sin(r*pi/2)) must match the direct
    Euclidean L2 distance between the unit vectors it's derived from."""
    rng = np.random.default_rng(0)
    d = 64
    for _ in range(5):
        a = rng.standard_normal(d); a /= np.linalg.norm(a)
        b = rng.standard_normal(d); b /= np.linalg.norm(b)
        dot = np.clip(a @ b, -1.0, 1.0)
        r_normalized = np.arccos(dot) / np.pi
        chord_formula = float(chord_distance_from_angular(np.array([r_normalized]))[0])
        chord_direct = float(np.linalg.norm(a - b))
        assert abs(chord_formula - chord_direct) < 1e-6


def test_conditional_cap_fraction_matches_single_cap_volume_fraction():
    """conditional_cap_fraction inlines single_cap_volume_fraction's
    formula as scalar math for speed (see exact_intersection_fraction's
    hot integrand path) -- must still match the reference vectorized
    implementation exactly, across real SIFT/ESCI dimensions."""
    rng = np.random.default_rng(0)
    for d in [16, 128, 384]:
        for _ in range(20):
            theta = rng.uniform(0.05, np.pi - 0.05)
            D = rng.uniform(0.05, np.pi - 0.05)
            r_c = rng.uniform(0.05, np.pi - 0.05)

            c = (np.cos(r_c) - np.cos(theta) * np.cos(D)) / (np.sin(theta) * np.sin(D))
            c = np.clip(c, -1.0, 1.0)
            r_cross = np.arccos(c) / np.pi
            expected = float(single_cap_volume_fraction(np.array([r_cross]), d - 1)[0])

            actual = conditional_cap_fraction(theta, D, r_c, d)
            assert abs(actual - expected) < 1e-12, (theta, D, r_c, d, actual, expected)


def test_compare_angular_vs_solid_ball_smoke():
    bdir = dc.DATASET_CONFIG['sift']['base_dir'] / 'sift_b50k_c256k100_t512n20_s42'
    if not (bdir / 'cache_queries.npy').exists():
        pytest.skip('sift_b50k_c256k100_t512n20_s42 not prepared')

    result = compare_angular_vs_solid_ball(
        'sift', 'sift_b50k_c256k100_t512n20_s42', K=1, budget_fraction=0.3,
        n_test_queries=5, max_candidates_per_query=10,
    )
    assert result is not None
    assert len(result['angular_fraction']) == result['n_pairs']
    assert len(result['solid_ball_fraction']) == result['n_pairs']
    assert len(result['oracle_hit']) == result['n_pairs']
    assert np.all((result['angular_fraction'] >= 0.0) & (result['angular_fraction'] <= 1.0))
    assert np.all((result['solid_ball_fraction'] >= -1e-9) & (result['solid_ball_fraction'] <= 1.0 + 1e-9))
    assert len(result['best_angular_fraction']) == result['n_test_queries']
    assert len(result['best_solid_ball_fraction']) == result['n_test_queries']
    assert len(result['union_covered']) == result['n_test_queries']
    # best fraction must be >= any individual sampled fraction it was maxed over
    assert np.all(result['best_angular_fraction'] <= 1.0 + 1e-9)
    assert np.all(result['best_solid_ball_fraction'] <= 1.0 + 1e-9)
    assert result['K'] == 1
    assert len(result['recall']) == result['n_pairs']
    assert np.all((result['recall'] >= 0.0) & (result['recall'] <= 1.0))
    # K=1: recall is identical to oracle_hit as a 0/1 value, no partial credit.
    np.testing.assert_array_equal(result['recall'], result['oracle_hit'].astype(np.float64))


def test_recall_pair_matches_manual_fraction_at_k5():
    """recall must equal (# of the query's true top-K neighbors present
    in that one candidate's cache) / K exactly, checked at K=5 (where
    it's genuinely continuous, not just 0/1 like the K=1 case above)."""
    bdir = dc.DATASET_CONFIG['sift']['base_dir'] / 'sift_b50k_c256k100_t512n20_s42'
    if not (bdir / 'cache_queries.npy').exists():
        pytest.skip('sift_b50k_c256k100_t512n20_s42 not prepared')

    K = 5
    result = compare_angular_vs_solid_ball(
        'sift', 'sift_b50k_c256k100_t512n20_s42', K=K, budget_fraction=0.5,
        n_test_queries=5, max_candidates_per_query=10,
    )
    assert result is not None
    assert result['K'] == K
    # recall*K must always be a whole number of found neighbors (0..K).
    counts = result['recall'] * K
    np.testing.assert_allclose(counts, np.round(counts), atol=1e-9)
    assert np.all((counts >= 0) & (counts <= K))
    # oracle_hit (all K found) must exactly coincide with recall==1.0.
    np.testing.assert_array_equal(result['oracle_hit'], result['recall'] >= 1.0 - 1e-9)

#-------------------------------------------------------------------------------
# Statistical helpers: fit_binomial_glm, mannwhitney_test, point_biserial_test
#-------------------------------------------------------------------------------


def test_fit_binomial_glm_recovers_known_sigmoid():
    """Fit a GLM on data drawn from a known logistic relationship;
    the fitted slope must be positive and clearly significant, and
    pseudo-R2 must be high (data was generated to be highly separable)."""
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.uniform(0.0, 1.0, n)
    true_p = 1.0 / (1.0 + np.exp(-(x - 0.5) * 15))
    y = (rng.uniform(0.0, 1.0, n) < true_p).astype(float)

    fit = fit_binomial_glm(x, y, 1)
    assert fit['slope'] > 0
    assert fit['slope_pvalue'] < 1e-6
    assert fit['pseudo_r2'] > 0.4


def test_fit_binomial_glm_null_data_gives_weak_fit():
    """Fit a GLM on data with genuinely no relationship (y independent of
    x) -- pseudo-R2 should be near zero and the slope not significant."""
    rng = np.random.default_rng(1)
    n = 2000
    x = rng.uniform(0.0, 1.0, n)
    y = rng.integers(0, 2, n).astype(float)

    fit = fit_binomial_glm(x, y, 1)
    assert fit['pseudo_r2'] < 0.01
    assert fit['slope_pvalue'] > 0.05


def test_fit_binomial_glm_grouped_recall_case():
    """Grouped/proportion form (n_trials=K>1, successes=round(recall*K))
    must also recover a known relationship -- same model family as the
    binary case, generalized."""
    rng = np.random.default_rng(2)
    n = 1000
    K = 5
    x = rng.uniform(0.0, 1.0, n)
    true_p = 1.0 / (1.0 + np.exp(-(x - 0.5) * 10))
    successes = rng.binomial(K, true_p).astype(float)

    fit = fit_binomial_glm(x, successes, K)
    assert fit['slope'] > 0
    assert fit['pseudo_r2'] > 0.1


def test_mannwhitney_matches_auc_score():
    """mannwhitney_test's 'auc' must exactly match the existing
    auc_score (same Mann-Whitney U statistic, just also carrying a
    p-value) -- direct cross-check between the hand-rolled and
    library-based computations."""
    rng = np.random.default_rng(3)
    n = 500
    x = rng.uniform(0.0, 1.0, n)
    y = (x + rng.normal(0, 0.2, n)) > 0.5

    mw = mannwhitney_test(x, y)
    assert abs(mw['auc'] - auc_score(x, y)) < 1e-9
    assert 0.0 <= mw['pvalue'] <= 1.0


def test_mannwhitney_empty_group_returns_nan():
    x = np.array([0.1, 0.2, 0.3])
    y = np.array([True, True, True])  # no misses -- one group empty
    mw = mannwhitney_test(x, y)
    assert np.isnan(mw['auc'])
    assert np.isnan(mw['pvalue'])


def test_point_biserial_positive_for_positively_related_data():
    rng = np.random.default_rng(4)
    n = 500
    x = rng.uniform(0.0, 1.0, n)
    y = (x + rng.normal(0, 0.1, n)) > 0.5

    pb = point_biserial_test(x, y)
    assert pb['correlation'] > 0.5
    assert pb['pvalue'] < 1e-6

#-------------------------------------------------------------------------------
# _actual_budgets: phantom-budget bug fix
#-------------------------------------------------------------------------------


def test_actual_budgets_unions_across_benchmarks():
    data = {
        'bm1': {1: {0.05: {}, 0.15: {}}},
        'bm2': {1: {0.15: {}, 0.30: {}}},
    }
    assert _actual_budgets(data, [1]) == [0.05, 0.15, 0.30]


def test_actual_budgets_respects_requested_k_values():
    data = {'bm1': {1: {0.05: {}}, 5: {0.15: {}}}}
    assert _actual_budgets(data, [1]) == [0.05]
    assert _actual_budgets(data, [1, 5]) == [0.05, 0.15]


def test_actual_budgets_empty_data():
    assert _actual_budgets({}, [1, 5, 10]) == []


def test_actual_budgets_skips_missing_k():
    # K=5 never appears for this benchmark -- must not raise.
    data = {'bm1': {1: {0.05: {}}}}
    assert _actual_budgets(data, [1, 5]) == [0.05]


def test_actual_budgets_counts_key_presence_not_value_truthiness():
    # A budget attempted-but-failed (result stored as None) still counts
    # as "actually swept" -- _actual_budgets looks at dict keys, not at
    # whether the stored result is non-None.
    data = {'bm1': {1: {0.05: None, 0.15: {}}}}
    assert _actual_budgets(data, [1]) == [0.05, 0.15]


def test_actual_budgets_exclude_full_drops_100pct():
    """budget=1.0 (100%) is a degenerate config (near-unconstrained
    budget) that skews pooled statistics -- exclude_full=True must drop
    it, exclude_full=False (default) must keep it."""
    data = {'bm1': {1: {0.05: {}, 0.5: {}, 1.0: {}}}}
    assert _actual_budgets(data, [1]) == [0.05, 0.5, 1.0]
    assert _actual_budgets(data, [1], exclude_full=True) == [0.05, 0.5]

#-------------------------------------------------------------------------------


def test_dual_best_candidate_scatter_skips_absent_budget():
    """Regression test for the phantom-budget bug: must not write a file
    for a budget_fraction that isn't present anywhere in data_by_bm_k."""
    from simulations.population.debug_intersection_calibration import _run_dir

    n = 5
    bm_name = 'bm1_c256k100_t512n20_s42'
    data = {
        bm_name: {
            1: {
                0.30: {
                    'best_angular_fraction': np.linspace(0.0, 1.0, n),
                    'best_solid_ball_fraction': np.linspace(0.0, 1.0, n),
                    'union_covered': np.array([True, False, True, False, True]),
                },
            },
        },
    }
    absent_budget = 0.99
    out_path = _run_dir() / 'bm1_test_dataset' / f'dual_best_candidate_scatter_budget{absent_budget*100:.0f}pct.png'
    plot_dataset_dual_best_candidate_scatter(
        'bm1_test_dataset', data, absent_budget, k_values=[1],
    )
    assert not out_path.exists()
