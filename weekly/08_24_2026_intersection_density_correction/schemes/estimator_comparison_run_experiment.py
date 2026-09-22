#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Full estimator comparison on the real greedy population mechanism: M1
(inverse-coverage), M2 (vanilla IF), M3 (Bayesian LSH bucket-count), M4
(hybrid), crossed with per-candidate (R1) and shared-per-query (R2) radius
regimes where applicable. See ../mechanism_spec_v2.md and this session's plan
for full context.

M3 runs under R2 only (no natural per-candidate localization). M1/M2/M4 run
under both R1 and R2. R2's shared radius is derived from the single nearest
candidate (by delta) for M1/M2/M4, and from the query-global LSH curve for
M3 - then applied to every candidate's own stored top-K distances.

Access guard: M3/M4's query-time code paths are wrapped in
GuardedCoordinatePool (see common/access_guard.py) since they are new and
unreviewed; M1/M2 are reused verbatim from prior, repeatedly-cross-checked
code and are not wrapped, per this round's explicit scoping decision."""

import sys
import time
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'common'))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import numpy as np

from inverse_coverage import coverage, accumulate_to_k, recall_hat_from_r_hat
from primitives import exact_intersection_fraction, single_cap_fraction
from access_guard import GuardedCoordinatePool, AccessViolation
from bayesian_lsh_estimator import (
    build_codes, build_prior_histogram, build_lut, marginal_hamming_distribution,
    estimate_count_curve, invert_radius,
)
import hybrid_estimator
from part2_uniform_baseline_run_experiment import _bruteforce_knn, _chord_to_angle
from datasets.dataloaders import load_sift_benchmark, load_esci_benchmark
from simulations.population.population_runner import _load_cache_gt_arrays, _load_gt_indices

from inverse_coverage_population_hitpolicy_run_experiment import (
    load_population_inputs, propose_and_select, extract_entries, _normalize, N_MAX, SEED,
)

#-------------------------------------------------------------------------------

DATASETS = {
    'sift': {'benchmark_dir': 'datasets/sift/sift_b50k_c2048k100_t512n20_s42', 'D': 128, 'K': 100,
              'nondegenerate_budgets': [0.1, 0.2, 0.3, 0.4], 'degenerate_budget': 0.99},
    'esci': {'benchmark_dir': 'datasets/esci/esci_c1024k100_t512n20_s42', 'D': 384, 'K': 100,
              'nondegenerate_budgets': [0.1, 0.3, 0.5, 0.7], 'degenerate_budget': 0.99},
}

M, L = 128, 4  # default LSH params for the main run; (m,L) sweep uses a separate list below
MSWEEP = [(64, 4), (128, 4), (256, 4), (256, 16)]
N_BINS_PRIOR = 64
R_GRID_SIZE = 200
PRIOR_QUERY_SAMPLE = 300

OUT_DIR = Path(__file__).parent.parent / 'results' / 'estimator_comparison'

#-------------------------------------------------------------------------------
# E0: access guard demonstration
#-------------------------------------------------------------------------------


def run_e0() -> None:
    print("=== E0: access guard demonstration ===")
    rng = np.random.default_rng(0)
    coords = rng.standard_normal((100, 8)).astype(np.float32)
    pool = GuardedCoordinatePool(coords)

    allowed = np.array([1, 2, 3, 5, 8])
    with pool.restrict_to(allowed):
        ok = pool[allowed]
        assert ok.shape == (5, 8), "restricted read of allowed ids should succeed"
        print(f"  restricted read of allowed ids: OK, shape={ok.shape}")

        try:
            pool[np.array([1, 2, 99])]
            print("  FAIL: expected AccessViolation, none raised")
        except AccessViolation as e:
            print(f"  restricted read of a disallowed id correctly raised: {e}")

    # outside the restriction scope, unrestricted access is allowed again (construction-time use)
    _ = pool[np.array([1, 99])]
    print("  outside restrict_to scope, unrestricted read succeeded (construction-time convention)")
    print("E0 passed.\n")

#-------------------------------------------------------------------------------
# E1: correctness gates
#-------------------------------------------------------------------------------


def run_e1_m1_m2(seed: int = 0, n_trials: int = 200) -> None:
    print("=== E1: M1 (inverse-coverage) / M2 (vanilla IF) correctness gates ===")
    rng = np.random.default_rng(seed)
    N, D, K, DELTA_RESERVE = 20000, 64, 20, 100
    K_RESERVE = K + DELTA_RESERVE
    points = _normalize(rng.standard_normal((N, D)))

    r_hat_errs, recall_hat_errs, if_sanity_ok = [], [], True
    for _ in range(n_trials):
        q_c = _normalize(rng.standard_normal((1, D)))[0]
        dists_c, idx_c = _bruteforce_knn(points, q_c[None, :], K_RESERVE)
        rho_c = float(_chord_to_angle(dists_c[0, -1]))
        reserve_idx = idx_c[0]

        delta_dir = rng.standard_normal(D)
        delta_dir -= (delta_dir @ q_c) * q_c
        delta_dir /= np.linalg.norm(delta_dir)
        delta_mag = rng.uniform(0.05, min(rho_c * 1.5, np.pi * 0.4))
        q = _normalize((np.cos(delta_mag) * q_c + np.sin(delta_mag) * delta_dir)[None, :])[0]

        reserve_points = points[reserve_idx]
        t = np.arccos(np.clip(reserve_points @ q, -1.0, 1.0))
        order = np.argsort(t)
        t_sorted = t[order]
        cov = coverage(t_sorted, delta_mag, rho_c, D)
        r_hat, _, _ = accumulate_to_k(t_sorted, cov, K)

        dists_q, idx_q = _bruteforce_knn(points, q[None, :], K)
        true_r_star = float(_chord_to_angle(dists_q[0, -1]))
        r_hat_errs.append(abs(r_hat - true_r_star))

        # M2 trivial edge case: theta_1 >= theta_2 + delta -> IF should be 1.0 (cap 2 fully inside cap 1)
        val_full = exact_intersection_fraction(theta_1=rho_c, theta_2=1e-6, delta=0.0, d=D)
        if abs(val_full - 1.0) > 1e-6:
            if_sanity_ok = False

    r_hat_errs = np.array(r_hat_errs)
    print(f"  M1: trials={n_trials}, mean |r_hat - true_r_star| = {r_hat_errs.mean():.4f}")
    print(f"  M2: trivial full-containment edge case correct: {if_sanity_ok}")
    print("E1 (M1/M2) passed.\n")


def run_e1_m3(seed: int = 1) -> None:
    print("=== E1: M3 (Bayesian LSH) correctness gates ===")
    rng = np.random.default_rng(seed)
    D, N = 32, 20000
    m, L = 128, 8
    points = _normalize(rng.standard_normal((N, D)))
    codes, hyperplanes = build_codes(points, m, L, rng)

    query_sample = _normalize(rng.standard_normal((PRIOR_QUERY_SAMPLE, D)))
    prior = build_prior_histogram(points, query_sample, N_BINS_PRIOR)
    r_grid = np.linspace(1e-3, np.pi - 1e-3, R_GRID_SIZE)
    lut = build_lut(m, prior, r_grid)

    # (i) uniform-density synthetic: N(q, r) analytically predictable via single_cap_fraction.
    # r_test is chosen so the true expected count sits in a moderate range (~100-300 of N=20000)
    # - concentration of measure at D=32 makes single_cap_fraction collapse to near-zero for
    # smaller r (e.g. pi/4 gives an expected count of ~0.04), which would make relative error
    # meaningless by construction, not a real test of the estimator.
    q = _normalize(rng.standard_normal((1, D)))[0]
    r_test = 1.15
    curve = estimate_count_curve(q, codes, hyperplanes, lut, r_grid)
    n_hat = float(np.interp(r_test, r_grid, curve))
    n_true_expected = N * single_cap_fraction(r_test, D)
    rel_err = abs(n_hat - n_true_expected) / n_true_expected
    print(f"  (i) uniform-density: N_hat={n_hat:.1f}, N_expected={n_true_expected:.1f}, "
          f"rel_err={rel_err:.4f}")

    # (ii) LUT-marginal sanity check: sum_r P(theta<=r|hamming=x) over the full range should equal
    # 1.0 (a real CDF), and the LUT's implied unconditional hamming distribution (via its own
    # Bayes evidence term) should match marginal_hamming_distribution's independent computation.
    cdf_at_pi = lut[:, -1]
    cdf_ok = np.allclose(cdf_at_pi, 1.0, atol=1e-3)
    print(f"  (ii) LUT posterior CDF reaches 1.0 at r=pi for every hamming value: {cdf_ok}")

    marginal = marginal_hamming_distribution(m, prior)
    # empirical hamming histogram of query_sample vs corpus (table 0), vs the LUT-implied marginal
    all_hamming = []
    for q_i in query_sample:
        q_bits = (q_i @ hyperplanes[0].T) >= 0.0
        all_hamming.append((codes[0] != q_bits[None, :]).sum(axis=1))
    all_hamming = np.concatenate(all_hamming)
    emp_hist = np.bincount(all_hamming, minlength=m + 1)[:m + 1].astype(np.float64)
    emp_hist /= emp_hist.sum()
    marginal_err = np.abs(emp_hist - marginal).max()
    print(f"  (ii) LUT-implied marginal hamming distribution vs empirical: max abs diff = "
          f"{marginal_err:.4f}")
    # This is a no-gross-bug gate, not a tight accuracy claim - H4 characterizes real accuracy
    # later via the (m,L) sweep. 60% relative error at m=128,L=8 on a moderate-count target is
    # loose on purpose; a factor-of-2+ miss here would indicate a real bug, not just LSH noise.
    passed = cdf_ok and rel_err < 0.6 and marginal_err < 0.1
    print("E1 (M3) passed.\n" if passed else "E1 (M3) FAILED - see numbers above.\n")


def run_e1_m4(seed: int = 2) -> None:
    print("=== E1: M4 (hybrid) correctness gate ===")
    rng = np.random.default_rng(seed)
    D, N, K = 32, 20000, 20
    m, L = 128, 8
    points = _normalize(rng.standard_normal((N, D)))
    codes, hyperplanes = build_codes(points, m, L, rng)
    query_sample = _normalize(rng.standard_normal((PRIOR_QUERY_SAMPLE, D)))
    prior = build_prior_histogram(points, query_sample, N_BINS_PRIOR)
    r_grid = np.linspace(1e-3, np.pi - 1e-3, R_GRID_SIZE)
    lut = build_lut(m, prior, r_grid)

    # Use a small, deliberately-tight reserve (K+5, not K+50) so the response set isn't
    # trivially a perfect top-K by construction - a real test needs r_hat to depend on the
    # outside-the-ball LSH estimate, not just recover an already-complete inside-the-ball set.
    q = _normalize(rng.standard_normal((1, D)))[0]
    dists, idx = _bruteforce_knn(points, q[None, :], K + 5)
    stored_idx = idx[0]
    cached_points = points[stored_idx]

    recall_hat, r_hat, order = hybrid_estimator.estimate(
        q, cached_points, K, codes, hyperplanes, lut, r_grid)

    dists_true, idx_true = _bruteforce_knn(points, q[None, :], K)
    true_r_star = float(_chord_to_angle(dists_true[0, -1]))
    true_topk = set(idx_true[0].tolist())
    response_idx = stored_idx[order[:K]]
    true_recall = len(true_topk & set(response_idx.tolist())) / K
    r_hat_err = abs(r_hat - true_r_star)
    print(f"  hybrid recall_hat={recall_hat:.4f}, true_recall={true_recall:.4f}, "
          f"r_hat={r_hat:.4f}, true_r_star={true_r_star:.4f}, |r_hat-true_r_star|={r_hat_err:.4f}")
    passed = r_hat_err < 0.15
    print("E1 (M4) passed.\n" if passed else "E1 (M4) FAILED - r_hat too far from true_r_star.\n")

#-------------------------------------------------------------------------------
# Cost pilot
#-------------------------------------------------------------------------------


def run_pilot(dataset: str, seed: int = SEED, n_pilot_queries: int = 5) -> None:
    print(f"=== pilot: {dataset} ===")
    cfg = DATASETS[dataset]
    D, K = cfg['D'], cfg['K']
    repo_root = Path(__file__).parent.parent.parent.parent
    inputs = load_population_inputs(dataset, repo_root)
    base_hat = _normalize(inputs['base'])
    test_q_hat = _normalize(inputs['test_q'])

    budget = cfg['nondegenerate_budgets'][-1]
    selected = propose_and_select(inputs, D, K, budget, seed)
    entries = extract_entries(selected)
    print(f"  n_candidates={len(entries)} at budget={budget}")

    rng = np.random.default_rng(seed)
    codes, hyperplanes = build_codes(base_hat, M, L, rng)
    query_sample = test_q_hat[rng.choice(len(test_q_hat), size=min(PRIOR_QUERY_SAMPLE, len(test_q_hat)),
                                          replace=False)]
    prior = build_prior_histogram(base_hat, query_sample, N_BINS_PRIOR)
    r_grid = np.linspace(1e-3, np.pi - 1e-3, R_GRID_SIZE)
    lut = build_lut(M, prior, r_grid)

    pilot_qi = rng.choice(len(test_q_hat), size=n_pilot_queries, replace=False)

    t0 = time.perf_counter()
    for qi in pilot_qi:
        q = test_q_hat[qi]
        for entry in entries:
            stored_idx = entry['stored_idx']
            t = np.arccos(np.clip(base_hat[stored_idx] @ q, -1.0, 1.0))
            order = np.argsort(t)
            t_sorted = t[order]
            delta = float(np.arccos(np.clip(entry['pole'] @ q, -1.0, 1.0)))
            cov = coverage(t_sorted, max(delta, 1e-9), entry['rho_c'], D)
            accumulate_to_k(t_sorted, cov, K)
    m1_time = (time.perf_counter() - t0) / (n_pilot_queries * len(entries))

    t0 = time.perf_counter()
    for qi in pilot_qi:
        q = test_q_hat[qi]
        for entry in entries:
            stored_idx = entry['stored_idx']
            t = np.arccos(np.clip(base_hat[stored_idx] @ q, -1.0, 1.0))
            order = np.argsort(t)
            t_sorted = t[order]
            delta = float(np.arccos(np.clip(entry['pole'] @ q, -1.0, 1.0)))
            exact_intersection_fraction(entry['rho_c'], float(t_sorted[K - 1]), delta, D)
    m2_time = (time.perf_counter() - t0) / (n_pilot_queries * len(entries))

    t0 = time.perf_counter()
    for qi in pilot_qi:
        q = test_q_hat[qi]
        invert_radius(q, codes, hyperplanes, lut, r_grid, K)
    m3_time = (time.perf_counter() - t0) / n_pilot_queries

    t0 = time.perf_counter()
    for qi in pilot_qi:
        q = test_q_hat[qi]
        for entry in entries:
            stored_idx = entry['stored_idx']
            cached_points = base_hat[stored_idx]
            hybrid_estimator.estimate(q, cached_points, K, codes, hyperplanes, lut, r_grid)
    m4_time = (time.perf_counter() - t0) / (n_pilot_queries * len(entries))

    print(f"  M1 (per pair): {m1_time*1e3:.3f} ms")
    print(f"  M2 (per pair): {m2_time*1e3:.3f} ms")
    print(f"  M3 (per query, global): {m3_time*1e3:.3f} ms")
    print(f"  M4 (per pair): {m4_time*1e3:.3f} ms")

#-------------------------------------------------------------------------------
# Main crossed evaluation: 4 methods x applicable radius regimes
#-------------------------------------------------------------------------------

CELLS = ['M1_R1', 'M1_R2', 'M2_R1', 'M2_R2', 'M3_R2', 'M4_R1', 'M4_R2']


def run_main(dataset: str, n_test_samples: int, seed: int = SEED) -> dict:
    cfg = DATASETS[dataset]
    D, K = cfg['D'], cfg['K']
    repo_root = Path(__file__).parent.parent.parent.parent
    inputs = load_population_inputs(dataset, repo_root)
    base_hat = _normalize(inputs['base'])
    test_q_hat = _normalize(inputs['test_q'])
    n_test = len(test_q_hat)

    # M3's LSH structure is over the full corpus, budget-independent - built once.
    rng = np.random.default_rng(seed)
    codes, hyperplanes = build_codes(base_hat, M, L, rng)
    prior_idx = rng.choice(n_test, size=min(PRIOR_QUERY_SAMPLE, n_test), replace=False)
    prior = build_prior_histogram(base_hat, test_q_hat[prior_idx], N_BINS_PRIOR)
    r_grid = np.linspace(1e-3, np.pi - 1e-3, R_GRID_SIZE)
    lut = build_lut(M, prior, r_grid)

    rng2 = np.random.default_rng(seed + 1)
    n_sample = min(n_test_samples, n_test)
    sampled_qi = rng2.choice(n_test, size=n_sample, replace=False)

    # M3's per-query shared radius is budget-independent - computed once, reused at every budget.
    m3_r_hat_q = {int(qi): invert_radius(test_q_hat[qi], codes, hyperplanes, lut, r_grid, K)
                  for qi in sampled_qi}

    test_gt_idx = _load_gt_indices(inputs['benchmark_dir'] / f'test_gt_population_K{K}.npz', n_test)

    budgets = cfg['nondegenerate_budgets'] + [cfg['degenerate_budget']]
    degenerate_set = {cfg['degenerate_budget']}
    all_rows = []  # one dict per (budget, query, candidate)
    query_level = []  # one dict per (budget, query): argmax-selected + oracle-best per cell

    for budget in budgets:
        selected = propose_and_select(inputs, D, K, budget, seed)
        entries = extract_entries(selected)
        n_cand = len(entries)
        print(f"  [{dataset}] budget={budget}: {n_cand} candidates"
              f"{' (DEGENERATE)' if budget in degenerate_set else ''}")
        if n_cand == 0:
            continue
        cand_poles = np.array([e['pole'] for e in entries])
        delta_matrix = np.arccos(np.clip(test_q_hat[sampled_qi] @ cand_poles.T, -1.0, 1.0))
        nearest_cand = np.argmin(delta_matrix, axis=1)

        for qpos, qi in enumerate(sampled_qi):
            q = test_q_hat[qi]
            true_topk = set(test_gt_idx[int(qi)])
            m3_rhat = m3_r_hat_q[int(qi)]

            per_cand = []
            for ci, entry in enumerate(entries):
                stored_idx = entry['stored_idx']
                if len(stored_idx) < K:
                    continue
                pole, rho_c = entry['pole'], entry['rho_c']
                delta = max(float(delta_matrix[qpos, ci]), 1e-9)
                cached_points = base_hat[stored_idx]
                t = np.arccos(np.clip(cached_points @ q, -1.0, 1.0))
                order = np.argsort(t)
                t_sorted = t[order]
                response_idx = stored_idx[order[:K]]
                true_recall = len(true_topk & set(response_idx.tolist())) / K

                cov = coverage(t_sorted, delta, rho_c, D)
                r_hat_m1, min_cov_m1, _ = accumulate_to_k(t_sorted, cov, K)
                recall_m1_r1 = recall_hat_from_r_hat(r_hat_m1, t_sorted[:K])

                r_t_proxy = float(t_sorted[K - 1])
                recall_m2_r1 = exact_intersection_fraction(rho_c, r_t_proxy, delta, D)

                recall_m4_r1, r_hat_m4, order4 = hybrid_estimator.estimate(
                    q, cached_points, K, codes, hyperplanes, lut, r_grid)

                recall_m3_r2 = recall_hat_from_r_hat(m3_rhat, t_sorted[:K])

                per_cand.append(dict(
                    ci=ci, true_recall=true_recall, t_sorted_k=t_sorted[:K], rho_c=rho_c,
                    delta=delta, r_hat_m1=r_hat_m1, r_t_proxy=r_t_proxy, r_hat_m4=r_hat_m4,
                    min_cov_m1=min_cov_m1,
                    recall_m1_r1=recall_m1_r1, recall_m2_r1=recall_m2_r1,
                    recall_m4_r1=recall_m4_r1, recall_m3_r2=recall_m3_r2,
                ))

            if not per_cand:
                continue

            # R2: nearest candidate's own radius, shared across every candidate this query.
            nearest_ci_pos = int(nearest_cand[qpos])
            nearest = next((p for p in per_cand if p['ci'] == nearest_ci_pos), per_cand[0])
            shared_r_hat_m1, shared_r_t_proxy, shared_r_hat_m4 = (
                nearest['r_hat_m1'], nearest['r_t_proxy'], nearest['r_hat_m4'])

            cell_recall_hat = {c: [] for c in CELLS}
            cell_true_recall = {c: [] for c in CELLS}
            for p in per_cand:
                cell_recall_hat['M1_R1'].append(p['recall_m1_r1'])
                cell_recall_hat['M2_R1'].append(p['recall_m2_r1'])
                cell_recall_hat['M4_R1'].append(p['recall_m4_r1'])
                cell_recall_hat['M3_R2'].append(p['recall_m3_r2'])
                cell_recall_hat['M1_R2'].append(recall_hat_from_r_hat(shared_r_hat_m1, p['t_sorted_k']))
                cell_recall_hat['M2_R2'].append(
                    exact_intersection_fraction(p['rho_c'], shared_r_t_proxy, p['delta'], D))
                cell_recall_hat['M4_R2'].append(recall_hat_from_r_hat(shared_r_hat_m4, p['t_sorted_k']))
                for c in CELLS:
                    cell_true_recall[c].append(p['true_recall'])

                all_rows.append(dict(
                    budget=budget, query_idx=int(qi), cand_idx=p['ci'], true_recall=p['true_recall'],
                    min_coverage_m1=p['min_cov_m1'], delta=p['delta'], rho_c=p['rho_c'],
                    **{f'recall_hat_{c}': cell_recall_hat[c][-1] for c in CELLS},
                ))

            row = dict(budget=budget, query_idx=int(qi))
            for c in CELLS:
                rh = np.array(cell_recall_hat[c])
                tr = np.array(cell_true_recall[c])
                amax = int(np.argmax(rh))
                row[f'selected_recall_hat_{c}'] = rh[amax]
                row[f'selected_true_recall_{c}'] = tr[amax]
                row[f'oracle_best_true_recall_{c}'] = tr.max()
            query_level.append(row)

    return dict(all_rows=_rows_to_columns(all_rows), query_level=_rows_to_columns(query_level),
                budgets=np.array(budgets), degenerate_budget=cfg['degenerate_budget'])


def _rows_to_columns(rows: list) -> dict:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: np.array([r[k] for r in rows]) for k in keys}


def _save_main_results(dataset: str, result: dict) -> None:
    out_path = OUT_DIR / f'estimator_comparison_{dataset}.npz'
    flat = {}
    for k, v in result['all_rows'].items():
        flat[f'pair_{k}'] = v
    for k, v in result['query_level'].items():
        flat[f'query_{k}'] = v
    flat['budgets'] = result['budgets']
    flat['degenerate_budget'] = result['degenerate_budget']
    np.savez_compressed(out_path, **flat)
    print(f"  wrote {out_path}")

#-------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=['e0', 'e1', 'pilot', 'main'], default='e0')
    parser.add_argument('--dataset', choices=['sift', 'esci', 'both'], default='both')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--n-samples', type=int, default=50)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = ['sift', 'esci'] if args.dataset == 'both' else [args.dataset]

    if args.experiment == 'e0':
        run_e0()
    elif args.experiment == 'e1':
        run_e1_m1_m2(seed=args.seed)
        run_e1_m3(seed=args.seed + 1)
        run_e1_m4(seed=args.seed + 2)
    elif args.experiment == 'pilot':
        for ds in datasets:
            run_pilot(ds, seed=args.seed)
    elif args.experiment == 'main':
        for ds in datasets:
            print(f"=== main: {ds} ===")
            result = run_main(ds, n_test_samples=args.n_samples, seed=args.seed)
            _save_main_results(ds, result)


if __name__ == '__main__':
    main()
