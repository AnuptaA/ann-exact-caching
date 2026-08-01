#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Shared population pipeline logic for SIFT and ESCI benchmarks."""

import sys
import json
import csv
import logging
import time
import tracemalloc
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from tqdm import tqdm

from simulations.core.simulate import CacheSimulator, SimulationResult
from utils.base import KVCache
from utils.base.selection import select_candidates

#-------------------------------------------------------------------------------

logger = logging.getLogger(__name__)

#-------------------------------------------------------------------------------

def _timed_stage(on_stage: Optional[Callable[[str, float, float], None]], name: str,
                  fn: Callable, *args, **kwargs):
    """Run fn under time.perf_counter()/tracemalloc and report to on_stage(name,
    elapsed_seconds, peak_mb) if given; otherwise just run fn with no overhead
    beyond the timing itself."""
    if on_stage is None:
        return fn(*args, **kwargs)
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    on_stage(name, elapsed, peak / 1e6)
    return result

#-------------------------------------------------------------------------------

def _load_cache_gt_arrays(
    benchmark_dir: Path,
    K: int,
    n_cache: int,
    base_vectors: np.ndarray,
) -> tuple:
    data = np.load(benchmark_dir / f"cache_gt_population_K{K}.npz")
    tv_all, ti_all, hg_all = [], [], []
    for i in range(n_cache):
        indices = data[f'indices_{i}'].tolist()
        tv_all.append([base_vectors[idx] for idx in indices])
        ti_all.append(indices)
        hg_all.append(float(data[f'half_gap_{i}']))
    return tv_all, ti_all, np.array(hg_all, dtype=np.float32)


def _load_gt_indices(path: Path, n: int) -> list:
    """Load just the ground-truth index lists from a *_gt_population_K{K}.npz file."""
    data = np.load(path)
    return [data[f'indices_{i}'].tolist() for i in range(n)]


def _run_simulation(
    kv_cache,
    base_vectors: np.ndarray,
    test_queries: np.ndarray,
    K: int,
    algorithms: list,
    benchmark_name: str,
    strategy: str,
    budget_B: int,
    eval_mode: str,
    metric: str,
    sigma: float = 1.0,
    test_gt_indices: Optional[list] = None,
    r_t_hat: Optional[float] = None,
    overlap_threshold: float = 0.9,
) -> list:
    """r_t_hat/overlap_threshold: only used by algorithm="overlap_threshold"
    (see utils.base.lemmas_sigma.overlap_threshold_hit_policy); every other
    algorithm ignores them."""
    simulator = CacheSimulator(
        base_vectors=base_vectors,
        cache_queries=test_queries[:1],
        test_queries=test_queries,
        K=K,
        N=K,
        metric=metric,
        r_t_hat=r_t_hat,
        overlap_threshold=overlap_threshold,
    )
    simulator.cache = kv_cache

    n_test = len(test_queries)
    all_results = []
    for algorithm in algorithms:
        cache_hits = 0
        total_distance_calcs = 0
        total_time_us = 0.0
        lemma1_hits = 0
        lemma2_hits = 0
        recall_sum = 0.0
        n_hits_with_recall = 0
        query_results = []

        for i, query in enumerate(tqdm(
            test_queries, desc=f"  [{eval_mode}] {algorithm:>20}", file=sys.stderr
        )):
            result = simulator.run_query(
                i, query, algorithm, ground_truth_indices=[], sigma=sigma
            )
            query_results.append(result)

            if result.cache_hit:
                cache_hits += 1
                if result.hit_source == 'lemma1':
                    lemma1_hits += 1
                elif result.hit_source == 'lemma2':
                    lemma2_hits += 1
                # K=0 (containment mode): there's no K-sized target to recall
                # a fraction of, but containment alone is the full task at
                # K=0, so a hit counts as recall=1.0 (not skipped/undefined).
                if K == 0:
                    recall_sum += 1.0
                    n_hits_with_recall += 1
                elif (
                    result.certified_indices is not None
                    and test_gt_indices is not None
                ):
                    true_gt = set(test_gt_indices[i])
                    recall_sum += len(set(result.certified_indices) & true_gt) / K
                    n_hits_with_recall += 1
            if result.distance_calculations is not None:
                total_distance_calcs += result.distance_calculations
            if result.time_us is not None:
                total_time_us += result.time_us

        mean_recall = recall_sum / n_hits_with_recall if n_hits_with_recall > 0 else 0.0

        sim_result = SimulationResult(
            dataset_name=f'{benchmark_name}/B{budget_B}/{eval_mode}/{strategy}',
            algorithm=algorithm,
            metric=metric,
            total_queries=n_test,
            cache_hits=cache_hits,
            correct_results=0,
            total_distance_calcs=total_distance_calcs,
            total_time_us=total_time_us,
            lemma1_hits=lemma1_hits,
            lemma2_hits=lemma2_hits,
            sigma=sigma,
            mean_recall=mean_recall,
            query_results=query_results,
        )
        all_results.append(sim_result)

        hit_rate = sim_result.hit_rate
        avg_dc = f"{sim_result.avg_distance_calcs:.1f}" if sim_result.avg_distance_calcs else "N/A"
        recall_str = f"  |  Mean recall: {mean_recall:.2%}" if n_hits_with_recall > 0 else ""
        print(f"  [{eval_mode}] Hit rate: {hit_rate:.2%}  |  Avg dist calcs: {avg_dc}{recall_str}")

    return all_results


def _write_outputs(
    out_base: Path,
    benchmark_name: str,
    strategy: str,
    config: dict,
    all_results: list,
    algorithms: list,
    metric: str,
) -> None:
    logs_dir = out_base / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)

    for sim_result in all_results:
        algo = sim_result.algorithm
        algo_dict = sim_result.to_dict()
        algo_dict.pop('query_results', None)
        with open(out_base / f"{algo}_{metric}.json", 'w') as f:
            json.dump(algo_dict, f, indent=2)

        log_file = logs_dir / f"{algo}_{metric}.log"
        with open(log_file, 'w') as lf:
            for qr in sim_result.query_results:
                if qr.cache_hit:
                    lf.write(
                        f"Query {qr.query_id}: HIT via {qr.hit_source} | "
                        f"dist_calcs={qr.distance_calculations}\n"
                    )
                else:
                    err = f" | error={qr.error}" if qr.error else ""
                    lf.write(
                        f"Query {qr.query_id}: MISS{err} | "
                        f"dist_calcs={qr.distance_calculations}\n"
                    )

    with open(out_base / 'per_query.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'algorithm', 'metric', 'query_id',
                         'cache_hit', 'hit_source', 'distance_calcs', 'time_us'])
        for sr in all_results:
            for qr in (sr.query_results or []):
                writer.writerow([
                    sr.dataset_name,
                    sr.algorithm,
                    sr.metric,
                    qr.query_id,
                    int(qr.cache_hit),
                    qr.hit_source or '',
                    qr.distance_calculations,
                    round(qr.time_us, 3) if qr.time_us is not None else None,
                ])

    summary = {
        "dataset": benchmark_name,
        "strategy": strategy,
        "config": config,
        "results": [r.to_dict() for r in all_results],
    }
    for r in summary["results"]:
        r.pop('query_results', None)
    with open(out_base / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    with open(logs_dir / 'run.log', 'w') as out_log:
        for algo in algorithms:
            log_file = logs_dir / f"{algo}_{metric}.log"
            if log_file.exists():
                out_log.write(f"--- {algo} / {metric} ---\n")
                out_log.write(log_file.read_text())
                out_log.write("\n")


def _write_selection_log(
    out_path: Path,
    selected: list,
    trajectory: list,
) -> None:
    log = []
    cached_before = np.zeros(len(selected[0].R_c), dtype=bool) if selected else None

    # min_dist_to_prev is computed via one vectorized batch (this step's
    # query against the matrix of all previously-selected queries) instead
    # of a Python-level angular_distance() call per (step, prior-step) pair.
    # Same O(n^2) total distance evaluations, but via BLAS matmuls instead
    # of up to ~n^2/2 individual Python-level calls -- at budget_fraction=1.0
    # on real data (~13,855 selected entries), the latter took ~900s of
    # untimed wall-clock (not visible in any profiled stage), dwarfing every
    # other stage combined. Numerically equivalent to angular_distance
    # (same dot-product/norm/arccos formula, including its zero-norm
    # fallback to distance 1.0); only used for a rounded (6-decimal),
    # informational log field, not selection/simulation output.
    if selected:
        queries_matrix = np.array([c.query for c in selected])
        query_norms = np.linalg.norm(queries_matrix, axis=1)
        query_norms_safe = np.where(query_norms == 0.0, 1.0, query_norms)

    for step, (cand, traj_entry) in enumerate(zip(selected, trajectory)):
        new_vecs = int(np.sum(cand.R_c & ~cached_before))
        cached_before |= cand.R_c
        rc_size = cand.rc_size
        overlap_pct = round((rc_size - new_vecs) / max(rc_size, 1), 4)
        if step > 0:
            dots = queries_matrix[:step] @ queries_matrix[step]
            denom = query_norms_safe[:step] * query_norms_safe[step]
            cosine_sim = np.clip(dots / denom, -1.0, 1.0)
            dists = np.arccos(cosine_sim) / np.pi
            zero_mask = (query_norms[:step] == 0.0) | (query_norms[step] == 0.0)
            if np.any(zero_mask):
                dists = np.where(zero_mask, 1.0, dists)
            min_dist = round(float(dists.min()), 6)
        else:
            min_dist = 0.0
        log.append({
            "step": step + 1,
            "alpha": round(float(cand.alpha), 4) if cand.alpha is not None else None,
            "r_Q": round(float(cand.r_Q), 6),
            "r_c": round(float(cand.r_c), 6),
            "vectors_in_R_c": rc_size,
            "new_vectors_added": new_vecs,
            "overlap_pct": overlap_pct,
            "min_dist_to_prev": min_dist,
            "queries_covered_total": int(cand.covered_train_indices.sum()),
            "new_queries_covered": traj_entry.get("new_train_coverage", 0),
            "cig_coverage": round(traj_entry["cig_train_coverage"], 6),
        })
    with open(out_path, 'w') as f:
        json.dump(log, f, indent=2)


def run_population_benchmark(
    benchmark_name: str,
    load_benchmark_fn: Callable,
    metric: str,
    output_dir: Path,
    benchmark_base_dir: Path,
    propose_fn: Callable,
    budget_values: list,
    strategies: list,
    algorithms: list,
    seed: int,
    run_ts: str = None,
    override_K: int = None,
    k_test_override: Optional[int] = None,
    sigma_values: Optional[list] = None,
    on_stage: Optional[Callable[[str, float, float], None]] = None,
) -> None:
    """Run the full population pipeline for a single benchmark.

    propose_fn is called with:
        (obs_queries, obs_r_Q, obs_tv, obs_ti, obs_hg,
         train_queries, train_r_t, test_queries, test_r_t,
         base_vectors, metric)
    and must return List[CandidateEntry].

    k_test_override=0 handles the containment-only case: cache is populated at
    K=1 but test_r_t is zeroed out so the hit condition is pure containment.
    For k_test_override >= 1 or None, override_K is used as before.

    sigma_values: recall thresholds to simulate at (only algorithm='lemma1_no_union'
        or 'lemma2_no_union' support sigma != 1.0). Candidate proposal and selection
        run once per (budget, strategy) regardless of sigma_values; only the
        simulation step repeats per sigma. Defaults to [1.0] (exact, non-relaxed
        behavior), in which case output paths are unchanged from the single-sigma
        layout.
        For any other sigma_values, oracle/non_oracle output nests one level
        deeper under 'sigma_{value:.2f}/'.

    on_stage: optional callback(stage_name, elapsed_seconds, peak_mb), called
        after each major stage (benchmark load, GT array load, candidate
        proposal, and per-budget select/build-cache/oracle-simulate/
        non_oracle-simulate) for profiling. No effect on behavior when None.
    """
    sigma_values = sigma_values if sigma_values else [1.0]
    single_default_sigma = sigma_values == [1.0]
    if run_ts is None:
        from datetime import datetime
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    benchmark_dir = benchmark_base_dir / benchmark_name

    if not (benchmark_dir / 'cache_r_Q.npy').exists():
        logger.warning(
            f"Skipping {benchmark_name}: run prepare_benchmark.py --extend-population first"
        )
        return

    print(f"\n{'='*80}")
    print(f"POPULATION BENCHMARK: {benchmark_name}")
    print(f"{'='*80}")

    def _load_benchmark_and_resolve_K():
        base, cache_q, test_q, config = load_benchmark_fn(str(benchmark_dir))
        config_K = config['cache_K']
        n_cache = len(cache_q)
        n_test = len(test_q)

        # Determine K for cache GT files and test radii
        if k_test_override == 0:
            # Containment-only: cache entries still populated at K=1 (real
            # top-1 vectors/half-gaps needed for entry construction), but
            # obs_r_Q -- also passed as train_r_t at the propose_fn call
            # site below -- is zeroed, symmetric with test_r_t. This makes
            # the coverage formula pure distance containment at K=0,
            # matching propose_candidates_ball's K=0 margin (also 0) and
            # simulate-time's K=0 hit condition (see lemmas_sigma.py).
            # obs_r_Q has no other functional use: to_cache_entry() builds
            # the real CacheEntry from r_c, never observed_r_Q, which is
            # otherwise only an informational debug-log field.
            cache_load_K = 1
            obs_r_Q = np.zeros(n_cache, dtype=np.float32)
            test_r_t = np.zeros(n_test, dtype=np.float32)
            target_K = 0
        else:
            target_K = override_K if override_K is not None else config_K
            cache_load_K = target_K
            if target_K != config_K:
                gt_k = np.load(benchmark_dir / f'cache_gt_population_K{target_K}.npz')
                obs_r_Q = np.array(
                    [float(gt_k[f'r_Q_{i}']) for i in range(n_cache)], dtype=np.float32
                )
                test_gt_k = np.load(benchmark_dir / f'test_gt_population_K{target_K}.npz')
                test_r_t = np.array(
                    [float(test_gt_k[f'r_Q_{i}']) for i in range(n_test)], dtype=np.float32
                )
            else:
                obs_r_Q = np.load(benchmark_dir / 'cache_r_Q.npy')
                test_r_t = np.load(benchmark_dir / 'test_r_t.npy')

        return base, cache_q, test_q, config, config_K, n_cache, n_test, obs_r_Q, test_r_t, cache_load_K, target_K

    (base, cache_q, test_q, config, config_K, n_cache, n_test,
     obs_r_Q, test_r_t, cache_load_K, target_K) = _timed_stage(
        on_stage, "load_benchmark", _load_benchmark_and_resolve_K
    )

    K = target_K
    obs_queries = cache_q
    obs_n = n_cache

    logger.info(f"Loading cache GT arrays ({n_cache} entries, K={cache_load_K})...")
    obs_tv, obs_ti, obs_hg = _timed_stage(
        on_stage, "load_cache_gt_arrays",
        _load_cache_gt_arrays, benchmark_dir, cache_load_K, n_cache, base,
    )

    # Ground truth for mean_recall (lemma1 sigma runs only; harmless to load otherwise).
    test_gt_path = (
        benchmark_dir / 'test_gt_population_K1.npz' if k_test_override == 0
        else benchmark_dir / f'test_gt_population_K{target_K}.npz'
    )
    test_gt_indices = _load_gt_indices(test_gt_path, n_test) if test_gt_path.exists() else None

    logger.info(f"Proposing candidates ({obs_n} queries)...")
    candidates = _timed_stage(
        on_stage, "propose_candidates",
        propose_fn,
        obs_queries, obs_r_Q, obs_tv, obs_ti, obs_hg,
        obs_queries, obs_r_Q,
        test_q, test_r_t,
        base, metric,
    )
    logger.info(f"Proposed {len(candidates)} candidates")

    effective_name = (
        benchmark_name.replace(f'k{config_K}', f'k{target_K}')
        if target_K != config_K else benchmark_name
    )

    for budget_B in budget_values:
        print(f"\n{'='*60}")
        print(f"Budget B={budget_B}")
        print(f"{'='*60}")

        b_dir = output_dir / run_ts / effective_name / f'B{budget_B}'

        for strategy in strategies:
            print(f"\n{'-'*40}")
            print(f"Strategy: {strategy}")
            print(f"{'-'*40}")

            selected, trajectory = _timed_stage(
                on_stage, f"select_candidates(B={budget_B})",
                select_candidates,
                candidates, budget_B, strategy,
                n_train_queries=obs_n,
                n_test_queries=n_test,
                seed=seed,
            )
            logger.info(f"B={budget_B} {strategy}: {len(selected)} entries, "
                        f"trajectory length {len(trajectory)}")

            traj_file = b_dir / f'{strategy}_trajectory.json'
            traj_file.parent.mkdir(parents=True, exist_ok=True)
            with open(traj_file, 'w') as f:
                json.dump(trajectory, f, indent=2)

            sel_file = b_dir / f'{strategy}_selection.json'
            _write_selection_log(sel_file, selected, trajectory)

            def _build_kv_cache():
                kv_cache = KVCache(metric=metric)
                for cand in selected:
                    entry = cand.to_cache_entry()
                    kv_cache.add_entry(
                        entry.query, entry.top_k_vectors,
                        entry.r_Q, entry.half_gap, entry.top_k_indices,
                        entry.top_k_matrix,
                    )
                return kv_cache

            kv_cache = _timed_stage(on_stage, f"build_kv_cache(B={budget_B})", _build_kv_cache)
            assert kv_cache.size() == len(selected), "KVCache size mismatch"

            # K=0 (containment mode) flows through unchanged: the CIG
            # algorithm's existing O1 gate + len(passing) >= threshold check
            # already reduces to "hit iff contained" at threshold=0, with no
            # special-casing needed (see the comment at that check in
            # lemmas_sigma.py). No clamp to K >= 1 required.
            sim_K = K

            for sigma in sigma_values:
                sigma_dir = b_dir if single_default_sigma else b_dir / f'sigma_{sigma:.2f}'

                oracle_results = _timed_stage(
                    on_stage, f"simulate_oracle(B={budget_B},sigma={sigma})",
                    _run_simulation,
                    kv_cache, base, obs_queries, sim_K, algorithms,
                    effective_name, strategy, budget_B, 'oracle', metric, sigma=sigma,
                    test_gt_indices=obs_ti,
                )
                out_oracle = sigma_dir / 'oracle' / strategy
                _write_outputs(out_oracle, effective_name, strategy, config,
                               oracle_results, algorithms, metric)

                non_oracle_results = _timed_stage(
                    on_stage, f"simulate_non_oracle(B={budget_B},sigma={sigma})",
                    _run_simulation,
                    kv_cache, base, test_q, sim_K, algorithms,
                    effective_name, strategy, budget_B, 'non_oracle', metric, sigma=sigma,
                    test_gt_indices=test_gt_indices,
                )
                out_non_oracle = sigma_dir / 'non_oracle' / strategy
                _write_outputs(out_non_oracle, effective_name, strategy, config,
                               non_oracle_results, algorithms, metric)

                logger.info(f"Results written to {sigma_dir}")
