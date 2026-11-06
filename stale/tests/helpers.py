#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Shared test-fixture helpers for creating mini SIFT/ESCI benchmarks.

Datasets differ only in vector dimension and which KNN method mirrors their
real prepare_benchmark.py pipeline (SIFT uses MainMemory.top_k_search; ESCI
uses a brute-force matmul, matching datasets/esci/prepare_benchmark.py).
"""

import sys
import json
import tempfile
import shutil
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from utils.base import MainMemory

#-------------------------------------------------------------------------------

POPULATION_K = 10

DATASET_CONFIG = {
    'sift': dict(dimension=128, dataset_label='test_mini'),
    'esci': dict(dimension=384, dataset_label='ESCI_test_mini'),
}

#-------------------------------------------------------------------------------
# KNN methods (one per dataset, mirroring each dataset's real prepare_benchmark.py)

def _brute_force_knn_angular(queries: np.ndarray, base: np.ndarray, K: int) -> list:
    """Brute-force top-K angular KNN. Returns list of (indices, distances, gap)."""
    q = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    b = base / np.linalg.norm(base, axis=1, keepdims=True)
    kp1 = min(K + 1, len(base))
    results = []
    for i in range(len(queries)):
        scores = q[i] @ b.T
        top_idx = np.argsort(-scores)[:kp1]
        top_scores = scores[top_idx]
        dists = (np.arccos(np.clip(top_scores, -1.0, 1.0)) / np.pi).astype(np.float32)
        idx = top_idx[:K]
        d = dists[:K]
        gap = float(dists[K] - dists[K - 1]) if kp1 > K else 0.0
        results.append((idx, d, gap))
    return results


def _mainmemory_knn_angular(queries: np.ndarray, base: np.ndarray, K: int) -> list:
    """MainMemory.top_k_search-based top-K angular KNN."""
    mm = MainMemory(vectors=list(base))
    results = []
    for q in queries:
        _, top_k_dists, gap, indices = mm.top_k_search(
            q, k=K, metric='angular', return_indices=True
        )
        results.append((np.array(indices), top_k_dists, gap))
    return results


_KNN_FN = {
    'sift': _mainmemory_knn_angular,
    'esci': _brute_force_knn_angular,
}

#-------------------------------------------------------------------------------


def _write_gt(knn_fn, queries: np.ndarray, base: np.ndarray, K: int,
              benchmark_dir: Path, filename: str) -> dict:
    results = knn_fn(queries, base, K)
    save_dict = {}
    for i, (idx, d, gap) in enumerate(results):
        save_dict[f'indices_{i}'] = idx
        save_dict[f'r_Q_{i}'] = np.float32(d[-1])
        save_dict[f'half_gap_{i}'] = np.float32(gap / 2.0)
    np.savez(benchmark_dir / filename, **save_dict)
    return save_dict


def _write_base_gt_esci(cache_queries: np.ndarray, base: np.ndarray, K: int,
                         benchmark_dir: Path) -> None:
    """Write cache_gt_K{K}.npz in ESCI's base-pipeline format: full per-entry
    vectors/distances/gap (not just the K-th distance), matching
    datasets/esci/prepare_benchmark.py's _build_cache_gt output."""
    results = _brute_force_knn_angular(cache_queries, base, K)
    save_dict = {}
    for i, (idx, d, gap) in enumerate(results):
        save_dict[f'indices_{i}'] = idx
        save_dict[f'vectors_{i}'] = base[idx]
        save_dict[f'distances_{i}'] = d
        save_dict[f'gap_{i}'] = np.float32(gap)
    np.savez_compressed(benchmark_dir / f'cache_gt_K{K}.npz', **save_dict)


def create_mini_benchmark(dataset: str, benchmark_dir: Path, seed: int = 42) -> dict:
    """Create a minimal dataset-style benchmark for testing.

    Files created:
      base_vectors.npy, cache_queries.npy, test_queries.npy,
      cache_gt_population_K{K}.npz, config.json
    """
    cfg = DATASET_CONFIG[dataset]
    knn_fn = _KNN_FN[dataset]
    np.random.seed(seed)

    dimension = cfg['dimension']
    num_base = 100
    num_cache = 20
    num_test = 10
    K = POPULATION_K
    N = 5

    base_vectors = np.random.randn(num_base, dimension).astype(np.float32)
    base_vectors /= np.linalg.norm(base_vectors, axis=1, keepdims=True)

    cache_queries = np.random.randn(num_cache, dimension).astype(np.float32)
    cache_queries /= np.linalg.norm(cache_queries, axis=1, keepdims=True)

    test_queries = np.random.randn(num_test, dimension).astype(np.float32)
    test_queries /= np.linalg.norm(test_queries, axis=1, keepdims=True)

    benchmark_dir.mkdir(parents=True, exist_ok=True)
    np.save(benchmark_dir / 'base_vectors.npy', base_vectors)
    np.save(benchmark_dir / 'cache_queries.npy', cache_queries)
    np.save(benchmark_dir / 'test_queries.npy', test_queries)

    _write_gt(knn_fn, cache_queries, base_vectors, K, benchmark_dir,
              f'cache_gt_population_K{K}.npz')

    if dataset == 'esci':
        # ESCI's base (non-population) hit-policy pipeline (run_esci.py) reads a
        # separate cache_gt_K{K}.npz with full per-entry vectors/distances/gap,
        # matching datasets/esci/prepare_benchmark.py's base GT format.
        _write_base_gt_esci(cache_queries, base_vectors, K, benchmark_dir)

    config = {
        'dataset': cfg['dataset_label'],
        'benchmark_name': benchmark_dir.name,
        'num_base_vectors': num_base,
        'num_cache_queries': num_cache,
        'num_test_queries': num_test,
        'cache_K': K,
        'test_N': N,
        'dimension': dimension,
        'seed': seed,
        'description': (
            f'Mini {dataset} test benchmark: {num_base} base, '
            f'{num_cache} cache (K={K}), {num_test} test (N={N})'
        ),
    }
    with open(benchmark_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    return config


def create_mini_benchmark_with_population_data(
    dataset: str, benchmark_dir: Path, seed: int = 42,
) -> dict:
    """create_mini_benchmark() plus population-pipeline extras:
    cache_r_Q.npy, test_gt_population_K{K}.npz, test_r_t.npy,
    cache_gt_population_K{1,5,20}.npz (K-microbenchmark tests),
    test_gt_population_K{1,5,10}.npz (containment microbenchmark sweep).
    Patches config.json with has_population_data=true.
    """
    knn_fn = _KNN_FN[dataset]
    config = create_mini_benchmark(dataset, benchmark_dir, seed=seed)
    K = config['cache_K']
    num_cache = config['num_cache_queries']
    num_test = config['num_test_queries']

    base = np.load(benchmark_dir / 'base_vectors.npy')
    cache_q = np.load(benchmark_dir / 'cache_queries.npy')
    test_q = np.load(benchmark_dir / 'test_queries.npy')

    cache_gt = np.load(benchmark_dir / f'cache_gt_population_K{K}.npz')
    r_Q = np.array(
        [float(cache_gt[f'r_Q_{i}']) for i in range(num_cache)], dtype=np.float32,
    )
    np.save(benchmark_dir / 'cache_r_Q.npy', r_Q)

    test_gt_data = _write_gt(knn_fn, test_q, base, K, benchmark_dir,
                              f'test_gt_population_K{K}.npz')
    r_t = np.array(
        [float(test_gt_data[f'r_Q_{i}']) for i in range(num_test)], dtype=np.float32,
    )
    np.save(benchmark_dir / 'test_r_t.npy', r_t)

    for k in [1, 5]:
        _write_gt(knn_fn, cache_q, base, k, benchmark_dir, f'cache_gt_population_K{k}.npz')
    _write_gt(knn_fn, cache_q, base, 20, benchmark_dir, 'cache_gt_population_K20.npz')
    for k in [1, 5, 10]:
        _write_gt(knn_fn, test_q, base, k, benchmark_dir, f'test_gt_population_K{k}.npz')

    config['has_population_data'] = True
    with open(benchmark_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    return config

#-------------------------------------------------------------------------------


def make_clustered_candidate_benchmark(
    seed: int = 0,
    dim: int = 16,
    n_clusters: int = 5,
    n_base_per_cluster: int = 16,
    n_cache_per_cluster: int = 2,
    n_test_per_cluster: int = 2,
    K: int = 2,
    noise_scale: float = 0.05,
) -> dict:
    """Clustered benchmark for utils.base.population's candidate-proposal tests
    (propose_candidates/_point/_ball, selection.py). Guarantees the CIG
    condition is satisfiable: cluster centers keep inter-cache distances much
    smaller than inter-cluster distances, so K-th base distances stay small.

    Returns a dict with base/cache/test vectors, per-cache-query K-NN ground
    truth (tv/ti/r_Q/hg), and per-test-query r_t, plus n_base/n_cache/n_test.
    """
    rng = np.random.default_rng(seed)

    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    def _cluster_vectors(n_per_cluster):
        parts = []
        for c in centers:
            vecs = c + rng.standard_normal((n_per_cluster, dim)).astype(np.float32) * noise_scale
            parts.append(vecs)
        vectors = np.vstack(parts)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors

    base = _cluster_vectors(n_base_per_cluster)
    cache = _cluster_vectors(n_cache_per_cluster)
    test = _cluster_vectors(n_test_per_cluster)

    def _knn(queries):
        q = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        sims = q @ base.T
        top_idx = np.argsort(-sims, axis=1)[:, :K]
        dists = np.arccos(np.clip(sims, -1.0, 1.0)) / np.pi
        r_Q = np.array([dists[i, top_idx[i, K - 1]] for i in range(len(queries))], dtype=np.float32)
        tv = [[base[j] for j in top_idx[i]] for i in range(len(queries))]
        ti = [top_idx[i].tolist() for i in range(len(queries))]
        hg = np.zeros(len(queries), dtype=np.float32)
        return tv, ti, r_Q, hg

    tv, ti, r_Q, hg = _knn(cache)
    _, _, test_r_t, _ = _knn(test)

    return dict(
        base=base, cache=cache, test=test,
        tv=tv, ti=ti, r_Q=r_Q, hg=hg, test_r_t=test_r_t,
        n_base=len(base), n_cache=len(cache), n_test=len(test),
    )

#-------------------------------------------------------------------------------


def check_selection_invariants(selected, trajectory, budget_B: int, label: str = "") -> None:
    """Shared invariant checks for utils.base.selection.select_candidates output:
    budget respected, trajectory length matches selection, monotone non-decreasing
    coverage/vectors_cached, and cig_train_coverage stays in [0, 1]. Used across
    tests/utils/test_selection.py and test_selection_with_new_candidates.py.
    """
    # budget invariant
    if selected:
        cached = np.zeros(len(selected[0].R_c), dtype=bool)
        for c in selected:
            cached |= c.R_c
        n_cached = int(cached.sum())
    else:
        n_cached = 0
    assert n_cached <= budget_B, \
        f"{label}: cached={n_cached} > budget_B={budget_B}"

    # trajectory length
    assert len(trajectory) == len(selected), \
        f"{label}: trajectory length {len(trajectory)} != selected length {len(selected)}"

    # monotone non-decreasing
    for i in range(1, len(trajectory)):
        assert trajectory[i]['cig_train_coverage'] >= trajectory[i - 1]['cig_train_coverage'] - 1e-9, \
            f"{label}: cig_train_coverage not monotone at step {i}"
        assert trajectory[i]['vectors_cached'] >= trajectory[i - 1]['vectors_cached'], \
            f"{label}: vectors_cached not monotone at step {i}"

    # cig_train_coverage in [0, 1]
    for t in trajectory:
        assert 0.0 <= t['cig_train_coverage'] <= 1.0 + 1e-9, \
            f"{label}: cig_train_coverage {t['cig_train_coverage']} out of range"

#-------------------------------------------------------------------------------


def get_temp_dir(prefix: str = "test_") -> Path:
    """Create and return a temporary directory."""
    return Path(tempfile.mkdtemp(prefix=prefix))


def cleanup_temp_dir(temp_dir: Path) -> None:
    """Clean up temporary directory."""
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
