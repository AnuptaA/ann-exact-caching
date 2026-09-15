#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""E2 (fixed-table bias vs L), E3 (cost/accuracy frontier), E4
(scaling), E6 (memory) for the Horvitz-Thompson LSH estimator."""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'common'))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import numpy as np
from tqdm import tqdm

from datasets.dataloaders import load_fvecs
from lsh_horvitz_thompson import (
    build_tables, query_retrieval, estimate_count, estimate_count_batch,
    make_anisotropic_basis, sample_anisotropic,
)
from part2_uniform_baseline_run_experiment import _sample_uniform_sphere, _bruteforce_knn, _chord_to_angle

#-------------------------------------------------------------------------------

N = 50000
K = 100
N_QUERIES = 300
SEED = 42
ANISO_EPS = 0.05

M_VALUES = [14, 17, 20, 24, 28, 32]
X_VALUES = [1, 2, 3, 4]
L_VALUES = [1, 4, 16, 64]

OUT_DIR = Path(__file__).parent.parent / 'results' / 'lsh_ht'

#-------------------------------------------------------------------------------


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vecs / norms


def _direct_radii(points: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    _, idx = _bruteforce_knn(points, queries, k)
    kth_points = points[idx[:, k - 1]]
    dots = np.clip(np.einsum('ij,ij->i', queries, kth_points), -1.0, 1.0)
    return np.arccos(dots)

#-------------------------------------------------------------------------------


def dataset_sift(n_points: int = N, n_queries: int = N_QUERIES, seed: int = SEED) -> dict:
    repo_root = Path(__file__).parent.parent.parent.parent
    points = _normalize(np.load(repo_root / 'datasets/sift/base_vectors_50k.npy').astype(np.float64))
    if n_points < N:
        points = points[:n_points]
    all_queries = _normalize(load_fvecs(str(repo_root / 'datasets/sift/sift_query.fvecs')).astype(np.float64))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_queries), size=n_queries, replace=False)
    return {'points': points, 'queries': all_queries[idx]}


def dataset_esci(n_points: int = N, n_queries: int = N_QUERIES, seed: int = SEED) -> dict:
    repo_root = Path(__file__).parent.parent.parent.parent
    points = _normalize(np.load(repo_root / 'datasets/esci/base_vectors_50k.npy').astype(np.float64))
    if n_points < N:
        points = points[:n_points]
    query_data = np.load(repo_root / 'datasets/esci/queries.npz')
    all_queries = _normalize(query_data['query_vectors'].astype(np.float64))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_queries), size=n_queries, replace=False)
    return {'points': points, 'queries': all_queries[idx]}


def dataset_isotropic(d: int, n_points: int = N, n_queries: int = N_QUERIES, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    return {'points': _sample_uniform_sphere(n_points, d, rng),
            'queries': _sample_uniform_sphere(n_queries, d, rng)}


def dataset_anisotropic(d: int, r_eff: int, n_points: int = N, n_queries: int = N_QUERIES,
                         seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    basis = make_anisotropic_basis(d, r_eff, rng)
    return {'points': sample_anisotropic(n_points, basis, ANISO_EPS, rng),
            'queries': sample_anisotropic(n_queries, basis, ANISO_EPS, rng)}


E3_DATASETS = {
    'sift': lambda: dataset_sift(),
    'esci': lambda: dataset_esci(),
    'isotropic_d128': lambda: dataset_isotropic(128),
    'isotropic_d384': lambda: dataset_isotropic(384),
    'anisotropic_d128_r5': lambda: dataset_anisotropic(128, 5),
    'anisotropic_d128_r15': lambda: dataset_anisotropic(128, 15),
    'anisotropic_d128_r40': lambda: dataset_anisotropic(128, 40),
}

#-------------------------------------------------------------------------------


def run_config(data: dict, radii: np.ndarray, true_counts: np.ndarray, m: int, x: int, L: int,
               rng: np.random.Generator) -> dict:
    points, queries = data['points'], data['queries']

    t0 = time.perf_counter()
    tables = build_tables(points, m, L, rng)
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    retrieved = query_retrieval(tables, queries, m, x)
    t_retrieve = time.perf_counter() - t0

    t0 = time.perf_counter()
    n_hat = estimate_count_batch(queries, retrieved, points, radii, m, x, L)
    t_estimate = time.perf_counter() - t0

    retrieved_counts = np.array([len(r) for r in retrieved])
    valid = true_counts > 0
    rel_err = np.full(len(queries), np.nan)
    rel_err[valid] = np.abs(n_hat[valid] - true_counts[valid]) / true_counts[valid]
    ratio = np.full(len(queries), np.nan)
    ratio[valid] = n_hat[valid] / true_counts[valid]

    return {
        'n_hat': n_hat, 'true_counts': true_counts, 'retrieved_counts': retrieved_counts,
        'rel_err': rel_err, 'ratio': ratio,
        't_build': t_build, 't_retrieve': t_retrieve, 't_estimate': t_estimate,
    }


def save_config(out_dir: Path, dataset_key: str, m: int, x: int, L: int, result: dict) -> None:
    d = out_dir / 'E3' / dataset_key
    d.mkdir(parents=True, exist_ok=True)
    path = d / f'm{m}_x{x}_L{L}.npz'
    np.savez_compressed(path, **{k: v for k, v in result.items() if isinstance(v, np.ndarray)},
                         t_build=result['t_build'], t_retrieve=result['t_retrieve'],
                         t_estimate=result['t_estimate'])

#-------------------------------------------------------------------------------


def run_e3(progress_path: Path) -> None:
    total = len(E3_DATASETS) * len(M_VALUES) * len(X_VALUES) * len(L_VALUES)
    done = 0
    start_time = time.time()

    for dataset_key, loader in E3_DATASETS.items():
        data = loader()
        radii = _direct_radii(data['points'], data['queries'], K)
        true_counts = np.array([
            int((np.arccos(np.clip(data['points'] @ data['queries'][i], -1.0, 1.0)) <= radii[i] + 1e-9).sum())
            for i in range(len(data['queries']))
        ])

        for m in tqdm(M_VALUES, desc=f'{dataset_key} m'):
            for x in X_VALUES:
                if x >= m:
                    done += len(L_VALUES)
                    continue
                for L in L_VALUES:
                    rng = np.random.default_rng(SEED + m * 1000 + x * 100 + L)
                    result = run_config(data, radii, true_counts, m, x, L, rng)
                    save_config(OUT_DIR, dataset_key, m, x, L, result)
                    done += 1
                    tmp = progress_path.with_suffix('.tmp')
                    tmp.write_text(json.dumps({'stage': 'E3', 'done': done, 'total': total,
                                                'current': f'{dataset_key} m={m} x={x} L={L}',
                                                'elapsed_seconds': time.time() - start_time,
                                                'timestamp': datetime.now().isoformat()}))
                    tmp.replace(progress_path)


def run_e2(progress_path: Path) -> None:
    """Fixed-table bias vs L: 30 independent table-set draws per L,
    on SIFT with a fixed moderate (m,x)."""
    m, x = 20, 2
    n_draws = 30
    data = dataset_sift(n_queries=150)
    radii = _direct_radii(data['points'], data['queries'], K)
    true_counts = np.array([
        int((np.arccos(np.clip(data['points'] @ data['queries'][i], -1.0, 1.0)) <= radii[i] + 1e-9).sum())
        for i in range(len(data['queries']))
    ])

    out_dir = OUT_DIR / 'E2'
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(L_VALUES) * n_draws
    done = 0
    start_time = time.time()

    for L in L_VALUES:
        mean_ratios = []
        mean_retrieved_fracs = []
        for draw in range(n_draws):
            rng = np.random.default_rng(10000 + L * 100 + draw)
            tables = build_tables(data['points'], m, L, rng)
            retrieved = query_retrieval(tables, data['queries'], m, x)
            n_hat = estimate_count_batch(data['queries'], retrieved, data['points'], radii, m, x, L)
            valid = true_counts > 0
            mean_ratios.append(float(np.mean(n_hat[valid] / true_counts[valid])))
            mean_retrieved_fracs.append(float(np.mean([len(r) for r in retrieved])) / len(data['points']))
            done += 1
            tmp = progress_path.with_suffix('.tmp')
            tmp.write_text(json.dumps({'stage': 'E2', 'done': done, 'total': total,
                                        'current': f'L={L} draw={draw}',
                                        'elapsed_seconds': time.time() - start_time,
                                        'timestamp': datetime.now().isoformat()}))
            tmp.replace(progress_path)
        np.savez_compressed(out_dir / f'L{L}.npz', mean_ratios=np.array(mean_ratios),
                             mean_retrieved_fracs=np.array(mean_retrieved_fracs))


def run_e4(progress_path: Path) -> None:
    """Scaling: N in {4K,8K,16K,32K,50K}, m=log2(N)+4, fixed L=16 first;
    then repeat holding error roughly constant by increasing L."""
    import math
    n_values = [4000, 8000, 16000, 32000, 50000]
    x = 2
    out_dir = OUT_DIR / 'E4'
    out_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    total = len(n_values) * 2
    done = 0

    for fixed_L in [True, False]:
        L_current = 16
        for n_pts in n_values:
            m = int(round(math.log2(n_pts) + 4))
            data = dataset_isotropic(128, n_points=n_pts, n_queries=200, seed=SEED)
            radii = _direct_radii(data['points'], data['queries'], K)
            true_counts = np.array([
                int((np.arccos(np.clip(data['points'] @ data['queries'][i], -1.0, 1.0)) <= radii[i] + 1e-9).sum())
                for i in range(len(data['queries']))
            ])

            if fixed_L:
                L = 16
            else:
                # increase L until mean relative error <= 0.20 or L=128 cap
                L = 4
                while L <= 128:
                    rng = np.random.default_rng(SEED + n_pts + L)
                    tables = build_tables(data['points'], m, L, rng)
                    retrieved = query_retrieval(tables, data['queries'], m, x)
                    n_hat = estimate_count_batch(data['queries'], retrieved, data['points'], radii, m, x, L)
                    valid = true_counts > 0
                    rel_err = np.mean(np.abs(n_hat[valid] - true_counts[valid]) / true_counts[valid])
                    if rel_err <= 0.20:
                        break
                    L *= 2

            rng = np.random.default_rng(SEED + n_pts + L + 1)
            tables = build_tables(data['points'], m, L, rng)
            retrieved = query_retrieval(tables, data['queries'], m, x)
            n_hat = estimate_count_batch(data['queries'], retrieved, data['points'], radii, m, x, L)
            retrieved_counts = np.array([len(r) for r in retrieved])
            valid = true_counts > 0
            rel_err = float(np.mean(np.abs(n_hat[valid] - true_counts[valid]) / true_counts[valid]))

            label = 'fixedL' if fixed_L else 'targeterr'
            np.savez_compressed(out_dir / f'{label}_N{n_pts}.npz', m=m, L=L,
                                 retrieved_counts=retrieved_counts, rel_err=rel_err,
                                 n_hat=n_hat, true_counts=true_counts)
            done += 1
            tmp = progress_path.with_suffix('.tmp')
            tmp.write_text(json.dumps({'stage': 'E4', 'done': done, 'total': total,
                                        'current': f'{label} N={n_pts} m={m} L={L}',
                                        'elapsed_seconds': time.time() - start_time,
                                        'timestamp': datetime.now().isoformat()}))
            tmp.replace(progress_path)


def run_e6() -> None:
    """Memory: L x N point-id storage (measured, deterministic given
    table structure) plus direction storage, full-per-table vs a
    hypothetical shared pool of P=256 candidate directions indexed per
    table (analytical, mirroring the tree design's H5 comparison)."""
    out_dir = OUT_DIR / 'E6'
    out_dir.mkdir(parents=True, exist_ok=True)
    P = 256
    rows = []
    for dataset_key, d in [('sift', 128), ('esci', 384)]:
        data = dataset_sift(n_points=2000) if dataset_key == 'sift' else dataset_esci(n_points=2000)
        for m in [20, 32]:
            for L in [16, 64]:
                rng = np.random.default_rng(SEED)
                tables = build_tables(data['points'], m, L, rng)
                point_id_bytes = sum(t['flat_points'].nbytes for t in tables)
                full_direction_bytes = L * m * d * 8
                pooled_direction_bytes = P * d * 8 + L * m * 4  # pool once + int32 index per (table,direction)
                rows.append({'dataset': dataset_key, 'd': d, 'm': m, 'L': L,
                             'point_id_bytes_measured': int(point_id_bytes),
                             'full_direction_bytes': int(full_direction_bytes),
                             'pooled_direction_bytes': int(pooled_direction_bytes)})
    with open(out_dir / 'memory.json', 'w') as f:
        json.dump(rows, f, indent=2)

#-------------------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    progress_path = OUT_DIR / 'progress.json'

    print("Running E6 (memory, cheap)...")
    run_e6()

    print("Running E2 (fixed-table bias vs L)...")
    run_e2(progress_path)

    print("Running E4 (scaling)...")
    run_e4(progress_path)

    print("Running E3 (cost/accuracy frontier grid)...")
    run_e3(progress_path)

    print("\nDone.")


if __name__ == '__main__':
    main()
