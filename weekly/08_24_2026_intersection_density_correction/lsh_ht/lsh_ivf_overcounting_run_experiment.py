#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Microbenchmark: overcounting factor of the crudest possible density
correction (sum full bucket/cluster counts for every bucket touching the
query cap, no overlap-fraction weighting) for random-hyperplane LSH vs.
IVF (spherical k-means), across dimension, resolution, and cluster
structure. Both touching tests are exact analytic checks; no Monte Carlo
sampling is used anywhere in the sweep."""

import json
import math
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.stats import vonmises_fisher, qmc, norm
from tqdm import tqdm

#-------------------------------------------------------------------------------

D_VALUES = [2, 3, 5]
M_VALUES = [2, 3, 4, 5, 6, 8]
N_CLUSTERS_VALUES = [1, 3, 8]
N_TRIALS = 300
N_SWEEP = 720
N_CLUSTER_POINTS = 980
N_BACKGROUND_POINTS = 20
N_KMEANS_ITER = 20

# Calibrated empirically (see calibrate_kappa_and_radius.py output):
# loose end ~1 radian typical spread, tight end ~0.04-0.07 radian typical
# spread, per dimension (concentration-to-spread relationship is not
# dimension-independent, so this is not one fixed range reused for all d).
KAPPA_RANGE = {2: (1.0, 300.0), 3: (1.0, 300.0), 5: (1.0, 1000.0)}
CAP_RADIUS = {2: 0.02, 3: 0.035, 5: 0.03}

OUT_DIR = Path(__file__).parent.parent / 'results' / 'lsh_ivf_overcounting'

#-------------------------------------------------------------------------------


def resolution_count(m: int, d: int) -> int:
    """R(m, d) = 2 * sum_{k=0}^{d-1} C(m-1, k); reduces to 2m at d=2."""
    return 2 * sum(math.comb(m - 1, k) for k in range(d))


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vecs / norms

#-------------------------------------------------------------------------------
# Sweep-position generation (low-discrepancy point sets on S^{d-1}).
#-------------------------------------------------------------------------------


def sweep_positions(d: int, n_sweep: int) -> np.ndarray:
    if d == 2:
        angles = np.linspace(0.0, 2.0 * np.pi, n_sweep, endpoint=False)
        return np.stack([np.cos(angles), np.sin(angles)], axis=1)
    if d == 3:
        i = np.arange(n_sweep)
        phi = np.arccos(1.0 - 2.0 * (i + 0.5) / n_sweep)
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        theta = golden_angle * i
        return np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=1)
    # d=5 (or any d without a direct Fibonacci-sphere analog): Sobol in the
    # hypercube, mapped through the inverse-normal CDF, then normalized.
    n_base2 = 1 << (n_sweep - 1).bit_length()
    sampler = qmc.Sobol(d=d, seed=0)
    u = sampler.random_base2(m=n_base2.bit_length() - 1)[:n_sweep]
    u = np.clip(u, 1e-10, 1.0 - 1e-10)
    g = norm.ppf(u)
    return _normalize(g)

#-------------------------------------------------------------------------------
# Data generation.
#-------------------------------------------------------------------------------


def generate_trial_data(d: int, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    """n_clusters vMF clusters (980 points total) plus 20 uniform
    background points, all on S^{d-1}."""
    kappa_lo, kappa_hi = KAPPA_RANGE[d]
    centers = _normalize(rng.standard_normal((n_clusters, d)))

    if n_clusters == 1:
        fractions = np.array([1.0])
    else:
        while True:
            fractions = rng.dirichlet(np.full(n_clusters, 0.5))
            if fractions.min() >= 0.01:
                break
    sizes = np.round(fractions * N_CLUSTER_POINTS).astype(int)
    sizes[-1] = N_CLUSTER_POINTS - sizes[:-1].sum()

    kappas = np.exp(rng.uniform(np.log(kappa_lo), np.log(kappa_hi), size=n_clusters))

    cluster_points = []
    for c in range(n_clusters):
        if sizes[c] <= 0:
            continue
        dist = vonmises_fisher(centers[c], float(kappas[c]))
        cluster_points.append(dist.rvs(size=int(sizes[c]), random_state=rng))
    cluster_points = np.vstack(cluster_points)

    background = _normalize(rng.standard_normal((N_BACKGROUND_POINTS, d)))
    return np.vstack([cluster_points, background])

#-------------------------------------------------------------------------------
# LSH: exact candidate-bucket touching test.
#-------------------------------------------------------------------------------


def build_lsh(points: np.ndarray, m: int, d: int, rng: np.random.Generator) -> tuple:
    hyperplanes = _normalize(rng.standard_normal((m, d)))
    bits = (points @ hyperplanes.T) >= 0.0  # (n_points, m)
    keys = (bits.astype(np.int64) * (1 << np.arange(m))).sum(axis=1)
    unique_keys, first_idx, counts = np.unique(keys, return_index=True, return_counts=True)
    unique_bits = bits[first_idx]  # (n_buckets, m)
    return hyperplanes, unique_bits, counts


def lsh_estimate(hyperplanes: np.ndarray, unique_bits: np.ndarray, counts: np.ndarray,
                  sweep_points: np.ndarray, cap_radius: float) -> np.ndarray:
    projections = sweep_points @ hyperplanes.T  # (n_sweep, m)
    margins = np.arcsin(np.clip(np.abs(projections), 0.0, 1.0))
    certain = margins >= cap_radius  # (n_sweep, m)
    query_bits = projections >= 0.0  # (n_sweep, m)

    mismatch = certain[:, None, :] & (unique_bits[None, :, :] != query_bits[:, None, :])  # (n_sweep, n_buckets, m)
    touches = ~mismatch.any(axis=2)  # (n_sweep, n_buckets)
    return touches @ counts  # (n_sweep,)

#-------------------------------------------------------------------------------
# IVF: spherical k-means and bounding-cap touching test.
#-------------------------------------------------------------------------------


def spherical_kmeans(points: np.ndarray, k: int, rng: np.random.Generator,
                      n_iter: int = N_KMEANS_ITER) -> tuple:
    n, d = points.shape
    k = min(k, n)
    init_idx = rng.choice(n, size=k, replace=False)
    centroids = points[init_idx].copy()

    for _ in range(n_iter):
        sims = points @ centroids.T  # (n, k)
        labels = sims.argmax(axis=1)
        sums = np.zeros((k, d))
        np.add.at(sums, labels, points)
        counts = np.bincount(labels, minlength=k)
        nonempty = counts > 0
        norms = np.linalg.norm(sums[nonempty], axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        centroids[nonempty] = sums[nonempty] / norms

    sims = points @ centroids.T
    labels = sims.argmax(axis=1)
    return centroids, labels


def build_ivf(points: np.ndarray, k: int, rng: np.random.Generator) -> tuple:
    centroids, labels = spherical_kmeans(points, k, rng)
    counts = np.bincount(labels, minlength=centroids.shape[0])
    populated = counts > 0

    ang_to_own_centroid = np.arccos(np.clip(np.einsum('ij,ij->i', points, centroids[labels]), -1.0, 1.0))
    bounding_radius = np.zeros(centroids.shape[0])
    np.maximum.at(bounding_radius, labels, ang_to_own_centroid)
    return centroids, counts, bounding_radius, populated


def ivf_estimate(centroids: np.ndarray, counts: np.ndarray, bounding_radius: np.ndarray,
                  populated: np.ndarray, sweep_points: np.ndarray, cap_radius: float) -> np.ndarray:
    ang_query_centroid = np.arccos(np.clip(sweep_points @ centroids.T, -1.0, 1.0))  # (n_sweep, k)
    touches = (ang_query_centroid < (bounding_radius[None, :] + cap_radius)) & populated[None, :]
    return touches @ counts  # (n_sweep,)

#-------------------------------------------------------------------------------


def true_captured_counts(points: np.ndarray, sweep_points: np.ndarray, cap_radius: float) -> np.ndarray:
    ang = np.arccos(np.clip(sweep_points @ points.T, -1.0, 1.0))  # (n_sweep, n_points)
    return (ang <= cap_radius).sum(axis=1)


def run_combo(d: int, m: int, n_clusters: int, n_trials: int, sweep: np.ndarray, seed: int) -> dict:
    cap_radius = CAP_RADIUS[d]
    k_ivf = resolution_count(m, d)
    rng = np.random.default_rng(seed)

    est_lsh = np.empty((n_trials, N_SWEEP))
    est_ivf = np.empty((n_trials, N_SWEEP))
    true_cap = np.empty((n_trials, N_SWEEP))

    for t in range(n_trials):
        points = generate_trial_data(d, n_clusters, rng)
        hyperplanes, unique_bits, lsh_counts = build_lsh(points, m, d, rng)
        centroids, ivf_counts, bounding_radius, populated = build_ivf(points, k_ivf, rng)

        est_lsh[t] = lsh_estimate(hyperplanes, unique_bits, lsh_counts, sweep, cap_radius)
        est_ivf[t] = ivf_estimate(centroids, ivf_counts, bounding_radius, populated, sweep, cap_radius)
        true_cap[t] = true_captured_counts(points, sweep, cap_radius)

    return {'estimate_lsh': est_lsh, 'estimate_ivf': est_ivf, 'true_captured': true_cap}

#-------------------------------------------------------------------------------


def _write_progress(progress_path: Path, done: int, total: int, start_time: float) -> None:
    tmp_path = progress_path.with_suffix('.tmp')
    payload = {
        'combos_done': done, 'combos_total': total,
        'elapsed_seconds': time.time() - start_time,
        'timestamp': datetime.now().isoformat(),
    }
    tmp_path.write_text(json.dumps(payload))
    tmp_path.replace(progress_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--d-values', type=int, nargs='+', default=D_VALUES, dest='d_values')
    parser.add_argument('--m-values', type=int, nargs='+', default=M_VALUES, dest='m_values')
    parser.add_argument('--n-clusters-values', type=int, nargs='+', default=N_CLUSTERS_VALUES, dest='n_clusters_values')
    parser.add_argument('--n-trials', type=int, default=N_TRIALS, dest='n_trials')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-dir', type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    combos = [(d, m, c) for d in args.d_values for m in args.m_values for c in args.n_clusters_values]
    print(f"Combos: {len(combos)}, n_trials={args.n_trials}, n_sweep={N_SWEEP}")

    sweep_cache = {d: sweep_positions(d, N_SWEEP) for d in args.d_values}

    progress_path = out_dir / 'progress.json'
    start_time = time.time()
    for i, (d, m, n_clusters) in enumerate(tqdm(combos, desc='combos', mininterval=1.0)):
        seed = args.seed + i
        result = run_combo(d, m, n_clusters, args.n_trials, sweep_cache[d], seed)
        path = out_dir / f'd{d}_m{m}_c{n_clusters}.npz'
        np.savez_compressed(path, d=d, m=m, n_clusters=n_clusters, cap_radius=CAP_RADIUS[d],
                             k_ivf=resolution_count(m, d), **result)
        _write_progress(progress_path, i + 1, len(combos), start_time)

    print(f"Done: {out_dir}")


if __name__ == '__main__':
    main()
