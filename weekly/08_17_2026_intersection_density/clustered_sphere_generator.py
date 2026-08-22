#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Synthetic clustered points on the unit sphere, ground-truth density
known by construction (mixture of von Mises-Fisher distributions).
"""

import numpy as np
from scipy.stats import vonmises_fisher

#-------------------------------------------------------------------------------

def _sample_uniform_sphere(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    pts = rng.standard_normal((n, d))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def sample_clustered_sphere(n_points: int, d: int, n_clusters: int, dense_kappa: float, sparse_kappa: float,
                             dense_fraction: float, rng: np.random.Generator) -> tuple:
    """Points on S^(d-1) drawn from a mixture of von Mises-Fisher
    clusters, each tagged 'dense' or 'sparse' by its concentration.

    Returns (points, cluster_id, tier) where tier is a bool array,
    True for dense.
    """
    centers = _sample_uniform_sphere(n_clusters, d, rng)
    is_dense = rng.random(n_clusters) < dense_fraction

    cluster_id = rng.integers(0, n_clusters, size=n_points)
    points = np.empty((n_points, d), dtype=np.float64)
    for c in range(n_clusters):
        members = np.where(cluster_id == c)[0]
        if len(members) == 0:
            continue
        kappa = dense_kappa if is_dense[c] else sparse_kappa
        points[members] = vonmises_fisher.rvs(mu=centers[c], kappa=kappa, size=len(members),
                                               random_state=rng)

    return points, cluster_id, is_dense[cluster_id]

#-------------------------------------------------------------------------------

if __name__ == '__main__':
    import math

    results = []

    def check(name: str, cond: bool, detail: str = '') -> None:
        results.append((name, cond, detail))
        status = 'PASS' if cond else 'FAIL'
        print(f"[{status}] {name}" + (f"  ({detail})" if detail else ''))

    rng = np.random.default_rng(0)
    D = 128
    points, cluster_id, is_dense = sample_clustered_sphere(
        n_points=20000, d=D, n_clusters=20, dense_kappa=500.0, sparse_kappa=3.0, dense_fraction=0.5, rng=rng,
    )

    norms = np.linalg.norm(points, axis=1)
    check("unit-norm", bool(np.allclose(norms, 1.0, atol=1e-8)), f"max diff {np.max(np.abs(norms-1.0)):.2e}")

    def knn_radius(pts: np.ndarray, query: np.ndarray, k: int) -> float:
        sims = pts @ query
        top = np.partition(-sims, k - 1)[:k]
        return float(np.arccos(np.clip(-top[-1], -1.0, 1.0)))

    dense_clusters = np.unique(cluster_id[is_dense])
    sparse_clusters = np.unique(cluster_id[~is_dense])
    dense_radii, sparse_radii = [], []
    for c in dense_clusters[:5]:
        members = points[cluster_id == c]
        if len(members) < 60:
            continue
        dense_radii.append(knn_radius(members, members[0], 50))
    for c in sparse_clusters[:5]:
        members = points[cluster_id == c]
        if len(members) < 60:
            continue
        sparse_radii.append(knn_radius(members, members[0], 50))

    dense_mean, sparse_mean = np.mean(dense_radii), np.mean(sparse_radii)
    check("dense-tier K-NN radius smaller than sparse-tier", dense_mean < sparse_mean,
          f"dense={dense_mean:.4f} sparse={sparse_mean:.4f}")

    def pairwise_spread(pts: np.ndarray, center: np.ndarray) -> float:
        return float(np.arccos(np.clip(pts @ center, -1.0, 1.0)).mean())

    dense_spread = np.mean([pairwise_spread(points[cluster_id == c], points[cluster_id == c].mean(axis=0) /
                                             np.linalg.norm(points[cluster_id == c].mean(axis=0)))
                             for c in dense_clusters[:5] if (cluster_id == c).sum() > 1])
    sparse_spread = np.mean([pairwise_spread(points[cluster_id == c], points[cluster_id == c].mean(axis=0) /
                                              np.linalg.norm(points[cluster_id == c].mean(axis=0)))
                              for c in sparse_clusters[:5] if (cluster_id == c).sum() > 1])
    check("dense clusters tighter pairwise spread than sparse", dense_spread < sparse_spread,
          f"dense={dense_spread:.4f} sparse={sparse_spread:.4f}")

    n_fail = sum(1 for _, cond, _ in results if not cond)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed.")
    import sys
    sys.exit(1 if n_fail else 0)
