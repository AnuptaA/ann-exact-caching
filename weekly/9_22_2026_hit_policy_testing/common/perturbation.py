#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Generate synthetic test queries at controlled angular distances from an
anchor, using the tangent-plane construction on the unit hypersphere.

Distances are in this codebase's normalized angular units (arccos(dot)/pi,
range [0,1]).
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import pandas as pd

from utils.base.distance_metrics import pairwise_angular_distance

#-------------------------------------------------------------------------------

def random_tangent_direction(anchor: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """DEPRECATED: no longer used by generate_radial_sweep/generate_random_pool.
    """
    g = rng.standard_normal(anchor.shape[0])
    v = g - np.dot(g, anchor) * anchor
    return v / np.linalg.norm(v)

#-------------------------------------------------------------------------------

def direction_toward_target(anchor: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The tangent direction at anchor pointing toward a real target point,
    along their connecting great circle. Same projection as
    random_tangent_direction, but steered by a real nearby point instead of
    Gaussian noise."""
    v = target - np.dot(target, anchor) * anchor
    return v / np.linalg.norm(v)

#-------------------------------------------------------------------------------

def tangent_perturb(anchor: np.ndarray, direction: np.ndarray, target_norm_dist: float) -> tuple[np.ndarray, float]:
    """Perturb anchor along direction to target_norm_dist (clipped to 1.0,
    the antipodal maximum). Returns (perturbed_point, realized_norm_dist)."""
    clipped = min(target_norm_dist, 1.0)
    theta_rad = clipped * np.pi
    perturbed = np.cos(theta_rad) * anchor + np.sin(theta_rad) * direction
    perturbed = perturbed / np.linalg.norm(perturbed)
    realized = float(pairwise_angular_distance(anchor[None, :], perturbed[None, :])[0, 0])
    return perturbed, realized

#-------------------------------------------------------------------------------

def generate_radial_sweep(
    anchor: np.ndarray,
    n_directions: int,
    relative_distances: list[float],
    r_c: float,
    rng: np.random.Generator,
    target_pool: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """n_directions independent rays, each steered toward a random real point
    from target_pool (e.g. the anchor's own top-500 nearest neighbors, not a
    uniformly random tangent direction); each ray is reused across all
    relative_distances (fixed direction, varying distance along it). Returns
    (metadata_df, vectors)."""
    rows = []
    vectors = []
    for direction_idx in range(n_directions):
        target = target_pool[rng.integers(len(target_pool))]
        direction = direction_toward_target(anchor, target)
        for nominal_rel_dist in relative_distances:
            target_norm_dist = nominal_rel_dist * r_c
            perturbed, realized_norm_dist = tangent_perturb(anchor, direction, target_norm_dist)
            vectors.append(perturbed)
            rows.append(dict(
                direction_idx=direction_idx,
                nominal_relative_distance=nominal_rel_dist,
                realized_relative_distance=realized_norm_dist / r_c,
                clipped=target_norm_dist > 1.0,
            ))
    return pd.DataFrame(rows), np.stack(vectors)

#-------------------------------------------------------------------------------

def generate_random_pool(
    anchor: np.ndarray,
    n_draws: int,
    max_relative_distance: float,
    r_c: float,
    rng: np.random.Generator,
    target_pool: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """n_draws independent (random real target from target_pool, random
    Uniform(0, max_rel_dist)) draws. Returns (metadata_df, vectors)."""
    rows = []
    vectors = []
    for _ in range(n_draws):
        target = target_pool[rng.integers(len(target_pool))]
        direction = direction_toward_target(anchor, target)
        nominal_rel_dist = rng.uniform(0.0, max_relative_distance)
        target_norm_dist = nominal_rel_dist * r_c
        perturbed, realized_norm_dist = tangent_perturb(anchor, direction, target_norm_dist)
        vectors.append(perturbed)
        rows.append(dict(
            nominal_relative_distance=nominal_rel_dist,
            realized_relative_distance=realized_norm_dist / r_c,
            clipped=target_norm_dist > 1.0,
        ))
    return pd.DataFrame(rows), np.stack(vectors)
