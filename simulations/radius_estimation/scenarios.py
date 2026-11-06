#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Data/query scenario generators for the radius-estimation analysis.

Each scenario fixes three independent spatial fields:
  - data vectors (determine the true radius field r_true(x))
  - train queries (where we observe the field + count weights)
  - test queries (where we evaluate the estimator)

Scenarios are designed to switch the two bias mechanisms on and off:
  - regression-to-the-mean / smoothing bias: driven by variation of r_true
    within a Hamming ball (i.e. non-uniform data)
  - count-weighting bias: driven by correlation between train-query density
    and r_true inside the ball (i.e. queries tracking the data)
"""

import numpy as np
from dataclasses import dataclass

#-------------------------------------------------------------------------------


@dataclass
class Scenario:
    name: str
    description: str
    data: np.ndarray
    train_queries: np.ndarray
    test_queries: np.ndarray


#-------------------------------------------------------------------------------


def _uniform_cube(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    return rng.uniform(0.0, 1.0, size=(n, dim))


def _gaussian_clusters(
    rng: np.random.Generator,
    n: int,
    dim: int,
    centers: np.ndarray,
    spread: float,
) -> np.ndarray:
    """Draw n points from a mixture of isotropic Gaussians at `centers`."""
    n_clusters = len(centers)
    assign = rng.integers(0, n_clusters, size=n)
    pts = centers[assign] + spread * rng.standard_normal((n, dim))
    return pts


def _sample_near_points(
    rng: np.random.Generator,
    anchors: np.ndarray,
    n: int,
    jitter: float,
) -> np.ndarray:
    """Sample n points, each a small jitter away from a random anchor point."""
    idx = rng.integers(0, len(anchors), size=n)
    return anchors[idx] + jitter * rng.standard_normal((n, anchors.shape[1]))


#-------------------------------------------------------------------------------


def build_scenarios(
    dim: int = 16,
    n_data: int = 5000,
    n_train: int = 4000,
    n_test: int = 2000,
    n_clusters: int = 20,
    cluster_spread: float = 0.03,
    tight_jitter: float = 0.01,
    seed: int = 42,
) -> list:
    """Construct the five analysis scenarios.

    1. uniform_data_uniform_queries   (control: both bias mechanisms off)
    2. uniform_data_queries_from_data (count-weighting on, flat field)
    3. clustered_data_uniform_queries (smoothing on, count-weighting off)
    4. clustered_data_queries_from_data (both on, aligned; realistic case)
    5. clustered_data_tight_queries   (both on globally, but low LOCAL variation)
    """
    rng = np.random.default_rng(seed)
    scenarios = []

    # ---- cluster centers shared by all clustered scenarios ----
    centers = rng.uniform(0.0, 1.0, size=(n_clusters, dim))

    # 1. uniform data + uniform queries
    data_u = _uniform_cube(rng, n_data, dim)
    scenarios.append(Scenario(
        name="uniform_data_uniform_queries",
        description="Uniform data, uniform train/test. Control: ~flat field, "
                    "flat query density. Both bias mechanisms off.",
        data=data_u,
        train_queries=_uniform_cube(rng, n_train, dim),
        test_queries=_uniform_cube(rng, n_test, dim),
    ))

    # 2. uniform data + queries sampled from data
    data_u2 = _uniform_cube(rng, n_data, dim)
    scenarios.append(Scenario(
        name="uniform_data_queries_from_data",
        description="Uniform data, train/test sampled near data points. "
                    "Count-weighting active but field is flat, so weighting a "
                    "flat field stays ~unbiased.",
        data=data_u2,
        train_queries=_sample_near_points(rng, data_u2, n_train, tight_jitter),
        test_queries=_sample_near_points(rng, data_u2, n_test, tight_jitter),
    ))

    # 3. clustered data + uniform queries
    data_c = _gaussian_clusters(rng, n_data, dim, centers, cluster_spread)
    scenarios.append(Scenario(
        name="clustered_data_uniform_queries",
        description="Clustered data, uniform train/test. Non-flat field with "
                    "count-weighting OFF. Isolates regression-to-the-mean.",
        data=data_c,
        train_queries=_uniform_cube(rng, n_train, dim),
        test_queries=_uniform_cube(rng, n_test, dim),
    ))

    # 4. clustered data + queries sampled from data
    data_c2 = _gaussian_clusters(rng, n_data, dim, centers, cluster_spread)
    scenarios.append(Scenario(
        name="clustered_data_queries_from_data",
        description="Clustered data, train/test sampled near data points. Both "
                    "mechanisms on and aligned. Realistic case; expect strong "
                    "downward bias at large radius.",
        data=data_c2,
        train_queries=_sample_near_points(rng, data_c2, n_train, cluster_spread),
        test_queries=_sample_near_points(rng, data_c2, n_test, cluster_spread),
    ))

    # 5. clustered data + tightly correlated train/test
    data_c3 = _gaussian_clusters(rng, n_data, dim, centers, cluster_spread)
    # pick shared anchor locations; train and test both jitter tightly around them
    anchors = _sample_near_points(rng, data_c3, n_test, tight_jitter)
    train_tight = anchors[rng.integers(0, len(anchors), size=n_train)] \
        + tight_jitter * rng.standard_normal((n_train, dim))
    scenarios.append(Scenario(
        name="clustered_data_tight_queries",
        description="Clustered data, train/test tightly co-located. Field "
                    "varies globally but little WITHIN the sampled neighborhood. "
                    "Best case: low bias despite clustered data.",
        data=data_c3,
        train_queries=train_tight,
        test_queries=anchors,
    ))

    return scenarios
