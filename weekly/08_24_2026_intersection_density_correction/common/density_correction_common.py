#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Shared Part 1 (overlap-count sanity check) and Part 2 (local-density
estimator) computations for the density-correction stage."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np

from primitives import single_cap_fraction, sample_cap_points_direct

#-------------------------------------------------------------------------------

_BOUNDARY_EPS = 1e-9  # absolute angular tolerance on the K-th-NN boundary comparison

def _angular_distance_consistent(background: np.ndarray, pole: np.ndarray) -> np.ndarray:
    """Angular distance from every corpus point to pole, via the same
    chord-then-angle formula used to derive r_test/r_cache elsewhere."""
    sims = background @ pole
    chord = np.sqrt(np.clip(2.0 - 2.0 * sims, 0.0, None))
    return 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))


def true_recall_consistent(background: np.ndarray, idx_topk: np.ndarray, c_star: np.ndarray, r_cache: float) -> float:
    """Recall of q from c*, using _angular_distance_consistent's formula
    for consistency with r_cache (not np.linalg.norm)."""
    dist_c = _angular_distance_consistent(background[idx_topk], c_star)
    return float(np.mean(dist_c <= r_cache + _BOUNDARY_EPS))


def covered_mask(candidate_coords: np.ndarray, pole: np.ndarray, radius: float) -> np.ndarray:
    """Boundary-tolerant "within radius of pole" test. candidate_coords
    may have any shape (..., d); returns a mask of the leading shape."""
    orig_shape = candidate_coords.shape[:-1]
    flat = candidate_coords.reshape(-1, candidate_coords.shape[-1])
    dist = _angular_distance_consistent(flat, pole)
    return (dist <= radius + _BOUNDARY_EPS).reshape(orig_shape)


def independent_overlap_count(background: np.ndarray, q: np.ndarray, c_star: np.ndarray,
                               r_test: float, r_cache: float) -> int:
    """Brute-force count of corpus points within r_test of q and within
    r_cache of c_star, independent of the already-saved recall value."""
    dist_q = _angular_distance_consistent(background, q)
    dist_c = _angular_distance_consistent(background, c_star)
    return int(np.sum((dist_q <= r_test + _BOUNDARY_EPS) & (dist_c <= r_cache + _BOUNDARY_EPS)))


def compute_corrected_recall_exact(overlap_count: int, k: int, if_value: float, r_test: float, d: int) -> float:
    """corrected_recall = IF * (density_overlap / density_q), with
    density_q = K / area(r_test) and density_overlap = overlap_count /
    (IF * area(r_test))."""
    area_q = single_cap_fraction(r_test, d)
    if area_q <= 0 or if_value <= 0:
        return float('nan')
    density_q = k / area_q
    density_overlap = overlap_count / (if_value * area_q)
    return float(if_value * density_overlap / density_q)


#-------------------------------------------------------------------------------
# Part 2: LSH (random-hyperplane / SimHash) local-density estimator.
#-------------------------------------------------------------------------------

def build_lsh_tables(background: np.ndarray, k_bits: int, l_tables: int, rng: np.random.Generator,
                      d: int) -> list:
    """Precompute L independent hash tables over one corpus, each with
    K random hyperplanes and a bucket-key -> point-count map."""
    tables = []
    weights = (1 << np.arange(k_bits))
    for _ in range(l_tables):
        hyperplanes = rng.standard_normal((k_bits, d))
        bits = (background @ hyperplanes.T) >= 0.0  # (n_corpus, K)
        keys = (bits.astype(np.int64) * weights).sum(axis=1)  # (n_corpus,)
        unique_keys, counts = np.unique(keys, return_counts=True)
        tables.append({'hyperplanes': hyperplanes, 'key_to_count': dict(zip(unique_keys.tolist(), counts.tolist()))})
    return tables


def hash_count_at_point(tables: list, point: np.ndarray) -> float:
    """Average, across the L tables, of how many corpus points share
    point's bucket key in that table."""
    counts = []
    for table in tables:
        bits = (table['hyperplanes'] @ point) >= 0.0  # (K,)
        weights = (1 << np.arange(len(bits)))
        key = int((bits.astype(np.int64) * weights).sum())
        counts.append(table['key_to_count'].get(key, 0))
    return float(np.mean(counts))


def estimate_range_count(tables: list, point: np.ndarray, exclude_points: np.ndarray = None) -> tuple:
    """Per-table bucket count at point (raw) and, minus any
    exclude_points sharing that bucket (extra), averaged over tables."""
    raw_counts = []
    extra_counts = [] if exclude_points is not None else None
    for table in tables:
        weights = (1 << np.arange(table['hyperplanes'].shape[0]))
        bits = (table['hyperplanes'] @ point) >= 0.0
        key = int((bits.astype(np.int64) * weights).sum())
        raw = table['key_to_count'].get(key, 0)
        raw_counts.append(raw)
        if exclude_points is not None:
            excl_bits = (exclude_points @ table['hyperplanes'].T) >= 0.0  # (n_excl, K)
            excl_keys = (excl_bits.astype(np.int64) * weights).sum(axis=1)  # (n_excl,)
            n_matching = int(np.sum(excl_keys == key))
            extra_counts.append(max(raw - n_matching, 0))

    raw_estimate = float(np.mean(raw_counts))
    extra_estimate = float(np.mean(extra_counts)) if exclude_points is not None else None
    return raw_estimate, extra_estimate


def find_overlap_sample_point(q: np.ndarray, c_star: np.ndarray, r_test: float, r_cache: float,
                               rng: np.random.Generator, max_batches: int = 20,
                               batch_size: int = 200) -> np.ndarray:
    """Rejection-samples one point inside the intersection of q's and
    c_star's caps. Returns None if no sample is accepted."""
    if r_test <= r_cache:
        small_pole, small_radius = q, r_test
    else:
        small_pole, small_radius = c_star, r_cache

    for _ in range(max_batches):
        candidates = sample_cap_points_direct(small_pole, small_radius, batch_size, rng)
        ok = covered_mask(candidates, q, r_test) & covered_mask(candidates, c_star, r_cache)
        if np.any(ok):
            return candidates[ok][0]
    return None


def compute_corrected_recall_estimated(tables: list, q: np.ndarray, c_star: np.ndarray, r_test: float,
                                        r_cache: float, if_value: float, rng: np.random.Generator) -> float:
    """Part 2: corrected_recall = IF * (hash_count_overlap /
    hash_count_q), the LSH-based density-ratio estimator."""
    overlap_point = find_overlap_sample_point(q, c_star, r_test, r_cache, rng)
    if overlap_point is None:
        return float('nan')
    hash_count_q = hash_count_at_point(tables, q)
    hash_count_overlap = hash_count_at_point(tables, overlap_point)
    if hash_count_q <= 0:
        return float('nan')
    return float(if_value * hash_count_overlap / hash_count_q)
