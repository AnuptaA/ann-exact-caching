#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""M3: Bayesian LSH bucket-count density estimator. Construction time: build
per-table point codes from random hyperplanes (reusing this project's
established _sample_uniform_sphere convention), sample a prior histogram of
query-to-corpus angular distance, and Bayes-invert the exact
P(hamming=x | theta) = Binomial(m, theta/pi) against that prior into a LUT
P(theta <= r | hamming=x). Query time: hash the query, compute Hamming
distance to every stored code (a global summary - codes/LUT, not raw
coordinates, per the assumptions this round explicitly permits), and read
off an expected point count via the LUT.

Codes are stored as plain (N, m) boolean bit arrays, not packed into a
single int64 (lsh_horvitz_thompson.py's _bit_weights/_popcount64 packing is
only valid up to ~62-63 bits per int64 - confirmed directly: at m=128 that
packing silently wrapped and produced a Hamming distribution with half the
correct mean. This project's (m,L) sweep goes up to m=256, so the packed
representation is unusable here regardless of code size; direct boolean
Hamming distance is used instead, dimension-independent).

No raw corpus coordinate is read at query time - estimate_count_curve only
touches `codes` (boolean arrays) and `hyperplanes`/`lut` (both built once at
construction time). This module's query-time functions are wrapped by
GuardedCoordinatePool in the calling experiment script, since a coordinate
read here would be exactly the bug class this project has been burned by
before."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
from scipy.stats import binom

from part2_uniform_baseline_run_experiment import _sample_uniform_sphere

#-------------------------------------------------------------------------------


def build_codes(points: np.ndarray, m: int, L: int, rng: np.random.Generator) -> tuple:
    """Per-table (N, m) boolean bit arrays and their (L, m, d) hyperplane
    matrices - construction-time, full corpus."""
    d = points.shape[1]
    codes = np.empty((L, points.shape[0], m), dtype=bool)
    hyperplanes = np.empty((L, m, d), dtype=np.float64)
    for l in range(L):
        H = _sample_uniform_sphere(m, d, rng)
        codes[l] = (points @ H.T) >= 0.0
        hyperplanes[l] = H
    return codes, hyperplanes


def build_prior_histogram(base: np.ndarray, query_sample: np.ndarray, n_bins: int) -> dict:
    """Offline, construction-time-only: histogram of angular distance from a
    sample of queries to the full corpus. Returns bin edges/centers/density
    over [0, pi]."""
    dists = np.arccos(np.clip(query_sample @ base.T, -1.0, 1.0)).ravel()
    edges = np.linspace(0.0, np.pi, n_bins + 1)
    counts, _ = np.histogram(dists, bins=edges)
    density = counts / counts.sum()
    centers = 0.5 * (edges[:-1] + edges[1:])
    return dict(edges=edges, centers=centers, density=density)

#-------------------------------------------------------------------------------


def build_lut(m: int, prior: dict, r_grid: np.ndarray) -> np.ndarray:
    """Bayes-inverts P(hamming=x|theta) (exact Binomial pmf) against the
    prior over theta into P(theta<=r | hamming=x), for every (x, r) grid
    cell. Returns shape (m+1, len(r_grid))."""
    centers, density = prior['centers'], prior['density']
    # likelihood[x, theta_bin] = P(hamming=x | theta=theta_bin)
    p_arg = centers[None, :] / np.pi  # (1, n_theta)
    x_arg = np.arange(m + 1)[:, None]  # (m+1, 1)
    likelihood = binom.pmf(x_arg, m, p_arg)  # (m+1, n_theta)

    joint = likelihood * density[None, :]  # (m+1, n_theta)
    evidence = joint.sum(axis=1, keepdims=True)  # (m+1, 1)
    evidence = np.clip(evidence, 1e-300, None)
    posterior = joint / evidence  # (m+1, n_theta), posterior[x, :] sums to 1

    # cumulative posterior mass at theta <= r, for each r in r_grid
    zeros_col = np.zeros((m + 1, 1))
    cum_edges = np.concatenate([zeros_col, np.cumsum(posterior, axis=1)], axis=1)  # (m+1, n_theta+1)
    bin_edges = prior['edges']
    lut = np.empty((m + 1, len(r_grid)))
    for xi in range(m + 1):
        lut[xi] = np.interp(r_grid, bin_edges, cum_edges[xi])
    return lut


def marginal_hamming_distribution(m: int, prior: dict) -> np.ndarray:
    """Unconditional P(hamming=x) implied by the prior - used only for the
    E1 LUT-marginal sanity check (build_lut's evidence, normalized)."""
    centers, density = prior['centers'], prior['density']
    p_arg = centers[None, :] / np.pi
    x_arg = np.arange(m + 1)[:, None]
    likelihood = binom.pmf(x_arg, m, p_arg)
    joint = likelihood * density[None, :]
    marginal = joint.sum(axis=1)
    return marginal / marginal.sum()

#-------------------------------------------------------------------------------


def _hash_query(query: np.ndarray, H: np.ndarray) -> np.ndarray:
    return (query @ H.T) >= 0.0  # (m,) bool


def estimate_count_curve(query: np.ndarray, codes: np.ndarray, hyperplanes: np.ndarray,
                          lut: np.ndarray, r_grid: np.ndarray) -> np.ndarray:
    """N_hat(r) for every r in r_grid, averaged across L tables. r_grid must
    be the exact grid lut was built with (lut's columns are indexed by
    position, not interpolated here) - to evaluate at an arbitrary r not on
    that grid, call this with the full grid and np.interp the result.
    Touches only codes/hyperplanes/lut (the precomputed global summary) -
    never a raw corpus coordinate."""
    L, m, _ = hyperplanes.shape
    total = np.zeros(len(r_grid))
    for l in range(L):
        q_bits = _hash_query(query, hyperplanes[l])
        hamming = (codes[l] != q_bits[None, :]).sum(axis=1)
        counts = np.bincount(hamming, minlength=m + 1)[:m + 1].astype(np.float64)
        total += counts @ lut
    return total / L


def invert_radius(query: np.ndarray, codes: np.ndarray, hyperplanes: np.ndarray,
                   lut: np.ndarray, r_grid: np.ndarray, K: int) -> float:
    """Smallest r on r_grid where the estimated count first reaches K."""
    curve = estimate_count_curve(query, codes, hyperplanes, lut, r_grid)
    idx = np.searchsorted(curve, K, side='left')
    if idx >= len(r_grid):
        return float('inf')
    return float(r_grid[idx])
