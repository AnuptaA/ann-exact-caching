"""Request streams over a universe. A stream is an array of request ids."""

import numpy as np

from workloads.q1.universe import Universe

N_REQUESTS = 100_000
HISTORY_FRACTION = 0.3
ZIPF_EXPONENTS = (0.30, 0.60, 0.99)


def zipf_probabilities(n_groups: int, s: float) -> np.ndarray:
    """P(rank r) proportional to r^(-s), r = 1..n_groups."""
    p = np.arange(1, n_groups + 1, dtype=np.float64) ** (-s)
    return p / p.sum()


def zipf_stream(uni: Universe, n_requests: int, s: float, seed: int = 0) -> np.ndarray:
    """Zipf over groups (random rank permutation), then a uniform member."""
    if s < 0:
        raise ValueError("zipf exponent must be non-negative")
    rng = np.random.default_rng([seed, 2, int(round(s * 1000))])
    group_at_rank = rng.permutation(uni.n_groups)
    ranks = rng.choice(uni.n_groups, size=n_requests, p=zipf_probabilities(uni.n_groups, s))
    members = rng.integers(0, uni.group_size, size=n_requests)
    return group_at_rank[ranks] * uni.group_size + members


def uniform_stream(uni: Universe, n_requests: int, seed: int = 0) -> np.ndarray:
    """Base queries sampled uniformly with replacement."""
    rng = np.random.default_rng([seed, 3])
    return uni.n_members + rng.integers(0, uni.n_groups, size=n_requests)


def no_locality_stream(uni: Universe, seed: int = 0) -> np.ndarray:
    """Every distinct official query once, in random order."""
    rng = np.random.default_rng([seed, 4])
    return uni.n_members + uni.n_groups + rng.permutation(uni.n_official)


def history_split(n_requests: int, history_fraction: float = HISTORY_FRACTION) -> int:
    """Index where the test window starts (requests before it are history)."""
    if not 0.0 <= history_fraction < 1.0:
        raise ValueError("history_fraction must be in [0, 1)")
    return int(round(n_requests * history_fraction))
