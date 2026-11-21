"""Proximity thresholds: percentiles of inter-query squared Euclidean distances."""

import numpy as np

from workloads.q1.distance import sq_l2_rows

MAX_PAIRS = 50_000_000
PAIR_CHUNK = 1_000_000


def pairwise_sq_percentiles(
    vectors: np.ndarray,
    percentiles: tuple[float, ...],
    max_pairs: int = MAX_PAIRS,
    seed: int = 0,
) -> dict[float, float]:
    """Percentiles (in percent) of squared distances over distinct pairs.

    Uses all pairs when there are at most max_pairs of them, otherwise max_pairs
    pairs sampled with a fixed seed.
    """
    n = len(vectors)
    if n < 2:
        raise ValueError("need at least two vectors")
    total = n * (n - 1) // 2
    if total <= max_pairs:
        dists = np.concatenate([
            sq_l2_rows(np.broadcast_to(vectors[i], (n - i - 1, vectors.shape[1])), vectors[i + 1:])
            for i in range(n - 1)])
    else:
        rng = np.random.default_rng(seed)
        parts = []
        for start in range(0, max_pairs, PAIR_CHUNK):
            m = min(PAIR_CHUNK, max_pairs - start)
            i = rng.integers(0, n, size=m)
            j = (i + rng.integers(1, n, size=m)) % n   # never the same vector
            parts.append(sq_l2_rows(vectors[i], vectors[j]))
        dists = np.concatenate(parts)
    return {p: float(np.percentile(dists, p)) for p in percentiles}


def history_vectors(stream_ids: np.ndarray, history_len: int, vectors_of) -> np.ndarray:
    """Vectors of the distinct queries among the first history_len requests."""
    unique = np.unique(np.asarray(stream_ids)[:history_len])
    return vectors_of(unique)
