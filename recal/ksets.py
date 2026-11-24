"""Request-size streams: a fixed K, or a K tied to the query.

Mixed K: every distinct query vector gets one K drawn uniformly from K_SET, seeded by a hash of the vector bytes,
so exact repeats keep their K and different queries (perturbations included) draw independently.
"""

import hashlib

import numpy as np

K_SET = (5, 10, 25, 50, 75, 100)


def k_sequence(n: int, k: int) -> np.ndarray:
    return np.full(n, int(k), dtype=np.int32)


def k_for_vector(vector, k_set=K_SET, seed: int = 0) -> int:
    digest = hashlib.blake2b(np.ascontiguousarray(vector, dtype=np.float32).tobytes() + int(seed).to_bytes(8, "little"),
                             digest_size=8).digest()
    return int(k_set[int.from_bytes(digest, "little") % len(k_set)])


def k_by_vectors(vectors, k_set=K_SET, seed: int = 0) -> np.ndarray:
    """One K per row; identical rows get the same K."""
    return np.array([k_for_vector(v, k_set, seed) for v in vectors], dtype=np.int32)


def k_by_query(q, idx, k_set=K_SET, seed: int = 0) -> np.ndarray:
    """K of every request of a stream (idx: query index per request, q: the query vectors)."""
    unique, inverse = np.unique(np.asarray(idx, dtype=np.int64), return_inverse=True)
    per_query = np.array([k_for_vector(q[i], k_set, seed) for i in unique], dtype=np.int32)
    return per_query[inverse.ravel()]
