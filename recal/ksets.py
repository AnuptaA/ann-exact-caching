"""Request-size streams: a fixed K, or K drawn per request from a set."""

import numpy as np

K_SET = (5, 10, 50, 100)


def k_sequence(n: int, k: int, mixed: bool = False, k_set=K_SET, seed: int = 0) -> np.ndarray:
    if not mixed:
        return np.full(n, int(k), dtype=np.int32)
    return np.random.default_rng([seed, 41]).choice(np.asarray(k_set, dtype=np.int32), size=n)
