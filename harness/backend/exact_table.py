"""Exact backend for answer tables (tests and the exact-backend arm)."""

import numpy as np

from workloads.q1.groundtruth import exact_topk_chunk


class ExactTableBackend:
    name = "exact"

    def __init__(self, base, device: str = "numpy"):
        self.base, self.device = base, device

    def search(self, queries: np.ndarray, list_size: int, ef_search: int) -> np.ndarray:
        ids, _ = exact_topk_chunk(np.asarray(queries, dtype=np.float32), self.base, list_size,
                                  self.device)
        return ids
