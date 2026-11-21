"""Backend interface. Every system reads answers from a table built through it."""

from typing import Protocol

import numpy as np


class Backend(Protocol):
    name: str

    def search(self, queries: np.ndarray, list_size: int, ef_search: int) -> np.ndarray:
        """Top `list_size` ids per query, nearest first, shape (n, list_size)."""
        ...


class DiskANNBackend:
    """Scaffold for a DiskANN backend (not implemented for Q1).

    A DiskANN backend would build a Vamana index over the same float32 base with
    ids equal to base row positions, expose its search list size as `ef_search`,
    and return the same (n, list_size) id arrays, so tables and the runner stay
    unchanged. Keep recall@10 tuned to about 0.9 with the same ef_search rule.
    """

    name = "diskann"

    def search(self, queries, list_size, ef_search):
        raise NotImplementedError("DiskANN is out of scope for Q1")
