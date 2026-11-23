"""DiskANN node cache sizing at stored precision, and node choice from history queries."""

import numpy as np

HEADER_BYTES = 4 + 8   # neighbor count and node header


def node_bytes(dim: int, coord_bytes: float, degree: int) -> float:
    """One cached node: its vector at stored precision, its graph links (4 bytes each) and the header."""
    return dim * coord_bytes + 4.0 * degree + HEADER_BYTES


def nodes_for_bytes(budget: float, dim: int, coord_bytes: float, degree: int) -> int:
    return max(0, int(budget // node_bytes(dim, coord_bytes, degree)))


def nodes_from_index(budget: float, max_node_len: int) -> int:
    """With the index header's max_node_len (vector, links and header of the on-disk node)."""
    return max(0, int(budget // max_node_len))


def pick_nodes(visits: list, n_nodes: int) -> np.ndarray:
    """The n_nodes most visited nodes over the history queries (visits: one id array per query)."""
    if n_nodes <= 0 or not len(visits):
        return np.zeros(0, np.int64)
    ids, counts = np.unique(np.concatenate([np.asarray(v, np.int64) for v in visits]), return_counts=True)
    order = np.lexsort((ids, -counts))
    return ids[order[:n_nodes]]
