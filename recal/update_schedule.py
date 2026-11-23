"""Data-update schedules: held-out inserts and deletes over the test window, uniform or targeted."""

import numpy as np

FRACTIONS = (0.01, 0.05, 0.10)


def build_schedule(vector_cluster: np.ndarray, cluster_queries: np.ndarray, fraction: float, first: int, last: int,
                   mode: str = "uniform", top_clusters: int = 8, seed: int = 0) -> dict:
    """fraction x N inserts (held out initially) and fraction x N deletes, evenly over requests [first, last).

    uniform: vectors drawn from all clusters. targeted: only from the top_clusters clusters with the most
    queries (cluster_queries: query count per cluster).
    """
    n = len(vector_cluster)
    count = int(round(fraction * n))
    if mode == "uniform":
        pool = np.arange(n)
    elif mode == "targeted":
        top = np.argsort(-np.asarray(cluster_queries), kind="stable")[:top_clusters]
        pool = np.flatnonzero(np.isin(vector_cluster, top))
    else:
        raise ValueError(f"unknown update mode {mode}")
    if 2 * count > len(pool):
        raise ValueError(f"{len(pool)} vectors in the update pool, {2 * count} needed")
    chosen = np.random.default_rng([seed, 53]).choice(pool, size=2 * count, replace=False)
    inserts, deletes = chosen[:count], chosen[count:]
    initial = np.ones(n, dtype=bool)
    initial[inserts] = False
    span = last - first
    t_ins = first + (np.arange(count) * span) // max(count, 1)
    t_del = first + (np.arange(count) * span + span // 2) // max(count, 1)
    t = np.r_[t_ins, t_del]
    order = np.argsort(t, kind="stable")
    return {"initial": initial, "t": t[order].astype(np.int64),
            "insert": np.r_[np.ones(count, bool), np.zeros(count, bool)][order],
            "id": np.r_[inserts, deletes][order].astype(np.int64)}


def live_truth(gt_row: np.ndarray, present: np.ndarray, k: int) -> np.ndarray:
    """The exact top-k among the vectors present, from a ground truth row over all vectors."""
    live = gt_row[present[gt_row]]
    if len(live) < k:
        raise ValueError(f"ground truth row has {len(live)} live vectors, {k} needed (deepen the ground truth)")
    return live[:k]


class UpdateStream:
    """Applies the scheduled events up to a request position; keeps the presence mask."""

    def __init__(self, schedule: dict):
        self.s = schedule
        self.present = schedule["initial"].copy()
        self.next = 0

    def due(self, position: int) -> tuple[np.ndarray, np.ndarray]:
        """(inserted ids, deleted ids) scheduled at or before position and not yet applied."""
        j = int(np.searchsorted(self.s["t"], position, side="right"))
        if j <= self.next:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        sl = slice(self.next, j)
        ins, ids = self.s["insert"][sl], self.s["id"][sl]
        self.next = j
        self.present[ids[ins]] = True
        self.present[ids[~ins]] = False
        return ids[ins], ids[~ins]
