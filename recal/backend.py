"""Backends: any top-K search. ExactBackend is brute force (answer tables from exact search)."""

import numpy as np


class ExactBackend:
    """Exact Euclidean top-K over an in-memory array. Counts the searches it serves by kind."""

    def __init__(self, vectors: np.ndarray, ids: np.ndarray | None = None, block: int = 2048):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self._identity = ids is None
        self.ids = np.arange(len(vectors), dtype=np.int64) if ids is None else np.asarray(ids, dtype=np.int64)
        self.block = block
        self.version = 0
        self.dim = self.vectors.shape[1]
        self.stats: dict[str, dict] = {}
        self._norms = np.einsum("ij,ij->i", self.vectors, self.vectors)

    def __len__(self) -> int:
        return len(self.vectors)

    def _count(self, kind: str, n: int, depth: int) -> None:
        entry = self.stats.setdefault(kind, {"searches": 0, "depth_sum": 0})
        entry["searches"] += n
        entry["depth_sum"] += n * depth

    def search(self, queries: np.ndarray, depth: int, kind: str = "serve") -> tuple[np.ndarray, np.ndarray]:
        """Ranked ids and Euclidean distances (float32) of the depth nearest vectors."""
        single = queries.ndim == 1
        q = np.atleast_2d(np.asarray(queries, dtype=np.float32))
        depth = min(depth, len(self.vectors))
        self._count(kind, len(q), depth)
        out_ids = np.empty((len(q), depth), dtype=np.int64)
        out_d = np.empty((len(q), depth), dtype=np.float32)
        for s in range(0, len(q), self.block):
            qb = q[s:s + self.block]
            sq = self._norms[None, :] - 2.0 * (qb @ self.vectors.T) + np.einsum("ij,ij->i", qb, qb)[:, None]
            part = np.argpartition(sq, depth - 1, axis=1)[:, :depth] if depth < sq.shape[1] else \
                np.tile(np.arange(sq.shape[1]), (len(qb), 1))
            for i in range(len(qb)):
                rows = self.vectors[part[i]].astype(np.float64)
                dist = np.linalg.norm(rows - qb[i].astype(np.float64), axis=1)
                order = np.argsort(dist, kind="stable")
                out_ids[s + i] = self.ids[part[i][order]]
                out_d[s + i] = dist[order]
        return (out_ids[0], out_d[0]) if single else (out_ids, out_d)

    def vectors_of(self, ids: np.ndarray) -> np.ndarray:
        """Vectors of database ids (ids are unique, ascending positions are not assumed)."""
        pos = self._positions(ids)
        return self.vectors[pos]

    def _positions(self, ids: np.ndarray) -> np.ndarray:
        if self._identity:
            return np.asarray(ids, dtype=np.int64)
        if not hasattr(self, "_pos"):
            self._pos = {int(v): i for i, v in enumerate(self.ids)}
        return np.array([self._pos[int(i)] for i in np.asarray(ids).ravel()], dtype=np.int64).reshape(np.shape(ids))

    # ---- data changes (M3): ids are never reused after a delete (X6) ---------------------------------

    def insert(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        ids = np.asarray(ids, dtype=np.int64)
        if np.isin(ids, self.ids).any() or (hasattr(self, "_deleted") and np.isin(ids, list(self._deleted)).any()):
            raise ValueError("database ids are never reused")
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.vectors = np.vstack([self.vectors, vectors])
        self.ids = np.r_[self.ids, ids]
        self._norms = np.einsum("ij,ij->i", self.vectors, self.vectors)
        self._identity = False
        self._reindex()

    def delete(self, ids: np.ndarray) -> None:
        ids = np.asarray(ids, dtype=np.int64)
        keep = ~np.isin(self.ids, ids)
        self._deleted = getattr(self, "_deleted", set()) | set(ids.tolist())
        self.vectors, self.ids, self._norms = self.vectors[keep], self.ids[keep], self._norms[keep]
        self._identity = False
        self._reindex()

    def _reindex(self) -> None:
        self._pos = {int(v): i for i, v in enumerate(self.ids)}
        self.version += 1
