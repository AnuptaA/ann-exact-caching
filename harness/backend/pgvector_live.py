"""pgvector backend with live inserts and deletes, for ReCal's update experiments (cluster only).

Searches, inserts and deletes go to the table that holds the HNSW index, so a miss or an audit sees the
current data. Ids are never reused after a delete.
"""

import numpy as np

from harness.backend.pgvector import vector_literal


class LivePgvectorBackend:
    def __init__(self, conn, dim: int, n_vectors: int, table: str = "base", column: str = "embedding",
                 ef_search: int = 1000):
        self.conn, self.dim, self.table, self.column = conn, dim, table, column
        self.ef_search, self.version, self.n = ef_search, 0, n_vectors
        self.stats: dict = {}
        self._deleted: set = set()

    def __len__(self) -> int:
        return self.n

    def _count(self, kind: str, n: int, depth: int) -> None:
        e = self.stats.setdefault(kind, {"searches": 0, "depth_sum": 0})
        e["searches"] += n
        e["depth_sum"] += n * depth

    def search(self, queries: np.ndarray, depth: int, kind: str = "serve"):
        single = queries.ndim == 1
        q = np.atleast_2d(np.asarray(queries, dtype=np.float32))
        self._count(kind, len(q), depth)
        ids = np.full((len(q), depth), -1, dtype=np.int64)
        dists = np.full((len(q), depth), np.inf, dtype=np.float32)
        with self.conn.cursor() as cur:
            cur.execute("SET enable_seqscan = off")
            cur.execute(f"SET hnsw.ef_search = {int(self.ef_search)}")
            for i, v in enumerate(q):
                cur.execute(f"SELECT id, {self.column} <-> %s::vector FROM {self.table} "
                            f"ORDER BY {self.column} <-> %s::vector LIMIT {int(depth)}",
                            (vector_literal(v), vector_literal(v)))
                rows = cur.fetchall()
                ids[i, :len(rows)] = [r[0] for r in rows]
                dists[i, :len(rows)] = [r[1] for r in rows]
        return (ids[0], dists[0]) if single else (ids, dists)

    def vectors_of(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT id, {self.column}::real[] FROM {self.table} WHERE id = ANY(%s)", (ids.tolist(),))
            found = {r[0]: np.asarray(r[1], dtype=np.float32) for r in cur.fetchall()}
        return np.stack([found[int(i)] for i in ids.ravel()]).reshape(ids.shape + (self.dim,))

    def insert(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        ids = np.asarray(ids, dtype=np.int64)
        if self._deleted & set(ids.tolist()):
            raise ValueError("database ids are never reused")
        with self.conn.cursor() as cur:
            for i, v in zip(ids.tolist(), np.asarray(vectors, dtype=np.float32)):
                cur.execute(f"INSERT INTO {self.table} (id, {self.column}) VALUES (%s, %s::vector)",
                            (i, vector_literal(v)))
        self.conn.commit()
        self.n += len(ids)
        self.version += 1

    def delete(self, ids: np.ndarray) -> None:
        ids = np.asarray(ids, dtype=np.int64)
        with self.conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.table} WHERE id = ANY(%s)", (ids.tolist(),))
        self.conn.commit()
        self._deleted |= set(ids.tolist())
        self.n -= len(ids)
        self.version += 1
