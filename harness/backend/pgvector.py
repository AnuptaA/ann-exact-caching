"""pgvector HNSW backend (PostgreSQL 16, pgvector 0.8.0).

pgvector silently answers exactly when the planner picks a sequential scan, so
every session disables seq scans and asserts that the HNSW index is used.
"""

import numpy as np

EF_SEARCH_MAX = 1000   # pgvector's upper limit for hnsw.ef_search
EXPECTED_VERSION = "0.8.0"


def extension_version(conn) -> str:
    with conn.cursor() as cur:
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        row = cur.fetchone()
    if row is None:
        raise RuntimeError("the vector extension is not installed in this database")
    return row[0]


def check_version(conn, expected: str = EXPECTED_VERSION) -> str:
    """Fail unless pgvector is the expected version (override only on purpose)."""
    found = extension_version(conn)
    if found != expected:
        raise RuntimeError(f"pgvector {found} installed, expected {expected}")
    return found


def expected_result_count(ef_search: int, limit: int, n_vectors: int) -> int:
    """How many rows an HNSW scan can return: at most ef_search (iterative scans off)."""
    return min(limit, ef_search, n_vectors)


def vector_literal(v: np.ndarray) -> str:
    return "[" + ",".join(f"{float(x):.9g}" for x in v) + "]"


class PgvectorBackend:
    name = "pgvector-hnsw"

    def __init__(self, conn, table: str = "base", column: str = "embedding",
                 index_name: str | None = None):
        self.conn = conn
        self.table, self.column = table, column
        self.index_name = index_name or f"{table}_{column}_hnsw"
        self._ready = False

    def _setup(self, cur, ef_search: int) -> None:
        if not 1 <= ef_search <= EF_SEARCH_MAX:
            raise ValueError(f"ef_search must be in [1, {EF_SEARCH_MAX}], got {ef_search}")
        cur.execute("SET enable_seqscan = off")
        cur.execute("SET hnsw.iterative_scan = off")
        cur.execute(f"SET hnsw.ef_search = {int(ef_search)}")

    def assert_index_used(self, cur, query: np.ndarray, list_size: int) -> None:
        """Fail unless EXPLAIN shows the HNSW index for this kind of query."""
        cur.execute(f"EXPLAIN SELECT id FROM {self.table} ORDER BY {self.column} <-> "
                    f"%s::vector LIMIT {int(list_size)}", (vector_literal(query),))
        plan = "\n".join(row[0] for row in cur.fetchall())
        if self.index_name not in plan or "Index Scan" not in plan:
            raise RuntimeError(f"query does not use index {self.index_name}:\n{plan}")

    def search(self, queries: np.ndarray, list_size: int, ef_search: int) -> np.ndarray:
        out = np.full((len(queries), list_size), -1, dtype=np.int64)
        with self.conn.cursor() as cur:
            self._setup(cur, ef_search)
            if len(queries) and not self._ready:
                self.assert_index_used(cur, queries[0], list_size)
                self._ready = True
            for i, q in enumerate(queries):
                cur.execute(f"SELECT id FROM {self.table} ORDER BY {self.column} <-> "
                            f"%s::vector LIMIT {int(list_size)}", (vector_literal(q),))
                ids = [row[0] for row in cur.fetchall()]
                out[i, :len(ids)] = ids
        return out
