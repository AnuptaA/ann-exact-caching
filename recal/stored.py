"""A backend that serves stored answers: exact top-R tables, DiskANN answer rows or a pgvector answer table.

The answer of a request is looked up by the hash of its query vector. The ids come from the stored table,
the distances are recomputed from the database vectors, so every path returns ranked ids with distances (X1).
"""

import numpy as np

from recal.backend import ExactBackend


class StoredAnswers(ExactBackend):
    """Stored answers for known query vectors; unknown vectors raise (no live search is available).

    `answers[i]` holds the ranked ids stored for `queries[i]`; depth requests up to the stored length.
    """

    def __init__(self, vectors, queries: np.ndarray, answers: np.ndarray, ids=None, shallow=None):
        """`shallow` (optional): the answers of the K the backend was tuned for (for example DiskANN's
        answer rows), used for depth up to its width; `answers` holds the deeper lists."""
        super().__init__(vectors, ids)
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        self.shallow = None if shallow is None else np.asarray(shallow, dtype=np.int64)
        self.answers = np.asarray(answers, dtype=np.int64)
        self.row = {queries[i].tobytes(): i for i in range(len(queries))}
        self.depth = self.answers.shape[1]

    def search(self, queries: np.ndarray, depth: int, kind: str = "serve"):
        single = queries.ndim == 1
        q = np.atleast_2d(np.ascontiguousarray(queries, dtype=np.float32))
        if depth > self.depth:
            raise ValueError(f"stored answers hold {self.depth} ids, {depth} requested")
        self._count(kind, len(q), depth)
        ids = np.empty((len(q), depth), dtype=np.int64)
        dists = np.empty((len(q), depth), dtype=np.float32)
        for i, v in enumerate(q):
            r = self.row.get(v.tobytes())
            if r is None:
                raise KeyError("no stored answer for this query vector")
            table = self.shallow if self.shallow is not None and depth <= self.shallow.shape[1] else self.answers
            ids[i] = table[r, :depth]
            rows = self.vectors_of(ids[i]).astype(np.float64)
            dists[i] = np.linalg.norm(rows - v.astype(np.float64), axis=1)
        return (ids[0], dists[0]) if single else (ids, dists)


def from_pgvector(vectors, queries: np.ndarray, table_dir):
    """Stored answers from main's pgvector answer table (harness/backend/answer_table.py): block files of ranked
    ids for the request vectors, in the order of `queries`. Needs the harness package, not PostgreSQL."""
    from harness.backend.answer_table import load_answer_table
    ids, _ = load_answer_table(table_dir)
    return StoredAnswers(vectors, queries[:len(ids)], ids)
