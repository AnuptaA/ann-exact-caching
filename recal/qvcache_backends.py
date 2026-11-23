"""Backend adapters with the interface of the QVCache artifact: search(query, K) and fetch_vectors_by_ids(ids).

The artifact compares squared L2 distances (DiskANN's convention), so the adapters return squared distances.
Every fetch_vectors_by_ids call is a vector re-read (QVCache reads the vectors it inserts through it); the
counters give searches, re-reads and the time spent in the backend, so the driver can attribute them per request.
"""

import time

import numpy as np


class CountingBackend:
    def __init__(self):
        self.searches = 0
        self.depth_sum = 0
        self.rereads = 0
        self.seconds = 0.0

    def snapshot(self) -> tuple:
        return self.searches, self.rereads, self.seconds

    def stats(self) -> dict:
        return {"miss": {"searches": self.searches, "depth_sum": self.depth_sum}}


class StoredBackend(CountingBackend):
    """Stored answers (exact, or an approximate backend's answer tables): the hit-rate runs."""

    def __init__(self, backend):
        super().__init__()
        self.b = backend

    def search(self, query, K: int):
        t0 = time.perf_counter()
        ids, dists = self.b.search(np.asarray(query, dtype=np.float32), int(K), kind="miss")
        self.seconds += time.perf_counter() - t0
        self.searches += 1
        self.depth_sum += int(K)
        return np.ascontiguousarray(ids, dtype=np.uint32), np.ascontiguousarray(np.asarray(dists) ** 2, np.float32)

    def fetch_vectors_by_ids(self, ids):
        t0 = time.perf_counter()
        vectors = self.b.vectors_of(np.asarray(list(ids), dtype=np.int64))
        self.seconds += time.perf_counter() - t0
        self.rereads += len(vectors)
        return [np.ascontiguousarray(v, dtype=np.float32) for v in vectors]


class LiveBackend(CountingBackend):
    """A live DiskANN index (searches) or pgvector (LivePgvectorBackend) with vectors read from the base array."""

    def __init__(self, search_fn, base):
        super().__init__()
        self.search_fn, self.base = search_fn, base

    def search(self, query, K: int):
        t0 = time.perf_counter()
        ids, dists = self.search_fn(np.asarray(query, dtype=np.float32), int(K))
        self.seconds += time.perf_counter() - t0
        self.searches += 1
        self.depth_sum += int(K)
        return np.ascontiguousarray(ids, dtype=np.uint32), np.ascontiguousarray(dists, dtype=np.float32)

    def fetch_vectors_by_ids(self, ids):
        t0 = time.perf_counter()
        vectors = [np.asarray(self.base[int(i)], dtype=np.float32) for i in ids]
        self.seconds += time.perf_counter() - t0
        self.rereads += len(vectors)
        return vectors
