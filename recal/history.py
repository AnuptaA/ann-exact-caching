"""History log and hash log on disk (B1, B2).

The history holds the last H main-path requests: query vector, K, kind and the backend's response when it
has one. The hash log holds the 64-bit hash of every request, mini cache hits included, for the last H
requests. Both are ring buffers in memory-mapped files, so they cost no RAM against the budget (S10).
"""

from pathlib import Path

import numpy as np

MISS, HIT, AUDITED = 0, 1, 2   # record kinds: miss, unaudited main-cache hit, audited hit


class HistoryLog:
    def __init__(self, directory: str | Path, dim: int, capacity: int, width: int):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dim, self.capacity, self.width = dim, capacity, width
        self.count = 0
        mm = lambda name, dtype, shape: np.lib.format.open_memmap(  # noqa: E731
            self.dir / name, mode="w+", dtype=dtype, shape=shape)
        self.vec = mm("history_vec.npy", np.float32, (capacity, dim))
        self.k = mm("history_k.npy", np.int32, (capacity,))
        self.kind = mm("history_kind.npy", np.uint8, (capacity,))
        self.version = mm("history_version.npy", np.int64, (capacity,))   # data version of the response
        self.resp_ids = mm("history_resp_ids.npy", np.int64, (capacity, width))
        self.resp_dists = mm("history_resp_dists.npy", np.float32, (capacity, width))
        self.has_resp = mm("history_has_resp.npy", np.uint8, (capacity,))
        self.hashes = mm("hash_log.npy", np.uint64, (capacity,))
        self.hash_count = 0

    def __len__(self) -> int:
        return min(self.count, self.capacity)

    def append_hash(self, h: int) -> None:
        self.hashes[self.hash_count % self.capacity] = np.uint64(h)
        self.hash_count += 1

    def append(self, vector, K: int, kind: int, ids=None, dists=None, version: int = 0) -> int:
        """Add one main-path request; returns its absolute position."""
        pos = self.count
        i = pos % self.capacity
        self.vec[i] = vector
        self.k[i] = K
        self.kind[i] = kind
        if ids is not None:
            n = min(len(ids), self.width)
            self.resp_ids[i, :n] = ids[:n]
            self.resp_dists[i, :n] = dists[:n]
            self.has_resp[i] = 1
            self.version[i] = version
        else:
            self.has_resp[i] = 0
        self.count += 1
        return pos

    def order(self) -> np.ndarray:
        """Physical indices of the stored records, oldest first."""
        n = len(self)
        start = self.count % self.capacity if self.count > self.capacity else 0
        return (start + np.arange(n)) % self.capacity

    def snapshot(self) -> dict:
        idx = self.order()
        return {"vec": np.asarray(self.vec[idx]), "k": np.asarray(self.k[idx]), "kind": np.asarray(self.kind[idx]),
                "ids": np.asarray(self.resp_ids[idx]), "dists": np.asarray(self.resp_dists[idx]),
                "has": np.asarray(self.has_resp[idx]).astype(bool), "version": np.asarray(self.version[idx]),
                "index": idx}

    def store_response(self, index: np.ndarray, ids: np.ndarray, dists: np.ndarray, version: int) -> None:
        """Write fetched responses into physical records."""
        n = min(ids.shape[1], self.width)
        self.resp_ids[index, :n] = ids[:, :n]
        self.resp_dists[index, :n] = dists[:, :n]
        self.has_resp[index] = 1
        self.version[index] = version

    def hash_snapshot(self) -> np.ndarray:
        n = min(self.hash_count, self.capacity)
        start = self.hash_count % self.capacity if self.hash_count > self.capacity else 0
        return np.asarray(self.hashes[(start + np.arange(n)) % self.capacity])

    def flush(self) -> None:
        for array in (self.vec, self.k, self.kind, self.version, self.resp_ids, self.resp_dists, self.has_resp,
                      self.hashes):
            array.flush()


def reuse_distances(hashes: np.ndarray) -> np.ndarray:
    """For each request that repeats an earlier one: distinct queries since its previous occurrence."""
    n = len(hashes)
    tree = np.zeros(n + 1, dtype=np.int64)

    def add(i: int, v: int) -> None:
        i += 1
        while i <= n:
            tree[i] += v
            i += i & -i

    def prefix(i: int) -> int:   # sum of positions [0, i)
        s = 0
        while i > 0:
            s += tree[i]
            i -= i & -i
        return s

    last: dict = {}
    out = []
    for pos, h in enumerate(hashes.tolist()):
        prev = last.get(h)
        if prev is not None:
            out.append(prefix(pos) - prefix(prev + 1))   # distinct queries strictly between the two
            add(prev, -1)
        add(pos, 1)
        last[h] = pos
    return np.asarray(out, dtype=np.int64)
