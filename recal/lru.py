"""The LRU ablation of the main cache: no greedy selection, entries admitted on every miss.

A miss admits one entry of 2K vectors for its query, with a shell of the next 2K ranks, and evicts
least recently used entries until the bytes fit. An entry counts as used when it contributes a vector
to a served answer. Serving follows the core: read the m nearest centers, union their lists, score the
answer with the offset, size and shell-intruder features. Centers are searched by brute force.
"""

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np

from recal.core import MAIN_HIT, MISS_ABSTAIN, MISS_NO_CANDIDATE, MISS_SCORE

EPS = 1e-12
LEVELS = 15
INDEX_BYTES_PER_ENTRY = 164.0


class LruCache:
    def __init__(self, dim: int, budget: float, coord_bytes: float = 4.0, lo=None, hi=None):
        self.d, self.budget, self.cb = dim, float(budget), float(coord_bytes)
        self.lo = np.zeros(dim, np.float32) if lo is None else np.asarray(lo, np.float32)
        span = np.ones(dim, np.float32) if hi is None else np.maximum(np.asarray(hi, np.float32) - self.lo, EPS)
        self.step = (span / LEVELS).astype(np.float32)
        self.pool: dict = {}       # id -> [vector, refs]
        self.shell: dict = {}      # id -> [codes, refs]
        self.entries: dict = {}    # entry id -> dict(resp, shell, radius, row)
        self.order: OrderedDict = OrderedDict()
        self.centers = np.zeros((64, dim), np.float32)
        self.free_rows: list = []
        self.n_rows = 0
        self.next_id = 0
        self.model = None
        self.lists = 0             # list slots held by entries (response and shell)

    # ---- bookkeeping -----------------------------------------------------------------------------

    def n_entries(self) -> int:
        return len(self.entries)

    def n_data_slots(self) -> int:
        return len(self.pool)

    def entry_sizes(self) -> np.ndarray:
        return np.array([len(e["resp"]) for e in self.entries.values()], dtype=np.int32)

    def bytes_by_structure(self) -> dict:
        d, n_e = self.d, len(self.entries)
        return {
            "data_pool": len(self.pool) * (self.cb * d + 12.0),
            "shell_pool": len(self.shell) * float((d + 1) // 2) + 8.0 * d,
            "entries": n_e * (self.cb * d + 20.0) + 4.0 * self.lists,
            "reverse_map": len(self.pool) * 12.0,
            "center_index": n_e * INDEX_BYTES_PER_ENTRY,
        }

    def total_bytes(self) -> float:
        return sum(self.bytes_by_structure().values())

    def set_model(self, model: dict) -> None:
        self.model = model

    # ---- serving ---------------------------------------------------------------------------------

    def _nearest(self, q: np.ndarray, m: int) -> list:
        rows = np.fromiter((e["row"] for e in self.entries.values()), dtype=np.int64, count=len(self.entries))
        ids = list(self.entries)
        c = self.centers[rows]
        d2 = ((c - q) ** 2).sum(axis=1)
        take = len(ids) if m <= 0 or m >= len(ids) else m
        part = np.argpartition(d2, take - 1)[:take] if take < len(ids) else np.arange(len(ids))
        part = part[np.argsort(d2[part], kind="stable")]
        return [(ids[i], float(np.sqrt(d2[i]))) for i in part]

    def lookup(self, q: np.ndarray, K: int, lam: float | None, m: int) -> SimpleNamespace:
        res = SimpleNamespace(path=MISS_NO_CANDIDATE, ids=np.zeros(0, np.int64), dists=np.zeros(0, np.float32),
                              has_score=False, score=float("nan"), n_read=0, n_candidates=0, n_contrib=-1,
                              best_entry=-1, contrib=[])
        if not self.entries:
            return res
        read = self._nearest(q, m)
        res.n_read = len(read)
        cand = {}
        for eid, _ in read:
            for i in self.entries[eid]["resp"]:
                if i not in cand:
                    cand[i] = float(np.linalg.norm(self.pool[i][0].astype(np.float64) - q))
        res.n_candidates = len(cand)
        if len(cand) < K:
            return res
        order = sorted(cand, key=lambda i: (cand[i], i))[:K]
        res.ids = np.array(order, np.int64)
        res.dists = np.array([cand[i] for i in order], np.float32)
        nr = max(cand[order[-1]], EPS)
        served = set(order)
        best, best_e, best_de, best_rad = np.inf, -1, 0.0, 1.0
        for eid, de in read:
            rad = max(self.entries[eid]["radius"], EPS)
            fit = (de + nr) / rad
            if fit < best:
                best, best_e, best_de, best_rad = fit, eid, de, rad
        res.best_entry = best_e
        res.contrib = [eid for eid, _ in read if served & set(self.entries[eid]["resp"])]
        res.n_contrib = len(res.contrib)
        offset, size = best_de / best_rad, nr / best_rad
        seen, count = set(), 0
        for eid, _ in read:
            for s in self.entries[eid]["shell"]:
                if s in seen:
                    continue
                seen.add(s)
                v = self.lo + self.shell[s][0].astype(np.float32) * self.step
                if float(np.sum((q.astype(np.float64) - v) ** 2)) < nr * nr:
                    count += 1
        rate = np.log((count + 1.0) / K)
        res.features = np.array([offset, size, rate])
        if self.model is not None:
            z = self.model["intercept"] + float(((res.features - self.model["mean"]) / self.model["scale"])
                                                 @ self.model["coef"])
            res.score, res.has_score = 1.0 / (1.0 + np.exp(-z)), True
        res.path = MISS_ABSTAIN if lam is None else (MAIN_HIT if res.has_score and res.score >= lam else MISS_SCORE)
        if res.path == MISS_ABSTAIN and not res.has_score:
            res.path = MISS_ABSTAIN
        return res

    def touch(self, entry_ids) -> None:
        for eid in entry_ids:
            if eid in self.order:
                self.order.move_to_end(eid)

    # ---- admission and eviction ------------------------------------------------------------------

    def _quantize(self, v: np.ndarray) -> np.ndarray:
        return np.clip(np.rint((v - self.lo) / self.step), 0, LEVELS).astype(np.uint8)

    def admit(self, center: np.ndarray, K: int, ids, dists, vectors) -> int:
        """Entry from the first 2K ranks, shell from the next 2K; returns the vectors newly held."""
        n = 2 * K
        ids = np.asarray(ids, np.int64)
        new = 0
        resp = [int(i) for i in ids[:n]]
        for i, v in zip(resp, vectors[:n]):
            if i in self.pool:
                self.pool[i][1] += 1
            else:
                self.pool[i] = [np.asarray(v, np.float32).copy(), 1]
                new += 1
        shell = []
        for i, v in zip(ids[n:2 * n].tolist(), vectors[n:2 * n]):
            if i in self.pool:
                continue
            if i in self.shell:
                self.shell[i][1] += 1
            else:
                self.shell[i] = [self._quantize(np.asarray(v, np.float32)), 1]
                new += 1
            shell.append(i)
        if self.free_rows:
            row = self.free_rows.pop()
        else:
            if self.n_rows == len(self.centers):
                self.centers = np.vstack([self.centers, np.zeros_like(self.centers)])
            row = self.n_rows
            self.n_rows += 1
        self.centers[row] = center
        eid = self.next_id
        self.next_id += 1
        self.entries[eid] = {"resp": resp, "shell": shell, "radius": float(dists[n - 1]), "row": row, "K": K}
        self.order[eid] = None
        self.lists += len(resp) + len(shell)
        while self.total_bytes() > self.budget and len(self.entries) > 1:
            self._evict(next(iter(self.order)))
        return new

    def _evict(self, eid: int) -> None:
        e = self.entries.pop(eid)
        self.order.pop(eid)
        self.free_rows.append(e["row"])
        self.lists -= len(e["resp"]) + len(e["shell"])
        for i in e["resp"]:
            self.pool[i][1] -= 1
            if self.pool[i][1] == 0:
                del self.pool[i]
        for i in e["shell"]:
            self.shell[i][1] -= 1
            if self.shell[i][1] == 0:
                del self.shell[i]
