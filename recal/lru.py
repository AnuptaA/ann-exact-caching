"""The LRU ablation of the main cache: no greedy selection, entries admitted on every miss.

A miss admits one entry of 2K vectors for its query, with a shell of the next 2K ranks, and evicts
least recently used entries until the bytes fit. An entry counts as used when it contributes a vector
to a served answer. Serving follows the core: read the m nearest centers (exact search), union their lists,
score the answer with the offset, size and shell-intruder features. Shell vectors that are also cached are not
counted, as the core never stores them as shell vectors.

Vectors live in arrays (a slot per cached id), so a lookup is a few array operations; candidate distances use
a float32 shortlist that is re-ranked in float64, with ties to the lower id, like the core.
"""

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np

from recal.core import MAIN_HIT, MISS_ABSTAIN, MISS_NO_CANDIDATE, MISS_SCORE
from recal.shell_quant import default_step, shell_codes, shell_step

EPS = 1e-12
INDEX_BYTES_PER_ENTRY = 164.0
SHORTLIST = 3
SHELL_CHUNK = 20_000
START_SLOTS = 1024


class LruCache:
    def __init__(self, dim: int, budget: float, coord_bytes: float = 4.0, lo=None, hi=None):
        self.d, self.budget, self.cb = dim, float(budget), float(coord_bytes)
        self.lo = np.zeros(dim, np.float32) if lo is None else np.asarray(lo, np.float32)
        # the core's quantizer: float64 span, step rounded to float32, codes rounded half away from zero
        self.step = default_step(dim) if hi is None else shell_step(self.lo, hi)
        self.pool: dict = {}       # id -> pool slot
        self.shell: dict = {}      # id -> shell slot
        self.entries: dict = {}    # entry id -> dict(resp (pool slots), shell (shell slots), radius, row, K)
        self.order: OrderedDict = OrderedDict()
        self.centers = np.zeros((64, dim), np.float32)
        self.row_entry = np.full(64, -1, np.int64)
        self.free_rows: list = []
        self.n_rows = 0
        self.next_id = 0
        self.model = None
        self.lists = 0             # list slots held by entries (response and shell)
        self.holders: dict = {}    # pool slot -> set of entry ids listing it
        self._all_resp = None
        self._init_slots()

    def _init_slots(self) -> None:
        d = self.d
        self.pvec = np.zeros((START_SLOTS, d), np.float32)
        self.pid = np.zeros(START_SLOTS, np.int64)
        self.pref = np.zeros(START_SLOTS, np.int32)
        self.psq = np.zeros(START_SLOTS, np.float64)
        self.pfree: list = []
        self.n_pslots = 0
        self.scode = np.zeros((START_SLOTS, d), np.uint8)
        self.sid = np.zeros(START_SLOTS, np.int64)
        self.sref = np.zeros(START_SLOTS, np.int32)
        self.s_pooled = np.zeros(START_SLOTS, bool)
        self.sfree: list = []
        self.n_sslots = 0

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
        """The m entries with the nearest centers, nearest first, with exact (float64) center distances."""
        rows = np.flatnonzero(self.row_entry[:self.n_rows] >= 0)
        ids = self.row_entry[rows]
        d = np.linalg.norm(self.centers[rows].astype(np.float64) - q.astype(np.float64), axis=1)
        take = len(ids) if m <= 0 or m >= len(ids) else m
        part = np.argpartition(d, take - 1)[:take] if take < len(ids) else np.arange(len(ids))
        part = part[np.argsort(d[part], kind="stable")]
        return [(int(ids[i]), float(d[i])) for i in part]

    def _resp_union(self, read: list, m: int) -> np.ndarray:
        if m <= 0 or m >= len(self.entries):
            if self._all_resp is None:
                self._all_resp = np.unique(np.concatenate([e["resp"] for e in self.entries.values()]))
            return self._all_resp
        return np.unique(np.concatenate([self.entries[eid]["resp"] for eid, _ in read]))

    def _nearest_candidates(self, q: np.ndarray, slots: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
        """The K nearest of the pool slots: float32 shortlist, float64 re-rank, ties to the lower id."""
        approx = self.psq[slots] - 2.0 * (self.pvec[slots] @ q.astype(np.float32)) + float(q.astype(np.float64) @ q)
        s = min(SHORTLIST * K, len(slots))
        short = slots[np.argpartition(approx, s - 1)[:s]] if s < len(slots) else slots
        dist = np.linalg.norm(self.pvec[short].astype(np.float64) - q.astype(np.float64), axis=1)
        order = np.lexsort((self.pid[short], dist))[:K]
        return short[order], dist[order]

    def _intruders(self, q: np.ndarray, read: list, nr: float) -> int:
        parts = [self.entries[eid]["shell"] for eid, _ in read if len(self.entries[eid]["shell"])]
        if not parts:
            return 0
        slots = np.unique(np.concatenate(parts))
        slots = slots[~self.s_pooled[slots]]
        q64 = q.astype(np.float64)
        lo, step = self.lo.astype(np.float64), self.step.astype(np.float64)
        count = 0
        for i in range(0, len(slots), SHELL_CHUNK):
            v = lo + self.scode[slots[i:i + SHELL_CHUNK]].astype(np.float64) * step
            count += int((((q64 - v) ** 2).sum(axis=1) < nr * nr).sum())
        return count

    def lookup(self, q: np.ndarray, K: int, lam: float | None, m: int) -> SimpleNamespace:
        res = SimpleNamespace(path=MISS_NO_CANDIDATE, ids=np.zeros(0, np.int64), dists=np.zeros(0, np.float32),
                              has_score=False, score=float("nan"), n_read=0, n_candidates=0, n_contrib=-1,
                              best_entry=-1, contrib=[])
        if not self.entries:
            return res
        read = self._nearest(q, m)
        res.n_read = len(read)
        slots = self._resp_union(read, m)
        res.n_candidates = len(slots)
        if len(slots) < K:
            return res
        near, dist = self._nearest_candidates(q, slots, K)
        res.ids = self.pid[near].copy()
        res.dists = dist.astype(np.float32)
        nr = max(float(dist[-1]), EPS)
        best, best_e, best_de, best_rad = np.inf, -1, 0.0, 1.0
        for eid, de in read:
            rad = max(self.entries[eid]["radius"], EPS)
            fit = (de + nr) / rad
            if fit < best:
                best, best_e, best_de, best_rad = fit, eid, de, rad
        res.best_entry = best_e
        held = set().union(*(self.holders[int(s)] for s in near))
        res.contrib = [eid for eid, _ in read if eid in held]
        res.n_contrib = len(res.contrib)
        offset, size = best_de / best_rad, nr / best_rad
        count = self._intruders(q, read, nr)
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

    def entries_needed(self, queries: np.ndarray, truth: np.ndarray) -> np.ndarray:
        """Per query, the entries (nearest center first) that reach every true neighbor the pool holds.

        -1 when none of its neighbors is pooled (the core's entries_needed_batch).
        """
        out = np.full(len(queries), -1, dtype=np.int64)
        if not self.entries:
            return out
        ids = list(self.entries)
        rows = np.array([self.entries[e]["row"] for e in ids])
        holders: dict = {}
        for pos, e in enumerate(ids):
            for i in self.pid[self.entries[e]["resp"]].tolist():
                holders.setdefault(i, []).append(pos)
        for n, q in enumerate(queries):
            slots = [holders[i] for i in truth[n].tolist() if i in holders]
            if not slots:
                continue
            d2 = ((self.centers[rows] - q) ** 2).sum(axis=1)
            rank = np.empty(len(ids), dtype=np.int64)
            rank[np.argsort(d2, kind="stable")] = np.arange(len(ids))
            out[n] = max(min(rank[p] for p in held) + 1 for held in slots)
        return out

    def touch(self, entry_ids) -> None:
        for eid in entry_ids:
            if eid in self.order:
                self.order.move_to_end(eid)

    # ---- admission and eviction ------------------------------------------------------------------

    def _quantize(self, v: np.ndarray) -> np.ndarray:
        return shell_codes(v, self.lo, self.step)

    def _grow(self, name: str, n: int) -> None:
        for attr in {"p": ("pvec", "pid", "pref", "psq"), "s": ("scode", "sid", "sref", "s_pooled")}[name]:
            a = getattr(self, attr)
            if n >= len(a):
                setattr(self, attr, np.concatenate([a, np.zeros_like(a)]))

    def _pool_slot(self) -> int:
        if self.pfree:
            return self.pfree.pop()
        self._grow("p", self.n_pslots)
        self.n_pslots += 1
        return self.n_pslots - 1

    def _shell_slot(self) -> int:
        if self.sfree:
            return self.sfree.pop()
        self._grow("s", self.n_sslots)
        self.n_sslots += 1
        return self.n_sslots - 1

    def admit(self, center: np.ndarray, K: int, ids, dists, vectors) -> int:
        """Entry from the first 2K ranks, shell from the next 2K; returns the vectors newly held."""
        n = 2 * K
        ids = np.asarray(ids, np.int64)
        new = 0
        resp = []
        for i, v in zip(ids[:n].tolist(), vectors[:n]):
            slot = self.pool.get(i)
            if slot is None:
                slot = self._pool_slot()
                v32 = np.asarray(v, np.float32)
                self.pvec[slot] = v32
                self.pid[slot] = i
                self.pref[slot] = 1
                self.psq[slot] = float(v32.astype(np.float64) @ v32)
                self.pool[i] = slot
                if i in self.shell:
                    self.s_pooled[self.shell[i]] = True
                new += 1
            else:
                self.pref[slot] += 1
            resp.append(slot)
        shell, fresh = [], []
        for i in ids[n:2 * n].tolist():
            if i in self.pool:
                continue
            fresh.append(i)
        rows = {i: v for i, v in zip(ids[n:2 * n].tolist(), vectors[n:2 * n])}
        new_ids = [i for i in dict.fromkeys(fresh) if i not in self.shell]
        codes = dict(zip(new_ids, self._quantize(np.asarray([rows[i] for i in new_ids], np.float32)))) if new_ids else {}
        for i in fresh:
            slot = self.shell.get(i)
            if slot is None:
                slot = self._shell_slot()
                self.scode[slot] = codes[i]
                self.sid[slot] = i
                self.sref[slot] = 1
                self.s_pooled[slot] = False
                self.shell[i] = slot
                new += 1
            else:
                self.sref[slot] += 1
            shell.append(slot)
        if self.free_rows:
            row = self.free_rows.pop()
        else:
            if self.n_rows == len(self.centers):
                self.centers = np.vstack([self.centers, np.zeros_like(self.centers)])
                self.row_entry = np.concatenate([self.row_entry, np.full(len(self.row_entry), -1, np.int64)])
            row = self.n_rows
            self.n_rows += 1
        self.centers[row] = center
        eid = self.next_id
        self.next_id += 1
        self.row_entry[row] = eid
        resp_slots = np.asarray(resp, np.int64)
        self.entries[eid] = {"resp": resp_slots, "shell": np.asarray(shell, np.int64),
                             "radius": float(dists[n - 1]), "row": row, "K": K}
        for slot in resp_slots.tolist():
            self.holders.setdefault(slot, set()).add(eid)
        self.order[eid] = None
        self.lists += len(resp) + len(shell)
        self._all_resp = None
        while self.total_bytes() > self.budget and len(self.entries) > 1:
            self._evict(next(iter(self.order)))
        return new

    def _evict(self, eid: int) -> None:
        e = self.entries.pop(eid)
        self.order.pop(eid)
        self.row_entry[e["row"]] = -1
        self.free_rows.append(e["row"])
        self.lists -= len(e["resp"]) + len(e["shell"])
        self._all_resp = None
        for slot in e["resp"].tolist():
            self.holders[slot].discard(eid)
            self.pref[slot] -= 1
            if self.pref[slot] == 0:
                del self.holders[slot]
                i = int(self.pid[slot])
                del self.pool[i]
                self.pfree.append(slot)
                if i in self.shell:
                    self.s_pooled[self.shell[i]] = False
        for slot in e["shell"].tolist():
            self.sref[slot] -= 1
            if self.sref[slot] == 0:
                del self.shell[int(self.sid[slot])]
                self.sfree.append(slot)
