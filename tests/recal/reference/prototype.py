"""Reference functions copied unchanged from the local branch system_param_microbenchmarks
(utils/cache/features.py, utils/cache/shell.py, utils/cache/minicache.py, utils/cache/lambda_policy.py).
They are the test oracle for the ReCal core and policy and are not imported by recal/."""

from collections import OrderedDict

import numpy as np

EPS = 1e-12


def fit_balls(nr, de, radius, n, fill):
    """Relative offset (de / r) and size (nr / r) of the n best-fitting balls.

    nr: neighborhood radius per query; de: distance to every ball center
    (queries x balls); radius: ball radii. Balls are ranked by the worst-case
    fit (nr + de) / r; columns of missing balls (fewer than n) hold `fill`.
    Shared by the offline evaluation and the online simulator.
    """
    radius = np.maximum(radius, EPS)  # duplicate-vector balls can have radius 0
    F = np.full((len(nr), 2 * n), fill, dtype=np.float64)
    m = min(n, de.shape[1])
    fit = (nr[:, None] + de) / radius[None, :]
    cand = (np.argpartition(fit, m - 1, axis=1)[:, :m] if m < fit.shape[1]
            else np.tile(np.arange(m), (len(nr), 1)))
    order = np.take_along_axis(cand, np.argsort(np.take_along_axis(fit, cand, axis=1),
                                                axis=1), axis=1)
    rows = np.arange(len(nr))
    for t in range(m):
        j = order[:, t]
        F[:, 2 * t] = de[rows, j] / radius[j]
        F[:, 2 * t + 1] = nr / radius[j]
    return F


def quantize(x, bits):
    """b-bit uniform scalar quantization per dimension (reconstructed)."""
    if bits < 1:
        raise ValueError(f"bits must be at least 1, got {bits}")
    if bits >= 32:
        return x
    lo, hi = x.min(axis=0), x.max(axis=0)
    step = np.maximum(hi - lo, EPS) / (2 ** bits - 1)
    return lo + np.round((x - lo) / step) * step


ROW_BASE_BYTES = 40


def row_bytes(k: int) -> int:
    return ROW_BASE_BYTES + 4 * k


class MiniCache:
    """LRU or CLOCK hash map keyed by a 64-bit hash of the query vector bytes.

    A row is [kind, k, score, epoch, ids]. lookup returns "backend", "served",
    "refused" (served row whose score is below lambda or lambda undefined) or "miss".
    """

    def __init__(self, rows: int, k: int = 10, policy: str = "lru", epoch: int = 0,
                 store_served: bool = True):
        if policy not in ("lru", "clock"):
            raise ValueError(policy)
        self.rows, self.k, self.policy, self.epoch = rows, k, policy, epoch
        self.store_served = store_served
        self.map = OrderedDict() if policy == "lru" else {}
        if policy == "clock":
            self.slot_key = [None] * rows
            self.bit = np.zeros(rows, dtype=bool)
            self.slot_of = {}
            self.hand = 0
        self.evictions = self.conversions = self.refused = 0
        self.hits = {"backend": 0, "served": 0}

    def __len__(self):
        return len(self.map)

    def count_kinds(self) -> dict:
        kinds = [r[0] for r in self.map.values()]
        return {"backend": kinds.count("backend"), "served": kinds.count("served")}

    def _touch(self, key):
        if self.policy == "lru":
            self.map.move_to_end(key)
        else:
            self.bit[self.slot_of[key]] = True

    def _drop(self, key):
        del self.map[key]
        if self.policy == "clock":
            s = self.slot_of.pop(key)
            self.slot_key[s] = None
            self.bit[s] = False

    def _insert(self, key, row):
        if key in self.map:
            self.map[key] = row
            self._touch(key)
            return
        if self.policy == "lru":
            if len(self.map) >= self.rows:
                self.map.popitem(last=False)
                self.evictions += 1
            self.map[key] = row
            return
        if len(self.map) >= self.rows:
            while True:
                s = self.hand
                self.hand = (self.hand + 1) % self.rows
                if self.slot_key[s] is None:
                    break
                if self.bit[s]:
                    self.bit[s] = False
                    continue
                del self.map[self.slot_key[s]]
                del self.slot_of[self.slot_key[s]]
                self.evictions += 1
                break
        else:
            s = self.slot_key.index(None)
        self.slot_key[s] = key
        self.slot_of[key] = s
        self.bit[s] = True
        self.map[key] = row

    def lookup(self, key, k: int, lam):
        row = self.map.get(key)
        if row is None:
            return "miss"
        if row[3] != self.epoch or k > row[1]:
            self._drop(key)
            return "miss"
        if row[0] == "backend":
            self._touch(key)
            self.hits["backend"] += 1
            return "backend"
        if lam is None or row[2] < lam:
            self.refused += 1
            return "refused"
        self._touch(key)
        self.hits["served"] += 1
        return "served"

    def write_backend(self, key, k: int, ids=None):
        old = self.map.get(key)
        if old is not None and old[0] == "served" and old[3] == self.epoch:
            self.conversions += 1
        self._insert(key, ["backend", k, None, self.epoch, ids])

    def write_served(self, key, k: int, score: float, ids=None):
        if not self.store_served:
            return
        old = self.map.get(key)
        if old is not None and old[0] == "backend" and old[3] == self.epoch and old[1] >= k:
            self._touch(key)
            return
        self._insert(key, ["served", k, float(score), self.epoch, ids])

    def bump_epoch(self):
        self.epoch += 1
