"""Reference functions copied unchanged from the local branch system_param_microbenchmarks
(utils/cache/features.py, utils/cache/shell.py, utils/cache/minicache.py, utils/cache/lambda_policy.py).
They are the test oracle for the ReCal core and policy and are not imported by recal/."""

from collections import OrderedDict
from statistics import NormalDist

import numpy as np

EPS = 1e-12
Z_CAL = NormalDist().inv_cdf(0.95)
FLOOR = 100.0


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


class BinnedDeque:
    def __init__(self, thr_desc, size, target, floor=FLOOR, z=Z_CAL, skew=False):
        self.skew = skew
        self.last = None
        self.s3 = np.zeros(len(thr_desc) + 1)
        self.thr = np.asarray(thr_desc, dtype=np.float64)
        self.nb = len(self.thr)
        self.size, self.target, self.floor, self.z = size, target, floor, z
        self.b = np.zeros(size, dtype=np.int64)
        self.y = np.zeros(size)
        self.w = np.zeros(size)
        self.head, self.n = 0, 0
        self.sums = np.zeros((4, self.nb + 1))   # W, sum w y, sum w y^2, sum w^2

    def bin_of(self, s):
        """First threshold (strict to lenient) the score clears; nb = below all."""
        return np.searchsorted(-self.thr, -np.asarray(s), side="left")

    def _add(self, b, y, w, sign):
        self.sums[0, b] += sign * w
        self.sums[1, b] += sign * w * y
        self.sums[2, b] += sign * w * y * y
        self.sums[3, b] += sign * w * w
        self.s3[b] += sign * w * y * y * y

    def push(self, b, y, w):
        i = (self.head + self.n) % self.size
        if self.n == self.size:
            self._add(self.b[self.head], self.y[self.head], self.w[self.head], -1.0)
            self.head = (self.head + 1) % self.size
            i = (self.head + self.n - 1) % self.size
        else:
            self.n += 1
        self.b[i], self.y[i], self.w[i] = b, y, w
        self._add(b, y, w, 1.0)

    def lam(self):
        """(threshold or None, effective sample size at it)."""
        c = np.cumsum(self.sums[:, :self.nb], axis=1)
        W, S1, S2, SW2 = c
        best, best_ess = None, 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            ess = W * W / SW2
            mean = S1 / W
            denom = W - SW2 / W
            sd = np.sqrt(np.maximum(S2 - W * mean ** 2, 0.0) / denom)
            mu3 = (np.cumsum(self.s3[:self.nb]) - 3 * mean * S2 + 3 * mean ** 2 * S1
                   - mean ** 3 * W) / W
            gamma = mu3 / np.maximum(S2 / W - mean ** 2, 1e-12) ** 1.5
            mult = self.z
            if self.skew:
                mult = self.z + np.maximum(0.0, -gamma) * (2 * self.z ** 2 + 1) / (
                    6 * np.sqrt(ess))
            lcb = mean - mult * sd / np.sqrt(ess)
        best_i = None
        for i in range(self.nb):
            if not np.isfinite(ess[i]) or ess[i] < self.floor:
                continue
            if lcb[i] >= self.target:
                best, best_ess, best_i = float(self.thr[i]), float(ess[i]), i
            else:
                break
        self.last = None if best_i is None else {
            "ess": best_ess, "sd": float(sd[best_i]), "gamma": float(gamma[best_i]),
            "mean": float(mean[best_i]), "lcb": float(lcb[best_i])}
        return best, best_ess
