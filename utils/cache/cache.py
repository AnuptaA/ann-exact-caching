#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Online cache state: entries (center query, K) with ref-counted membership, shell sample, features."""

import numpy as np

from utils.base.knn import sq_dist
from utils.cache.features import N_BALLS, POOR_FIT, fit_balls
from utils.cache.shell import quantize

EPS = 1e-12
CHUNK = 1024

#-------------------------------------------------------------------------------


class Cache:
    """Cache entries plus shell sample over a fixed database."""

    def __init__(self, D, k, R, bits, budget):
        self.D, self.k, self.R, self.bits, self.budget = D, k, R, bits, budget
        self.n_db = len(D["db"])
        self.ref = np.zeros(self.n_db, dtype=np.int32)
        self.entries = {}          # center query -> K
        self.p = 0.0
        self.last_used = {}
        self.vectors_fetched = 0
        self._dirty = True

    # -- membership --------------------------------------------------------

    def members(self, c, K):
        return self.D["top"][c, :K]

    def add(self, c, K):
        """Add (or resize) an entry; vectors already cached are not refetched."""
        ids = self.members(c, K)
        self.vectors_fetched += int((self.ref[ids] == 0).sum())
        self.ref[ids] += 1
        if c in self.entries:
            self.ref[self.members(c, self.entries[c])] -= 1
        self.entries[c] = K
        self._dirty = True

    def remove(self, c):
        K = self.entries.pop(c)
        self.ref[self.members(c, K)] -= 1
        self.last_used.pop(c, None)
        self._dirty = True

    def replace(self, entries):
        """Swap to a new entry set: add new / resized entries first, then drop
        leaving ones, so shared vectors are never refetched."""
        for c, K in entries.items():
            if self.entries.get(c) != K:
                self.add(c, K)
        for c in [c for c in self.entries if c not in entries]:
            self.remove(c)

    @property
    def cached(self):
        return self.ref > 0

    def n_cached(self):
        return int((self.ref > 0).sum())

    # -- shell sample ------------------------------------------------------

    def shell_ids(self):
        top = self.D["top"]
        parts = [top[c, K:self.R] for c, K in self.entries.items() if K < self.R]
        if not parts:
            return np.empty(0, dtype=np.int64)
        ids = np.unique(np.concatenate(parts))
        return ids[self.ref[ids] == 0]

    def set_p_to_fill(self):
        """Largest p (<= 1) such that cache + sample fits the budget."""
        full = len(self.shell_ids()) * self.bits / 32
        room = self.budget - self.n_cached()
        self.p = float(np.clip(room / full, 0.0, 1.0)) if full > 0 else 0.0
        self._dirty = True

    def memory(self):
        """Cached vectors + sample vector-equivalents (no array rebuild)."""
        sid = self.shell_ids()
        n_samp = int((self.D["draw"][sid] < self.p).sum())
        return self.n_cached() + n_samp * self.bits / 32

    # -- arrays for features -----------------------------------------------

    def prepare(self):
        if not self._dirty:
            return
        db, Q = self.D["db"], self.D["q"]
        self.cached_ids = np.flatnonzero(self.ref > 0)
        self.cv = db[self.cached_ids]
        self.cv_sq = (self.cv * self.cv).sum(axis=1)
        self.centers = np.array(list(self.entries), dtype=np.int64)
        cq = Q[self.centers]
        self.cq, self.cq_sq = cq, (cq * cq).sum(axis=1)
        kth = self.D["top"][self.centers, np.array([self.entries[c] for c in self.centers]) - 1]
        self.radius = np.sqrt(((cq - db[kth]) ** 2).sum(axis=1))
        sid = self.shell_ids()
        self.samp_ids = sid[self.D["draw"][sid] < self.p]
        sv = quantize(db[self.samp_ids].astype(np.float64), self.bits).astype(np.float32) \
            if len(self.samp_ids) else np.zeros((0, db.shape[1]), np.float32)
        self.sv, self.sv_sq = sv, (sv * sv).sum(axis=1)
        self._dirty = False

    def features(self, qidx):
        """7 features, neighborhood radius and the k nearest cached ids."""
        self.prepare()
        Q, k = self.D["q"], self.k
        n = len(qidx)
        X = np.full((n, 2 * N_BALLS + 1), POOR_FIT, dtype=np.float64)
        near = np.empty((n, k), dtype=np.int64)
        for s in range(0, n, CHUNK):
            q = Q[qidx[s:s + CHUNK]]
            m = len(q)
            d2 = sq_dist(q, self.cv, self.cv_sq)
            part = np.argpartition(d2, k - 1, axis=1)[:, :k]
            near[s:s + m] = self.cached_ids[part]
            nr = np.sqrt(np.take_along_axis(d2, part, axis=1).max(axis=1)) + EPS
            de = np.sqrt(sq_dist(q, self.cq, self.cq_sq))
            X[s:s + m, :2 * N_BALLS] = fit_balls(nr, de, self.radius, N_BALLS, POOR_FIT)
            if len(self.sv):
                cnt = (sq_dist(q, self.sv, self.sv_sq) < (nr ** 2)[:, None]).sum(axis=1)
            else:
                cnt = np.zeros(m)
            X[s:s + m, -1] = np.log1p(cnt)
        return X, near

    def retention(self, qidx):
        return self.cached[self.D["top"][qidx, :self.k]].mean(axis=1)

    def mark_used(self, near_ids, t):
        """LRU: entries supplying any of the given vector ids were used at t."""
        if not len(near_ids) or not self.entries:
            return
        hit = np.zeros(self.n_db, dtype=bool)
        hit[near_ids.ravel()] = True
        for c, K in self.entries.items():
            if hit[self.members(c, K)].any():
                self.last_used[c] = t
