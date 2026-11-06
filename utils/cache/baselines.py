#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Prior approximate vector caches for the online simulator (online_sim.py).

Both start empty and process the warm-up queries like any others (misses
fill the cache); metrics are logged after warm-up, costs from the start.
As for our methods, cache contents change only between batches; per-query
state (thresholds) is updated in arrival order within a batch.

Aker (density-aware approximate cache):
  representative entry per inserted miss: the query, its backend top-k ids
  (neighbor reference list), minD = distance to its nearest neighbor, and a
  threshold tau initialized to minD / 4 and capped at minD.
  lookup: exact repeat of a cached query (equality hit; tau *= 1.1), repeat
  of an earlier approximate-hit query (alias hit), or the nearest
  representative with d(query, representative) < tau (approximate hit; tau
  *= 0.25). The answer is that entry's top-k (not a union over the cache).
  miss: backend top-k inserted as a new representative. Capacity: distinct
  vectors in the neighbor object pool (budget M); FIFO eviction of
  representative entries (with their aliases).

QVCache (query-aware vector cache):
  M vectors split into --qv-mini mini-indexes; a miss inserts the backend's
  top-k vectors (those not cached yet) into the current mini-index; when it
  is full, an empty mini-index is used, else the least recently used one is
  evicted whole and reused. Hit iff the distance to the k-th nearest cached
  vector is <= (1 + D) theta[region]; regions = cells of --qv-buckets equal
  buckets per dimension of a --qv-dims-dimensional PCA projection (fit on
  the warm-up queries); on a miss theta[region] <- (1 - alpha) theta + alpha
  d_backend[k] (initialized to d_backend[k] in a new region). Answer = the k
  nearest cached vectors. Cache search is exact here (the paper uses graph
  indexes), which can only help QVCache.
"""

import numpy as np

from utils.base.knn import sq_dist

#-------------------------------------------------------------------------------


class Base:
    def __init__(self, D, stream, args):
        self.D, self.stream, self.a = D, stream, args
        self.k = args.k
        self.M = args.budget * len(D["db"])
        self.ref = np.zeros(len(D["db"]), dtype=np.int32)
        self.log, self.events = [], []
        self.cost = {"topk": 0, "topR": 0}
        self.fetched = 0
        self.n_c = 0            # distinct cached vectors

    def _acquire(self, ids):
        new = int((self.ref[ids] == 0).sum())
        self.fetched += new
        self.n_c += new
        self.ref[ids] += 1

    def _release(self, ids):
        self.ref[ids] -= 1
        self.n_c -= int((self.ref[ids] == 0).sum())

    def n_cached(self):
        return self.n_c

    def _dist(self, qi, ids):
        x = self.D["db"][ids]
        return np.sqrt(((x - self.D["q"][qi]) ** 2).sum(axis=-1))

    def run(self):
        a = self.a
        for s in range(0, len(self.stream), a.batch):
            qb = self.stream[s:s + a.batch]
            served, ret = self.batch(qb, s)
            self.cost["topk"] += int((~served).sum())
            if s + len(qb) > a.warmup:
                keep = np.arange(len(qb)) + s >= a.warmup
                self.log.append({"t": s + len(qb), "n": int(keep.sum()),
                                 "served": int(served[keep].sum()),
                                 "served_retention_sum": float(ret[served & keep].sum()),
                                 "audited_served": 0, "lambda": None,
                                 "memory": self.n_cached(), "cached": self.n_cached(),
                                 "entries": self.n_entries()})
        return {"log": self.log, "events": self.events,
                "cost": dict(self.cost, vectors_fetched=self.fetched)}


class Aker(Base):
    def __init__(self, D, stream, args):
        super().__init__(D, stream, args)
        self.reps = {}          # center query -> [tau, minD]
        self.fifo = []          # insertion order of representatives
        self.alias = {}         # query -> representative
        self.children = {}      # representative -> its aliases

    def n_entries(self):
        return len(self.reps)

    def _insert(self, c):
        ids = self.D["top"][c, :self.k]
        min_d = float(self._dist(c, ids[:1])[0])
        self._acquire(ids)
        self.reps[c] = [min_d / 4, min_d]
        self.fifo.append(c)
        while self.n_cached() > self.M and self.fifo:
            v = self.fifo.pop(0)
            if v not in self.reps:
                continue
            self._release(self.D["top"][v, :self.k])
            del self.reps[v]
            for al in self.children.pop(v, []):
                self.alias.pop(al, None)

    def batch(self, qb, s):
        k, top, Q = self.k, self.D["top"], self.D["q"]
        n = len(qb)
        served = np.zeros(n, dtype=bool)
        ret = np.zeros(n)
        centers = np.array(list(self.reps), dtype=np.int64)
        if len(centers):
            cq = Q[centers]
            d = np.sqrt(sq_dist(Q[qb], cq, (cq * cq).sum(axis=1)))
            near, near_d = centers[d.argmin(axis=1)], d.min(axis=1)
        misses = []
        for i, q in enumerate(qb):
            q = int(q)
            e = None
            if q in self.reps:                       # equality hit
                e = q
                self.reps[e][0] = min(self.reps[e][0] * self.a.aker_alpha_l, self.reps[e][1])
            elif q in self.alias and self.alias[q] in self.reps:
                e = self.alias[q]                    # alias hit (approximate)
                self.reps[e][0] *= self.a.aker_alpha_t
            elif len(centers) and int(near[i]) in self.reps and near_d[i] < self.reps[int(near[i])][0]:
                e = int(near[i])                     # approximate hit
                self.reps[e][0] *= self.a.aker_alpha_t
                self.alias[q] = e
                self.children.setdefault(e, []).append(q)
            if e is None:
                misses.append(q)
            else:
                served[i] = True
                ret[i] = np.isin(top[q, :k], top[e, :k]).mean()
        for q in misses:
            if q not in self.reps:
                self._insert(q)
        return served, ret


class QVCache(Base):
    def __init__(self, D, stream, args):
        super().__init__(D, stream, args)
        self.n_mini = args.qv_mini
        self.cap = self.M / self.n_mini
        self.mini = [[] for _ in range(self.n_mini)]   # vector ids per mini-index
        self.size = [0] * self.n_mini
        self.last_used = [0] * self.n_mini
        self.fill = 0
        self.theta = {}
        self.owner = np.full(len(D["db"]), -1, dtype=np.int64)  # mini-index of a cached id
        w = np.asarray(D["q"][stream[:args.warmup]], dtype=np.float64)
        self.mu = w.mean(axis=0)
        _, _, vt = np.linalg.svd(w - self.mu, full_matrices=False)
        self.P = vt[:args.qv_dims].T
        z = (w - self.mu) @ self.P
        self.lo, self.hi = z.min(axis=0), z.max(axis=0)

    def n_entries(self):
        return sum(1 for m in self.mini if m)

    def region(self, qi):
        z = (np.asarray(self.D["q"][qi], dtype=np.float64) - self.mu) @ self.P
        b = np.floor((z - self.lo) / np.maximum(self.hi - self.lo, 1e-12) * self.a.qv_buckets)
        return tuple(np.clip(b, 0, self.a.qv_buckets - 1).astype(int))

    def _next_fill(self, t):
        empty = [i for i in range(self.n_mini) if self.size[i] == 0]
        if empty:
            return empty[0]
        v = min(range(self.n_mini), key=lambda i: self.last_used[i])
        ids = np.asarray(self.mini[v], dtype=np.int64)
        self._release(ids)
        self.owner[ids] = -1
        self.mini[v], self.size[v] = [], 0
        self.events.append({"t": t, "event": "evict_mini", "mini": v})
        return v

    def _insert(self, q, t):
        ids = self.D["top"][q, :self.k]
        new = ids[self.ref[ids] == 0]
        if self.size[self.fill] + len(new) > self.cap:
            self.fill = self._next_fill(t)
        self._acquire(new)
        self.owner[new] = self.fill
        self.mini[self.fill].extend(new.tolist())
        self.size[self.fill] += len(new)
        self.last_used[self.fill] = t

    def batch(self, qb, s):
        k, top, Q, a = self.k, self.D["top"], self.D["q"], self.a
        n = len(qb)
        served = np.zeros(n, dtype=bool)
        ret = np.zeros(n)
        ids = np.flatnonzero(self.ref > 0)
        if len(ids) >= k:
            cv = self.D["db"][ids]
            d2 = sq_dist(Q[qb], cv, (cv * cv).sum(axis=1))
            part = np.argpartition(d2, k - 1, axis=1)[:, :k]
            dk = np.sqrt(np.take_along_axis(d2, part, axis=1).max(axis=1))
            near = ids[part]
        cached = self.ref > 0
        misses = []
        for i, q in enumerate(qb):
            q = int(q)
            key = self.region(q)
            th = self.theta.get(key)
            if len(ids) >= k and th is not None and dk[i] <= (1 + a.qv_D) * th:
                served[i] = True
                ret[i] = cached[top[q, :k]].mean()
                for m in set(self.owner[near[i]].tolist()):
                    if m >= 0:
                        self.last_used[m] = s + i
            else:
                db_k = float(self._dist(q, top[q, k - 1:k])[0])
                self.theta[key] = db_k if th is None else (1 - a.qv_alpha) * th + a.qv_alpha * db_k
                misses.append(q)
        for q in misses:
            self._insert(q, s + len(qb))
        return served, ret
