#!/usr/bin/env python3
"""The simulator's cache (online_sim.Cache) backed by the C++ core (doc 9ah, milestone 2).

online_sim.Cache keeps the cheap bookkeeping (entries, ball / shell geometry from the backend's ranked
lists, vector refcounts, the shell sample's sampling rate p, labels) and rebuilds every array the hit
decision needs (cached vectors, quantized shell sample, per-entry sample lists) in prepare() after any
change: fine once per 1,000-request batch, ~1 s per change on large caches, i.e. impractical per request.
CoreCache keeps that bookkeeping (it subclasses Cache) and mirrors every add / remove into the core
incrementally; features() asks the core. The controllers (greedy, rebuild, LRU, continuous greedy,
calibration, drift tests) run unchanged on it.

Differences to Cache, by design:
  - the shell sample's quantization range (per dimension) is fixed when the sample is first filled (the
    warm-up) instead of recomputed at every prepare(); identical while the cache is static
  - database changes (inserts / deletes) are not supported (the data-change experiments use Cache)
Run with .venv-core (Python 3.11): the core is built for it (cachecore/build.sh).
"""
import numpy as np

from simulations.approximate_hits.online_sim import Cache

EPS = 1e-12


class CoreCache(Cache):
    def __init__(self, D, k, R, bits, budget, maintain=True, threads=8, ef1=0, ef2=0):
        super().__init__(D, k, R, bits, budget, maintain=maintain)
        if not np.all(self.D["present"]):
            raise ValueError("CoreCache: database changes are not supported")
        if not maintain:
            raise ValueError("CoreCache: frozen (unmaintained) balls are not supported")
        from simulations.approximate_hits.cachecore import Core
        self._db = self.D["db"]
        self._q = self.D["q"]
        self.dim = self._db.shape[1]
        # exact search (headline, ef1 = 0) needs no graph: none is maintained
        self.core = Core(self.dim, k, 0, capacity=int(max(budget, 1024)) + 1024, graph=bool(ef1 > 0))
        self.core.set_threads(threads)
        self.ef1, self.ef2 = ef1, ef2        # 0 / 0: exact search over the cached vectors (headline)
        self.qlo = self.qhi = None           # fixed quantization range of the shell sample
        self._holders = {}                   # vector id -> set of entry centers (LRU use marks)
        self._core_p = None                  # sampling rate the core's samples were built with
        self._scope_set = None

    # -- the core mirror ------------------------------------------------------------------
    def _scope(self):
        sb = self.shell_balls if self.shell_scope == "balls" else 0
        if sb != self._scope_set:
            self.core.set_shell_balls(int(sb))
            self._scope_set = sb

    def _quantize(self, x):
        if self.bits >= 32 or not len(x):
            return np.asarray(x, np.float32)
        step = np.maximum(self.qhi - self.qlo, EPS) / (2 ** self.bits - 1)
        x = np.clip(np.asarray(x, np.float64), self.qlo, self.qhi)
        return (self.qlo + np.round((x - self.qlo) / step) * step).astype(np.float32)

    def _entry_parts(self, c):
        ball, shell = self._ball_shell(int(c))
        ball = ball[ball >= 0]
        shell = shell[shell >= 0]
        shell = shell[self.D["draw"][shell] < self.p] if self.qlo is not None or self.bits >= 32 else shell[:0]
        return ball.astype(np.int64), shell.astype(np.int64)

    def _core_add(self, c):
        ball, samp = self._entry_parts(c)
        self.core.add_entry(int(c), np.asarray(self._q[c], np.float32), float(self.geo[c][2]), ball,
                            np.asarray(self._db[ball], np.float32), samp, self._quantize(self._db[samp]),
                            float(self.last_used.get(c, 0)))
        for v in ball.tolist():
            self._holders.setdefault(v, set()).add(int(c))

    def _core_remove(self, c):
        ball, _ = self._ball_shell(int(c))
        for v in ball[ball >= 0].tolist():
            h = self._holders.get(v)
            if h is not None:
                h.discard(int(c))
                if not h:
                    del self._holders[v]
        self.core.remove_entry(int(c))
        # evicted vectors stay as deleted graph nodes / dead slots: compact when they pile up (background)
        if self.core.dead_slots() > 0.3 * max(self.core.n_cached(), 1) and self.core.dead_slots() > 1000:
            self.core.rebuild_index(32)      # frees evicted vectors' slots; rebuilds the graph if there is one

    def _resync_samples(self):
        """p changed: every entry's sampled shell changes (warm-up, greedy rebuilds)."""
        if self.qlo is None and self.bits < 32:
            sid = super().shell_ids()
            samp = sid[self.D["draw"][sid] < self.p]
            if len(samp):
                x = np.asarray(self._db[samp], np.float64)
                self.qlo, self.qhi = x.min(axis=0), x.max(axis=0)
        for c in list(self.entries):
            self._core_remove(c)
            self._core_add(c)
        self._core_p = self.p

    # -- Cache interface ------------------------------------------------------------------
    def add(self, c, K):
        if c in self.entries:
            self._core_remove(c)
        super().add(c, K)
        self._core_add(c)

    def remove(self, c):
        self._core_remove(c)
        super().remove(c)

    def set_p_to_fill(self):
        super().set_p_to_fill()
        if self.p != self._core_p:
            self._resync_samples()

    def prepare(self):
        """No array rebuild: the core is current. cached_ids for callers that read it (continuous greedy)."""
        self._update_members()
        self.cached_ids = np.flatnonzero(self._count > 0)

    def full_prepare(self):
        """The arrays of Cache.prepare (for cache_state / save_state only)."""
        self._dirty = True
        Cache.prepare(self)

    def features(self, qidx):
        self._scope()
        if self._core_p != self.p and len(self.entries):
            self._resync_samples()
        qidx = np.asarray(qidx, dtype=np.int64)
        xs = np.ascontiguousarray(np.asarray(self._q[qidx], np.float32))
        # features for the controllers (no decision here); with a stage 2 (timing), near = the answer that
        # would be served (lambda -inf: every query takes the stage-2 path), as the labels need
        lam = float("-inf") if self.ef2 else float("inf")
        hit, score, near, nr, feat = self.core.lookup_batch(xs, lam, int(self.ef1), int(self.ef2))
        X = np.asarray(feat, dtype=np.float64)
        near = np.asarray(near, dtype=np.int64)
        if self.core.n_cached() < self.k:
            near[:] = 0
        return X, near

    def sync(self, model):
        """Before core decisions: shell scope, samples (p), model coefficients."""
        self._scope()
        if self._core_p != self.p and len(self.entries):
            self._resync_samples()
        if getattr(self, "_model_id", None) != id(model):
            mp = model.params
            self.core.set_model(list(map(float, mp["mu"])), list(map(float, mp["sd"])),
                                list(map(float, mp["coef"])), float(mp["intercept"]))
            self._model_id = id(model)

    def mark_used(self, near_ids, t):
        """LRU: entries supplying any of the given vector ids were used at t."""
        if not len(near_ids) or not self.entries:
            return
        for v in np.unique(np.asarray(near_ids).ravel()).tolist():
            for c in self._holders.get(v, ()):
                self.last_used[c] = t
