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
FAST_SCORE = 1e9      # the core's score of an exact repeat (fast path, doc 9ak)


class CoreCache(Cache):
    def __init__(self, D, k, R, bits, budget, maintain=True, threads=8, ef1=0, ef2=0, ball_mode="fit", ball_m=64,
                 ball_ef=64, ball_scan_below=1000, repeat_fast=False, shell_code="sq", insert_ef=40, shell_range="minmax", serve_balls=0):
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
        if insert_ef != 40:
            self.core.set_insert_ef(int(insert_ef))
        self.serve_balls = int(serve_balls)       # doc 9an: serve hits from the union of the n best-fit balls
        if self.serve_balls:
            self.core.set_serve_balls(self.serve_balls, int(ef1))
        self.ef1, self.ef2 = ef1, ef2        # 0 / 0: exact search over the cached vectors (headline)
        # how a request's balls are found (doc 9ai): fit = every entry, best fit; fit-m = best fit among the
        # m nearest centers; near = the nearest centers (graph over the centers at search list ball_ef)
        # near-scan = the nearest centers by a full scan (same features as near, exact); near scans caches of at
        # most ball_scan_below entries too (cheaper than the graph there: speed only)
        mode = {"fit": 0, "fit-m": 1, "near": 2, "near-scan": 3}[ball_mode]
        if mode:
            self.core.set_ball_mode(mode, int(ball_m), int(ball_ef), int(ball_scan_below))
        # exact-repeat fast path (doc 9ak): a request equal to a cached center is served that entry's own top-k
        self.repeat_fast = bool(repeat_fast)
        if self.repeat_fast:
            self.core.set_repeat_fast(True)
        # shell sample codes: sq = the simulator's b-bit scalar quantization (4 bits: packed codes in the core);
        # rabitq = 1-bit RaBitQ codes of the raw vectors (doc 9ak; memory still charged at b bits, conservative)
        if shell_code not in ("sq", "rabitq"):
            raise ValueError(shell_code)
        self.shell_code = shell_code
        # 4-bit code range per dimension: the sample's min / max, or its 1st / 99th percentile (robust, doc 9al:
        # outliers stretch min / max and coarsen the codes; the robust range matches float counts in the ladder)
        if shell_range not in ("minmax", "robust"):
            raise ValueError(shell_range)
        self.shell_range = shell_range
        self.last_fast = None
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
        sv = (np.asarray(self._db[samp], np.float32) if self.shell_code == "rabitq"
              else self._quantize(self._db[samp]))
        self.core.add_entry(int(c), np.asarray(self._q[c], np.float32), float(self.geo[c][2]), ball,
                            np.asarray(self._db[ball], np.float32), samp, sv,
                            float(self.last_used.get(c, 0)))
        for v in ball.tolist():
            self._holders.setdefault(v, set()).add(int(c))
        fresh = np.unique(ball[self._count[ball] == 0])
        self._fetched += len(fresh)
        self.samp_fetched = getattr(self, "samp_fetched", 0) + len(samp)     # shell-sample vectors (re)coded (doc 9as)
        # 11-21: vectors a real cache must read for its shell sample: only ones no current entry samples already (a
        # resized or re-added entry re-codes its sample, but those vectors are in memory)
        if not hasattr(self, "_samp_count"):
            self._samp_count = np.zeros(len(self._db), dtype=np.int32)
        if len(samp):
            su = np.unique(samp)
            self.samp_new = getattr(self, "samp_new", 0) + int((self._samp_count[su] == 0).sum())
            self._samp_count[su] += 1
        np.add.at(self._count, ball, 1)

    def _core_remove(self, c):
        if hasattr(self, "_samp_count"):
            _, samp = self._entry_parts(c)
            if len(samp):
                su = np.unique(samp)
                self._samp_count[su] = np.maximum(self._samp_count[su] - 1, 0)
        ball, _ = self._ball_shell(int(c))
        ball = ball[ball >= 0].astype(np.int64)
        np.subtract.at(self._count, ball, 1)
        for v in ball.tolist():
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
        ents = list(self.entries)
        new_range = False
        if self.qlo is None and self.bits < 32:
            # the range over the current sample (shell minus cached vectors), before the entries leave the core
            sid = self.shell_ids()
            samp = sid[self.D["draw"][sid] < self.p]
            if len(samp):
                x = np.asarray(self._db[samp], np.float64)
                if self.shell_range == "robust":
                    self.qlo, self.qhi = np.percentile(x, 1, axis=0), np.percentile(x, 99, axis=0)
                else:
                    self.qlo, self.qhi = x.min(axis=0), x.max(axis=0)
                self._rq_c = x.mean(axis=0)          # RaBitQ centroid: the first sample's mean
                new_range = True
        for c in ents:
            self._core_remove(c)
        if new_range and self.shell_code == "rabitq":
            # random orthogonal rotation, as ballfit_experiments.RaBitQ (seed 0); the core takes P^T
            P, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(self.dim, self.dim)))
            self.core.set_rabitq(list(map(float, self._rq_c)), np.ascontiguousarray(P.T, dtype=np.float32))
        elif new_range:   # packed 4-bit codes in the core (identical counts; doc 9ai); other widths stay float32
            self.core.set_quantizer(list(map(float, self.qlo)), list(map(float, self.qhi)), int(self.bits))
        for c in ents:
            self._core_add(c)
        self._core_p = self.p

    # -- Cache interface ------------------------------------------------------------------
    # bookkeeping kept incrementally (Cache recomputes it from every entry after any change, which per-request
    # updates cannot afford): vector refcounts (_count) in _core_add / _core_remove, sizes from the core
    def _update_members(self):
        self._members_dirty = False

    def n_cached(self):
        return int(self.core.n_cached())

    def memory(self):
        return float(self.core.memory(self.bits))

    def shell_ids(self):
        """The shell: union of the entries' shells minus cached vectors (on demand: set_p_to_fill)."""
        if not self.entries:
            return np.empty(0, dtype=np.int64)
        sh = np.unique(np.concatenate([self._ball_shell(int(c))[1] for c in self.entries]))
        sh = sh[sh >= 0]
        return sh[self._count[sh] == 0]

    def add(self, c, K, shallow=False):
        if c in self.entries:
            self._core_remove(c)
        super().add(c, K, shallow=shallow)
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
        lam = float("-inf") if (self.ef2 or self.serve_balls) else float("inf")
        res = self.core.lookup_batch(xs, lam, int(self.ef1), int(self.ef2))
        hit, score, near, nr, feat = res[:5]
        self.last_fast = np.asarray(res[5], dtype=bool) if len(res) > 5 else np.zeros(len(qidx), bool)
        X = np.asarray(feat, dtype=np.float64)
        near = np.asarray(near, dtype=np.int64)
        if self.core.n_cached() < self.k:
            near[:] = 0
        return X, near

    def scores(self, qidx, model):
        """Model scores; exact repeats (fast path) score FAST_SCORE: served at every cutoff, so calibration
        counts them as hits at every cutoff (their labels: audits, as any hit)."""
        X, near = self.features(qidx)
        s = np.asarray(model(X), dtype=np.float64)
        if self.repeat_fast and self.last_fast is not None and self.last_fast.any():
            s = np.where(self.last_fast, FAST_SCORE, s)
        return s, X, near

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
