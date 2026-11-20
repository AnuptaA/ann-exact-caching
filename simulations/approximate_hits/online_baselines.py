#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Prior approximate vector caches for the online simulator (online_sim.py).

Both start empty and process the warm-up queries like any others (misses
fill the cache); metrics are logged after warm-up, costs from the start.
As for our methods, cache contents change only between batches; per-query
state (thresholds) is updated in arrival order within a batch. Retention of
a served query = share of its current exact top-k in the returned answer.

Aker (density-aware approximate cache):
  representative entry per inserted miss: the query, its backend top-(k + D)
  ids (neighbor reference list; D = --aker-reserve, paper default 0), minD
  = distance to its nearest neighbor, and a threshold tau initialized to
  --aker-tau-init x minD (paper: 1/4) and capped at --aker-cap x minD
  (paper: 1).
  lookup: exact repeat of a cached query (equality hit; tau *= 1.1), repeat
  of an earlier approximate-hit query (alias hit), or the nearest
  representative with d(query, representative) < tau (approximate hit; tau
  *= 0.25). The answer is that entry's first k neighbors.
  miss: backend top-(k + D) inserted as a new representative. Capacity:
  distinct vectors in the neighbor object pool (budget M, reserves
  included); FIFO eviction of representative entries (with their aliases).
  database changes ("del-consistency", paper section 4.6):
    delete  removed from every entry holding it; the next reserve neighbor
            moves up; an entry left with fewer than k neighbors is dropped
    insert  appended to an insert log; fast path: the entry whose query is
            closest to the new vector takes it if it is closer than that
            entry's farthest stored neighbor (the farthest is then dropped);
            slow path, whenever risk R >= --aker-risk (paper 0.3), on every
            insert and lookup: the next entry in round-robin order checks
            the next --aker-batch (paper 16) log records from its checkpoint;
            R = mean over entries of d_k / d_{k+D} (1 if D = 0) x mean
            unchecked records per entry / log size

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
  database changes: the paper does not handle them (writes go to the
  backend only). --qv-deletes keep: as published (deleted vectors stay
  cached and can be returned; counted as deleted_returned); remove: a
  minimal fix that drops deleted vectors from the mini-indexes. Inserts
  reach the cache only through later misses.
"""

import numpy as np

from simulations.approximate_hits.online_sim import db_view, sq_dist, topk_present

#-------------------------------------------------------------------------------


class Base:
    def __init__(self, D, stream, args):
        self.D, self.stream, self.a = db_view(D), stream, args
        self.k = args.k
        self.M = args.budget * len(D["db"])
        self.ref = np.zeros(len(D["db"]), dtype=np.int32)
        self.log, self.events = [], []
        self.cost = {"topk": 0, "topR": 0, "refresh_dist": 0, "deleted_returned": 0}
        self.fetched = 0
        self.n_c = 0            # distinct cached vectors
        self.ev = D.get("events")
        self.ev_i = 0

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

    def apply_events(self, s):
        """Inserts / deletes scheduled before stream position s, in order."""
        ev = self.ev
        if ev is None:
            return 0
        j = int(np.searchsorted(ev["t"], s, side="right"))
        if j <= self.ev_i:
            return 0
        ids, ins = ev["id"][self.ev_i:j], ev["insert"][self.ev_i:j]
        pres = self.D["present"]
        pres[ids[ins]] = True
        pres[ids[~ins]] = False
        self.on_delete(ids[~ins])
        self.on_insert(ids[ins])
        n = j - self.ev_i
        self.ev_i = j
        return n

    def on_delete(self, ids):
        pass

    def on_insert(self, ids):
        pass

    def run(self):
        a = self.a
        for s in range(0, len(self.stream), a.batch):
            n_ev = self.apply_events(s)
            qb = self.stream[s:s + a.batch]
            # labels = the backend's answer; with an approximate backend the
            # exact top-k is also passed, to log retention against exact truth
            truth = topk_present(self.D, qb, self.k, key="answer")
            truth_x = None if self.D["truth"] is self.D["answer"] else \
                topk_present(self.D, qb, self.k, key="truth")
            # oracle reference: share of each query's top-k cached
            counts = (self.ref[truth] > 0).sum(axis=1)
            served, ret, ret_x = self.batch(qb, s, truth, truth_x)
            self.cost["topk"] += int((~served).sum())
            if s + len(qb) > a.warmup:
                keep = np.arange(len(qb)) + s >= a.warmup
                self.log.append({"t": s + len(qb), "n": int(keep.sum()),
                                 "served": int(served[keep].sum()),
                                 "served_retention_sum": float(ret[served & keep].sum()),
                                 "served_retention_exact_sum":
                                     float(ret_x[served & keep].sum()),
                                 "ret_hist": np.bincount(counts[keep],
                                                         minlength=self.k + 1).tolist(),
                                 "audited_served": 0, "lambda": None,
                                 "memory": self.n_cached(), "cached": self.n_cached(),
                                 "entries": self.n_entries(), "events": n_ev})
                if "initial" in self.D:
                    aff = (~self.D["initial"][truth]).any(axis=1) & keep
                    self.log[-1].update(aff_n=int(aff.sum()), aff_served=int((aff & served).sum()),
                                        aff_ret_sum=float(ret[aff & served].sum()))
        res = {"log": self.log, "events": self.events,
               "cost": dict(self.cost, vectors_fetched=self.fetched)}
        T = getattr(self, "timing", None)
        if T is not None:
            lat, hit = np.array(T["lat"]) * 1e3, np.array(T["hit"], dtype=bool)
            summ = lambda v: None if not len(v) else {"mean_ms": float(np.mean(v)), "p50_ms": float(np.median(v)),
                                                      "p99_ms": float(np.percentile(v, 99)), "n": int(len(v))}
            res["timing"] = {"every": T["every"], "n_timed": int(len(lat)), "hit_rate_timed": float(hit.mean()) if len(hit) else None,
                             "cache": dict(summ(lat) or {}, hits_ms=summ(lat[hit]), misses_ms=summ(lat[~hit])),
                             "lat_ms": lat.tolist(), "hit": hit.tolist(), "dt_ms": (np.array(T["dt"]) * 1e3).tolist()}
        return res


class Aker(Base):
    def __init__(self, D, stream, args):
        super().__init__(D, stream, args)
        self.n_keep = args.k + getattr(args, "aker_reserve", 0)
        self.reps = {}          # center -> {"tau", "cap", "ids", "d", "lc"}
        self.fifo = []          # insertion order of representatives
        self.alias = {}         # query -> representative
        self.children = {}      # representative -> its aliases
        self.holders = {}       # vector id -> representatives holding it
        self.ilog = []          # insert log (vector ids)
        self.sum_lc = 0         # sum of log checkpoints over representatives
        self.sum_r = 0.0        # sum of d_k / d_{k+D} over representatives
        self.rr = []            # round-robin order for the slow path

    def n_entries(self):
        return len(self.reps)

    # -- entry bookkeeping ---------------------------------------------------

    def _r(self, e):
        d = e["d"]
        if self.n_keep == self.k or len(d) < self.n_keep:
            return 1.0
        return float(d[self.k - 1] / max(d[-1], 1e-12))

    def _hold(self, c, ids):
        for x in ids.tolist():
            self.holders.setdefault(x, set()).add(c)

    def _unhold(self, c, ids):
        for x in ids.tolist():
            h = self.holders.get(x)
            if h is not None:
                h.discard(c)
                if not h:
                    del self.holders[x]

    def _insert(self, c):
        ids = topk_present(self.D, [c], self.n_keep)[0]
        d = self._dist(c, ids)
        self._acquire(ids)
        self._hold(c, ids)
        # tau starts at tau_init * minD and never exceeds cap * minD
        e = {"tau": self.a.aker_tau_init * float(d[0]), "cap": self.a.aker_cap * float(d[0]),
             "ids": ids, "d": d, "lc": len(self.ilog)}
        self.reps[c] = e
        self.sum_lc += e["lc"]
        self.sum_r += self._r(e)
        self.fifo.append(c)
        while self.n_cached() > self.M and self.fifo:
            v = self.fifo.pop(0)
            if v in self.reps:
                self._drop(v)

    def _drop(self, v):
        e = self.reps.pop(v)
        self._release(e["ids"])
        self._unhold(v, e["ids"])
        self.sum_lc -= e["lc"]
        self.sum_r -= self._r(e)
        for al in self.children.pop(v, []):
            self.alias.pop(al, None)

    def _offer(self, c, x, dx):
        """Insert vector x (distance dx to c's query) into c's neighbor list
        if it is closer than the farthest stored neighbor."""
        e = self.reps[c]
        if dx >= e["d"][-1] or x in set(e["ids"].tolist()):
            return
        self.sum_r -= self._r(e)
        pos = int(np.searchsorted(e["d"], dx))
        ids, d = np.insert(e["ids"], pos, x), np.insert(e["d"], pos, dx)
        self._acquire(np.array([x]))
        self._hold(c, np.array([x]))
        if len(ids) > self.n_keep:
            gone = ids[-1:]
            self._release(gone)
            self._unhold(c, gone)
            ids, d = ids[:-1], d[:-1]
        e["ids"], e["d"] = ids, d
        self.sum_r += self._r(e)

    # -- database changes ----------------------------------------------------

    def on_delete(self, ids):
        for x in ids.tolist():
            for c in list(self.holders.get(x, ())):
                e = self.reps[c]
                keep = e["ids"] != x
                self.sum_r -= self._r(e)
                e["ids"], e["d"] = e["ids"][keep], e["d"][keep]
                self._release(np.array([x]))
                self._unhold(c, np.array([x]))
                self.sum_r += self._r(e)
                if len(e["ids"]) < self.k:
                    self._drop(c)

    def risk(self):
        W = len(self.ilog)
        n = len(self.reps)
        if not W or not n:
            return 0.0
        return (self.sum_r / n) * (W - self.sum_lc / n) / W

    def slow_step(self):
        """Next representative (round-robin) checks the next batch of log
        records since its checkpoint."""
        if not self.reps:
            return
        while True:
            if not self.rr:
                self.rr = list(self.reps)
            c = self.rr.pop()
            if c in self.reps:
                break
        e = self.reps[c]
        lo, hi = e["lc"], min(e["lc"] + self.a.aker_batch, len(self.ilog))
        if hi > lo:
            xs = np.array(self.ilog[lo:hi], dtype=np.int64)
            xs = xs[self.D["present"][xs]]
            self.cost["refresh_dist"] += len(xs)
            for x, dx in zip(xs.tolist(), self._dist(c, xs).tolist()):
                self._offer(c, x, dx)
        self.sum_lc += hi - e["lc"]
        e["lc"] = hi

    def on_insert(self, ids):
        if not len(ids):
            return
        # fast path: for each new vector, the entry whose query is closest
        # (query filter); entries do not change while inserts are offered
        best = None
        if self.reps:
            cs = np.fromiter(self.reps, dtype=np.int64)
            cq = self.D["q"][cs]
            cq_sq = (cq * cq).sum(axis=1)
            best, best_d = [], []
            for s in range(0, len(ids), 4096):
                d2 = sq_dist(self.D["db"][ids[s:s + 4096]], cq, cq_sq)
                j = d2.argmin(axis=1)
                best.append(cs[j])
                best_d.append(np.sqrt(d2[np.arange(len(j)), j]))
            best, best_d = np.concatenate(best), np.concatenate(best_d)
        for i, x in enumerate(ids.tolist()):
            self.ilog.append(x)
            if best is not None and int(best[i]) in self.reps:
                self.cost["refresh_dist"] += 1
                self._offer(int(best[i]), x, float(best_d[i]))
            if self.risk() >= self.a.aker_risk:
                self.slow_step()

    # -- lookups -------------------------------------------------------------

    def batch(self, qb, s, truth, truth_x=None):
        k, Q = self.k, self.D["q"]
        n = len(qb)
        served = np.zeros(n, dtype=bool)
        ret = np.zeros(n)
        ret_x = ret if truth_x is None else np.zeros(n)
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
                r = self.reps[e]
                r["tau"] = min(r["tau"] * self.a.aker_alpha_l, r["cap"])
            elif q in self.alias and self.alias[q] in self.reps:
                e = self.alias[q]                    # alias hit (approximate)
                self.reps[e]["tau"] *= self.a.aker_alpha_t
            elif len(centers) and int(near[i]) in self.reps \
                    and near_d[i] < self.reps[int(near[i])]["tau"]:
                e = int(near[i])                     # approximate hit
                self.reps[e]["tau"] *= self.a.aker_alpha_t
                self.alias[q] = e
                self.children.setdefault(e, []).append(q)
            if e is None:
                misses.append(q)
            else:
                served[i] = True
                ret[i] = np.isin(truth[i], self.reps[e]["ids"][:k]).mean()
                if truth_x is not None:
                    ret_x[i] = np.isin(truth_x[i], self.reps[e]["ids"][:k]).mean()
            if self.ilog and self.risk() >= self.a.aker_risk:
                self.slow_step()
        for q in misses:
            if q not in self.reps:
                self._insert(q)
        return served, ret, ret_x


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
        self.remove_deleted = getattr(args, "qv_deletes", "keep") == "remove"
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

    def on_delete(self, ids):
        if not self.remove_deleted:
            return
        gone = ids[self.ref[ids] > 0]
        for x in gone.tolist():
            self.size[int(self.owner[x])] -= 1
        self._release(gone)
        self.owner[gone] = -1

    def _next_fill(self, t):
        empty = [i for i in range(self.n_mini) if self.size[i] == 0]
        if empty:
            return empty[0]
        v = min(range(self.n_mini), key=lambda i: self.last_used[i])
        ids = np.asarray(self.mini[v], dtype=np.int64)
        ids = ids[self.owner[ids] == v] if len(ids) else ids
        self._release(ids)
        self.owner[ids] = -1
        self.mini[v], self.size[v] = [], 0
        self.events.append({"t": t, "event": "evict_mini", "mini": v})
        return v

    def _insert(self, q, t):
        ids = topk_present(self.D, [q], self.k)[0]
        new = ids[self.ref[ids] == 0]
        if self.size[self.fill] + len(new) > self.cap:
            self.fill = self._next_fill(t)
        self._acquire(new)
        self.owner[new] = self.fill
        self.mini[self.fill].extend(new.tolist())
        self.size[self.fill] += len(new)
        self.last_used[self.fill] = t

    def batch(self, qb, s, truth, truth_x=None):
        k, Q, a = self.k, self.D["q"], self.a
        n = len(qb)
        served = np.zeros(n, dtype=bool)
        ret = np.zeros(n)
        ret_x = ret if truth_x is None else np.zeros(n)
        ids = np.flatnonzero(self.ref > 0)
        if len(ids) >= k:
            cv = self.D["db"][ids]
            d2 = sq_dist(Q[qb], cv, (cv * cv).sum(axis=1))
            part = np.argpartition(d2, k - 1, axis=1)[:, :k]
            dk = np.sqrt(np.take_along_axis(d2, part, axis=1).max(axis=1))
            # answer = the k nearest cached vectors by exact distance (float32
            # distances can misorder near-ties): re-rank a few extra candidates
            m = min(k + 10, len(ids))
            cand = ids[np.argpartition(d2, m - 1, axis=1)[:, :m]]
            exact = ((self.D["db"][cand].astype(np.float64)
                      - Q[qb][:, None, :].astype(np.float64)) ** 2).sum(axis=2)
            near = np.take_along_axis(cand, np.argsort(exact, axis=1)[:, :k], axis=1)
            near_dk = np.sqrt(np.sort(exact, axis=1)[:, k - 1])
        present = self.D["present"]
        cached_now = self.ref > 0
        misses = []
        for i, q in enumerate(qb):
            q = int(q)
            key = self.region(q)
            th = self.theta.get(key)
            if len(ids) >= k and th is not None and dk[i] <= (1 + a.qv_D) * th:
                served[i] = True
                # a cached true neighbor is returned if it is at least as close as
                # the answer's k-th vector (equal-distance duplicates count;
                # cached deleted vectors that are closer push true ones out)
                dt = self._dist(q, truth[i])
                ret[i] = (cached_now[truth[i]] & (dt <= near_dk[i] + 1e-6)).mean()
                if truth_x is not None:
                    dx = self._dist(q, truth_x[i])
                    ret_x[i] = (cached_now[truth_x[i]] & (dx <= near_dk[i] + 1e-6)).mean()
                self.cost["deleted_returned"] += int((~present[near[i]]).sum())
                for m in set(self.owner[near[i]].tolist()):
                    if m >= 0:
                        self.last_used[m] = s + i
            else:
                db_k = float(self._dist(q, truth[i, k - 1:k])[0])
                self.theta[key] = db_k if th is None else (1 - a.qv_alpha) * th + a.qv_alpha * db_k
                misses.append(q)
        for q in misses:
            self._insert(q, s + len(qb))
        return served, ret, ret_x


class QVCacheCore(QVCache):
    """QVCache with its request path in the C++ core (doc 9ah; cachecore.QVCore, exact search over the cached
    vectors, as QVCache.batch), one request at a time: a miss's answer is inserted right away (QVCache.batch
    inserts the batch's misses at its end, so a repeat within the batch could never hit). Retention of a hit:
    share of the backend's top-k in the served answer. Run with .venv-core (Python 3.11)."""

    def __init__(self, D, stream, args):
        super().__init__(D, stream, args)
        from simulations.approximate_hits.cachecore import QVCore
        # timing (--timing-compose): mini-index graphs (hnswlib, search list 64) and every request timed;
        # headline: exact search over the cached vectors
        self.timing = None
        graph = bool(getattr(args, "timing_compose", None)) or bool(getattr(args, "qv_graph", False))
        if getattr(args, "timing_compose", None):
            self.timing = {"compose": np.load(args.timing_compose), "every": args.timing_every, "lat": [], "hit": [], "dt": []}
        self.core = QVCore(self.D["db"].shape[1], self.k, self.n_mini, int(self.cap), float(args.qv_D),
                           float(args.qv_alpha), self.mu, np.ascontiguousarray(self.P), self.lo, self.hi,
                           int(args.qv_buckets), graph)

    def n_entries(self):
        return self.core.n_minis_used()

    def batch(self, qb, s, truth, truth_x=None):
        n = len(qb)
        served = np.zeros(n, dtype=bool)
        ret = np.zeros(n)
        ret_x = ret if truth_x is None else np.zeros(n)
        Q, db = self.D["q"], self.D["db"]
        import time as _t
        T = self.timing
        for i, q in enumerate(qb):
            x = np.ascontiguousarray(np.asarray(Q[int(q)], np.float32))
            a0 = _t.perf_counter()
            hit, near, _ = self.core.lookup(x, float(s + i))
            dt = _t.perf_counter() - a0
            pos = s + i - self.a.warmup
            if T is not None and pos >= 0 and pos % T["every"] == 0:
                j = pos // T["every"]
                T["lat"].append(dt + (0.0 if hit else float(T["compose"][j]) if j < len(T["compose"]) else np.nan))
                T["hit"].append(bool(hit))
                T["dt"].append(dt)
            if hit:
                served[i] = True
                near = np.asarray(near, dtype=np.int64)
                ret[i] = np.isin(truth[i], near).mean()
                if truth_x is not None:
                    ret_x[i] = np.isin(truth_x[i], near).mean()
            else:
                ans = np.ascontiguousarray(np.asarray(truth[i], dtype=np.int64))
                self.core.on_miss(x, ans, np.ascontiguousarray(np.asarray(db[ans], np.float32)), float(s + i))
        ids = np.asarray(self.core.cached_ids(), dtype=np.int64)       # for the oracle reference counts
        self.ref[:] = 0
        self.ref[ids] = 1
        self.n_c = len(ids)
        return served, ret, ret_x


class AkerRealSim(Base):
    """The real Aker library (aker_real/libakershim*.so, ctypes; doc 9q) in the shared driver (doc 9ah), one
    request at a time as eval_aker_real.run_quality: lookup; on a miss, insert the backend's top k + delta.
    Settings: --aker-real-delta, --aker-real-at (alpha tighten), --aker-real-al (alpha loosen), risk 0.35,
    or Proximity mode (--aker-real-prox f: global threshold f x the median squared exact-NN distance of
    5,000 queries; needs the proximity build first in LD_LIBRARY_PATH). Timed mode as QVCacheCore."""

    def __init__(self, D, stream, args):
        super().__init__(D, stream, args)
        import os
        import sys as _sys
        from pathlib import Path as _P
        _sys.path.insert(0, str(_P(__file__).parent / "aker_real"))
        from eval_aker_real import Aker as _Aker
        if args.aker_real_prox is not None:
            q, db, top = self.D["q"], self.D["db"], self.D["truth"]
            rows = np.random.default_rng(0).choice(len(q), min(5000, len(q)), replace=False)
            nn = np.asarray(db[np.asarray(top[rows, 0], dtype=np.int64)], dtype=np.float32)
            thr = args.aker_real_prox * float(np.median(((np.asarray(q[rows], np.float32) - nn) ** 2).sum(axis=1)))
            os.environ["AKER_GLOBAL_THRESH"] = repr(thr)
        self.delta = args.aker_real_delta
        self.ak = _Aker(self.D["q"].shape[1], int(args.budget * len(self.D["db"])), self.k, self.delta, 0.35,
                        args.aker_real_at, args.aker_real_al)
        self.timing = None
        if getattr(args, "timing_compose", None):
            self.timing = {"compose": np.load(args.timing_compose), "every": args.timing_every, "lat": [], "hit": [], "dt": []}

    def n_entries(self):
        return 0

    def batch(self, qb, s, truth, truth_x=None):
        import time as _t
        n = len(qb)
        served = np.zeros(n, dtype=bool)
        ret = np.zeros(n)
        ret_x = ret if truth_x is None else np.zeros(n)
        Q, db, T = self.D["q"], self.D["db"], self.timing
        for i, q in enumerate(qb):
            q = int(q)
            x = np.ascontiguousarray(np.asarray(Q[q], np.float32))
            a0 = _t.perf_counter()
            kind, ids, _, _ = self.ak.lookup(q + 1, x)
            dt = _t.perf_counter() - a0
            hit = kind in (1, 2, 4)
            pos = s + i - self.a.warmup
            if T is not None and pos >= 0 and pos % T["every"] == 0:
                j = pos // T["every"]
                T["lat"].append(dt + (0.0 if hit else float(T["compose"][j]) if j < len(T["compose"]) else np.nan))
                T["hit"].append(bool(hit))
                T["dt"].append(dt)
            if hit:
                got = np.asarray(ids, dtype=np.int64) - 1
                served[i] = True
                ret[i] = np.isin(truth[i], got).mean()
                if truth_x is not None:
                    ret_x[i] = np.isin(truth_x[i], got).mean()
            else:
                nid = np.ascontiguousarray(topk_present(self.D, [q], self.k + self.delta, key="answer")[0].astype(np.int64))
                self.ak.insert(q + 1, x, nid + 1, np.ascontiguousarray(np.asarray(db[nid], np.float32)))
        return served, ret, ret_x
