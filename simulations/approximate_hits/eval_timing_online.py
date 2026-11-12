#!/usr/bin/env python3
"""
End-to-end timing of every technique in front of the SAME real DiskANN (context doc 9k).

One request at a time, single thread, quiet-machine guard (timing_env.py). The request loop is
identical for every technique; per request (timed together):
    lookup in the cache; on a hit the cached answer is returned (ours / LRU: with probability
    --audit-rate the backend is also asked, to label the hit for calibration); on a miss DiskANN
    answers (search list --L, the effort tuned to recall@10 0.95), then the technique updates
    itself (ours / LRU: label for calibration; LRU: the miss becomes an entry, which needs the
    backend's top-500 for its ball and shell, evicting least recently used entries; Aker: insert
    via its C library; QVCache: insert into the current mini-index); ours / LRU recalibrate the
    cutoff every --cal-every requests on the labels of the last --cal-span requests (as the
    simulator does), inside the request that triggers it.
Then the same requests against DiskANN alone, and DiskANN with its node cache given the
technique's memory ("equal memory").

Techniques (--method):
  ours     the greedy cache of online_sim.py --save-state (built from DiskANN's answers), hit path
           = HNSW 32 over cached vectors + intruder count over the 10 best-fitting balls
  lru      our hit rule + LRU population (the simulator's lru: every miss an entry with K =
           --k-lru, shell from its top-500 at the state's sampling rate p), starting from the same
           warm-up state; the same hit path, maintained incrementally (HNSW: vectors added on
           insert, evicted ones masked; graph rebuilt when > 30% are dead)
  aker     the real Aker library (aker_real/, libakershim[_prox].so), warmed up on the first
           --warmup requests with DiskANN's answers (diskann.npz, the same search), untimed
  qvcache  QVCache as described in its paper (C++ with Python bindings; mini-indexes are
           FreshVamana graphs): mini-indexes = diskannpy dynamic in-memory indexes (FreshVamana),
           region thresholds over a PCA projection as in online_baselines.QVCache, warmed up like
           Aker
Fidelity notes: LRU's newly added shell vectors are compared at full precision (the simulator
quantizes them to 4 bits; memory is counted at 4 bits as in the simulator); QVCache searches its
mini-indexes with graph search (the simulator's reimplementation searches exactly).

  .venv-ann/bin/python simulations/approximate_hits/eval_timing_online.py --method ours \
      --prep simulations/approximate_hits/results/online/spacev-aker-0.99 --budget 0.01 \
      --state .../state_b0.01.npz --out .../ours_b0.01.json
"""

import sys
import json
import time
import argparse
import logging
from pathlib import Path
from collections import deque

sys.path.append(str(Path(__file__).parent.parent.parent))
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
RES = Path("simulations/approximate_hits/results")
SHELL_BALLS = 10      # the hit path's shell scope (FastHitPath "balls10", online_sim --shell-balls 10)


def summary(lat):
    lat = np.asarray(lat)
    if not len(lat):
        return None
    return {"mean_ms": float(lat.mean() * 1e3), "p50_ms": float(np.percentile(lat, 50) * 1e3),
            "p99_ms": float(np.percentile(lat, 99) * 1e3), "n": int(len(lat))}


# -- calibration (ours, LRU): the simulator's rule ----------------------------------------------

class Calibrator:
    """Labels (score, retention, weight) of misses (free: the backend answered) and audited hits
    (weight 1 / audit rate); lambda = select_weighted over the last cal_span requests, every
    cal_every requests."""

    def __init__(self, lam, target, delta, cal_span, cal_every):
        self.lam, self.T, self.delta = lam, target, delta
        self.span, self.every = cal_span, cal_every
        self.lab = deque()

    def add(self, t, s, y, w):
        self.lab.append((t, s, y, w))

    def seed(self, st):
        """The warm-up's calibration labels (online_sim --save-state cal_*), kept as the simulator
        keeps them until they fall out of the last cal_span requests."""
        if "cal_s" in st:
            for t, s, y, w in zip(st["cal_t"], st["cal_s"], st["cal_y"], st["cal_w"]):
                self.add(int(t), float(s), float(y), float(w))

    def maybe(self, t):
        if t % self.every:
            return
        from simulations.approximate_hits.threshold_selection import select_weighted, threshold_grid
        while self.lab and self.lab[0][0] <= t - self.span:
            self.lab.popleft()
        if len(self.lab) < 2:
            return
        a = np.array([(s, y, w) for _, s, y, w in self.lab])
        a = a[np.isfinite(a).all(axis=1)]          # safety: a non-finite score would void every cutoff
        if len(a) < 2:
            return
        self.lam = select_weighted(a[:, 0], a[:, 1], a[:, 2], threshold_grid(a[:, 0]), self.T, self.delta)


# -- ours: static greedy cache ------------------------------------------------------------------

class Ours:
    """Static after warm-up. Default (stationary workloads, decided with Jialin 11-11): no
    recalibration, no audits, no drift tests (in the simulator these never fire on the stationary
    traces, and the static cache keeps the target); --ours-recal: recalibrate as the simulator."""

    def __init__(self, args, db, q, st):
        self.audits = self.recal = bool(getattr(args, "ours_recal", False))
        from simulations.approximate_hits.eval_timing import FastHitPath
        self.hp = FastHitPath(db, q, st, "hnsw32", "balls10")
        self.k = int(st["k"])
        self.cached = np.zeros(len(db), dtype=bool)
        self.cached[st["cached_ids"]] = True
        lam = float(st["lam"])
        self.cal = Calibrator(None if np.isnan(lam) else lam, args.target, args.cal_delta,
                              args.cal_span, args.cal_every)
        self.cal.seed(st)
        self.miss_k = self.k
        self.memory_bytes = int(sum(self.hp.bytes.values()) + self.hp.extra_bytes)
        self.t = {"knn": 0.0, "balls": 0.0, "shell": 0.0, "model": 0.0}

    def lookup(self, qi, x):
        self.hp.lam = np.inf if self.cal.lam is None else self.cal.lam
        hit, s, near = self.hp.decide(x, self.t)
        near = np.asarray(near)
        if len(near) < self.k or np.any(near < 0) or not np.isfinite(s):
            hit = False        # graph search came back short (never seen for the static graph)
            self.fallbacks = getattr(self, "fallbacks", 0) + 1
        self._s = s
        return bool(hit), near

    def label(self, t, answer, w):
        self.cal.add(t, self._s, float(self.cached[answer[:self.k]].mean()), w)

    def retention(self, answer):
        """The simulator's retention: share of the backend's top-k answer that is cached."""
        return float(self.cached[answer[:self.k]].mean())

    def on_miss(self, t, qi, x, ids, ix):
        if self.recal:
            self.label(t, ids, 1.0)

    def after(self, t):
        if self.recal:
            self.cal.maybe(t)


# -- LRU: our hit rule + LRU population, maintained incrementally -------------------------------

class LRU(Ours):
    def __init__(self, args, db, q, st, deep):
        import faiss
        self.faiss = faiss
        self.db, self.q, self.k, self.K, self.R = db, q, int(st["k"]), args.k_lru, 500
        self.deep = deep
        self.dim = db.shape[1]
        self.budget = float(st["budget"])
        self.p, self.bits = float(st["p"]), int(st["bits"])
        self.draw = np.random.default_rng(1).random(len(db))     # the simulator's per-vector draw
        self.mu, self.sd, self.coef, self.b = st["mu"], st["sd"], st["coef"], float(st["intercept"])
        lam = float(st["lam"])
        self.cal = Calibrator(None if np.isnan(lam) else lam, args.target, args.cal_delta,
                              args.cal_span, args.cal_every)
        self.cal.seed(st)
        self.audits = self.recal = True       # LRU's cache changes on every miss: recalibration needed
        self.miss_k = self.R
        self.ref = np.zeros(len(db), dtype=np.int32)              # entries holding a vector
        self.slot = np.full(len(db), -1, dtype=np.int64)          # vector id -> row in cv
        self.cv = np.zeros((1024, self.dim), dtype=np.float32)
        self.cv_id = np.zeros(1024, dtype=np.int64)
        self.n_cv, self.dead = 0, 0
        self.entries = {}                                         # center -> entry dict
        self.holders = {}                                         # vector id -> set of centers
        self.n_samp = 0
        self.sref = np.zeros(len(db), dtype=np.int32)
        sv0 = np.asarray(st["sv"], dtype=np.float32)
        self.qlo, self.qhi = (sv0.min(axis=0), sv0.max(axis=0)) if len(sv0) else (None, None)
        if self.qlo is None:
            self.bits = 32
        self.background = self.rebuild_s = 0.0
        self.fallbacks = 0
        self.hnsw = None
        self._rebuild()
        self._load_state(st)
        self.t = {"knn": 0.0, "balls": 0.0, "shell": 0.0, "model": 0.0, "insert": 0.0, "evict": 0.0}

    # vectors
    def _rebuild(self):
        faiss = self.faiss
        live = np.flatnonzero(self.ref[self.cv_id[:self.n_cv]] > 0) if self.n_cv else np.empty(0, int)
        ids = self.cv_id[live]
        vec = self.cv[live]
        self.cv = np.zeros((max(1024, 2 * len(ids)), self.dim), dtype=np.float32)
        self.cv_id = np.zeros(len(self.cv), dtype=np.int64)
        self.cv[:len(ids)], self.cv_id[:len(ids)] = vec, ids
        self.slot[:] = -1
        self.slot[ids] = np.arange(len(ids))
        self.n_cv, self.dead = len(ids), 0
        a = time.perf_counter()
        self.hnsw = faiss.IndexHNSWFlat(self.dim, 32)
        self.hnsw.hnsw.efConstruction = 200
        self.hnsw.hnsw.efSearch = 32
        if len(ids):
            # a rebuild is background work in a real system: built with 32 threads, its time
            # reported separately ("rebuild_s") and not charged to the request that triggers it
            faiss.omp_set_num_threads(32)
            self.hnsw.add(self.cv[:len(ids)])
            faiss.omp_set_num_threads(1)
        self.hnsw.hnsw.efConstruction = 40    # incremental inserts: 2x cheaper than 200 (measured)
        self.background += time.perf_counter() - a
        self.rebuild_s += time.perf_counter() - a

    def _add_vectors(self, ids, index=True):
        new = ids[self.slot[ids] < 0]
        if len(new):
            if self.n_cv + len(new) > len(self.cv):
                grow = max(len(self.cv), len(new))
                self.cv = np.concatenate([self.cv, np.zeros((grow, self.dim), np.float32)])
                self.cv_id = np.concatenate([self.cv_id, np.zeros(grow, np.int64)])
            v = np.asarray(self.db[new], dtype=np.float32)
            self.cv[self.n_cv:self.n_cv + len(new)], self.cv_id[self.n_cv:self.n_cv + len(new)] = v, new
            self.slot[new] = np.arange(self.n_cv, self.n_cv + len(new))
            self.n_cv += len(new)
            if index:
                self.hnsw.add(v)
        self.dead -= int(((self.ref[ids] == 0) & (self.slot[ids] >= 0)).sum()) - len(new)  # revived rows
        old = self._eff(ids)
        self.ref[ids] += 1
        self.n_samp += int(self._eff(ids).sum() - old.sum())

    def _drop_vectors(self, ids):
        old = self._eff(ids)
        self.ref[ids] -= 1
        self.n_samp += int(self._eff(ids).sum() - old.sum())
        gone = ids[self.ref[ids] == 0]
        self.dead += len(gone)
        if self.dead > 0.3 * max(self.n_cv, 1):
            self._rebuild()

    def n_cached(self):
        return self.n_cv - self.dead

    def memory(self):
        return self.n_cached() + self.n_samp * self.bits / 32

    def _eff(self, ids):
        # a vector counts as a shell sample iff some entry's shell sample holds it and it is not
        # cached (the simulator's shell = union of the entries' shells minus cached vectors)
        return (self.sref[ids] > 0) & (self.ref[ids] == 0)

    def _samp(self, sid, inc):
        # distinct shell-sample vectors (one sample is shared by every entry whose shell holds it)
        old = self._eff(sid)
        self.sref[sid] += 1 if inc > 0 else -1
        self.n_samp += int(self._eff(sid).sum() - old.sum())

    # entries
    def _add_entry(self, c, x, row, K):
        ball = np.asarray(row[:K], dtype=np.int64)
        shell = np.asarray(row[K:self.R], dtype=np.int64)
        shell = shell[self.draw[shell] < self.p]
        rad = float(np.sqrt(((np.asarray(self.db[ball[-1]], np.float32) - x) ** 2).sum())) + 1e-12
        self._add_vectors(ball)
        sv = np.asarray(self.db[shell], dtype=np.float32) if len(shell) else np.zeros((0, self.dim), np.float32)
        if len(sv) and self.bits < 32:
            # b-bit scalar quantization as the simulator stores the sample (eval_core.quantize),
            # over the warm-up sample's per-dimension range
            step = np.maximum(self.qhi - self.qlo, 1e-12) / (2 ** self.bits - 1)
            sv = (self.qlo + np.round((np.clip(sv, self.qlo, self.qhi) - self.qlo) / step) * step).astype(np.float32)
        self.entries[c] = {"x": x.astype(np.float32), "r": rad, "ball": ball, "sv": sv, "sid": shell,
                           "sv_sq": (sv * sv).sum(axis=1), "used": 0}
        self._samp(shell, 1)
        for v in ball.tolist():
            self.holders.setdefault(v, set()).add(c)
        self._arr_add(c)

    def _remove_entry(self, c):
        e = self.entries.pop(c)
        for v in e["ball"].tolist():
            h = self.holders.get(v)
            if h is not None:
                h.discard(c)
                if not h:
                    del self.holders[v]
        self._samp(e["sid"], -1)
        self._drop_vectors(e["ball"])
        self._arr_del(c)

    def _load_state(self, st):
        # entries of the warm-up cache: centers and radii are in the state; a ball is the prefix of
        # the center's deep list (DiskANN top-500, the simulator's D["rank"] row) within the radius,
        # its shell the state's shell sample (entry_samp)
        sv_all, es, eo = st["sv"].astype(np.float32), st["entry_samp"], st["entry_samp_off"]
        for j, (c, r) in enumerate(zip(st["centers"], st["radius"])):
            x = np.asarray(self.q[c], dtype=np.float32)
            row = np.asarray(self.deep[c], dtype=np.int64)
            row = row[row >= 0]
            d = np.sqrt(((np.asarray(self.db[row], np.float32) - x) ** 2).sum(axis=1))
            ball = row[d <= r * (1 + 1e-6)]
            pos = es[eo[j]:eo[j + 1]]
            sv = sv_all[pos]
            self._add_vectors(ball, index=False)
            sid = np.asarray(st["samp_ids"][pos], dtype=np.int64)
            self.entries[int(c)] = {"x": x, "r": float(r), "ball": ball, "sv": sv, "sid": sid,
                                    "sv_sq": (sv * sv).sum(axis=1), "used": 0}
            self._samp(sid, 1)
            for v in ball.tolist():
                self.holders.setdefault(v, set()).add(int(c))
        self._rebuild()                       # one multi-threaded graph build over the warm-up cache
        self.background = self.rebuild_s = 0.0
        self._arr_build()
        self.memory_bytes = self._bytes()
        logger.info(f"LRU warm-up cache: {len(self.entries)} entries, {self.n_cached()} vectors, "
                    f"{self.n_samp} shell samples; memory {self.memory():.0f} of {self.budget:.0f}")

    # entry centers and radii as arrays, maintained in place (append; delete = move the last row)
    def _arr_build(self):
        self.keys = list(self.entries)
        n = self.n_e = len(self.keys)
        cap = max(1024, 2 * n)
        self.cq = np.zeros((cap, self.dim), dtype=np.float32)
        self.cq_sq, self.rad = np.zeros(cap), np.ones(cap)
        for i, c in enumerate(self.keys):
            self.cq[i], self.rad[i] = self.entries[c]["x"], self.entries[c]["r"]
        self.cq_sq[:n] = (self.cq[:n] * self.cq[:n]).sum(axis=1)
        self.row = {c: i for i, c in enumerate(self.keys)}

    def _arr_add(self, c):
        if self.n_e == len(self.cq):
            g = len(self.cq)
            self.cq = np.concatenate([self.cq, np.zeros((g, self.dim), np.float32)])
            self.cq_sq, self.rad = np.concatenate([self.cq_sq, np.zeros(g)]), np.concatenate([self.rad, np.ones(g)])
        i, e = self.n_e, self.entries[c]
        self.cq[i], self.rad[i], self.cq_sq[i] = e["x"], e["r"], float(e["x"] @ e["x"])
        self.keys.append(c)
        self.row[c] = i
        self.n_e += 1

    def _arr_del(self, c):
        i, last = self.row.pop(c), self.n_e - 1
        if i != last:
            self.cq[i], self.rad[i], self.cq_sq[i] = self.cq[last], self.rad[last], self.cq_sq[last]
            self.keys[i] = self.keys[last]
            self.row[self.keys[i]] = i
        self.keys.pop()
        self.n_e -= 1

    def _bytes(self):
        return int(self.n_cached() * self.dim * 4 + self.n_samp * self.dim * self.bits / 8
                   + self.hnsw.ntotal * 32 * 2 * 4 + len(self.entries) * (self.dim * 4 + 8))

    def lookup(self, qi, x):
        from simulations.approximate_hits.eval_timing import N_BALLS, POOR_FIT, EPS
        k, t = self.k, self.t
        a = time.perf_counter()
        kk = k + 8
        while True:
            dd, ii = self.hnsw.search(x[None], min(kk, max(self.hnsw.ntotal, 1)))
            ok = ii[0] >= 0
            ii, dd = ii[0][ok], dd[0][ok]
            live = self.ref[self.cv_id[ii]] > 0
            if live.sum() >= k or kk >= self.hnsw.ntotal:
                break
            kk *= 4
        ii, dd = ii[live][:k], dd[live][:k]
        if len(ii) < k and self.n_cached() >= k:
            # the graph search found fewer than k live vectors (rare: FAISS HNSW with incremental
            # inserts returned no result at all once in ~15,000 lookups): exact search over the
            # live cached vectors, as the simulator does
            self.fallbacks += 1
            rows = np.flatnonzero(self.ref[self.cv_id[:self.n_cv]] > 0)
            d2 = ((self.cv[rows] - x) ** 2).sum(axis=1)
            o = np.argpartition(d2, k - 1)[:k]
            o = o[np.argsort(d2[o])]
            ii, dd = rows[o], d2[o]
        near = self.cv_id[ii]
        nr = float(np.sqrt(max(dd.max(), 0.0))) + EPS if len(dd) else np.inf
        b = time.perf_counter()
        n_e = self.n_e
        x_sq = float(x @ x)
        de = np.sqrt(np.maximum(self.cq_sq[:n_e] - 2.0 * (self.cq[:n_e] @ x) + x_sq, 0.0))
        fit = (nr + de) / self.rad[:n_e]
        m = min(N_BALLS, len(fit))
        best = np.argpartition(fit, m - 1)[:m] if m < len(fit) else np.arange(m)
        best = best[np.argsort(fit[best])]
        f = np.full(2 * N_BALLS + 1, POOR_FIT)
        f[0:2 * m:2] = de[best] / self.rad[best]
        f[1:2 * m:2] = nr / self.rad[best]
        c = time.perf_counter()
        # intruders: distinct sampled shell vectors of the SHELL_BALLS best-fitting balls inside
        # the neighborhood radius (as FastHitPath "balls10"; a vector sampled by several entries
        # counts once)
        nb = min(SHELL_BALLS, len(fit))
        sel = np.argpartition(fit, nb - 1)[:nb] if nb < len(fit) else np.arange(len(fit))
        inside = []
        for j in sel:
            e = self.entries[self.keys[j]]
            if len(e["sv"]):
                sid = e["sid"][(e["sv_sq"] - 2.0 * (e["sv"] @ x) + x_sq) < nr * nr]
                inside.append(sid[self.ref[sid] == 0])       # cached vectors are not shell samples
        cnt = len(np.unique(np.concatenate(inside))) if inside else 0
        f[-1] = np.log1p(cnt)
        d = time.perf_counter()
        s = float(((f - self.mu) / self.sd) @ self.coef + self.b)
        if not np.isfinite(s) and not getattr(self, "_warned", False):
            self._warned = True
            logger.warning(f"non-finite score at t={getattr(self, '_now', None)}: n near {len(near)}, "
                           f"nr {nr}, hnsw ntotal {self.hnsw.ntotal}, n_cv {self.n_cv}, dead {self.dead}, "
                           f"entries {self.n_e}, min radius {self.rad[:self.n_e].min() if self.n_e else None}, "
                           f"f {f.tolist()}")
        hit = len(near) >= k and self.cal.lam is not None and s >= self.cal.lam
        e = time.perf_counter()
        t["knn"] += b - a
        t["balls"] += c - b
        t["shell"] += d - c
        t["model"] += e - d
        self._s = s
        if hit:      # LRU: entries supplying a served vector were used now
            for v in near.tolist():
                for cc in self.holders.get(v, ()):
                    self.entries[cc]["used"] = self._now
        return bool(hit), near

    def label(self, t, answer, w):
        self.cal.add(t, self._s, float((self.ref[answer[:self.k]] > 0).mean()), w)

    def retention(self, answer):
        return float((self.ref[answer[:self.k]] > 0).mean())

    def on_miss(self, t, qi, x, ids, ix):
        self.label(t, ids, 1.0)
        a = time.perf_counter()
        if int(qi) not in self.entries:
            self._add_entry(int(qi), x, ids, self.K)
            self.entries[int(qi)]["used"] = t
        b = time.perf_counter()
        if self.memory() > self.budget:
            order = sorted(self.entries, key=lambda c: self.entries[c]["used"])
            for c in order:
                if self.memory() <= self.budget:
                    break
                if c != int(qi):
                    self._remove_entry(c)
        self.t["insert"] += b - a
        self.t["evict"] += time.perf_counter() - b

    def after(self, t):
        self._now = t + 1
        self.cal.maybe(t)


# -- Aker (real library) ------------------------------------------------------------------------

class AkerReal:
    audits = False

    def __init__(self, args, db, q, warm_q, warm_ans):
        sys.path.insert(0, str(Path(__file__).parent / "aker_real"))
        import os
        from eval_aker_real import Aker
        if args.prox_thresh is not None:
            os.environ["AKER_GLOBAL_THRESH"] = repr(args.prox_thresh)
        self.k, self.delta, self.db = args.k, args.aker_delta, db
        self.miss_k = args.k + args.aker_delta
        self.ak = Aker(q.shape[1], int(args.budget * len(db)), args.k, args.aker_delta, 0.35,
                       args.alpha_tighten, args.alpha_loosen)
        for qi, row in zip(warm_q, warm_ans):        # warm-up, untimed
            kind, _, _, _ = self.ak.lookup(int(qi) + 1, q[qi])
            if kind in (0, 3):
                nid = np.ascontiguousarray(row[:self.miss_k], dtype=np.int64)
                self.ak.insert(int(qi) + 1, q[qi], nid + 1, np.ascontiguousarray(db[nid], dtype=np.float32))
        import re
        self.memory_bytes = int(float(re.search(r"Total: ([0-9.]+) MB", self.ak.status()).group(1)) * 1e6)

    def lookup(self, qi, x):
        kind, ids, _, _ = self.ak.lookup(int(qi) + 1, x)
        return kind in (1, 2, 4), np.array(ids - 1)

    def on_miss(self, t, qi, x, ids, ix):
        nid = np.ascontiguousarray(ids[:self.miss_k], dtype=np.int64)
        self.ak.insert(int(qi) + 1, x, nid + 1, np.ascontiguousarray(self.db[nid], dtype=np.float32))

    def after(self, t):
        pass


# -- QVCache (FreshVamana mini-indexes) ---------------------------------------------------------

class QVCacheFast:
    audits = False

    def __init__(self, args, db, q, warm_q, warm_ans):
        import diskannpy
        self.dk = diskannpy
        self.db, self.q, self.k, self.dim = db, q, args.k, db.shape[1]
        self.miss_k = args.k
        self.n_mini, self.D, self.alpha = args.qv_mini, args.qv_D, 0.9
        self.cap = int(args.budget * len(db) / self.n_mini)
        w = np.asarray(q[warm_q], dtype=np.float64)                    # PCA on the warm-up queries
        self.mu = w.mean(axis=0)
        _, _, vt = np.linalg.svd(w - self.mu, full_matrices=False)
        self.P = vt[:args.qv_dims].T
        z = (w - self.mu) @ self.P
        self.lo, self.hi, self.nb = z.min(axis=0), z.max(axis=0), args.qv_buckets
        self.theta = {}
        self.idx = [None] * self.n_mini
        self.ids = [[] for _ in range(self.n_mini)]                   # vector ids per mini-index
        self.owner = np.full(len(db), -1, dtype=np.int32)
        self.last_used = [0] * self.n_mini
        self.fill = 0
        self.t = {"lookup": 0.0, "insert": 0.0}
        for i, (qi, row) in enumerate(zip(warm_q, warm_ans)):         # warm-up, untimed
            hit, _ = self.lookup(qi, np.asarray(q[qi], dtype=np.float32))
            if not hit:
                self.on_miss(i, qi, np.asarray(q[qi], dtype=np.float32), np.asarray(row), None)
        self.memory_bytes = self._bytes()

    def _bytes(self):
        n = int((self.owner >= 0).sum())
        return n * (self.dim * 4 + 32 * 4) + len(self.theta) * 16

    def _new_index(self):
        return self.dk.DynamicMemoryIndex(distance_metric="l2", vector_dtype=np.float32,
                                          dimensions=self.dim, max_vectors=self.cap + 1,
                                          complexity=64, graph_degree=32, num_threads=1)

    def region(self, x):
        z = (np.asarray(x, dtype=np.float64) - self.mu) @ self.P
        b = np.floor((z - self.lo) / np.maximum(self.hi - self.lo, 1e-12) * self.nb)
        return tuple(np.clip(b, 0, self.nb - 1).astype(int))

    def lookup(self, qi, x):
        a = time.perf_counter()
        ds, ii = [], []
        for m, ix in enumerate(self.idx):
            if ix is not None and len(self.ids[m]):
                r = ix.search(x, k_neighbors=min(self.k, len(self.ids[m])), complexity=64)
                ds.append(r.distances)
                ii.append(r.identifiers.astype(np.int64) - 1)
        hit, near = False, np.empty(0, dtype=np.int64)
        if ds:
            d = np.concatenate(ds)
            i = np.concatenate(ii)
            o = np.argsort(d)[:self.k]
            near, dk = i[o], float(np.sqrt(max(d[o][-1], 0.0)))
            th = self.theta.get(self.region(x))
            hit = len(near) >= self.k and th is not None and dk <= (1 + self.D) * th
            if hit:
                for m in set(self.owner[near].tolist()):
                    if m >= 0:
                        self.last_used[m] = getattr(self, "_now", 0)
        self.t["lookup"] += time.perf_counter() - a
        return bool(hit), near

    def on_miss(self, t, qi, x, ids, ix):
        a = time.perf_counter()
        ids = np.asarray(ids[:self.k], dtype=np.int64)
        key = self.region(x)
        dk = float(np.sqrt(((np.asarray(self.db[ids[-1]], np.float32) - x) ** 2).sum()))
        th = self.theta.get(key)
        self.theta[key] = dk if th is None else (1 - self.alpha) * th + self.alpha * dk
        new = ids[self.owner[ids] < 0]
        if len(self.ids[self.fill]) + len(new) > self.cap:
            empty = [i for i in range(self.n_mini) if not self.ids[i]]
            v = empty[0] if empty else min(range(self.n_mini), key=lambda i: self.last_used[i])
            if self.ids[v]:
                old = np.asarray(self.ids[v], dtype=np.int64)
                self.owner[old[self.owner[old] == v]] = -1
            self.ids[v], self.idx[v], self.fill = [], None, v
        if self.idx[self.fill] is None:
            self.idx[self.fill] = self._new_index()
        vec = np.asarray(self.db[new], dtype=np.float32)
        for vid, v in zip(new.tolist(), vec):
            self.idx[self.fill].insert(v, np.uint32(vid + 1))
        self.owner[new] = self.fill
        self.ids[self.fill].extend(new.tolist())
        self.last_used[self.fill] = t
        self.t["insert"] += time.perf_counter() - a

    def after(self, t):
        self._now = t + 1


# -- main ---------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--method", choices=["ours", "lru", "aker", "qvcache"], required=True)
    ap.add_argument("--prep", required=True)
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--state", default=None, help="ours / lru: online_sim --save-state file")
    ap.add_argument("--target-recall", default="0.95", help="DiskANN effort tuned to this recall@10")
    ap.add_argument("--diskann", default=str(RES / "backends" / "spacev10m" / "diskann"))
    ap.add_argument("--n", type=int, default=None,
                    help="requests after the warm-up (default: the rest of the stream, as the headline)")
    ap.add_argument("--sample-every", type=int, default=10,
                    help="time every this-many-th request end to end (live DiskANN); the rest replay "
                         "DiskANN's stored answers")
    ap.add_argument("--ours-recal", action="store_true",
                    help="ours: recalibrate (with audits) as the simulator; default static")
    ap.add_argument("--warm", type=int, default=200, help="untimed requests first (each arm)")
    ap.add_argument("--warmup", type=int, default=20000,
                    help="aker / qvcache: warm-up requests (ours / lru: the state's warm-up)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--cal-delta", type=float, default=0.05, help="ours / lru: calibration delta")
    ap.add_argument("--audit-rate", type=float, default=0.1)
    ap.add_argument("--cal-span", type=int, default=20000)
    ap.add_argument("--cal-every", type=int, default=1000)
    ap.add_argument("--k-lru", type=int, default=20)
    ap.add_argument("--maintenance", choices=["sync", "background"], default="sync",
                    help="the cache's update after a miss is on the request's path (sync; LRU's miss "
                         "is answered by its top-500 search), or background work after a normal top-k "
                         "search (its time reported apart, not in the latency)")
    ap.add_argument("--aker-delta", type=int, default=5, help="Aker reserve neighbors (top_delta)")
    ap.add_argument("--alpha-tighten", type=float, default=0.9)
    ap.add_argument("--alpha-loosen", type=float, default=1.1)
    ap.add_argument("--prox-thresh", type=float, default=None, help="Aker Proximity mode: global threshold")
    ap.add_argument("--prox-frac", type=float, default=None,
                    help="Aker Proximity mode: threshold = this fraction of the median squared NN distance "
                         "(as eval_aker_real.py --prox-frac)")
    ap.add_argument("--qv-D", type=float, default=0.0)
    ap.add_argument("--qv-mini", type=int, default=4)
    ap.add_argument("--qv-dims", type=int, default=16)
    ap.add_argument("--qv-buckets", type=int, default=8)
    ap.add_argument("--no-equal-memory", action="store_true")
    ap.add_argument("--out", required=True)
    from simulations.approximate_hits.timing_env import add_args, make_guard
    add_args(ap)
    args = ap.parse_args()
    guard = make_guard(args)
    guard.start()
    from simulations.approximate_hits.eval_timing import open_diskann
    from simulations.approximate_hits.run_e2e_oracle_trace import load_base, load_base_f32
    prep = Path(args.prep)
    meta = json.loads((prep / "meta.json").read_text())
    db_native = load_base(meta["trace"])                      # what the DiskANN index takes (int8 / f32)
    db = load_base_f32(meta["trace"])                         # float32 view for the caches
    q = np.ascontiguousarray(np.load(prep / "q.npy"), dtype=np.float32)
    stream = np.load(prep / "streams.npz")["trace"]
    top = np.load(prep / "top500.npy", mmap_mode="r")
    bk = json.loads((RES / "backends" / prep.name / "diskann.json").read_text())
    L = int(bk["targets"][args.target_recall]["effort"])
    ans = np.load(RES / "backends" / prep.name / "diskann.npz")[f"answer_{args.target_recall}"]
    st = dict(np.load(args.state, allow_pickle=False)) if args.state else None
    if st is not None:                  # ours / LRU: the requests after the state's warm-up
        args.warmup = int(st["warmup"])
    warm_q = np.asarray(stream[:args.warmup], dtype=np.int64)
    end = len(stream) if args.n is None else min(len(stream), args.warmup + args.n)
    req = np.asarray(stream[args.warmup:end], dtype=np.int64)
    # requests timed end to end with live DiskANN calls: every --sample-every-th, from offset 0;
    # the others are replayed with DiskANN's stored answers (identical to live ones: checked on
    # 300 queries, top-10 at the tuned search list and top-500, 100% equal), so the cache evolves
    # exactly as with live calls, over the whole stream (doc 9m)
    timed_mask = (np.arange(len(req)) % args.sample_every) == 0
    logger.info(f"{args.method} on {prep.name}, budget {args.budget}, DiskANN search list {L} "
                f"(recall@10 {bk['targets'][args.target_recall]['recall_answer_at_k']:.3f}); "
                f"{len(req)} requests, {int(timed_mask.sum())} timed (1 in {args.sample_every})")
    deep = None
    if args.method in ("ours", "lru"):
        if args.method == "ours":
            m = Ours(args, db, q, st)
        else:
            deep = np.load(RES / "backends" / prep.name / "diskann.npz")[f"deep_{args.target_recall}"]
            m = LRU(args, db, q, st, deep)
            m._now = args.warmup
    elif args.method == "aker":
        if args.prox_frac is not None:      # eval_aker_real.py's rule (same rows, exact NN)
            rows = np.random.default_rng(0).choice(len(q), min(5000, len(q)), replace=False)
            nn = np.asarray(db[np.asarray(top[rows, 0], dtype=np.int64)], dtype=np.float32)
            args.prox_thresh = args.prox_frac * float(np.median(((q[rows] - nn) ** 2).sum(axis=1)))
            logger.info(f"Aker Proximity mode: global threshold {args.prox_thresh:.4g}")
        m = AkerReal(args, db, q, warm_q, ans[warm_q])
    else:
        m = QVCacheFast(args, db, q, warm_q, ans[warm_q])
    if deep is None and max(getattr(m, "miss_k", args.k), args.k) > ans.shape[1]:
        deep = np.load(RES / "backends" / prep.name / "diskann.npz")[f"deep_{args.target_recall}"]
    mem0 = m.memory_bytes
    for s_ in getattr(m, "t", {}):            # stage times: per request, over all requests
        m.t[s_] = 0.0
    logger.info(f"cache ready: {mem0 / 1e6:.1f} MB")
    ix = open_diskann(args.diskann, 0)
    xs_native = q.astype(db_native.dtype) if db_native.dtype != np.float32 else q

    def live(qi, kk):
        r = ix.search(xs_native[qi], k_neighbors=kk, complexity=max(L, kk), beam_width=2)
        return np.asarray(r.identifiers if hasattr(r, "identifiers") else r[0], dtype=np.int64)

    def stored(qi, kk):
        # top-kk at search list max(L, kk), as live() would return
        if kk <= ans.shape[1] and kk <= L:
            return np.asarray(ans[qi, :kk], dtype=np.int64)
        return np.asarray(deep[qi, :kk], dtype=np.int64)

    for qi in warm_q[-args.warm:]:                             # warm the OS page cache, untimed
        live(int(qi), args.k)
    rng = np.random.default_rng(0)
    n = len(req)
    hits = np.zeros(n, dtype=bool)
    audited_all = np.zeros(n, dtype=bool)
    ret, ret_b, ret_sim = np.zeros(n), np.zeros(n), np.full(n, np.nan)
    total = []
    maint = 0.0
    trace = []
    t0 = time.time()
    for i, qi in enumerate(req.tolist()):
        t_req = args.warmup + i
        x = q[qi]
        timed = bool(timed_mask[i])
        backend = live if timed else stored
        if hasattr(m, "retention"):      # not timed: the cache's content before this request
            ret_sim[i] = m.retention(ans[qi])
        bg0 = getattr(m, "background", 0.0) + maint
        a = time.perf_counter()
        hit, near = m.lookup(qi, x)
        audited = False
        if hit:
            if m.audits and rng.random() < args.audit_rate:
                m.label(t_req, backend(qi, args.k), 1.0 / args.audit_rate)
                audited = True
            answer = near[:args.k]
        elif args.maintenance == "background":
            # answer from the normal top-k search; the cache's update after it (LRU: the top-500
            # search its new entry needs, the insert, evictions; Aker / QVCache: the insert) is
            # background work, its time reported apart
            answer = backend(qi, args.k)[:args.k]
            b_, r_ = time.perf_counter(), getattr(m, "background", 0.0)
            dl = backend(qi, m.miss_k) if m.miss_k > args.k else answer
            m.on_miss(t_req, qi, x, dl, ix)
            if timed:
                maint += time.perf_counter() - b_ - (getattr(m, "background", 0.0) - r_)
        else:
            ids = backend(qi, m.miss_k)
            m.on_miss(t_req, qi, x, ids, ix)
            answer = ids[:args.k]
        m.after(t_req)
        if timed:
            total.append(time.perf_counter() - a - (getattr(m, "background", 0.0) + maint - bg0))
        hits[i], audited_all[i] = hit, audited
        ret[i] = float(np.isin(answer, top[qi, :args.k]).mean())           # vs exact (not timed)
        ret_b[i] = float(np.isin(answer, ans[qi, :args.k]).mean())         # vs DiskANN's answer
        if (i + 1) % 1000 == 0:      # per-1,000 trace (cutoff, labels, cache size)
            cal = getattr(m, "cal", None)
            trace.append({"t": t_req + 1, "hits": int(hits[i - 999:i + 1].sum()),
                          "lam": None if cal is None or cal.lam is None else float(cal.lam),
                          "labels": None if cal is None else len(cal.lab),
                          "entries": len(getattr(m, "entries", {})) or None,
                          "memory": float(m.memory()) if hasattr(m, "memory") else None})
        if (i + 1) % 10000 == 0:
            logger.info(f"  {i + 1} / {n} requests ({time.time() - t0:.0f} s), hit rate so far "
                        f"{hits[:i + 1].mean():.3f}")
    total = np.array(total)
    th = hits[timed_mask]
    has_sim = not np.all(np.isnan(ret_sim))
    res = {"args": vars(args), "search_list": L, "n": n, "n_timed": int(timed_mask.sum()),
           "hit_rate": float(hits.mean()), "hit_rate_timed": float(th.mean()),
           "audited_share": float(audited_all.mean()),
           "recall_vs_exact_all": float(ret.mean()),
           "recall_vs_exact_hits": float(ret[hits].mean()) if hits.any() else None,
           "retention_vs_backend_hits": float(ret_b[hits].mean()) if hits.any() else None,
           # the simulator's (and the guarantee's) retention: share of DiskANN's answer cached
           "retention_sim_hits": float(np.nanmean(ret_sim[hits])) if has_sim and hits.any() else None,
           "cache_memory_bytes": int(mem0),
           "background_rebuild_s": float(getattr(m, "rebuild_s", 0.0)),
           "background_maintenance_s_timed": float(maint),
           "cache": dict(summary(total), hits_ms=summary(total[th]), misses_ms=summary(total[~th])),
           "stage_ms": {s_: v / n * 1e3 for s_, v in getattr(m, "t", {}).items()},
           "replay_s": time.time() - t0, "trace": trace,
           "knn_fallbacks": int(getattr(m, "fallbacks", 0))}
    logger.info(f"hit rate {res['hit_rate']:.3f} (timed sample {res['hit_rate_timed']:.3f}); recall vs "
                f"exact {res['recall_vs_exact_all']:.3f}; cache + DiskANN {res['cache']['mean_ms']:.2f} ms")
    req_all = req
    req = req_all[timed_mask]          # the DiskANN-alone arms run the timed sample
    # DiskANN alone, and with the cache's memory as its node cache
    from simulations.approximate_hits.eval_timing import time_backend
    xq = np.ascontiguousarray(xs_native[req])
    time_backend(ix, xq[:args.warm], args.k, L)
    lat = time_backend(ix, xq, args.k, L)
    res["backend"] = summary(lat)
    lat_mem = None
    if not args.no_equal_memory:
        node_bytes = db_native.shape[1] * np.dtype(db_native.dtype).itemsize + 4 + 64 * 4
        nodes = int(mem0 // node_bytes)
        del ix
        ix2 = open_diskann(args.diskann, nodes)
        time_backend(ix2, xq[:args.warm], args.k, L)
        lat_mem = time_backend(ix2, xq, args.k, L)
        res["backend_mem"] = dict(summary(lat_mem), nodes_cached=nodes)
    b0 = res["backend"]["mean_ms"]
    res["speedup"] = {"overall": b0 / res["cache"]["mean_ms"],
                      "limit_1_over_1_minus_h": 1 / max(1 - res["hit_rate"] + res["audited_share"], 1e-9),
                      "vs_equal_memory": (res["backend_mem"]["mean_ms"] / res["cache"]["mean_ms"])
                      if "backend_mem" in res else None}
    guard.stop()
    res["environment"] = guard.report()
    logger.info(f"speedup {res['speedup']}; environment {res['environment']['verdict']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(Path(args.out).with_suffix("")) + "_lat.npz", idx=np.flatnonzero(timed_mask),
                        cache=total, hit=th, backend=lat,
                        backend_mem=lat_mem if lat_mem is not None else np.empty(0))
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
