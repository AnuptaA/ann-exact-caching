#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Online simulator (fixed database, fixed k, workload streams).

Inputs from online_prepare.py: exact top-R database ids of every query (the
backend's answers and the ground truth), query streams (random order, an
abrupt workload shift between two groups of query clusters, or a trace).

Cache: entries = (center query, K, its exact top-K ids); one entry per center.
Memory = distinct cached vectors + shell sample (b-bit vectors, b / 32 each).
Shell sample: uncached ids at ranks K+1..R around each entry's center, each
kept iff a fixed per-vector uniform draw is < p.

Serve decision: 7 features against the current cache (relative offset and
size for the 3 best-fitting balls, log1p intruder count), logistic model fit
once at warm-up, serve from cache iff score >= lambda; else miss -> backend.
Retention of a served query = fraction of its true top-k that is cached.

Labels: every miss has its backend answer (free); each served query is
audited (sent to the backend too) with probability a. Calibration data =
the (score, retention) pairs of all misses (weight 1) and audited hits
(weight 1 / a), computed at serve time against the cache then in force,
among the last --cal-span queries since the last reset; lambda = weighted
normal lower bound, fixed order (threshold_selection.select_weighted).

Warm-up (all methods): the first W0 queries all go to the backend (top-R);
split into candidate / scoring / fit / calibration parts; greedy cache of f M
vectors plus a shell sample filling the rest of the budget M; model fit;
lambda on the calibration part.

Methods after warm-up (batches of B queries; cache and lambda change only
between batches):
  static   nothing changes (no recalibration, no tests)
  greedy   the warm-up cache; lambda recalibrated every batch
  lru      every miss becomes an entry with K = K_lru (top-R fetched for the
           shell); least recently used entries are evicted to fit M; lambda
           recalibrated every batch
  aker, qvcache   external baselines (online_baselines.py)

Detection (greedy, lru; --detect, any subset), after every batch, each test
at level delta_alarm / (m j (j + 1)) for the j-th check since the last reset
(m enabled tests; jointly valid over the run):
  score      two-sample KS test, scores of the last --drift-queries queries
             vs the earlier scores since the last reset (label-free)
  retention  Hoeffding upper bound on the last --drift-window audited served
             queries' retention is < T
  hitrate    two-proportion z test, share of scores >= lambda, same samples
             as the score test
An alarm resets calibration (only labels after the alarm are used; nothing
is served until two arrive) and the rebuild window.

Response (--response): recal = the reset only; rebuild (greedy) = also
shadow rebuild checks at --alarm-checks queries after each alarm: build a
proposal on half of the window's labeled queries, compare oracle hits with
the current cache on the other half, install a cache built from all of them
iff >= (1 + g) better; a held-out share of the window calibrates it.

Reference: --oracle-cal labels every query for calibration at no cost.

Backend work is counted separately: top-k searches (misses, audited served
queries), top-R searches (warm-up, rebuild candidates, LRU insertions), and
vectors fetched into the cache or sample. Writes <out>/<method><tag>.json.
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from statistics import NormalDist

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from simulations.approximate_hits.eval_core import quantize
from simulations.approximate_hits.eval_generalize import fit_logistic
from simulations.approximate_hits.greedy_population import GreedyPopulator, hits_from_histograms
from simulations.approximate_hits.threshold_selection import select_weighted, threshold_grid

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ESCI = Path("datasets/esci/data_vectors")
K_VALUES = (10, 20, 50, 100, 200, 500)
N_BALLS = 3
POOR_FIT = 10.0
EPS = 1e-12
CHUNK = 1024

#-------------------------------------------------------------------------------


def sq_dist(a, b, b_sq):
    """Squared Euclidean distances (float32), clipped at 0."""
    d = (a * a).sum(axis=1)[:, None] + b_sq[None, :] - 2.0 * (a @ b.T)
    return np.maximum(d, 0.0)


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
        Ks = np.array([self.entries[c] for c in self.centers], dtype=np.int64)
        kth = self.D["top"][self.centers, Ks - 1]
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
        if len(self.cv) < k:
            # fewer than k cached vectors: nothing can be served
            X[:, -1] = 0.0
            near[:] = 0
            return X, near
        for s in range(0, n, CHUNK):
            q = Q[qidx[s:s + CHUNK]]
            m = len(q)
            d2 = sq_dist(q, self.cv, self.cv_sq)
            part = np.argpartition(d2, k - 1, axis=1)[:, :k]
            near[s:s + m] = self.cached_ids[part]
            nr = np.sqrt(np.take_along_axis(d2, part, axis=1).max(axis=1)) + EPS
            de = np.sqrt(sq_dist(q, self.cq, self.cq_sq))
            fit = (nr[:, None] + de) / self.radius[None, :]
            t = min(N_BALLS, fit.shape[1])
            cand = (np.argpartition(fit, t - 1, axis=1)[:, :t] if t < fit.shape[1]
                    else np.tile(np.arange(t), (m, 1)))
            order = np.take_along_axis(cand, np.argsort(np.take_along_axis(fit, cand, axis=1),
                                                        axis=1), axis=1)
            rows = np.arange(m)
            for j in range(t):
                e = order[:, j]
                X[s:s + m, 2 * j] = de[rows, e] / self.radius[e]
                X[s:s + m, 2 * j + 1] = nr / self.radius[e]
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


#-------------------------------------------------------------------------------


def greedy_entries(D, cand, score, score_w, budget, k, target):
    """Greedy cache (dict center -> K) and its oracle hits on the scoring set.

    cand: list of (center query, largest K whose top-K ids are known); each
    expands to the K_VALUES up to that K.
    """
    top = D["top"]
    pairs = [(c, K) for c, kmax in cand for K in K_VALUES if K <= kmax]
    pop = GreedyPopulator([top[c, :K] for c, K in pairs], top[score, :k], len(D["db"]), k,
                          target, score_weights=np.asarray(score_w, dtype=np.int64))
    while pop.step(int(budget)) is not None:
        pass
    entries = {}
    for j in pop.selected:
        c, K = pairs[j]
        entries[int(c)] = max(entries.get(int(c), 0), K)
    return entries, int(pop.score_hits)


def oracle_hits(cache, score, score_w, k, target):
    counts = cache.cached[cache.D["top"][score, :k]].sum(axis=1)
    hist = np.bincount(counts, weights=score_w, minlength=k + 1).astype(np.int64)
    return int(hits_from_histograms(hist, target, k)[0])


def hits_of_entries(D, entries, score, score_w, k, target):
    """Oracle hits on a scoring set of a cache given as {center: K}."""
    cached = np.zeros(len(D["db"]), dtype=bool)
    for c, K in entries.items():
        cached[D["top"][c, :K]] = True
    counts = cached[D["top"][score, :k]].sum(axis=1)
    hist = np.bincount(counts, weights=score_w, minlength=k + 1).astype(np.int64)
    return int(hits_from_histograms(hist, target, k)[0])


def hoeffding_upper(y, level):
    return float(np.mean(y) + np.sqrt(np.log(1 / level) / (2 * len(y))))


def two_prop_pvalue(x1, n1, x0, n0):
    """Two-sided p-value of the two-proportion z test (1.0 if degenerate)."""
    p = (x1 + x0) / (n1 + n0)
    if p <= 0 or p >= 1:
        return 1.0
    z = (x1 / n1 - x0 / n0) / np.sqrt(p * (1 - p) * (1 / n1 + 1 / n0))
    return float(2 * (1 - NormalDist().cdf(abs(z))))


class Sim:
    def __init__(self, D, stream, args):
        self.D, self.stream, self.a = D, stream, args
        self.k, self.T = args.k, args.target
        self.rng = np.random.default_rng(args.seed)
        self.M = args.budget * len(D["db"])
        self.cache = Cache(D, args.k, args.R, args.bits, self.M)
        self.log = []
        self.cost = {"topk": 0, "topR": 0}
        self.events = []
        self.lam = None
        self.reset_detection()
        self.window = []            # (query, served, audited, audit rate), since the last alarm

    def reset_detection(self):
        """Start calibration and the drift tests afresh (after an alarm or a
        rebuild): only labels arriving from now on are used."""
        self.cal = []               # (t, scores, retentions, weights) of labeled queries
        self.ref = []               # (t, scores) of all queries (score / hit-rate tests)
        self.recent_ret = []        # retentions of audited served queries (retention test)
        self.n_checks = 0

    # -- calibration -------------------------------------------------------

    def calibrate(self, t):
        """lambda from the labeled queries of the last --cal-span queries since
        the last reset (weighted normal bound, fixed order)."""
        lo = t - self.a.cal_span
        self.cal = [c for c in self.cal if c[0] > lo]
        self.ref = [r for r in self.ref if r[0] > lo]
        if not self.cal:
            self.lam = None
            return
        s = np.concatenate([c[1] for c in self.cal])
        y = np.concatenate([c[2] for c in self.cal])
        w = np.concatenate([c[3] for c in self.cal])
        self.lam = select_weighted(s, y, w, threshold_grid(s), self.T, self.a.delta) \
            if len(s) >= 2 else None

    # -- warm-up -----------------------------------------------------------

    def warmup(self):
        a, rng = self.a, self.rng
        w = self.stream[:a.warmup]
        self.cost["topR"] += len(w)
        perm = rng.permutation(w)
        n = len(w) // 5
        cand, score, fit, cal = perm[:n], perm[n:3 * n], perm[3 * n:4 * n], perm[4 * n:]
        entries, _ = greedy_entries(self.D, [(int(c), a.R) for c in cand], score,
                                    np.ones(len(score)), a.f * self.M, self.k, self.T)
        self.cache.replace(entries)
        self.cache.set_p_to_fill()
        X, _ = self.cache.features(fit)
        self.model = fit_logistic(X, self.cache.retention(fit))
        X, _ = self.cache.features(cal)
        s = self.model(X)
        self.cal = [(a.warmup, s, self.cache.retention(cal), np.ones(len(cal)))]
        self.ref = [(a.warmup, s)]
        self.calibrate(a.warmup)
        logger.info(f"warm-up: {len(entries)} entries, {self.cache.n_cached()} cached vectors, "
                    f"p = {self.cache.p:.3f}, memory {self.cache.memory():.0f} / {self.M:.0f}, "
                    f"lambda {self.lam}")

    # -- drift tests -------------------------------------------------------

    def detect(self):
        """Run the enabled tests; each at level delta_alarm / (m j (j + 1)) for
        the j-th check since the last reset (m tests; jointly valid). Returns
        the name of the first test that fires, or None."""
        a = self.a
        tests = a.detect
        if not tests:
            return None
        self.n_checks += 1
        level = a.delta_alarm / (len(tests) * self.n_checks * (self.n_checks + 1))
        if "retention" in tests:
            r = self.recent_ret[-a.drift_window:]
            if len(r) >= 30 and hoeffding_upper(r, level) < self.T:
                return "retention"
        allsc = np.concatenate([r[1] for r in self.ref]) if self.ref else np.empty(0)
        recent, base = allsc[-a.drift_queries:], allsc[:-a.drift_queries]
        if len(recent) < a.drift_queries or len(base) < a.min_ref:
            return None
        if "score" in tests:
            from scipy.stats import ks_2samp
            if ks_2samp(recent, base).pvalue < level:
                return "score"
        if "hitrate" in tests and self.lam is not None:
            if two_prop_pvalue((recent >= self.lam).sum(), len(recent),
                               (base >= self.lam).sum(), len(base)) < level:
                return "hitrate"
        return None

    # -- rebuild (shadow check) --------------------------------------------

    def _greedy(self, qs, wts):
        """Candidates: the current entries (already cached, free) plus new
        queries (top-R must be fetched). Candidates and scoring queries are
        disjoint (a query's own ball would trivially serve it): a random
        --cand-share of the labeled queries proposes candidates, the rest
        score them (offline used 1 : 4)."""
        perm = self.rng.permutation(len(qs))
        n_c = int(self.a.cand_share * len(qs))
        pool, sc = qs[perm[:n_c]], perm[n_c:]
        new = [int(c) for c in self.rng.choice(pool, min(self.a.c_max, len(pool)), replace=False)
               if int(c) not in self.cache.entries]
        self.cost["topR"] += len(new)
        cand = [(c, self.a.R) for c in new] + list(self.cache.entries.items())
        return greedy_entries(self.D, cand, qs[sc], wts[sc], self.a.f * self.M, self.k,
                              self.T)[0]

    def try_rebuild(self, t):
        """Shadow rebuild on the window (queries since the last alarm).

        A random --cal-share of ALL window queries is held out to calibrate a
        new cache (a cache is never calibrated on its own build data; the
        held-out queries' scores also form the reference of the score tests).
        The labeled rest (misses, weight 1; audited hits, weight 1 / audit
        rate; integer units of 1/10) builds: a proposal from a random half is
        compared with the current cache by oracle hits on the other half and
        a cache built from all of it is installed iff >= (1 + g) better."""
        a = self.a
        if len(self.window) < a.min_window:
            return False
        q = np.array([w[0] for w in self.window], dtype=np.int64)
        srv = np.array([w[1] for w in self.window])
        aud = np.array([w[2] for w in self.window])
        rate = np.array([w[3] for w in self.window])
        lab = np.ones(len(q), bool) if a.oracle_cal else (~srv | aud)
        pi = np.where(srv & (not a.oracle_cal), rate, 1.0)
        hold = self.rng.random(len(q)) < a.cal_share
        b = np.flatnonzero(~hold & lab)
        if len(b) > a.s_max:
            b = self.rng.choice(b, a.s_max, replace=False)
        qs, wts = q[b], np.rint(10 / pi[b]).astype(np.int64)
        half = self.rng.random(len(qs)) < 0.5
        proposal = self._greedy(qs[half], wts[half])
        new_hits = hits_of_entries(self.D, proposal, qs[~half], wts[~half], self.k, self.T)
        cur_hits = oracle_hits(self.cache, qs[~half], wts[~half], self.k, self.T)
        if new_hits < (1 + a.gain) * cur_hits:
            self.events.append({"t": t, "event": "shadow_keep", "new": new_hits, "cur": cur_hits})
            return False
        self.cache.replace(self._greedy(qs, wts))
        self.cache.set_p_to_fill()
        hq = q[hold]
        X, _ = self.cache.features(hq)
        s = self.model(X)
        self.reset_detection()
        hl = lab[hold]
        self.cal = [(t, s[hl], self.cache.retention(hq)[hl], 1 / pi[hold][hl])]
        self.ref = [(t, s)]
        self.calibrate(t)
        self.events.append({"t": t, "event": "rebuild", "new": new_hits, "cur": cur_hits,
                            "lambda": self.lam})
        return True

    # -- LRU ---------------------------------------------------------------

    def lru_update(self, qb, serve, near, t, p_lru):
        """Every miss becomes an entry with K = K_lru (its top-R fetched for
        the shell); least recently used entries are evicted to fit M, in
        chunks sized by the average memory per entry (never more than half
        the entries at once, so the cache cannot be emptied by overshoot)."""
        a, cache = self.a, self.cache
        cache.mark_used(near[serve], t)
        miss = [int(q) for q in qb[~serve] if int(q) not in cache.entries]
        self.cost["topR"] += len(miss)
        for q in miss:
            cache.add(q, a.k_lru)
            cache.last_used[q] = t
        cache.p = p_lru
        order = sorted(cache.entries, key=lambda c: cache.last_used.get(c, 0))
        i = 0
        mem = cache.memory()
        while len(cache.entries) > 1 and mem > self.M:
            per = mem / len(cache.entries)
            n_ev = max(1, min(int((mem - self.M) / per), len(cache.entries) // 2))
            for c in order[i:i + n_ev]:
                cache.remove(c)
            i += n_ev
            mem = cache.memory()

    # -- main loop ---------------------------------------------------------

    def run(self, method):
        a, rng, cache = self.a, self.rng, self.cache
        self.warmup()
        p_lru = cache.p
        last_alarm = None
        last_check = a.warmup
        pending = []                # scheduled post-alarm rebuild checks (times)
        for s in range(a.warmup, len(self.stream), a.batch):
            qb = self.stream[s:s + a.batch]
            t = s + len(qb)
            X, near = cache.features(qb)
            score = self.model(X)
            serve = (score >= self.lam if self.lam is not None and cache.n_cached() >= self.k
                     else np.zeros(len(qb), bool))
            y = cache.retention(qb)
            rate = (a.alarm_audit_rate if last_alarm is not None
                    and s - last_alarm < a.alarm_audit_span else a.audit_rate)
            audit = serve & (rng.random(len(qb)) < rate)
            self.cost["topk"] += int((~serve).sum() + audit.sum())
            self.log.append({"t": t, "n": len(qb), "served": int(serve.sum()),
                             "served_retention_sum": float(y[serve].sum()),
                             "ret_hist": np.bincount(np.rint(y * self.k).astype(int),
                                                     minlength=self.k + 1).tolist(),
                             "audited_served": int(audit.sum()),
                             "lambda": self.lam, "memory": cache.memory(),
                             "cached": cache.n_cached(), "entries": len(cache.entries)})
            if method != "static":
                # labels: every miss (free) and audited hits (weight 1 / rate);
                # serve-time (score, retention) pairs, i.e. against the cache
                # in force before this batch's updates
                lab = np.ones(len(qb), bool) if a.oracle_cal else (~serve | audit)
                wt = np.where(serve & (not a.oracle_cal), 1 / rate, 1.0)
                # the window is used only to build caches (greedy rebuilds)
                self.cal.append((t, score[lab], y[lab], wt[lab]))
                self.ref.append((t, score))
                self.recent_ret = (self.recent_ret + list(y[audit]))[-a.drift_window:]
                self.window = (self.window + [(int(q), bool(v), bool(u), rate)
                                              for q, v, u in zip(qb, serve, audit)])[-a.window:]
                if method == "lru":
                    self.lru_update(qb, serve, near, t, p_lru)
                kind = self.detect()
                if kind:
                    self.events.append({"t": t, "event": "alarm", "kind": kind})
                    self.reset_detection()
                    self.window = []
                    last_alarm = last_check = t
                    pending = [t + d for d in a.alarm_checks]
                elif a.response == "rebuild" and method == "greedy" and (
                        (pending and t >= pending[0])
                        or (a.check_every and t - last_check >= a.check_every)):
                    while pending and t >= pending[0]:
                        pending.pop(0)
                    self.try_rebuild(t)
                    last_check = t
                self.calibrate(t)
            if (s - a.warmup) // a.batch % 10 == 0:
                done = self.log[-1]
                logger.info(f"{method} t={t}: served {done['served'] / done['n']:.3f} "
                            f"ret {done['served_retention_sum'] / max(done['served'], 1):.3f} "
                            f"lambda {self.lam} entries {len(cache.entries)} mem {done['memory']:.0f}")
        return {"method": method, "args": vars(a), "log": self.log, "events": self.events,
                "cost": dict(self.cost, vectors_fetched=cache.vectors_fetched,
                             warmup_topR=a.warmup)}


#-------------------------------------------------------------------------------


def load(prep, args):
    meta_path = Path(prep) / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"kind": "esci"}
    if meta["kind"] == "trace":
        from simulations.approximate_hits.run_e2e_oracle_trace import load_base
        db = np.asarray(load_base(meta["trace"]), dtype=np.float32)
        q = np.load(Path(prep) / "q.npy").astype(np.float32)
    else:
        db = np.load(ESCI / f"esci_db{meta.get('n_db', args.n_db)}_s42.npy").astype(np.float32)
        q = np.load(ESCI / "esci_queries.npy").astype(np.float32)
    top = np.load(Path(prep) / f"top{args.R}.npy")
    draw = np.random.default_rng(1).random(len(db))
    streams = np.load(Path(prep) / "streams.npz")
    return {"db": db, "q": q, "top": top, "draw": draw}, streams


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--prep", type=str, default="simulations/approximate_hits/results/online/esci1m")
    ap.add_argument("--method", choices=["static", "greedy", "lru", "aker", "qvcache"],
                    required=True)
    ap.add_argument("--detect", nargs="*", choices=["score", "retention", "hitrate"],
                    default=["score", "retention", "hitrate"],
                    help="drift tests (give the flag with no values for none)")
    ap.add_argument("--response", choices=["recal", "rebuild"], default="rebuild",
                    help="after an alarm: recalibrate only, or also shadow-rebuild "
                         "(greedy only; lru always recalibrates only)")
    ap.add_argument("--oracle-cal", action="store_true",
                    help="reference: every query is labeled for calibration at no cost")
    ap.add_argument("--stream", choices=["random", "shift", "trace"], default="shift")
    ap.add_argument("--n-db", type=int, default=1000000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--budget", type=float, default=0.1, help="memory, fraction of database")
    ap.add_argument("--f", type=float, default=0.85, help="share of budget for the cache")
    ap.add_argument("--R", type=int, default=500)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=20000)
    ap.add_argument("--audit-rate", type=float, default=0.1, help="share of hits audited")
    ap.add_argument("--cal-span", type=int, default=20000,
                    help="calibrate on the labeled queries among the last this many queries")
    ap.add_argument("--cal-share", type=float, default=0.2,
                    help="share of the window held out at a rebuild to calibrate the new cache")
    ap.add_argument("--window", type=int, default=30000,
                    help="recent queries (since the last alarm) used to build a cache")
    ap.add_argument("--cand-share", type=float, default=0.2,
                    help="share of the building queries proposing candidates")
    ap.add_argument("--drift-window", type=int, default=500,
                    help="recent audited served queries in the retention test")
    ap.add_argument("--drift-queries", type=int, default=2000,
                    help="recent queries in the score and hit-rate tests")
    ap.add_argument("--min-ref", type=int, default=1000,
                    help="reference queries needed before the score and hit-rate tests run")
    ap.add_argument("--min-window", type=int, default=5000,
                    help="queries needed in the window before a shadow rebuild check")
    ap.add_argument("--alarm-checks", type=int, nargs="*", default=[5000, 15000, 30000],
                    help="shadow rebuild checks this many queries after an alarm")
    ap.add_argument("--check-every", type=int, default=0,
                    help="also a shadow rebuild check every this many queries (0 = never)")
    ap.add_argument("--gain", type=float, default=0.1)
    ap.add_argument("--delta-alarm", type=float, default=0.05)
    ap.add_argument("--c-max", type=int, default=3000)
    ap.add_argument("--s-max", type=int, default=25000)
    ap.add_argument("--k-lru", type=int, default=50)
    ap.add_argument("--aker-alpha-l", type=float, default=1.1)
    ap.add_argument("--aker-alpha-t", type=float, default=0.25)
    ap.add_argument("--aker-tau-init", type=float, default=0.25)
    ap.add_argument("--aker-cap", type=float, default=1.0)
    ap.add_argument("--qv-mini", type=int, default=4)
    ap.add_argument("--qv-D", type=float, default=0.075, help="QVCache deviation factor")
    ap.add_argument("--qv-alpha", type=float, default=0.9)
    ap.add_argument("--qv-dims", type=int, default=16)
    ap.add_argument("--qv-buckets", type=int, default=8)
    ap.add_argument("--alarm-audit-rate", type=float, default=None,
                    help="audit rate for --alarm-audit-span queries after an alarm")
    ap.add_argument("--alarm-audit-span", type=int, default=10000)
    ap.add_argument("--tag", type=str, default="", help="suffix of the output file name")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-queries", type=int, default=None, help="truncate the stream")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    D, streams = load(args.prep, args)
    stream = streams[args.stream][:args.max_queries]
    out = Path(args.out or Path(args.prep) / f"sim_{args.stream}_b{args.budget:g}")
    out.mkdir(parents=True, exist_ok=True)
    if args.alarm_audit_rate is None:
        args.alarm_audit_rate = args.audit_rate
    if args.method == "lru":
        args.response = "recal"
    if args.method in ("aker", "qvcache"):
        from simulations.approximate_hits import online_baselines as ob
        cls = ob.Aker if args.method == "aker" else ob.QVCache
        res = dict(cls(D, stream, args).run(), method=args.method, args=vars(args))
    else:
        res = Sim(D, stream, args).run(args.method)
    key = f"phase_{args.stream}"
    if key in streams.files or (args.stream == "shift" and "phase" in streams.files):
        phase = streams[key] if key in streams.files else streams["phase"]
        res["phase_start"] = int(np.argmax(phase[:len(stream)] == 1))
    name = f"{args.method}{args.tag}.json"
    (out / name).write_text(json.dumps(res))
    logger.info(f"wrote {out / name}")


if __name__ == "__main__":
    main()
