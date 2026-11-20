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
  ogreedy  continuous greedy (11-14): every batch, greedy on the recent misses
           (fully labeled) adds new entries (any K) and grows existing ones,
           evicting by credit per vector or LRU (see ogreedy_update); lambda
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
from simulations.approximate_hits.threshold_selection import select_weighted, threshold_grid, start_from_fit

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ESCI = Path("datasets/esci/data_vectors")
K_VALUES = (10, 20, 50, 100, 200, 500)
K_LIMIT = [None]        # --max-ball-k: largest ball size greedy may choose (None = all of K_VALUES)
N_BALLS = 3
POOR_FIT = 10.0
EPS = 1e-12
CHUNK = 1024
SCAN = (1000, 2000, 5000, 10000, 20000)   # retention test windows (queries)

#-------------------------------------------------------------------------------


def sq_dist(a, b, b_sq):
    """Squared Euclidean distances (float32), clipped at 0."""
    d = (a * a).sum(axis=1)[:, None] + b_sq[None, :] - 2.0 * (a @ b.T)
    return np.maximum(d, 0.0)


def db_view(D):
    """Fill defaults for a fixed database: ranking = top-R lists, every
    vector present. What the backend returns for each query: D["rank"] =
    its deep list (top-R, asked when building cache entries and shells) and
    D["answer"] = its top-k answer (misses, audits); both exact unless an
    approximate backend is used. D["truth"] = the exact ranking, used only
    to evaluate."""
    if "rank" not in D:
        D["rank"] = D["top"]
    D.setdefault("answer", D["rank"])
    D.setdefault("truth", D["rank"])
    if "present" not in D:
        D["present"] = np.ones(len(D["db"]), dtype=bool)
    D.setdefault("short", [0])
    return D


def approximate_rows(rows, recall, seed=123, chunk=4096):
    """Backend answer lists from exact ranked rows: the index misses each
    database vector independently with probability 1 - recall for each
    query (a fixed draw per query and rank position), and returns the best
    of the vectors it sees, still in order. Recall at any k is then 1 -
    miss probability (a surviving true top-k vector is always inside the
    returned top-k). Rows keep their width; the tail is padded with -1."""
    out = np.empty_like(rows)
    rng = np.random.default_rng(seed)
    for s in range(0, len(rows), chunk):
        r = rows[s:s + chunk]
        drop = (rng.random(r.shape) < 1.0 - recall) | (r < 0)
        order = np.argsort(drop, axis=1, kind="stable")
        o = np.take_along_axis(r, order, axis=1)
        keep = (~drop).sum(axis=1)
        o[np.arange(r.shape[1])[None, :] >= keep[:, None]] = -1
        out[s:s + chunk] = o
    return out


def measured_recall(truth_rows, answer_rows, k, n=20000, seed=0):
    """Mean share of the exact top-k (first k ids of each truth row) found
    in the first k ids of the answer row, on a sample of queries."""
    idx = np.random.default_rng(seed).choice(len(truth_rows), min(n, len(truth_rows)),
                                             replace=False)
    t, a = truth_rows[idx, :k], answer_rows[idx, :k]
    return float(np.mean([len(np.intersect1d(x, y)) / k for x, y in zip(t, a)]))


def topk_present(D, qidx, n, key="rank"):
    """First n present ids of each query's list: the backend's deep list
    (key "rank"), its top-k answer ("answer") or the exact ranking
    ("truth"); with an exact backend
    both are the exact top-n of the current database. Rows with fewer than
    n present ids are padded with their last present id and counted in
    D["short"]."""
    R = D[key][np.asarray(qidx)]
    m = (R >= 0) & D["present"][np.maximum(R, 0)]
    c = np.cumsum(m, axis=1)
    sel = m & (c <= n)
    cnt = sel.sum(axis=1)
    if (cnt == n).all():
        return R[sel].reshape(len(R), n)
    D["short"][0] += int((cnt < n).sum())
    out = np.empty((len(R), n), dtype=R.dtype)
    for i in range(len(R)):
        ids = R[i][sel[i]]
        if not len(ids):
            ids = R[i][:1]
        out[i] = np.r_[ids, np.full(n - len(ids), ids[-1])]
    return out


class Cache:
    """Cache entries plus shell sample; the database may change.

    An entry (center c, K) is a ball: center c, radius = distance to its K-th
    present neighbor when the entry was added. Members = present vectors
    inside the ball, i.e. present ids in the prefix of c's ranking up to that
    neighbor; its shell = present ids after it up to the R-th present
    neighbor at creation. With --maintain (default) inserts inside a ball
    become members and deleted vectors leave, so every ball stays exact;
    without, the member and shell sets are frozen at creation (deleted
    vectors still leave: a cache never returns a deleted vector).
    """

    def __init__(self, D, k, R, bits, budget, maintain=True):
        self.D = db_view(D)
        self.k, self.R, self.bits, self.budget = k, R, bits, budget
        self.maintain = maintain
        self.shell_scope = "all"     # intruder count over the whole shell sample, or "balls"
        self.shell_balls = 3         # how many best-fitting balls (scope "balls")
        self.n_db = len(D["db"])
        self.entries = {}          # center query -> K
        self.geo = {}              # center -> (ball prefix end, shell prefix end, radius)
        self.frozen = {}           # center -> (ball ids, shell ids) without maintenance
        self.p = 0.0
        self.last_used = {}
        self.shallow = set()       # centers of recency-segment entries (no shell; doc 9ar)
        self._fetched = 0
        self._count = np.zeros(self.n_db, dtype=np.int32)
        self._shell = np.empty(0, dtype=np.int64)
        self._members_dirty = True
        self._dirty = True

    # -- membership --------------------------------------------------------

    def add(self, c, K, shallow=False):
        """Add (or resize) an entry; vectors already cached are not refetched. shallow (doc 9ar, recency
        segment): the entry is the miss's ordinary top-k answer only, no deep list, so it has no shell."""
        row = self.D["rank"][c]
        pres = (row >= 0) & self.D["present"][np.maximum(row, 0)]
        cum = np.cumsum(pres)
        pe = int(np.searchsorted(cum, K)) + 1 if cum[-1] >= K else len(row)
        ps = int(np.searchsorted(cum, self.R)) + 1 if cum[-1] >= self.R else len(row)
        ps = pe if shallow else max(ps, pe)
        if shallow:
            self.shallow.add(c)
        else:
            self.shallow.discard(c)
        kth = row[pe - 1]
        rad = float(np.sqrt(((self.D["q"][c] - self.D["db"][kth]) ** 2).sum()))
        self.entries[c] = K
        self.geo[c] = (pe, ps, rad)
        if not self.maintain:
            self.frozen[c] = (row[:pe][pres[:pe]], row[pe:ps][pres[pe:ps]])
        self._members_dirty = self._dirty = True

    def remove(self, c):
        self.entries.pop(c)
        self.geo.pop(c)
        self.frozen.pop(c, None)
        self.last_used.pop(c, None)
        self.shallow.discard(c)
        self._members_dirty = self._dirty = True

    def replace(self, entries):
        """Swap to a new entry set (vectors kept across the swap are not
        refetched)."""
        for c, K in entries.items():
            if self.entries.get(c) != K:
                self.add(c, K)
        for c in [c for c in self.entries if c not in entries]:
            self.remove(c)

    def data_changed(self):
        """The database changed (inserts / deletes): recompute membership."""
        self._members_dirty = self._dirty = True

    def _ball_shell(self, c):
        if self.maintain:
            pe, ps, _ = self.geo[c]
            row = self.D["rank"][c]
            return row[:pe], row[pe:ps]
        return self.frozen[c]

    def _update_members(self):
        if not self._members_dirty:
            return
        present = self.D["present"]
        balls, shells = [], []
        for c in self.entries:
            b, s = self._ball_shell(c)
            balls.append(b)
            shells.append(s)
        old = self._count > 0
        if balls:
            b = np.concatenate(balls)
            b = b[(b >= 0)]
            b = b[present[b]]
            self._count = np.bincount(b, minlength=self.n_db).astype(np.int32)
            s = np.unique(np.concatenate(shells))
            s = s[s >= 0]
            self._shell = s[present[s] & (self._count[s] == 0)]
        else:
            self._count = np.zeros(self.n_db, dtype=np.int32)
            self._shell = np.empty(0, dtype=np.int64)
        self._fetched += int(((self._count > 0) & ~old).sum())
        self._members_dirty = False

    @property
    def vectors_fetched(self):
        self._update_members()
        return self._fetched

    @property
    def ref(self):
        self._update_members()
        return self._count

    @property
    def cached(self):
        return self.ref > 0

    def n_cached(self):
        return int((self.ref > 0).sum())

    def members(self, c):
        b, _ = self._ball_shell(c)
        b = b[b >= 0]
        return b[self.D["present"][b]]

    # -- shell sample ------------------------------------------------------

    def shell_ids(self):
        self._update_members()
        return self._shell

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
        self._update_members()
        if not self._dirty:
            return
        db, Q = self.D["db"], self.D["q"]
        self.cached_ids = np.flatnonzero(self._count > 0)
        self.cv = db[self.cached_ids]
        self.cv_sq = (self.cv * self.cv).sum(axis=1)
        self.centers = np.array(list(self.entries), dtype=np.int64)
        cq = Q[self.centers]
        self.cq, self.cq_sq = cq, (cq * cq).sum(axis=1)
        self.radius = np.array([self.geo[c][2] for c in self.centers], dtype=np.float64)
        sid = self._shell
        self.samp_ids = sid[self.D["draw"][sid] < self.p]
        sv = quantize(db[self.samp_ids].astype(np.float64), self.bits).astype(np.float32) \
            if len(self.samp_ids) else np.zeros((0, db.shape[1]), np.float32)
        self.sv, self.sv_sq = sv, (sv * sv).sum(axis=1)
        # each entry's own sampled shell vectors, as positions in samp_ids
        # (for --shell-scope balls: count intruders only among the shells of
        # the best-fitting balls)
        present, draw = self.D["present"], self.D["draw"]
        self.entry_samp = []
        for c in self.centers:
            _, sh = self._ball_shell(int(c))
            sh = sh[sh >= 0]
            sh = sh[present[sh] & (self._count[sh] == 0) & (draw[sh] < self.p)]
            self.entry_samp.append(np.searchsorted(self.samp_ids, sh))
        self._dirty = False

    def scores(self, qidx, model):
        """Model scores, features and the would-be served ids (CoreCache: exact repeats on the fast path
        score FAST_SCORE, doc 9ak)."""
        X, near = self.features(qidx)
        return model(X), X, near

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
            if not len(self.sv):
                cnt = np.zeros(m)
            elif self.shell_scope == "balls":
                # intruders among the sampled shell vectors of the best-fitting balls only
                cnt = np.zeros(m)
                ns = min(self.shell_balls, fit.shape[1])
                sel = (order[:, :min(ns, t)] if ns <= t
                       else np.argpartition(fit, ns - 1, axis=1)[:, :ns])
                for i in range(m):
                    pos = np.unique(np.concatenate([self.entry_samp[e] for e in sel[i]]))
                    if len(pos):
                        d2s = self.sv_sq[pos] - 2.0 * (self.sv[pos] @ q[i]) + (q[i] @ q[i])
                        cnt[i] = (d2s < nr[i] ** 2).sum()
            else:
                cnt = (sq_dist(q, self.sv, self.sv_sq) < (nr ** 2)[:, None]).sum(axis=1)
            X[s:s + m, -1] = np.log1p(cnt)
        return X, near

    def retention(self, qidx):
        """Share of each query's current top-k, as the backend answers it,
        that is cached (the label we can observe)."""
        return self.cached[topk_present(self.D, qidx, self.k, key="answer")].mean(axis=1)

    def served_retention(self, qidx, near, key="answer"):
        """Share of each query's top-k (the backend's answer, or "truth") contained in the answer the
        cache serves for it (near: its k nearest cached vectors). With an exact backend this equals
        retention() (a cached true top-k vector is among the k nearest cached ones)."""
        A = topk_present(self.D, qidx, self.k, key=key)
        return (np.asarray(near)[:, :, None] == A[:, None, :]).any(axis=2).mean(axis=1)

    def retention_exact(self, qidx):
        """Share of each query's exact top-k that is cached (evaluation
        only; equals retention() with an exact backend)."""
        return self.cached[topk_present(self.D, qidx, self.k, key="truth")].mean(axis=1)

    def mark_used(self, near_ids, t):
        """LRU: entries supplying any of the given vector ids were used at t."""
        if not len(near_ids) or not self.entries:
            return
        hit = np.zeros(self.n_db, dtype=bool)
        hit[near_ids.ravel()] = True
        for c in self.entries:
            if hit[self.members(c)].any():
                self.last_used[c] = t


#-------------------------------------------------------------------------------


def og_build_half(q):
    """ogreedy --og-split: a fixed half of the query ids (multiplicative hash) whose misses score
    candidates; the other half's misses calibrate (never the cache's own build data)."""
    q = np.asarray(q, dtype=np.int64)
    return (((q * 2654435761) >> 7) & 1) == 0


def greedy_entries(D, cand, score, score_w, budget, k, target):
    """Greedy cache (dict center -> K) and its oracle hits on the scoring set.

    cand: list of (center query, largest K whose top-K ids are known); each
    expands to the K_VALUES up to that K. Uses the current database.
    """
    D = db_view(D)
    kvals = tuple(K for K in K_VALUES if K <= (K_LIMIT[0] or max(K_VALUES)))
    centers = np.array([c for c, _ in cand], dtype=np.int64)
    kmax = max([km for _, km in cand] + [1])
    tops = topk_present(D, centers, min(kmax, max(kvals))) if len(centers) else None
    pairs = [(i, c, K) for i, (c, km) in enumerate(cand) for K in kvals if K <= km]
    pop = GreedyPopulator([tops[i, :K] for i, _, K in pairs],
                          topk_present(D, score, k, key="answer"),
                          len(D["db"]), k, target,
                          score_weights=np.asarray(score_w, dtype=np.int64))
    while pop.step(int(budget)) is not None:
        pass
    entries = {}
    for j in pop.selected:
        _, c, K = pairs[j]
        entries[int(c)] = max(entries.get(int(c), 0), K)
    return entries, int(pop.score_hits)


def oracle_hits(cache, score, score_w, k, target):
    counts = cache.cached[topk_present(cache.D, score, k, key="answer")].sum(axis=1)
    hist = np.bincount(counts, weights=score_w, minlength=k + 1).astype(np.int64)
    return int(hits_from_histograms(hist, target, k)[0])


def hits_of_entries(D, entries, score, score_w, k, target):
    """Oracle hits on a scoring set of a cache given as {center: K} (current
    database)."""
    D = db_view(D)
    cached = np.zeros(len(D["db"]), dtype=bool)
    if entries:
        cs = np.array(list(entries), dtype=np.int64)
        tops = topk_present(D, cs, max(entries.values()))
        for i, c in enumerate(cs):
            cached[tops[i, :entries[int(c)]]] = True
    counts = cached[topk_present(D, score, k, key="answer")].sum(axis=1)
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
        if getattr(args, "cache_impl", "python") == "core":     # the C++ core (doc 9ah), .venv-core
            from simulations.approximate_hits.core_cache import CoreCache
            self.cache = CoreCache(D, args.k, args.R, args.bits, self.M,
                                   maintain=not getattr(args, "no_maintain", False), threads=args.core_threads,
                                   ef1=args.core_ef1, ef2=args.core_ef2, ball_mode=args.core_ball_mode,
                                   ball_m=args.core_ball_m, ball_ef=args.core_ball_ef,
                                   ball_scan_below=args.core_ball_scan_below,
                                   repeat_fast=getattr(args, "core_repeat_fast", False),
                                   shell_code=getattr(args, "core_shell_code", "sq"),
                                   insert_ef=getattr(args, "core_insert_ef", 40),
                                   shell_range=getattr(args, "shell_range", "minmax"),
                                   serve_balls=getattr(args, "core_serve_balls", 0))
        else:
            self.cache = Cache(D, args.k, args.R, args.bits, self.M,
                               maintain=not getattr(args, "no_maintain", False))
        self.cache.shell_scope = getattr(args, "shell_scope", "all")
        self.cache.shell_balls = getattr(args, "shell_balls", 3)
        self.log = []
        self.cost = {"topk": 0, "topR": 0, "center_checks": 0, "label_updates": 0}
        self.maintained = getattr(args, "cal_labels", "serve") == "maintained"
        self.cal_stale = False
        self.since_refresh = 0      # batches since maintained labels were refreshed
        ev = D.get("events")
        self.ev = ev                # database events (data-change runs) or None
        self.ev_i = 0
        self.events = []
        self.lam = None
        self.timing = None
        if getattr(args, "timing_compose", None):
            if getattr(args, "cache_impl", "python") != "core":
                raise SystemExit("--timing-compose needs --cache-impl core")
            self.timing = {"compose": np.load(args.timing_compose), "every": args.timing_every,
                           "lat": [], "hit": [], "stages": [], "dt": []}
        self.reset_detection()
        self.window = []            # (query, served, audited, audit rate), since the last alarm

    def reset_detection(self):
        """Start calibration and the drift tests afresh (after an alarm or a
        rebuild): only labels arriving from now on are used."""
        self.cal = []               # (t, scores, retentions, weights, queries) of labeled queries
        self.ref = []               # (t, scores) of all queries (score / hit-rate tests)
        self.audits = []            # (batch end t, retentions, weights) of audited hits
        self.n_checks = 0

    # -- calibration -------------------------------------------------------

    def y(self, qidx, near):
        """Label of each query: --label served (default since 11-12, doc 9o): share of the
        backend's top-k in the answer the cache would serve; cached: share of it that is cached
        (identical with an exact backend)."""
        if getattr(self.a, "label", "served") == "served":
            return self.cache.served_retention(qidx, near)
        return self.cache.retention(qidx)

    def calibrate(self, t):
        """lambda from the labeled queries of the last --cal-span queries since
        the last reset (weighted normal bound, fixed order)."""
        lo = t - self.a.cal_span
        self.cal = [c for c in self.cal if c[0] > lo]
        self.ref = [r for r in self.ref if r[0] > lo]
        self.audits = [r for r in self.audits if r[0] > lo]
        if not self.cal:
            self.lam = None
            return
        self.since_refresh += 1
        every = getattr(self.a, "maintain_every", 1)
        if self.maintained and self.cal_stale and (self.since_refresh >= every
                                                   or self.cal_stale == "now"):
            # maintained labels: each stored query's answer is kept current
            # (updated on inserts / deletes), so its score and retention are
            # recomputed against the current cache and database
            qs = np.concatenate([c[4] for c in self.cal])
            sc, X, nq = self.cache.scores(qs, self.model)
            yy = self.y(qs, nq)
            out, i = [], 0
            for c in self.cal:
                n = len(c[4])
                out.append((c[0], sc[i:i + n], yy[i:i + n], c[3], c[4]))
                i += n
            self.cal = out
            self.cal_stale = False
            self.since_refresh = 0
        s = np.concatenate([c[1] for c in self.cal])
        y = np.concatenate([c[2] for c in self.cal])
        w = np.concatenate([c[3] for c in self.cal])
        self.lam = select_weighted(s, y, w, threshold_grid(s), self.T, self.a.delta,
                                   start=getattr(self, "lam_start", None)) \
            if len(s) >= 2 else None

    # -- warm-up -----------------------------------------------------------

    def warmup(self):
        a, rng = self.a, self.rng
        w = self.stream[:a.warmup]
        self.cost["topR"] += len(w)
        perm = rng.permutation(w)
        if getattr(a, "oracle_score", None) is not None:
            # ORACLE probe: greedy scores on stream[START:END] (requests that come later, i.e. it
            # sees the future); candidates = 60% of the warm-up, 20% fit the model, 20% calibrate
            n = len(w) // 5
            cand, fit, cal = perm[:3 * n], perm[3 * n:4 * n], perm[4 * n:]
            score = self.stream[a.oracle_score[0]:a.oracle_score[1]]
        elif getattr(a, "warm_split", None) is None:
            n = len(w) // 5
            cand, score, fit, cal = perm[:n], perm[n:3 * n], perm[3 * n:4 * n], perm[4 * n:]
        else:       # --warm-split: shares of the warm-up for candidates / scoring / model fit / calibration
            cut = [0] + [int(round(x * len(w))) for x in np.cumsum(a.warm_split)]
            cut[-1] = len(w)
            cand, score, fit, cal = (perm[cut[i]:cut[i + 1]] for i in range(4))
        entries, score_hits = greedy_entries(self.D, [(int(c), a.R) for c in cand], score,
                                             np.ones(len(score)), a.f * self.M, self.k, self.T)
        self.cache.replace(entries)
        self.cache.set_p_to_fill()
        X, nf = self.cache.features(fit)
        y_fit = self.y(fit, nf)
        self.model = fit_logistic(X, y_fit)
        # starting threshold of the calibration walk, from the fit split (doc 9p)
        self.fit_q = fit
        self.lam_start = (start_from_fit(self.model(X), y_fit, self.T, a.delta)
                          if getattr(a, "start", "fit") == "fit" else None)
        s, X, nc = self.cache.scores(cal, self.model)
        self.cal = [(a.warmup, s, self.y(cal, nc), np.ones(len(cal)), cal)]
        self.ref = [(a.warmup, s)]
        self.calibrate(a.warmup)
        # in-sample check (11-14): the warm-up cache on the scoring requests greedy optimized for:
        # greedy's oracle hits (any retention-T admission), the cache's oracle hits with the shell
        # sample, and the calibrated policy (model score >= lambda) with its mean retention
        ss, Xs, ns = self.cache.scores(score, self.model)
        ys = self.y(score, ns)
        srv = ss >= self.lam if self.lam is not None else np.zeros(len(score), bool)
        self.insample = {"n_score": int(len(score)), "n_cand": int(len(cand)),
                         "distinct_cand": int(len(np.unique(cand))),
                         "greedy_oracle_hit_rate": float(score_hits / max(len(score), 1)),
                         "cache_oracle_hit_rate": float(oracle_hits(self.cache, score, np.ones(len(score)),
                                                                    self.k, self.T) / max(len(score), 1)),
                         "calibrated_hit_rate": float(srv.mean()),
                         "calibrated_retention": float(ys[srv].mean()) if srv.any() else None,
                         "entries": int(len(entries)), "cached_vectors": int(self.cache.n_cached())}
        logger.info(f"in-sample (scoring split): {self.insample}")
        logger.info(f"warm-up: {len(entries)} entries, {self.cache.n_cached()} cached vectors, "
                    f"p = {self.cache.p:.3f}, memory {self.cache.memory():.0f} / {self.M:.0f}, "
                    f"lambda {self.lam}")

    # -- warm-up product (shared by timing runs of the same cell, doc 9ad) -------

    WARM_KEYS = ("budget", "seed", "warmup", "stream", "target", "delta", "k", "f", "R", "bits", "warm_split",
                 "oracle_score", "backend_file", "backend_target", "shell_scope", "shell_balls", "max_ball_k",
                 "start", "label", "audit_rate", "cal_span")

    def warm_key(self):
        return {k: getattr(self.a, k, None) for k in self.WARM_KEYS}

    def warm_product(self):
        """Everything warmup() produced, picklable (the scorer as its coefficients)."""
        return dict(key=self.warm_key(), entries=dict(self.cache.entries), params=self.model.params, lam=self.lam,
                    lam_start=getattr(self, "lam_start", None), fit_q=self.fit_q, cal=self.cal, ref=self.ref,
                    insample=self.insample, rng=self.rng.bit_generator.state, cost=dict(self.cost))

    def restore_warm(self, prod):
        """Install a warm_product() instead of running warmup(): the same cache, model, cutoff,
        calibration set and random state, without the greedy selection (minutes at 10%)."""
        if prod["key"] != self.warm_key():
            raise ValueError("warm-up product made with other options")
        self.cache.replace(prod["entries"])
        self.cache.set_p_to_fill()
        mp = prod["params"]
        self.model = (lambda A, mp=mp: ((A - mp["mu"]) / mp["sd"]) @ mp["coef"] + mp["intercept"])
        self.model.params = mp
        self.lam, self.lam_start, self.fit_q = prod["lam"], prod["lam_start"], prod["fit_q"]
        self.cal, self.ref, self.insample = prod["cal"], prod["ref"], prod["insample"]
        self.rng.bit_generator.state = prod["rng"]
        self.cost = prod["cost"]

    # -- drift tests -------------------------------------------------------

    def retention_alarm(self, t, level):
        """Audited hits of the last w queries, for each w in SCAN (level split
        over them): normal upper bound on their (weighted) mean retention
        < T. Weights 1 / audit rate; effective n = (sum w)^2 / sum w^2."""
        if not self.audits:
            return False
        ts = np.concatenate([np.full(len(x[1]), x[0]) for x in self.audits])
        ys = np.concatenate([x[1] for x in self.audits])
        ws = np.concatenate([x[2] for x in self.audits])
        z = NormalDist().inv_cdf(1 - level / len(SCAN))
        for w in SCAN:
            m = ts > t - w
            if m.sum() < 30:
                continue
            W = ws[m].sum()
            mu = (ws[m] * ys[m]).sum() / W
            ne = W ** 2 / (ws[m] ** 2).sum()
            sd = np.sqrt((ws[m] * (ys[m] - mu) ** 2).sum() / W * ne / max(ne - 1, 1))
            if mu + z * sd / np.sqrt(ne) < self.T:
                return True
        return False

    def detect(self, t):
        """Run the enabled tests; each at level delta_alarm / (m j (j + 1)) for
        the j-th check since the last reset (m tests; jointly valid), or at
        a fixed per-check level (--alarm-level; false alarms accumulate).
        Returns the name of the first test that fires, or None."""
        a = self.a
        tests = a.detect
        if not tests:
            return None
        self.n_checks += 1
        fixed = getattr(a, "alarm_level", None)
        level = (fixed / len(tests) if fixed else
                 a.delta_alarm / (len(tests) * self.n_checks * (self.n_checks + 1)))
        if "retention" in tests and self.retention_alarm(t, level):
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
        s, X, nh = self.cache.scores(hq, self.model)
        self.reset_detection()
        hl = lab[hold]
        self.last_hold = (hq, hl, 1 / pi[hold])   # for callers that re-score them (timing harness)
        self.cal = [(t, s[hl], self.y(hq, nh)[hl], 1 / pi[hold][hl], hq[hl])]
        self.ref = [(t, s)]
        self.calibrate(t)
        self.events.append({"t": t, "event": "rebuild", "new": new_hits, "cur": cur_hits,
                            "lambda": self.lam})
        self.cal_stale = "now"
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
        p_adm = getattr(a, "lru_admit_p", 1.0)
        if p_adm < 1.0:            # doc 9aq: admit (and deep-search) only a share of the misses
            if not hasattr(self, "lru_rng"):
                self.lru_rng = np.random.default_rng(a.seed + 11)
            miss = [q for q in miss if self.lru_rng.random() < p_adm]
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

    # -- continuous greedy (ogreedy, 11-14) --------------------------------

    def ogreedy_update(self, qb, serve, near, t):
        """Continuous greedy, then (--og-recent-share > 0) the recency segment."""
        self._ogreedy_step(qb, serve, near, t)
        if getattr(self.a, "og_recent_share", 0.0) > 0:
            self._recent_step(qb, serve, t)

    def _recent_step(self, qb, serve, t):
        """Recency segment (doc 9ar): every miss not already an entry becomes a shallow entry, its ordinary top-k
        answer as a K = k ball with no shell. No deep search: the backend read these k vectors for the miss
        anyway (DiskANN reranks with full vectors). Least recently used shallow entries leave when the segment
        holds more than --og-recent-share x M / k entries, or while the cache is over M. Continuous greedy
        neither evicts nor grows shallow entries (growing one would need its deep list)."""
        a, cache, k = self.a, self.cache, self.k
        miss = [int(q) for q in np.unique(qb[~serve]) if int(q) not in cache.entries]
        for q in miss:
            cache.add(q, k, shallow=True)
            cache.last_used[q] = t
        self.cost["recent_adds"] = self.cost.get("recent_adds", 0) + len(miss)
        cap = int(a.og_recent_share * self.M / k)
        order = sorted(cache.shallow, key=lambda c: cache.last_used.get(c, 0))
        n_ev, i = len(order) - cap, 0
        for c in order[:max(n_ev, 0)]:
            cache.remove(c)
        i = max(n_ev, 0)
        mem = cache.memory()
        while mem > self.M and i < len(order):
            ch = order[i:i + 64]
            for c in ch:
                cache.remove(c)
            i += len(ch)
            mem = cache.memory()
        self.cost["recent_evicts"] = self.cost.get("recent_evicts", 0) + i

    def _shallow_only_vectors(self):
        """Cached vectors held by shallow (recency-segment) entries and by no other entry."""
        cache = self.cache
        if not cache.shallow:
            return 0
        sh = np.unique(np.concatenate([cache.members(c) for c in cache.shallow]))
        deep = [cache.members(c) for c in cache.entries if c not in cache.shallow]
        if deep:
            sh = sh[~np.isin(sh, np.concatenate(deep))]
        return int(len(sh))

    def _ogreedy_step(self, qb, serve, near, t):
        """Continuous utility-driven maintenance (doc 9ac). After each batch:
          scoring set   the misses of the last --og-span requests BEFORE this batch (fully
                        labeled: the backend answered them; no audited-hit up-weighting), with
                        multiplicities; this batch's misses are the candidates, so scoring them on
                        themselves would make every miss serve itself (a first version did: ~90% of
                        the entries replaced every batch);
          candidates    this batch's misses as new entries at every K of K_VALUES, and every
                        current entry grown to each larger K;
          credits       per entry: served answer vectors it supplied over the last --og-span
                        requests / k (hits attributable to it);
          exchange      greedy picks the candidate with the most marginal scoring hits per new
                        vector (the offline greedy's rule, starting from the current cache);
                        when it does not fit, victims are evicted, in order of credit per vector
                        (--og-evict utility; only while the victim's credit per vector is below the
                        candidate's gain per vector) or least recently used first (--og-evict lru;
                        whenever room is needed). Entries added this batch are not victims.
        The populator still counts victims' vectors as cached within the batch (a slightly
        optimistic gain for candidates sharing them); the next batch scores the real cache."""
        a, cache, k = self.a, self.cache, self.k
        span = a.og_span or a.cal_span
        if not getattr(self, "og_seeded", False):
            # seed both windows with the warm-up requests (all labeled), served by the warm-up cache
            # and cutoff: otherwise the first batches would see ~1,000 requests of credit and evict
            # most warm-up entries as "unused" (a first version evicted 582 of 826 in batch one)
            self.og_seeded = True
            w = np.asarray(self.stream[:a.warmup], dtype=np.int64)
            sw0, Xw, nw = cache.scores(w, self.model)
            sw_ = (sw0 >= self.lam) if self.lam is not None else np.zeros(len(w), bool)
            hv0 = np.bincount(np.asarray(nw[sw_]).ravel(), minlength=cache.n_db) if sw_.any() else None
            cred0 = {}
            if hv0 is not None:
                for c in cache.entries:
                    v = hv0[cache.members(c)].sum()
                    if v:
                        cred0[c] = v / k
            self.og_credit = [(a.warmup, cred0)]
            wm = w[~sw_]
            self.og_miss = [(a.warmup, wm[og_build_half(wm)] if a.og_split else wm)]
        own = serve & ~getattr(self, "_tmask", np.zeros(len(serve), bool))[:len(serve)] \
            if len(getattr(self, "_tmask", ())) == len(serve) else serve
        cache.mark_used(near[own], t)
        # credits of this batch
        hv = np.bincount(np.asarray(near[own]).ravel(), minlength=cache.n_db) if own.any() else None
        cred = {}
        if hv is not None:
            for c in cache.entries:
                v = hv[cache.members(c)].sum()
                if v:
                    cred[c] = v / k
        self.og_credit.append((t, cred))
        self.og_credit = [x for x in self.og_credit if x[0] > t - span]
        miss = qb[~serve]
        if a.og_split:
            miss = miss[og_build_half(miss)]
        self.og_miss = [x for x in self.og_miss if x[0] > t - span]
        past = [x[1] for x in self.og_miss]        # misses of earlier batches only
        self.og_miss.append((t, miss))
        new = [int(q) for q in np.unique(miss) if int(q) not in cache.entries]
        # doc 9aq: which misses get the deep top-R search (candidates). The other misses still score candidates
        # (their answers are labeled); only their deep lists are not fetched.
        n_miss_new = len(new)
        trig_m = getattr(a, "og_trigger_m", 0)
        if trig_m > 0 and new:
            # cluster trigger: a miss is a candidate if >= trig_m ids of its top-k answer also occur in the answer of another
            # distinct miss within the last --og-trigger-window requests (this batch included)
            D0 = db_view(self.D)
            recent = np.unique(np.concatenate([x[1] for x in self.og_miss if x[0] > t - a.og_trigger_window]))
            cnt = np.bincount(topk_present(D0, recent, k, key="answer").ravel(), minlength=cache.n_db)
            shared = (cnt[topk_present(D0, np.array(new, dtype=np.int64), k, key="answer")] >= 2).sum(axis=1)
            new = [q for q, sh in zip(new, shared) if sh >= trig_m]
        smp = getattr(a, "og_sample", 1.0)
        if smp < 1.0 and new:
            if not hasattr(self, "og_rng"):
                self.og_rng = np.random.default_rng(a.seed + 7)
            new = [q for q in new if self.og_rng.random() < smp]
        self.cost["topR"] += len(new)
        self.cost["new_misses"] = self.cost.get("new_misses", 0) + n_miss_new
        if not past:
            return
        sq, sw = np.unique(np.concatenate(past), return_counts=True)
        kvals = tuple(K for K in K_VALUES if K <= (K_LIMIT[0] or max(K_VALUES)))
        pairs = [(c, K) for c in new for K in kvals]
        pairs += [(c, K) for c, K0 in cache.entries.items() for K in kvals if K > K0 and c not in cache.shallow]
        if not pairs or not len(sq):
            return
        D = db_view(self.D)
        cs = np.array(sorted({c for c, _ in pairs}), dtype=np.int64)
        tops = topk_present(D, cs, max(kvals))
        row = {int(c): i for i, c in enumerate(cs)}
        pop = GreedyPopulator([tops[row[c], :K] for c, K in pairs], topk_present(D, sq, k, key="answer"),
                              cache.n_db, k, self.T, score_weights=sw.astype(np.int64))
        cache.prepare()
        pop._cache_vectors(np.asarray(cache.cached_ids, dtype=np.int64))
        # vector budget for cached vectors: what is left of M after the shell sample, at the current
        # ratio of memory to cached vectors
        n_c = max(cache.n_cached(), 1)
        per = cache.memory() / n_c
        # doc 9ar: with a recency segment, greedy keeps to (1 - share) x M; vectors held only by shallow entries
        # are the segment's (greedy's cap counts every cached vector)
        share = getattr(a, "og_recent_share", 0.0)
        sh_only = self._shallow_only_vectors() if share > 0 else 0
        # doc 9as: the mini cache's memory is outside the main cache
        limit = self.M * (1 - share - getattr(a, "mini_cache_share", 0.0)) + sh_only * per
        cap = int(limit / per)
        credit = {}
        for _, cr in self.og_credit:
            for c, v in cr.items():
                credit[c] = credit.get(c, 0.0) + v
        cache._update_members()      # victim order by credit per vector held only by that entry
        size = {c: max(int((cache._count[cache.members(c)] == 1).sum()), 1) for c in cache.entries}
        if a.og_evict == "lru":
            order = sorted(cache.entries, key=lambda c: cache.last_used.get(c, 0))
        else:
            order = sorted(cache.entries, key=lambda c: credit.get(c, 0.0) / size[c])
        order = [c for c in order if c not in cache.shallow]       # the recency segment evicts its own
        # candidates' gains are ORACLE hits on past misses (perfect hit decisions); incumbents'
        # credits are hits actually served under the calibrated cutoff. Put both in served hits:
        # scale gains by rho = served / oracle hits over the same span, from this run's own log (the
        # warm-up's in-sample rates as a prior worth its scoring split); without it candidates
        # always outrank incumbents (TripClick 10%: ~1,400 changes per batch, retention 0.892 < T)
        from simulations.approximate_hits.greedy_population import hits_from_histograms
        # online information only (11-15): served hits are known; the oracle hit rate is
        # estimated from the LABELED requests (misses, weight 1; audited hits, weight 1 / audit
        # rate), never from unaudited hits' retention (a first version used every request's
        # retention from the log: unavailable to a real cache)
        recent = [b for b in self.log if b["t"] > t - span]
        labs = [c for c in self.cal if c[0] > t - span]
        ins = getattr(self, "insample", None) or {}
        n0 = ins.get("n_score", 0)
        n_req = sum(b["n"] for b in recent)
        srv = sum(b["served"] for b in recent) + n0 * ins.get("calibrated_hit_rate", 0.0)
        orc = n0 * ins.get("cache_oracle_hit_rate", 0.0)
        if labs and n_req:
            yy = np.concatenate([c[2] for c in labs])
            ww = np.concatenate([c[3] for c in labs])
            hist = np.bincount(np.rint(yy * k).astype(int), weights=ww, minlength=k + 1)
            hi = np.rint(hist * 1000.0 / max(ww.sum(), 1e-9)).astype(np.int64)   # ~1,000 units
            orc += float(hits_from_histograms(hi, self.T, k)[0]) / max(hi.sum(), 1) * n_req
        rho = float(np.clip(srv / orc, 0.0, 1.0)) if orc > 0 else 1.0
        # --og-kcal (11-15, v2): temporal cross-check per entry size K, in marginal hits on both
        # sides: the span's misses split into an earlier and a later half; the same candidates
        # (this batch's pairs) scored against each half over the current cache; among the
        # candidates of size K that greedy would favor on the earlier half (top fifth by gain per
        # vector), phi_K = their gains on the later half / on the earlier half (clipped to [0, 1];
        # 1 without data). Candidates of size K are ranked by phi_K x gain. Recomputed every 5
        # batches. (v1 compared attributed credits with marginal predictions: shared vectors
        # inflated credits, phi stuck at 1.) At 10% greedy otherwise prefers top-200 / 500 entries
        # whose past coverage does not recur (a hand cap at K 100: SPACEV 0.6 31.2 -> 42.2).
        phi = getattr(self, "og_phi", {}) if a.og_kcal else {}
        if a.og_kcal and (not hasattr(self, "og_phi") or (t - self.og_phi_t) >= 5 * a.batch):
            t_mid = t - span / 2
            pe = [x[1] for x in self.og_miss[:-1] if x[0] <= t_mid]
            pl = [x[1] for x in self.og_miss[:-1] if x[0] > t_mid]
            phi = {}
            if pe and pl:
                gains = []
                for part in (pe, pl):
                    pq, pw = np.unique(np.concatenate(part), return_counts=True)
                    pp = GreedyPopulator([tops[row[c], :K] for c, K in pairs], topk_present(D, pq, k, key="answer"),
                                         cache.n_db, k, self.T, score_weights=pw.astype(np.int64))
                    pp._cache_vectors(np.asarray(cache.cached_ids, dtype=np.int64))
                    dh, _, _ = pp._evaluate_candidates(pp.new_vectors > 0)
                    gains.append((dh, pp.new_vectors.copy()))
                (ge, nv), (gl, _) = gains
                kk = np.array([K for _, K in pairs])
                for K in kvals:
                    m = np.flatnonzero((kk == K) & (nv > 0))
                    if len(m) < 20:
                        continue
                    r = ge[m] / nv[m]
                    top = m[np.argsort(-r)[:max(10, len(m) // 5)]]
                    if ge[top].sum() > 0:
                        phi[K] = float(np.clip(gl[top].sum() / ge[top].sum(), 0.0, 1.0))
            self.og_phi, self.og_phi_t = phi, t
            self.og_phi_log.append((t, dict(phi)))
        kof = np.array([K for _, K in pairs])
        phiv = np.array([phi.get(int(K), 1.0) for K in kof])
        # vectors an eviction actually frees: those no other remaining entry holds
        cache._update_members()
        cnt = cache._count.copy()
        vi, freed, victims, chosen = 0, 0, [], []
        while True:
            alive = (pop.new_vectors > 0)
            if not alive.any():
                break
            d_hits, _, _ = pop._evaluate_candidates(alive)
            ratio = np.where(alive, rho * phiv * d_hits / np.maximum(pop.new_vectors, 1), -1.0)
            j = int(np.argmax(ratio))
            if ratio[j] <= 0:
                break
            need = pop.cache_size + int(pop.new_vectors[j]) - (cap + freed)
            while need > 0 and vi < len(order):
                v = order[vi]
                if v in chosen:
                    vi += 1
                    continue
                mv = cache.members(v)
                own = int((cnt[mv] == 1).sum())       # freed if v goes (shared ones stay cached)
                if own == 0:
                    vi += 1
                    continue
                if a.og_evict == "utility" and credit.get(v, 0.0) / own >= ratio[j]:
                    break
                cnt[mv] -= 1
                victims.append(v)
                freed += own
                need -= own
                vi += 1
            if need > 0:
                break
            if a.og_kcal:
                # commit j itself (the populator's own step ranks without phi)
                vecs = pop.cand_v[pop.cand_ptr[j]:pop.cand_ptr[j + 1]]
                pop._cache_vectors(vecs[~pop.cached[vecs]])
                pop.selected.append(j)
                chosen.append(pairs[j][0])
                continue
            res = pop.step(cap + freed)
            if res is None:
                break
            chosen.append(pairs[res["candidate"]][0])
        if a.og_fresh:
            # fresh admission (11-15): a new miss has no evidence of its own; its expected value is
            # measured in hindsight on this run: earlier misses of the window, as if admitted at
            # each K, against the later misses (gain over the current cache per new vector, mean
            # over the earlier misses). New misses not chosen above are admitted at the best K
            # while that value beats the next victim's credit per vector.
            t_mid = t - span / 2
            pe = [x[1] for x in self.og_miss[:-1] if x[0] <= t_mid]
            pl = [x[1] for x in self.og_miss[:-1] if x[0] > t_mid]
            if pe and pl:
                early = np.unique(np.concatenate(pe))
                lq, lw = np.unique(np.concatenate(pl), return_counts=True)
                early = early[~np.isin(early, lq)][:4000]        # no self-service
                if len(early) and len(lq):
                    te = topk_present(D, early, max(kvals))
                    fp = [(i, K) for i in range(len(early)) for K in kvals]
                    pop2 = GreedyPopulator([te[i, :K] for i, K in fp], topk_present(D, lq, k, key="answer"),
                                           cache.n_db, k, self.T, score_weights=lw.astype(np.int64))
                    pop2._cache_vectors(np.asarray(cache.cached_ids, dtype=np.int64))
                    dh, _, _ = pop2._evaluate_candidates(pop2.new_vectors > 0)
                    best_K, best_v = None, 0.0
                    for K in kvals:
                        m = np.array([kk == K for _, kk in fp])
                        v = rho * dh[m].sum() / max(pop2.new_vectors[m].sum(), 1)
                        if v > best_v:
                            best_K, best_v = K, v
                    # the later part spans (t - t_mid) requests; credits span `span`: same units
                    best_v *= span / max(t - t_mid, 1)
                    self.og_fresh_log.append((t, best_K, best_v))
                    if best_K is not None:
                        for c in [c for c in new if c not in chosen]:
                            add_v = int((~pop.cached[tops[row[c], :best_K]]).sum())
                            need = pop.cache_size + add_v - (cap + freed)
                            while need > 0 and vi < len(order):
                                v = order[vi]
                                if v in chosen:
                                    vi += 1
                                    continue
                                mv = cache.members(v)
                                own = int((cnt[mv] == 1).sum())
                                if own == 0:
                                    vi += 1
                                    continue
                                if credit.get(v, 0.0) / own >= best_v:
                                    break
                                cnt[mv] -= 1
                                victims.append(v)
                                freed += own
                                need -= own
                                vi += 1
                            if need > 0:
                                break
                            j = pairs.index((c, best_K))
                            vecs = pop.cand_v[pop.cand_ptr[j]:pop.cand_ptr[j + 1]]
                            pop._cache_vectors(vecs[~pop.cached[vecs]])
                            pop.selected.append(j)
                            chosen.append(c)
        for v in victims:
            if v in cache.entries:
                cache.remove(v)
        for jj in pop.selected:
            c, K = pairs[jj]
            if cache.entries.get(c, 0) < K:
                cache.add(c, K)
                cache.last_used[c] = t
        mem = cache.memory()
        if mem > limit:             # shell growth: trim old entries by the victim order, then the
            # latest additions (lowest gain per vector first, i.e. last chosen first)
            for v in [v for v in order if v not in chosen] + chosen[::-1]:
                if mem <= limit or len(cache.entries) <= 1:
                    break
                if v in cache.entries:
                    cache.remove(v)
                    mem = cache.memory()
        self.og_log.append({"t": t, "rho": rho, "added": len(pop.selected), "evicted": len(victims),
                            "entries": len(cache.entries), "cached": cache.n_cached(), "shallow": len(cache.shallow)})

    # -- database events ---------------------------------------------------

    def apply_events(self, s):
        """Apply inserts / deletes scheduled before stream position s. Cache
        upkeep cost: one distance per (insert, entry center); maintained
        labels: one per (insert, stored calibration query)."""
        ev = self.ev
        if ev is None:
            return 0
        j = int(np.searchsorted(ev["t"], s, side="right"))
        if j <= self.ev_i:
            return 0
        sl = slice(self.ev_i, j)
        ins = ev["insert"][sl]
        ids = ev["id"][sl]
        pres = self.D["present"]
        pres[ids[ins]] = True
        pres[ids[~ins]] = False
        self.ev_i = j
        n_ins = int(ins.sum())
        self.cost["center_checks"] += n_ins * len(self.cache.entries)
        if self.maintained:
            self.cost["label_updates"] += n_ins * sum(len(c[4]) for c in self.cal)
        self.cache.data_changed()
        if self.a.method != "lru" and self.cache.memory() > self.M:
            self.cache.set_p_to_fill()
        self.cal_stale = True
        return j - sl.start

    # -- main loop ---------------------------------------------------------

    # -- pool statistics (doc 9ai): how much of a hit's answer one cache entry could give ------------------
    def _pool_acc(self, qs, serve, near):
        """For each served request (the cache in force when it was served): distinct entries holding a served
        vector, the size of a greedy set cover of the served vectors by entries, the most served vectors one
        entry holds, and the best retention (vs the backend's top-k) any single entry's ball gives."""
        if not serve.any():
            return
        cache, k = self.cache, self.k
        holders = getattr(cache, "_holders", None)
        if holders is None:                       # online_sim.Cache: build vector -> entries for this call
            holders = {}
            for c in cache.entries:
                for v in cache.members(c).tolist():
                    holders.setdefault(v, set()).add(c)
        P = self._pool
        for q, nn in zip(np.asarray(qs)[serve], np.asarray(near)[serve]):
            nn = [int(v) for v in nn[:k]]
            hold = {}
            for v in nn:
                for c in holders.get(v, ()):
                    hold.setdefault(c, set()).add(v)
            left, cover = set(nn), 0
            while left and hold:
                c = max(hold, key=lambda e: len(hold[e] & left))
                if not hold[c] & left:
                    break
                left -= hold[c]
                cover += 1
            ans = set(topk_present(self.D, [int(q)], k, key="answer")[0].tolist())
            best_ret = max((len(set(cache.members(c).tolist()) & ans) for c in hold), default=0) / k
            P["hits"] += 1
            P["holders"] += len(hold)
            P["cover"] += cover
            P["best_cov"] += max((len(x) for x in hold.values()), default=0) / k
            P["single_ret"] += best_ret

    # -- timing (doc 9ah): the request path through the core, one request at a time ------------
    def _timed_lookup(self, q, t):
        """One request: the core's lookup (stage 1, decision, stage 2 on a hit) timed; a miss is charged that
        request's measured DiskANN time (composed, --timing-compose); the would-be served answer of a miss
        (its label) is computed untimed. Every --timing-every-th request after the warm-up is recorded."""
        import time as _t
        cache, T = self.cache, self.timing
        cache.sync(self.model)
        x = np.ascontiguousarray(np.asarray(self.D["q"][q], np.float32))
        lam = self.lam if (self.lam is not None and cache.n_cached() >= self.k) else float("inf")
        a0 = _t.perf_counter()
        r = cache.core.lookup(x, float(lam), int(cache.ef1), int(cache.ef2))
        dt = _t.perf_counter() - a0
        near = np.asarray(r.near, dtype=np.int64)
        if not r.hit and (cache.ef2 or getattr(cache, "serve_balls", 0)):
            near = np.asarray(cache.core.serve(x, int(cache.ef2)), dtype=np.int64)
        if len(near) < self.k:
            near = np.concatenate([near, np.zeros(self.k - len(near), np.int64)])
        pos = t - self.a.warmup
        if pos % T["every"] == 0:
            j = pos // T["every"]
            charge = 0.0 if r.hit else float(T["compose"][j]) if j < len(T["compose"]) else np.nan
            T["lat"].append(dt + charge)
            T["hit"].append(bool(r.hit))
            T["dt"].append(dt)                     # the cache path alone (doc 9ap: replayed by throughput_replay.py)
            T["stages"].append((r.t_knn, r.t_balls, r.t_shell, r.t_model, r.t_serve))
        return np.array([r.score]), np.array([bool(r.hit)]), near[None, :self.k]

    def cache_state(self):
        """The current cache as the timing harness's hit path takes it (eval_timing.FastHitPath):
        cached vector ids, entry centers and radii, the shell sample (ids and the b-bit vectors as
        used), model coefficients, lambda, k, sampling rate, budget."""
        c = self.cache
        (c.full_prepare if hasattr(c, "full_prepare") else c.prepare)()
        mp = self.model.params
        return dict(cached_ids=c.cached_ids, centers=c.centers, radius=c.radius,
                    samp_ids=c.samp_ids, sv=c.sv, mu=mp["mu"], sd=mp["sd"], coef=mp["coef"],
                    intercept=mp["intercept"], lam=np.nan if self.lam is None else self.lam,
                    k=self.k, p=c.p, bits=c.bits, budget=self.M, memory=c.memory(),
                    entry_samp=np.concatenate(c.entry_samp) if c.entry_samp else np.empty(0, int),
                    entry_samp_off=np.cumsum([0] + [len(x) for x in c.entry_samp]),
                    shell_scope=c.shell_scope)

    def save_state(self, path):
        """Cache state after warm-up for eval_timing.py: cached vector ids,
        entry centers and radii, the shell sample (ids and the b-bit vectors
        as used), model coefficients, lambda, k; plus the warm-up length."""
        c = self.cache
        # the simulator's own scores for the next queries, to validate the
        # per-query re-implementation in eval_timing.py
        test_q = self.stream[self.a.warmup:self.a.warmup + 3000]
        st = self.cache_state()
        test_scores = self.model(c.features(test_q)[0])
        np.savez(path, **st, warmup=self.a.warmup, test_q=test_q, test_scores=test_scores,
                 # calibration labels at save time (the warm-up's calibration set): the timing
                 # harness seeds its recalibration with them, as the simulator keeps them
                 cal_t=np.concatenate([np.full(len(x[1]), x[0]) for x in self.cal]) if self.cal else np.empty(0),
                 cal_s=np.concatenate([x[1] for x in self.cal]) if self.cal else np.empty(0),
                 cal_y=np.concatenate([x[2] for x in self.cal]) if self.cal else np.empty(0),
                 cal_w=np.concatenate([x[3] for x in self.cal]) if self.cal else np.empty(0),
                 cal_q=np.concatenate([x[4] for x in self.cal]) if self.cal else np.empty(0, int),
                 fit_q=getattr(self, "fit_q", np.empty(0, int)),
                 lam_start=np.nan if getattr(self, "lam_start", None) is None else self.lam_start)
        logger.info(f"saved cache state to {path}")

    def run(self, method):
        a, rng, cache = self.a, self.rng, self.cache
        self.warmup()
        if getattr(a, "save_state", None):
            self.save_state(a.save_state)
        p_lru = cache.p
        self.og_credit, self.og_miss, self.og_log, self.og_fresh_log = [], [], [], []
        self.og_pred, self.og_phi_log = {}, []
        self.served_pos, self.audit_pos = [], []
        from collections import OrderedDict
        xt = getattr(a, "mini_cache_share", 0.0)
        self.mini = OrderedDict() if xt > 0 else None
        # an entry: the query's key plus its k result vectors (hits return vectors, as Aker's cache does), so k
        # vectors per entry (the key and ids are negligible; vectors shared with the main cache are not deduplicated)
        self.mini_entry_cost = float(self.k)
        self.mini_cap = int(xt * self.M / self.mini_entry_cost)
        if self.mini is not None and method != "ogreedy":
            raise SystemExit("--mini-cache-share: continuous greedy (ogreedy) only")
        if self.mini is not None:            # mini cache seeded with the warm-up's requests (all answered by the backend)
            for q in np.asarray(self.stream[:a.warmup]).tolist():
                self.mini[q] = True
                self.mini.move_to_end(q)
                if len(self.mini) > self.mini_cap:
                    self.mini.popitem(last=False)
        last_alarm = None
        last_check = a.warmup
        pending = []                # scheduled post-alarm rebuild checks (times)
        hits_prev = None
        for s in range(a.warmup, len(self.stream), a.batch):
            qb = self.stream[s:s + a.batch]
            t = s + len(qb)
            n_ev = self.apply_events(s)
            reset_frac = getattr(a, "reset_on_change", 0)
            if reset_frac and n_ev >= reset_frac * len(self.D["db"]):
                # a known large change: calibrate only on labels from now on
                self.events.append({"t": t, "event": "known_change", "n": n_ev})
                self.reset_detection()
                self.window = []
            # --update-every U (doc 9ag): LRU / ogreedy admit and evict after every U requests of the
            # batch (U = 1: every request, as the timing harness's LRU and Aker do); labels, calibration
            # and drift tests stay per batch. Default: once per batch.
            # policy updates (LRU / ogreedy admission and eviction) every UE requests of the batch
            # (--update-every; default once per batch); lookups one request at a time when timed
            UE = getattr(a, "update_every", 0) or len(qb)
            timed = self.timing is not None
            UL = 1 if timed else UE
            upd = method in ("lru", "ogreedy") and UE < len(qb)
            parts, pend = [], []
            self._tparts = []
            self._pool = {"hits": 0, "holders": 0, "cover": 0, "best_cov": 0.0, "single_ret": 0.0}
            step = UL if (upd or timed) else len(qb)
            for u0 in range(0, len(qb), step):
                qs = qb[u0:u0 + step]
                if timed:
                    q0 = int(qs[0])
                    if self.mini is not None and q0 in self.mini:
                        # doc 9as: a mini-cache hit (hash lookup; its time recorded as measured, ~1 us)
                        import time as _t
                        a0 = _t.perf_counter()
                        _ = self.mini[q0]
                        dt0 = _t.perf_counter() - a0
                        T = self.timing
                        pos = s + u0 - self.a.warmup
                        if pos % T["every"] == 0:
                            T["lat"].append(dt0)
                            T["hit"].append(True)
                            T["dt"].append(dt0)
                            T["stages"].append((0.0, 0.0, 0.0, 0.0, 0.0))
                        sc, sv = np.array([1e9]), np.array([True])
                        ns = topk_present(self.D, qs[:1], self.k, key="answer")
                        self._tparts.append(np.array([True]))
                        self.cost["mini_hits"] = self.cost.get("mini_hits", 0) + 1
                    else:
                        sc, sv, ns = self._timed_lookup(q0, s + u0)
                        if self.mini is not None:
                            self._tparts.append(np.array([False]))
                else:
                    sc, Xs, ns = cache.scores(qs, self.model)
                    sv = (sc >= self.lam if self.lam is not None and cache.n_cached() >= self.k
                          else np.zeros(len(qs), bool))
                    if self.mini is not None and len(self.mini):
                        # doc 9as: exact repeats in the mini cache are served its stored backend answer, outside the
                        # hit model; they count as hits at every cutoff (FAST_SCORE), as the main cache's fast path
                        th = np.array([int(q) in self.mini for q in qs])
                        self._tparts.append(th)
                        if th.any():
                            sc = np.where(th, 1e9, sc)
                            sv = sv | th
                            ns = np.array(ns, copy=True)
                            ns[th] = topk_present(self.D, qs[th], self.k, key="answer")
                            self.cost["mini_hits"] = self.cost.get("mini_hits", 0) + int(th.sum())
                ys = self.y(qs, ns)
                ye = ys if self.D["truth"] is self.D["answer"] else (
                    cache.served_retention(qs, ns, key="truth") if self.a.label == "served"
                    else cache.retention_exact(qs))
                parts.append((sc, sv, ys, ye, ns))
                if getattr(a, "pool_stats", False):
                    self._pool_acc(qs, sv, ns)
                if upd:
                    pend.append((qs, sv, ns))
                    if sum(len(p_[0]) for p_ in pend) >= UE or u0 + len(qs) >= len(qb):
                        q_, s_, n_ = (np.concatenate([p_[i] for p_ in pend]) for i in range(3))
                        tt = s + u0 + len(qs)
                        if method == "lru":
                            self.lru_update(q_, s_, n_, tt, p_lru)
                        else:
                            self.ogreedy_update(q_, s_, n_, tt)
                        self.cal_stale = True
                        pend = []
            score, serve, y, y_exact = (np.concatenate([p_[i] for p_ in parts]) for i in range(4))
            # mini-cache hits (doc 9as) do not credit the main cache's entries
            self._tmask = (np.concatenate(self._tparts) if self.mini is not None and len(self._tparts) == len(parts)
                           else np.zeros(len(qb), bool))
            near = np.concatenate([p_[4] for p_ in parts])
            if self.D["truth"] is self.D["answer"]:
                b_rec = np.ones(len(qb))
            else:
                # the backend's own recall (its top-k answer vs the exact top-k):
                # what each query would get without the cache
                ta = topk_present(self.D, qb, self.k, key="truth")
                aa = topk_present(self.D, qb, self.k, key="answer")
                b_rec = (ta[:, :, None] == aa[:, None, :]).any(axis=2).mean(axis=1)
            rate = (a.alarm_audit_rate if last_alarm is not None
                    and s - last_alarm < a.alarm_audit_span else a.audit_rate)
            target = getattr(a, "audit_target", 0)
            if target and hits_prev is not None:
                # adaptive: aim at `target` audits per batch when hits are few
                rate = float(np.clip(target / max(hits_prev, 1), rate, 1.0))
            audit = serve & (rng.random(len(qb)) < rate)
            hits_prev = int(serve.sum())
            if self.mini is not None:
                # mini cache upkeep (LRU): every request of a stored query refreshes it; backend answers (misses,
                # audited hits) are admitted; capacity --mini-cache-share x M / entry cost
                for q, got in zip(qb.tolist(), (~serve | audit).tolist()):
                    if q in self.mini:
                        self.mini.move_to_end(q)
                    elif got:
                        self.mini[q] = True
                        if len(self.mini) > self.mini_cap:
                            self.mini.popitem(last=False)
            if getattr(a, "log_served", False):      # doc 9as: per-request hit / audit positions (stream index)
                self.served_pos.extend((s + np.flatnonzero(serve)).tolist())
                self.audit_pos.extend((s + np.flatnonzero(audit)).tolist())
            self.cost["topk"] += int((~serve).sum() + audit.sum())
            self.log.append({"t": t, "n": len(qb), "served": int(serve.sum()),
                             "served_retention_sum": float(y[serve].sum()),
                             "served_retention_exact_sum": float(y_exact[serve].sum()),
                             "served_backend_recall_sum": float(b_rec[serve].sum()),
                             "backend_recall_sum": float(b_rec.sum()),
                             "ret_hist": np.bincount(np.rint(y * self.k).astype(int),
                                                     minlength=self.k + 1).tolist(),
                             "audited_served": int(audit.sum()),
                             "lambda": self.lam, "memory": cache.memory(),
                             "cached": cache.n_cached(), "entries": len(cache.entries),
                             # cumulative vectors read for the cache (ball members; shell samples, core only), doc 9as
                             "fetched": int(cache.vectors_fetched), "samp_fetched": int(getattr(cache, "samp_fetched", 0)),
                             "events": n_ev, "audit_rate": rate})
            if getattr(a, "pool_stats", False):
                self.log[-1].update({f"pool_{k_}": v_ for k_, v_ in self._pool.items()})
            if "initial" in self.D:
                # queries whose current exact top-k contains inserted vectors
                aff = (~self.D["initial"][topk_present(self.D, qb, self.k,
                                                         key="truth")]).any(axis=1)
                self.log[-1].update(aff_n=int(aff.sum()), aff_served=int((aff & serve).sum()),
                                    aff_ret_sum=float(y[aff & serve].sum()))
            if method != "static":
                # labels: every miss (free) and audited hits (weight 1 / rate);
                # serve-time (score, retention) pairs, i.e. against the cache
                # in force before this batch's updates
                lab = np.ones(len(qb), bool) if a.oracle_cal else (~serve | audit)
                if method == "ogreedy" and getattr(a, "og_split", False):
                    # misses of build-half queries score ogreedy's candidates: not calibration
                    lab &= serve | ~og_build_half(qb)
                wt = np.where(serve & (not a.oracle_cal), 1 / rate, 1.0)
                # the window is used only to build caches (greedy rebuilds)
                self.cal.append((t, score[lab], y[lab], wt[lab], qb[lab]))
                self.ref.append((t, score))
                self.audits.append((t, y[audit], np.full(int(audit.sum()), 1 / rate)))
                self.window = (self.window + [(int(q), bool(v), bool(u), rate)
                                              for q, v, u in zip(qb, serve, audit)])[-a.window:]
                if upd:
                    pass                                    # applied per U requests above
                elif method == "lru":
                    self.lru_update(qb, serve, near, t, p_lru)
                    self.cal_stale = True
                elif method == "ogreedy":
                    self.ogreedy_update(qb, serve, near, t)
                    self.cal_stale = True
                kind = self.detect(t)
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
        res = {"method": method, "args": vars(a), "log": self.log, "events": self.events,
               "og_log": self.og_log, "og_fresh_log": self.og_fresh_log, "og_phi_log": self.og_phi_log,
               "cost": dict(self.cost, vectors_fetched=cache.vectors_fetched,
                            warmup_topR=a.warmup, short_rows=self.D["short"][0])}
        if getattr(a, "log_served", False):
            res["served_pos"], res["audit_pos"] = self.served_pos, self.audit_pos
        if self.timing is not None:
            T = self.timing
            lat, hit = np.array(T["lat"]) * 1e3, np.array(T["hit"], dtype=bool)
            st = np.array(T["stages"]) * 1e3 if T["stages"] else np.zeros((0, 5))
            summ = lambda v: None if not len(v) else {"mean_ms": float(np.mean(v)), "p50_ms": float(np.median(v)),
                                                      "p99_ms": float(np.percentile(v, 99)), "n": int(len(v))}
            res["timing"] = {"every": T["every"], "n_timed": int(len(lat)), "hit_rate_timed": float(hit.mean()) if len(hit) else None,
                             "cache": dict(summ(lat) or {}, hits_ms=summ(lat[hit]), misses_ms=summ(lat[~hit])),
                             "stage_ms": dict(zip(("knn", "balls", "shell", "model", "serve"),
                                                  map(float, st.mean(axis=0)))) if len(st) else None,
                             "lat_ms": lat.tolist(), "hit": hit.tolist(), "dt_ms": (np.array(T["dt"]) * 1e3).tolist(),
                             "ef1": self.cache.ef1, "ef2": self.cache.ef2}
        return res


#-------------------------------------------------------------------------------


def load(prep, args):
    meta_path = Path(prep) / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"kind": "esci"}
    if meta["kind"] == "trace":
        from simulations.approximate_hits.run_e2e_oracle_trace import load_base_f32
        db = load_base_f32(meta["trace"])
        q = np.load(Path(prep) / "q.npy").astype(np.float32)
    else:
        db = np.load(ESCI / f"esci_db{meta.get('n_db', args.n_db)}_s42.npy").astype(np.float32)
        q = np.load(ESCI / "esci_queries.npy").astype(np.float32)
    topf = Path(prep) / f"top{args.R}.npy"
    # doc 9aq: a shell depth R below the stored 500 uses the first R columns of top500.npy
    top = np.load(topf) if topf.exists() else np.load(Path(prep) / "top500.npy", mmap_mode="r")[:, :args.R].copy()
    draw = np.random.default_rng(1).random(len(db))
    streams = np.load(Path(prep) / "streams.npz")
    D = {"db": db, "q": q, "top": top, "draw": draw}
    kind = getattr(args, "data_change", "none")
    if kind != "none":
        # changing database (online_prepare_data.py): union ranking (merged
        # with an initial-database ranking for flagged queries), initial
        # presence, insert / delete schedule
        topU = np.load(Path(prep) / "topU1000.npy")
        extra = np.load(Path(prep) / f"rank_extra_{kind}.npz")
        width = extra["rows"].shape[1] if len(extra["flagged"]) else topU.shape[1]
        rank = np.full((len(q), width), -1, dtype=np.int32)
        rank[:, :topU.shape[1]] = topU
        if len(extra["flagged"]):
            rank[extra["flagged"]] = extra["rows"]
        dc = np.load(Path(prep) / f"data_{kind}.npz")
        n = len(dc["ins_ids"])
        if args.schedule == "burst":
            t_ins = t_del = np.full(n, int(dc["burst_at"]))
        else:
            t_ins, t_del = dc["ins_t"], dc["del_t"]
        t = np.r_[t_ins, t_del]
        order = np.argsort(t, kind="stable")
        D.update(rank=rank, present=dc["initial"].copy(), top=rank, initial=dc["initial"],
                 events={"t": t[order], "id": np.r_[dc["ins_ids"], dc["del_ids"]][order],
                         "insert": np.r_[np.ones(n, bool), np.zeros(n, bool)][order]})
    recall = getattr(args, "backend_recall", 1.0)
    if recall < 1.0:
        # approximate backend: its answers miss each vector independently
        # per query; replacements come from deeper ranks, so rows must be
        # deeper than R (the union ranking, 1000 wide)
        if kind == "none":
            wide = Path(prep) / "topU1000.npy"
            if not wide.exists():
                raise SystemExit(f"{wide} needed for --backend-recall < 1 "
                                 "(online_prepare_data.py computes it)")
            D["rank"] = np.load(wide)
        D["truth"] = D["rank"]
        D["rank"] = D["top"] = D["answer"] = approximate_rows(D["truth"], recall,
                                                             args.backend_seed)
        D["backend_recall_measured"] = {
            f"at_{k}": measured_recall(D["truth"], D["rank"], k) for k in (args.k, args.R)}
    bfile = getattr(args, "backend_file", None)
    if bfile:
        # a real index (eval_real_backend.py): its deep lists and top-k
        # answers at the tuned search effort for --backend-target
        if kind != "none":
            raise SystemExit("real-index backends support a fixed database only")
        z = np.load(bfile)
        # exact truth: the union ranking where it exists (ESCI, SPACEV preps), else the prep's own
        # exact top-R (TriviaQA, TripClick, MedRAG: only top500.npy; the truth needs only top-k)
        wide = Path(prep) / "topU1000.npy"
        D["truth"] = np.load(wide) if wide.exists() else (
            np.load(Path(prep) / f"top{args.R}.npy") if (Path(prep) / f"top{args.R}.npy").exists()
            else D["top"])                       # doc 9aq: top500.npy sliced to R (the truth needs only the top-k)
        D["rank"] = D["top"] = z[f"deep_{args.backend_target}"]
        D["answer"] = z[f"answer_{args.backend_target}"]
        D["backend_recall_measured"] = {
            f"answer_at_{args.k}": measured_recall(D["truth"], D["answer"], args.k),
            f"deep_at_{args.R}": measured_recall(D["truth"], D["rank"], args.R)}
    return D, streams


def make_parser():
    """The simulator's options (also used by eval_timing_online.py's drift-handling ours, which
    runs this simulator's controller live)."""
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--prep", type=str, default="simulations/approximate_hits/results/online/esci1m")
    ap.add_argument("--og-evict", choices=["utility", "lru"], default="utility",
                    help="ogreedy: victims by credit per vector (only when below the candidate's gain "
                         "per vector) or least recently used")
    ap.add_argument("--og-kcal", action="store_true",
                    help="ogreedy: rank candidates of entry size K by phi_K x gain, phi_K = realized / "
                         "predicted hits of this run's entries of size K (no cap on K)")
    ap.add_argument("--og-fresh", action="store_true",
                    help="ogreedy: also admit fresh misses (no evidence of their own) at the K whose "
                         "hindsight value on this run (earlier misses vs later ones) is best, while it "
                         "beats the victims' credit per vector")
    ap.add_argument("--og-split", action="store_true",
                    help="ogreedy: misses of a fixed half of the queries score candidates, the other "
                         "half calibrates (use with --cal-labels maintained: calibration pairs "
                         "re-scored against the current cache every batch)")
    ap.add_argument("--og-sample", type=float, default=1.0,
                    help="ogreedy: share of the new misses that become candidates (get the deep top-R search), doc 9aq")
    ap.add_argument("--og-trigger-m", type=int, default=0,
                    help="ogreedy: a miss is a candidate only if >= this many ids of its answer occur in the answer of another "
                         "distinct recent miss (0 = off), doc 9aq")
    ap.add_argument("--og-trigger-window", type=int, default=2000, help="requests the trigger looks back over")
    ap.add_argument("--lru-admit-p", type=float, default=1.0, help="lru: probability a miss is admitted (doc 9aq)")
    ap.add_argument("--mini-cache-share", type=float, default=0.0,
                    help="ogreedy: share of the budget for the mini cache (query -> the backend's top-k result vectors, "
                         "LRU, k vectors per entry), checked before the main cache, doc 9as; 0 = off")
    ap.add_argument("--og-recent-share", type=float, default=0.0,
                    help="ogreedy: share of the budget for a recency segment of misses' top-k answers (shallow K = k "
                         "entries, no deep search, LRU), doc 9ar; 0 = off")
    ap.add_argument("--og-span", type=int, default=0,
                    help="ogreedy: requests whose misses score candidates and whose hits credit entries "
                         "(0 = --cal-span)")
    ap.add_argument("--method", choices=["static", "greedy", "lru", "ogreedy", "aker", "qvcache"],
                    required=True)
    ap.add_argument("--detect", nargs="*", choices=["score", "retention", "hitrate"],
                    default=["score", "retention", "hitrate"],
                    help="drift tests (give the flag with no values for none)")
    ap.add_argument("--response", choices=["recal", "rebuild"], default="rebuild",
                    help="after an alarm: recalibrate only, or also shadow-rebuild "
                         "(greedy only; lru always recalibrates only)")
    ap.add_argument("--oracle-cal", action="store_true",
                    help="reference: every query is labeled for calibration at no cost")
    ap.add_argument("--stream", choices=["random", "shift", "trace", "multishift"],
                    default="shift")
    ap.add_argument("--data-change", choices=["none", "churn", "drift"], default="none",
                    help="database inserts / deletes (online_prepare_data.py)")
    ap.add_argument("--schedule", choices=["gradual", "burst"], default="gradual")
    ap.add_argument("--backend-recall", type=float, default=1.0,
                    help="simulated approximate backend: recall of its answers (1 = exact); "
                         "labels and caches use its answers, truth is logged separately")
    ap.add_argument("--backend-seed", type=int, default=123)
    ap.add_argument("--backend-file", type=str, default=None,
                    help="real-index answers (results/backends/<prep>/<index>.npz)")
    ap.add_argument("--backend-target", type=str, default="0.95",
                    help="which tuned recall target of --backend-file to use")
    ap.add_argument("--no-maintain", action="store_true",
                    help="no cache upkeep on inserts (balls go stale; deletes still leave)")
    ap.add_argument("--cal-labels", choices=["serve", "maintained"], default="serve",
                    help="calibration pairs as measured at serve time, or kept current")
    ap.add_argument("--maintain-every", type=int, default=5,
                    help="refresh maintained calibration labels every this many batches "
                         "(and right after a rebuild)")
    ap.add_argument("--reset-on-change", type=float, default=0.0,
                    help="reset calibration when one batch changes >= this share of the "
                         "database (0 = never)")
    ap.add_argument("--alarm-level", type=float, default=None,
                    help="fixed per-check false-alarm level (default: delta_alarm "
                         "spread over checks, no accumulation)")
    ap.add_argument("--audit-target", type=int, default=30,
                    help="raise the audit rate to aim at this many audits per batch "
                         "(0 = fixed rate)")
    ap.add_argument("--n-db", type=int, default=1000000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--start", choices=["fit", "none"], default="fit",
                    help="calibration walk: start at the fit split's most certifiable threshold "
                         "(default since 11-12, doc 9p), or at the strictest")
    ap.add_argument("--label", choices=["served", "cached"], default="served",
                    help="retention label: overlap of the served answer with the backend's top-k, or the "
                         "share of the backend's top-k that is cached (the same with an exact backend)")
    ap.add_argument("--budget", type=float, default=0.1, help="memory, fraction of database")
    ap.add_argument("--f", type=float, default=0.85, help="share of budget for the cache")
    ap.add_argument("--R", type=int, default=500)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=None,
                    help="warm-up requests (default: min(20000, a fifth of the full stream); 11-13: MedRAG-Zipf has 10k)")
    ap.add_argument("--audit-rate", type=float, default=0.1, help="share of hits audited")
    ap.add_argument("--cal-span", type=int, default=20000,
                    help="calibrate on the labeled queries among the last this many queries")
    ap.add_argument("--cal-share", type=float, default=0.2,
                    help="share of the window held out at a rebuild to calibrate the new cache")
    ap.add_argument("--window", type=int, default=30000,
                    help="recent queries (since the last alarm) used to build a cache")
    ap.add_argument("--cand-share", type=float, default=0.2,
                    help="share of the building queries proposing candidates")
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
    ap.add_argument("--warm-split", type=float, nargs=4, default=None, metavar=("CAND", "SCORE", "FIT", "CAL"),
                    help="shares of the warm-up used for candidate queries / scoring queries / model "
                         "fit / calibration (default: .2 .4 .2 .2)")
    ap.add_argument("--oracle-score", type=int, nargs=2, default=None, metavar=("START", "END"),
                    help="ORACLE probe: the warm-up cache is chosen by greedy on the requests "
                         "stream[START:END] (later than the warm-up); candidates are 60%% of the warm-up")
    ap.add_argument("--max-ball-k", type=int, default=None,
                    help="largest ball size K greedy may choose (default: 500; sizes are 10 20 50 100 200 500)")
    ap.add_argument("--c-max", type=int, default=3000)
    ap.add_argument("--s-max", type=int, default=25000)
    ap.add_argument("--k-lru", type=int, default=50)
    ap.add_argument("--cache-impl", choices=["python", "core"], default="python",
                    help="core: the cache's request path in the C++ core (core_cache.CoreCache; Python 3.11 / "
                         ".venv-core; doc 9ah); python: online_sim.Cache (arrays rebuilt after every change)")
    ap.add_argument("--core-threads", type=int, default=8, help="--cache-impl core: threads of batch lookups")
    ap.add_argument("--core-ef1", type=int, default=0,
                    help="--cache-impl core: stage-1 search list over the cached vectors (0: exact scan, headline)")
    ap.add_argument("--core-ef2", type=int, default=0, help="--cache-impl core: stage-2 (served points) search list")
    ap.add_argument("--core-ball-mode", choices=["fit", "fit-m", "near", "near-scan"], default="fit",
                    help="--cache-impl core (doc 9ai): fit = best-fitting balls over every entry; fit-m = best fit among "
                         "the --core-ball-m nearest centers; near = the nearest centers (3 for the score, the shell scope "
                         "for the intruder count); a graph over the centers at search list --core-ball-ef")
    ap.add_argument("--core-ball-m", type=int, default=64)
    ap.add_argument("--core-ball-ef", type=int, default=64)
    ap.add_argument("--core-ball-scan-below", type=int, default=1000,
                    help="near: caches of at most this many entries are scanned (the graph pays off above ~1,000 "
                         "entries at 769-d, measured on TripClick, doc 9ai); speed only")
    ap.add_argument("--core-repeat-fast", action="store_true",
                    help="exact-repeat fast path: a request equal to a cached entry's center is served that entry's "
                         "own top-k, skipping the hit path (doc 9ak)")
    ap.add_argument("--core-serve-balls", type=int, default=0,
                    help="core: serve a hit from the k nearest among the members of the n best-fitting entries "
                         "(exact scan) instead of a stage-2 search over the whole cache (doc 9an)")
    ap.add_argument("--shell-range", choices=["minmax", "robust"], default="minmax",
                    help="core: 4-bit code range per dimension, min / max or 1st / 99th percentile (doc 9al)")
    ap.add_argument("--core-insert-ef", type=int, default=40,
                    help="efConstruction of incremental graph inserts (bulk builds use 200; doc 9al)")
    ap.add_argument("--core-shell-code", choices=["sq", "rabitq"], default="sq",
                    help="shell sample codes in the core: b-bit scalar quantization (4 bits: packed) or 1-bit RaBitQ "
                         "(doc 9ak; memory still charged at --bits)")
    ap.add_argument("--timing-compose", default=None,
                    help="timing (doc 9ah): DiskANN-alone latency (s) of every --timing-every-th request after the "
                         "warm-up (eval_timing_backend.py backend_s<every>_lat.npy); each request's core lookup is "
                         "timed and a miss charged its DiskANN latency; needs --cache-impl core")
    ap.add_argument("--timing-every", type=int, default=50)
    ap.add_argument("--aker-real", action="store_true", help="--method aker: the real Aker library (doc 9ah)")
    ap.add_argument("--qv-graph", action="store_true",
                    help="--method qvcache --cache-impl core: search the mini-indexes' graphs (hnswlib) instead of "
                         "an exact scan, without timing (headline runs, doc 9ai)")
    ap.add_argument("--pool-stats", action="store_true",
                    help="log per batch, over the hits: entries holding served vectors, greedy cover size, the best "
                         "entry's share of the served top-k, the best single-entry retention (doc 9ai)")
    ap.add_argument("--aker-real-delta", type=int, default=5)
    ap.add_argument("--aker-real-at", type=float, default=0.9)
    ap.add_argument("--aker-real-al", type=float, default=1.1)
    ap.add_argument("--aker-real-prox", type=float, default=None)
    ap.add_argument("--update-every", type=int, default=0,
                    help="lru / ogreedy: admit and evict after every this many requests (1 = every request; "
                         "0 = once per --batch); calibration and drift tests stay per batch (doc 9ag)")
    ap.add_argument("--aker-alpha-l", type=float, default=1.1)
    ap.add_argument("--aker-alpha-t", type=float, default=0.25)
    ap.add_argument("--aker-tau-init", type=float, default=0.25)
    ap.add_argument("--aker-cap", type=float, default=1.0)
    ap.add_argument("--aker-reserve", type=int, default=0,
                    help="extra neighbors stored per entry for deletes (paper: 0; 10 / 30 "
                         "in its refresh test)")
    ap.add_argument("--aker-risk", type=float, default=0.3,
                    help="risk threshold that triggers slow-path refresh of inserts")
    ap.add_argument("--aker-batch", type=int, default=16,
                    help="insert log records checked per slow-path step")
    ap.add_argument("--qv-deletes", choices=["keep", "remove"], default="keep",
                    help="QVCache on deletes: keep (as published) or remove from the cache")
    ap.add_argument("--qv-mini", type=int, default=4)
    ap.add_argument("--qv-D", type=float, default=0.075, help="QVCache deviation factor")
    ap.add_argument("--qv-alpha", type=float, default=0.9)
    ap.add_argument("--qv-dims", type=int, default=16)
    ap.add_argument("--qv-buckets", type=int, default=8)
    ap.add_argument("--alarm-audit-rate", type=float, default=None,
                    help="audit rate for --alarm-audit-span queries after an alarm")
    ap.add_argument("--alarm-audit-span", type=int, default=10000)
    ap.add_argument("--shell-scope", choices=["all", "balls"], default="all",
                    help="intruder count over the whole shell sample, or only the shells "
                         "of the 3 best-fitting balls")
    ap.add_argument("--shell-balls", type=int, default=3,
                    help="with --shell-scope balls: how many best-fitting balls' shells "
                         "the intruder count covers")
    ap.add_argument("--save-state", type=str, default=None,
                    help="save the cache state after warm-up (for eval_timing.py)")
    ap.add_argument("--tag", type=str, default="", help="suffix of the output file name")
    ap.add_argument("--log-served", action="store_true",
                    help="record the stream positions of hits and audited hits (doc 9as: offline mini-cache oracle)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-queries", type=int, default=None, help="truncate the stream")
    ap.add_argument("--out", type=str, default=None)
    return ap


def main():
    args = make_parser().parse_args()
    K_LIMIT[0] = args.max_ball_k
    D, streams = load(args.prep, args)
    if args.warmup is None:         # from the FULL stream (--max-queries truncates it for the states)
        args.warmup = min(20000, len(streams[args.stream]) // 5)
    stream = streams[args.stream][:args.max_queries]
    out = Path(args.out or Path(args.prep) / f"sim_{args.stream}_b{args.budget:g}")
    out.mkdir(parents=True, exist_ok=True)
    if args.alarm_audit_rate is None:
        args.alarm_audit_rate = args.audit_rate
    if args.method in ("lru", "ogreedy"):
        args.response = "recal"
    if args.method in ("aker", "qvcache"):
        from simulations.approximate_hits import online_baselines as ob
        cls = (ob.AkerRealSim if getattr(args, "aker_real", False) else ob.Aker) if args.method == "aker" else (
            ob.QVCacheCore if getattr(args, "cache_impl", "python") == "core" else ob.QVCache)
        res = dict(cls(D, stream, args).run(), method=args.method, args=vars(args))
    else:
        res = Sim(D, stream, args).run(args.method)
    if "backend_recall_measured" in D:
        res["backend"] = dict(D["backend_recall_measured"],
                              target=args.backend_target if args.backend_file
                              else args.backend_recall, file=args.backend_file)
        logger.info(f"backend recall measured {D['backend_recall_measured']}")
    key = f"phase_{args.stream}"
    if key in streams.files or (args.stream == "shift" and "phase" in streams.files):
        phase = streams[key] if key in streams.files else streams["phase"]
        res["phase_start"] = int(np.argmax(phase[:len(stream)] == 1))
        # start position of every phase (multishift: phases 0..3)
        ph = phase[:len(stream)]
        res["phase_starts"] = [0] + (np.flatnonzero(np.diff(ph)) + 1).tolist()
    name = f"{args.method}{args.tag}.json"
    (out / name).write_text(json.dumps(res))
    logger.info(f"wrote {out / name}")


if __name__ == "__main__":
    main()

