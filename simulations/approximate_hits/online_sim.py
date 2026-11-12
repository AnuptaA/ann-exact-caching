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
        self._fetched = 0
        self._count = np.zeros(self.n_db, dtype=np.int32)
        self._shell = np.empty(0, dtype=np.int64)
        self._members_dirty = True
        self._dirty = True

    # -- membership --------------------------------------------------------

    def add(self, c, K):
        """Add (or resize) an entry; vectors already cached are not refetched."""
        row = self.D["rank"][c]
        pres = (row >= 0) & self.D["present"][np.maximum(row, 0)]
        cum = np.cumsum(pres)
        pe = int(np.searchsorted(cum, K)) + 1 if cum[-1] >= K else len(row)
        ps = int(np.searchsorted(cum, self.R)) + 1 if cum[-1] >= self.R else len(row)
        ps = max(ps, pe)
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
            X, _ = self.cache.features(qs)
            sc, yy = self.model(X), self.cache.retention(qs)
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
        self.lam = select_weighted(s, y, w, threshold_grid(s), self.T, self.a.delta) \
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
        entries, _ = greedy_entries(self.D, [(int(c), a.R) for c in cand], score,
                                    np.ones(len(score)), a.f * self.M, self.k, self.T)
        self.cache.replace(entries)
        self.cache.set_p_to_fill()
        X, _ = self.cache.features(fit)
        self.model = fit_logistic(X, self.cache.retention(fit))
        X, _ = self.cache.features(cal)
        s = self.model(X)
        self.cal = [(a.warmup, s, self.cache.retention(cal), np.ones(len(cal)), cal)]
        self.ref = [(a.warmup, s)]
        self.calibrate(a.warmup)
        logger.info(f"warm-up: {len(entries)} entries, {self.cache.n_cached()} cached vectors, "
                    f"p = {self.cache.p:.3f}, memory {self.cache.memory():.0f} / {self.M:.0f}, "
                    f"lambda {self.lam}")

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
        X, _ = self.cache.features(hq)
        s = self.model(X)
        self.reset_detection()
        hl = lab[hold]
        self.cal = [(t, s[hl], self.cache.retention(hq)[hl], 1 / pi[hold][hl], hq[hl])]
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

    def save_state(self, path):
        """Cache state after warm-up for eval_timing.py: cached vector ids,
        entry centers and radii, the shell sample (ids and the b-bit vectors
        as used), model coefficients, lambda, k; plus the warm-up length."""
        c = self.cache
        c.prepare()
        mp = self.model.params
        # the simulator's own scores for the next queries, to validate the
        # per-query re-implementation in eval_timing.py
        test_q = self.stream[self.a.warmup:self.a.warmup + 3000]
        test_scores = self.model(c.features(test_q)[0])
        np.savez(path, cached_ids=c.cached_ids, centers=c.centers, radius=c.radius,
                 samp_ids=c.samp_ids, sv=c.sv, mu=mp["mu"], sd=mp["sd"], coef=mp["coef"],
                 intercept=mp["intercept"], lam=np.nan if self.lam is None else self.lam,
                 k=self.k, warmup=self.a.warmup, p=c.p, bits=c.bits, budget=self.M,
                 memory=c.memory(), test_q=test_q, test_scores=test_scores,
                 entry_samp=np.concatenate(c.entry_samp) if c.entry_samp else np.empty(0, int),
                 entry_samp_off=np.cumsum([0] + [len(x) for x in c.entry_samp]),
                 shell_scope=c.shell_scope,
                 # calibration labels at save time (the warm-up's calibration set): the timing
                 # harness seeds its recalibration with them, as the simulator keeps them
                 cal_t=np.concatenate([np.full(len(x[1]), x[0]) for x in self.cal]) if self.cal else np.empty(0),
                 cal_s=np.concatenate([x[1] for x in self.cal]) if self.cal else np.empty(0),
                 cal_y=np.concatenate([x[2] for x in self.cal]) if self.cal else np.empty(0),
                 cal_w=np.concatenate([x[3] for x in self.cal]) if self.cal else np.empty(0))
        logger.info(f"saved cache state to {path}")

    def run(self, method):
        a, rng, cache = self.a, self.rng, self.cache
        self.warmup()
        if getattr(a, "save_state", None):
            self.save_state(a.save_state)
        p_lru = cache.p
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
            X, near = cache.features(qb)
            score = self.model(X)
            serve = (score >= self.lam if self.lam is not None and cache.n_cached() >= self.k
                     else np.zeros(len(qb), bool))
            y = cache.retention(qb)
            y_exact = y if self.D["truth"] is self.D["answer"] else cache.retention_exact(qb)
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
                             "events": n_ev, "audit_rate": rate})
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
                wt = np.where(serve & (not a.oracle_cal), 1 / rate, 1.0)
                # the window is used only to build caches (greedy rebuilds)
                self.cal.append((t, score[lab], y[lab], wt[lab], qb[lab]))
                self.ref.append((t, score))
                self.audits.append((t, y[audit], np.full(int(audit.sum()), 1 / rate)))
                self.window = (self.window + [(int(q), bool(v), bool(u), rate)
                                              for q, v, u in zip(qb, serve, audit)])[-a.window:]
                if method == "lru":
                    self.lru_update(qb, serve, near, t, p_lru)
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
        return {"method": method, "args": vars(a), "log": self.log, "events": self.events,
                "cost": dict(self.cost, vectors_fetched=cache.vectors_fetched,
                             warmup_topR=a.warmup, short_rows=self.D["short"][0])}


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
    top = np.load(Path(prep) / f"top{args.R}.npy")
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
        D["truth"] = np.load(Path(prep) / "topU1000.npy")
        D["rank"] = D["top"] = z[f"deep_{args.backend_target}"]
        D["answer"] = z[f"answer_{args.backend_target}"]
        D["backend_recall_measured"] = {
            f"answer_at_{args.k}": measured_recall(D["truth"], D["answer"], args.k),
            f"deep_at_{args.R}": measured_recall(D["truth"], D["rank"], args.R)}
    return D, streams


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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-queries", type=int, default=None, help="truncate the stream")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    K_LIMIT[0] = args.max_ball_k
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

