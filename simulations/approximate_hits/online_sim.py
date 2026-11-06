#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Online simulator, milestone 1 (fixed database, fixed k, workload streams).

Inputs from online_prepare.py: exact top-R database ids of every query (the
backend's answers and the ground truth), query streams (random order, or an
abrupt workload shift between two groups of query clusters).

Cache: entries = (center query, K, its exact top-K ids); one entry per center.
Memory = distinct cached vectors + shell sample (b-bit vectors, b / 32 each).
Shell sample: uncached ids at ranks K+1..R around each entry's center, each
kept iff a fixed per-vector uniform draw is < p.

Serve decision (every method): 7 features against the current cache
(relative offset and size for the 3 best-fitting balls, log1p intruder
count), logistic model fit once at warm-up, serve from cache iff score >=
lambda; else miss -> backend. Retention of a served query = fraction of its
true top-k that is cached (the k nearest cached vectors contain all of them).

Audits: each query independently with probability a also goes to the backend
(costs a top-k search only if served). A share of the audited queries
(--cal-share) is reserved for calibration: these are never used to build a
cache (rebuild candidates / scoring, LRU insertions), so calibration stays
independent of population, as offline. The calibration window = the most
recent --n-cal of them, labels recomputed against the current cache (no
backend cost). All audited served queries feed the drift test.

Warm-up (all methods): the first W0 queries all go to the backend (top-R);
split into candidate / scoring / fit / calibration parts; greedy cache of f M
vectors plus a shell sample filling the rest of the budget M; model fit;
lambda on the calibration part.

Methods after warm-up (batches of B queries; the cache and lambda change only
between batches):
  static     nothing changes
  periodic   every E queries: rebuild with the greedy on the last --window
             queries (candidates: the current entries, free, plus C_max random
             misses / build-role audits, whose top-R must be fetched; scoring:
             misses with weight 1, build-role audited served queries with the
             inverse ratio of their observation probabilities), then
             recalibrate lambda
  triggered  lambda recalibrated after every batch (cheap); two drift tests
             after every batch, at level delta_alarm / (2 j (j + 1)) each for
             the j-th check since the last alarm (jointly valid):
               retention  Hoeffding upper bound on the last --drift-window
                          audited served queries is < T
               scores     two-sample KS test between the last --drift-queries
                          incoming queries' scores and the calibration
                          window's scores (label-free, no backend cost)
             alarm -> calibrate only on audits after the alarm (nothing is
             served until two arrive) and restart the rebuild window; also a
             shadow check every E_check queries (once --min-window queries
             are in the window): run the greedy on a random
             half of the window's labeled queries and swap in a new cache
             (rebuilt on the whole window) iff on the other half its oracle
             hits beat the current cache's by a factor (1 + g)
  lru        every miss becomes an entry with K = K_lru (its top-R fetched for
             the shell); an entry is used when it supplies one of the k
             vectors of a served answer; least recently used entries are
             evicted to fit M; lambda recalibrated after every batch

Backend work is counted separately: top-k searches (misses, audited served
queries), top-R searches (warm-up, rebuild candidates, LRU insertions), and
vectors fetched into the cache or sample. Writes <out>/<method>.json.
"""

import json
import argparse
import logging
from pathlib import Path


import numpy as np

from utils.base.logs import setup_logging

from utils.cache.models import fit_logistic
from utils.cache.cache import Cache
from utils.cache.population import greedy_entries, hits_of_entries, oracle_hits
from utils.cache.calibration import hoeffding_upper, select, threshold_grid

logger = logging.getLogger(__name__)

EPS = 1e-12
CHUNK = 1024

#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------


class Sim:
    def __init__(self, D, stream, args):
        self.D, self.stream, self.a = D, stream, args
        self.k, self.T = args.k, args.target
        self.rng = np.random.default_rng(args.seed)
        self.M = args.budget * len(D["db"])
        self.cache = Cache(D, args.k, args.R, args.bits, self.M)
        self.log = []
        self.cost = {"topk": 0, "topR": 0}
        self.audit_q = []           # calibration-role audited queries (never used to build)
        self.window = []            # (query, served, audited, cal role), last --window
        self.events = []
        self.cal_since = 0          # calibration window start (moved by drift alarms)
        self.recent_served = []     # audited served retentions (drift test)
        self.recent_scores = []     # incoming queries' scores (label-free drift test)
        self.cal_scores = np.empty(0)
        self.n_checks = 0

    # -- calibration -------------------------------------------------------

    def calibrate(self, qidx):
        if len(qidx) < 2:
            self.lam, self.cal_scores = None, np.empty(0)
            return
        X, _ = self.cache.features(qidx)
        s = self.model(X)
        y = self.cache.retention(qidx)
        self.cal_scores = s
        self.lam = select("normal_fixed", s, y, threshold_grid(s), self.T, self.a.delta)

    def cal_window(self):
        """Most recent n_cal calibration-role audits; after a drift alarm, only
        those collected since the alarm."""
        start = max(len(self.audit_q) - self.a.n_cal, self.cal_since)
        return np.array(self.audit_q[max(start, 0):], dtype=np.int64)

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
        self.audit_q = list(cal)
        self.calibrate(self.cal_window())
        logger.info(f"warm-up: {len(entries)} entries, {self.cache.n_cached()} cached vectors, "
                    f"p = {self.cache.p:.3f}, memory {self.cache.memory():.0f} / {self.M:.0f}, "
                    f"lambda {self.lam}")

    # -- rebuild -----------------------------------------------------------

    def rebuild_inputs(self):
        """Labeled window queries usable for building (never calibration-role
        audits): misses with weight 1, build-role audited served queries with
        the inverse ratio of their observation probabilities."""
        a = self.a
        lab = [(q, srv) for q, srv, aud, cal in self.window[-a.window:]
               if not cal and ((not srv) or aud)]
        if not lab:
            return None
        if a.audit_rate * (1 - a.cal_share) <= 0:
            raise ValueError("rebuilding needs build-role audits: audit_rate > 0 and cal_share < 1")
        w_srv = int(round((1 - a.audit_rate * a.cal_share)
                          / (a.audit_rate * (1 - a.cal_share))))
        qs = np.array([q for q, _ in lab], dtype=np.int64)
        wts = np.array([w_srv if srv else 1 for _, srv in lab], dtype=np.int64)
        if len(qs) > a.s_max:
            pick = self.rng.choice(len(qs), a.s_max, replace=False)
            qs, wts = qs[pick], wts[pick]
        return qs, wts

    def _greedy(self, qs, wts):
        """Candidates: the current entries (already cached, free) plus new
        window queries (top-R must be fetched). Candidates and scoring
        queries are disjoint (a query's own ball would trivially serve it):
        a random --cand-share of the window's labeled queries proposes
        candidates, the rest score them (offline used 1 : 4)."""
        perm = self.rng.permutation(len(qs))
        n_c = int(self.a.cand_share * len(qs))
        pool, sc = qs[perm[:n_c]], perm[n_c:]
        new = [int(c) for c in self.rng.choice(pool, min(self.a.c_max, len(pool)), replace=False)
               if int(c) not in self.cache.entries]
        self.cost["topR"] += len(new)
        cand = [(c, self.a.R) for c in new] + list(self.cache.entries.items())
        return greedy_entries(self.D, cand, qs[sc], wts[sc], self.a.f * self.M, self.k,
                              self.T)[0]

    def try_rebuild(self, t, force):
        inp = self.rebuild_inputs()
        if inp is None:
            return False
        qs, wts = inp
        new_hits = cur_hits = None
        if not force:
            # shadow check: build on one half, compare on the other (no
            # candidate comes from the comparison half)
            half = self.rng.random(len(qs)) < 0.5
            proposal = self._greedy(qs[half], wts[half])
            hq, hw = qs[~half], wts[~half]
            new_hits = hits_of_entries(self.D, proposal, hq, hw, self.k, self.T)
            cur_hits = oracle_hits(self.cache, hq, hw, self.k, self.T)
            if new_hits < (1 + self.a.gain) * cur_hits:
                self.events.append({"t": t, "event": "shadow_keep", "new": new_hits,
                                    "cur": cur_hits})
                return False
        entries = self._greedy(qs, wts)
        self.cache.replace(entries)
        self.cache.set_p_to_fill()
        self.calibrate(self.cal_window())
        # scores of earlier queries were computed against the old cache
        self.recent_scores, self.recent_served, self.n_checks = [], [], 0
        self.events.append({"t": t, "event": "rebuild", "new": new_hits, "cur": cur_hits,
                            "lambda": self.lam})
        return True

    # -- main loop ---------------------------------------------------------

    def validate(self):
        a = self.a
        if len(self.stream) <= a.warmup:
            raise ValueError(f"stream of {len(self.stream)} queries is not longer than the "
                             f"warm-up ({a.warmup})")
        if a.warmup < 5:
            raise ValueError("warm-up needs at least 5 queries (five-way split)")
        if not 1 <= self.k <= a.R:
            raise ValueError(f"k={self.k} must be between 1 and R={a.R}")
        if not 1 <= a.k_lru <= a.R:
            raise ValueError(f"k_lru={a.k_lru} must be between 1 and R={a.R}")
        if not 0.0 <= a.audit_rate <= 1.0 or not 0.0 <= a.cal_share < 1.0:
            raise ValueError("audit_rate must be in [0, 1] and cal_share in [0, 1)")

    def run(self, method):
        a, rng, cache = self.a, self.rng, self.cache
        self.validate()
        self.warmup()
        last_rebuild = last_check = a.warmup
        p_lru = cache.p
        recal_due = False
        for s in range(a.warmup, len(self.stream), a.batch):
            qb = self.stream[s:s + a.batch]
            X, near = cache.features(qb)
            score = self.model(X)
            serve = score >= self.lam if self.lam is not None else np.zeros(len(qb), bool)
            serve &= cache.can_serve
            y = cache.retention(qb)
            audit = rng.random(len(qb)) < a.audit_rate
            cal = audit & (rng.random(len(qb)) < a.cal_share)
            self.cost["topk"] += int((~serve).sum() + (serve & audit).sum())
            self.audit_q += list(qb[cal])
            self.window += [(int(q), bool(v), bool(u), bool(c))
                            for q, v, u, c in zip(qb, serve, audit, cal)]
            self.window = self.window[-a.window:]
            t = s + len(qb)
            self.log.append({"t": t, "n": len(qb), "served": int(serve.sum()),
                             "served_retention_sum": float(y[serve].sum()),
                             "audited_served": int((serve & audit).sum()),
                             "lambda": self.lam, "memory": cache.memory(),
                             "cached": cache.n_cached(), "entries": len(cache.entries)})
            if method == "periodic" and t - last_rebuild >= a.epoch:
                self.try_rebuild(t, force=True)
                last_rebuild = t
            elif method == "triggered":
                self.recent_served = (self.recent_served + list(y[serve & audit]))[-a.drift_window:]
                self.recent_scores = (self.recent_scores + list(score))[-a.drift_queries:]
                self.n_checks += 1
                level = a.delta_alarm / (self.n_checks * (self.n_checks + 1))
                labeled = (not a.no_ret_alarm and len(self.recent_served) >= 30
                           and hoeffding_upper(self.recent_served, level / 2) < self.T)
                from scipy.stats import ks_2samp
                shifted = (not a.no_score_alarm and len(self.recent_scores) >= a.drift_queries
                           and len(self.cal_scores) >= 200
                           and ks_2samp(self.recent_scores, self.cal_scores).pvalue < level / 2)
                if labeled or shifted:
                    # fresh start: calibrate only on audits from now on, and
                    # build the next cache only from queries from now on
                    self.events.append({"t": t, "event": "alarm",
                                        "kind": "retention" if labeled else "scores"})
                    self.recent_served, self.recent_scores, self.n_checks = [], [], 0
                    self.cal_since = len(self.audit_q)
                    self.window = []
                    last_check = t
                    recal_due = True
                elif t - last_check >= a.check_every:
                    if not a.no_shadow and len(self.window) >= a.min_window:
                        if self.try_rebuild(t, force=False):
                            last_rebuild = t
                    last_check = t
                    recal_due = True
                # cheap continuous recalibration on the calibration window;
                # without it, lambda is reset only at alarms, checks and rebuilds
                if not a.no_recal or recal_due:
                    self.calibrate(self.cal_window())
                    recal_due = False
            elif method == "lru":
                cache.mark_used(near[serve], t)
                # calibration-role queries are held out from the cache
                miss = [int(q) for q in qb[~serve & ~cal] if int(q) not in cache.entries]
                self.cost["topR"] += len(miss)
                for q in miss:
                    cache.add(q, a.k_lru)
                    cache.last_used[q] = t
                cache.p = p_lru
                order = sorted(cache.entries, key=lambda c: cache.last_used.get(c, 0))
                i = 0
                while cache.entries and cache.memory() > self.M:
                    # evict in chunks of about enough entries to cover the excess
                    n_ev = max(1, int((cache.memory() - self.M) / (2 * a.k_lru)))
                    for c in order[i:i + n_ev]:
                        cache.remove(c)
                    i += n_ev
                self.calibrate(self.cal_window())
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
    db = np.load(Path(args.esci_dir) / f"esci_db{args.n_db}_s42.npy").astype(np.float32)
    q = np.load(Path(args.esci_dir) / "esci_queries.npy").astype(np.float32)
    top = np.load(Path(prep) / f"top{args.R}.npy")
    draw = np.random.default_rng(1).random(len(db))
    streams = np.load(Path(prep) / "streams.npz")
    return {"db": db, "q": q, "top": top, "draw": draw}, streams


def main():
    setup_logging()
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--prep", type=str, default="simulations/approximate_hits/results/online/esci1m")
    ap.add_argument("--method", choices=["static", "periodic", "triggered", "lru", "aker",
                                         "qvcache"], required=True)
    ap.add_argument("--stream", choices=["random", "shift"], default="shift")
    ap.add_argument("--n-db", type=int, default=1000000)
    ap.add_argument("--esci-dir", type=str, default="datasets/esci/data_vectors")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--budget", type=float, default=0.1, help="memory, fraction of database")
    ap.add_argument("--f", type=float, default=0.85, help="share of budget for the cache")
    ap.add_argument("--R", type=int, default=500)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=20000)
    ap.add_argument("--audit-rate", type=float, default=0.1)
    ap.add_argument("--cal-share", type=float, default=0.6,
                    help="share of audited queries reserved for calibration")
    ap.add_argument("--n-cal", type=int, default=3000)
    ap.add_argument("--epoch", type=int, default=10000)
    ap.add_argument("--window", type=int, default=30000,
                    help="recent queries used to build a cache")
    ap.add_argument("--cand-share", type=float, default=0.2,
                    help="share of the window's labeled queries proposing candidates")
    ap.add_argument("--drift-window", type=int, default=500,
                    help="recent audited served queries in the retention test")
    ap.add_argument("--drift-queries", type=int, default=2000,
                    help="recent queries in the label-free score-distribution test")
    ap.add_argument("--min-window", type=int, default=5000,
                    help="queries needed in the window before a shadow rebuild check")
    ap.add_argument("--check-every", type=int, default=5000)
    ap.add_argument("--gain", type=float, default=0.1)
    ap.add_argument("--delta-alarm", type=float, default=0.05)
    ap.add_argument("--c-max", type=int, default=3000)
    ap.add_argument("--s-max", type=int, default=25000)
    ap.add_argument("--k-lru", type=int, default=50)
    ap.add_argument("--aker-alpha-l", type=float, default=1.1)
    ap.add_argument("--aker-alpha-t", type=float, default=0.25)
    ap.add_argument("--qv-mini", type=int, default=4)
    ap.add_argument("--qv-D", type=float, default=0.075, help="QVCache deviation factor")
    ap.add_argument("--qv-alpha", type=float, default=0.9)
    ap.add_argument("--qv-dims", type=int, default=16)
    ap.add_argument("--qv-buckets", type=int, default=8)
    # ablation switches for the triggered method
    ap.add_argument("--no-recal", action="store_true",
                    help="no per-batch recalibration (only at alarms / checks / rebuilds)")
    ap.add_argument("--no-ret-alarm", action="store_true", help="disable the retention alarm")
    ap.add_argument("--no-score-alarm", action="store_true",
                    help="disable the score-distribution alarm")
    ap.add_argument("--no-shadow", action="store_true", help="disable shadow rebuild checks")
    ap.add_argument("--tag", type=str, default="", help="suffix of the output file name")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-queries", type=int, default=None, help="truncate the stream")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    D, streams = load(args.prep, args)
    stream = streams[args.stream][:args.max_queries]
    out = Path(args.out or Path(args.prep) / f"sim_{args.stream}_b{args.budget:g}")
    out.mkdir(parents=True, exist_ok=True)
    if args.method in ("aker", "qvcache"):
        from utils.cache import baselines as ob
        cls = ob.Aker if args.method == "aker" else ob.QVCache
        res = dict(cls(D, stream, args).run(), method=args.method, args=vars(args))
    else:
        res = Sim(D, stream, args).run(args.method)
    if args.stream == "shift":
        phase = streams["phase"]
        res["phase_start"] = int(np.argmax(phase == 1)) if (phase == 1).any() else len(phase)
    name = f"{args.method}{args.tag}.json"
    (out / name).write_text(json.dumps(res))
    logger.info(f"wrote {out / name}")


if __name__ == "__main__":
    main()
