#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Selection engine on an abstract incidence structure.

Structure: n train queries with K slots each (the vector id in every slot of each query's top-K), and candidates, each a
group of train queries plus a vector set R. State: a pool of vectors and, per (query, slot), a coverage count
cnt = number of SELECTED candidates whose group does NOT contain the query and whose R contains the slot's vector.
Recall of a query is the fraction of its slots with cnt > 0 (leave-group-out, provenance: a candidate gives no credit
to any query in its group). Deficits are integers in units of 1/K recall. Budget and cost use the FULL pool:
cost(c) = |R_c minus pool|, constraint |pool| <= B.

TWO EVALUATION PATHS, NEVER MIXED.
1. The greedy (this module) uses exact deficits and the (query, slot) coverage counters only.
2. LSH train deficits (HP3 cap, LSH hindsight) are computed once per FINAL pool at evaluation time, by
   `group_aware_counts` below. The LSH estimate counts pool vectors within r_hat, and r_hat can exceed theta, so the
   count includes vectors outside the query's true top-K that the slot counters cannot see. A pool vector within
   r_hat counts for train query q only if some selected candidate containing it has a group that excludes q. This is
   exact and is not an approximation of the slot counters.

Scores (a function of the state; gain = score after adding the candidate minus score now; utility = gain / cost):
SEL1 number of queries with recall >= T, SEL2 hindsight hit count h*, SEL3 sum of recall, SEL4 minus the sum of the h0
smallest deficits, SEL5 SEL4 with h0 = the current h*, taken before the candidate is added. All gains are integers,
so utilities compare exactly. Ties break by smaller cost, then smaller candidate id; cost 0 with positive gain is
taken first; the greedy stops when no fitting candidate has positive gain.

Exact incremental refresh: an inverted index vector -> candidates and cached per-candidate deltas (number of still
uncovered slots of each query that the candidate would cover). After a pick only candidates sharing a vector with it
get their cost and deltas updated; all scores are then recomputed from the cached deltas and the global deficit
histogram. `greedy_reference` is a slow independent implementation used to validate the engine on small instances."""

import time
from dataclasses import dataclass
from fractions import Fraction

import numba as nb
import numpy as np

import ee_config as cfg

SCORES = ("SEL1", "SEL2", "SEL3", "SEL4", "SEL5")

#-------------------------------------------------------------------------------
# Structure

@dataclass
class Incidence:
    n: int
    K: int
    n_vec: int
    slot_vec: np.ndarray      # (n, K) int32 vector id in each slot
    cand_ptr: np.ndarray      # (C + 1,) int64
    cand_vec: np.ndarray      # int32, vector set of each candidate (unique within a candidate)
    grp_ptr: np.ndarray       # (C + 1,) int64
    grp_q: np.ndarray         # int32, group queries of each candidate
    cand_source: np.ndarray   # (C,) int32 source query of each candidate (-1 if unknown)
    cnt0: np.ndarray = None   # (n, K) initial coverage counts (toy states); real runs start from zero

    @property
    def n_cand(self) -> int:
        return len(self.cand_ptr) - 1

def csr_from_lists(lists) -> tuple:
    ptr = np.zeros(len(lists) + 1, dtype=np.int64)
    ptr[1:] = np.cumsum([len(x) for x in lists])
    flat = np.concatenate([np.asarray(x, dtype=np.int32) for x in lists]) if len(lists) else np.empty(0, np.int32)
    return ptr, flat

def slack_tenths(T: float, n: int, K: int) -> int:
    """Total slack n (1 - T) in deficit units of 1/K recall (floor, exact for integer sums)."""
    return cfg.slack_units(T, n) * K // cfg.UNITS_PER_RECALL

def cover_threshold(T: float, K: int) -> int:
    """Smallest number of covered slots with recall >= T."""
    return -(-cfg.target_units(T) * K // cfg.UNITS_PER_RECALL)

#-------------------------------------------------------------------------------
# Numba kernels

@nb.njit(cache=True)
def _hstar(H, S_t):
    h = 0
    rem = S_t
    for u in range(len(H)):
        cnt = H[u]
        if u == 0:
            h += cnt
        else:
            take = min(cnt, rem // u)
            h += take
            rem -= take * u
            if take < cnt:
                break
    return h

@nb.njit(cache=True)
def _sum_smallest(H, h0):
    total = 0
    left = h0
    for u in range(len(H)):
        take = min(H[u], left)
        total += take * u
        left -= take
        if left == 0:
            break
    return total

@nb.njit(cache=True)
def _invert(n_out, ptr, idx):
    """CSR transpose: for every value v in idx, the list of rows (positions in ptr) that contain it."""
    counts = np.zeros(n_out + 1, np.int64)
    for k in range(len(idx)):
        counts[idx[k] + 1] += 1
    out_ptr = np.cumsum(counts)
    pos = out_ptr[:-1].copy()
    out = np.empty(len(idx), np.int32)
    for r in range(len(ptr) - 1):
        for k in range(ptr[r], ptr[r + 1]):
            v = idx[k]
            out[pos[v]] = r
            pos[v] += 1
    return out_ptr, out

@nb.njit(cache=True)
def _build_aff(cand_ptr, cand_vec, grp_ptr, grp_q, vs_ptr, vs_slot, cnt0, K, n):
    C = len(cand_ptr) - 1
    tmp = np.zeros(n, np.int32)
    gmark = np.zeros(n, np.int64)
    touched = np.empty(n, np.int32)
    lens = np.zeros(C, np.int64)
    for rnd in range(2):
        if rnd == 1:
            aff_ptr = np.zeros(C + 1, np.int64)
            for c in range(C):
                aff_ptr[c + 1] = aff_ptr[c] + lens[c]
            aff_q = np.empty(aff_ptr[C], np.int32)
            aff_d = np.empty(aff_ptr[C], np.int32)
        for c in range(C):
            for k in range(grp_ptr[c], grp_ptr[c + 1]):
                gmark[grp_q[k]] = c + 1
            nt = 0
            for k in range(cand_ptr[c], cand_ptr[c + 1]):
                v = cand_vec[k]
                for t in range(vs_ptr[v], vs_ptr[v + 1]):
                    s = vs_slot[t]
                    q = s // K
                    if gmark[q] == c + 1 or cnt0[s] > 0:
                        continue
                    if tmp[q] == 0:
                        touched[nt] = q
                        nt += 1
                    tmp[q] += 1
            if rnd == 0:
                lens[c] = nt
            else:
                order = np.sort(touched[:nt])
                base = aff_ptr[c]
                for j in range(nt):
                    aff_q[base + j] = order[j]
                    aff_d[base + j] = tmp[order[j]]
            for j in range(nt):
                tmp[touched[j]] = 0
    return aff_ptr, aff_q, aff_d

@nb.njit(parallel=True, cache=True)
def _score_all(mode, aff_ptr, aff_q, aff_d, covered, K, H, S_t, thr, h0, h_now, alive, cost, budget_left, gains):
    C = len(aff_ptr) - 1
    sum_now = _sum_smallest(H, h0)
    for c in nb.prange(C):
        if (not alive[c]) or cost[c] > budget_left:
            gains[c] = 0
            continue
        g = 0
        if mode == 1:
            for k in range(aff_ptr[c], aff_ptr[c + 1]):
                d = aff_d[k]
                if d > 0:
                    u = K - covered[aff_q[k]]
                    if u > thr and u - d <= thr:
                        g += 1
        elif mode == 3:
            for k in range(aff_ptr[c], aff_ptr[c + 1]):
                g += aff_d[k]
        else:
            hh = H.copy()
            for k in range(aff_ptr[c], aff_ptr[c + 1]):
                d = aff_d[k]
                if d > 0:
                    u = K - covered[aff_q[k]]
                    hh[u] -= 1
                    hh[u - d] += 1
            if mode == 2:
                g = _hstar(hh, S_t) - h_now
            else:
                g = sum_now - _sum_smallest(hh, h0)
        gains[c] = g

@nb.njit(cache=True)
def _pick(gains, cost, alive, budget_left):
    best = -1
    best_util = -1.0
    best_cost = 0
    for c in range(len(gains)):
        if (not alive[c]) or gains[c] <= 0 or cost[c] > budget_left:
            continue
        util = np.inf if cost[c] == 0 else gains[c] / cost[c]
        if best < 0 or util > best_util or (util == best_util and cost[c] < best_cost):
            best, best_util, best_cost = c, util, cost[c]
    return best

@nb.njit(cache=True)
def _bsearch(arr, lo, hi, x):
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo

@nb.njit(cache=True)
def _apply(p, stamp, cand_ptr, cand_vec, grp_ptr, grp_q, vs_ptr, vs_slot, vc_ptr, vc_cand, aff_ptr, aff_q, aff_d,
           cnt, covered, H, cost, in_pool, gstamp, K):
    for k in range(grp_ptr[p], grp_ptr[p + 1]):
        gstamp[grp_q[k]] = stamp
    added = 0
    for k in range(cand_ptr[p], cand_ptr[p + 1]):
        v = cand_vec[k]
        if not in_pool[v]:
            in_pool[v] = True
            added += 1
            for t in range(vc_ptr[v], vc_ptr[v + 1]):
                cost[vc_cand[t]] -= 1
        for t in range(vs_ptr[v], vs_ptr[v + 1]):
            s = vs_slot[t]
            q = s // K
            if gstamp[q] == stamp:
                continue
            if cnt[s] == 0:
                u = K - covered[q]
                H[u] -= 1
                H[u - 1] += 1
                covered[q] += 1
                for j in range(vc_ptr[v], vc_ptr[v + 1]):
                    c2 = vc_cand[j]
                    lo, hi = aff_ptr[c2], aff_ptr[c2 + 1]
                    pos = _bsearch(aff_q, lo, hi, q)
                    if pos < hi and aff_q[pos] == q:
                        aff_d[pos] -= 1
            cnt[s] += 1
    return added

#-------------------------------------------------------------------------------
# Index and greedy

@dataclass
class Index:
    vs_ptr: np.ndarray
    vs_slot: np.ndarray
    vc_ptr: np.ndarray
    vc_cand: np.ndarray
    aff_ptr: np.ndarray
    aff_q: np.ndarray
    aff_d0: np.ndarray

def build_index(inc: Incidence) -> Index:
    flat = inc.slot_vec.ravel()
    order = np.argsort(flat, kind="stable").astype(np.int64)
    vs_ptr = np.zeros(inc.n_vec + 1, np.int64)
    vs_ptr[1:] = np.cumsum(np.bincount(flat, minlength=inc.n_vec))
    vc_ptr, vc_cand = _invert(inc.n_vec, inc.cand_ptr, inc.cand_vec)
    cnt0 = np.zeros(inc.n * inc.K, np.int32) if inc.cnt0 is None else inc.cnt0.ravel().astype(np.int32)
    aff_ptr, aff_q, aff_d0 = _build_aff(inc.cand_ptr, inc.cand_vec, inc.grp_ptr, inc.grp_q, vs_ptr, order, cnt0,
                                        inc.K, inc.n)
    return Index(vs_ptr, order, vc_ptr, vc_cand, aff_ptr, aff_q, aff_d0)

@dataclass
class Result:
    selected: np.ndarray
    pool_mask: np.ndarray
    covered: np.ndarray       # per query, covered slots (leave-group-out recall count)
    h_star: int
    pool_size: int
    stalled: bool
    unused_budget: int
    n_distinct_sources: int
    steps: dict
    seconds: float

class _Run:
    """Mutable greedy state plus the kernel calls."""

    def __init__(self, inc: Incidence, idx: Index, T: float):
        self.inc, self.idx = inc, idx
        n, K, C = inc.n, inc.K, inc.n_cand
        self.cnt = np.zeros(n * K, np.int32) if inc.cnt0 is None else inc.cnt0.ravel().astype(np.int32).copy()
        self.covered = (self.cnt.reshape(n, K) > 0).sum(axis=1).astype(np.int32)
        self.H = np.bincount(K - self.covered, minlength=K + 1).astype(np.int64)
        self.S_t, self.thr = slack_tenths(T, n, K), K - cover_threshold(T, K)
        self.cost = np.diff(inc.cand_ptr).astype(np.int64)
        self.aff_d = idx.aff_d0.copy()
        self.alive = np.ones(C, dtype=bool)
        self.in_pool = np.zeros(inc.n_vec, dtype=bool)
        self.gstamp = np.zeros(n, np.int64)
        self.gains = np.zeros(C, np.int64)
        self.pool_size = 0
        self.picks = []

    def h_star(self) -> int:
        return int(_hstar(self.H, self.S_t))

    def compute_gains(self, score: str, budget_left: int, h0: int = None) -> np.ndarray:
        mode = SCORES.index(score) + 1
        h_now = self.h_star()
        h0_eff = h_now if mode == 5 else (h0 if mode == 4 else 0)
        _score_all(mode, self.idx.aff_ptr, self.idx.aff_q, self.aff_d, self.covered, self.inc.K, self.H, self.S_t,
                   self.thr, h0_eff, h_now, self.alive, self.cost, budget_left, self.gains)
        return self.gains

    def apply(self, c: int) -> int:
        inc, idx = self.inc, self.idx
        added = _apply(c, len(self.picks) + 1, inc.cand_ptr, inc.cand_vec, inc.grp_ptr, inc.grp_q, idx.vs_ptr,
                       idx.vs_slot, idx.vc_ptr, idx.vc_cand, idx.aff_ptr, idx.aff_q, self.aff_d, self.cnt,
                       self.covered, self.H, self.cost, self.in_pool, self.gstamp, inc.K)
        self.alive[c] = False
        self.pool_size += added
        self.picks.append(c)
        return added

def score_gains(inc: Incidence, idx: Index, score: str, T: float, h0: int = None, replay: tuple = (),
                budget: int = 10 ** 9) -> tuple:
    """(gains, costs) of every candidate in the state reached by buying the `replay` candidates in order."""
    run = _Run(inc, idx, T)
    for c in replay:
        run.apply(c)
    run.compute_gains(score, budget - run.pool_size, h0)
    return run.gains.copy(), run.cost.copy()

def replay_h_star(inc: Incidence, idx: Index, order: tuple, T: float) -> int:
    run = _Run(inc, idx, T)
    for c in order:
        run.apply(c)
    return run.h_star()

def greedy(inc: Incidence, idx: Index, score: str, T: float, budget: int, h0: int = None,
           max_steps: int = None) -> Result:
    """Exact incremental greedy. SEL4 needs h0; SEL5 takes h0 from the current h* before every purchase."""
    assert score in SCORES and (score != "SEL4" or h0 is not None)
    t0 = time.time()
    run = _Run(inc, idx, T)
    log = dict(cand=[], gain=[], cost=[], h_before=[], h_after=[], n_positive=[], pool_size=[])
    while max_steps is None or len(run.picks) < max_steps:
        budget_left = budget - run.pool_size
        h_now = run.h_star()
        run.compute_gains(score, budget_left, h0)
        best = int(_pick(run.gains, run.cost, run.alive, budget_left))
        if best < 0:
            break
        g, c_best = int(run.gains[best]), int(run.cost[best])
        n_pos = int((run.gains > 0).sum())
        added = run.apply(best)
        assert added == c_best, "cost bookkeeping drifted"
        for key, val in zip(log, (best, g, c_best, h_now, run.h_star(), n_pos, run.pool_size)):
            log[key].append(val)
    picks = np.array(run.picks, dtype=np.int64)
    src = inc.cand_source[picks] if len(picks) else np.empty(0, np.int32)
    return Result(selected=picks, pool_mask=run.in_pool, covered=run.covered.copy(), h_star=run.h_star(),
                  pool_size=run.pool_size, stalled=(budget - run.pool_size) > 0, unused_budget=budget - run.pool_size,
                  n_distinct_sources=int(len(np.unique(src[src >= 0]))),
                  steps={k: np.array(v) for k, v in log.items()}, seconds=time.time() - t0)

#-------------------------------------------------------------------------------
# Slow independent reference (small instances only)

def greedy_reference(inc: Incidence, score: str, T: float, budget: int, h0: int = None) -> dict:
    """Straightforward recomputation of every candidate's gain from scratch each step, with exact fractions."""
    n, K = inc.n, inc.K
    S_t, thr = slack_tenths(T, n, K), K - cover_threshold(T, K)
    cnt = np.zeros((n, K), np.int64) if inc.cnt0 is None else inc.cnt0.astype(np.int64).copy()
    slots_of = {}
    for q in range(n):
        for j in range(K):
            slots_of.setdefault(int(inc.slot_vec[q, j]), []).append((q, j))
    R = [set(inc.cand_vec[inc.cand_ptr[c]:inc.cand_ptr[c + 1]].tolist()) for c in range(inc.n_cand)]
    G = [set(inc.grp_q[inc.grp_ptr[c]:inc.grp_ptr[c + 1]].tolist()) for c in range(inc.n_cand)]

    def deficits(cn):
        return np.sort(K - (cn > 0).sum(axis=1))

    def hstar(def_sorted):
        return int((np.cumsum(def_sorted) <= S_t).sum())

    def value(cn, h0_fixed):
        d = deficits(cn)
        if score == "SEL1":
            return int((d <= thr).sum())
        if score == "SEL2":
            return hstar(d)
        if score == "SEL3":
            return int((cn > 0).sum())
        return -int(d[:h0_fixed].sum())

    pool, alive, picks, gains_log = set(), set(range(inc.n_cand)), [], []
    while True:
        h0_fixed = hstar(deficits(cnt)) if score == "SEL5" else h0
        now = value(cnt, h0_fixed)
        best, best_key = None, None
        for c in sorted(alive):
            cost = len(R[c] - pool)
            if cost > budget - len(pool):
                continue
            cn = cnt.copy()
            for v in R[c]:
                for q, j in slots_of.get(v, []):
                    if q not in G[c]:
                        cn[q, j] += 1
            gain = value(cn, h0_fixed) - now
            if gain <= 0:
                continue
            util = Fraction(10 ** 18) if cost == 0 else Fraction(gain, cost)
            key = (util, -cost, -c)
            if best_key is None or key > best_key:
                best, best_key, best_gain = c, key, gain
        if best is None:
            break
        for v in R[best]:
            for q, j in slots_of.get(v, []):
                if q not in G[best]:
                    cnt[q, j] += 1
        pool |= R[best]
        alive.discard(best)
        picks.append(best)
        gains_log.append(best_gain)
    return dict(selected=np.array(picks, dtype=np.int64), gains=np.array(gains_log), pool=pool,
                covered=(cnt > 0).sum(axis=1), h_star=hstar(deficits(cnt)))

#-------------------------------------------------------------------------------
# Group-aware LSH train counts (evaluation path 2)

@nb.njit(parallel=True, cache=True)
def _group_counts_kernel(dist, r_hat, q_lo, pos_of, total, qc_ptr, qc_list, sel_ptr, sel_vec, n_pool):
    n_rows = dist.shape[0]
    out = np.zeros(n_rows, np.int64)
    for i in nb.prange(n_rows):
        q = q_lo + i
        blocked = np.zeros(n_pool, np.int32)
        for t in range(qc_ptr[q], qc_ptr[q + 1]):
            c = qc_list[t]
            for k in range(sel_ptr[c], sel_ptr[c + 1]):
                blocked[pos_of[sel_vec[k]]] += 1
        cnt = 0
        for p in range(n_pool):
            if dist[i, p] <= r_hat[q] and total[p] - blocked[p] > 0:
                cnt += 1
        out[i] = cnt
    return out

def group_aware_counts(inc: Incidence, selected: np.ndarray, pool_ids: np.ndarray, r_hat: np.ndarray,
                       train: np.ndarray, base: np.ndarray, chunk: int = 500) -> np.ndarray:
    """Estimated recall counts of the LSH arm for train queries: the number of pool vectors within r_hat of the query
    that are available to it, meaning some selected candidate containing the vector has a group excluding the query.
    Built once per final pool; independent of the greedy and of the top-K slot counters."""
    from utils.base.distance_metrics import pairwise_angular_distance
    pool_ids = np.asarray(pool_ids, dtype=np.int64)
    pos_of = np.full(inc.n_vec, -1, np.int64)
    pos_of[pool_ids] = np.arange(len(pool_ids))
    sel = np.asarray(selected, dtype=np.int64)
    sel_ptr, sel_vec = csr_from_lists([inc.cand_vec[inc.cand_ptr[c]:inc.cand_ptr[c + 1]] for c in sel])
    total = np.bincount(pos_of[sel_vec], minlength=len(pool_ids)).astype(np.int32)
    grp_lists = [inc.grp_q[inc.grp_ptr[c]:inc.grp_ptr[c + 1]] for c in sel]
    qc_ptr, qc_list = _invert(inc.n, *csr_from_lists(grp_lists))
    out = np.empty(inc.n, np.int64)
    pool_vecs = base[pool_ids]
    for s in range(0, inc.n, chunk):
        e = min(inc.n, s + chunk)
        dist = pairwise_angular_distance(train[s:e], pool_vecs)
        out[s:e] = _group_counts_kernel(dist, np.asarray(r_hat, dtype=np.float64), s, pos_of, total, qc_ptr, qc_list,
                                        sel_ptr, sel_vec.astype(np.int32), len(pool_ids))
    return out
