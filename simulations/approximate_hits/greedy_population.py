#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Offline cache population for approximate hits under a retention target.

Hit policy (hits-only running average): a set of queries may be served from
cache as long as the mean retention over the served (hit) queries is at least
the target T. Misses are answered by the backend (retention 1), so the overall
average is then also >= T.

Oracle setting: the backend is exact, and retention for a query is
|true top-k ids present in the cache| / k. Retention therefore only takes the
values 0/k, ..., k/k, so a set of queries is summarized by a histogram over
those k+1 values.

Greedy population: repeatedly add the candidate entry with the largest
(additional scoring-query hits) / (additional distinct cached vectors),
breaking ties by (additional retained top-k vectors) / (additional cached
vectors). The tie-break keeps greedy moving when no single candidate adds a
hit on its own.
"""

import copy
from typing import List, Optional

import numpy as np

EPS = 1e-9

#-------------------------------------------------------------------------------


def hits_from_histograms(hist: np.ndarray, target: float, k: int):
    """Best-case number of hits under the hits-only average-retention policy.

    Admitting the highest-retention queries first is optimal, so the admitted
    set is the largest top-retention prefix whose mean retention is >= target.

    Args:
        hist: (..., k+1) counts; hist[..., b] = number of queries whose
            retention count is b (retention b/k).
        target: retention target T in [0, 1].
        k: result count per query.

    Returns:
        (n_hits, kept) arrays of shape hist.shape[:-1]; kept is the summed
        retention count (units of 1/k) over admitted queries.
    """
    hist = np.asarray(hist, dtype=np.int64)
    batch_shape = hist.shape[:-1]
    n_hits = np.zeros(batch_shape, dtype=np.int64)
    kept = np.zeros(batch_shape, dtype=np.int64)
    still_open = np.ones(batch_shape, dtype=bool)
    tk = target * k
    for b in range(k, -1, -1):
        n_b = hist[..., b]
        if b >= tk - EPS:
            take = n_b
        else:
            # mean stays >= T while kept + m*b >= tk*(n_hits + m)
            slack = kept - tk * n_hits
            take = np.floor(slack / (tk - b) + EPS).astype(np.int64)
            take = np.clip(take, 0, n_b)
        take = np.where(still_open, take, 0)
        n_hits += take
        kept += take * b
        if b < tk - EPS:
            still_open &= take == n_b
    return n_hits, kept


def online_hits(retention_counts: np.ndarray, target: float, k: int):
    """Online hits-only policy: admit a query if the running mean retention
    over admitted queries, including it, stays >= target. Order-dependent.

    Returns (n_hits, kept).
    """
    tk = target * k
    n_hits = 0
    kept = 0
    for c in np.asarray(retention_counts).tolist():
        if kept + c >= tk * (n_hits + 1) - EPS:
            n_hits += 1
            kept += c
    return n_hits, kept


#-------------------------------------------------------------------------------


def _csr_by_key(keys: np.ndarray, values: np.ndarray, n_keys: int):
    """Group values by integer key. Returns (ptr, grouped_values)."""
    order = np.argsort(keys, kind="stable")
    ptr = np.zeros(n_keys + 1, dtype=np.int64)
    np.cumsum(np.bincount(keys, minlength=n_keys), out=ptr[1:])
    return ptr, values[order]


def _gather(ptr: np.ndarray, data: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Concatenate data[ptr[key]:ptr[key+1]] for each key, vectorized."""
    starts = ptr[keys]
    lens = ptr[keys + 1] - starts
    total = int(lens.sum())
    if total == 0:
        return np.empty(0, dtype=data.dtype)
    offsets = np.repeat(starts - np.cumsum(lens) + lens, lens)
    return data[offsets + np.arange(total)]


class _QueryTracker:
    """Retention counts for a fixed set of queries as vectors get cached.

    weights[i] is the multiplicity of query i (e.g. how often it repeats in a
    workload); histograms are weighted by it.
    """

    def __init__(self, topk: np.ndarray, n_db: int, k: int,
                 weights: Optional[np.ndarray] = None):
        self.k = k
        self.n = len(topk)
        self.weights = (np.ones(self.n, dtype=np.int64) if weights is None
                        else np.asarray(weights, dtype=np.int64))
        owners = np.repeat(np.arange(self.n), topk.shape[1])
        self.ptr, self.query_of = _csr_by_key(topk.ravel(), owners, n_db)
        self.counts = np.zeros(self.n, dtype=np.int64)

    def add_vectors(self, new_v: np.ndarray) -> None:
        np.add.at(self.counts, _gather(self.ptr, self.query_of, new_v), 1)

    def copy(self) -> "_QueryTracker":
        """Copy sharing the static index arrays; only counts are duplicated."""
        new = copy.copy(self)
        new.counts = self.counts.copy()
        return new

    def histogram(self) -> np.ndarray:
        return np.rint(np.bincount(self.counts, weights=self.weights,
                                   minlength=self.k + 1)).astype(np.int64)


#-------------------------------------------------------------------------------


class GreedyPopulator:
    """Incremental greedy cache population under the hits-only policy.

    Args:
        cand_vectors: list of int arrays; candidate j caches these vector ids.
        score_topk: (n_score, k) true top-k ids of the scoring (train) queries.
        n_db: number of database vectors.
        k: result count per query.
        target: retention target T.
        eval_topk: optional (n_eval, k) true top-k ids of held-out queries
            whose retention is tracked (not used for selection).
        score_weights, eval_weights: optional per-query multiplicities.
    """

    def __init__(
        self,
        cand_vectors: List[np.ndarray],
        score_topk: np.ndarray,
        n_db: int,
        k: int,
        target: float,
        eval_topk: Optional[np.ndarray] = None,
        score_weights: Optional[np.ndarray] = None,
        eval_weights: Optional[np.ndarray] = None,
    ):
        self.k = k
        self.target = target
        self.n_db = n_db
        self.n_cand = len(cand_vectors)
        self.n_score = len(score_topk)

        sizes = np.array([len(c) for c in cand_vectors], dtype=np.int64)
        self.cand_ptr = np.zeros(self.n_cand + 1, dtype=np.int64)
        np.cumsum(sizes, out=self.cand_ptr[1:])
        self.cand_v = np.concatenate(cand_vectors).astype(np.int64)
        cand_owner = np.repeat(np.arange(self.n_cand), sizes)

        # vector -> candidates containing it
        self.vc_ptr, self.vc_cand = _csr_by_key(self.cand_v, cand_owner, n_db)
        cands_per_vec = np.diff(self.vc_ptr)

        # (candidate, scoring query, shared vector) triples
        s_vec = score_topk.ravel().astype(np.int64)
        s_owner = np.repeat(np.arange(self.n_score), score_topk.shape[1])
        rep = cands_per_vec[s_vec]
        tri_s = np.repeat(s_owner, rep)
        tri_v = np.repeat(s_vec, rep)
        tri_j = _gather(self.vc_ptr, self.vc_cand, s_vec)

        pair_key = tri_j * self.n_score + tri_s
        uniq, tri_p = np.unique(pair_key, return_inverse=True)
        self.pair_j = uniq // self.n_score
        self.pair_s = uniq % self.n_score
        self.n_pairs = len(uniq)
        self.n_triples = len(tri_p)
        # overlap of candidate's UNCACHED vectors with the query's top-k
        self.pair_overlap = np.bincount(tri_p, minlength=self.n_pairs).astype(np.int64)
        # vector -> pairs it contributes to (for incremental decrements)
        self.vp_ptr, self.vp_pair = _csr_by_key(tri_v, tri_p.astype(np.int64), n_db)

        self.new_vectors = sizes.copy()  # uncached vectors per candidate
        self.cached = np.zeros(n_db, dtype=bool)
        self.cache_size = 0
        self.selected: List[int] = []

        self.score = _QueryTracker(score_topk, n_db, k, score_weights)
        self.eval = (_QueryTracker(eval_topk, n_db, k, eval_weights)
                     if eval_topk is not None else None)
        self.score_hits = 0
        self.last_predicted_hits = None

    def clone_for_target(self, target: float) -> "GreedyPopulator":
        """Fresh copy for another target that shares the static index arrays.

        Only the mutable state is duplicated, so forked workers can share one
        large precomputed structure (copy-on-write). Call on an unstepped
        instance.
        """
        new = copy.copy(self)
        new.target = target
        new.pair_overlap = self.pair_overlap.copy()
        new.new_vectors = self.new_vectors.copy()
        new.cached = self.cached.copy()
        new.selected = list(self.selected)
        new.score = self.score.copy()
        new.eval = self.eval.copy() if self.eval is not None else None
        new.score_hits = int(hits_from_histograms(
            new.score.histogram(), target, self.k)[0])
        new.last_predicted_hits = None
        return new

    def _evaluate_candidates(self, alive: np.ndarray):
        """Marginal hits and retained-vector gains for all candidates."""
        L = self.k + 1
        active = (self.pair_overlap > 0) & alive[self.pair_j]
        pj = self.pair_j[active]
        gain = self.pair_overlap[active]
        ps = self.pair_s[active]
        w = self.score.weights[ps]
        old = self.score.counts[ps]
        new = old + gain
        size = self.n_cand * L
        delta = np.rint(
            np.bincount(pj * L + new, weights=w, minlength=size)
            - np.bincount(pj * L + old, weights=w, minlength=size)
        ).astype(np.int64).reshape(self.n_cand, L)
        hist_new = self.score.histogram()[None, :] + delta
        hits_new, _ = hits_from_histograms(hist_new, self.target, self.k)
        d_hits = hits_new - self.score_hits
        d_kept = np.bincount(pj, weights=gain * w, minlength=self.n_cand)
        return d_hits, d_kept, hits_new

    def step(self, max_size: int) -> Optional[dict]:
        """Add the best candidate that fits within max_size. None if done."""
        alive = (self.new_vectors > 0) & (self.cache_size + self.new_vectors <= max_size)
        if not alive.any():
            return None
        d_hits, d_kept, hits_new = self._evaluate_candidates(alive)
        idx = np.flatnonzero(alive)
        if d_kept[idx].max() <= 0:
            return None  # nothing helps any scoring query

        cost = self.new_vectors[idx].astype(np.float64)
        primary = d_hits[idx] / cost
        tied = idx[primary == primary.max()]
        secondary = d_kept[tied] / self.new_vectors[tied]
        j = int(tied[int(np.argmax(secondary))])

        self.last_predicted_hits = int(hits_new[j])
        vecs = self.cand_v[self.cand_ptr[j]:self.cand_ptr[j + 1]]
        new_v = vecs[~self.cached[vecs]]
        self._cache_vectors(new_v)
        self.selected.append(j)
        return {
            "candidate": j,
            "added_vectors": int(len(new_v)),
            "cache_size": self.cache_size,
            "score_hits": self.score_hits,
        }

    def _cache_vectors(self, new_v: np.ndarray) -> None:
        self.cached[new_v] = True
        self.cache_size += len(new_v)
        self.new_vectors -= np.bincount(
            _gather(self.vc_ptr, self.vc_cand, new_v), minlength=self.n_cand)
        np.subtract.at(self.pair_overlap, _gather(self.vp_ptr, self.vp_pair, new_v), 1)
        self.score.add_vectors(new_v)
        self.score_hits = int(hits_from_histograms(
            self.score.histogram(), self.target, self.k)[0])
        if self.eval is not None:
            self.eval.add_vectors(new_v)
