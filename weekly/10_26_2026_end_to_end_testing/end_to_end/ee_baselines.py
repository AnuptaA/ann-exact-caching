#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Baseline pools: no cache, Popularity, Random Own and threshold_baseline.

Every pool carries the train recall counts that feed the HP3 cap: leave-group-out for candidate based pools (the source
query is excluded from its own candidate), leave-one-out for Popularity (the pool is rebuilt without the query's own
top-K set, exactly).

threshold_baseline is the old pipeline population policy (population.py of the 09_22 hit-policy pipeline, kept in a local
tarball), adapted to K = 10 and this incidence structure: for every train query a, the 20 other train queries j with the
smallest R(a, j) = dist(a, j) + theta_ceil(0.8 K)(j) give 20 candidates with radius R_i; a candidate's vectors are all
database vectors within R_i of a, its utility numerator is the number of other train queries j with R(a, j) <= R_i, and
the greedy repeatedly takes the best numerator / (new vectors) ratio until nothing fits. Candidates with no new vectors
are dropped, candidates larger than the budget are never built. Ties break by smaller cost, then smaller candidate id."""

import math
from dataclasses import dataclass

import numpy as np

import ee_config as cfg
import ee_estimators as est
import ee_incidence as ei

T_POP = 0.8
N_NEIGHBORS = 20

@dataclass
class Pool:
    name: str
    pool_mask: np.ndarray
    train_true_counts: np.ndarray   # (n,) recall counts of the train queries (leave-group-out / leave-one-out)
    selected: np.ndarray = None     # candidate ids when the pool was built from candidates
    n_selected: int = 0
    inc: ei.Incidence = None        # incidence the candidate ids refer to

#-------------------------------------------------------------------------------

def no_cache(w, n_db: int = cfg.N_DB) -> Pool:
    return Pool("no_cache", np.zeros(n_db, dtype=bool), np.zeros(w.n, dtype=np.int64))

#-------------------------------------------------------------------------------
# Popularity
def popularity_order(counts: np.ndarray) -> np.ndarray:
    """Vectors with count >= 1 by decreasing count, ties by id."""
    ids = np.flatnonzero(counts > 0)
    return ids[np.lexsort((ids, -counts[ids]))]

def _loo_pool(counts: np.ndarray, head: np.ndarray, own: np.ndarray, budget: int) -> np.ndarray:
    """Popularity pool rebuilt without the query's own top-K set."""
    c = counts[head] - np.isin(head, own)
    keep = c > 0
    h, cq = head[keep], c[keep]
    return h[np.lexsort((h, -cq))][:budget]

def popularity_lsh_train_counts(w, budget: int, r_hat: np.ndarray, base: np.ndarray, n_db: int = cfg.N_DB) -> np.ndarray:
    """LSH-arm estimated recall counts of the train queries on the Popularity pool: pool vectors within r_hat[q] of query q,
    with the pool rebuilt without q's own top-K set."""
    from utils.base.distance_metrics import pairwise_angular_distance
    counts = np.bincount(w.train_top_ids.ravel(), minlength=n_db).astype(np.int64)
    order = popularity_order(counts)
    head = order[:budget + 2 * w.K]
    out = np.empty(w.n, dtype=np.int64)
    for q in range(w.n):
        pool_q = _loo_pool(counts, head, w.train_top_ids[q], budget)
        out[q] = int((pairwise_angular_distance(w.train[q:q + 1], base[pool_q])[0] <= r_hat[q]).sum())
    return out

def popularity(w, budget: int, n_db: int = cfg.N_DB) -> Pool:
    """The `budget` vectors contained in the most train top-K sets (ties by id); train counts are leave-one-out."""
    top = w.train_top_ids
    counts = np.bincount(top.ravel(), minlength=n_db).astype(np.int64)
    order = popularity_order(counts)
    pool = order[:budget]
    mask = np.zeros(n_db, dtype=bool)
    mask[pool] = True
    # leave-one-out: rebuild the pool without query q's own top-K set; only the first budget + 2K vectors can matter
    head = order[:budget + 2 * w.K]
    loo = np.empty(w.n, dtype=np.int64)
    for q in range(w.n):
        loo[q] = np.isin(top[q], _loo_pool(counts, head, top[q], budget)).sum()
    return Pool("popularity", mask, loo)

#-------------------------------------------------------------------------------
# Random Own

def random_own(w, budget: int, seed: int, own_inc: ei.Incidence = None, own_idx: ei.Index = None) -> Pool:
    """Random train queries' top-K sets added in random order while they fit; the source is excluded from its own
    candidate in the leave-group-out recall."""
    if own_inc is None:
        import ee_proposals as ep
        own_inc, _ = ep.make_incidence(w, "own")
    own_idx = ei.build_index(own_inc) if own_idx is None else own_idx
    run = ei._Run(own_inc, own_idx, 0.9)
    remaining = list(np.random.default_rng([seed, 41]).permutation(w.n))
    changed = True
    while changed:  # costs only fall as the pool grows, so repeat until no remaining candidate fits
        changed, still = False, []
        for c in remaining:
            if run.cost[c] <= budget - run.pool_size:
                run.apply(int(c))
                changed = True
            else:
                still.append(c)
        remaining = still
    return Pool("random_own", run.in_pool.copy(), run.covered.astype(np.int64), np.array(run.picks), len(run.picks), own_inc)

#-------------------------------------------------------------------------------
# threshold_baseline

def threshold_candidates(w, budget: int, T_pop: float = T_POP, n_neighbors: int = N_NEIGHBORS) -> dict:
    assert budget < w.prefix_ids.shape[1], "the nearest-vector prefix must be longer than the budget"
    col = math.ceil(T_pop * w.K) - 1
    theta_pop = w.train_top_dist[:, col]
    R, G, source, numer = [], [], [], []
    others_all = np.arange(w.n)
    for a in range(w.n):
        others = others_all[others_all != a]
        R_all = w.tt_dist[a, others] + theta_pop[others]
        order = np.argsort(R_all, kind="stable")
        sorted_R = R_all[order]
        for r_i in sorted_R[:n_neighbors]:
            count = int(np.searchsorted(w.prefix_dist[a], r_i, side="right"))
            if count > budget:
                continue
            R.append(w.prefix_ids[a, :count].copy())
            G.append([a])
            source.append(a)
            numer.append(int(np.searchsorted(sorted_R, r_i, side="right")))
    return dict(R=R, G=G, source=np.array(source, dtype=np.int32), numer=np.array(numer, dtype=np.int64))

def threshold_greedy(cands: dict, budget: int, n_db: int) -> np.ndarray:
    """Candidate ids in selection order: best numerator / new-vector ratio first."""
    ptr, vec = ei.csr_from_lists(cands["R"])
    vc_ptr, vc_cand = ei._invert(n_db, ptr, vec)
    cost = np.diff(ptr).astype(np.int64)
    numer = cands["numer"].astype(np.float64)
    alive = np.ones(len(cost), dtype=bool)
    in_pool = np.zeros(n_db, dtype=bool)
    picks, size = [], 0
    while True:
        alive &= cost > 0                       # nothing new to offer: dropped for good
        ok = alive & (cost <= budget - size)
        if not ok.any():
            break
        ratio = np.where(ok, numer / np.maximum(cost, 1), -1.0)
        best_ratio = ratio.max()
        tied = np.flatnonzero(ratio == best_ratio)
        c = int(tied[np.argmin(cost[tied])])    # smaller cost, then smaller id (argmin keeps the first)
        picks.append(c)
        alive[c] = False
        new = vec[ptr[c]:ptr[c + 1]]
        new = new[~in_pool[new]]
        in_pool[new] = True
        size += len(new)
        for v in new:
            cost[vc_cand[vc_ptr[v]:vc_ptr[v + 1]]] -= 1
    return np.array(picks, dtype=np.int64)

def threshold_baseline(w, budget: int, n_db: int = cfg.N_DB) -> Pool:
    cands = threshold_candidates(w, budget)
    picks = threshold_greedy(cands, budget, n_db)
    cptr, cvec = ei.csr_from_lists(cands["R"])
    gptr, gq = ei.csr_from_lists(cands["G"])
    inc = ei.Incidence(w.n, w.K, n_db, w.train_top_ids.astype(np.int32), cptr, cvec, gptr, gq, cands["source"])
    run = ei.replay(inc, ei.build_index(inc), picks)
    assert run.pool_size <= budget
    return Pool("threshold_baseline", run.in_pool.copy(), run.covered.astype(np.int64), picks, len(picks), inc)

#-------------------------------------------------------------------------------

def pool_from_result(name: str, result: ei.Result, inc: ei.Incidence = None) -> Pool:
    """Wrap a greedy result (train counts are the leave-group-out slot counters)."""
    return Pool(name, result.pool_mask, result.covered.astype(np.int64), result.selected, len(result.selected), inc)

def test_true_counts(w, pool: Pool) -> np.ndarray:
    return est.true_recall_counts(pool.pool_mask, w.test_top_ids)
