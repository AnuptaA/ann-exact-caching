#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Radius arms and estimated recall.

Oracle arm: r_hat = theta, estimated recall = |pool intersect top-K ids| / K (set intersection).

LSH arm: the Hamming-LSH estimator math of radius_eval_tables/rt_engine.py, unchanged (plain arithmetic mean of
theta, no logs). One matrix W of 2048 x d entries in {+1, -1} per seed, code = sign(W x) with 0 counted as +1,
table t of a configuration (L, m) uses rows [t*m, (t+1)*m). Labeled points are all train queries (label = exact
theta) plus optionally S sampled database vectors (label = K-th nearest OTHER vector). Test queries are never
inserted. Per table the store keeps, per distinct code, the float64 sum of theta and the integer count. A query
takes, in each table, codes in increasing m-bit Hamming distance until the cumulative count first reaches nu,
including every code at that same distance (ties); the table value is sum / count and r_hat is the plain mean of
the L table values. For a train query leave-one-out subtracts its own theta and count from its code in every table.

Estimated recall of the LSH arm is (number of pool vectors within r_hat of the query) / K, unclipped. Counts are
returned as integers, so recall units of 0.05 stay exact."""

from dataclasses import dataclass

import numpy as np

import ee_config as cfg
from rt_engine import make_hyperplanes, sign_codes
from utils.base.distance_metrics import pairwise_angular_distance

#-------------------------------------------------------------------------------
# Recall counts

def pool_mask(pool_ids, n_db: int = cfg.N_DB) -> np.ndarray:
    mask = np.zeros(n_db, dtype=bool)
    mask[np.asarray(pool_ids, dtype=np.int64)] = True
    return mask

def true_recall_counts(mask: np.ndarray, top_ids: np.ndarray) -> np.ndarray:
    """|pool intersect top-K ids| per query, by set membership (not by comparing distances)."""
    return mask[top_ids].sum(axis=1).astype(np.int64)

def vectors_within_counts(queries: np.ndarray, r_hat: np.ndarray, pool_vectors: np.ndarray,
                          chunk: int = 500) -> np.ndarray:
    """Number of pool vectors with exact distance to the query <= r_hat (unclipped, can exceed K)."""
    out = np.empty(len(queries), dtype=np.int64)
    for s in range(0, len(queries), chunk):
        e = min(len(queries), s + chunk)
        d = pairwise_angular_distance(queries[s:e], pool_vectors)
        out[s:e] = (d <= r_hat[s:e, None]).sum(axis=1)
    return out

#-------------------------------------------------------------------------------
# LSH store

@dataclass
class TableStore:
    rows: tuple          # (row_lo, m) rows of W used by this table
    codes_pm: np.ndarray # (D, m) float32 +/-1 distinct codes
    sums: np.ndarray     # (D,) float64 sum of theta per distinct code
    counts: np.ndarray   # (D,) int64 number of labeled points per distinct code
    inverse: np.ndarray  # (n_labeled,) distinct code index of every labeled point
    labels: np.ndarray   # (n_labeled,) float64 theta of every labeled point

def build_table(W: np.ndarray, labeled: np.ndarray, labels: np.ndarray, row_lo: int, m: int) -> TableStore:
    codes = sign_codes(labeled, W[row_lo:row_lo + m])
    packed = np.packbits(codes > 0, axis=1)
    _, first, inverse = np.unique(packed, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    return TableStore((row_lo, m), codes[first].astype(np.float32),
                      np.bincount(inverse, weights=labels, minlength=len(first)),
                      np.bincount(inverse, minlength=len(first)).astype(np.int64), inverse, labels)

def table_value(store: TableStore, W: np.ndarray, queries: np.ndarray, nu: int, own: np.ndarray = None) -> np.ndarray:
    """Table value per query. `own` gives, for train queries, the labeled index of the query itself (-1 for none),
    which is removed from its code before estimating."""
    row_lo, m = store.rows
    q_codes = sign_codes(queries, W[row_lo:row_lo + m]).astype(np.float32)
    n_rows = len(queries)
    ham = np.rint((m - q_codes @ store.codes_pm.T) * 0.5).astype(np.int64)
    base = (np.arange(n_rows, dtype=np.int64) * (m + 1))[:, None]
    idx = (ham + base).ravel()
    cnt = np.bincount(idx, weights=np.broadcast_to(store.counts, ham.shape).ravel(), minlength=n_rows * (m + 1))
    sm = np.bincount(idx, weights=np.broadcast_to(store.sums, ham.shape).ravel(), minlength=n_rows * (m + 1))
    cnt, sm = cnt.reshape(n_rows, m + 1), sm.reshape(n_rows, m + 1)
    if own is not None:
        rows = np.flatnonzero(own >= 0)
        j = store.inverse[own[rows]]
        assert (ham[rows, j] == 0).all(), "a labeled train query must sit at Hamming distance 0 from itself"
        cnt[rows, 0] -= 1
        sm[rows, 0] -= store.labels[own[rows]]
    cc, cs = np.cumsum(cnt, axis=1), np.cumsum(sm, axis=1)
    cut = np.minimum((cc < nu).sum(axis=1), m)
    ar = np.arange(n_rows)
    assert (cc[ar, cut] > 0).all(), "empty neighborhood"
    return cs[ar, cut] / cc[ar, cut]

def table_values(store: TableStore, W: np.ndarray, queries: np.ndarray, nus: list, own: np.ndarray = None) -> np.ndarray:
    """Table values for several nu at once, (len(nus), rows): one Hamming pass, one histogram, many cutoffs."""
    row_lo, m = store.rows
    q_codes = sign_codes(queries, W[row_lo:row_lo + m]).astype(np.float32)
    n_rows = len(queries)
    ham = np.rint((m - q_codes @ store.codes_pm.T) * 0.5).astype(np.int64)
    idx = (ham + (np.arange(n_rows, dtype=np.int64) * (m + 1))[:, None]).ravel()
    cnt = np.bincount(idx, weights=np.broadcast_to(store.counts, ham.shape).ravel(), minlength=n_rows * (m + 1))
    sm = np.bincount(idx, weights=np.broadcast_to(store.sums, ham.shape).ravel(), minlength=n_rows * (m + 1))
    cnt, sm = cnt.reshape(n_rows, m + 1), sm.reshape(n_rows, m + 1)
    if own is not None:
        rows = np.flatnonzero(own >= 0)
        assert (ham[rows, store.inverse[own[rows]]] == 0).all()
        cnt[rows, 0] -= 1
        sm[rows, 0] -= store.labels[own[rows]]
    cc, cs = np.cumsum(cnt, axis=1), np.cumsum(sm, axis=1)
    ar = np.arange(n_rows)
    out = np.empty((len(nus), n_rows))
    for j, nu in enumerate(nus):
        cut = np.minimum((cc < nu).sum(axis=1), m)
        assert (cc[ar, cut] > 0).all(), "empty neighborhood"
        out[j] = cs[ar, cut] / cc[ar, cut]
    return out

class LSHStore:
    """L tables of m bits over one labeled set. Test queries are never inserted."""

    def __init__(self, W: np.ndarray, L: int, m: int, labeled: np.ndarray, labels: np.ndarray):
        assert L * m <= W.shape[0]
        self.W, self.L, self.m = W, L, m
        self.labels = np.asarray(labels, dtype=np.float64)
        self.tables = [build_table(W, labeled, self.labels, t * m, m) for t in range(L)]

    def estimate_multi(self, queries: np.ndarray, nus: list, own: np.ndarray = None, chunk: int = 250) -> np.ndarray:
        """r_hat for several nu at once, (len(nus), n_queries); equals `estimate` per nu."""
        out = np.empty((len(nus), len(queries)))
        for s in range(0, len(queries), chunk):
            e = min(len(queries), s + chunk)
            own_c = None if own is None else own[s:e]
            vals = np.stack([table_values(t, self.W, queries[s:e], nus, own_c) for t in self.tables])
            out[:, s:e] = vals.mean(axis=0)
        return out

    def estimate(self, queries: np.ndarray, nu: int, own: np.ndarray = None, chunk: int = 500) -> np.ndarray:
        out = np.empty(len(queries))
        for s in range(0, len(queries), chunk):
            e = min(len(queries), s + chunk)
            own_c = None if own is None else own[s:e]
            vals = [table_value(t, self.W, queries[s:e], nu, own_c) for t in self.tables]
            out[s:e] = np.mean(vals, axis=0)
        return out

#-------------------------------------------------------------------------------
# Arms

class OracleArm:
    name = "oracle"

    def __init__(self, workload):
        self.w = workload

    def radius(self, split: str) -> np.ndarray:
        return self.w.theta_train if split == "train" else self.w.theta_test

    def recall_counts(self, split: str, mask: np.ndarray, base: np.ndarray = None) -> np.ndarray:
        ids = self.w.train_top_ids if split == "train" else self.w.test_top_ids
        return true_recall_counts(mask, ids)

class LSHArm:
    """Hamming-LSH radius arm for one (L, m, nu, S) over one seed."""
    name = "lsh"

    def __init__(self, workload, base: np.ndarray, seed: int, L: int, m: int, nu: int, sample_size: int = 0,
                 db_theta: np.ndarray = None):
        self.w, self.base, self.nu = workload, base, nu
        self.W = make_hyperplanes(seed, base.shape[1])
        labeled, labels = [workload.train], [workload.theta_train]
        self.sample_ids = np.empty(0, dtype=np.int64)
        if sample_size:
            assert db_theta is not None, "sampled database labels need db_theta"
            self.sample_ids = np.random.default_rng([seed, 21]).permutation(len(base))[:sample_size]
            labeled.append(base[self.sample_ids])
            labels.append(db_theta[self.sample_ids])
        self.store = LSHStore(self.W, L, m, np.vstack(labeled), np.concatenate(labels))
        self.n_train = len(workload.train)
        self._cache = {}

    def radius(self, split: str) -> np.ndarray:
        """r_hat per query; train queries use leave-one-out."""
        if split not in self._cache:
            if split == "train":
                self._cache[split] = self.store.estimate(self.w.train, self.nu, own=np.arange(self.n_train))
            else:
                self._cache[split] = self.store.estimate(self.w.test, self.nu)
        return self._cache[split]

    def recall_counts(self, split: str, mask: np.ndarray, base: np.ndarray = None) -> np.ndarray:
        """Estimated recall counts against the whole pool (the group-aware train version is built in the
        evaluation module)."""
        base = self.base if base is None else base
        queries = self.w.train if split == "train" else self.w.test
        return vectors_within_counts(queries, self.radius(split), base[np.flatnonzero(mask)])
