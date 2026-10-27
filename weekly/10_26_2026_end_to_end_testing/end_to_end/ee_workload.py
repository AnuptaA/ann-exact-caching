#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Workloads and exact ground truth for the end-to-end tests.

lambda is the local fraction: A = round(lambda * n / 50) anchors from the official query pool, each with 50 train
and 50 test independent perturbations (angle uniform in [0, 10] degrees, random direction orthogonal to the
anchor). The remaining n - 50 A train and test queries are official pool queries drawn without replacement,
excluding the anchors, with train and test disjoint. Anchors are nested across lambda for a seed (and so are
their copies, each anchor has its own random stream); the random component is drawn per lambda."""

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import ee_config as cfg
from data_prep import load_base_pool, load_official_query_pool
from ground_truth import batched_topk_full_skip_self
from perturbation import random_tangent_direction, tangent_perturb
from utils.base.distance_metrics import pairwise_angular_distance

LOG = logging.getLogger("ee_workload")

#-------------------------------------------------------------------------------

@dataclass
class Workload:
    dataset: str
    seed: int
    lam: float
    n: int
    K: int
    n_anchors: int
    train: np.ndarray            # (n, d) float64 unit vectors
    test: np.ndarray
    train_anchor: np.ndarray     # anchor rank per query, -1 for random queries
    test_anchor: np.ndarray
    train_angle_deg: np.ndarray  # realized angle to the anchor, nan for random queries
    test_angle_deg: np.ndarray
    anchor_source: np.ndarray    # official pool index of each anchor
    train_top_ids: np.ndarray    # (n, K) int32
    train_top_dist: np.ndarray   # (n, K) float64 ascending
    test_top_ids: np.ndarray
    test_top_dist: np.ndarray
    prefix_ids: np.ndarray = None   # (n, P) int32, nearest P DB vectors of each train query, ascending distance
    prefix_dist: np.ndarray = None  # (n, P) float64
    tt_dist: np.ndarray = None      # (n, n) float64 train-train distances, zero diagonal
    tt_order: np.ndarray = None     # (n, n - 1) int32, other train queries by increasing distance (ties by index)

    @property
    def theta_train(self) -> np.ndarray:
        return self.train_top_dist[:, self.K - 1]

    @property
    def theta_test(self) -> np.ndarray:
        return self.test_top_dist[:, self.K - 1]

#-------------------------------------------------------------------------------
# Exact search

def exact_neighbors(queries: np.ndarray, base: np.ndarray, count: int, chunk: int = cfg.GT_CHUNK) -> tuple:
    """Nearest `count` base vectors of every query by brute force: (ids int32, distances float64), ascending
    distance, ties by id."""
    n = len(queries)
    ids = np.empty((n, count), dtype=np.int32)
    dist = np.empty((n, count), dtype=np.float64)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        rows = pairwise_angular_distance(queries[s:e], base)
        part = np.argpartition(rows, count - 1, axis=1)[:, :count]
        part.sort(axis=1)
        part_d = np.take_along_axis(rows, part, axis=1)
        order = np.argsort(part_d, axis=1, kind="stable")
        ids[s:e] = np.take_along_axis(part, order, axis=1)
        dist[s:e] = np.take_along_axis(part_d, order, axis=1)
    return ids, dist

def train_train(train: np.ndarray) -> tuple:
    """Train-train distance matrix (zero diagonal) and each query's other train queries by increasing distance."""
    d = pairwise_angular_distance(train, train)
    np.fill_diagonal(d, 0.0)
    masked = d.copy()
    np.fill_diagonal(masked, np.inf)
    order = np.argsort(masked, axis=1, kind="stable")[:, :-1].astype(np.int32)
    return d, order

#-------------------------------------------------------------------------------
# Queries

def _local_copies(anchor: np.ndarray, count: int, rng: np.random.Generator, max_angle_deg: float) -> tuple:
    out = np.empty((count, anchor.shape[0]))
    angles = np.empty(count)
    for i in range(count):
        direction = random_tangent_direction(anchor, rng)
        alpha = rng.uniform(0.0, max_angle_deg)
        q, realized = tangent_perturb(anchor, direction, alpha / 180.0)
        out[i], angles[i] = q, realized * 180.0
    return out, angles

def build_queries(pool: np.ndarray, seed: int, lam: float, n: int, per_anchor: int,
                  max_angle_deg: float = cfg.MAX_ANGLE_DEG, lam_index: int = None) -> dict:
    """Train and test queries (train first by anchor, then random). Raises if the pool is too small."""
    A = int(round(lam * n / per_anchor))
    n_local = A * per_anchor
    n_random = n - n_local
    anchor_perm = np.random.default_rng([seed, 11]).permutation(len(pool))
    anchor_source = anchor_perm[:A]
    train_parts, test_parts, tr_anchor, te_anchor, tr_ang, te_ang = [], [], [], [], [], []
    for rank, src in enumerate(anchor_source):
        rng = np.random.default_rng([seed, 12, rank])
        tr, a_tr = _local_copies(pool[src], per_anchor, rng, max_angle_deg)
        te, a_te = _local_copies(pool[src], per_anchor, rng, max_angle_deg)
        train_parts.append(tr)
        test_parts.append(te)
        tr_anchor += [rank] * per_anchor
        te_anchor += [rank] * per_anchor
        tr_ang.append(a_tr)
        te_ang.append(a_te)
    random_src = np.empty(0, dtype=np.int64)
    if n_random > 0:
        eligible = np.setdiff1d(np.arange(len(pool)), anchor_source)
        if len(eligible) < 2 * n_random:
            raise ValueError(f"official pool too small: {len(eligible)} eligible queries for {2 * n_random} random "
                             f"train and test queries (seed {seed}, lambda {lam})")
        idx = lam_index if lam_index is not None else int(round(lam * 100))
        pick = np.random.default_rng([seed, 13, idx]).permutation(eligible)[:2 * n_random]
        train_parts.append(pool[pick[:n_random]])
        test_parts.append(pool[pick[n_random:]])
        tr_anchor += [-1] * n_random
        te_anchor += [-1] * n_random
        tr_ang.append(np.full(n_random, np.nan))
        te_ang.append(np.full(n_random, np.nan))
        random_src = pick
    return dict(train=np.vstack(train_parts), test=np.vstack(test_parts), train_anchor=np.array(tr_anchor),
                test_anchor=np.array(te_anchor), train_angle_deg=np.concatenate(tr_ang),
                test_angle_deg=np.concatenate(te_ang), anchor_source=anchor_source, random_source=random_src,
                n_anchors=A)

def check_queries(train: np.ndarray, test: np.ndarray, base: np.ndarray, tol: float = 1e-9) -> dict:
    """Asserts that no query equals a base vector or another query; returns the minimum distances."""
    both = np.vstack([train, test])
    nn = np.empty(len(both))
    for s in range(0, len(both), 500):
        nn[s:s + 500] = np.arccos(np.clip((both[s:s + 500] @ base.T).max(axis=1), -1, 1)) / np.pi
    qq = pairwise_angular_distance(both, both)
    np.fill_diagonal(qq, np.inf)
    out = dict(min_dist_to_base=float(nn.min()), min_query_distance=float(qq.min()))
    assert out["min_dist_to_base"] > tol, f"a query coincides with a base vector: {out}"
    assert out["min_query_distance"] > tol, f"duplicate queries: {out}"
    return out

#-------------------------------------------------------------------------------

def build_workload(dataset: str, base: np.ndarray, pool: np.ndarray, seed: int, lam: float, n: int = cfg.N_QUERIES,
                   K: int = cfg.K, per_anchor: int = cfg.QUERIES_PER_ANCHOR, prefix_len: int = cfg.PREFIX_LEN,
                   max_angle_deg: float = cfg.MAX_ANGLE_DEG) -> Workload:
    q = build_queries(pool, seed, lam, n, per_anchor, max_angle_deg)
    check = check_queries(q["train"], q["test"], base)
    LOG.info(f"{dataset} seed {seed} lambda {lam}: {q['n_anchors']} anchors, min distance to base "
             f"{check['min_dist_to_base']:.4g}, min query distance {check['min_query_distance']:.4g}")
    prefix_ids, prefix_dist = exact_neighbors(q["train"], base, prefix_len)
    test_ids, test_dist = exact_neighbors(q["test"], base, K)
    tt_dist, tt_order = train_train(q["train"])
    return Workload(dataset=dataset, seed=seed, lam=lam, n=n, K=K, n_anchors=q["n_anchors"], train=q["train"],
                    test=q["test"], train_anchor=q["train_anchor"], test_anchor=q["test_anchor"],
                    train_angle_deg=q["train_angle_deg"], test_angle_deg=q["test_angle_deg"],
                    anchor_source=q["anchor_source"], train_top_ids=prefix_ids[:, :K].copy(),
                    train_top_dist=prefix_dist[:, :K].copy(), test_top_ids=test_ids, test_top_dist=test_dist,
                    prefix_ids=prefix_ids, prefix_dist=prefix_dist, tt_dist=tt_dist, tt_order=tt_order)

#-------------------------------------------------------------------------------
# Cache. Only the queries and the top-K lists are stored (small); the 5,001-nearest prefix and the train-train
# matrices are recomputed on load (a few seconds) because they would take hundreds of MB per workload.

CACHE_FIELDS = ["train", "test", "train_anchor", "test_anchor", "train_angle_deg", "test_angle_deg", "anchor_source",
                "train_top_ids", "train_top_dist", "test_top_ids", "test_top_dist"]

def cache_path(dataset: str, seed: int, lam: float, n: int = cfg.N_QUERIES, K: int = cfg.K,
               directory: Path = cfg.GROUND_TRUTH_DIR) -> Path:
    return Path(directory) / f"{dataset}_lam{lam:g}_seed{seed}_n{n}_K{K}.npz"

def save_workload(w: Workload, path: Path) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    meta = json.dumps(dict(dataset=w.dataset, seed=w.seed, lam=w.lam, n=w.n, K=w.K, n_anchors=w.n_anchors,
                           prefix_len=int(w.prefix_ids.shape[1]) if w.prefix_ids is not None else cfg.PREFIX_LEN))
    with open(tmp, "wb") as f:
        np.savez(f, meta=np.array(meta), **{k: getattr(w, k) for k in CACHE_FIELDS})
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def attach_derived(w: Workload, base: np.ndarray, prefix_len: int) -> Workload:
    w.prefix_ids, w.prefix_dist = exact_neighbors(w.train, base, prefix_len)
    assert np.array_equal(w.prefix_ids[:, :w.K], w.train_top_ids), "stored top-K disagrees with the recomputed prefix"
    w.tt_dist, w.tt_order = train_train(w.train)
    return w

def load_workload(path: Path, base: np.ndarray) -> Workload:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    w = Workload(dataset=meta["dataset"], seed=meta["seed"], lam=meta["lam"], n=meta["n"], K=meta["K"],
                 n_anchors=meta["n_anchors"], **{k: z[k] for k in CACHE_FIELDS})
    return attach_derived(w, base, meta["prefix_len"])

def dedupe_pool(pool: np.ndarray) -> tuple:
    """Official pools can contain identical query vectors. Keep the first occurrence of each: returns
    (unique pool in original order, original index of each kept row)."""
    first, keep = {}, []
    for i in range(len(pool)):
        key = pool[i].tobytes()
        if key not in first:
            first[key] = i
            keep.append(i)
    keep = np.array(keep, dtype=np.int64)
    return pool[keep], keep

def get_workload(dataset: str, seed: int, lam: float, base: np.ndarray = None, pool: np.ndarray = None,
                 n: int = cfg.N_QUERIES, K: int = cfg.K, directory: Path = cfg.GROUND_TRUTH_DIR) -> Workload:
    """Load the cached workload, or build it from the dataset's base and de-duplicated official query pool and
    cache it. `anchor_source` indexes the original (not de-duplicated) official pool."""
    base = load_base_pool(dataset) if base is None else base
    assert len(base) == cfg.N_DB, f"{dataset}: expected {cfg.N_DB} base vectors, got {len(base)}"
    path = cache_path(dataset, seed, lam, n, K, directory)
    if path.exists():
        w = load_workload(path, base)
        assert (w.dataset, w.seed, w.lam, w.n, w.K) == (dataset, seed, lam, n, K), f"{path.name}: stale cache"
        return w
    pool = load_official_query_pool(dataset) if pool is None else pool
    unique_pool, kept = dedupe_pool(pool)
    if len(unique_pool) < len(pool):
        LOG.warning(f"{dataset}: official pool has {len(pool) - len(unique_pool)} duplicate vectors of {len(pool)}; "
                    f"drawing from the {len(unique_pool)} distinct ones")
    w = build_workload(dataset, base, unique_pool, seed, lam, n, K)
    w.anchor_source = kept[w.anchor_source]
    save_workload(w, path)
    return w

#-------------------------------------------------------------------------------
# Theta of database points (K-th nearest OTHER vector), the labels of sampled DB points in the LSH arm

def db_theta(dataset: str, base: np.ndarray, K: int = cfg.K, directory: Path = cfg.GROUND_TRUTH_DIR,
             batch_size: int = 256) -> np.ndarray:
    digest = hashlib.sha1(np.ascontiguousarray(base).tobytes()).hexdigest()
    path = Path(directory) / f"{dataset}_db_theta_K{K}.npz"
    if path.exists():
        z = np.load(path)
        assert str(z["digest"]) == digest and int(z["K"]) == K, f"{path.name}: stale cache"
        return z["theta"]
    _, dist = batched_topk_full_skip_self(base, base, np.arange(len(base)), K, batch_size=batch_size)
    theta = dist[:, K - 1].astype(np.float64)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, theta=theta, digest=digest, K=K)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return theta
