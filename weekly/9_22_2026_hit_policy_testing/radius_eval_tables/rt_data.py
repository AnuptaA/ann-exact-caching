#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Databases, train/test query sets for stages 0 to 4, and the per-database theta
cache."""

import hashlib
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

import rt_config
from data_prep import load_base_pool, load_official_query_pool
from ground_truth import batched_topk_full_skip_self
from perturbation import random_tangent_direction, tangent_perturb
from utils.base.distance_metrics import pairwise_angular_distance

#-------------------------------------------------------------------------------
# Samplers

def sample_uniform_sphere(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """n unit vectors uniform on the sphere, stored as float32 then returned as renormalized float64."""
    v = rng.standard_normal((n, d))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v = v.astype(np.float32).astype(np.float64)
    return v / np.linalg.norm(v, axis=1, keepdims=True)

def sample_permutation(N: int, seed: int) -> np.ndarray:
    return np.random.default_rng([seed, 0]).permutation(N)

def build_database(dataset: str, stage: int) -> np.ndarray:
    if stage <= 2:
        d = rt_config.DIM[dataset]
        return sample_uniform_sphere(rt_config.N_DB, d, np.random.default_rng([rt_config.DATA_SEED, d]))
    return load_base_pool(dataset)

#-------------------------------------------------------------------------------
# Validity helpers

def nn_distance(queries: np.ndarray, database: np.ndarray, chunk: int = 500) -> np.ndarray:
    """Distance from each query to its nearest database vector."""
    out = np.empty(len(queries))
    for s in range(0, len(queries), chunk):
        max_dot = (queries[s:s + chunk] @ database.T).max(axis=1)
        out[s:s + chunk] = np.arccos(np.clip(max_dot, -1.0, 1.0)) / np.pi
    return out

def duplicate_of_earlier(X: np.ndarray, tol: float = rt_config.DUP_TOL, chunk: int = 500) -> np.ndarray:
    """True for rows within tol of an earlier row of X."""
    n = len(X)
    dup = np.zeros(n, dtype=bool)
    cols = np.arange(n)[None, :]
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        dist = np.arccos(np.clip(X[s:e] @ X.T, -1.0, 1.0)) / np.pi
        dist[cols >= np.arange(s, e)[:, None]] = np.inf
        dup[s:e] = dist.min(axis=1) <= tol
    return dup

def select_valid(candidates: np.ndarray, database: np.ndarray, need: int, logger: logging.Logger,
                 label: str) -> np.ndarray:
    """Indices of the first `need` candidates that are not within tol of a database vector or of an earlier
    candidate. Rejects are logged."""
    near_db = nn_distance(candidates, database) <= rt_config.DUP_TOL
    dup = duplicate_of_earlier(candidates)
    ok = np.flatnonzero(~near_db & ~dup)
    logger.info(f"{label}: {len(candidates)} candidates, rejected {int(near_db.sum())} zero-distance to database, "
                f"{int(dup.sum())} duplicates of an earlier candidate")
    assert len(ok) >= need, f"{label}: only {len(ok)} valid candidates for {need} needed"
    return ok[:need]

#-------------------------------------------------------------------------------
# Query sets

def _meta(role: str, n: int, anchor_id=None, pert_dist=None, source=None) -> pd.DataFrame:
    return pd.DataFrame(dict(
        role=role, query_id=np.arange(n),
        anchor_id=-1 if anchor_id is None else anchor_id,
        pert_dist=np.nan if pert_dist is None else pert_dist,
        source_index=-1 if source is None else source))

def _split_plain(candidates: np.ndarray, source: np.ndarray, database: np.ndarray, n_train: int, n_test: int,
                 logger: logging.Logger, label: str) -> dict:
    ok = select_valid(candidates, database, n_train + n_test, logger, label)
    tr, te = ok[:n_train], ok[n_train:]
    return dict(train=candidates[tr], test=candidates[te],
                meta_train=_meta("train", n_train, source=source[tr]),
                meta_test=_meta("test", n_test, source=source[te]))

def _build_locality(pool: np.ndarray, database: np.ndarray, n_train: int, n_test: int, logger: logging.Logger,
                    label: str) -> dict:
    """Every query is an independent perturbation of one of 50 anchors, with an angle uniform in
    [0, MAX_ANGLE_DEG] and a random direction orthogonal to the anchor. Invalid draws are redrawn for the same anchor."""
    rng = np.random.default_rng([rt_config.DATA_SEED, 300])
    n_anchors = rt_config.N_ANCHORS
    anchor_src = rng.choice(len(pool), n_anchors, replace=False)
    anchors = pool[anchor_src]
    max_dist = rt_config.MAX_ANGLE_DEG / 180.0
    accepted = np.empty((0, pool.shape[1]))
    sets, redraws = {}, 0
    for role, n in (("train", n_train), ("test", n_test)):
        base, extra = divmod(n, n_anchors)
        per_anchor = [base + (1 if a < extra else 0) for a in range(n_anchors)]
        q_list, a_list, p_list = [], [], []
        for a in range(n_anchors):
            need, got = per_anchor[a], 0
            while got < need:
                batch = max(8, 2 * (need - got))
                cand, real = [], []
                for _ in range(batch):
                    u = rng.uniform(0.0, max_dist)
                    q, realized = tangent_perturb(anchors[a], random_tangent_direction(anchors[a], rng), u)
                    cand.append(q)
                    real.append(realized)
                cand, real = np.array(cand), np.array(real)
                valid = nn_distance(cand, database) > rt_config.DUP_TOL
                if len(accepted):
                    valid &= ~(np.arccos(np.clip(cand @ accepted.T, -1, 1)).min(axis=1) / np.pi <= rt_config.DUP_TOL)
                valid &= ~duplicate_of_earlier(cand)
                redraws += int((~valid).sum())
                keep = np.flatnonzero(valid)[:need - got]
                accepted = np.vstack([accepted, cand[keep]])
                q_list.append(cand[keep])
                a_list.append(np.full(len(keep), a))
                p_list.append(real[keep])
                got += len(keep)
        sets[role] = (np.vstack(q_list), np.concatenate(a_list), np.concatenate(p_list))
    assert set(sets["train"][1].tolist()) == set(sets["test"][1].tolist()) == set(range(n_anchors)), \
        f"{label}: train and test must share the same {n_anchors} anchors"
    pd_all = np.concatenate([sets["train"][2], sets["test"][2]]) * 180.0
    logger.info(f"{label}: locality, {n_anchors} anchors, realized perturbation degrees: min {pd_all.min():.3f}, "
                f"mean {pd_all.mean():.3f}, max {pd_all.max():.3f}; {redraws} invalid draws redrawn")
    out = dict(train=sets["train"][0], test=sets["test"][0])
    for role in ("train", "test"):
        out[f"meta_{role}"] = _meta(role, len(sets[role][0]), anchor_id=sets[role][1], pert_dist=sets[role][2],
                                    source=anchor_src[sets[role][1]])
    return out

def build_stage(dataset: str, stage: int, logger: logging.Logger, n_train: int = None, n_test: int = None) -> dict:
    """Returns database, train, test (float64 unit vectors), meta_train, meta_test, db_key."""
    n_train = rt_config.N_TRAIN if n_train is None else n_train
    n_test = rt_config.N_TEST if n_test is None else n_test
    label = f"{dataset}/stage{stage}"
    database = build_database(dataset, stage)
    kind = rt_config.stage_kind(stage)
    if kind == "uniform":
        d = rt_config.DIM[dataset]
        rng = np.random.default_rng([rt_config.DATA_SEED, d, 100])
        cand = sample_uniform_sphere(n_train + n_test + 50, d, rng)
        out = _split_plain(cand, np.full(len(cand), -1), database, n_train, n_test, logger, label)
    elif kind == "real":
        pool = load_official_query_pool(dataset)
        order = np.random.default_rng([rt_config.DATA_SEED, 200]).permutation(len(pool))
        take = order[:min(len(pool), n_train + n_test + 500)]
        out = _split_plain(pool[take], take, database, n_train, n_test, logger, label)
    else:
        out = _build_locality(load_official_query_pool(dataset), database, n_train, n_test, logger, label)
    out.update(database=database, db_key=rt_config.stage_db_key(dataset, stage))
    check_query_sets(out, logger, label)
    return out

def check_query_sets(qs: dict, logger: logging.Logger, label: str) -> None:
    """Asserts every query is farther than DUP_TOL from every database vector and from every other query."""
    both = np.vstack([qs["train"], qs["test"]])
    nn_db = nn_distance(both, qs["database"])
    dup = duplicate_of_earlier(both)
    assert nn_db.min() > rt_config.DUP_TOL, f"{label}: a query coincides with a database vector"
    assert not dup.any(), f"{label}: duplicate queries"
    train_test = np.arccos(np.clip(qs["train"] @ qs["test"].T, -1, 1)).min() / np.pi
    assert train_test > rt_config.DUP_TOL, f"{label}: a test query equals a train query"
    logger.info(f"{label}: query sets valid, train {len(qs['train'])}, test {len(qs['test'])}, min distance to "
                f"database {nn_db.min():.4g}, min train-test distance {train_test:.4g}")

#-------------------------------------------------------------------------------
# Per-database theta cache

def database_digest(database: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(database).tobytes()).hexdigest()

def theta_cache_path(db_key: str, N: int) -> Path:
    return rt_config.THETA_CACHE_DIR / f"{db_key}_N{N}_K{rt_config.K_TOPK}_seed{rt_config.DATA_SEED}_theta.npz"

def build_or_load_theta(db_key: str, database: np.ndarray, logger: logging.Logger) -> tuple:
    """theta of every database vector (K-th nearest OTHER vector). Returns (theta, cache_hit)."""
    N, K = len(database), rt_config.K_TOPK
    path = theta_cache_path(db_key, N)
    digest = database_digest(database)
    if path.exists():
        z = np.load(path)
        assert int(z["N"]) == N and int(z["K"]) == K and len(z["theta"]) == N, f"{path.name}: stale cache"
        assert str(z["digest"]) == digest, f"{path.name}: database hash mismatch"
        logger.info(f"{db_key}: theta cache hit ({path.name})")
        return z["theta"], True
    logger.info(f"{db_key}: building theta cache for N={N} (exact, O(N^2))")
    _, dist = batched_topk_full_skip_self(database, database, np.arange(N), K, batch_size=256)
    theta = dist[:, K - 1].astype(np.float64)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, theta=theta, N=N, K=K, digest=digest)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return theta, False
