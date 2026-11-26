"""The v10 cache built on a stream's history with exact labels, and the features of history and held-out queries.

The history is the first part of the stream (recal.history_mode), labeled with the stored exact answers
(top500.npy). The cache is the first construction of recal.system.ReCal, so entries, pool, model and m are the
ones the implementation builds. Features come from the compiled core (offset, size, intruder rate).
"""

import multiprocessing
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from recal import accounting, build as builder, history_mode, ksets, sim
from recal.params import Params
from recal.stored import StoredAnswers
from recal.system import ReCal

TARGET = 0.9
DELTA = 0.05
AUDIT = 0.1
ROLE_CODES = {name: i for i, name in enumerate(builder.ROLE_NAMES)}
MIN_CHUNK = 256
_FORKED: dict = {}


def fork_map(fn, n: int, workers: int) -> list:
    """fn(lo, hi) over consecutive ranges covering range(n), results in order.

    With workers > 1 the ranges run in forked processes, which see the parent's memory (the built cache)
    copy-on-write; fn must not depend on state another range changes."""
    if workers <= 1 or n <= MIN_CHUNK:
        return [fn(0, n)]
    parts = min(4 * workers, -(-n // MIN_CHUNK))
    bounds = np.linspace(0, n, parts + 1).astype(int)
    _FORKED["fn"] = fn
    try:
        with ProcessPoolExecutor(min(workers, parts), mp_context=multiprocessing.get_context("fork"),
                                 initializer=_one_blas_thread) as pool:
            return list(pool.map(_call_forked, bounds[:-1], bounds[1:]))
    finally:
        _FORKED.clear()


def _call_forked(lo: int, hi: int):
    return _FORKED["fn"](int(lo), int(hi))


def _one_blas_thread() -> None:
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)


@dataclass
class Features:
    """Per query: has a candidate answer, the three features, the candidate answer's ids and its recall."""
    has: np.ndarray
    F: np.ndarray
    cand_ids: np.ndarray
    recall: np.ndarray
    nr: np.ndarray


@dataclass
class World:
    name: str
    budget: float
    p: Params
    M: float
    db: np.ndarray
    q: np.ndarray
    top: np.ndarray
    stream: np.ndarray
    history_len: int
    built: builder.Built
    snap: dict
    role: np.ndarray
    k: int
    cache: dict = field(default_factory=dict)
    workers: int = 1

    @property
    def core(self):
        return self.built.core

    @property
    def m(self) -> int:
        return self.built.m

    def entries(self) -> list[dict]:
        return [dict(self.core.entry(i), center=self.built.centers[i]) for i in range(self.built.n_entries)]

    def pool_vectors(self) -> np.ndarray:
        ids = np.asarray(self.built.pool_ids, dtype=np.int64)
        return np.asarray(self.db[ids], dtype=np.float32) if len(ids) else np.zeros((0, self.db.shape[1]), np.float32)

    def features(self, Q: np.ndarray, truth: np.ndarray, m: int | None = None) -> Features:
        """Features and candidate recall of queries against the cache; m defaults to the build's m."""
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        if not len(Q) or not self.built.n_entries:
            n = len(Q)
            return Features(np.zeros(n, bool), np.zeros((n, 3)), -np.ones((n, self.k), np.int64), np.zeros(n),
                            np.zeros(n))
        m = self.m if m is None else m
        parts = fork_map(lambda lo, hi: self.core.candidates_batch(Q[lo:hi], self.k, m)[:2], len(Q), self.workers)
        feat = np.concatenate([part[0] for part in parts])
        cand = np.concatenate([part[1] for part in parts])
        has = feat[:, 0] == 1
        recall = np.zeros(len(Q))
        if has.any():
            recall[has] = builder._recall(cand[has], truth[has][:, :self.k])
        return Features(has, feat[:, [2, 3, 4]].copy(), cand, recall, feat[:, 1].copy())

    def history(self) -> dict:
        """Every history query: features, candidate recall and role code."""
        if "history" not in self.cache:
            sel = self.snap["k"] == self.k
            vec, ids = self.snap["vec"][sel], self.snap["ids"][sel][:, :self.k]
            self.cache["history"] = dict(features=self.features(vec, ids), role=self.role[sel], vec=vec)
        return self.cache["history"]

    def role_rows(self, name: str) -> tuple[Features, np.ndarray]:
        h = self.history()
        mask = h["role"] == ROLE_CODES[name]
        f = h["features"]
        return Features(f.has[mask], f.F[mask], f.cand_ids[mask], f.recall[mask], f.nr[mask]), h["vec"][mask]

    def heldout(self) -> dict:
        """The scored requests: unique queries with their features and an index from request to unique query."""
        if "heldout" not in self.cache:
            qi = self.stream[self.history_len:]
            unique, inverse = np.unique(qi, return_inverse=True)
            vec = np.asarray(self.q[unique], dtype=np.float32)
            truth = np.asarray(self.top[unique][:, :self.k], dtype=np.int64)
            self.cache["heldout"] = dict(unique=unique, inverse=inverse, vec=vec, truth=truth,
                                         features=self.features(vec, truth))
        return self.cache["heldout"]


def build_world(prep: str | Path, stream: str, budget: float, coord_bytes: float, name: str = "",
                k: int = 10, seed: int = 0, history_frac: float = history_mode.HISTORY_FRACTION,
                max_requests: int = 0, params: dict | None = None, idealized: bool = False) -> World:
    prep = Path(prep)
    db, q, top, streams = sim.load_prep(prep)
    if top is None:
        raise SystemExit(f"{prep} has no top500.npy (run the prepare stages fetch and prep)")
    idx = np.asarray(streams[stream], dtype=np.int64)
    if max_requests:
        idx = idx[:max_requests]
    history_len = history_mode.history_length(len(idx), history_frac)
    M = accounting.budget_bytes(len(db), db.shape[1], budget, coord_bytes)
    p = Params(K=k, T=TARGET, delta=DELTA, a=AUDIT, coord_bytes=coord_bytes, seed=seed, idealized=idealized,
               **(params or {}))
    unique = np.unique(idx)
    backend = StoredAnswers(db, q[unique], top[unique])
    work = tempfile.mkdtemp(prefix="micro_world_")
    try:
        system = ReCal(backend, p, M, work)
        system.ingest_history(np.asarray(q[idx[:history_len]], dtype=np.float32), ksets.k_sequence(history_len, k))
        built = system.built
        snap = {key: np.array(value) for key, value in system.log.snapshot().items()}
    finally:
        shutil.rmtree(work, ignore_errors=True)
    role = roles_of(snap, p, built)
    return World(name or prep.name, budget, p, M, db, q, top, idx, history_len, built, snap, role, k)


def roles_of(snap: dict, p: Params, built: builder.Built) -> np.ndarray:
    """Role code of every snapshot row, drawn the way the first construction drew them."""
    seed = int(np.random.default_rng(p.seed + 1).integers(2**31))
    roles = builder.split_roles(len(snap["vec"]), p.roles, np.random.default_rng(seed), snap["vec"],
                                p.split_by_query)
    code = np.full(len(snap["vec"]), -1, np.int8)
    for name, rows in roles.items():
        code[rows] = ROLE_CODES[name]
    sizes = {n: int((code == c).sum()) for n, c in ROLE_CODES.items()}
    if sizes != built.info["roles"]:
        raise RuntimeError(f"role split differs from the construction's: {sizes} against {built.info['roles']}")
    return code
