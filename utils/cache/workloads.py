#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Workloads and saved-run inputs: SIFT/ESCI query pools, simZipf, Aker traces."""

import json
import argparse
import logging
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Optional

import numpy as np

from utils.base.io import load_fvecs_fast
from utils.base.knn import exact_topk
from utils.cache.entries import build_entry_tables

logger = logging.getLogger(__name__)



@dataclass
class DataPaths:
    """Where the raw data lives; the command line can override it (see cli.py)."""
    sift: Path = Path("datasets/sift/raw_sift1m/sift")
    aker: Path = Path("datasets/aker")
    esci: Path = Path("datasets/esci/data_vectors")


PATHS = DataPaths()

EVAL_SPLITS = ("fit", "dev", "cal", "test")
SCORE_SPLITS = ("fit", "cal", "test")
TRACE_SPLITS = ("cand", "score", "fit", "dev", "cal", "test")
TRACE_FRACTIONS = {"cand": 0.08, "score": 0.32, "fit": 0.15, "dev": 0.15,
                   "cal": 0.15, "test": 0.15}
WORKLOADS = {
    **{f"spacev-aker-{s}": {"base": "spacev-10m.npy",
                            "trace": [f"spacev-sim-100k-{s}.npy"]}
       for s in ("0.3", "0.6", "0.99")},
    **{f"triviaqa-aker-{s}": {"base": "sphere-10m.npy",
                              "trace": [f"triviaqa-sim-100k-{s}.npy"]}
       for s in ("0.3", "0.6", "0.99")},
    "spacev-real": {"base": "spacev-10m.npy",
                    "trace": ["spacev1b/query_log.bin", "spacev1b/query.bin"]},
}

#-------------------------------------------------------------------------------


def _take(perm, start, n, pool_size, what):
    """perm[start:start + n], refusing to silently return a short slice."""
    if start + n > pool_size:
        raise ValueError(f"{what}: need queries [{start}, {start + n}) but the pool has "
                         f"only {pool_size}")
    return perm[start:start + n]


def load_inputs(args):
    """Database, candidate-generating, scoring, and test query vectors.

    Deterministic given args (dataset, sizes, seed), so evaluation scripts can
    rebuild a run's exact inputs from its saved config.
    """
    rng = np.random.default_rng(args.seed)
    if args.dataset == "sift":
        logger.info("Loading SIFT1M...")
        base = load_fvecs_fast(str(PATHS.sift / "sift_base.fvecs"))
        learn = load_fvecs_fast(str(PATHS.sift / "sift_learn.fvecs"))
        test = load_fvecs_fast(str(PATHS.sift / "sift_query.fvecs"))
        if args.n_test > len(test):
            raise ValueError(f"n_test={args.n_test} exceeds the {len(test)} SIFT queries")
        test = test[:args.n_test]
        db = base[rng.choice(len(base), size=args.n_db, replace=False)]
        train_idx = rng.choice(len(learn), size=args.n_cand_queries + args.n_score_queries,
                               replace=False)
        cand_q = learn[train_idx[:args.n_cand_queries]]
        score_q = learn[train_idx[args.n_cand_queries:]]
    else:
        # ESCI: database = pre-built product subsample; train and test are
        # disjoint random splits of the real ESCI queries
        esci_dir = Path(args.esci_dir)
        logger.info(f"Loading ESCI from {esci_dir}...")
        db = np.load(esci_dir / f"esci_db{args.n_db}_s{args.seed}.npy")
        pool = np.load(esci_dir / "esci_queries.npy")
        perm = rng.permutation(len(pool))
        n_c, n_s = args.n_cand_queries, args.n_score_queries
        test = pool[_take(perm, 0, args.n_test, len(pool), "test")]
        cand_q = pool[_take(perm, args.n_test, n_c, len(pool), "candidates")]
        score_q = pool[_take(perm, args.n_test + n_c, n_s, len(pool), "scoring")]
    return db, cand_q, score_q, test


class SimZipfWorkload:
    """Lazily materialized similar-query groups with Zipf group popularity."""

    def __init__(self, base_queries, group_size, zipf_s, method,
                 lambda_max, noise_rel, seed):
        if len(base_queries) < 2 or group_size < 1:
            raise ValueError("need at least 2 base queries and a group size of at least 1")
        self.q = base_queries.astype(np.float64)
        self.n_groups = len(base_queries)
        self.group_size = group_size
        self.method = method
        self.noise_rel = noise_rel
        self.seed = seed
        rng = np.random.default_rng(seed)
        # interpolation partners (never the anchor itself) and mixing weights
        partners = rng.integers(0, self.n_groups - 1, size=(self.n_groups, group_size))
        partners += partners >= np.arange(self.n_groups)[:, None]
        self.partners = partners
        self.lam = rng.uniform(0.0, lambda_max, size=(self.n_groups, group_size))
        # Zipf popularity over a random rank order of groups
        rank_of_group = rng.permutation(self.n_groups)
        p = 1.0 / np.power(rank_of_group + 1.0, zipf_s)
        self.group_prob = p / p.sum()

    def sample(self, n, rng):
        """Sample n requests; returns variant ids (group * group_size + j)."""
        g = rng.choice(self.n_groups, size=n, p=self.group_prob)
        j = rng.integers(0, self.group_size, size=n)
        return g * self.group_size + j

    def vectors(self, ids):
        g = ids // self.group_size
        j = ids % self.group_size
        anchor = self.q[g]
        if self.method == "interp":
            lam = self.lam[g, j][:, None]
            return (1.0 - lam) * anchor + lam * self.q[self.partners[g, j]]
        out = np.empty_like(anchor)
        dim = anchor.shape[1]
        for i, vid in enumerate(ids.tolist()):
            z = np.random.default_rng([self.seed, vid]).standard_normal(dim)
            out[i] = anchor[i] + self.noise_rel * np.linalg.norm(anchor[i]) * z / np.sqrt(dim)
        return out


def dedup(ids):
    uniq, inv, counts = np.unique(ids, return_inverse=True, return_counts=True)
    return uniq, counts.astype(np.int64), inv


def build_inputs(args):
    """Database and workload draws; deterministic given args (sizes, seed).

    Returns a dict with db, the workload, deduplicated candidate / scoring /
    test variant ids, their weights, the test access order, and stats.
    """
    # same database subsample as load_inputs for the same seed
    rng = np.random.default_rng(args.seed)
    logger.info("Loading SIFT1M...")
    sift_base = load_fvecs_fast(str(PATHS.sift / "sift_base.fvecs"))
    db = sift_base[rng.choice(len(sift_base), size=args.n_db, replace=False)]
    base_queries = load_fvecs_fast(str(PATHS.sift / "sift_query.fvecs"))

    wl = SimZipfWorkload(base_queries, args.group_size, args.zipf, args.method,
                         args.lambda_max, args.noise_rel, args.seed)
    wrng = np.random.default_rng(args.seed + 1)
    cand_ids, _, _ = dedup(wl.sample(args.n_cand_requests, wrng))
    score_ids, score_w, _ = dedup(wl.sample(args.n_score_requests, wrng))
    test_req = wl.sample(args.n_test_requests, wrng)
    test_ids, test_w, test_seq = dedup(test_req)

    stats = {
        "distinct_cand": int(len(cand_ids)),
        "distinct_score": int(len(score_ids)),
        "distinct_test": int(len(test_ids)),
        "test_requests_exact_repeat_of_score": float(np.isin(test_req, score_ids).mean()),
        "test_requests_group_in_cand": float(np.isin(
            test_req // args.group_size, cand_ids // args.group_size).mean()),
        "top1pct_groups_request_share": float(
            np.sort(wl.group_prob)[::-1][:max(1, wl.n_groups // 100)].sum()),
    }
    return {"db": db, "wl": wl, "cand_ids": cand_ids, "score_ids": score_ids,
            "score_w": score_w, "test_ids": test_ids, "test_w": test_w,
            "test_seq": test_seq, "stats": stats}


def _read_spacev_bin(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        n, d = np.frombuffer(f.read(8), dtype=np.int32)
        return np.frombuffer(f.read(int(n) * int(d)), dtype=np.int8).reshape(n, d)


def _load_vectors(path: Path) -> np.ndarray:
    return _read_spacev_bin(path) if path.suffix == ".bin" else np.load(path)


def load_base(name: str) -> np.ndarray:
    """Database vectors: int8 kept as is; floating data as float32."""
    arr = np.load(PATHS.aker / WORKLOADS[name]["base"], mmap_mode="r")
    if np.issubdtype(arr.dtype, np.integer):
        return np.ascontiguousarray(arr)
    out = np.empty(arr.shape, dtype=np.float32)
    for b in range(0, len(arr), 500_000):
        out[b:b + 500_000] = arr[b:b + 500_000]
    return out


def split_boundaries(name: str, parts) -> dict:
    """Consecutive request ranges per split."""
    if name == "spacev-real":
        log_n, test_n = len(parts[0]), len(parts[1])
        sizes = {"cand": 10000, "score": 34162, "fit": 16000, "dev": 17000,
                 "cal": log_n - 10000 - 34162 - 16000 - 17000, "test": test_n}
    else:
        n = sum(len(p) for p in parts)
        sizes = {s: int(TRACE_FRACTIONS[s] * n) for s in TRACE_SPLITS}
        sizes["test"] = n - sum(v for s, v in sizes.items() if s != "test")
    short = [s for s, v in sizes.items() if v < 1]
    if short:
        raise ValueError(f"trace {name!r} is too short for splits {short}: sizes {sizes}")
    bounds, start = {}, 0
    for s in TRACE_SPLITS:
        bounds[s] = (start, start + sizes[s])
        start += sizes[s]
    return bounds


def build_trace(name: str):
    """Requests (in order) as indices into globally distinct vectors."""
    parts = [_load_vectors(PATHS.aker / p) for p in WORKLOADS[name]["trace"]]
    seq = np.concatenate(parts)
    uniq, inv = np.unique(seq, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    return uniq, inv, split_boundaries(name, parts)


def split_view(inv, bounds, split):
    """Distinct global ids in a split, their weights, and the request order."""
    lo, hi = bounds[split]
    ids, local, counts = np.unique(inv[lo:hi], return_inverse=True, return_counts=True)
    return ids, counts.astype(np.int64), local.reshape(-1)


def load_trace_run(run_dir: Path):
    """Everything an evaluation script needs from a saved trace run."""
    data = json.loads((run_dir / "results.json").read_text())
    cfg = argparse.Namespace(**data["config"])
    db = load_base(cfg.trace)
    z = np.load(run_dir / "topk.npz")
    uniq, inv, topk_all = z["uniq"], z["inv"], z["topk"]
    bounds = {s: tuple(z[f"bounds_{s}"]) for s in TRACE_SPLITS}
    cand_ids = split_view(inv, bounds, "cand")[0]
    sets, topk, seqs = {}, {}, {}
    for s in ("score", "fit", "dev", "cal", "test"):
        ids, w, local = split_view(inv, bounds, s)
        sets[s] = (uniq[ids], w)
        topk[s] = topk_all[ids][:, :cfg.k]
        seqs[s] = local
    return {"data": data, "cfg": cfg, "db": db, "cand_q": uniq[cand_ids],
            "cand_top": z["cand_top"], "sets": sets, "topk": topk, "seqs": seqs}


@dataclass
class RunData:
    """Everything an evaluation needs from a saved greedy run (explicit, no globals)."""
    k: int
    k_values: list
    db: np.ndarray
    cand_q: np.ndarray
    sets: dict                     # split -> (query vectors, request weights)
    topk: dict                     # split -> exact top-k ids
    cached_ids: list               # per entry index: ids cached by that entry
    radius: np.ndarray             # per entry index: ball radius
    cand_ext: Optional[np.ndarray] = None   # exact top-R ids per candidate (deeper shells)

    def __post_init__(self):
        kv = list(self.k_values)
        if not kv or kv[0] < 1 or any(a >= b for a, b in zip(kv, kv[1:])):
            raise ValueError(f"k_values must be positive and strictly increasing, got {kv}")
        if self.k < 1:
            raise ValueError(f"k must be at least 1, got {self.k}")
        n_ent = None if self.cand_q is None else len(self.cand_q) * len(kv)
        for name, arr in (("cached_ids", self.cached_ids), ("radius", self.radius)):
            if n_ent is not None and arr is not None and len(arr) != n_ent:
                raise ValueError(f"{name} has {len(arr)} entries, expected {n_ent}")
        for name, (q, wts) in self.sets.items():
            if len(q) != len(wts):
                raise ValueError(f"split {name}: {len(q)} queries but {len(wts)} weights")
            if name in self.topk and (len(self.topk[name]) != len(q)
                                      or self.topk[name].shape[1] < self.k):
                raise ValueError(f"split {name}: top-k table does not match the queries or k")

    @cached_property
    def draw(self):
        """Fixed uniform draw per database vector (shell-sample membership)."""
        return np.random.default_rng(1).random(len(self.db))


def make_run_data(cfg, db, cand_q, sets, cand_top=None, pre_topk=None):
    """Entry tables and exact top-k for the given splits."""
    cached_ids, radius, _, _ = build_entry_tables(cand_q, db, cfg.k_values, 1,
                                                  cached_top=cand_top)
    topk = pre_topk if pre_topk is not None else {
        name: exact_topk(q, db, cfg.k) for name, (q, _) in sets.items()}
    return RunData(k=cfg.k, k_values=cfg.k_values, db=db, cand_q=cand_q, sets=sets,
                   topk=topk, cached_ids=cached_ids, radius=radius)


def load_run_data(run_dir: Path, n_each: int):
    """Load a run's splits, exact neighbors and entry tables: (results, cfg, RunData)."""
    run_cfg = json.loads((run_dir / "results.json").read_text())["config"]
    cand_top = pre_topk = None
    if "trace" in run_cfg:
        # trace runs: fixed time-ordered splits and saved exact neighbors
        t = load_trace_run(run_dir)
        data, cfg, db, cand_q = t["data"], t["cfg"], t["db"], t["cand_q"]
        sets = {s: t["sets"][s] for s in EVAL_SPLITS}
        pre_topk = {s: t["topk"][s] for s in EVAL_SPLITS}
        cand_top = t["cand_top"]
    else:
        data, cfg, db, cand_q, sets = load_splits(run_dir, n_each)
    logger.info(f"{run_dir.name}: " + ", ".join(
        f"{k}={len(v[0])} distinct ({int(v[1].sum())} requests)" for k, v in sets.items()))
    return data, cfg, make_run_data(cfg, db, cand_q, sets, cand_top, pre_topk)


def load_splits(run_dir: Path, n_each: int):
    data = json.loads((run_dir / "results.json").read_text())
    cfg = argparse.Namespace(**data["config"])
    if "zipf" in data["config"]:
        inp = build_inputs(cfg)
        wl = inp["wl"]
        rng = np.random.default_rng(cfg.seed + 4)  # independent of all other draws
        sets = {}
        for name in ("fit", "dev", "cal"):
            ids, w, _ = dedup(wl.sample(n_each, rng))
            sets[name] = (wl.vectors(ids), w)
        sets["test"] = (wl.vectors(inp["test_ids"]), inp["test_w"])
        return data, cfg, inp["db"], wl.vectors(inp["cand_ids"]), sets
    db, cand_q, _, test_q = load_inputs(cfg)
    pool = np.load(Path(cfg.esci_dir) / "esci_queries.npy")
    perm = np.random.default_rng(cfg.seed).permutation(len(pool))
    start = cfg.n_test + cfg.n_cand_queries + cfg.n_score_queries
    sets = {}
    for i, name in enumerate(("fit", "dev", "cal")):
        q = pool[_take(perm, start + i * n_each, n_each, len(pool), f"{name} queries")]
        sets[name] = (q, np.ones(len(q), dtype=np.int64))
    sets["test"] = (test_q, np.ones(len(test_q), dtype=np.int64))
    return data, cfg, db, cand_q, sets


def load_fit_cal_splits(run_dir: Path, n_fit: int, n_calib: int):
    """Rebuild a run's inputs plus fresh fit and calibration queries."""
    data = json.loads((run_dir / "results.json").read_text())
    cfg = argparse.Namespace(**data["config"])
    if "zipf" in data["config"]:
        inp = build_inputs(cfg)
        wl = inp["wl"]
        rng = np.random.default_rng(cfg.seed + 2)  # independent of all run draws
        fit_ids, fit_w, _ = dedup(wl.sample(n_fit, rng))
        cal_ids, cal_w, _ = dedup(wl.sample(n_calib, rng))
        sets = {
            "fit": (wl.vectors(fit_ids), fit_w),
            "cal": (wl.vectors(cal_ids), cal_w),
            "test": (wl.vectors(inp["test_ids"]), inp["test_w"]),
        }
        db = inp["db"]
        cand_q = wl.vectors(inp["cand_ids"])
    else:
        db, cand_q, _, test_q = load_inputs(cfg)
        pool = np.load(Path(cfg.esci_dir) / "esci_queries.npy")
        perm = np.random.default_rng(cfg.seed).permutation(len(pool))
        start = cfg.n_test + cfg.n_cand_queries + cfg.n_score_queries
        fit_q = pool[_take(perm, start, n_fit, len(pool), "fit queries")]
        cal_q = pool[_take(perm, start + n_fit, n_calib, len(pool), "calibration queries")]
        sets = {
            "fit": (fit_q, np.ones(len(fit_q), dtype=np.int64)),
            "cal": (cal_q, np.ones(len(cal_q), dtype=np.int64)),
            "test": (test_q, np.ones(len(test_q), dtype=np.int64)),
        }
    return data, cfg, db, cand_q, sets
