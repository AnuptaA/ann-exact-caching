#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Multi-table Hamming estimator, one pass per query chunk.

Labeled set per seed: all N database vectors in permutation order (columns [0, N)),
then the train queries (columns [N, N + n_train)). The sample at S is the column
prefix [0, S); sample-only uses that prefix, sample+train adds the train columns.

Per table: m-bit Hamming distances come from 64-bit +/-1 float32 dot-product blocks.
Per-table averages and cutoffs h_(nu) come from Hamming histograms of theta sums
and counts (exactly tie-inclusive). Union quantities across the L tables (distinct
counts and group means) come from membership masks, never from summed histograms."""

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import rt_config

#-------------------------------------------------------------------------------
# Seed state

@dataclass
class SeedState:
    seed: int
    N: int
    n_train: int
    W: np.ndarray            # (bits_max, d) float32, +/-1
    perm: np.ndarray         # (N,) database indices in column order
    codes: np.ndarray        # (N + n_train, bits_max) int8, +/-1
    theta_lab: np.ndarray    # (N + n_train,) float64
    s_grid: list
    bounds: list = field(default_factory=list)

    @property
    def n_labeled(self) -> int:
        return self.N + self.n_train

    def labeled_ids(self) -> np.ndarray:
        """Database index for sample columns, N + train index for train columns."""
        return np.concatenate([self.perm, self.N + np.arange(self.n_train)])

def make_hyperplanes(seed: int, d: int, bits_max: int = rt_config.BITS_MAX) -> np.ndarray:
    rng = np.random.default_rng([seed, 1])
    return (rng.integers(0, 2, size=(bits_max, d)) * 2 - 1).astype(np.float32)

def sign_codes(X: np.ndarray, W: np.ndarray, block: int = 4000) -> np.ndarray:
    """sign(W x) as int8 +/-1, with 0 counted as +1."""
    out = np.empty((len(X), len(W)), dtype=np.int8)
    for s in range(0, len(X), block):
        proj = X[s:s + block].astype(np.float32) @ W.T
        out[s:s + block] = np.where(proj >= 0, 1, -1)
    return out

def build_seed_state(seed: int, database: np.ndarray, theta_db: np.ndarray, train: np.ndarray,
                     theta_train: np.ndarray, s_grid: list = None, bits_max: int = rt_config.BITS_MAX,
                     perm: np.ndarray = None) -> SeedState:
    from rt_data import sample_permutation
    s_grid = list(rt_config.S_GRID if s_grid is None else s_grid)
    N = len(database)
    assert s_grid[-1] == N, "the last sample size is the full database"
    perm = sample_permutation(N, seed) if perm is None else perm
    W = make_hyperplanes(seed, database.shape[1], bits_max)
    codes = np.concatenate([sign_codes(database[perm], W), sign_codes(train, W)])
    theta_lab = np.concatenate([theta_db[perm], theta_train]).astype(np.float64)
    return SeedState(seed, N, len(train), W, perm, codes, theta_lab, s_grid, [0] + s_grid + [N + len(train)])

#-------------------------------------------------------------------------------
# Hamming distances per table

def table_hamming(q_codes: np.ndarray, state: SeedState, tables: list, block_bits: int = rt_config.BLOCK_BITS) -> dict:
    """(row_lo, m) -> (rows, n_labeled) uint16 Hamming distances. Block dots are exact integers in float32;
    per-table sums are uint16 (m = 2048 exceeds uint8)."""
    rows = len(q_codes)
    H = {tb: np.zeros((rows, state.n_labeled), dtype=np.uint16) for tb in tables}
    for lo in range(0, state.W.shape[0], block_bits):
        members = [tb for tb in tables if tb[0] <= lo < tb[0] + tb[1]]
        if not members:
            continue
        lab = state.codes[:, lo:lo + block_bits].astype(np.float32)
        dot = q_codes[:, lo:lo + block_bits] @ lab.T
        h = ((block_bits - dot) * 0.5).astype(np.uint16)
        for tb in members:
            H[tb] += h
    return H

#-------------------------------------------------------------------------------
# Per-table estimates from histograms

def _segment_hist(Hseg: np.ndarray, theta_seg: np.ndarray, nbins: int) -> tuple:
    rows = Hseg.shape[0]
    idx = (Hseg.astype(np.int64) + (np.arange(rows, dtype=np.int64) * nbins)[:, None]).ravel()
    cnt = np.bincount(idx, minlength=rows * nbins).reshape(rows, nbins).astype(np.float64)
    wts = np.bincount(idx, weights=np.broadcast_to(theta_seg, Hseg.shape).ravel(),
                      minlength=rows * nbins).reshape(rows, nbins)
    return cnt, wts

def population_keys(s_grid: list) -> list:
    return [(S, kind) for S in s_grid for kind in rt_config.POP_KINDS]

def table_estimates(Htab: np.ndarray, state: SeedState, m: int, nu_grid: list) -> dict:
    """(S, kind) -> (values (n_nu, rows) float64, cutoffs h_(nu) (n_nu, rows) int) for one table."""
    nbins = m + 1
    b = state.bounds
    segs = [_segment_hist(Htab[:, b[i]:b[i + 1]], state.theta_lab[b[i]:b[i + 1]], nbins)
            for i in range(len(b) - 1)]
    train_cnt, train_w = segs[-1]
    out, run_cnt, run_w = {}, 0.0, 0.0
    ar = np.arange(Htab.shape[0])
    for i, S in enumerate(state.s_grid):
        run_cnt = run_cnt + segs[i][0]
        run_w = run_w + segs[i][1]
        for kind in rt_config.POP_KINDS:
            cnt, wts = (run_cnt, run_w) if kind == "sample_only" else (run_cnt + train_cnt, run_w + train_w)
            cc, cw = np.cumsum(cnt, axis=1), np.cumsum(wts, axis=1)
            vals = np.empty((len(nu_grid), Htab.shape[0]))
            hs = np.empty((len(nu_grid), Htab.shape[0]), dtype=np.int64)
            for j, nu in enumerate(nu_grid):
                h = (cc < nu).sum(axis=1)
                assert h.max() < nbins, "population smaller than nu"
                vals[j] = cw[ar, h] / cc[ar, h]
                hs[j] = h
            out[(S, kind)] = (vals, hs)
    return out

#-------------------------------------------------------------------------------
# Union quantities over the L tables of a configuration

def union_stats(H: dict, tbs: list, cutoffs: list, state: SeedState, S: int, kind: str) -> dict:
    """Distinct labeled points across tables (union of H <= h_(nu)), split into sample and train members."""
    N = state.N
    rows = H[tbs[0]].shape[0]
    use_train = kind == "sample_train"
    un_s = np.zeros((rows, S), dtype=bool)
    un_t = np.zeros((rows, state.n_train), dtype=bool) if use_train else None
    for tb, h in zip(tbs, cutoffs):
        un_s |= H[tb][:, :S] <= h[:, None]
        if use_train:
            un_t |= H[tb][:, N:] <= h[:, None]
    n_s = un_s.sum(axis=1).astype(np.float64)
    sum_s = un_s.astype(np.float64) @ state.theta_lab[:S]
    if use_train:
        n_t = un_t.sum(axis=1).astype(np.float64)
        sum_t = un_t.astype(np.float64) @ state.theta_lab[N:]
    else:
        n_t, sum_t = np.zeros(rows), np.zeros(rows)
    with np.errstate(invalid="ignore", divide="ignore"):
        return dict(n_total=n_s + n_t, n_train=n_t, n_sample=n_s,
                    mean_theta_train=np.where(n_t > 0, sum_t / n_t, np.nan),
                    mean_theta_sample=np.where(n_s > 0, sum_s / n_s, np.nan))

#-------------------------------------------------------------------------------
# Neighbor dump

def dump_neighbors(H: dict, state: SeedState, S: int, local_rows: np.ndarray, query_ids: np.ndarray, configs: list,
                   nu_max: int = rt_config.DUMP_NU_MAX, tie_cap: int = rt_config.DUMP_TIE_CAP) -> pd.DataFrame:
    """One row per retrieved neighbor per table for sample+train at S: the nu_max smallest by
    (Hamming, labeled id), ties at the cutoff capped at tie_cap rows per table."""
    N = state.N
    ids_all = state.labeled_ids()
    cols = np.concatenate([np.arange(S), np.arange(N, state.n_labeled)])
    ids, theta, from_train = ids_all[cols], state.theta_lab[cols], cols >= N
    recs = []
    for L, m in configs:
        for t, tb in enumerate(rt_config.config_tables(L, m)):
            for r, qid in zip(local_rows, query_ids):
                hq = H[tb][r, cols]
                kth = min(nu_max, len(hq)) - 1
                cutoff = np.partition(hq, kth)[kth]
                sel = np.flatnonzero(hq <= cutoff)
                sel = sel[np.lexsort((ids[sel], hq[sel]))][:tie_cap]
                recs.append(pd.DataFrame(dict(
                    query_id=np.int32(qid), config=rt_config.config_name(L, m), table=np.int8(t),
                    rank_in_table=np.arange(1, len(sel) + 1, dtype=np.int16), hamming=hq[sel].astype(np.int16),
                    theta_neighbor=theta[sel].astype(np.float32), from_train=from_train[sel].astype(np.int8),
                    labeled_id=ids[sel].astype(np.int32))))
    return pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()

#-------------------------------------------------------------------------------
# One pass over a query chunk

@dataclass
class ChunkOptions:
    configs: list = field(default_factory=lambda: list(rt_config.CONFIGS))
    nu_grid: list = field(default_factory=lambda: list(rt_config.NU_GRID))
    ref_S: int = rt_config.REF_S
    union_all_seeds: list = None  # S values with exact union counts for every seed; default [ref_S]
    dump_local: np.ndarray = None  # chunk-local row indices to dump (None for no dump)
    block_bits: int = rt_config.BLOCK_BITS

def _stat_sums(theta: np.ndarray, r_hat: np.ndarray) -> dict:
    rel = (r_hat - theta) / theta
    return dict(n=len(theta), sum_theta=theta.sum(), sum_rhat=r_hat.sum(), sum_theta2=(theta ** 2).sum(),
                sum_rhat2=(r_hat ** 2).sum(), sum_theta_rhat=(theta * r_hat).sum(), sum_relerr=rel.sum(),
                sum_relerr2=(rel ** 2).sum())

def process_chunk(state: SeedState, Xq: np.ndarray, theta_q: np.ndarray, qids: np.ndarray,
                  opts: ChunkOptions) -> dict:
    """Returns dict(rows, agg, dump, timings, tables_est) for one seed and one chunk of test queries."""
    union_all = [opts.ref_S] if opts.union_all_seeds is None else list(opts.union_all_seeds)
    assert opts.ref_S in union_all and opts.ref_S in state.s_grid
    timings = {}
    t0 = time.time()
    tables = rt_config.unique_tables(opts.configs)
    q_codes = sign_codes(Xq, state.W).astype(np.float32)
    H = table_hamming(q_codes, state, tables, opts.block_bits)
    timings["matmul"] = time.time() - t0

    t0 = time.time()
    est = {tb: table_estimates(H[tb], state, tb[1], opts.nu_grid) for tb in tables}
    timings["histograms"] = time.time() - t0

    t0 = time.time()
    agg, rows_parts = [], []
    ref_pop = (opts.ref_S, "sample_train")
    union_time = 0.0
    for L, m in opts.configs:
        tbs = rt_config.config_tables(L, m)
        cname = rt_config.config_name(L, m)
        for pop in population_keys(state.s_grid):
            S, kind = pop
            for j, nu in enumerate(opts.nu_grid):
                vals = np.stack([est[tb][pop][0][j] for tb in tbs])
                hs = [est[tb][pop][1][j] for tb in tbs]
                r_hat = vals.mean(axis=0)
                row = dict(config=cname, L=L, m=m, S=S, pop=kind, nu=nu, seed=state.seed, **_stat_sums(theta_q, r_hat))
                scope = "all_seeds" if S in union_all else "seed0_only"
                do_union = S in union_all or state.seed == 0
                row["distinct_scope"] = scope
                row.update(dict(n_distinct=np.nan, sum_distinct_total=np.nan, sum_distinct_train=np.nan,
                                sum_distinct_sample=np.nan, sum_train_share=np.nan))
                if do_union:
                    tu = time.time()
                    u = union_stats(H, tbs, hs, state, S, kind)
                    union_time += time.time() - tu
                    share = u["n_train"] / u["n_total"]
                    row.update(n_distinct=len(theta_q), sum_distinct_total=u["n_total"].sum(),
                               sum_distinct_train=u["n_train"].sum(), sum_distinct_sample=u["n_sample"].sum(),
                               sum_train_share=share.sum())
                agg.append(row)
                if S == opts.ref_S and kind == "sample_train":
                    so = np.stack([est[tb][(S, "sample_only")][0][j] for tb in tbs]).mean(axis=0)
                    cutoff = np.mean([h / m for h in hs], axis=0)
                    spread = (vals.std(axis=0) / vals.mean(axis=0)) if L > 1 else np.full(len(theta_q), np.nan)
                    rows_parts.append(pd.DataFrame(dict(
                        query_id=qids.astype(np.int32), seed=np.int8(state.seed), config=cname, nu=np.int8(nu),
                        r_hat=r_hat.astype(np.float32), r_hat_sample_only=so.astype(np.float32),
                        n_distinct_total=u["n_total"].astype(np.int32), n_distinct_train=u["n_train"].astype(np.int32),
                        n_distinct_sample=u["n_sample"].astype(np.int32),
                        mean_theta_train_group=u["mean_theta_train"].astype(np.float32),
                        mean_theta_sample_group=u["mean_theta_sample"].astype(np.float32),
                        mean_cutoff_h_over_m=cutoff.astype(np.float32), table_spread=spread.astype(np.float32))))
    timings["estimate_and_union"] = time.time() - t0 - union_time
    timings["union_masks"] = union_time

    t0 = time.time()
    dump = None
    if opts.dump_local is not None and len(opts.dump_local) and state.seed == 0:
        dump = dump_neighbors(H, state, opts.ref_S, np.asarray(opts.dump_local), qids[opts.dump_local], opts.configs)
    timings["dump"] = time.time() - t0
    return dict(rows=pd.concat(rows_parts, ignore_index=True), agg=pd.DataFrame(agg), dump=dump, timings=timings)

def constant_value(state: SeedState, S: int) -> float:
    """Median stored theta of the sample-only labeled points at S."""
    return float(np.median(state.theta_lab[:S]))

#-------------------------------------------------------------------------------

class MultiTableEstimator:
    """The estimator over a fitted labeled set. predict_radius takes only query vectors."""

    def __init__(self, state: SeedState, configs: list = None, nu_grid: list = None, ref_S: int = None):
        self.state = state
        self.opts = ChunkOptions(configs=list(configs or rt_config.CONFIGS), nu_grid=list(nu_grid or rt_config.NU_GRID),
                                 ref_S=ref_S or rt_config.REF_S)

    def predict_radius(self, Q: np.ndarray) -> pd.DataFrame:
        """Per-query rows (config, nu, r_hat, r_hat_sample_only, ...) at the reference S."""
        placeholder = np.ones(len(Q))  # theta only feeds the aggregate sums, never the estimates
        return process_chunk(self.state, Q, placeholder, np.arange(len(Q)), self.opts)["rows"]
