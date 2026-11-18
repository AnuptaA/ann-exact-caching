#!/usr/bin/env python3
"""Hit-rate feature tables (doc 9al): for one static warm-up cache, every request's raw ingredients for
the hit decision, computed once, so that every variant of the decision (oracle, model, calibration size,
feature subsets, shell size / scope / codes, graph search, single-ball serving) is a cheap fit + calibration
on stored numbers (hit_ladder.py). Exact pieces are batched matrix products over all requests; nothing
calls a backend (the exact top-500 lists and the DiskANN answers already exist).

Cache: the simulator's warm-up cache (online_sim.Sim.warmup: greedy over the warm-up candidates, shell
sample filling the rest of the budget at sampling rate p0) for a workload / budget and a cache variant:
  default   f 0.85, entries of any K in 10..500 (the headline cache)
  maxk10    entries of K = k only (past top-k results only, like Aker / QVCache)
  f95, f100 cache share 0.95 / 1.0 of the memory (shell 5% / none)
Backend: exact (balls, shells and labels from the exact lists) or diskann (from DiskANN's deep lists and
answers; truth stays exact).

Rows: the distinct queries of the warm-up fit / calibration splits and of the stream after the warm-up
(requests map to them: post_q, fit_q, cal_q). Per distinct query:
  knn      exact k nearest cached vectors: nr (k-th distance), served answer's retention vs the backend's
           top-k (y) and vs the exact top-k (y_truth)
  balls    the 10 best-fitting entries (fit = (nr + d) / radius) and the 10 nearest centers, with distances
  single   retention when serving only one entry's own k nearest members: the best-fitting entry, the
           nearest center, and the best of the 10 best-fitting (oracle single entry)
  shell    intruder counts (sampled shell vectors within nr, not cached): scope = the shells of the 3 / 10
           best-fitting or nearest entries, at sampling rates P_MULT x p0 (nested draws: one pass), float
           and 4-bit codes; 1-bit RaBitQ at p0; the whole sample at p0 (float, 4-bit); vectors examined
  graph    with the C++ core's HNSW over the cached vectors at (ef1, ef2) pairs: stage-1 nr, the fit order
           and the deployed counts under it, and the stage-2 served answer's retention

  PYTHONPATH=. .venv-core/bin/python simulations/approximate_hits/hit_table.py --workload tripclick-ip --budget 0.01
"""
import sys
import json
import time
import argparse
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
RES = Path("simulations/approximate_hits/results")
EPS = 1e-12
NB = 10                                   # balls kept per order
P_MULT = (4.0, 2.0, 1.0, 0.5, 0.25, 0.1)  # sampling rates as multiples of the deployed p0 (capped at 1)
GRAPH = ((16, 64), (64, 256), (256, 1024))
CHUNK = 1024


def build_cache(w, budget, variant, backend, seed=0):
    from simulations.approximate_hits import online_sim as OS
    argv = ["--prep", str(w.prep_dir), "--stream", w.stream, "--method", "static", "--seed", str(seed),
            "--budget", f"{budget:g}", "--detect"]
    if backend == "diskann":
        argv += ["--backend-file", str(w.answers_file), "--backend-target", w.target]
    if variant == "maxk10":
        argv += ["--max-ball-k", "10"]
    elif variant in ("f95", "f100"):
        argv += ["--f", {"f95": "0.95", "f100": "1.0"}[variant]]
    elif variant != "default":
        raise ValueError(variant)
    args = OS.make_parser().parse_args(argv)
    OS.K_LIMIT[0] = args.max_ball_k
    D, streams = OS.load(args.prep, args)
    if args.warmup is None:
        args.warmup = min(20000, len(streams[args.stream]) // 5)
    stream = streams[args.stream][:args.max_queries]
    sim = OS.Sim(D, stream, args)
    sim.warmup()
    return OS, args, D, stream, sim


def sqd(a, a_sq, b, b_sq):
    return np.maximum(a_sq[:, None] + b_sq[None, :] - 2.0 * (a @ b.T), 0.0)


def topn(M, n, smallest=True):
    """Row-wise indices of the n smallest entries, sorted."""
    n = min(n, M.shape[1])
    part = np.argpartition(M, n - 1, axis=1)[:, :n] if n < M.shape[1] else np.tile(np.arange(M.shape[1]), (len(M), 1))
    v = np.take_along_axis(M, part, axis=1)
    o = np.argsort(v, axis=1, kind="stable")
    return np.take_along_axis(part, o, axis=1)


def main():
    from simulations.approximate_hits.workloads import get
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True)
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--variant", default="default", choices=["default", "maxk10", "f95", "f100"])
    ap.add_argument("--backend", default="exact", choices=["exact", "diskann"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-post", type=int, default=0, help="cap on post-warm-up requests (0 = all; tests)")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    w = get(a.workload)
    out = Path(a.out or RES / "hit_tables" / w.name / f"b{a.budget:g}_{a.variant}_{a.backend}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    OS, args, D, stream, sim = build_cache(w, a.budget, a.variant, a.backend)
    cache, k = sim.cache, sim.k
    cache.prepare()
    t_build = time.time() - t0
    logger.info(f"cache: {len(cache.entries)} entries, {cache.n_cached()} cached, p0 {cache.p:.4f}, "
                f"{len(cache.samp_ids)} samples ({t_build:.0f} s)")

    # ---- rows: distinct queries of the fit / cal splits and the post-warm-up stream
    post = np.asarray(stream[args.warmup:], dtype=np.int64)
    if a.max_post:
        post = post[:a.max_post]
    fit_q = np.asarray(sim.fit_q, dtype=np.int64)
    cal_q = np.asarray(sim.cal[0][4], dtype=np.int64)
    U = np.unique(np.concatenate([post, fit_q, cal_q]))
    nU = len(U)
    Q = D["q"]
    A = OS.topk_present(D, U, k, key="answer")         # backend's top-k (labels)
    Tt = OS.topk_present(D, U, k, key="truth")         # exact top-k (evaluation)

    # ---- cache arrays
    cids, cv = cache.cached_ids, np.ascontiguousarray(cache.cv, dtype=np.float32)
    cv_sq = (cv.astype(np.float64) ** 2).sum(axis=1).astype(np.float32)
    cen = cache.centers
    cq = np.ascontiguousarray(Q[cen], dtype=np.float32)
    cq_sq = (cq.astype(np.float64) ** 2).sum(axis=1).astype(np.float32)
    rad = cache.radius.astype(np.float64)
    nE = len(cen)
    mpos = [np.searchsorted(cids, cache.members(int(c))) for c in cen]   # members as positions in cids
    # shell vectors (not cached) of every entry with their draws, up to the largest sampling rate
    p0 = float(cache.p)
    pgrid = np.array([min(1.0, m * p0) for m in P_MULT])
    pmax = float(pgrid.max()) if p0 > 0 else 0.0
    draw, cnt = D["draw"], cache._count
    eshell = []
    for c in cen:
        _, sh = cache._ball_shell(int(c))
        sh = sh[sh >= 0]
        sh = sh[D["present"][sh] & (cnt[sh] == 0) & (draw[sh] < pmax)]
        eshell.append(sh)
    S_ids = np.unique(np.concatenate(eshell)) if eshell and pmax > 0 else np.empty(0, np.int64)
    epos = [np.searchsorted(S_ids, sh) for sh in eshell]
    S = np.ascontiguousarray(D["db"][S_ids], dtype=np.float32)
    S_draw = draw[S_ids].astype(np.float64)
    # 4-bit codes with the deployed sample's range (the simulator's quantize over the sample at p0)
    x0 = D["db"][cache.samp_ids].astype(np.float64) if len(cache.samp_ids) else np.zeros((1, Q.shape[1]))
    lo, hi = x0.min(axis=0), x0.max(axis=0)
    step = np.maximum(hi - lo, EPS) / 15.0
    S4 = (lo + np.round((np.clip(S, lo, hi) - lo) / step) * step).astype(np.float32) if len(S) else S
    # RaBitQ (1 bit, centroid = the deployed sample's mean, rotation seed 0) for the vectors drawn at p0
    dim = Q.shape[1]
    rq_c = x0.mean(axis=0)
    P, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(dim, dim)))
    P = P.astype(np.float32)
    r = S.astype(np.float64) - rq_c
    rn = np.linalg.norm(r, axis=1)
    u = ((r / np.maximum(rn, EPS)[:, None]).astype(np.float32) @ P) if len(S) else np.zeros((0, dim), np.float32)
    rq_code = np.where(u >= 0, 1, -1).astype(np.int8)
    rq_xo = np.abs(u).sum(axis=1) / np.sqrt(dim)
    del u, r
    # the whole deployed sample (scope all): float and the simulator's 4-bit vectors
    all_pos = np.flatnonzero(S_draw < p0) if len(S) else np.empty(0, np.int64)
    Sa, Sa4 = S[all_pos], S4[all_pos]
    Sa_sq = (Sa.astype(np.float64) ** 2).sum(axis=1).astype(np.float32)
    Sa4_sq = (Sa4.astype(np.float64) ** 2).sum(axis=1).astype(np.float32)
    logger.info(f"{nU} distinct requests, {nE} entries, {len(S_ids)} shell vectors up to p {pmax:.3f} "
                f"({len(all_pos)} at p0)")

    # ---- graph variants (C++ core): stage-1 nr and stage-2 served answers
    graph = {}
    if not a.no_graph and len(cids) >= k:
        from simulations.approximate_hits.cachecore import Core
        tg = time.time()
        core = Core(dim, k, 0, capacity=len(cids) + 1024, graph=True)
        core.set_threads(a.threads)
        core.add_cached(cids.astype(np.int64), cv)
        e0 = np.empty(0, np.int64)
        z = np.zeros((0, dim), np.float32)
        for e, c in enumerate(cen):
            b = cids[mpos[e]].astype(np.int64)
            core.add_entry(int(c), cq[e], float(rad[e]), b, cv[mpos[e]], e0, z, 0.0)
        xs = np.ascontiguousarray(Q[U], dtype=np.float32)
        tf = RES / "tuning_core" / w.name / f"b{a.budget:g}.json"
        pairs = list(GRAPH)
        if tf.exists():
            d = json.loads(tf.read_text())
            pairs.append((int(d["stage1"]["ef"]), int(d["stage2"]["ef"])))
        for ef1, ef2 in dict.fromkeys(pairs):
            res = core.lookup_batch(xs, float("-inf"), ef1, ef2)
            near = np.asarray(res[2], dtype=np.int64)
            graph[(ef1, ef2)] = {"nr": np.asarray(res[3], dtype=np.float64),
                                 "y": (near[:, :, None] == A[:, None, :]).any(axis=2).mean(axis=1),
                                 "y_truth": (near[:, :, None] == Tt[:, None, :]).any(axis=2).mean(axis=1)}
        del core
        logger.info(f"graph variants {list(graph)} ({time.time() - tg:.0f} s)")

    # ---- main pass (chunks of distinct requests)
    nP = len(pgrid)
    o = {
        "nr": np.zeros(nU), "y": np.zeros(nU), "y_truth": np.zeros(nU),
        "fit_idx": np.zeros((nU, NB), np.int32), "fit_de": np.zeros((nU, NB), np.float32),
        "near_idx": np.zeros((nU, NB), np.int32), "near_de": np.zeros((nU, NB), np.float32),
        "y_single_fit": np.zeros(nU), "y_single_near": np.zeros(nU), "y_single_best": np.zeros(nU),
        "ytr_single_fit": np.zeros(nU), "ytr_single_near": np.zeros(nU),
        "all_f": np.zeros(nU, np.int32), "all_4": np.zeros(nU, np.int32),
        "rq_fit3": np.zeros(nU, np.int32), "rq_fit10": np.zeros(nU, np.int32),
        "rq_near3": np.zeros(nU, np.int32), "rq_near10": np.zeros(nU, np.int32),
    }
    for order in ("fit", "near"):
        for B in (3, 10):
            for code in ("f", "4"):
                o[f"cnt_{order}{B}_{code}"] = np.zeros((nU, nP), np.int32)
            o[f"scan_{order}{B}"] = np.zeros((nU, nP), np.int32)
    for g in graph:
        tag = f"g{g[0]}_{g[1]}"
        o[f"{tag}_fit_idx"] = np.zeros((nU, 3), np.int32)
        o[f"{tag}_fit_de"] = np.zeros((nU, 3), np.float32)
        o[f"{tag}_all_4"] = np.zeros(nU, np.int32)
        o[f"{tag}_fit10_4"] = np.zeros(nU, np.int32)
        o[f"{tag}_near10_4"] = np.zeros(nU, np.int32)
    i_p0 = P_MULT.index(1.0)
    tm = time.time()
    for s in range(0, nU, CHUNK):
        qi = U[s:s + CHUNK]
        x = np.ascontiguousarray(Q[qi], dtype=np.float32)
        x_sq = (x.astype(np.float64) ** 2).sum(axis=1).astype(np.float32)
        m = len(qi)
        d2 = sqd(x, x_sq, cv, cv_sq) if len(cids) else np.zeros((m, 0), np.float32)
        if len(cids) >= k:
            nn = topn(d2, k)
            nr = np.sqrt(np.take_along_axis(d2, nn[:, -1:], axis=1)[:, 0].astype(np.float64)) + EPS
            served = cids[nn]
        else:
            nr = np.full(m, np.inf)
            served = np.zeros((m, k), np.int64)
        As, Ts = A[s:s + m], Tt[s:s + m]
        o["nr"][s:s + m] = nr
        o["y"][s:s + m] = (served[:, :, None] == As[:, None, :]).any(axis=2).mean(axis=1)
        o["y_truth"][s:s + m] = (served[:, :, None] == Ts[:, None, :]).any(axis=2).mean(axis=1)
        de = np.sqrt(sqd(x, x_sq, cq, cq_sq).astype(np.float64))
        fit = (nr[:, None] + de) / rad[None, :]
        fidx, nidx = topn(fit, NB), topn(de, NB)
        nb = fidx.shape[1]
        o["fit_idx"][s:s + m, :nb] = fidx
        o["fit_de"][s:s + m, :nb] = np.take_along_axis(de, fidx, axis=1)
        o["near_idx"][s:s + m, :nb] = nidx
        o["near_de"][s:s + m, :nb] = np.take_along_axis(de, nidx, axis=1)
        # the whole sample at p0
        if len(all_pos):
            nr2 = (nr * nr)[:, None]
            o["all_f"][s:s + m] = (sqd(x, x_sq, Sa, Sa_sq) < nr2).sum(axis=1)
            o["all_4"][s:s + m] = (sqd(x, x_sq, Sa4, Sa4_sq) < nr2).sum(axis=1)
        gfit = {}
        for g, G in graph.items():
            ng = G["nr"][s:s + m]
            gf = topn((ng[:, None] + de) / rad[None, :], max(NB, 3))
            gfit[g] = (ng, gf)
            tag = f"g{g[0]}_{g[1]}"
            o[f"{tag}_fit_idx"][s:s + m, :min(3, gf.shape[1])] = gf[:, :3]
            o[f"{tag}_fit_de"][s:s + m, :min(3, gf.shape[1])] = np.take_along_axis(de, gf[:, :3], axis=1)
            if len(all_pos):
                o[f"{tag}_all_4"][s:s + m] = (sqd(x, x_sq, Sa4, Sa4_sq) < (ng * ng)[:, None]).sum(axis=1)
        # per request: single-entry serving and the ball-scoped shells
        for i in range(m):
            row = d2[i]
            yb = []
            for j, e in enumerate(fidx[i]):
                mp = mpos[e]
                if len(mp) < k:
                    yb.append((0.0, 0.0))
                    continue
                top = mp[np.argpartition(row[mp], k - 1)[:k]] if len(mp) > k else mp
                ids = cids[top]
                yb.append((np.isin(As[i], ids).mean(), np.isin(Ts[i], ids).mean()))
            o["y_single_fit"][s + i], o["ytr_single_fit"][s + i] = yb[0]
            o["y_single_best"][s + i] = max(v[0] for v in yb)
            e = nidx[i, 0]
            mp = mpos[e]
            if len(mp) >= k:
                top = mp[np.argpartition(row[mp], k - 1)[:k]] if len(mp) > k else mp
                ids = cids[top]
                o["y_single_near"][s + i] = np.isin(As[i], ids).mean()
                o["ytr_single_near"][s + i] = np.isin(Ts[i], ids).mean()
            if not len(S):
                continue
            nr2 = nr[i] * nr[i]
            xi = x[i]
            qr = xi.astype(np.float64) - rq_c
            qn = float(np.linalg.norm(qr))
            qp = (qr / max(qn, EPS)).astype(np.float32) @ P
            for order, ix in (("fit", fidx[i]), ("near", nidx[i])):
                pos10 = np.unique(np.concatenate([epos[e] for e in ix[:10]])) if len(ix) else np.empty(0, np.int64)
                in3 = np.isin(pos10, np.concatenate([epos[e] for e in ix[:3]])) if len(pos10) else np.zeros(0, bool)
                if not len(pos10):
                    continue
                V = S[pos10]
                dd = ((V - xi) ** 2).sum(axis=1)
                dd4 = ((S4[pos10] - xi) ** 2).sum(axis=1)
                dr = S_draw[pos10]
                for B, msk in ((3, in3), (10, np.ones(len(pos10), bool))):
                    for j, pg in enumerate(pgrid):
                        sel = msk & (dr < pg)
                        o[f"scan_{order}{B}"][s + i, j] = sel.sum()
                        o[f"cnt_{order}{B}_f"][s + i, j] = (sel & (dd < nr2)).sum()
                        o[f"cnt_{order}{B}_4"][s + i, j] = (sel & (dd4 < nr2)).sum()
                    sel = msk & (dr < p0)
                    if sel.any():
                        pp = pos10[sel]
                        ip = (rq_code[pp].astype(np.float32) @ qp) / np.sqrt(dim) / np.maximum(rq_xo[pp], 1e-6)
                        est = rn[pp] ** 2 + qn * qn - 2 * rn[pp] * qn * ip
                        o[f"rq_{order}{B}"][s + i] = int((est < nr2).sum())
            # graph variants: the deployed counts (4-bit, p0) over the 10 best-fitting / nearest under graph nr
            for g, (ng, gf) in gfit.items():
                tag = f"g{g[0]}_{g[1]}"
                ng2 = ng[i] * ng[i]
                for name, ix in ((f"{tag}_fit10_4", gf[i, :10]), (f"{tag}_near10_4", nidx[i, :10])):
                    pp = np.unique(np.concatenate([epos[e] for e in ix]))
                    pp = pp[S_draw[pp] < p0]
                    if len(pp):
                        o[name][s + i] = int((((S4[pp] - xi) ** 2).sum(axis=1) < ng2).sum())
        if (s // CHUNK) % 5 == 0:
            el = time.time() - tm
            logger.info(f"rows {s + m} / {nU} ({el:.0f} s, ~{el / (s + m) * (nU - s - m):.0f} s left)")
    for g, G in graph.items():
        tag = f"g{g[0]}_{g[1]}"
        o[f"{tag}_nr"], o[f"{tag}_y"], o[f"{tag}_y_truth"] = G["nr"], G["y"], G["y_truth"]
    # ---- cache composition: share of cached vectors within their entries' first k members
    first_k = np.zeros(len(cids), bool)
    for e in range(nE):
        mp = mpos[e]
        if len(mp):
            row_ids = cids[mp]
            # members are a prefix of the center's ranking: rank order = order in the ranking
            rk = cache.D["rank"][cen[e]]
            inprefix = np.isin(rk[:k], row_ids)
            first_k[np.searchsorted(cids, rk[:k][inprefix])] = True
    Ks = np.array([cache.entries[int(c)] for c in cen])
    meta = {"workload": w.name, "budget": a.budget, "variant": a.variant, "backend": a.backend, "k": k,
            "T": sim.T, "delta": args.delta, "p0": p0, "pgrid": pgrid.tolist(), "p_mult": list(P_MULT),
            "bits": cache.bits, "entries": int(nE), "cached": int(len(cids)), "samples_p0": int(len(all_pos)),
            "memory": float(cache.memory()), "M": float(sim.M), "warmup": int(args.warmup),
            "share_cached_in_first_k": float(first_k.mean()) if len(cids) else 0.0,
            "shell_memory_share": float(len(all_pos) * cache.bits / 32 / max(sim.M, 1)),
            "K_hist": {int(x): int(c) for x, c in zip(*np.unique(Ks, return_counts=True))},
            "graph": [list(g) for g in graph], "t_build_s": t_build, "t_total_s": time.time() - t0,
            "lam_deployed": None if sim.lam is None else float(sim.lam), "insample": getattr(sim, "insample", {})}
    np.savez(out, U=U, post_q=np.searchsorted(U, post), fit_q=np.searchsorted(U, fit_q),
             cal_q=np.searchsorted(U, cal_q), rad=rad, entry_K=Ks, meta=json.dumps(meta), **o)
    logger.info(f"wrote {out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
