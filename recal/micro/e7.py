"""E7: an HNSW over the cached vectors.

For degree in DEGREES and ef in EFS: how often the search returns exactly the K nearest cached vectors, the mean
recall against them, the time per query (one thread), and the edge memory per vector. Sizes of the graph are
computed: level 0 holds 2 x degree neighbor ids and a 4-byte count, upper layers an expected 1/(degree-1) of that.
Also, per setting: an exact rerank of the top 4K candidates (share exact, time with the rerank), and for the fast
path (ef in FAST_EFS) how the v11 features computed from the returned answer differ from the exact answer's and
how often the hit or miss decision at the certified lambda changes.
"""

import time

import numpy as np

from recal.micro import e10, fit, topk, world as w
from recal.micro.e4 import sq_dist

NAME = "E7"
DEGREES = (4, 8, 12, 16)
EFS = (16, 32, 64, 128, 256, 512, 1024)
FAST_EFS = (32, 64, 128)
RERANK_FACTOR = 4
RERANK_BLOCK = 500
EF_CONSTRUCTION = 200
EXACT_TARGET = 0.99
FAST_TARGET = 0.90
MAX_QUERIES = 5000
BLOCK_ELEMENTS = 100_000_000


def exact_top(pool: np.ndarray, Q: np.ndarray, k: int) -> np.ndarray:
    block = max(1, BLOCK_ELEMENTS // max(len(pool), 1))
    out = np.empty((len(Q), k), np.int64)
    for lo in range(0, len(Q), block):
        D = sq_dist(Q[lo:lo + block], pool)
        for i in range(len(D)):
            out[lo + i] = topk.exact_topk(pool, Q[lo + i], D[i], k)[0]
    return out


def id_bytes_for(n: int) -> int:
    return 2 if n < 65_536 else 3 if n < 16_777_216 else 4


def edge_bytes_per_vector(degree: int, id_bytes: int) -> float:
    level0 = 2 * degree * id_bytes + 4
    upper = (degree * id_bytes + 4) / max(degree - 1, 1)
    return level0 + upper


def same_set(found: np.ndarray, exact: np.ndarray) -> np.ndarray:
    return np.array([set(a) == set(b) for a, b in zip(np.sort(found, axis=1), np.sort(exact, axis=1))])


def rerank(pool: np.ndarray, Q: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """The k nearest of each row's candidate ids, by exact distance."""
    out = np.empty((len(Q), k), np.int64)
    for lo in range(0, len(Q), RERANK_BLOCK):
        cand = labels[lo:lo + RERANK_BLOCK]
        diff = pool[cand].astype(np.float64) - Q[lo:lo + RERANK_BLOCK, None, :].astype(np.float64)
        order = np.argsort((diff ** 2).sum(axis=2), axis=1)[:, :k]
        out[lo:lo + RERANK_BLOCK] = np.take_along_axis(cand, order, axis=1)
    return out


def run(world: w.World, max_queries: int = MAX_QUERIES, units=None) -> tuple[list[dict], dict]:
    pool = world.pool_vectors()
    k = world.k
    held = world.heldout()
    Q = held["vec"]
    if len(Q) > max_queries:
        Q = Q[np.linspace(0, len(Q) - 1, max_queries).astype(int)]
    if len(pool) < k:
        for degree in DEGREES:
            if units is not None:
                units.cached(f"degree{degree}", list)
        return [], dict(workload=world.name, budget=world.budget, pool_vectors=len(pool), note="pool smaller than K")
    exact = exact_top(pool, Q, k)
    decisions = decision_context(world, Q)
    rows = []
    for degree in DEGREES:
        compute = lambda degree=degree: degree_rows(world, pool, Q, exact, decisions, degree)  # noqa: E731
        rows += units.cached(f"degree{degree}", compute, queries=len(Q)) if units is not None else compute()
    return rows, summary(world.name, world.budget, rows)


def decision_context(world: w.World, Q: np.ndarray) -> dict | None:
    """Linear v11 model, its certified lambda and the exact answer's decisions on the queries (None: no model)."""
    ctx = e10.context(world)
    if ctx["model"] is None or not ctx["cal"]["has"].any():
        return None
    p, cal = world.p, ctx["cal"]
    ok = cal["has"]
    lam = fit.certify(ctx["train_scores"], cal["score"][ok], cal["y"][ok], p.T, p.delta, p.min_ess).lam
    F = ctx["feats"].exact(Q)
    has = np.isfinite(F).all(axis=1)
    score = np.full(len(Q), -np.inf)
    score[has] = fit.score(ctx["model"], F[has])
    return dict(ctx=ctx, lam=lam, F=F, has=has, hit=fit.decide(score, has, lam))


def fast_path_columns(decisions: dict | None, Q: np.ndarray, pool_dists: np.ndarray) -> dict:
    """Feature differences and decision changes when nr and the answer come from the approximate result."""
    if decisions is None or decisions["lam"] is None:
        return dict(fast_unchanged_decision_share=float("nan"), fast_offset_diff_p95=float("nan"),
                    fast_size_diff_p95=float("nan"), fast_rate_diff_p95=float("nan"))
    ctx = decisions["ctx"]
    nr = np.sqrt(np.maximum(pool_dists.max(axis=1), 0.0))
    F = ctx["feats"].exact(Q, nr=nr)
    has = np.isfinite(F).all(axis=1)
    score = np.full(len(Q), -np.inf)
    score[has] = fit.score(ctx["model"], F[has])
    hit = fit.decide(score, has, decisions["lam"])
    diff = np.abs(F - decisions["F"])
    keep = has & decisions["has"]
    return dict(fast_unchanged_decision_share=float((hit == decisions["hit"]).mean()),
                fast_offset_diff_p95=pct(diff[keep, 0]), fast_size_diff_p95=pct(diff[keep, 1]),
                fast_rate_diff_p95=pct(diff[keep, 2]))


def pct(x: np.ndarray) -> float:
    return float(np.percentile(x, 95)) if len(x) else float("nan")


def degree_rows(world: w.World, pool: np.ndarray, Q: np.ndarray, exact: np.ndarray, decisions, degree: int) -> list:
    import hnswlib
    k = world.k
    vector_bytes = pool.shape[1] * world.p.coord_bytes
    index = hnswlib.Index(space="l2", dim=pool.shape[1])
    t0 = time.perf_counter()
    index.init_index(max_elements=len(pool), ef_construction=EF_CONSTRUCTION, M=degree)
    index.set_num_threads(1)     # a threaded build gives a different graph on every run
    index.add_items(pool, np.arange(len(pool)))
    build_s = time.perf_counter() - t0
    ib = id_bytes_for(len(pool))
    k_rerank = min(RERANK_FACTOR * k, len(pool))
    rows = []
    for ef in EFS:
        index.set_ef(max(ef, k))
        t0 = time.perf_counter()
        labels, dists = index.knn_query(Q, k=k, num_threads=1)
        per_query = (time.perf_counter() - t0) / len(Q)
        labels = labels.astype(np.int64)
        recall = np.array([len(set(a) & set(b)) / k for a, b in zip(labels, exact)])
        index.set_ef(max(ef, k_rerank))
        t0 = time.perf_counter()
        wide, _ = index.knn_query(Q, k=k_rerank, num_threads=1)
        reranked = rerank(pool, Q, wide.astype(np.int64), k)
        rerank_per_query = (time.perf_counter() - t0) / len(Q)
        fast = fast_path_columns(decisions, Q, dists) if ef in FAST_EFS else {}
        rows.append(dict(
            workload=world.name, budget=world.budget, degree=degree, ef=ef, pool_vectors=len(pool),
            n_queries=len(Q), exact_share=float(same_set(labels, exact).mean()), mean_recall=float(recall.mean()),
            us_per_query=per_query * 1e6, rerank_exact_share=float(same_set(reranked, exact).mean()),
            rerank_us_per_query=rerank_per_query * 1e6, build_s=build_s, id_bytes_pool=ib,
            edge_bytes_per_vector_4=edge_bytes_per_vector(degree, 4),
            edge_bytes_per_vector_sized=edge_bytes_per_vector(degree, ib),
            edge_share_of_vector_4=edge_bytes_per_vector(degree, 4) / vector_bytes,
            edge_share_of_vector_sized=edge_bytes_per_vector(degree, ib) / vector_bytes,
            **{**dict(fast_unchanged_decision_share=float("nan"), fast_offset_diff_p95=float("nan"),
                      fast_size_diff_p95=float("nan"), fast_rate_diff_p95=float("nan")), **fast}))
    return rows


def summary(name: str, budget: float, rows: list[dict]) -> dict:
    """Smallest degree (then ef) with 99% exact answers, and the smallest ef reaching 90% (per degree and overall)."""
    exact = [r for r in rows if r["exact_share"] >= EXACT_TARGET]
    best = min(exact, key=lambda r: (r["degree"], r["ef"])) if exact else None
    fast = {}
    for degree in DEGREES:
        ok = [r["ef"] for r in rows if r["degree"] == degree and r["exact_share"] >= FAST_TARGET]
        fast[degree] = min(ok) if ok else None
    reached = [e for e in fast.values() if e is not None]
    rerank_ok = [r for r in rows if r["rerank_exact_share"] >= EXACT_TARGET]
    rerank_best = min(rerank_ok, key=lambda r: (r["degree"], r["ef"])) if rerank_ok else None
    unchanged = [r for r in rows if r["fast_unchanged_decision_share"] >= EXACT_TARGET]
    unchanged_best = min(unchanged, key=lambda r: (r["degree"], r["ef"])) if unchanged else None
    return dict(workload=name, budget=budget, pool_vectors=rows[0]["pool_vectors"],
                smallest_exact_99=None if best is None else dict(degree=best["degree"], ef=best["ef"],
                                                                 edge_bytes_per_vector_sized=best[
                                                                     "edge_bytes_per_vector_sized"]),
                smallest_ef_90_by_degree=fast, smallest_ef_90=min(reached) if reached else None,
                smallest_rerank_exact_99=None if rerank_best is None else dict(
                    degree=rerank_best["degree"], ef=rerank_best["ef"]),
                smallest_fast_unchanged_decisions_99=None if unchanged_best is None else dict(
                    degree=unchanged_best["degree"], ef=unchanged_best["ef"]))
