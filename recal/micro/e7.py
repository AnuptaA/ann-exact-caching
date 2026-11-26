"""E7: an HNSW over the cached vectors.

For degree in DEGREES and ef in EFS: how often the search returns exactly the K nearest cached vectors, the mean
recall against them, the time per query (one thread), and the edge memory per vector. Sizes of the graph are
computed: level 0 holds 2 x degree neighbor ids and a 4-byte count, upper layers an expected 1/(degree-1) of that.
"""

import time

import numpy as np

from recal.micro import world as w
from recal.micro.e4 import sq_dist

NAME = "E7"
DEGREES = (4, 8, 12, 16)
EFS = (16, 32, 64, 128, 256)
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
        out[lo:lo + block] = np.argpartition(D, k - 1, axis=1)[:, :k]
    return out


def id_bytes_for(n: int) -> int:
    return 2 if n < 65_536 else 3 if n < 16_777_216 else 4


def edge_bytes_per_vector(degree: int, id_bytes: int) -> float:
    level0 = 2 * degree * id_bytes + 4
    upper = (degree * id_bytes + 4) / max(degree - 1, 1)
    return level0 + upper


def same_set(found: np.ndarray, exact: np.ndarray) -> np.ndarray:
    return np.array([set(a) == set(b) for a, b in zip(np.sort(found, axis=1), np.sort(exact, axis=1))])


def run(world: w.World, max_queries: int = MAX_QUERIES) -> tuple[list[dict], dict]:
    import hnswlib
    pool = world.pool_vectors()
    k = world.k
    held = world.heldout()
    Q = held["vec"]
    if len(Q) > max_queries:
        Q = Q[np.linspace(0, len(Q) - 1, max_queries).astype(int)]
    if len(pool) < k:
        return [], dict(workload=world.name, budget=world.budget, pool_vectors=len(pool), note="pool smaller than K")
    exact = exact_top(pool, Q, k)
    vector_bytes = pool.shape[1] * world.p.coord_bytes
    rows = []
    for degree in DEGREES:
        index = hnswlib.Index(space="l2", dim=pool.shape[1])
        t0 = time.perf_counter()
        index.init_index(max_elements=len(pool), ef_construction=EF_CONSTRUCTION, M=degree)
        index.add_items(pool, np.arange(len(pool)))
        build_s = time.perf_counter() - t0
        index.set_num_threads(1)
        ib = id_bytes_for(len(pool))
        for ef in EFS:
            index.set_ef(max(ef, k))
            t0 = time.perf_counter()
            labels, _ = index.knn_query(Q, k=k, num_threads=1)
            per_query = (time.perf_counter() - t0) / len(Q)
            labels = labels.astype(np.int64)
            recall = np.array([len(set(a) & set(b)) / k for a, b in zip(labels, exact)])
            rows.append(dict(
                workload=world.name, budget=world.budget, degree=degree, ef=ef, pool_vectors=len(pool),
                n_queries=len(Q), exact_share=float(same_set(labels, exact).mean()), mean_recall=float(recall.mean()),
                us_per_query=per_query * 1e6, build_s=build_s, id_bytes_pool=ib,
                edge_bytes_per_vector_4=edge_bytes_per_vector(degree, 4),
                edge_bytes_per_vector_sized=edge_bytes_per_vector(degree, ib),
                edge_share_of_vector_4=edge_bytes_per_vector(degree, 4) / vector_bytes,
                edge_share_of_vector_sized=edge_bytes_per_vector(degree, ib) / vector_bytes))
    return rows, summary(world, rows)


def summary(world: w.World, rows: list[dict]) -> dict:
    """Smallest degree (then ef) with 99% exact answers, and the smallest ef reaching 90% (per degree and overall)."""
    exact = [r for r in rows if r["exact_share"] >= EXACT_TARGET]
    best = min(exact, key=lambda r: (r["degree"], r["ef"])) if exact else None
    fast = {}
    for degree in DEGREES:
        ok = [r["ef"] for r in rows if r["degree"] == degree and r["exact_share"] >= FAST_TARGET]
        fast[degree] = min(ok) if ok else None
    reached = [e for e in fast.values() if e is not None]
    return dict(workload=world.name, budget=world.budget, pool_vectors=rows[0]["pool_vectors"],
                smallest_exact_99=None if best is None else dict(degree=best["degree"], ef=best["ef"],
                                                                 edge_bytes_per_vector_sized=best[
                                                                     "edge_bytes_per_vector_sized"]),
                smallest_ef_90_by_degree=fast, smallest_ef_90=min(reached) if reached else None)
