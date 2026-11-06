#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Distance and exact nearest-neighbor primitives (numpy, no global state)."""

import numpy as np

#-------------------------------------------------------------------------------


def sq_norms(x):
    return np.einsum("ij,ij->i", x, x)


def pairwise_dist(a, b, b_sq=None):
    """Euclidean distances between rows of a and rows of b (float64)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if b_sq is None:
        b_sq = sq_norms(b)
    d2 = sq_norms(a)[:, None] + b_sq[None, :] - 2.0 * (a @ b.T)
    return np.sqrt(np.maximum(d2, 0.0))


def sq_dist(a, b, b_sq):
    """Squared Euclidean distances (float32), clipped at 0."""
    d = (a * a).sum(axis=1)[:, None] + b_sq[None, :] - 2.0 * (a @ b.T)
    return np.maximum(d, 0.0)


def fetch_rows(db, ids):
    """Rows of db (possibly memory-mapped) for ids, in ids order, float64."""
    ids = np.asarray(ids, dtype=np.int64)
    order = np.argsort(ids)
    rows = np.empty((len(ids), db.shape[1]))
    rows[order] = db[ids[order]]
    return rows


def exact_topk(queries: np.ndarray, db: np.ndarray, k: int, chunk: int = 1000) -> np.ndarray:
    """Exact top-k ids (sorted by distance) under squared Euclidean distance.

    Large databases (> 2M vectors) use exact_topk_blocked to bound memory.
    """
    if len(db) > 2_000_000:
        return exact_topk_blocked(queries, db, k)
    db64 = db.astype(np.float64)
    db_sq = np.einsum("ij,ij->i", db64, db64)
    out = np.empty((len(queries), k), dtype=np.int64)
    for s in range(0, len(queries), chunk):
        q = queries[s:s + chunk].astype(np.float64)
        d = db_sq[None, :] - 2.0 * (q @ db64.T)
        part = np.argpartition(d, k - 1, axis=1)[:, :k]
        order = np.argsort(np.take_along_axis(d, part, axis=1), axis=1)
        out[s:s + chunk] = np.take_along_axis(part, order, axis=1)
    return out


def exact_topk_blocked(queries: np.ndarray, db: np.ndarray, k: int,
                       q_chunk: int = 4096, db_block: int = 500_000) -> np.ndarray:
    """Exact top-k ids for large databases, blocked over queries and database.

    Two stages per database block. Screen: one float32 matmul of [q, 1] with
    [x, -|x|^2/2] gives s = q.x - |x|^2/2 (larger s = closer, since
    |q - x|^2 = |q|^2 - 2s); keep vectors with s >= (threshold implied by the
    current k-th best) - slack. Refine: exact float64 distances for the
    survivors, merged into the running top-k. Every true top-k vector passes
    the screen, so the result is exact. Ties are broken by lower id. Output
    rows are sorted by (distance, id).
    """
    n, dim = db.shape
    xa = np.empty((n, dim + 1), dtype=np.float32)
    db_sq = np.empty(n, dtype=np.float64)
    for b in range(0, n, db_block):
        xb = np.asarray(db[b:b + db_block], dtype=np.float64)
        db_sq[b:b + len(xb)] = np.einsum("ij,ij->i", xb, xb)
        xa[b:b + len(xb), :dim] = xb
        xa[b:b + len(xb), dim] = -0.5 * db_sq[b:b + len(xb)]
    max_db_sq = float(db_sq.max())
    out = np.empty((len(queries), k), dtype=np.int64)
    buf = np.empty((min(q_chunk, len(queries)), min(db_block, n)), dtype=np.float32)
    for s0 in range(0, len(queries), q_chunk):
        q64 = np.asarray(queries[s0:s0 + q_chunk], dtype=np.float64)
        c = len(q64)
        qa = np.ones((c, dim + 1), dtype=np.float32)
        qa[:, :dim] = q64
        q_sq = np.einsum("ij,ij->i", q64, q64)
        # float32 screening error is far below this slack
        slack = 1e-4 * (float(q_sq.max()) + max_db_sq + 1.0)
        best_d = np.full((c, k), np.inf)
        best_i = np.full((c, k), -1, dtype=np.int64)
        for b in range(0, n, db_block):
            nb = min(db_block, n - b)
            s = np.matmul(qa, xa[b:b + nb].T, out=buf[:c, :nb])
            if np.isinf(best_d[:, -1]).any():
                thr = np.partition(s, nb - k, axis=1)[:, nb - k].astype(np.float64)
                known = ~np.isinf(best_d[:, -1])
                thr[known] = np.maximum(thr[known], 0.5 * (q_sq[known] - best_d[known, -1]))
            else:
                thr = 0.5 * (q_sq - best_d[:, -1])
            rows, cols = np.nonzero(s >= (thr - slack)[:, None].astype(np.float32))
            ids = cols.astype(np.int64) + b
            diff = q64[rows] - np.asarray(db[ids], dtype=np.float64)
            d_new = np.einsum("ij,ij->i", diff, diff)
            # merge: k smallest by (distance, id) per row
            all_r = np.concatenate([np.repeat(np.arange(c), k), rows])
            all_d = np.concatenate([best_d.ravel(), d_new])
            all_i = np.concatenate([best_i.ravel(), ids])
            order = np.lexsort((all_i, all_d, all_r))
            all_r, all_d, all_i = all_r[order], all_d[order], all_i[order]
            starts = np.searchsorted(all_r, np.arange(c))
            keep = (np.arange(len(all_r)) - starts[all_r]) < k
            best_d = all_d[keep].reshape(c, k)
            best_i = all_i[keep].reshape(c, k)
        out[s0:s0 + c] = best_i
    return out


def knn_dists(q, db_c, db_c_sq, k, chunk=2048):
    """Sorted distances from each query to its k nearest rows of db_c (float64)."""
    out = np.empty((len(q), k))
    for s in range(0, len(q), chunk):
        d = pairwise_dist(q[s:s + chunk], db_c, db_c_sq)
        out[s:s + chunk] = np.sort(np.partition(d, k - 1, axis=1)[:, :k], axis=1)
    return out
