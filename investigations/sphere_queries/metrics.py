"""Relative contrast, hubness, norm and cosine statistics for query sets against a base."""

import numpy as np
from scipy.stats import skew

TOP = 10


def topk_and_median(queries: np.ndarray, base: np.ndarray, k: int = TOP,
                    chunk: int = 250) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-k ids, k-th NN Euclidean distance and median distance per query (float32)."""
    base = np.asarray(base, dtype=np.float32)
    base_sq = np.einsum("ij,ij->i", base, base)
    ids = np.empty((len(queries), k), dtype=np.int64)
    kth = np.empty(len(queries))
    med = np.empty(len(queries))
    for start in range(0, len(queries), chunk):
        q = np.asarray(queries[start:start + chunk], dtype=np.float32)
        d2 = np.einsum("ij,ij->i", q, q)[:, None] + base_sq[None, :] - 2.0 * (q @ base.T)
        np.maximum(d2, 0.0, out=d2)
        part = np.argpartition(d2, k - 1, axis=1)[:, :k]
        order = np.argsort(np.take_along_axis(d2, part, 1), axis=1)
        top = np.take_along_axis(part, order, 1)
        ids[start:start + len(q)] = top
        kth[start:start + len(q)] = np.sqrt(np.take_along_axis(d2, top, 1)[:, -1])
        med[start:start + len(q)] = np.sqrt(np.median(d2, axis=1))
    return ids, kth, med


def relative_contrast(kth: np.ndarray, med: np.ndarray) -> float:
    """Median over queries of the k-th NN distance over the median base distance."""
    return float(np.median(kth) / np.median(med))


def hubness(ids: np.ndarray, n_base: int, top_hubs: int = 100) -> dict:
    """Skewness of k-occurrence counts and the share held by the most frequent vectors."""
    counts = np.bincount(ids.ravel(), minlength=n_base)
    share = float(np.sort(counts)[::-1][:top_hubs].sum() / counts.sum())
    return {"skew": float(skew(counts)), "top_share": share,
            "never_retrieved": float(np.mean(counts == 0)), "max_count": int(counts.max())}


def norm_stats(queries: np.ndarray, base: np.ndarray) -> dict:
    q = np.asarray(queries, dtype=np.float64)
    b = np.asarray(base, dtype=np.float64)
    qn, bn = np.linalg.norm(q, axis=1), np.linalg.norm(b, axis=1)
    mq, mb = q.mean(0), b.mean(0)
    cos_means = float(mq @ mb / (np.linalg.norm(mq) * np.linalg.norm(mb)))
    return {"q_norm_med": float(np.median(qn)), "b_norm_med": float(np.median(bn)),
            "cos_mean_q_mean_b": cos_means,
            "mean_dist": float(np.linalg.norm(mq - mb))}


def nn_cosine(queries: np.ndarray, base: np.ndarray, ids: np.ndarray) -> float:
    """Median cosine between each query and its nearest neighbour."""
    q = np.asarray(queries, dtype=np.float64)
    nn = np.asarray(base, dtype=np.float64)[ids[:, 0]]
    cos = (q * nn).sum(1) / (np.linalg.norm(q, axis=1) * np.linalg.norm(nn, axis=1))
    return float(np.median(cos))


def stream_skew(vectors: np.ndarray) -> dict:
    """Skew of a released stream by exact vector identity."""
    _, inverse, counts = np.unique(np.ascontiguousarray(vectors), axis=0,
                                   return_inverse=True, return_counts=True)
    order = np.sort(counts)[::-1]
    total = counts.sum()
    return {"requests": int(total), "distinct": int(len(counts)),
            "top1_share": float(order[0] / total),
            "top10_share": float(order[:10].sum() / total),
            "top1pct_share": float(order[:max(1, len(order) // 100)].sum() / total)}
