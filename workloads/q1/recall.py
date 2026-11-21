"""recall@10 with tie handling; always divides by 10."""

import numpy as np

from workloads.q1 import K
from workloads.q1.distance import l2_rows

TIE_RTOL = 1e-6


def recall_at_k(
    query: np.ndarray,
    answer_ids: np.ndarray,
    base: np.ndarray,
    kth_dist: float,
    k: int = K,
) -> float:
    """Share of the exact top-k present in the first k answer ids.

    A returned vector is correct if its distance is at most kth_dist * (1 + 1e-6).
    Ids outside the base (such as -1 padding) and repeated ids count as incorrect;
    missing answers count as incorrect because the divisor is always k.
    """
    ids = np.asarray(answer_ids, dtype=np.int64)[:k]
    seen = set()
    keep = []
    for pos, vid in enumerate(ids.tolist()):
        if 0 <= vid < len(base) and vid not in seen:
            seen.add(vid)
            keep.append(pos)
    if not keep:
        return 0.0
    rows = np.asarray(base[ids[keep]], dtype=np.float32)
    dist = l2_rows(np.broadcast_to(np.asarray(query, dtype=np.float32), rows.shape), rows)
    return float((dist <= kth_dist * (1.0 + TIE_RTOL)).sum()) / k


def recall_batch(
    queries: np.ndarray,
    answers: np.ndarray,
    base: np.ndarray,
    kth_dist: np.ndarray,
    k: int = K,
) -> np.ndarray:
    """recall_at_k for each row; answers is (n, m) with -1 padding."""
    return np.array([recall_at_k(q, a, base, d, k)
                     for q, a, d in zip(queries, answers, kth_dist)], dtype=np.float64)
