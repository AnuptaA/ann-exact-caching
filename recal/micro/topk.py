"""K nearest cached vectors with the core's rule: float64 L2, ties go to the lower slot (the lower id)."""

import numpy as np

MARGIN = 3


def exact_topk(vectors: np.ndarray, q: np.ndarray, approx: np.ndarray, k: int,
               slots: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Slots and exact distances of the k nearest, from approximate distances `approx` (aligned with `slots`).

    A shortlist of MARGIN * k by the approximate distance is re-ranked by exact float64 distance and slot, so a
    duplicate vector with the lower id wins like in the core's (distance, slot) order. `slots` must ascend."""
    cand = np.arange(len(approx)) if slots is None else np.asarray(slots)
    s = min(MARGIN * k, len(cand))
    short = cand[np.argpartition(approx, s - 1)[:s]] if s < len(cand) else cand
    exact = np.linalg.norm(vectors[short].astype(np.float64) - q.astype(np.float64), axis=1)
    order = np.lexsort((short, exact))[:k]
    return short[order], exact[order]
