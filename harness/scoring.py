"""Score a replay: per-request recall against the exact top-10, hit rates, kinds."""

from dataclasses import dataclass

import numpy as np

from workloads.q1.recall import recall_batch

MISS, EQUALITY, APPROX, LINKED = 0, 1, 2, 3   # LINKED counts as approximate


@dataclass
class RunScore:
    recall: np.ndarray          # per request, always against the exact top-10
    hit: np.ndarray             # bool per request
    equality: np.ndarray        # bool: served by an entry stored under this exact vector id
    approximate: np.ndarray     # bool: any other hit (filter hit or linked repeat)
    seen_before_hit: np.ndarray  # diagnostic: hit on a vector that appeared earlier
    window: slice               # requests that are measured (the test window)

    def summary(self) -> dict:
        w = self.window
        n = len(self.recall[w])
        return {"requests_measured": n,
                "hit_rate": float(self.hit[w].mean()),
                "equality_hit_rate": float(self.equality[w].mean()),
                "approximate_hit_rate": float(self.approximate[w].mean()),
                "delivered_recall": float(self.recall[w].mean()),
                "recall_on_hits": float(self.recall[w][self.hit[w]].mean()) if self.hit[w].any() else None,
                "recall_on_misses": (float(self.recall[w][~self.hit[w]].mean())
                                     if (~self.hit[w]).any() else None)}


def seen_before(stream_ids: np.ndarray) -> np.ndarray:
    """True where the request's vector id appeared at an earlier request."""
    _, first = np.unique(stream_ids, return_index=True)
    out = np.ones(len(stream_ids), dtype=bool)
    out[first] = False
    return out


def score_run(
    outcome: np.ndarray,
    result_ids: np.ndarray,
    stream_ids: np.ndarray,
    request_vecs: np.ndarray,
    base: np.ndarray,
    kth_dist_of_request: np.ndarray,
    history_len: int,
) -> RunScore:
    """Recall of each request's returned list (hit list or backend answer).

    Equality is the driver's strict outcome code, not "seen before".
    """
    if not (len(outcome) == len(result_ids) == len(stream_ids) == len(request_vecs)):
        raise ValueError("per-request arrays have different lengths")
    hit = outcome != MISS
    seen = seen_before(stream_ids)
    return RunScore(
        recall=recall_batch(request_vecs, result_ids, base, kth_dist_of_request),
        hit=hit, equality=outcome == EQUALITY, approximate=(outcome == APPROX) | (outcome == LINKED),
        seen_before_hit=hit & seen, window=slice(history_len, len(outcome)))
