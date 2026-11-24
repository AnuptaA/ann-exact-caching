"""Memory split s between ReCal (s x B) and DiskANN's node cache ((1 - s) x B)."""

import numpy as np

SPLITS = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)


def arm_budgets(total_bytes: float, s: float) -> dict:
    if not 0.0 <= s <= 1.0:
        raise ValueError("split must be in [0, 1]")
    return {"split": s, "recal_bytes": s * total_bytes, "node_cache_bytes": (1.0 - s) * total_bytes}


class ReadsCurve:
    """Disk reads per backend search against node cache bytes (script node_cache_curve.py)."""

    def __init__(self, cache_bytes, reads_per_search):
        order = np.argsort(cache_bytes)
        self.bytes = np.asarray(cache_bytes, float)[order]
        self.reads = np.asarray(reads_per_search, float)[order]
        if len(self.bytes) < 2 or (self.reads <= 0).any():
            raise ValueError("a node cache curve needs at least two positive points")

    def at(self, cache_bytes: float) -> float:
        if cache_bytes < self.bytes[0] or cache_bytes > self.bytes[-1]:
            raise ValueError(f"{cache_bytes} bytes is outside the measured curve "
                             f"[{self.bytes[0]}, {self.bytes[-1]}]")
        return float(np.interp(cache_bytes, self.bytes, self.reads))


def expected_reads(hit_rate: float, search_reads: float, background_reads: float) -> float:
    """Disk reads per request: misses search the backend; background searches and re-reads are extra."""
    return (1.0 - hit_rate) * search_reads + background_reads


def background_reads_per_request(background: dict, n_requests: int, reads_at_depth, rereads: float = 0.0) -> float:
    """Reads of ReCal's non-miss backend work per scored request.

    background: backend stats by kind ({kind: {searches, depth_sum}}), miss excluded here. reads_at_depth(depth)
    gives the reads of one search of that depth; rereads: vector re-reads (one read each) in the run.
    """
    total = rereads
    for kind, e in background.items():
        if kind == "miss" or e.get("searches", 0) <= 0:
            continue
        depth = e["depth_sum"] / e["searches"]
        total += e["searches"] * reads_at_depth(depth)
    return total / max(n_requests, 1)
