"""Byte accounting of every compared system against one budget M (percent of the dataset's stored bytes).

Everything a system keeps in RAM to serve is charged: vectors at their stored precision, ids, graphs and
indexes, entry metadata, models and its own caches. The formulas are nominal sizes, as in the spec (S9).
"""

ID_BYTES = 8
FLOAT_BYTES = 4


def budget_bytes(n: int, d: int, fraction: float, coord_bytes: int = 4) -> float:
    """M: a fraction of the dataset's stored bytes."""
    return fraction * n * d * coord_bytes


def aker_bytes(pool_vectors: int, entries: int, k: int, delta: int, d: int, aliases: int = 0,
               filter_links: int = 4) -> dict:
    """Aker: pooled neighbor vectors, one representative query vector per entry (the filter index stores
    it), the filter's graph links, per-entry metadata (k + delta neighbor references, tau, minD, flags)."""
    out = {
        "neighbor_pool": pool_vectors * (FLOAT_BYTES * d + ID_BYTES),
        "representatives": entries * FLOAT_BYTES * d,
        "filter_graph": entries * (2 * filter_links * 4 + 8),
        "entry_metadata": entries * (4 * (k + delta) + 24),
        "aliases": aliases * 12,
    }
    out["total"] = sum(out.values())
    return out


def qvcache_bytes(vectors: int, d: int, graph_degree: int = 0, regions: int = 0) -> dict:
    """QVCache: cached vectors with ids, the mini-indexes' graph links (0 for exact search), the theta table."""
    out = {
        "vectors": vectors * (FLOAT_BYTES * d + ID_BYTES),
        "graphs": vectors * graph_degree * 4,
        "theta_table": regions * 16,
    }
    out["total"] = sum(out.values())
    return out


def prototype_bytes(vectors: int, shell_samples: int, entries: int, k_mean: float, d: int, bits: int = 4) -> dict:
    """The older prototype (online_sim.Cache): cached vectors, 4-bit shell samples, centers and id lists."""
    out = {
        "vectors": vectors * (FLOAT_BYTES * d + ID_BYTES),
        "shell_samples": shell_samples * (bits * d / 8 + ID_BYTES),
        "entries": entries * (FLOAT_BYTES * d + 4 * k_mean + 16),
    }
    out["total"] = sum(out.values())
    return out


def fits(total: float, M: float) -> bool:
    return total <= M


def scale_pool(pool: int, total: float, M: float) -> int:
    """Pool size for a rerun when a run used `total` bytes at `pool` vectors and M is smaller."""
    return max(1, int(pool * M / total * 0.995))
