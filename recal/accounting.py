"""Byte accounting of every compared system against one budget M (percent of the dataset's stored bytes).

Everything a system keeps in RAM to serve is charged: vectors at their stored precision, ids, graphs and
indexes, entry metadata, models and its own caches. The formulas are nominal sizes.
"""

import re

ID_BYTES = 8
FLOAT_BYTES = 4
# constants of the Aker library's own memory estimate (ak_anns_cache_telemetry.cc)
AKER_ENTRY_BYTES = 80
AKER_POOL_META_BYTES = 48
AKER_FILTER_LINK_BYTES = 4 * 8   # m = 4 neighbor identifiers of 8 bytes per filter entry


def budget_bytes(n: int, d: int, fraction: float, coord_bytes: int = 4) -> float:
    """M: a fraction of the dataset's stored bytes."""
    return fraction * n * d * coord_bytes


def aker_bytes(pool_vectors: int, entries: int, k: int, delta: int, d: int, aliases: int = 0,
               filter_links: int = 4, coord_bytes: int = FLOAT_BYTES) -> dict:
    """Aker: pooled neighbor vectors, one representative query vector per entry (the filter index stores
    it), the filter's graph links, per-entry metadata (k + delta neighbor references, tau, minD, flags)."""
    out = {
        "neighbor_pool": pool_vectors * (coord_bytes * d + ID_BYTES),
        "representatives": entries * coord_bytes * d,
        "filter_graph": entries * (2 * filter_links * 4 + 8),
        "entry_metadata": entries * (4 * (k + delta) + 24),
        "aliases": aliases * 12,
    }
    out["total"] = sum(out.values())
    return out


def qvcache_bytes(vectors: int, d: int, graph_degree: int = 0, regions: int = 0,
                  coord_bytes: int = FLOAT_BYTES) -> dict:
    """QVCache: cached vectors with ids, the mini-indexes' graph links (0 for exact search), the theta table."""
    out = {
        "vectors": vectors * (coord_bytes * d + ID_BYTES),
        "graphs": vectors * graph_degree * 4,
        "theta_table": regions * 16,
    }
    out["total"] = sum(out.values())
    return out


def prototype_bytes(vectors: int, shell_samples: int, entries: int, k_mean: float, d: int, bits: int = 4,
                    coord_bytes: int = FLOAT_BYTES) -> dict:
    """The earlier cache implementation (online_sim.Cache): cached vectors, 4-bit shell samples, centers and id lists."""
    out = {
        "vectors": vectors * (coord_bytes * d + ID_BYTES),
        "shell_samples": shell_samples * (bits * d / 8 + ID_BYTES),
        "entries": entries * (coord_bytes * d + 4 * k_mean + 16),
    }
    out["total"] = sum(out.values())
    return out


def fits(total: float, M: float) -> bool:
    return total <= M


def scale_pool(pool: int, total: float, M: float) -> int:
    """Pool size for a rerun when a run used `total` bytes at `pool` vectors and M is smaller."""
    return max(1, int(pool * M / total * 0.995))


AKER_STATUS_PATTERNS = {
    "total_entries": r"Total entries:\s*(\d+)",
    "physical_entries": r"Physical entries:\s*(\d+)",
    "pool_mb": r"Pool:\s*([0-9.]+) MB",
}


def parse_aker_status(text: str) -> dict:
    """Entry counts and the pool size from the Aker library's status text."""
    out = {}
    for name, pattern in AKER_STATUS_PATTERNS.items():
        found = re.search(pattern, text)
        if found is None:
            raise ValueError(f"Aker status has no {name}")
        out[name] = float(found.group(1)) if name.endswith("_mb") else int(found.group(1))
    return out


def aker_bytes_from_status(text: str, d: int, coord_bytes: int = FLOAT_BYTES) -> dict:
    """Aker's bytes at stored precision from the library's own counts.

    The library estimates its structures at float32; the counts (entries, physical entries,
    pooled vectors, recovered from its pool estimate) are re-charged at coord_bytes per
    coordinate with the library's per-structure overheads.
    """
    s = parse_aker_status(text)
    pool_vectors = round(s["pool_mb"] * 1e6 / (AKER_POOL_META_BYTES + FLOAT_BYTES * d))
    out = {
        "entries": s["total_entries"] * AKER_ENTRY_BYTES,
        "neighbor_pool": pool_vectors * (AKER_POOL_META_BYTES + coord_bytes * d),
        "representatives": s["physical_entries"] * coord_bytes * d,
        "filter_graph": s["physical_entries"] * AKER_FILTER_LINK_BYTES,
    }
    out["total"] = sum(out.values())
    out["pool_vectors"] = pool_vectors
    return out


def aker_pool_for_budget(M: float, d: int, k: int, coord_bytes: int = FLOAT_BYTES,
                         vectors_per_entry: float | None = None) -> int:
    """Pool slots whose pool, entries and filter fit M when each entry holds vectors_per_entry
    pooled vectors (default k: no sharing between entries, the worst case)."""
    share = float(k if vectors_per_entry is None else vectors_per_entry)
    per_vector = AKER_POOL_META_BYTES + coord_bytes * d
    per_entry = AKER_ENTRY_BYTES + coord_bytes * d + AKER_FILTER_LINK_BYTES
    return max(1, int(M / (per_vector + per_entry / share)))


def aker_unit_costs(d: int, coord_bytes: float = FLOAT_BYTES) -> dict:
    """Aker bytes per pooled neighbor object and per entry (with its filter vector and links)."""
    return {"per_object": AKER_POOL_META_BYTES + coord_bytes * d,
            "per_entry": AKER_ENTRY_BYTES + coord_bytes * d + AKER_FILTER_LINK_BYTES}


def aker_full_capacity_bytes(capacity: int, d: int, coord_bytes: float, objects_per_entry: float) -> float:
    """Bytes of a full cache: `capacity` pooled objects held by capacity / objects_per_entry entries."""
    c = aker_unit_costs(d, coord_bytes)
    return capacity * c["per_object"] + capacity / objects_per_entry * c["per_entry"]


def aker_capacity_for_budget(M: float, d: int, coord_bytes: float, objects_per_entry: float) -> int:
    """Pool capacity (neighbor objects) whose full cache costs M, given the pooled objects each entry holds."""
    c = aker_unit_costs(d, coord_bytes)
    return max(1, int(M / (c["per_object"] + c["per_entry"] / objects_per_entry)))


def calibrate_capacity(M: float, d: int, coord_bytes: float, measure_ratio, start: int, tol: float = 0.01) -> tuple:
    """Capacity so that a full cache costs M.

    measure_ratio(capacity) runs the library on the history with that pool capacity and returns the pooled
    objects per entry it ended with. The first measurement (at `start`) sets the capacity; one run at that
    capacity checks it and the capacity is set again from the new ratio when the full-cache bytes miss M by
    more than tol. Returns (capacity, info).
    """
    ratio = measure_ratio(start)
    capacity = aker_capacity_for_budget(M, d, coord_bytes, ratio)
    steps = [{"probe_capacity": start, "objects_per_entry": ratio, "capacity": capacity}]
    check = measure_ratio(capacity)
    full = aker_full_capacity_bytes(capacity, d, coord_bytes, check)
    if abs(full - M) > tol * M:
        capacity = aker_capacity_for_budget(M, d, coord_bytes, check)
        full = aker_full_capacity_bytes(capacity, d, coord_bytes, check)
        steps.append({"probe_capacity": steps[0]["capacity"], "objects_per_entry": check, "capacity": capacity,
                      "refined": True})
    return capacity, {"steps": steps, "objects_per_entry": check, "bytes_at_full_capacity": full,
                      "configured_capacity": capacity, "M": M}


def capacity_for_budget(system: str, M: float, d: int, coord_bytes: float, **kwargs) -> int:
    """The capacity knob of a system so that its full capacity costs M (aker: objects per entry required)."""
    if system in ("aker", "proximity"):
        return aker_capacity_for_budget(M, d, coord_bytes, kwargs["objects_per_entry"])
    if system == "qvcache":
        return qvcache_capacity_for_budget(M, d, coord_bytes, kwargs["R"], kwargs["pca_dim"], kwargs["regions"],
                                           kwargs.get("n_mini", 4))
    raise ValueError(f"no capacity rule for {system}")


# QVCache (artifact qvcache-vldb26): float32 mini-indexes of FreshVamana graphs, a PCA projection and a table of
# regional thresholds. Per cached point: the vector at stored precision, a 4-byte tag and the graph row of R
# 4-byte neighbors. Per active region: its key and a double threshold. PCA: pca_dim x dim float32 plus the mean.
QVCACHE_TAG_BYTES = 4
QVCACHE_REGION_BYTES = 16


def qvcache_unit_costs(d: int, coord_bytes: float, R: int) -> dict:
    return {"per_point": coord_bytes * d + QVCACHE_TAG_BYTES + 4.0 * R, "per_region": QVCACHE_REGION_BYTES}


def qvcache_fixed_bytes(d: int, pca_dim: int, regions: int) -> float:
    return pca_dim * d * 4.0 + d * 4.0 + regions * QVCACHE_REGION_BYTES


def qvcache_full_capacity_bytes(max_points: int, d: int, coord_bytes: float, R: int, pca_dim: int,
                                regions: int) -> float:
    return max_points * qvcache_unit_costs(d, coord_bytes, R)["per_point"] + qvcache_fixed_bytes(d, pca_dim, regions)


def qvcache_capacity_for_budget(M: float, d: int, coord_bytes: float, R: int, pca_dim: int, regions: int,
                                n_mini: int = 4) -> int:
    """memory_index_max_points whose full cache costs M, given the active regions measured on the history.

    The points are split evenly over the n_mini mini-indexes, so the result is a multiple of n_mini.
    """
    per_point = qvcache_unit_costs(d, coord_bytes, R)["per_point"]
    points = int((M - qvcache_fixed_bytes(d, pca_dim, regions)) // per_point)
    return max(n_mini, points // n_mini * n_mini)


def qvcache_bytes_used(vectors: int, regions: int, d: int, coord_bytes: float, R: int, pca_dim: int) -> dict:
    c = qvcache_unit_costs(d, coord_bytes, R)
    out = {"vectors": vectors * coord_bytes * d, "tags": vectors * QVCACHE_TAG_BYTES, "graphs": vectors * 4.0 * R,
           "pca": pca_dim * d * 4.0 + d * 4.0, "regions": regions * c["per_region"]}
    out["total"] = sum(out.values())
    return out
