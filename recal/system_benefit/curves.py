"""Measurements of the backend against its node cache and search depth (run on the cluster).

The measuring functions take the index opener and the searcher as arguments, so they are tested with mocks.
"""

import json
import time
from pathlib import Path

import numpy as np

SECTOR = 4096


def read_bytes_counter() -> float:
    """Bytes this process has read from storage (Linux /proc/self/io)."""
    with open("/proc/self/io") as f:
        for line in f:
            if line.startswith("read_bytes:"):
                return float(line.split()[1])
    raise OSError("no read_bytes in /proc/self/io")


def measure_node_cache_curve(open_index, queries, node_counts, node_bytes: float, k: int, L: int,
                             n_warm: int = 100, io_counter=read_bytes_counter) -> dict:
    """Reads per search and latency for each node cache size.

    open_index(nodes) returns an index with search(q, k_neighbors, complexity, beam_width); reads are the
    storage bytes read per search divided by the 4 KiB sector.
    """
    rows = []
    for nodes in node_counts:
        index = open_index(nodes)
        for q in queries[:n_warm]:
            index.search(q, k_neighbors=k, complexity=L, beam_width=2)
        before, began = io_counter(), time.perf_counter()
        lat = []
        for q in queries:
            t0 = time.perf_counter()
            index.search(q, k_neighbors=k, complexity=L, beam_width=2)
            lat.append(time.perf_counter() - t0)
        reads = (io_counter() - before) / SECTOR / max(len(queries), 1)
        rows.append({"nodes": int(nodes), "cache_bytes": float(nodes * node_bytes), "reads_per_search": reads,
                     "latency_ms": float(np.mean(lat) * 1e3), "p99_ms": float(np.percentile(lat, 99) * 1e3)})
        del index
    return {"k": k, "L": L, "rows": rows}


def measure_depth_latency(search, queries, depths, repeats: int = 1) -> dict:
    """Mean latency of search(q, depth) per depth, in the format DepthLatency.load reads."""
    mean_ms = []
    for depth in depths:
        times = []
        for _ in range(repeats):
            for q in queries:
                t0 = time.perf_counter()
                search(q, int(depth))
                times.append(time.perf_counter() - t0)
        mean_ms.append(float(np.mean(times) * 1e3))
    return {"depth_latency": {"depths": [int(d) for d in depths], "mean_ms": mean_ms, "n": len(queries) * repeats}}


def table_depths(k_values, delta: int = 0, factor: int = 7) -> list[int]:
    """Depths a workload needs: k, k + delta and factor x k for every K in use."""
    out = set()
    for k in k_values:
        out.update({int(k), int(k) + int(delta), factor * int(k)})
    return sorted(out)


def save(path, table: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, indent=1))
