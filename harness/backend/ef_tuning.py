"""ef_search tuning: mean recall@10 and latency over a grid, then the selection rule."""

import time

import numpy as np

from workloads.q1.recall import recall_batch

GRID = (10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400)
TARGET_RECALL = 0.9


def sweep(backend, queries, base, kth_dist, grid=GRID, list_size=10) -> list[dict]:
    """Mean recall@10 (against exact) and per-query latency at each ef_search."""
    rows = []
    for ef in grid:
        if ef < list_size:
            continue
        start = time.perf_counter()
        ids = backend.search(queries, list_size, ef)
        elapsed = time.perf_counter() - start
        rows.append({"ef_search": ef,
                     "recall": float(recall_batch(queries, ids, base, kth_dist).mean()),
                     "latency_ms": 1000.0 * elapsed / max(len(queries), 1)})
    return rows


def choose_ef(rows: list[dict], list_size: int, target: float = TARGET_RECALL) -> dict:
    """max(smallest ef_search reaching the target recall, list_size)."""
    reaching = [r for r in rows if r["recall"] >= target]
    if not reaching:
        raise ValueError(f"no ef_search in the grid reaches recall {target}; best "
                         f"{max(r['recall'] for r in rows):.3f}")
    smallest = min(reaching, key=lambda r: r["ef_search"])
    chosen = max(smallest["ef_search"], list_size)
    at_chosen = min((r for r in rows if r["ef_search"] >= chosen), key=lambda r: r["ef_search"],
                    default=smallest)
    return {"ef_search": chosen, "smallest_reaching_target": smallest["ef_search"],
            "list_size_binds": chosen > smallest["ef_search"], "grid_recall": at_chosen["recall"]}
