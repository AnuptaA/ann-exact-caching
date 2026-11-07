"""Exact top-k ground truth: GPU (or numpy) candidates, CPU float32 recompute.

Chunked over queries and over the base, resumable per query chunk. Candidates
come from a matmul-based distance (fast, slightly inexact); the final top-k is
recomputed from differences with the shared distance function and re-sorted.
When more vectors tie at the k-th distance than the candidate margin holds, which
tied ids are kept can differ between devices; the distances are always exact.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np

from workloads.q1.distance import l2_to_ids

logger = logging.getLogger(__name__)

TOP_K = 100
CANDIDATE_MARGIN = 28
QUERY_CHUNK = 1024
BASE_CHUNK = 262144


def _to_f32(rows: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(rows, dtype=np.float32)


def _candidates_numpy(queries: np.ndarray, base, n_cand: int, base_chunk: int) -> np.ndarray:
    """Candidate ids per query by matmul distance, running top over base chunks."""
    q_sq = np.einsum("ij,ij->i", queries, queries)
    best_d = np.full((len(queries), n_cand), np.inf, dtype=np.float32)
    best_i = np.full((len(queries), n_cand), -1, dtype=np.int64)
    for start in range(0, len(base), base_chunk):
        chunk = _to_f32(base[start:start + base_chunk])
        d = q_sq[:, None] + np.einsum("ij,ij->i", chunk, chunk)[None, :] - 2.0 * (queries @ chunk.T)
        ids = np.broadcast_to(np.arange(start, start + len(chunk)), d.shape)
        all_d = np.concatenate([best_d, d], axis=1)
        all_i = np.concatenate([best_i, ids], axis=1)
        keep = np.argpartition(all_d, n_cand - 1, axis=1)[:, :n_cand]
        best_d = np.take_along_axis(all_d, keep, axis=1)
        best_i = np.take_along_axis(all_i, keep, axis=1)
    return best_i


def free_device_bytes(device: str) -> int:
    """Free memory on the device (unknown devices count as unlimited)."""
    if device.startswith("cuda"):
        import torch
        return torch.cuda.mem_get_info()[0]
    return 1 << 62


def fit_chunks(device: str, query_chunk: int, base_chunk: int, margin: float = 0.4) -> tuple[int, int]:
    """Shrink the chunks until the (queries x base) float32 tile, its concatenation with the running
    candidates and the matmul workspace (about three tiles) fit in `margin` of the free memory."""
    budget = margin * free_device_bytes(device)
    while 3 * 4 * query_chunk * base_chunk > budget and base_chunk > 4096:
        base_chunk //= 2
    while 3 * 4 * query_chunk * base_chunk > budget and query_chunk > 64:
        query_chunk //= 2
    return query_chunk, base_chunk


class DeviceBase:
    """The float32 base uploaded once, in chunks, when it fits on the device."""

    def __init__(self, chunks, base_chunk: int):
        self.chunks, self.base_chunk = chunks, base_chunk

    @classmethod
    def try_load(cls, base, device: str, base_chunk: int, fraction: float = 0.5):
        if device == "numpy":
            return None
        nbytes = base.shape[0] * base.shape[1] * 4
        if nbytes > fraction * free_device_bytes(device):
            logger.info(f"base ({nbytes / 2 ** 30:.1f} GiB) does not fit on {device}; streaming chunks")
            return None
        import torch
        logger.info(f"uploading the base ({nbytes / 2 ** 30:.1f} GiB) to {device} once")
        chunks = [torch.from_numpy(_to_f32(base[i:i + base_chunk])).to(device)
                  for i in range(0, len(base), base_chunk)]
        return cls(chunks, base_chunk)


def _candidates_torch(queries: np.ndarray, base, n_cand: int, base_chunk: int,
                      device: str, resident: "DeviceBase | None" = None) -> np.ndarray:
    import torch
    q = torch.from_numpy(queries).to(device)
    q_sq = (q * q).sum(1)
    best_d = torch.full((len(q), n_cand), float("inf"), device=device)
    best_i = torch.full((len(q), n_cand), -1, dtype=torch.int64, device=device)
    n = len(base)
    for k, start in enumerate(range(0, n, base_chunk)):
        if resident is not None:
            chunk = resident.chunks[k]
        else:
            chunk = torch.from_numpy(_to_f32(base[start:start + base_chunk])).to(device)
        d = q_sq[:, None] + (chunk * chunk).sum(1)[None, :] - 2.0 * (q @ chunk.T)
        ids = torch.arange(start, start + len(chunk), device=device).expand_as(d)
        all_d = torch.cat([best_d, d], dim=1)
        all_i = torch.cat([best_i, ids], dim=1)
        best_d, pos = torch.topk(all_d, n_cand, dim=1, largest=False)
        best_i = torch.gather(all_i, 1, pos)
    return best_i.cpu().numpy()


def exact_topk_chunk(
    queries: np.ndarray,
    base,
    k: int = TOP_K,
    device: str = "numpy",
    base_chunk: int = BASE_CHUNK,
    margin: int = CANDIDATE_MARGIN,
    resident: "DeviceBase | None" = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(ids, euclidean distances) of the exact top-k for one chunk of queries."""
    if k > len(base):
        raise ValueError(f"k={k} exceeds base size {len(base)}")
    queries = _to_f32(queries)
    n_cand = min(k + margin, len(base))
    if device == "numpy":
        cand = _candidates_numpy(queries, base, n_cand, base_chunk)
    else:
        cand = _candidates_torch(queries, base, n_cand, base_chunk, device, resident)
    ids = np.empty((len(queries), k), dtype=np.int64)
    dist = np.empty((len(queries), k), dtype=np.float32)
    for i, (q, c) in enumerate(zip(queries, cand)):
        d = l2_to_ids(q, base, c)
        order = np.lexsort((c, d))[:k]   # distance, then id: deterministic ties
        ids[i], dist[i] = c[order], d[order]
    return ids, dist


def build_ground_truth(
    queries: np.ndarray,
    base,
    out_dir: Path,
    k: int = TOP_K,
    device: str = "numpy",
    query_chunk: int = QUERY_CHUNK,
    base_chunk: int = BASE_CHUNK,
) -> tuple[np.ndarray, np.ndarray]:
    """Resumable ground truth; chunk files are written atomically and reused."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"n_queries": len(queries), "n_base": len(base), "k": k, "query_chunk": query_chunk}
    meta_path = out_dir / "meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
        raise ValueError(f"{meta_path} was written for a different run: {meta_path.read_text()}")
    meta_path.write_text(json.dumps(meta))
    query_chunk_used, base_chunk_used = query_chunk, base_chunk
    if device != "numpy":
        query_chunk_used, base_chunk_used = fit_chunks(device, query_chunk, base_chunk)
        if (query_chunk_used, base_chunk_used) != (query_chunk, base_chunk):
            logger.warning(f"device memory is small: chunks reduced to queries={query_chunk_used} "
                           f"base={base_chunk_used}")
    # the resume metadata fixes the query chunk size; only the base chunk may change
    resident = DeviceBase.try_load(base, device, base_chunk_used)
    n_chunks = -(-len(queries) // query_chunk)
    began, built = time.time(), 0
    for c, start in enumerate(range(0, len(queries), query_chunk)):
        path = out_dir / f"chunk_{start:09d}.npz"
        if path.exists():
            continue
        block = queries[start:start + query_chunk]
        parts = [exact_topk_chunk(block[i:i + query_chunk_used], base, k, device, base_chunk_used,
                                  resident=resident) for i in range(0, len(block), query_chunk_used)]
        ids = np.concatenate([p[0] for p in parts])
        dist = np.concatenate([p[1] for p in parts])
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, ids=ids, dist=dist)
        tmp.rename(path)
        built += 1
        per_chunk = (time.time() - began) / built
        remaining = n_chunks - c - 1
        logger.info(f"ground truth chunk {c + 1}/{n_chunks} ({start + len(ids)}/{len(queries)} queries, "
                    f"{len(ids) / per_chunk:.0f} q/s, ETA {remaining * per_chunk / 60:.1f} min)")
    return load_ground_truth(out_dir)


def load_ground_truth(out_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "meta.json").read_text())
    files = sorted(out_dir.glob("chunk_*.npz"))
    expected = -(-meta["n_queries"] // meta["query_chunk"])
    if len(files) != expected:
        raise FileNotFoundError(f"{len(files)} of {expected} chunks present in {out_dir}")
    parts = [np.load(f) for f in files]
    return (np.concatenate([p["ids"] for p in parts]),
            np.concatenate([p["dist"] for p in parts]))
