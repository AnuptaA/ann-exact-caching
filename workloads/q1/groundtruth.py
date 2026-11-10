"""Exact top-k ground truth: GPU (or numpy) candidates, CPU float32 recompute.

Chunked over queries and over the base, resumable per query chunk. Candidates
come from a matmul-based distance (fast, slightly inexact); the final top-k is
recomputed from differences with the shared distance function and re-sorted.
When more vectors tie at the k-th distance than the candidate margin holds, which
tied ids are kept can differ between devices; the distances are always exact.

The CPU path (device "cpu", alias "numpy") needs only numpy. Its tiles are sized
from the RAM that is available, and the candidate merge and the exact recompute
run on a thread pool. "torch-cpu" runs the torch path on the CPU (tests only).
"""

import contextlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from workloads.q1 import atomic, device as device_mod
from workloads.q1.distance import l2_to_ids
from workloads.q1.progress import Progress

logger = logging.getLogger(__name__)

TOP_K = 100
CANDIDATE_MARGIN = 28
QUERY_CHUNK = 1024
BASE_CHUNK = 262144
MIN_QUERIES_PER_WORKER = 16
CPU_RAM_FRACTION = 0.5    # share of the available RAM one tile and its base chunk may use


def is_cpu_numpy(device: str) -> bool:
    return device in ("numpy", "cpu")


def _torch_device(device: str) -> str:
    return "cpu" if device == "torch-cpu" else device


def _to_f32(rows: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(rows, dtype=np.float32)


def _split(n: int, parts: int) -> list[slice]:
    bounds = np.linspace(0, n, parts + 1).astype(int)
    return [slice(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def _merge_numpy(q, q_sq, chunk, chunk_sq, start, best_d, best_i, n_cand):
    d = q_sq[:, None] + chunk_sq[None, :] - 2.0 * (q @ chunk.T)
    ids = np.broadcast_to(np.arange(start, start + len(chunk)), d.shape)
    all_d = np.concatenate([best_d, d], axis=1)
    all_i = np.concatenate([best_i, ids], axis=1)
    keep = np.argpartition(all_d, n_cand - 1, axis=1)[:, :n_cand]
    return np.take_along_axis(all_d, keep, axis=1), np.take_along_axis(all_i, keep, axis=1)


def _candidates_numpy(queries: np.ndarray, base, n_cand: int, base_chunk: int,
                      pool: ThreadPoolExecutor | None = None, workers: int = 1,
                      progress=None) -> np.ndarray:
    """Candidate ids per query by matmul distance, running top over base chunks.

    With a pool, the queries are split over `workers` slices that merge each base chunk in parallel.
    `progress(fraction_of_base_done)` is called after every base chunk.
    """
    slices = _split(len(queries), workers)
    q_sq = np.einsum("ij,ij->i", queries, queries)
    best = [(np.full((s.stop - s.start, n_cand), np.inf, dtype=np.float32),
             np.full((s.stop - s.start, n_cand), -1, dtype=np.int64)) for s in slices]

    def step(j, chunk, chunk_sq, start):
        s = slices[j]
        best[j] = _merge_numpy(queries[s], q_sq[s], chunk, chunk_sq, start, *best[j], n_cand)

    for start in range(0, len(base), base_chunk):
        chunk = _to_f32(base[start:start + base_chunk])
        chunk_sq = np.einsum("ij,ij->i", chunk, chunk)
        if pool is None or len(slices) == 1:
            for j in range(len(slices)):
                step(j, chunk, chunk_sq, start)
        else:
            list(pool.map(lambda j: step(j, chunk, chunk_sq, start), range(len(slices))))
        if progress is not None:
            progress(min(1.0, (start + len(chunk)) / len(base)))
    return np.concatenate([b[1] for b in best])


def free_device_bytes(device: str) -> int:
    """Free memory on the device (unknown devices count as unlimited)."""
    if device.startswith("cuda"):
        import torch
        return torch.cuda.mem_get_info()[0]
    return 1 << 62


TILE_BYTES_PER_ELEMENT = 20   # float32 distance tile, matmul/broadcast temporaries, concatenated
                              # distances (4 B) and int64 ids (8 B); measured peak is about 16 B


def fit_chunks(device: str, query_chunk: int, base_chunk: int, margin: float = 0.6) -> tuple[int, int]:
    """Shrink the chunks until one (queries x base) tile with its temporaries fits in `margin` of
    the memory that is free now (call after the resident base is uploaded)."""
    budget = margin * free_device_bytes(device)
    while TILE_BYTES_PER_ELEMENT * query_chunk * base_chunk > budget and base_chunk > 4096:
        base_chunk //= 2
    while TILE_BYTES_PER_ELEMENT * query_chunk * base_chunk > budget and query_chunk > 64:
        query_chunk //= 2
    if TILE_BYTES_PER_ELEMENT * query_chunk * base_chunk > budget:
        raise MemoryError(f"even a {query_chunk} x {base_chunk} tile does not fit in {device} memory "
                          f"({free_device_bytes(device) / 2 ** 30:.1f} GiB free)")
    return query_chunk, base_chunk


def fit_cpu_chunks(query_chunk: int, base_chunk: int, dim: int,
                   available: int | None = None) -> tuple[int, int]:
    """Shrink the chunks until one tile plus the float32 base chunk fits in the available RAM."""
    avail = device_mod.available_ram_bytes() if available is None else available
    budget = CPU_RAM_FRACTION * avail

    def need(q: int, b: int) -> float:
        return TILE_BYTES_PER_ELEMENT * q * b + 8.0 * b * dim

    while need(query_chunk, base_chunk) > budget and base_chunk > 4096:
        base_chunk //= 2
    while need(query_chunk, base_chunk) > budget and query_chunk > 64:
        query_chunk //= 2
    if need(query_chunk, base_chunk) > budget:
        raise MemoryError(f"even a {query_chunk} x {base_chunk} tile does not fit in RAM "
                          f"({avail / 2 ** 30:.1f} GiB available)")
    return query_chunk, base_chunk


class DeviceBase:
    """The float32 base uploaded once, in chunks, when it fits on the device."""

    def __init__(self, chunks, base_chunk: int):
        self.chunks, self.base_chunk = chunks, base_chunk

    @classmethod
    def try_load(cls, base, device: str, base_chunk: int, fraction: float = 0.5):
        if is_cpu_numpy(device):
            return None
        nbytes = base.shape[0] * base.shape[1] * 4
        if nbytes > fraction * free_device_bytes(device):
            logger.info(f"base ({nbytes / 2 ** 30:.1f} GiB) does not fit on {device}; streaming chunks")
            return None
        import torch
        logger.info(f"uploading the base ({nbytes / 2 ** 30:.1f} GiB) to {device} once")
        target = _torch_device(device)
        chunks = [torch.from_numpy(_to_f32(base[i:i + base_chunk])).to(target)
                  for i in range(0, len(base), base_chunk)]
        return cls(chunks, base_chunk)


def _candidates_torch(queries: np.ndarray, base, n_cand: int, base_chunk: int,
                      device: str, resident: "DeviceBase | None" = None, progress=None) -> np.ndarray:
    import torch
    device = _torch_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False   # TF32 would make candidate distances too coarse
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
        if progress is not None:
            progress(min(1.0, (start + len(chunk)) / n))
    return best_i.cpu().numpy()


def _usable_workers(workers: int, n_queries: int) -> int:
    """Threads for the CPU path: one without a way to cap BLAS threads (it would oversubscribe)."""
    if workers <= 1 or not device_mod.can_cap_threads():
        return 1
    return max(1, min(workers, n_queries // MIN_QUERIES_PER_WORKER))


def exact_topk_chunk(
    queries: np.ndarray,
    base,
    k: int = TOP_K,
    device: str = "numpy",
    base_chunk: int = BASE_CHUNK,
    margin: int = CANDIDATE_MARGIN,
    resident: "DeviceBase | None" = None,
    workers: int = 1,
    progress=None,
) -> tuple[np.ndarray, np.ndarray]:
    """(ids, euclidean distances) of the exact top-k for one chunk of queries.

    `workers` applies to the CPU path only. `progress(fraction_of_base_done)` is called per base chunk.
    """
    if k > len(base):
        raise ValueError(f"k={k} exceeds base size {len(base)}")
    queries = _to_f32(queries)
    n_cand = min(k + margin, len(base))
    workers = _usable_workers(workers, len(queries)) if is_cpu_numpy(device) else 1
    cap = device_mod.limited_threads(1) if workers > 1 else contextlib.nullcontext()
    ids = np.empty((len(queries), k), dtype=np.int64)
    dist = np.empty((len(queries), k), dtype=np.float32)

    def finish(rows: slice, cand: np.ndarray) -> None:
        for i in range(rows.start, rows.stop):
            c = cand[i]
            d = l2_to_ids(queries[i], base, c)
            order = np.lexsort((c, d))[:k]   # distance, then id: deterministic ties
            ids[i], dist[i] = c[order], d[order]

    with cap, ThreadPoolExecutor(workers) as pool:
        if is_cpu_numpy(device):
            cand = _candidates_numpy(queries, base, n_cand, base_chunk, pool, workers, progress)
        else:
            cand = _candidates_torch(queries, base, n_cand, base_chunk, device, resident, progress)
        if workers > 1:
            list(pool.map(lambda rows: finish(rows, cand), _split(len(queries), workers)))
        else:
            finish(slice(0, len(queries)), cand)
    return ids, dist


def _chunks_done(out_dir: Path, n_queries: int, query_chunk: int) -> int:
    """Queries already covered by chunk files (a resumed run)."""
    done = 0
    for start in range(0, n_queries, query_chunk):
        if (out_dir / f"chunk_{start:09d}.npz").exists():
            done += min(query_chunk, n_queries - start)
    return done


def _record_run(out_dir: Path, entry: dict) -> None:
    """Append this run to provenance.json (not part of the resume check)."""
    path = out_dir / "provenance.json"
    runs = json.loads(path.read_text())["runs"] if path.exists() else []
    atomic.write_json(path, {"runs": runs + [entry]})


def build_ground_truth(
    queries: np.ndarray,
    base,
    out_dir: Path,
    k: int = TOP_K,
    device: str = "numpy",
    query_chunk: int = QUERY_CHUNK,
    base_chunk: int = BASE_CHUNK,
    workers: int | None = None,
    progress_interval: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Resumable ground truth; chunk files are written atomically and reused.

    A CPU run and a GPU run can resume each other: the device and chunk sizes actually used go to
    provenance.json, and only the query chunk size is checked on resume.
    """
    if device == "auto":
        device = device_mod.resolve("auto").kind
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"n_queries": len(queries), "n_base": len(base), "k": k, "query_chunk": query_chunk}
    meta_path = out_dir / "meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
        raise ValueError(f"{meta_path} was written for a different run: {meta_path.read_text()}")
    atomic.write_json(meta_path, meta)
    atomic.remove_stale_temporaries(out_dir)
    cpu = is_cpu_numpy(device)
    # upload the base first (if it fits), then size the tiles from what is left
    resident = DeviceBase.try_load(base, device, base_chunk)
    query_chunk_used, base_chunk_used = query_chunk, base_chunk
    if cpu:
        query_chunk_used, base_chunk_used = fit_cpu_chunks(query_chunk, base_chunk, base.shape[1])
    else:
        query_chunk_used, base_chunk_used = fit_chunks(device, query_chunk, base_chunk)
    if (query_chunk_used, base_chunk_used) != (query_chunk, base_chunk):
        logger.warning(f"memory is small: chunks reduced to queries={query_chunk_used} "
                       f"base={base_chunk_used}")
        if resident is not None and base_chunk_used != base_chunk:
            resident = None    # the resident chunks were cut at the old size; stream instead
    threads = workers if workers is not None else device_mod.default_threads()
    used_workers = _usable_workers(threads, query_chunk_used) if cpu else 1
    _record_run(out_dir, {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "device": device,
                          "workers": used_workers, "query_chunk": query_chunk_used,
                          "base_chunk": base_chunk_used})
    n_chunks = -(-len(queries) // query_chunk)
    progress = Progress("ground truth", len(queries), "queries",
                        done=_chunks_done(out_dir, len(queries), query_chunk),
                        interval=progress_interval)
    began, built = time.time(), 0
    for c, start in enumerate(range(0, len(queries), query_chunk)):
        path = out_dir / f"chunk_{start:09d}.npz"
        if path.exists():
            continue
        block = queries[start:start + query_chunk]
        parts = []
        for i in range(0, len(block), query_chunk_used):
            sub = block[i:i + query_chunk_used]

            def report(fraction, offset=start + i, size=len(sub)):
                progress.update(offset + fraction * size)

            parts.append(exact_topk_chunk(sub, base, k, device, base_chunk_used, resident=resident,
                                          workers=used_workers, progress=report))
        ids = np.concatenate([p[0] for p in parts])
        dist = np.concatenate([p[1] for p in parts])
        atomic.save_npz(path, ids=ids, dist=dist)
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
