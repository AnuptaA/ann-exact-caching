"""QVCache from the authors' artifact (https://github.com/qvcache/qvcache-vldb26), through the common harness.

The artifact's Python binding (module qvcache) is driven with the same streams, history, records and byte rule as
every other system. The reimplementations in simulations/approximate_hits/online_baselines.py stay as cross-checks
and produce no reported number (recal.cells --method qvcache_sim).

Settings are the artifact's documented defaults except where the paper (arXiv 2602.02057, section 6.2) differs:
the deviation factor is 0.25 for SIFT and SPACEV and 0.075 for the other datasets, and the cache size is a byte
budget (the paper fixes 100,000 vectors). DIFFERENCES lists each one.

Memory (accounting.qvcache_*): memory_index_max_points is set so that a full cache costs M: per cached point the
vector at stored precision, a 4-byte tag and a graph row of R 4-byte neighbors; the PCA projection; 16 bytes per
active region, with the regions counted on the history. The mini-indexes split the points evenly.
"""

import importlib
import time
from pathlib import Path

import numpy as np

from recal import accounting, history_mode
from recal.qvcache_backends import StoredBackend
from recal.records import MAIN, MISS, Records

ARTIFACT_REPO = "https://github.com/qvcache/qvcache-vldb26"
ARTIFACT_COMMIT = "39731c6cfa428a8386441ffbc3c43a6bc00dc3f9"
SPACEV_DEVIATION, OTHER_DEVIATION = 0.25, 0.075

ARTIFACT_DEFAULTS = {
    "R": 64, "memory_L": 128, "B": 8, "M": 8, "alpha": 1.2, "build_threads": 8, "search_threads": 24,
    "use_reconstructed_vectors": False, "p": 0.9, "beamwidth": 2, "use_regional_theta": True, "pca_dim": 16,
    "buckets_per_dim": 8, "max_regions": 2**64 - 1, "n_async_insert_threads": 16, "lazy_theta_updates": True,
    "number_of_mini_indexes": 4, "search_mini_indexes_in_parallel": False, "max_search_threads": 32,
}
SEARCH_STRATEGY = "SEQUENTIAL_LRU_ADAPTIVE"

DIFFERENCES = [
    {"setting": "deviation_factor", "artifact": SPACEV_DEVIATION,
     "used": "0.25 for SPACEV (paper, section 6.2), 0.075 for the other datasets"},
    {"setting": "memory_index_max_points", "artifact": "set by the caller (the paper: 100,000 vectors)",
     "used": "set from the byte budget M (accounting.qvcache_capacity_for_budget)"},
    {"setting": "R, memory_L", "artifact": "64, 128", "used": "artifact values (not stated in the paper)"},
    {"setting": "alpha (adaptivity rate of the thresholds)", "artifact": "p 0.9", "used": "p 0.9 (the paper's alpha 0.9)"},
    {"setting": "vector precision charged", "artifact": "float32 inside the mini-indexes",
     "used": "stored precision, as for every system"},
]


def load_module():
    return importlib.import_module("qvcache")


def default_deviation(coord_bytes: float) -> float:
    """SPACEV is the int8 dataset (stored precision 1 byte)."""
    return SPACEV_DEVIATION if coord_bytes == 1 else OTHER_DEVIATION


def settings(threads: int | None = None) -> dict:
    s = dict(ARTIFACT_DEFAULTS)
    if threads:
        s.update(build_threads=threads, search_threads=threads, max_search_threads=threads,
                 n_async_insert_threads=min(16, threads))
    return s


def write_bin(path: Path, array, chunk: int = 100_000) -> Path:
    """DiskANN .bin: int32 points, int32 dimension, then float32 rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(np.array([len(array), array.shape[1]], dtype=np.int32).tobytes())
        for s in range(0, len(array), chunk):
            f.write(np.ascontiguousarray(array[s:s + chunk], dtype=np.float32).tobytes())
    return path


def pca_base(db, out_dir: Path, points: int, seed: int = 0) -> Path:
    """The file the artifact trains its PCA on (it samples 1 point in 1,000): a random subset of the base."""
    rows = np.sort(np.random.default_rng(seed).choice(len(db), min(points, len(db)), replace=False))
    return write_bin(Path(out_dir) / "pca_base.bin", np.asarray(db[rows], dtype=np.float32))


def make_cache(qvc, bin_path: Path, pca_prefix: Path, backend, conf: dict, deviation: float, max_points: int):
    cache = qvc.QVCache(
        data_path=str(bin_path), pca_prefix=str(pca_prefix), R=conf["R"], memory_L=conf["memory_L"], B=conf["B"],
        M=conf["M"], alpha=conf["alpha"], build_threads=conf["build_threads"], search_threads=conf["search_threads"],
        use_reconstructed_vectors=conf["use_reconstructed_vectors"], p=conf["p"], deviation_factor=deviation,
        memory_index_max_points=int(max_points), beamwidth=conf["beamwidth"],
        use_regional_theta=conf["use_regional_theta"], pca_dim=conf["pca_dim"],
        buckets_per_dim=conf["buckets_per_dim"], max_regions=conf["max_regions"],
        n_async_insert_threads=conf["n_async_insert_threads"], lazy_theta_updates=conf["lazy_theta_updates"],
        number_of_mini_indexes=conf["number_of_mini_indexes"],
        search_mini_indexes_in_parallel=conf["search_mini_indexes_in_parallel"],
        max_search_threads=conf["max_search_threads"], backend=backend)
    cache.set_search_strategy(getattr(qvc.SearchStrategy, SEARCH_STRATEGY))
    return cache


def replay(cache, backend, vectors, ks, first_scored: int, truth, records: Records | None = None) -> dict:
    """Serve the requests in order; those from first_scored on are scored (and recorded)."""
    hits, recall_sum, scored = 0, 0.0, 0
    for pos in range(len(vectors)):
        before = backend.snapshot()
        t0 = time.perf_counter()
        hit, tags, _ = cache.search(np.asarray(vectors[pos], dtype=np.float32), int(ks[pos]))
        elapsed = time.perf_counter() - t0
        if pos < first_scored:
            continue
        searches, rereads, seconds = (a - b for a, b in zip(backend.snapshot(), before))
        recall = float(np.isin(truth(pos, int(ks[pos])), tags).mean())
        hits += bool(hit)
        recall_sum += recall
        scored += 1
        if records is not None:
            records.add(pos, int(ks[pos]), MAIN if hit else MISS, recall, backend_kind=0 if hit else 1,
                        backend_depth=0 if hit else int(ks[pos]), rereads=int(rereads),
                        cache_ms=max(elapsed - seconds, 0.0) * 1e3)
    return {"scored": scored, "hits": hits, "recall_sum": recall_sum}


def calibrate(qvc, bin_path, pca_prefix, backend, conf, deviation, vectors, ks, history_len, M, d, coord_bytes):
    """Capacity at which a full cache costs M. The active regions are counted on a pass over the history and
    projected linearly to the whole stream (an upper bound: the region count grows sublinearly)."""
    R, pca = conf["R"], conf["pca_dim"]
    n_mini = conf["number_of_mini_indexes"]
    probe = accounting.qvcache_capacity_for_budget(M, d, coord_bytes, R, pca, 0, n_mini)
    cache = make_cache(qvc, bin_path, pca_prefix, backend, conf, deviation, probe)
    replay(cache, backend, vectors[:history_len], ks[:history_len], history_len, lambda p, k: np.zeros(0))
    seen = int(cache.get_number_of_active_pca_regions())
    regions = int(np.ceil(seen * len(vectors) / max(history_len, 1)))
    del cache
    capacity = accounting.qvcache_capacity_for_budget(M, d, coord_bytes, R, pca, regions, n_mini)
    full = accounting.qvcache_full_capacity_bytes(capacity, d, coord_bytes, R, pca, regions)
    return capacity, {"probe_capacity": probe, "regions_on_history": seen, "regions_projected": regions, "configured_capacity": capacity,
                      "bytes_at_full_capacity": full, "M": M, "source": "calibrated_on_history"}


def run(args, db, q, top, idx, ks, backend, out_dir: Path, records: Records | None = None, qvc=None) -> dict:
    qvc = qvc or load_module()
    d = db.shape[1]
    conf = settings(args.qvcache_threads)
    deviation = args.deviation_factor if args.deviation_factor is not None else default_deviation(args.coord_bytes)
    n = len(idx)
    history_len = history_mode.history_length(n, args.history_frac)
    vectors = np.asarray(q[idx], dtype=np.float32)
    M = accounting.budget_bytes(len(db), d, args.budget, args.coord_bytes)
    bin_path = pca_base(db, out_dir, args.pca_points)
    pca_prefix = Path(out_dir) / "pca"
    capacity, calibration = calibrate(qvc, bin_path, pca_prefix, backend, conf, deviation, vectors, ks, history_len,
                                      M, d, args.coord_bytes)
    cache = make_cache(qvc, bin_path, pca_prefix, backend, conf, deviation, capacity)
    backend.searches = backend.depth_sum = backend.rereads = 0
    backend.seconds = 0.0
    t0 = time.time()
    truth = (lambda pos, k: top[idx[pos], :k])
    totals = replay(cache, backend, vectors, ks, history_len, truth, records)
    used = accounting.qvcache_bytes_used(int(cache.get_number_of_vectors_in_memory_index()),
                                         int(cache.get_number_of_active_pca_regions()), d, args.coord_bytes,
                                         conf["R"], conf["pca_dim"])
    scored = max(totals["scored"], 1)
    out = {"method": "qvcache", "args": dict(vars(args), warmup=history_len),
           "summary": {"hit_rate": totals["hits"] / scored, "system_recall": totals["recall_sum"] / scored,
                       "queries": totals["scored"], "seconds": time.time() - t0},
           "background": backend.stats(), "bytes": used, "M": M, "within_M": bool(used["total"] <= M),
           "configured_capacity": capacity, "bytes_used_end": used["total"],
           "bytes_at_full_capacity": calibration["bytes_at_full_capacity"], "calibration": calibration,
           "qvcache": {"repo": ARTIFACT_REPO, "commit": ARTIFACT_COMMIT, "settings": conf,
                       "deviation_factor": deviation, "search_strategy": SEARCH_STRATEGY,
                       "threads": args.qvcache_threads, "differences": DIFFERENCES,
                       "build": "native, in the interpreter of the backend (no extra hop)"}}
    history_mode.stamp(out, n, history_len)
    if records is not None:
        records.meta.update(method="qvcache", K=int(ks[0]), history_len=history_len, M=M, n_scored=n - history_len)
    del cache
    return out


def run_cell(a, out_dir: Path, qvc=None) -> tuple[dict, Records]:
    from recal import sim
    db, q, top, streams = sim.load_prep(Path(a.prep))
    idx = np.asarray(streams[a.stream], dtype=np.int64)
    if a.max_requests:
        idx = idx[:a.max_requests]
    from recal import ksets
    ks = ksets.k_sequence(len(idx), a.k, a.k_mixed, a.k_set, a.seed)
    if not a.k_mixed and (Path(a.prep) / "k.npy").exists():
        ks = np.load(Path(a.prep) / "k.npy")[:len(idx)]
    backend = StoredBackend(sim.make_backend(a.backend, db, q, top, idx, a))
    records = Records()
    return run(a, db, q, top, idx, ks, backend, Path(out_dir), records, qvc), records
