"""Prep folders for the shift and update experiments, from request ids of the group universe.

A prep folder is what every driver reads: q.npy (the distinct request vectors), streams.npz (request positions
into q.npy under the stream name, plus phase_<name> and cluster_<name> per request), top500.npy (exact ranking
over all vectors, truncated to `depth`), k.npy (K per request) and meta.json (kind files, history length).
Update schedules add updates.npz for ReCal's update path and the three files the real Aker driver and the
simulators read for a data change (data_churn.npz, rank_extra_churn.npz, topU1000.npy), so all systems replay
the same events with the same live ground truth.

Request vectors come from existing official vectors through the group universe; no encoder runs. SPACEV:
the official query set. TriviaQA: the DPR-space question vectors already embedded for SPHERE (append a zero
coordinate for the inner-product form with append_zero).
"""

import json
from pathlib import Path

import numpy as np

from recal.history_mode import HISTORY_FRACTION
from workloads.q1 import shifts
from workloads.q1.universe import request_vectors

AKER_KIND = "churn"


def exact_top(queries: np.ndarray, base, depth: int, device: str = "numpy", chunk: int = 512) -> np.ndarray:
    """Exact top-depth ids over all base vectors, int32."""
    from workloads.q1.groundtruth import exact_topk_chunk
    out = np.empty((len(queries), depth), dtype=np.int32)
    for s in range(0, len(queries), chunk):
        ids, _ = exact_topk_chunk(np.asarray(queries[s:s + chunk], dtype=np.float32), base, depth, device)
        out[s:s + chunk] = ids
    return out


def k_sequence(n: int, k: int = 10, mixed: bool = False, seed: int = 0) -> np.ndarray:
    from recal import ksets
    return ksets.k_sequence(n, k, mixed, seed=seed)


def convert_shift(out: Path, kind: str, shift: dict, uni, official, base, base_path: Path, centroids: np.ndarray,
                  k_seq: np.ndarray | None = None, depth: int = 500, history_frac: float = HISTORY_FRACTION,
                  append_zero: bool = False, device: str = "numpy") -> Path:
    """One prep folder for a shift stream (shift: the output of shifts.shift_stream)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    vectors = request_vectors(uni, np.asarray(official), shift["ids"])
    if append_zero:
        vectors = np.hstack([vectors, np.zeros((len(vectors), 1), np.float32)])
    queries, inverse = np.unique(vectors, axis=0, return_inverse=True)
    n = len(vectors)
    stream = inverse.astype(np.int64).ravel()
    np.save(out / "q.npy", queries.astype(np.float32))
    np.save(out / "top500.npy", exact_top(queries, base, min(depth, len(base)), device))
    np.savez(out / "streams.npz", **{kind: stream, "random": stream, f"phase_{kind}": shift["phase"],
                                     f"cluster_{kind}": shifts.assign(vectors, centroids)})
    np.save(out / "k.npy", k_sequence(n) if k_seq is None else np.asarray(k_seq, dtype=np.int32))
    meta = {"kind": "files", "db": str(base_path), "q": str(out / "q.npy"), "stream": kind, "n_requests": n,
            "history_len": int(history_frac * n), "top_depth": min(depth, len(base)),
            "n_clusters": int(len(centroids))}
    (out / "meta.json").write_text(json.dumps(meta))
    return out


def convert_updates(prep: Path, schedule: dict, base, depth: int = 1000, device: str = "numpy") -> Path:
    """Add an update schedule to a prep folder, for ReCal and for the real Aker driver and the simulators."""
    prep = Path(prep)
    np.savez(prep / "updates.npz", **schedule)
    insert = schedule["insert"].astype(bool)
    ins_ids, ins_t = schedule["id"][insert], schedule["t"][insert]
    del_ids, del_t = schedule["id"][~insert], schedule["t"][~insert]
    np.savez(prep / f"data_{AKER_KIND}.npz", held=ins_ids, initial=schedule["initial"], ins_ids=ins_ids,
             del_ids=del_ids, ins_t=ins_t, del_t=del_t, burst_at=int(schedule["t"].min()))
    queries = np.load(prep / "q.npy", mmap_mode="r")
    top = exact_top(queries, base, min(depth, len(base)), device)
    np.save(prep / "topU1000.npy", top)
    np.savez(prep / f"rank_extra_{AKER_KIND}.npz", flagged=np.zeros(0, dtype=np.int64),
             rows=np.zeros((0, top.shape[1]), dtype=np.int32))
    meta = json.loads((prep / "meta.json").read_text())
    meta["updates"] = {"file": "updates.npz", "aker_kind": AKER_KIND, "events": int(len(schedule["t"]))}
    (prep / "meta.json").write_text(json.dumps(meta))
    return prep
