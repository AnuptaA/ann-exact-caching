"""Per-request records: one row per scored request, saved as a compressed npz.

Path codes: 0 mini hit, 1 main hit, 2 miss, 3 abstained (no threshold in force).
Baselines use hit (1) and miss (2) only. Missing values are NaN or -1.
"""

import json
from pathlib import Path

import numpy as np

MINI, MAIN, MISS, ABSTAIN = 0, 1, 2, 3
BACKEND_NONE, BACKEND_MISS, BACKEND_AUDIT = 0, 1, 2
BACKEND_CODES = {"miss": BACKEND_MISS, "audit": BACKEND_AUDIT}

FIELDS = {
    "index": np.int64,
    "K": np.int32,
    "path": np.int8,
    "recall": np.float32,
    "score": np.float32,
    "lam": np.float32,
    "n_read": np.int32,
    "n_contrib": np.int32,
    "cand_recall": np.float32,
    "backend_kind": np.int8,
    "backend_depth": np.int32,
    "rereads": np.int32,
    "cache_ms": np.float32,
}


class Records:
    def __init__(self, meta: dict | None = None):
        self.meta = dict(meta or {})
        self.columns = {name: [] for name in FIELDS}
        self.constructions: list[dict] = []

    def __len__(self) -> int:
        return len(self.columns["index"])

    def add(self, index: int, K: int, path: int, recall: float, score: float = np.nan,
            lam: float | None = None, n_read: int = -1, n_contrib: int = -1,
            cand_recall: float = np.nan, backend_kind: int = BACKEND_NONE,
            backend_depth: int = 0, rereads: int = 0, cache_ms: float = np.nan) -> None:
        row = {"index": index, "K": K, "path": path, "recall": recall,
               "score": np.nan if score is None else score,
               "lam": np.nan if lam is None else lam, "n_read": n_read,
               "n_contrib": n_contrib, "cand_recall": cand_recall,
               "backend_kind": backend_kind, "backend_depth": backend_depth,
               "rereads": rereads, "cache_ms": cache_ms}
        for name, value in row.items():
            self.columns[name].append(value)

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: np.asarray(self.columns[name], dtype=dtype)
                for name, dtype in FIELDS.items()}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        with open(tmp, "wb") as handle:
            np.savez_compressed(
                handle, **self.arrays(),
                constructions_json=np.array(json.dumps(self.constructions, default=float)),
                meta_json=np.array(json.dumps(self.meta, default=str)))
        tmp.replace(path)
        return path


def timing_section(cols: dict, history_len: int) -> dict:
    """The timing section the throughput replay reads (request ids, hit, cache time, re-reads), from records."""
    hit = np.isin(cols["path"], (MINI, MAIN))
    dt = np.nan_to_num(cols["cache_ms"].astype(np.float64), nan=0.0)
    return {"request_ids": cols["index"].tolist(), "hit": hit.tolist(), "dt_ms": dt.tolist(),
            "rereads": cols["rereads"].tolist(), "every": 1, "n_timed": int(len(hit)), "history_len": history_len}


def load(path: str | Path) -> tuple[dict[str, np.ndarray], list[dict], dict]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in FIELDS}
        constructions = json.loads(str(data["constructions_json"]))
        meta = json.loads(str(data["meta_json"]))
    return arrays, constructions, meta


def path_from_response(response) -> int:
    """Path code of a ReCal response."""
    from recal.core import MAIN_HIT, MINI_HIT, MISS_ABSTAIN
    if response.path == MINI_HIT:
        return MINI
    if response.path == MAIN_HIT:
        return MAIN
    if response.path == MISS_ABSTAIN:
        return ABSTAIN
    return MISS


def backend_fields(background: list) -> tuple[int, int]:
    """Kind and depth of the first backend search of the request (none: 0, 0)."""
    for kind, depth in background:
        if kind in BACKEND_CODES:
            return BACKEND_CODES[kind], int(depth)
    return BACKEND_NONE, 0


def add_response(records: Records, index: int, K: int, response, truth) -> None:
    """Add one scored ReCal request. truth: the exact top-K ids."""
    recall = float(np.isin(truth, response.ids).mean())
    cand_recall = np.nan
    if response.cand_ids is not None:
        cand_recall = float(np.isin(truth, response.cand_ids).mean())
    kind, depth = backend_fields(response.background)
    records.add(index, K, path_from_response(response), recall,
                score=response.score, lam=response.lam_used, n_read=response.n_read,
                n_contrib=response.n_contrib, cand_recall=cand_recall,
                backend_kind=kind, backend_depth=depth, rereads=response.rereads,
                cache_ms=response.cache_ms)
