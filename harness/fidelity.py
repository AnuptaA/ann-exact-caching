"""Fidelity check: Aker (paper settings) on Aker's released SPACEV stream.

Reports system delivered recall (criterion: >= 0.8) and the cumulative hit ratio
over the request sequence, to be compared by eye with the paper's Figures 8 and
9. Report only; nothing is tuned toward the paper.
"""

import hashlib
import io
import pickle
from pathlib import Path

import numpy as np

from harness import data, fingerprints
from harness.backend.answer_table import build_answer_table
from workloads.q1 import groundtruth

MIN_RECALL = 0.8
_ALLOWED = {("numpy", "ndarray"), ("numpy", "dtype"), ("numpy._core.multiarray", "_reconstruct"),
            ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "scalar"),
            ("numpy.core.multiarray", "scalar"), ("builtins", "list"), ("builtins", "dict"),
            ("builtins", "tuple"), ("builtins", "set"), ("builtins", "bytearray")}


class RestrictedUnpickler(pickle.Unpickler):
    """Only numpy arrays and builtin containers; every other global is rejected."""

    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"global {module}.{name} is not allowed")


def load_released_pickle(path: Path):
    """Inspect a third-party .pkl safely (run it in the isolated environment too)."""
    return RestrictedUnpickler(io.BytesIO(Path(path).read_bytes())).load()


class FidelityData:
    """Duck-typed dataset for run_cell: the released stream over its distinct vectors."""

    history_fraction = 0.0   # measure every request

    def __init__(self, root: Path, base_path: Path, payload: np.ndarray, backend, ef_search: int,
                 list_size: int = 50, gt_device: str = "numpy"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.base_path = Path(base_path)
        self.base = data.read_f32bin(self.base_path)
        unique, inverse = np.unique(np.asarray(payload, dtype=np.float32), axis=0, return_inverse=True)
        self.unique = unique
        self.streams = {"released": inverse.astype(np.int64).ravel()}
        build_answer_table(backend, unique, self.base, self.root / "answers", list_size, ef_search)
        self.answer_ids, self.answer_dist = _load(self.root / "answers")
        groundtruth.build_ground_truth(unique, self.base, self.root / "gt", device=gt_device)
        self.gt_ids, self.gt_dist = groundtruth.load_ground_truth(self.root / "gt")
        self.backend = {"backend": backend.name, "ef_search": ef_search, "list_size": list_size}
        stream_sha = hashlib.sha256(self.streams["released"].tobytes() + unique.tobytes()).hexdigest()
        self.fingerprints = {"base": fingerprints.base_signature(self.base_path), "official": None,
                             "universe": None, "streams": stream_sha,
                             "answers": fingerprints.answers_fingerprint(self.root),
                             "answers_meta": fingerprints.sha256_or_none(self.root / "answers" / "meta.json"),
                             "tune": None}
        self.input_id = fingerprints.combine(self.fingerprints)

    def vectors_of(self, ids: np.ndarray) -> np.ndarray:
        return self.unique[ids]


def _load(path):
    from harness.backend.answer_table import load_answer_table
    return load_answer_table(path)


def cumulative_hit_ratio(outcome: np.ndarray) -> np.ndarray:
    return np.cumsum(outcome != 0) / np.arange(1, len(outcome) + 1)


def check_recall(delivered: float, minimum: float = MIN_RECALL) -> dict:
    return {"delivered_recall": delivered, "minimum": minimum, "pass": delivered >= minimum}


def plot_hit_ratio(curves: dict, path: Path) -> Path:
    """Cumulative hit ratio versus request index, one line per pool size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, curve in curves.items():
        ax.plot(np.arange(1, len(curve) + 1), curve, label=label)
    ax.set_xlabel("request index")
    ax.set_ylabel("cumulative hit ratio")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return Path(path)


RELEASED_URL = "https://huggingface.co/datasets/sjoon-oh/aker/resolve/main/spacev-sim-100k-0.99"


def run_fidelity(data_dir: Path, dsn: str, driver_dir: Path, out_dir: Path, work_dir: Path,
                 ef_search: int, list_size: int = 50, device: str = "numpy",
                 inspect_pickle: bool = False, dataset: str = "spacev",
                 payload_path: Path | None = None) -> dict:
    """Aker (paper settings) on the released SPACEV stream at pools of 1% and 5%.

    Needs the prepared spacev dataset and its PostgreSQL database. Reports recall,
    the pass/fail against the criterion and writes the cumulative hit-ratio plot.
    """
    import psycopg
    from workloads.q1.datasets import common
    from harness.backend.pgvector import PgvectorBackend, check_version, with_keepalives
    from harness.runner import manifest, replay, results
    root = Path(data_dir) / "fidelity"
    if payload_path is None:
        payload_path = common.download_file(RELEASED_URL + ".npy", root / "spacev-sim-100k-0.99.npy")
    if inspect_pickle:
        pkl = common.download_file(RELEASED_URL + ".pkl", root / "spacev-sim-100k-0.99.pkl")
        obj = load_released_pickle(pkl)
        print(f"released pickle: {type(obj).__name__}, "
              f"{len(obj) if hasattr(obj, '__len__') else ''} items")
    payload = np.load(payload_path)
    with psycopg.connect(with_keepalives(dsn, f"q1_{dataset}"), autocommit=True) as conn:
        check_version(conn)
        ds = FidelityData(root / "tables", Path(data_dir) / dataset / "base.f32bin", payload,
                          PgvectorBackend(conn), ef_search, list_size, device)
    report, curves = {}, {}
    for pct in (1, 5):
        cell = manifest.Cell(f"{dataset}-fidelity", "released", "aker", "paper", pct)
        header, arrays = results.read_result(
            replay.run_cell(cell, ds, driver_dir, out_dir, work_dir))
        s = header["summary"]
        curves[f"pool {pct}%"] = cumulative_hit_ratio(arrays["outcome"])
        report[f"{pct}%"] = {**s, **check_recall(s["delivered_recall"])}
    plot = plot_hit_ratio(curves, Path(out_dir) / "fidelity_hit_ratio.png")
    return {"report": report, "plot": str(plot), "requests": int(len(payload))}
