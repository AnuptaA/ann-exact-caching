"""Run one grid cell: build driver inputs, call the replay driver, score, write results."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from harness import config, data, scoring, thresholds
from harness.backend.answer_table import load_answer_table
from harness.runner import results
from harness.runner.manifest import Cell
from workloads.q1 import K, groundtruth, streams as stream_mod, universe
from workloads.q1.cli import load_universe

AKER_REVISION = "e7639a52aec1320a47c5fa0b833c7214024931c9"


class DatasetDir:
    """Layout of one prepared dataset (built by the workloads and backend steps)."""

    def __init__(self, root: Path, dataset: str, seed: int = 0):
        self.root = Path(root) / dataset
        self.base = data.read_f32bin(self.root / "base.f32bin")
        self.official = data.read_f32bin(self.root / "official.f32bin")
        self.universe = load_universe(self.root / "universe.npz")
        z = np.load(self.root / f"streams_seed{seed}.npz")
        self.streams = {k: z[k] for k in z.files if k != "history_fraction"}
        self.history_fraction = float(z["history_fraction"])
        self.answer_ids, self.answer_dist = load_answer_table(self.root / "answers")
        self.gt_ids, self.gt_dist = groundtruth.load_ground_truth(self.root / "gt")
        self.backend = json.loads((self.root / "answers" / "meta.json").read_text())

    def vectors_of(self, ids: np.ndarray) -> np.ndarray:
        return universe.request_vectors(self.universe, self.official, ids)


def settings_for(cell: Cell, ds: DatasetDir, stream_ids: np.ndarray, history_len: int,
                 seed: int = 0) -> config.AkerSettings:
    pool = config.pool_size_for(len(ds.base), cell.budget_percent)
    if cell.system == "aker":
        return config.AkerSettings(mode="standard", pool_size=pool)
    hist = thresholds.history_vectors(stream_ids, history_len, ds.vectors_of)
    p = cell.percentile()
    thresh = thresholds.pairwise_sq_percentiles(hist, (p,), seed=seed)[p]
    return config.AkerSettings(mode="proximity", pool_size=pool, global_thresh=thresh)


def run_cell(cell: Cell, ds: DatasetDir, driver_dir: Path, out_dir: Path, work_dir: Path,
             timeout: float | None = None) -> Path:
    """Replay one cell; skips nothing here (the caller checks results.is_done)."""
    ids = ds.streams[cell.stream]
    history_len = stream_mod.history_split(len(ids), ds.history_fraction)
    settings = settings_for(cell, ds, ids, history_len, cell.stream_seed)
    work = Path(work_dir) / cell.cell_id
    work.mkdir(parents=True, exist_ok=True)
    data.write_f32bin(work / "requests.f32bin", ds.vectors_of(ids))
    ids.astype("<i8").tofile(work / "stream.i64")
    ds.answer_ids[ids][:, :K].astype("<i8").tofile(work / "answers.i64")
    ds.answer_dist[ids][:, :K].astype("<f4").tofile(work / "dists.f32")
    binary = Path(driver_dir) / f"ak_replay_{settings.mode}"
    subprocess.run([str(binary), "--base", str(ds.root / "base.f32bin"),
                    "--requests", str(work / "requests.f32bin"), "--stream", str(work / "stream.i64"),
                    "--answers", str(work / "answers.i64"), "--dists", str(work / "dists.f32"),
                    "--out-dir", str(work), *settings.driver_args()], check=True, timeout=timeout)
    outcome = np.fromfile(work / "outcome.u8", dtype=np.uint8)
    result_ids = np.fromfile(work / "result_ids.i64", dtype="<i8").reshape(len(ids), K)
    score = scoring.score_run(outcome, result_ids, ids, data.read_f32bin(work / "requests.f32bin"),
                              ds.base, ds.gt_dist[ids][:, K - 1], history_len)
    library_exact = outcome == scoring.EXACT_LIB
    header = {"cell": cell.__dict__, "settings": settings.as_dict(),
              "config_hash": hashlib.sha1(json.dumps([cell.__dict__, settings.as_dict()],
                                                     sort_keys=True).encode()).hexdigest(),
              **results.git_state(), "backend": ds.backend["backend"],
              "ef_search": ds.backend["ef_search"], "list_size": ds.backend["list_size"],
              "aker_revision": AKER_REVISION, "history_len": history_len,
              "summary": score.summary(),
              "kind_mismatch": int((score.equality != library_exact & score.hit).sum()),
              "driver_summary": (work / "summary.txt").read_text()}
    return results.write_result(out_dir, cell.cell_id, header, outcome=outcome,
                                recall=score.recall.astype(np.float32), equality=score.equality,
                                approximate=score.approximate, stream_ids=ids)
