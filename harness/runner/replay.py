"""Run one grid cell: build driver inputs, call the replay driver, score, write results."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np

from harness import config, data, scoring, thresholds
from harness.backend.answer_table import load_answer_table
from harness.runner import manifest, results
from harness.runner.manifest import Cell
from workloads.q1 import K, groundtruth, streams as stream_mod, universe
from workloads.q1.cli import load_universe

AKER_REVISION = "e7639a52aec1320a47c5fa0b833c7214024931c9"


class DatasetDir:
    """Layout of one prepared dataset (built by the workloads and backend steps)."""

    def __init__(self, root: Path, dataset: str, seed: int = 0):
        self.root = Path(root) / dataset
        self.base_path = self.root / "base.f32bin"
        self.base = data.read_f32bin(self.base_path)
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


def history_percentiles(cell: Cell, ds: DatasetDir, stream_ids: np.ndarray, history_len: int,
                        seed: int, cache_dir: Path | None) -> dict:
    """Percentiles of squared distances among the distinct history queries of this stream.

    Cached per (dataset, stream, seed) so the three budgets share one computation. The
    universe-wide percentile (a sample of member vectors) is stored as a diagnostic.
    """
    wanted = tuple(manifest.PROXIMITY_PERCENTILES)
    path = None
    if cache_dir is not None:
        path = Path(cache_dir) / f"{cell.dataset}-{cell.stream}-s{seed}.json"
        if path.exists():
            cached = json.loads(path.read_text())
            if all(str(p) in cached["history"] for p in wanted):
                return cached
    hist = thresholds.history_vectors(stream_ids, history_len, ds.vectors_of)
    history = thresholds.pairwise_sq_percentiles(hist, wanted, seed=seed)
    members = np.random.default_rng(seed).choice(ds.universe.n_members,
                                                 min(20000, ds.universe.n_members), replace=False)
    wide = thresholds.pairwise_sq_percentiles(ds.vectors_of(np.sort(members)), wanted,
                                              max_pairs=10_000_000, seed=seed)
    out = {"history": {str(p): v for p, v in history.items()},
           "universe_wide": {str(p): v for p, v in wide.items()}, "distinct_history_queries": len(hist)}
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out))
    return out


def settings_for(cell: Cell, ds: DatasetDir, stream_ids: np.ndarray, history_len: int,
                 seed: int = 0, cache_dir: Path | None = None) -> tuple[config.AkerSettings, dict]:
    pool = config.pool_size_for(len(ds.base), cell.budget_percent)
    if cell.system == "aker":
        return config.AkerSettings(mode="standard", pool_size=pool), {}
    pcts = history_percentiles(cell, ds, stream_ids, history_len, seed, cache_dir)
    thresh = pcts["history"][str(cell.percentile())]
    return config.AkerSettings(mode="proximity", pool_size=pool, global_thresh=thresh), pcts


POOL_LINE = re.compile(r"VectorSlotPool Status: (\d+) inserted out of total (\d+)")


def peak_pool_vectors(status_log: str) -> int:
    """Largest sampled count of unique pooled vectors in the driver's status log."""
    return max((int(m.group(1)) for m in POOL_LINE.finditer(status_log)), default=0)


def run_cell(cell: Cell, ds: DatasetDir, driver_dir: Path, out_dir: Path, work_dir: Path,
             timeout: float | None = None) -> Path:
    """Replay one cell; skips nothing here (the caller checks results.is_done)."""
    ids = ds.streams[cell.stream]
    history_len = stream_mod.history_split(len(ids), ds.history_fraction)
    settings, thresholds_info = settings_for(cell, ds, ids, history_len, cell.stream_seed,
                                             Path(work_dir) / "thresholds")
    work = Path(work_dir) / cell.cell_id
    work.mkdir(parents=True, exist_ok=True)
    data.write_f32bin(work / "requests.f32bin", ds.vectors_of(ids))
    ids.astype("<i8").tofile(work / "stream.i64")
    ds.answer_ids[ids][:, :K].astype("<i8").tofile(work / "answers.i64")
    ds.answer_dist[ids][:, :K].astype("<f4").tofile(work / "dists.f32")
    binary = Path(driver_dir) / f"ak_replay_{settings.mode}"
    with (work / "driver.log").open("w") as log:
        subprocess.run([str(binary), "--base", str(ds.base_path),
                        "--requests", str(work / "requests.f32bin"), "--stream", str(work / "stream.i64"),
                        "--answers", str(work / "answers.i64"), "--dists", str(work / "dists.f32"),
                        "--out-dir", str(work), "--status-every", "100", *settings.driver_args()],
                       check=True, timeout=timeout, stdout=log, stderr=subprocess.STDOUT)
    outcome = np.fromfile(work / "outcome.u8", dtype=np.uint8)
    result_ids = np.fromfile(work / "result_ids.i64", dtype="<i8").reshape(len(ids), K)
    score = scoring.score_run(outcome, result_ids, ids, data.read_f32bin(work / "requests.f32bin"),
                              ds.base, ds.gt_dist[ids][:, K - 1], history_len)
    w = score.window
    peak = peak_pool_vectors((work / "status.log").read_text())
    if peak > settings.pool_size:
        raise RuntimeError(f"{cell.cell_id}: peak pool {peak} exceeds pool_size {settings.pool_size}")
    header = {"cell": cell.__dict__, "settings": settings.as_dict(),
              "config_hash": hashlib.sha1(json.dumps([cell.__dict__, settings.as_dict()],
                                                     sort_keys=True).encode()).hexdigest(),
              **results.git_state(), "backend": ds.backend["backend"],
              "ef_search": ds.backend["ef_search"], "list_size": ds.backend["list_size"],
              "aker_revision": AKER_REVISION, "history_len": history_len,
              "summary": score.summary(), "peak_pool_vectors_sampled": peak,
              "thresholds": thresholds_info,
              "library_exact_hits_in_window": int((outcome[w] == scoring.EXACT_LIB).sum()),
              "equality_hits_in_window": int(score.equality[w].sum()),
              "driver_summary": (work / "summary.txt").read_text()}
    return results.write_result(out_dir, cell.cell_id, header, outcome=outcome,
                                recall=score.recall.astype(np.float32), equality=score.equality,
                                approximate=score.approximate, stream_ids=ids)
