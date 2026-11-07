"""Run one grid cell: build driver inputs, call the replay driver, score, write results."""

import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from harness import config, data, fingerprints, scoring, thresholds
from harness.backend.answer_table import load_answer_table
from harness.runner import manifest, results
from harness.runner.manifest import Cell
from workloads.q1 import K, groundtruth, streams as stream_mod, universe
from workloads.q1.cli import load_universe

logger = logging.getLogger(__name__)

AKER_REVISION = "e7639a52aec1320a47c5fa0b833c7214024931c9"
REPO = Path(__file__).resolve().parents[2]


class DatasetDir:
    """Layout of one prepared dataset (built by `make q1-generate`)."""

    def __init__(self, root: Path, dataset: str, seed: int = 0):
        self.root = Path(root) / dataset
        self.dataset = dataset
        self.seed = seed

        def need(path: Path, stage: str) -> Path:
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} is missing: run `make q1-generate DATASET={dataset}` "
                    f"(stage {stage}) before running systems on {dataset}")
            return path

        self.base_path = need(self.root / "base.f32bin", "to_f32")
        official_path = need(self.root / "official.f32bin", "to_f32")
        universe_path = need(self.root / "universe.npz", "inputs")
        streams_path = need(self.root / f"streams_seed{seed}.npz", "inputs")
        meta_path = need(self.root / "answers" / "meta.json", "answer_table")
        need(self.root / "gt", "groundtruth")
        self.fingerprints = fingerprints.verify(self.root, seed)
        self.input_id = fingerprints.combine(self.fingerprints)
        self.base = data.read_f32bin(self.base_path)
        self.official = data.read_f32bin(official_path)
        self.universe = load_universe(universe_path)
        z = np.load(streams_path)
        self.streams = {k: z[k] for k in z.files if k != "history_fraction"}
        self.history_fraction = float(z["history_fraction"])
        first10 = self.root / "answers_first10.npz"
        if first10.exists():
            f = np.load(first10)
            self.answer_ids, self.answer_dist = f["ids"], f["dist"]
        else:
            self.answer_ids, self.answer_dist = load_answer_table(self.root / "answers")
        self.gt_ids, self.gt_dist = groundtruth.load_ground_truth(self.root / "gt")
        self.backend = json.loads(meta_path.read_text())

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
        path = Path(cache_dir) / f"{cell.dataset}-{cell.stream}-s{seed}-{ds.input_id}.json"
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
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")      # parallel configs may race
        tmp.write_text(json.dumps(out))
        tmp.replace(path)
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


def parse_startup(driver_log: str) -> dict:
    """Key=value pairs of the driver's PARAM line and of its FILTER line(s)."""
    out: dict = {}
    for line in driver_log.splitlines():
        if line.startswith("PARAM "):
            out["param"] = dict(item.split("=", 1) for item in line.split()[1:])
        elif line.startswith("FILTER "):
            out.setdefault("filter", dict(item.split("=", 1) for item in line.split()[1:]))
    return out


def check_startup(startup: dict, settings: config.AkerSettings, n_base: int, n_requests: int) -> None:
    """The values the driver and FAISS actually hold must be the intended ones."""
    param, flt = startup.get("param"), startup.get("filter")
    if param is None or flt is None:
        raise RuntimeError("driver log has no PARAM/FILTER startup lines (stale driver build?)")
    want = {"mode": settings.mode, "pool_size": settings.pool_size, "k": settings.k,
            "top_delta": settings.top_delta, "alpha_tighten": settings.alpha_tighten,
            "alpha_loosen": settings.alpha_loosen, "filter_m": settings.filter_m,
            "filter_ef_construction": settings.filter_ef_construction,
            "filter_ef_search": settings.filter_ef_search, "base_n": n_base, "requests": n_requests,
            "global_thresh_sq_l2": settings.global_thresh}
    for key, expected in want.items():
        got = param.get(key)
        same = (got == expected) if isinstance(expected, str) else (
            got is not None and abs(float(got) - float(expected)) <= 1e-5 * max(1.0, abs(float(expected))))
        if not same:
            raise RuntimeError(f"driver started with {key}={got}, expected {expected}")
    faiss_view = {"faiss_M": settings.filter_m, "efConstruction": settings.filter_ef_construction,
                  "efSearch": settings.filter_ef_search}
    for key, expected in faiss_view.items():
        if int(flt.get(key, -1)) != expected:
            raise RuntimeError(f"FAISS holds {key}={flt.get(key)}, expected {expected}")


def counter_disagreements(outcome: np.ndarray, final_status: str) -> int:
    """Absolute differences between our outcome counts and the library's own hit counters.

    Library "hit" is a lookup served by the entry stored under the same id (our equality);
    "sim-hit" is every other hit, linked repeats included (our approximate); "miss" the rest.
    """
    counters = {name: int(re.search(rf"Cache {name}: (\d+)", final_status).group(1))
                for name in ("hit", "sim-hit", "miss")}
    ours = {"hit": int((outcome == scoring.EQUALITY).sum()),
            "sim-hit": int(((outcome == scoring.APPROX) | (outcome == scoring.LINKED)).sum()),
            "miss": int((outcome == scoring.MISS).sum())}
    return sum(abs(counters[k] - ours[k]) for k in counters)


def peak_pool_vectors(status_log: str) -> int:
    """Largest sampled count of unique pooled vectors in the driver's status log."""
    return max((int(m.group(1)) for m in POOL_LINE.finditer(status_log)), default=0)


def stream_inputs(cell: Cell, ds: DatasetDir, ids: np.ndarray, work_dir: Path) -> Path:
    """Driver inputs of one (dataset, stream, seed), shared by every system config.

    Built once into a temporary directory and renamed, so parallel runs never see a
    half-written set.
    """
    final = Path(work_dir) / "inputs" / f"{cell.dataset}-{cell.stream}-s{cell.stream_seed}-{ds.input_id}"
    if (final / "READY").exists():
        return final
    tmp = final.with_name(f"{final.name}.{os.getpid()}.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    t = time.time()
    data.write_f32bin(tmp / "requests.f32bin", ds.vectors_of(ids))
    ids.astype("<i8").tofile(tmp / "stream.i64")
    ds.answer_ids[ids][:, :K].astype("<i8").tofile(tmp / "answers.i64")
    ds.answer_dist[ids][:, :K].astype("<f4").tofile(tmp / "dists.f32")
    (tmp / "READY").write_text("")
    try:
        tmp.rename(final)
    except OSError:                       # another process finished first
        shutil.rmtree(tmp, ignore_errors=True)
    logger.info(f"built shared inputs for {final.name} in {time.time() - t:.0f}s")
    return final


def run_driver(binary: Path, args: list, work: Path, timeout: float | None) -> None:
    """Run the replay driver; a crash or timeout raises with the driver log tail."""
    log_path = work / "driver.log"
    with log_path.open("w") as log:
        try:
            subprocess.run([str(binary), *args], check=True, timeout=timeout, stdout=log,
                           stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"{binary.name} exited with code {exc.returncode}; driver log tail:\n"
                               f"{_tail(log_path)}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"{binary.name} timed out after {timeout}s; driver log tail:\n"
                               f"{_tail(log_path)}") from exc


def _generation_env(root: Path) -> dict:
    path = Path(root) / "env.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _tail(path: Path, lines: int = 30) -> str:
    return "\n".join(Path(path).read_text(errors="replace").splitlines()[-lines:])


def run_cell(cell: Cell, ds: DatasetDir, driver_dir: Path, out_dir: Path, work_dir: Path,
             timeout: float | None = None) -> Path:
    """Replay one cell; skips nothing here (the caller checks results.is_done)."""
    started = time.time()
    code = fingerprints.code_fingerprints(REPO, driver_dir)
    ids = ds.streams[cell.stream]
    history_len = stream_mod.history_split(len(ids), ds.history_fraction)
    logger.info(f"cell {cell.cell_id}: start ({len(ids)} requests)")
    settings, thresholds_info = settings_for(cell, ds, ids, history_len, cell.stream_seed,
                                             Path(work_dir) / "thresholds")
    inputs = stream_inputs(cell, ds, ids, work_dir)
    work = Path(work_dir) / cell.cell_id
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    binary = Path(driver_dir) / f"ak_replay_{settings.mode}"
    if not binary.exists():
        raise FileNotFoundError(f"{binary} not found: build the drivers (make q1-setup or make build-replay)")
    run_driver(binary, ["--base", str(ds.base_path), "--requests", str(inputs / "requests.f32bin"),
                        "--stream", str(inputs / "stream.i64"), "--answers", str(inputs / "answers.i64"),
                        "--dists", str(inputs / "dists.f32"), "--out-dir", str(work),
                        "--status-every", "100", *settings.driver_args()], work, timeout)
    outcome = np.fromfile(work / "outcome.u8", dtype=np.uint8)
    result_ids = np.fromfile(work / "result_ids.i64", dtype="<i8").reshape(len(ids), K)
    score = scoring.score_run(outcome, result_ids, ids, data.read_f32bin(inputs / "requests.f32bin"),
                              ds.base, ds.gt_dist[ids][:, K - 1], history_len)
    w = score.window
    startup = parse_startup((work / "driver.log").read_text())
    check_startup(startup, settings, len(ds.base), len(ids))
    driver_summary = dict(line.split("=", 1) for line in (work / "summary.txt").read_text().splitlines()
                          if "=" in line and not line.startswith("final_status"))
    peak = int(driver_summary["peak_pool"])
    if peak > settings.pool_size:
        raise RuntimeError(f"{cell.cell_id}: peak pool {peak} exceeds pool_size {settings.pool_size}")
    header = {"cell": cell.__dict__, "settings": settings.as_dict(),
              "config_hash": hashlib.sha1(json.dumps([cell.__dict__, settings.as_dict()],
                                                     sort_keys=True).encode()).hexdigest(),
              **results.git_state(), "backend": ds.backend["backend"],
              "ef_search": ds.backend["ef_search"], "list_size": ds.backend["list_size"],
              "aker_revision": AKER_REVISION, "history_len": history_len,
              "summary": score.summary(), "peak_pool_vectors": peak, "driver_startup": startup,
              "thresholds": thresholds_info,
              "counter_disagreements": counter_disagreements(outcome, (work / "summary.txt").read_text()),
              "equality_hits_in_window": int(score.equality[w].sum()),
              "approximate_filter_hits_in_window": int((outcome[w] == scoring.APPROX).sum()),
              "linked_hits_in_window": int((outcome[w] == scoring.LINKED).sum()),
              "seen_before_hits_in_window": int(score.seen_before_hit[w].sum()),
              "driver_summary": (work / "summary.txt").read_text(),
              "fingerprints": ds.fingerprints, "code": code,
              "reuse_key": fingerprints.reuse_key(ds.fingerprints, code),
              "generation_env": _generation_env(ds.root),
              "versions": {"python": platform.python_version(), "numpy": np.__version__}}
    path = results.write_result(out_dir, cell.cell_id, header, outcome=outcome,
                                recall=score.recall.astype(np.float32), equality=score.equality,
                                approximate=score.approximate, stream_ids=ids)
    summary = header["summary"]
    logger.info(f"cell {cell.cell_id}: done in {time.time() - started:.0f}s: hit rate "
                f"{summary['hit_rate']:.4f}, delivered recall {summary['delivered_recall']:.4f}, "
                f"peak pool {peak}/{settings.pool_size}")
    return path
