"""Q1 workload generation for one dataset: idempotent stages with done markers.

fetch, embed, to_f32, inputs, groundtruth, pg_load, pg_index, tune_ef,
answer_table, backend_recall. Each stage writes <data>/<dataset>/.done/<stage>.json
when it finishes; a rerun skips finished stages. --force clears a stage and all later ones.
"""

import contextlib
import json
import logging
import logging.handlers
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from harness import data, fingerprints
from harness.datasets import DatasetSpec, get_spec
from workloads.q1 import K, atomic, groundtruth, streams as stream_mod
from workloads.q1.recall import recall_batch

logger = logging.getLogger(__name__)

SLOW_NUMPY_FLOPS = 5e13        # ground truth on numpy beyond this needs Q1_ALLOW_SLOW=1 (about 15 min)
EF_MAX = 1000
LOG_TAIL_LINES = 40

STAGES = ("fetch", "embed", "to_f32", "inputs", "groundtruth", "pg_load", "pg_index",
          "tune_ef", "answer_table", "backend_recall")
LIST_SIZE = 200   # entries up to 50 (1,2,5 x K) plus shell ranks up to 200; ef_search >= L
TUNE_QUERIES = 2000
LOG_MAX_BYTES = 20 * 2 ** 20
LOG_BACKUPS = 5


@dataclass
class Context:
    data_dir: Path
    spec: DatasetSpec
    dsn: str = ""                     # base connection string; the database is q1_<dataset>
    device: str = "numpy"
    encode_python: str = sys.executable
    pg_workers: int = 4
    gt_query_chunk: int = groundtruth.QUERY_CHUNK
    gt_base_chunk: int = groundtruth.BASE_CHUNK
    list_size: int = LIST_SIZE
    tune_queries: int = TUNE_QUERIES
    seed: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def root(self) -> Path:
        return Path(self.data_dir) / self.spec.name

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def logs(self) -> Path:
        return self.root / ".logs"

    @property
    def n_ids(self) -> int:
        sp = self.spec
        return sp.n_groups * sp.group_size + sp.n_groups + self.n_official

    @property
    def n_official(self) -> int:
        return int(data.read_f32bin(self.root / "official.f32bin").shape[0])

    @property
    def db_name(self) -> str:
        return f"q1_{self.spec.name}"

    def db_dsn(self) -> str:
        from harness.backend.pgvector import with_keepalives
        return with_keepalives(self.dsn, self.db_name)


def auto_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "numpy"
    except ImportError:
        return "numpy"


def _marker(ctx: Context, stage: str) -> Path:
    return ctx.root / ".done" / f"{stage}.json"


def is_done(ctx: Context, stage: str) -> bool:
    return _marker(ctx, stage).exists()


def clear_from(ctx: Context, stage: str) -> None:
    for later in STAGES[STAGES.index(stage):]:
        _marker(ctx, later).unlink(missing_ok=True)




class StageError(RuntimeError):
    """A stage failed; the message names the dataset and stage and carries a log tail."""


def tail(path: Path, lines: int = LOG_TAIL_LINES) -> str:
    try:
        return "\n".join(Path(path).read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return f"(no log at {path})"


@contextlib.contextmanager
def file_log(ctx: "Context"):
    """Also write INFO logs to <data>/<dataset>/.logs/generate.log while a pipeline runs."""
    ctx.logs.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(ctx.logs / "generate.log", maxBytes=LOG_MAX_BYTES,
                                                   backupCount=LOG_BACKUPS)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    old_level = root.level
    root.addHandler(handler)
    root.setLevel(min(old_level or logging.INFO, logging.INFO))
    try:
        yield handler
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
        handler.close()


def env_info(ctx: "Context") -> dict:
    """Versions and identity of this run, written next to the data (best effort)."""
    from harness.runner import results
    info = {**results.git_state(), "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "device": ctx.device, "list_size": ctx.list_size}
    try:
        import torch
        info.update(torch=torch.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except ImportError:
        info["torch"] = None
    driver = Path(ctx.extra.get("driver_dir", "build/replay")) / "ak_replay_standard"
    toolchain = driver.parent / "TOOLCHAIN.txt"
    if toolchain.exists():
        info["driver_toolchain"] = toolchain.read_text().strip().splitlines()
    if driver.exists():
        import hashlib
        info["driver_sha1"] = hashlib.sha1(driver.read_bytes()).hexdigest()
    if ctx.dsn:
        try:
            import psycopg
            from psycopg.conninfo import make_conninfo
            with psycopg.connect(make_conninfo(ctx.dsn, dbname="postgres"), connect_timeout=5) as conn:
                info["postgres"] = conn.execute("SHOW server_version").fetchone()[0]
                row = conn.execute("SELECT default_version FROM pg_available_extensions "
                                   "WHERE name = 'vector'").fetchone()
                info["pgvector_available"] = row[0] if row else None
        except Exception as exc:
            info["postgres"] = f"unreachable: {exc!r}"
    return info


def total_ram_bytes() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        return 0


def pg_data_dir(ctx: "Context") -> Path | None:
    """Where the server writes: Q1_PGDATA, else asked from the server itself."""
    if os.environ.get("Q1_PGDATA"):
        return Path(os.environ["Q1_PGDATA"])
    try:
        import psycopg
        from psycopg.conninfo import make_conninfo
        with psycopg.connect(make_conninfo(ctx.dsn, dbname="postgres"), connect_timeout=10) as conn:
            return Path(conn.execute("SHOW data_directory").fetchone()[0])
    except Exception as exc:
        logger.warning(f"cannot ask PostgreSQL for its data directory: {exc!r}")
        return None


def check_resources(ctx: "Context", stage: str) -> None:
    """Refuse or warn before a heavy stage; every message says what to do."""
    from workloads.q1.datasets.common import check_free_disk
    n, d = ctx.spec.n_base, ctx.spec.dim
    gib = 2 ** 30
    if stage == "to_f32":
        check_free_disk(ctx.data_dir, n * d * 4 + ctx.spec.n_base // 10 * d * 4 + gib)
    elif stage == "groundtruth":
        check_free_disk(ctx.data_dir, ctx.n_ids * 100 * 12 + gib)
        flops = 2.0 * ctx.n_ids * n * d
        if ctx.device == "numpy" and flops > SLOW_NUMPY_FLOPS and not ctx.extra.get("allow_slow"):
            raise StageError(
                f"ground truth for {ctx.spec.name} on the numpy device needs about {flops:.1e} flops "
                f"(hours); run on a GPU (Q1_DEVICE=cuda) or set Q1_ALLOW_SLOW=1 to accept it")
        base_bytes = n * d * 4
        ram = total_ram_bytes()
        if ram and base_bytes > 0.6 * ram:
            logger.warning(f"the float32 base ({base_bytes / gib:.0f} GiB) is more than 60% of RAM "
                           f"({ram / gib:.0f} GiB): the CPU recompute step will read rows from disk")
    elif stage in ("pg_load", "pg_index"):
        from harness.deploy import preflight
        need = preflight.need_bytes(ctx.spec.name) if ctx.spec.name in preflight.DATASETS else None
        if need is None:
            return
        pgdata = pg_data_dir(ctx)
        if pgdata is None:
            logger.warning("PostgreSQL data directory unknown (set Q1_PGDATA): disk not checked")
        elif stage == "pg_load":
            check_free_disk(pgdata, need["table"] + need["wal"] + gib)
        else:
            check_free_disk(pgdata, need["index"] + need["wal"] + gib)
        ram = total_ram_bytes()
        if stage == "pg_index" and ram and ram < need["ram"]:
            raise StageError(f"HNSW build for {ctx.spec.name} needs about {need['ram'] / gib:.0f} GiB of RAM "
                             f"(graph in maintenance_work_mem); this machine has {ram / gib:.0f} GiB")
    elif stage == "answer_table":
        check_free_disk(ctx.data_dir, ctx.n_ids * ctx.list_size * 8 + gib)


# -- stages -----------------------------------------------------------------


def stage_fetch(ctx: Context) -> dict:
    ctx.raw.mkdir(parents=True, exist_ok=True)
    ctx.spec.fetch(ctx.raw)
    return {}


def stage_embed(ctx: Context) -> dict:
    if ctx.spec.embed is not None and ctx.spec.name in ("esci", "sphere") and ctx.device != "cuda" \
            and not ctx.extra.get("allow_slow"):
        raise StageError(f"encoding the {ctx.spec.name} texts on the CPU takes many hours; use a GPU "
                         "(Q1_DEVICE=cuda) or set Q1_ALLOW_SLOW=1 to accept it")
    if ctx.spec.embed is not None:
        ctx.spec.embed(ctx.raw, {"device": "cuda" if ctx.device == "cuda" else "cpu",
                                 "encode_python": ctx.encode_python})
    return {}


def stage_to_f32(ctx: Context) -> dict:
    base, official = ctx.spec.base(ctx.raw), ctx.spec.official(ctx.raw)
    if base.shape != (ctx.spec.n_base, ctx.spec.dim):
        raise ValueError(f"{ctx.spec.name} base is {base.shape}, expected {(ctx.spec.n_base, ctx.spec.dim)}")
    if official.shape[1] != ctx.spec.dim:
        raise ValueError(f"{ctx.spec.name} official queries have d={official.shape[1]}")
    for name, rows in (("base", base), ("official", official)):
        if not (ctx.root / f"{name}.f32bin").exists():
            data.write_f32bin(ctx.root / f"{name}.f32bin", rows)
    return {"n_base": int(base.shape[0]), "n_official": int(official.shape[0])}


def stage_inputs(ctx: Context) -> dict:
    from harness.runner.prepare import write_universe_and_streams
    official = data.read_f32bin(ctx.root / "official.f32bin")
    s = ctx.spec
    write_universe_and_streams(ctx.root, len(official), s.n_groups, s.group_size, s.n_requests,
                               no_locality=s.no_locality, stream_seed=ctx.seed)
    return {"n_official": len(official)}


def stage_groundtruth(ctx: Context) -> dict:
    from harness.runner.prepare import prepare_ground_truth
    prepare_ground_truth(ctx.root, ctx.device, ctx.gt_query_chunk, ctx.gt_base_chunk)
    return {"device": ctx.device}


def _ensure_database(ctx: Context) -> None:
    import psycopg
    from psycopg.conninfo import make_conninfo
    with psycopg.connect(make_conninfo(ctx.dsn, dbname="postgres"), autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (ctx.db_name,)).fetchone()
        if not exists:
            conn.execute(f'CREATE DATABASE "{ctx.db_name}"')


def stage_pg_load(ctx: Context) -> dict:
    import psycopg
    from harness.backend import load, pgvector
    _ensure_database(ctx)
    base = data.read_f32bin(ctx.root / "base.f32bin")
    with psycopg.connect(ctx.db_dsn()) as conn:
        load.load_base(conn, base)
        version = pgvector.check_version(conn)
    return {"rows": int(base.shape[0]), "pgvector": version}


def stage_pg_index(ctx: Context) -> dict:
    import psycopg
    from harness.backend import build_index
    base = data.read_f32bin(ctx.root / "base.f32bin")
    with psycopg.connect(ctx.db_dsn()) as conn:
        conn.execute("DROP INDEX IF EXISTS base_embedding_hnsw")
        conn.commit()
        info = build_index.build_index(conn, "base", "embedding", _build_key(ctx), base.shape[0],
                                       base.shape[1], workers=ctx.extra.get("index_workers", 8),
                                       dsn=ctx.db_dsn())
    return info


def _build_key(ctx: Context) -> str:
    return ctx.spec.name if ctx.spec.name in ("spacev", "sphere", "sift10m", "esci") else "spacev"


def _load_universe(ctx: Context):
    from workloads.q1.cli import load_universe
    return load_universe(ctx.root / "universe.npz")


def prewarm_index(conn, index: str = "base_embedding_hnsw") -> None:
    """Load the index into shared buffers (best effort; speeds tuning and the answer table)."""
    try:
        conn.execute("CREATE EXTENSION IF NOT EXISTS pg_prewarm")
        blocks = conn.execute("SELECT pg_prewarm(%s)", (index,)).fetchone()[0]
        logger.info(f"pg_prewarm loaded {blocks} blocks of {index}")
    except Exception as exc:
        logger.warning(f"pg_prewarm skipped: {exc!r}")


def stage_tune_ef(ctx: Context) -> dict:
    import psycopg
    from harness.backend import ef_tuning, pgvector
    base = data.read_f32bin(ctx.root / "base.f32bin")
    official = data.read_f32bin(ctx.root / "official.f32bin")
    uni = _load_universe(ctx)
    _, gt_dist = groundtruth.load_ground_truth(ctx.root / "gt")
    rng = np.random.default_rng(ctx.seed)
    pick = np.sort(rng.choice(len(official), min(ctx.tune_queries, len(official)), replace=False))
    queries = np.asarray(official[pick], dtype=np.float32)
    kth = gt_dist[uni.pool_id(0) + pick, K - 1]
    with psycopg.connect(ctx.db_dsn(), autocommit=True) as conn:
        pgvector.check_version(conn)
        prewarm_index(conn)
        rows = ef_tuning.sweep(pgvector.PgvectorBackend(conn), queries, base, kth)
    for row in rows:
        logger.info(f"ef_search {row['ef_search']}: recall@10 {row['recall']:.4f}, "
                    f"{row['latency_ms']:.2f} ms/query")
    choice = ef_tuning.choose_ef(rows, ctx.list_size)
    if choice["ef_search"] > EF_MAX:
        raise StageError(f"list size {ctx.list_size} needs ef_search {choice['ef_search']} > {EF_MAX} "
                         f"(pgvector's limit); lower Q1_LIST_SIZE")
    result = {"sweep": rows, "choice": choice, "ef_search": choice["ef_search"],
              "list_size": ctx.list_size, "n_queries": len(pick)}
    atomic.write_json(ctx.root / "tune.json", result, indent=1)
    return {"ef_search": choice["ef_search"], "recall": choice["grid_recall"]}


def stage_answer_table(ctx: Context) -> dict:
    from harness.backend.answer_table import load_answer_table
    tune = json.loads((ctx.root / "tune.json").read_text())
    if tune["list_size"] != ctx.list_size:
        raise StageError(f"tune.json was made for list size {tune['list_size']} but this run uses "
                         f"{ctx.list_size}; rerun with --force tune_ef")
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    n = max(1, ctx.pg_workers)
    ctx.logs.mkdir(parents=True, exist_ok=True)
    logs = [ctx.logs / f"answer_table_{i}.log" for i in range(n)]
    handles = [open(path, "w") for path in logs]
    try:
        procs = [subprocess.Popen([sys.executable, "-m", "harness", "answer-table",
                                   "--dataset-dir", str(ctx.root), "--dsn", ctx.db_dsn(),
                                   "--list-size", str(tune["list_size"]),
                                   "--ef-search", str(tune["ef_search"]), "--shard", f"{i}/{n}"],
                                  env=env, stdout=handles[i], stderr=subprocess.STDOUT)
                 for i in range(n)]
        logger.info(f"{n} answer-table shards running; logs in {ctx.logs}/answer_table_<i>.log")
        codes = [proc.wait() for proc in procs]
    finally:
        for handle in handles:
            handle.close()
    if any(codes):
        bad = [i for i, c in enumerate(codes) if c]
        raise StageError(f"answer-table shards {bad} failed (exit codes {codes}); last lines of shard "
                         f"{bad[0]}:\n{tail(logs[bad[0]])}")
    ids, dist = load_answer_table(ctx.root / "answers")
    atomic.save_npz(ctx.root / "answers_first10.npz", ids=ids[:, :K].astype(np.int32), dist=dist[:, :K])
    return {"rows": int(len(ids)), "list_size": int(ids.shape[1]), "ef_search": tune["ef_search"]}


def stage_backend_recall(ctx: Context) -> dict:
    from harness.backend.answer_table import load_answer_table
    from harness.runner.prepare import request_vectors_of
    base = data.read_f32bin(ctx.root / "base.f32bin")
    ids, _ = load_answer_table(ctx.root / "answers")
    _, gt_dist = groundtruth.load_ground_truth(ctx.root / "gt")
    vectors = request_vectors_of(ctx.root)
    recall = np.empty(len(ids))
    for start in range(0, len(ids), 20000):
        sl = slice(start, start + 20000)
        recall[sl] = recall_batch(vectors[sl], ids[sl, :K], base, gt_dist[sl, K - 1])
    z = np.load(ctx.root / f"streams_seed{ctx.seed}.npz")
    fraction = float(z["history_fraction"])
    per_stream = {}
    for name in z.files:
        if name == "history_fraction":
            continue
        req = z[name]
        test = req[stream_mod.history_split(len(req), fraction):]
        per_stream[name] = {"requests": int(len(req)), "mean_recall_all": float(recall[req].mean()),
                            "mean_recall_test_window": float(recall[test].mean())}
    out = {"mean_recall_all_ids": float(recall.mean()), "per_stream": per_stream,
           "ef_search": json.loads((ctx.root / "tune.json").read_text())["ef_search"]}
    atomic.write_json(ctx.root / "backend_recall.json", out, indent=1)
    fingerprints.store(ctx.root, ctx.seed)
    return {"mean_recall_all_ids": out["mean_recall_all_ids"]}


RUNNERS = {"fetch": stage_fetch, "embed": stage_embed, "to_f32": stage_to_f32,
           "inputs": stage_inputs, "groundtruth": stage_groundtruth, "pg_load": stage_pg_load,
           "pg_index": stage_pg_index, "tune_ef": stage_tune_ef,
           "answer_table": stage_answer_table, "backend_recall": stage_backend_recall}


# -- output validation (sampled, so it stays cheap on 30 GB files) --------------


def _sample_rows(arr, n: int = 4096, seed: int = 0) -> np.ndarray:
    idx = np.sort(np.random.default_rng(seed).choice(len(arr), min(n, len(arr)), replace=False))
    return np.asarray(arr[idx], dtype=np.float32)


def check_vectors(name: str, arr, n: int, d: int) -> None:
    if arr.shape != (n, d):
        raise ValueError(f"{name}: shape {arr.shape}, expected {(n, d)}")
    rows = _sample_rows(arr)
    if not np.isfinite(rows).all():
        raise ValueError(f"{name}: NaN or infinite values in a sample of {len(rows)} rows")
    if (np.abs(rows).sum(axis=1) == 0).mean() > 0.5:
        raise ValueError(f"{name}: more than half of a sample of rows are all zeros")


def validate_stage(ctx: Context, stage: str) -> None:
    """Raise ValueError when a finished stage left unusable output."""
    s = ctx.spec
    if stage == "fetch":
        if not any(ctx.raw.iterdir()):
            raise ValueError(f"fetch left {ctx.raw} empty")
    elif stage == "embed" and s.embed is not None:
        official = s.official(ctx.raw)
        check_vectors("official queries", official, official.shape[0], s.dim)
    elif stage == "to_f32":
        base = data.read_f32bin(ctx.root / "base.f32bin")
        check_vectors("base.f32bin", base, s.n_base, s.dim)
        official = data.read_f32bin(ctx.root / "official.f32bin")
        check_vectors("official.f32bin", official, official.shape[0], s.dim)
    elif stage == "inputs":
        uni = _load_universe(ctx)
        z = np.load(ctx.root / f"streams_seed{ctx.seed}.npz")
        for name in z.files:
            if name == "history_fraction":
                continue
            ids = z[name]
            if ids.min() < 0 or ids.max() >= uni.n_ids:
                raise ValueError(f"stream {name} has ids outside [0, {uni.n_ids})")
            if name != "no_locality" and len(ids) != s.n_requests:
                raise ValueError(f"stream {name} has {len(ids)} requests, expected {s.n_requests}")
        if s.no_locality and len(z["no_locality"]) != uni.n_official:
            raise ValueError("no_locality stream must contain every official query once")
    elif stage == "groundtruth":
        ids, dist = groundtruth.load_ground_truth(ctx.root / "gt")
        if ids.shape != (ctx.n_ids, 100) or dist.shape != ids.shape:
            raise ValueError(f"ground truth shape {ids.shape}, expected {(ctx.n_ids, 100)}")
        if ids.min() < 0 or ids.max() >= s.n_base or not np.isfinite(dist).all() or (dist < 0).any():
            raise ValueError("ground truth has out-of-range ids or invalid distances")
        if (np.diff(dist, axis=1) < -1e-6 * np.maximum(dist[:, 1:], 1e-30)).any():
            raise ValueError("ground truth distances are not sorted")
    elif stage == "pg_index":
        import psycopg
        from harness.backend.build_index import index_is_valid
        with psycopg.connect(ctx.db_dsn()) as conn:
            if not index_is_valid(conn, "base_embedding_hnsw"):
                raise ValueError("base_embedding_hnsw is missing or invalid")
    elif stage == "tune_ef":
        tune = json.loads((ctx.root / "tune.json").read_text())
        if not ctx.list_size <= tune["ef_search"] <= EF_MAX:
            raise ValueError(f"ef_search {tune['ef_search']} outside [{ctx.list_size}, {EF_MAX}]")
    elif stage == "answer_table":
        ids = np.load(ctx.root / "answers_first10.npz")["ids"]
        if ids.shape != (ctx.n_ids, K) or ids.min() < 0 or ids.max() >= s.n_base:
            raise ValueError(f"answer table first-10 has shape {ids.shape} or ids out of range")
        validate_answer_table(ctx)
    elif stage == "backend_recall":
        rec = json.loads((ctx.root / "backend_recall.json").read_text())
        if not 0.0 < rec["mean_recall_all_ids"] <= 1.0:
            raise ValueError(f"implausible backend recall {rec['mean_recall_all_ids']}")


RECALL_TOLERANCE = 0.02   # answer table vs tune.json on the tuning queries (same index, same ef_search)


def validate_answer_table(ctx: Context) -> dict:
    """Full block checks, stored distances against a recomputation, recall against tune.json."""
    from harness.backend.answer_table import load_answer_table, validate_blocks
    from harness.runner.prepare import request_vectors_of
    from workloads.q1.distance import l2_to_ids
    tune = json.loads((ctx.root / "tune.json").read_text())
    validate_blocks(ctx.root / "answers", ctx.spec.n_base, ctx.list_size, ctx.n_ids)
    ids, dist = load_answer_table(ctx.root / "answers")
    base = data.read_f32bin(ctx.root / "base.f32bin")
    vectors = request_vectors_of(ctx.root)
    rng = np.random.default_rng(ctx.seed + 1)
    for row in np.sort(rng.choice(len(ids), min(200, len(ids)), replace=False)):
        fresh = l2_to_ids(np.asarray(vectors[int(row):int(row) + 1][0], dtype=np.float32), base, ids[row])
        if not np.allclose(fresh, dist[row], rtol=1e-4, atol=1e-5):
            raise ValueError(f"answer table row {row}: stored distances differ from a recomputation")
    official = data.read_f32bin(ctx.root / "official.f32bin")
    uni = _load_universe(ctx)
    _, gt_dist = groundtruth.load_ground_truth(ctx.root / "gt")
    pick = np.sort(np.random.default_rng(ctx.seed).choice(
        len(official), min(tune["n_queries"], len(official)), replace=False))
    rows = uni.pool_id(0) + pick
    recall = float(recall_batch(np.asarray(official[pick], dtype=np.float32), ids[rows, :K], base,
                                gt_dist[rows, K - 1]).mean())
    expected = tune["choice"]["grid_recall"]
    if abs(recall - expected) > RECALL_TOLERANCE:
        raise ValueError(f"answer table recall@{K} on the tuning queries is {recall:.4f}, tune.json "
                         f"recorded {expected:.4f} (tolerance {RECALL_TOLERANCE})")
    return {"recall_on_tuning_queries": recall, "tune_recall": expected}


def run_pipeline(ctx: Context, stages=None, force=None, runners=None, validate=None) -> list[str]:
    """Run the requested stages in order, skipping finished ones; returns the stages that ran.

    Each stage is resource-checked first, validated afterwards, and logged to
    <data>/<dataset>/.logs/generate.log. A failure writes .failed/<stage>.json and raises
    StageError with the log tail; the stage marker is only written on success.
    """
    real = runners is None
    runners = runners or RUNNERS
    validate = real if validate is None else validate
    wanted = list(stages) if stages else list(STAGES)
    for name in wanted + ([force] if force else []):
        if name not in STAGES:
            raise ValueError(f"unknown stage {name!r}; choose from {STAGES}")
    ran = []
    with file_log(ctx):
        if force:
            clear_from(ctx, force)
        if real:
            from harness.deploy import preflight
            problems = preflight.check_data_volume(ctx.data_dir)
            if problems:
                raise StageError(problems[0])
            ctx.root.mkdir(parents=True, exist_ok=True)
            atomic.write_json(ctx.root / "env.json", env_info(ctx), indent=1)
        for stage in STAGES:
            if stage not in wanted or is_done(ctx, stage):
                continue
            logger.info(f"[{ctx.spec.name}] stage {stage} starting")
            start = time.time()
            failed = ctx.root / ".failed" / f"{stage}.json"
            try:
                if real:
                    check_resources(ctx, stage)
                info = runners[stage](ctx) or {}
                if validate:
                    validate_stage(ctx, stage)
            except Exception as exc:
                failed.parent.mkdir(parents=True, exist_ok=True)
                failed.write_text(json.dumps({"stage": stage, "error": repr(exc),
                                              "traceback": traceback.format_exc(),
                                              "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
                logger.exception(f"[{ctx.spec.name}] stage {stage} FAILED after "
                                 f"{(time.time() - start) / 60:.1f} min")
                raise StageError(
                    f"[{ctx.spec.name}] stage {stage} failed: {exc}\n"
                    f"details: {failed}\nlast log lines:\n{tail(ctx.logs / 'generate.log')}") from exc
            failed.unlink(missing_ok=True)
            marker = _marker(ctx, stage)
            marker.parent.mkdir(parents=True, exist_ok=True)
            seconds = time.time() - start
            atomic.write_json(marker, {"seconds": round(seconds, 1), **info})
            logger.info(f"[{ctx.spec.name}] stage {stage} done in {seconds / 60:.1f} min {info}")
            ran.append(stage)
    return ran


GPU_QUERY_CHUNK = 8192     # larger query chunks mean fewer passes over the base
GPU_BASE_CHUNK = 131072


def make_context(dataset: str, data_dir, dsn: str = "", device: str = "auto", **kw) -> Context:
    dev = auto_device() if device == "auto" else device
    if device == "auto" and dev == "numpy":
        logger.warning("no CUDA device found: ground truth and encoders will run on the CPU "
                       "(fine for the tiny smoke test, hours for real datasets)")
    if dev == "cuda" and auto_device() != "cuda":
        raise StageError("device cuda requested but torch cannot see a CUDA device "
                         "(driver missing, CPU-only torch build, or no GPU on this instance)")
    kw = {k: v for k, v in kw.items() if v is not None}
    if dev == "cuda":
        kw.setdefault("gt_query_chunk", GPU_QUERY_CHUNK)
        kw.setdefault("gt_base_chunk", GPU_BASE_CHUNK)
    return Context(Path(data_dir), get_spec(dataset), dsn, dev, **kw)
