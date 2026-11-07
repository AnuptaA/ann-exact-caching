"""Q1 workload generation for one dataset: idempotent stages with done markers.

fetch, embed, to_f32, inputs, groundtruth, pg_load, pg_index, tune_ef,
answer_table, backend_recall. Each stage writes <data>/<dataset>/.done/<stage>.json
when it finishes; a rerun skips finished stages. --force clears a stage and all later ones.
"""

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from harness import data
from harness.datasets import DatasetSpec, get_spec
from workloads.q1 import K, groundtruth, streams as stream_mod
from workloads.q1.recall import recall_batch

logger = logging.getLogger(__name__)

STAGES = ("fetch", "embed", "to_f32", "inputs", "groundtruth", "pg_load", "pg_index",
          "tune_ef", "answer_table", "backend_recall")
LIST_SIZE = 50
TUNE_QUERIES = 2000


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
    def db_name(self) -> str:
        return f"q1_{self.spec.name}"

    def db_dsn(self) -> str:
        from psycopg.conninfo import make_conninfo
        return make_conninfo(self.dsn, dbname=self.db_name)


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


# -- stages -----------------------------------------------------------------


def stage_fetch(ctx: Context) -> dict:
    ctx.raw.mkdir(parents=True, exist_ok=True)
    ctx.spec.fetch(ctx.raw)
    return {}


def stage_embed(ctx: Context) -> dict:
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
                                       base.shape[1], workers=ctx.extra.get("index_workers", 8))
    return info


def _build_key(ctx: Context) -> str:
    return ctx.spec.name if ctx.spec.name in ("spacev", "sphere", "sift10m", "esci") else "spacev"


def _load_universe(ctx: Context):
    from workloads.q1.cli import load_universe
    return load_universe(ctx.root / "universe.npz")


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
        rows = ef_tuning.sweep(pgvector.PgvectorBackend(conn), queries, base, kth)
    choice = ef_tuning.choose_ef(rows, ctx.list_size)
    result = {"sweep": rows, "choice": choice, "ef_search": choice["ef_search"],
              "list_size": ctx.list_size, "n_queries": len(pick)}
    (ctx.root / "tune.json").write_text(json.dumps(result, indent=1))
    return {"ef_search": choice["ef_search"], "recall": choice["grid_recall"]}


def stage_answer_table(ctx: Context) -> dict:
    from harness.backend.answer_table import load_answer_table
    tune = json.loads((ctx.root / "tune.json").read_text())
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    n = max(1, ctx.pg_workers)
    procs = [subprocess.Popen([sys.executable, "-m", "harness", "answer-table",
                               "--dataset-dir", str(ctx.root), "--dsn", ctx.db_dsn(),
                               "--list-size", str(tune["list_size"]),
                               "--ef-search", str(tune["ef_search"]), "--shard", f"{i}/{n}"],
                              env=env) for i in range(n)]
    codes = [p.wait() for p in procs]
    if any(codes):
        raise RuntimeError(f"answer-table shards failed with exit codes {codes}")
    ids, _ = load_answer_table(ctx.root / "answers")
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
    (ctx.root / "backend_recall.json").write_text(json.dumps(out, indent=1))
    return {"mean_recall_all_ids": out["mean_recall_all_ids"]}


RUNNERS = {"fetch": stage_fetch, "embed": stage_embed, "to_f32": stage_to_f32,
           "inputs": stage_inputs, "groundtruth": stage_groundtruth, "pg_load": stage_pg_load,
           "pg_index": stage_pg_index, "tune_ef": stage_tune_ef,
           "answer_table": stage_answer_table, "backend_recall": stage_backend_recall}


def run_pipeline(ctx: Context, stages=None, force=None, runners=None) -> list[str]:
    """Run the requested stages in order, skipping finished ones; returns the stages that ran."""
    runners = runners or RUNNERS
    wanted = list(stages) if stages else list(STAGES)
    for name in wanted + ([force] if force else []):
        if name not in STAGES:
            raise ValueError(f"unknown stage {name!r}; choose from {STAGES}")
    if force:
        clear_from(ctx, force)
    ran = []
    for stage in STAGES:
        if stage not in wanted or is_done(ctx, stage):
            continue
        logger.info(f"[{ctx.spec.name}] stage {stage}")
        start = time.time()
        info = runners[stage](ctx) or {}
        marker = _marker(ctx, stage)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"seconds": round(time.time() - start, 1), **info}))
        ran.append(stage)
    return ran


GPU_QUERY_CHUNK = 8192     # larger query chunks mean fewer passes over the base
GPU_BASE_CHUNK = 131072


def make_context(dataset: str, data_dir, dsn: str = "", device: str = "auto", **kw) -> Context:
    dev = auto_device() if device == "auto" else device
    kw = {k: v for k, v in kw.items() if v is not None}
    if dev == "cuda":
        kw.setdefault("gt_query_chunk", GPU_QUERY_CHUNK)
        kw.setdefault("gt_base_chunk", GPU_BASE_CHUNK)
    return Context(Path(data_dir), get_spec(dataset), dsn, dev, **kw)
