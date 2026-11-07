"""Command line: python -m harness <command> ..."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from harness import data
from harness.runner import manifest, replay, results

logger = logging.getLogger(__name__)


def _shard_arg(text: str) -> tuple[int, int]:
    index, count = text.split("/")
    return int(index), int(count)


def _cells(args):
    cells = manifest.build_grid(args.datasets, args.streams, seeds=tuple(args.seeds))
    if getattr(args, "systems", None):
        cells = [c for c in cells if c.system in args.systems]
    if getattr(args, "settings", None):
        cells = [c for c in cells if c.setting in args.settings]
    if getattr(args, "budgets", None):
        cells = [c for c in cells if c.budget_percent in args.budgets]
    return manifest.shard(cells, *args.shard)


def _grid(args):
    for cell in _cells(args):
        print(cell.cell_id)


def _run(args):
    cells = _cells(args)
    commit = results.git_state()["commit"]
    todo = [c for c in cells if not results.is_done(args.out_dir, c.cell_id, commit)]
    logger.info(f"{len(cells)} cells in this shard, {len(todo)} to run")
    loaded: dict = {}
    for cell in todo:
        key = (cell.dataset, cell.stream_seed)
        if key not in loaded:
            loaded = {key: replay.DatasetDir(Path(args.data_dir), cell.dataset, cell.stream_seed)}
        path = replay.run_cell(cell, loaded[key], Path(args.driver_dir), Path(args.out_dir),
                               Path(args.work_dir), timeout=args.timeout)
        logger.info(f"done {path.name}")


def _generate(args):
    from harness import pipeline
    datasets = args.dataset if args.dataset != ["all"] else list(pipeline_datasets())
    for name in datasets:
        ctx = pipeline.make_context(name, args.data_dir, args.dsn, args.device,
                                    encode_python=args.encode_python, pg_workers=args.pg_workers,
                                    gt_query_chunk=args.gt_query_chunk, gt_base_chunk=args.gt_base_chunk)
        ran = pipeline.run_pipeline(ctx, args.stages, args.force)
        print(f"{name}: ran {ran or 'nothing (all stages done)'}")


def pipeline_datasets():
    from harness.datasets import PRODUCTION
    return PRODUCTION


def _stages(args):
    from harness import pipeline
    print(" ".join(pipeline.STAGES))


def _collect(args):
    from harness import collect
    table = collect.rows(args.out_dir)
    print(collect.format_table(table))
    if args.csv:
        collect.write_csv(args.out_dir, args.csv)


def _fidelity(args):
    from harness import fidelity
    import json
    out = fidelity.run_fidelity(Path(args.data_dir), args.dsn, Path(args.driver_dir), Path(args.out_dir),
                                Path(args.work_dir), args.ef_search, device=args.device,
                                inspect_pickle=args.inspect_pickle)
    print(json.dumps(out, indent=1))


def _preflight(args):
    from harness.deploy import preflight
    extra = (["--require-gpu"] if args.require_gpu else []) + (["--dsn", args.dsn] if args.dsn else [])
    return preflight.main(["--datasets", *args.datasets, "--data-dir", args.data_dir, *extra])


def _tune_ef(args):
    import psycopg
    from harness.backend import ef_tuning, pgvector
    from workloads.q1 import groundtruth
    base = data.read_f32bin(Path(args.dataset_dir) / "base.f32bin")
    official = data.read_f32bin(Path(args.dataset_dir) / "official.f32bin")
    rng = np.random.default_rng(args.seed)
    sample = np.sort(rng.choice(len(official), min(args.n_queries, len(official)), replace=False))
    queries = np.asarray(official[sample], dtype=np.float32)
    _, dist = groundtruth.exact_topk_chunk(queries, base, k=10, device=args.gt_device)
    with psycopg.connect(args.dsn, autocommit=True) as conn:
        rows = ef_tuning.sweep(pgvector.PgvectorBackend(conn), queries, base, dist[:, 9])
    for row in rows:
        print(row)
    print(ef_tuning.choose_ef(rows, args.list_size))


def _answer_table(args):
    import psycopg
    from harness.backend import answer_table, pgvector
    from harness.runner.prepare import RequestVectors
    from workloads.q1.cli import load_universe
    root = Path(args.dataset_dir)
    base = data.read_f32bin(root / "base.f32bin")
    official = data.read_f32bin(root / "official.f32bin")
    vectors = RequestVectors(load_universe(root / "universe.npz"), official)
    with psycopg.connect(args.dsn, autocommit=True) as conn:
        answer_table.build_answer_table(pgvector.PgvectorBackend(conn), vectors, base,
                                        root / "answers", args.list_size, args.ef_search,
                                        shard=args.shard)


def _load_base(args):
    import psycopg
    from harness.backend import load
    base = data.read_f32bin(Path(args.dataset_dir) / "base.f32bin")
    with psycopg.connect(args.dsn) as conn:
        load.load_base(conn, base)


def _build_index(args):
    import psycopg
    from harness.backend import build_index
    base = data.read_f32bin(Path(args.dataset_dir) / "base.f32bin")
    with psycopg.connect(args.dsn) as conn:
        print(build_index.build_index(conn, "base", "embedding", args.dataset, base.shape[0],
                                      base.shape[1], workers=args.workers))


def _prepare_inputs(args):
    from harness.runner import prepare
    prepare.prepare_inputs(Path(args.data_dir), args.dataset, data.read_f32bin(args.base),
                           data.read_f32bin(args.official), no_locality=args.no_locality)


def _groundtruth(args):
    from harness.runner import prepare
    prepare.prepare_ground_truth(Path(args.data_dir) / args.dataset, args.device)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m harness")
    sub = ap.add_subparsers(dest="command", required=True)

    def grid_args(p):
        p.add_argument("--datasets", nargs="+", required=True)
        p.add_argument("--streams", nargs="+", default=None)
        p.add_argument("--seeds", type=int, nargs="+", default=[0])
        p.add_argument("--shard", type=_shard_arg, default=(0, 1), help="i/n")
        p.add_argument("--systems", nargs="+", default=None, choices=["aker", "proximity"])
        p.add_argument("--settings", nargs="+", default=None, help="paper, p0.1, p1")
        p.add_argument("--budgets", type=int, nargs="+", default=None, help="1 5 10")

    p = sub.add_parser("grid", help="list the cell ids of this shard")
    grid_args(p)
    p.set_defaults(run=_grid)

    p = sub.add_parser("run", help="run the cells of this shard that are not done yet")
    grid_args(p)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--driver-dir", required=True)
    p.add_argument("--timeout", type=float, default=None)
    p.set_defaults(run=_run)

    p = sub.add_parser("generate", help="Q1 workload generation for datasets, through answer tables")
    p.add_argument("--dataset", nargs="+", required=True, help="esci spacev sift10m sphere tiny, or all")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--dsn", default="", help="PostgreSQL connection string (database q1_<dataset> is used)")
    p.add_argument("--stages", nargs="+", default=None)
    p.add_argument("--force", default=None, help="clear this stage and all later ones first")
    p.add_argument("--device", default="auto", help="auto, numpy, cuda or mps (ground truth)")
    p.add_argument("--encode-python", default=sys.executable)
    p.add_argument("--pg-workers", type=int, default=4)
    p.add_argument("--gt-query-chunk", type=int, default=None, help="default 1024; 8192 on cuda")
    p.add_argument("--gt-base-chunk", type=int, default=None, help="default 262144; 131072 on cuda")
    p.set_defaults(run=_generate)

    p = sub.add_parser("collect", help="table (and optional CSV) of all result cells")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--csv", default=None)
    p.set_defaults(run=_collect)

    p = sub.add_parser("fidelity", help="Aker on the released SPACEV stream (report only)")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--dsn", required=True)
    p.add_argument("--driver-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--ef-search", type=int, required=True, help="the spacev tune.json choice")
    p.add_argument("--device", default="numpy")
    p.add_argument("--inspect-pickle", action="store_true")
    p.set_defaults(run=_fidelity)

    p = sub.add_parser("stages", help="list the generation stages in order")
    p.set_defaults(run=_stages)

    p = sub.add_parser("preflight", help="check the instance before a run")
    p.add_argument("--datasets", nargs="+", default=["spacev", "sift10m", "sphere", "esci"])
    p.add_argument("--data-dir", default="/data")
    p.add_argument("--require-gpu", action="store_true")
    p.add_argument("--dsn", default=None)
    p.set_defaults(run=_preflight)

    p = sub.add_parser("tune-ef", help="sweep ef_search on pgvector and apply the selection rule")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--dsn", required=True)
    p.add_argument("--n-queries", type=int, default=2000)
    p.add_argument("--list-size", type=int, default=50)
    p.add_argument("--gt-device", default="numpy")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(run=_tune_ef)

    p = sub.add_parser("answer-table", help="build the backend answer table (resumable, shardable)")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--dsn", required=True)
    p.add_argument("--list-size", type=int, default=50)
    p.add_argument("--ef-search", type=int, required=True)
    p.add_argument("--shard", type=_shard_arg, default=(0, 1))
    p.set_defaults(run=_answer_table)

    p = sub.add_parser("prepare-inputs", help="base, official, universe and streams from f32bin files")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--base", required=True, help="base.f32bin (see harness.data.write_f32bin)")
    p.add_argument("--official", required=True, help="official queries as f32bin")
    p.add_argument("--no-locality", action="store_true")
    p.set_defaults(run=_prepare_inputs)

    p = sub.add_parser("groundtruth", help="exact top-100 for every request vector (resumable)")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--device", default="numpy", help="numpy, cuda or mps")
    p.set_defaults(run=_groundtruth)

    p = sub.add_parser("load-base", help="load base.f32bin into PostgreSQL")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--dsn", required=True)
    p.set_defaults(run=_load_base)

    p = sub.add_parser("build-index", help="CREATE INDEX with sized memory; fails if the graph outgrows it")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--dataset", required=True, choices=["spacev", "sphere", "sift10m", "esci"])
    p.add_argument("--dsn", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(run=_build_index)
    return ap


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = build_parser().parse_args(argv)
    return args.run(args)
