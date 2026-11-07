"""Command line: python -m harness <command> ..."""

import argparse
import logging
import sys
from pathlib import Path

from harness import data, fingerprints
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
    code = fingerprints.code_fingerprints(replay.REPO, Path(args.driver_dir))
    keys: dict = {}

    def reuse_key(cell):
        key = (cell.dataset, cell.stream_seed)
        if key not in keys:
            stored = fingerprints.verify(Path(args.data_dir) / cell.dataset, cell.stream_seed)
            keys[key] = fingerprints.reuse_key(stored, code)
        return keys[key]

    todo = [c for c in cells if not results.is_done(args.out_dir, c.cell_id, commit, reuse_key(c))]
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
                                    gt_query_chunk=args.gt_query_chunk, gt_base_chunk=args.gt_base_chunk,
                                    list_size=args.list_size)
        ctx.extra.update(allow_slow=args.allow_slow, driver_dir=args.driver_dir)
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




def _status(args):
    from harness import status
    print(status.format_status(Path(args.data_dir)))


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
    p.add_argument("--list-size", type=int, default=200, help="neighbors stored per query (L)")
    p.add_argument("--allow-slow", action="store_true", help="accept ground truth on the numpy device")
    p.add_argument("--driver-dir", default="build/replay")
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


    p = sub.add_parser("status", help="background runs and generation progress")
    p.add_argument("--data-dir", required=True)
    p.set_defaults(run=_status)

    p = sub.add_parser("answer-table", help="build the backend answer table (resumable, shardable)")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--dsn", required=True)
    p.add_argument("--list-size", type=int, required=True)
    p.add_argument("--ef-search", type=int, required=True)
    p.add_argument("--shard", type=_shard_arg, default=(0, 1))
    p.set_defaults(run=_answer_table)
    return ap


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = build_parser().parse_args(argv)
    return args.run(args)
