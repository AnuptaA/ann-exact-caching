#!/usr/bin/env python3
"""Run the micro-experiments for one dataset and push each result to the store under micro/.

  python -m scripts.micro.run --dataset spacev --experiments E1,E3,E4,E7,E8,E9 [--no-upload] [--store URL]

Needs the exact prepare stages (make prepare DATASET=<d> STAGES=fetch,prep; MedRAG also working_set). Workloads:
SPACEV and TriviaQA run their Zipf 0.6 stream, LAION and MedRAG their one stream. Budgets are the workload's
(MedRAG: the working-set budgets). Results go to results/micro/<experiment>/<workload>/<experiment>.json (+ .csv).
Worlds (the v10 cache of one budget) are built once and shared by E1, E3, E4, E7 and E8; an experiment whose
result has the same config hash is skipped, and its results are pushed once its budgets are all done.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from recal import matrix, store as store_mod  # noqa: E402
from recal.measure import jobs_util  # noqa: E402
from recal.micro import common, e1, e3, e4, e7, e8, e9, world as world_mod  # noqa: E402

ORDER = ("E1", "E3", "E4", "E7", "E8", "E9")
WORLD_BASED = ("E1", "E3", "E4", "E7", "E8")
DEFAULT_WINDOW = e9.WINDOW


def parse_experiments(text: str) -> list[str]:
    names = [n.strip().upper() for n in text.split(",") if n.strip()]
    unknown = [n for n in names if n not in ORDER]
    if unknown:
        raise SystemExit(f"unknown experiment(s) {', '.join(unknown)}; choose from {', '.join(ORDER)}")
    return [n for n in ORDER if n in names]


def workloads_for(cfg: dict, dataset: str) -> dict:
    """The workloads of a dataset the micro-experiments run: the Zipf 0.6 stream where there are several."""
    found = {n: w for n, w in cfg["workloads"].items() if w["dataset"] == dataset}
    if not found:
        raise SystemExit(f"no workload for dataset {dataset!r} in the configuration")
    if len(found) > 1:
        found = {n: w for n, w in found.items() if n.endswith("-0.6")}
    return found


def budgets_for(cfg: dict, name: str, w: dict) -> list[float]:
    if w["dataset"] == "medrag":
        medrag = matrix.load_medrag_budgets(cfg)
        if medrag is None:
            raise SystemExit("MedRAG budgets are missing: run make prepare DATASET=medrag STAGES=fetch,prep,working_set")
        return list(medrag["q1_budgets"])
    return list(w["budgets"])


def settings_of(experiment: str, name: str, w: dict, budgets: list[float], params: dict, max_requests: int,
                window: int) -> dict:
    s = dict(experiment=experiment, workload=name, prep=w["prep"], stream=w["stream"], coord_bytes=w["coord_bytes"],
             budgets=budgets, k=10, target=world_mod.TARGET, delta=world_mod.DELTA, audit=world_mod.AUDIT, seed=0,
             history_frac=0.3, params=params, max_requests=max_requests, code_commit=jobs_util.code_commit())
    if experiment == "E3":
        s["bits"] = list(e3.BITS)
    if experiment == "E7":
        s.update(degrees=list(e7.DEGREES), efs=list(e7.EFS), ef_construction=e7.EF_CONSTRUCTION)
    if experiment == "E8":
        s["caps"] = list(e8.CAPS)
    if experiment == "E9":
        s["window"] = window
    return s


def push(s, experiment: str, name: str, directory: Path) -> None:
    if s is not None:
        n = s.push_tree(directory, f"micro/{experiment}/{name}")
        print(f"[{name}] {experiment}: pushed {n} files to micro/{experiment}/{name}")


def finish(experiment, name, settings, directory, rows, extra, s):
    results = dict(extra, rows=rows)
    common.write_result(directory, experiment, settings, results, rows)
    print(f"[{name}] {experiment}: wrote {common.result_path(directory, experiment)}")
    push(s, experiment, name, directory)


def run_workload(cfg: dict, name: str, w: dict, experiments: list[str], out_root: Path, s, params: dict,
                 max_requests: int, window: int, force: bool, jobs: int = 1, world_workers: int = 1) -> None:
    budgets = budgets_for(cfg, name, w)
    settings = {e: settings_of(e, name, w, budgets, params, max_requests, window) for e in experiments}
    dirs = {e: common.out_dir(e, name, out_root) for e in experiments}
    pending = []
    for e in experiments:
        if not force and common.is_fresh(dirs[e], e, settings[e]):
            print(f"[{name}] {e}: result with the same config hash exists, skipped")
        else:
            pending.append(e)
    based = [e for e in pending if e in WORLD_BASED]
    if "E7" in based and (jobs > 1 or world_workers > 1):
        raise SystemExit("E7 times single-thread searches on an idle machine: run it with --jobs 1 --world-workers 1")
    rows = {e: [] for e in based}
    extra = {e: {} for e in based}
    args = [(name, w, budget, based, max_requests, params, world_workers) for budget in budgets] if based else []
    for budget_rows, budget_extra in parallel_map(world_budget, args, jobs):
        for e in based:
            rows[e] += budget_rows[e]
            for key, value in budget_extra[e].items():
                extra[e].setdefault(key, []).append(value)
    for e in based:
        if e == "E1":
            extra[e]["summary"] = e1.summarize(rows[e])
        finish(e, name, settings[e], dirs[e], rows[e], extra[e], s)
    if "E9" in pending:
        extra9 = e9.sim_extra(params)
        tasks = [(name, w["prep"], w["stream"], budget, w["coord_bytes"], method, lru, mini, extra9, window)
                 for budget in budgets for method, lru, mini in e9.variants()]
        out = parallel_map(e9_task, tasks, jobs, on_result=lambda row: push_partial(row, name, dirs["E9"], s))
        finish("E9", name, settings["E9"], dirs["E9"], out, {}, s)


def push_partial(row: dict, name: str, directory: Path, s) -> None:
    """Write one finished E9 run under partial/ (and push it), so results show before all runs end."""
    path = directory / "partial" / f"{row['budget']:g}_{row['method']}_mini-{'on' if row['mini_cache'] else 'off'}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=1))
    if s is not None:
        s.push_file(f"micro/E9/{name}/partial/{path.name}", path)
    print(f"[{name}] E9 partial: {path.name} hit_rate={row['hit_rate']:.4f} "
          f"recall_of_hits={row['recall_of_hits']:.4f}", flush=True)


def world_budget(name: str, w: dict, budget: float, based: list[str], max_requests: int,
                 params: dict, world_workers: int = 1) -> tuple[dict, dict]:
    """Build the cache of one budget and run the world-based experiments on it: rows and extras per experiment."""
    print(f"[{name}] building the cache at budget {budget:g}", flush=True)
    world = world_mod.build_world(w["prep"], w["stream"], budget, w["coord_bytes"], name=name,
                                  max_requests=max_requests, params=params)
    world.workers = world_workers
    rows = {}
    extra = {e: {} for e in based}
    for e in based:
        print(f"[{name}] {e} at budget {budget:g}", flush=True)
        if e == "E1":
            rows[e] = e1.run(world)
        elif e == "E3":
            rows[e] = e3.run(world)
        elif e == "E4":
            rows[e], extra[e]["diagnostics"] = e4.run(world)
        elif e == "E7":
            rows[e], extra[e]["summaries"] = e7.run(world)
        elif e == "E8":
            rows[e] = e8.run(world)
    print(f"[{name}] budget {budget:g} done", flush=True)
    return rows, extra


def e9_task(name, prep, stream, budget, coord_bytes, method, lru, mini, extra, window) -> dict:
    print(f"[{name}] E9 at budget {budget:g}: {method}, mini cache {'on' if mini else 'off'}", flush=True)
    row = e9.run_variant(name, prep, stream, budget, coord_bytes, method, lru, mini, extra, window)
    print(f"[{name}] E9 at budget {budget:g}: {method}, mini cache {'on' if mini else 'off'} done", flush=True)
    return row


def _limit_blas(threads: int) -> None:
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=threads)


def parallel_map(fn, tasks: list[tuple], jobs: int, on_result=None) -> list:
    """fn(*task) for every task, in task order; with jobs > 1 in forked worker processes.

    on_result(result) runs in this process as each task finishes (in finishing order)."""
    if jobs <= 1 or len(tasks) <= 1:
        out = []
        for t in tasks:
            out.append(fn(*t))
            if on_result is not None:
                on_result(out[-1])
        return out
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    import os
    workers = min(jobs, len(tasks))
    threads = max(1, len(os.sched_getaffinity(0)) // workers)
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork"),
                             initializer=_limit_blas, initargs=(threads,)) as pool:
        futures = [pool.submit(fn, *t) for t in tasks]
        if on_result is not None:
            from concurrent.futures import as_completed
            for future in as_completed(futures):
                on_result(future.result())
        return [future.result() for future in futures]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--experiments", default=",".join(ORDER))
    ap.add_argument("--config", type=Path, default=Path("experiments/matrix.json"))
    ap.add_argument("--out", type=Path, default=common.RESULTS_ROOT)
    ap.add_argument("--store", default=None)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--force", action="store_true", help="rerun experiments whose result is up to date")
    ap.add_argument("--max-requests", type=int, default=0, help="shorten the streams (tests)")
    ap.add_argument("--params", default="{}", help="JSON of recal.params.Params overrides (tests)")
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="E9 window of main-path requests")
    ap.add_argument("--jobs", type=int, default=1,
                    help="worker processes: budgets of the world-based experiments, E9 runs (not with E7)")
    ap.add_argument("--world-workers", type=int, default=1,
                    help="forked processes per budget for the per-query work of E1, E3, E4, E8 (not with E7)")
    a = ap.parse_args(argv)
    cfg = json.loads(a.config.read_text())
    experiments = parse_experiments(a.experiments)
    s = None if a.no_upload else store_mod.open_store(a.store)
    for name, w in workloads_for(cfg, a.dataset).items():
        run_workload(cfg, name, w, experiments, a.out, s, json.loads(a.params), a.max_requests, a.window, a.force,
                     a.jobs, a.world_workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
