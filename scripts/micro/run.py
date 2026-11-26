#!/usr/bin/env python3
"""Run the micro-experiments for one dataset and push each finished unit to the store under micro/.

  python -m scripts.micro.run --dataset spacev --experiments E1,E3,E4,E7,E8,E9 [--no-upload] [--store URL]
  python -m scripts.micro.run --dataset spacev --experiments intruder_check,E10,E11,E12,E13,E7

Needs the exact prepare stages (make prepare DATASET=<d> STAGES=fetch,prep; MedRAG also working_set). Workloads:
SPACEV and TriviaQA run their Zipf 0.6 stream, LAION and MedRAG their one stream. Budgets are the workload's
(MedRAG: the working-set budgets; E11: the smallest and the middle one). Results go to
results/micro/<experiment>/<workload>/<experiment>.json (+ .csv).

Work is checkpointed in units (one budget of an experiment, one degree of an index, one E9 run): each is written to
<experiment>/<workload>/units/ and pushed when it finishes, and a rerun skips finished units. progress.json in the
same folder is refreshed every 30 seconds. Inputs are checked before any heavy work. The world (the v10 cache of one
budget) is built once per budget and shared by the world-based experiments.
"""

import argparse
import json
import shutil
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from recal import matrix, store as store_mod  # noqa: E402
from recal.measure import jobs_util  # noqa: E402
from recal.micro import (common, e1, e3, e4, e7, e8, e9, e10, e11, e12, e13, intruder_check,  # noqa: E402
                         preflight, units as units_mod, world as world_mod)

ORDER = ("intruder_check", "E1", "E3", "E4", "E7", "E8", "E9", "E10", "E11", "E12", "E13")
WORLD_BASED = tuple(e for e in ORDER if e != "E9")
INDEX_UNITS = {"E7": ("degree", e7.DEGREES), "E13": ("degree", e13.DEGREES)}
DEFAULT_WINDOW = e9.WINDOW
DEFAULT_EXPERIMENTS = "E1,E3,E4,E7,E8,E9"


def parse_experiments(text: str) -> list[str]:
    known = {name.lower(): name for name in ORDER}
    names = [n.strip() for n in text.split(",") if n.strip()]
    unknown = [n for n in names if n.lower() not in known]
    if unknown:
        raise SystemExit(f"unknown experiment(s) {', '.join(unknown)}; choose from {', '.join(ORDER)}")
    chosen = {known[n.lower()] for n in names}
    return [n for n in ORDER if n in chosen]


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


def experiment_budgets(experiment: str, budgets: list[float]) -> list[float]:
    return e11.pick_budgets(budgets) if experiment == "E11" else list(budgets)


def settings_of(experiment: str, name: str, w: dict, budgets: list[float], params: dict, max_requests: int,
                window: int) -> dict:
    s = dict(experiment=experiment, workload=name, prep=w["prep"], stream=w["stream"], coord_bytes=w["coord_bytes"],
             budgets=experiment_budgets(experiment, budgets), k=10, target=world_mod.TARGET, delta=world_mod.DELTA,
             audit=world_mod.AUDIT, seed=0, history_frac=0.3, params=params, max_requests=max_requests,
             code_commit=jobs_util.code_commit())
    extra = {
        "E3": dict(bits=list(e3.BITS)),
        "E7": dict(degrees=list(e7.DEGREES), efs=list(e7.EFS), ef_construction=e7.EF_CONSTRUCTION,
                   fast_efs=list(e7.FAST_EFS), rerank_factor=e7.RERANK_FACTOR),
        "E8": dict(caps=list(e8.CAPS)),
        "E9": dict(window=window),
        "intruder_check": dict(max_queries=intruder_check.MAX_QUERIES),
        "E10": dict(bits=list(e10.BITS), max_queries=e10.MAX_QUERIES, nearest=e10.NEAREST),
        "E11": dict(shares=list(e11.SHARES), precisions=list(e11.PRECISIONS), deltas=list(e11.DELTAS),
                    max_queries=e10.MAX_QUERIES, nearest=e10.NEAREST),
        "E12": dict(earlier_share=e12.EARLIER_SHARE, nearest=e10.NEAREST),
        "E13": dict(cs=list(e13.CS), degrees=list(e13.DEGREES), efs=list(e13.EFS), max_queries=e13.MAX_QUERIES),
    }
    s.update(extra.get(experiment, {}))
    return s


@dataclass
class Context:
    """What a worker process needs to write and push the units of a workload."""
    dirs: dict
    settings: dict
    store_url: str | None
    upload: bool
    name: str

    def push(self, experiment: str):
        """A function that pushes one file of the experiment's folder to the store, or None (no upload)."""
        if not self.upload:
            return None
        s = store_mod.open_store(self.store_url)
        directory = self.dirs[experiment]

        def push(path) -> None:
            key = f"micro/{experiment}/{self.name}/{Path(path).relative_to(directory).as_posix()}"
            s.push_file(key, Path(path))
        return push

    def units(self, experiment: str) -> units_mod.Units:
        return units_mod.Units(self.dirs[experiment], experiment, self.settings[experiment], self.push(experiment))


class Scoped:
    """Units of an experiment whose keys are prefixed (the budget)."""

    def __init__(self, units: units_mod.Units, prefix: str):
        self.units, self.prefix = units, prefix

    def cached(self, key: str, fn, queries: int = 0):
        return self.units.cached(f"{self.prefix}/{key}", fn, queries)


def budget_tag(budget: float) -> str:
    return f"b{budget:g}"


def unit_keys(experiment: str, budget: float) -> list[str]:
    tag = budget_tag(budget)
    if experiment in INDEX_UNITS:
        label, values = INDEX_UNITS[experiment]
        return [f"{tag}/{label}{v}" for v in values]
    return [tag]


def e9_key(budget: float, method: str, mini: bool) -> str:
    return f"run_{budget:g}_{method}_mini-{'on' if mini else 'off'}"


def e9_units(budgets: list[float]) -> list[tuple[float, str, bool, bool]]:
    return [(b, method, lru, mini) for b in budgets for method, lru, mini in e9.variants()]


def missing_budgets(experiment: str, ctx: Context, budgets: list[float]) -> list[float]:
    return [b for b in experiment_budgets(experiment, budgets)
            if any(common.read_unit(ctx.dirs[experiment], k, ctx.settings[experiment]) is None
                   for k in unit_keys(experiment, b))]


def payload_of(experiment: str, world: world_mod.World) -> dict:
    if experiment == "E4":
        rows, diagnostics = e4.run(world)
        return dict(rows=rows, extra=dict(diagnostics=diagnostics))
    if experiment == "intruder_check":
        rows, diagnostics = intruder_check.run(world)
        return dict(rows=rows, extra=dict(diagnostics=diagnostics))
    runners = {"E1": e1.run, "E3": e3.run, "E8": e8.run, "E10": e10.run, "E11": e11.run, "E12": e12.run}
    return dict(rows=runners[experiment](world), extra={})


def world_budget(name: str, w: dict, budget: float, based: list[str], max_requests: int, params: dict,
                 world_workers: int, ctx: Context) -> None:
    """Build the cache of one budget and run the world-based experiments on it; every unit is written and pushed."""
    print(f"[{name}] building the cache at budget {budget:g}", flush=True)
    world = world_mod.build_world(w["prep"], w["stream"], budget, w["coord_bytes"], name=name,
                                  max_requests=max_requests, params=params)
    world.workers = world_workers
    n_requests = len(world.stream) - world.history_len
    for e in based:
        print(f"[{name}] {e} at budget {budget:g}", flush=True)
        units = ctx.units(e)
        if e in INDEX_UNITS:
            module = {"E7": e7, "E13": e13}[e]
            module.run(world, units=Scoped(units, budget_tag(budget)))
        else:
            units.cached(budget_tag(budget), lambda e=e: payload_of(e, world), queries=n_requests)
    print(f"[{name}] budget {budget:g} done", flush=True)


def e9_task(name, prep, stream, budget, coord_bytes, method, lru, mini, extra, window, ctx: Context) -> dict:
    print(f"[{name}] E9 at budget {budget:g}: {method}, mini cache {'on' if mini else 'off'}", flush=True)
    units = ctx.units("E9")

    def compute() -> dict:
        return e9.run_variant(name, prep, stream, budget, coord_bytes, method, lru, mini, extra, window)

    row = units.cached(e9_key(budget, method, mini), compute)
    print(f"[{name}] E9 partial: {e9_key(budget, method, mini)} hit_rate={row['hit_rate']:.4f} "
          f"recall_of_hits={row['recall_of_hits']:.4f}", flush=True)
    return row


def collect(experiment: str, ctx: Context, budgets: list[float]) -> tuple[list[dict], dict]:
    """Rows and extras of an experiment assembled from its finished units."""
    directory, settings = ctx.dirs[experiment], ctx.settings[experiment]

    def read(key: str):
        payload = common.read_unit(directory, key, settings)
        if payload is None:
            raise RuntimeError(f"unit {key} of {experiment} is missing in {directory}")
        return payload

    rows: list[dict] = []
    extra: dict = {}
    if experiment == "E9":
        rows = [read(e9_key(b, method, mini)) for b, method, _, mini in e9_units(budgets)]
        return rows, extra
    for budget in experiment_budgets(experiment, budgets):
        if experiment in INDEX_UNITS:
            budget_rows = [row for key in unit_keys(experiment, budget) for row in read(key)]
            module = {"E7": e7, "E13": e13}[experiment]
            if budget_rows:
                extra.setdefault("summaries", []).append(module.summary(ctx.name, budget, budget_rows))
            rows += budget_rows
            continue
        payload = read(budget_tag(budget))
        rows += payload["rows"]
        for key, value in payload["extra"].items():
            extra.setdefault(key, []).append(value)
    if experiment == "E1":
        extra["summary"] = e1.summarize(rows)
    if experiment == "E11":
        extra["summary"] = e11.summary(rows)
    return rows, extra


def finish(experiment: str, ctx: Context, budgets: list[float], s) -> None:
    rows, extra = collect(experiment, ctx, budgets)
    directory = ctx.dirs[experiment]
    common.write_result(directory, experiment, ctx.settings[experiment], dict(extra, rows=rows), rows)
    print(f"[{ctx.name}] {experiment}: wrote {common.result_path(directory, experiment)}", flush=True)
    if s is not None:
        n = s.push_tree(directory, f"micro/{experiment}/{ctx.name}")
        print(f"[{ctx.name}] {experiment}: pushed {n} files to micro/{experiment}/{ctx.name}", flush=True)


def total_units(experiment: str, budgets: list[float]) -> int:
    if experiment == "E9":
        return len(e9_units(budgets))
    return sum(len(unit_keys(experiment, b)) for b in experiment_budgets(experiment, budgets))


def run_workload(cfg: dict, name: str, w: dict, experiments: list[str], out_root: Path, store_url, upload: bool,
                 params: dict, max_requests: int, window: int, force: bool, jobs: int = 1, world_workers: int = 1,
                 probe: bool = True) -> None:
    budgets = budgets_for(cfg, name, w)
    settings = {e: settings_of(e, name, w, budgets, params, max_requests, window) for e in experiments}
    dirs = {e: common.out_dir(e, name, out_root) for e in experiments}
    ctx = Context(dirs, settings, store_url, upload, name)
    pending = []
    for e in experiments:
        if not force and common.is_fresh(dirs[e], e, settings[e]):
            print(f"[{name}] {e}: result with the same config hash exists, skipped")
        else:
            pending.append(e)
    if not pending:
        return
    based = [e for e in pending if e in WORLD_BASED]
    if "E7" in based and (jobs > 1 or world_workers > 1):
        raise SystemExit("E7 times single-thread searches on an idle machine: run it with --jobs 1 --world-workers 1")
    preflight.run_checks(w["prep"], w["stream"], pending, max_requests, min(budgets), w["coord_bytes"], params, probe)
    if force:
        for e in pending:
            shutil.rmtree(dirs[e] / common.UNITS_DIR, ignore_errors=True)
    s = store_mod.open_store(store_url) if upload else None
    with ExitStack() as stack:
        for e in pending:
            stack.enter_context(units_mod.Heartbeat(dirs[e], e, name, total_units(e, budgets), push=ctx.push(e)))
        todo = {e: missing_budgets(e, ctx, budgets) for e in based}
        args = [(name, w, b, [e for e in based if b in todo[e]], max_requests, params, world_workers, ctx)
                for b in budgets if any(b in todo[e] for e in based)]
        parallel_map(world_budget, args, jobs)
        for e in based:
            finish(e, ctx, budgets, s)
        if "E9" in pending:
            extra9 = e9.sim_extra(params)
            tasks = [(name, w["prep"], w["stream"], b, w["coord_bytes"], method, lru, mini, extra9, window, ctx)
                     for b, method, lru, mini in e9_units(budgets)]
            parallel_map(e9_task, tasks, jobs)
            finish("E9", ctx, budgets, s)


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
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    threads = max(1, cores // workers)
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
    ap.add_argument("--experiments", default=DEFAULT_EXPERIMENTS)
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
    ap.add_argument("--no-probe", action="store_true", help="skip the preflight build of a small cache")
    a = ap.parse_args(argv)
    cfg = json.loads(a.config.read_text())
    experiments = parse_experiments(a.experiments)
    if a.store is not None or not a.no_upload:
        store_mod.open_store(a.store)       # fails now when the store is not configured
    for name, w in workloads_for(cfg, a.dataset).items():
        run_workload(cfg, name, w, experiments, a.out, a.store, not a.no_upload, json.loads(a.params),
                     a.max_requests, a.window, a.force, a.jobs, a.world_workers, not a.no_probe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
