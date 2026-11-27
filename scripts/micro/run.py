#!/usr/bin/env python3
"""Run the micro-experiments for one dataset and push each finished unit to the store under micro/.

  python -m scripts.micro.run --dataset spacev --experiments E1,E3,E4,E7,E8,E9 [--workers 16] [--no-upload]
  python -m scripts.micro.run --dataset spacev --tier quick --experiments intruder_check,E1,E3,E4,E10,E11,E13

Needs the exact prepare stages (make prepare DATASET=<d> STAGES=fetch,prep; MedRAG also working_set). Workloads:
SPACEV and TriviaQA run their Zipf 0.6 stream, LAION and MedRAG their one stream. Budgets are the workload's
(MedRAG: the working-set budgets; E11: the smallest and the middle one; E14: the middle one). Results go to
results/micro/<experiment>/<workload>/<experiment>.json (+ .csv); the quick tier writes to <workload>-quick.

Work is checkpointed in units (one budget of an experiment, one index degree, one E9 run): each is written to
<experiment>/<workload>/units/ and pushed when it finishes, and a rerun skips finished units. A unit is reused only
when the hash of the code the experiment uses, of the prepared inputs and of the libraries is unchanged.
progress.json in the same folder is refreshed every 30 seconds. Inputs are checked before any heavy work, and
every built cache is compared with the core's own definitions (recal/micro/golden.py) before an experiment uses it.

--workers N: budgets are built in parallel worker processes and the per-query work of each is split across forked
processes (results are identical to --workers 1). E7 times single-thread searches and needs --workers 1.
Quick tier: the middle budget only, a seeded sample of 2,000 held-out requests, reduced E11 and E13 grids.
"""

import argparse
import hashlib
import json
import shutil
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from recal import matrix, store as store_mod  # noqa: E402
from recal.micro import (common, e1, e3, e4, e7, e8, e9, e10, e11, e12, e13, e14, golden, intruder_check,  # noqa: E402
                         preflight, provenance, units as units_mod, world as world_mod)

ORDER = ("intruder_check", "E1", "E3", "E4", "E7", "E8", "E9", "E10", "E11", "E12", "E13", "E14")
WORLD_BASED = tuple(e for e in ORDER if e != "E9")
INDEX_UNITS = {"E7": ("degree", e7.DEGREES), "E13": ("degree", e13.DEGREES)}
SPEND_NO_M = ("E1", "E3", "E4", "E10", "E12", "intruder_check")      # results that do not depend on M, only on the cache
DEFAULT_WINDOW = e9.WINDOW
DEFAULT_EXPERIMENTS = "E1,E3,E4,E7,E8,E9"
QUICK_EXPERIMENTS = ("intruder_check", "E1", "E3", "E4", "E10", "E11", "E13")
QUICK_SAMPLE_REQUESTS = 2000
QUICK_E11_REQUESTS = 35_000
QUICK_E11_GRID = dict(shares=(0.01, 0.03), precisions=(8, 16), deltas=(0.05,))
GOLDEN = "golden"


def parse_experiments(text: str) -> list[str]:
    known = {name.lower(): name for name in ORDER}
    names = [n.strip() for n in text.split(",") if n.strip()]
    unknown = [n for n in names if n.lower() not in known]
    if unknown:
        raise SystemExit(f"unknown experiment(s) {', '.join(unknown)}; choose from {', '.join(ORDER)}")
    chosen = {known[n.lower()] for n in names}
    return [n for n in ORDER if n in chosen]


def check_tier(experiments: list[str], tier: str) -> None:
    if tier not in ("full", "quick"):
        raise SystemExit(f"unknown tier {tier!r}: choose full or quick")
    if tier == "quick":
        extra = [e for e in experiments if e not in QUICK_EXPERIMENTS]
        if extra:
            raise SystemExit(f"{', '.join(extra)} is not part of the quick tier ({', '.join(QUICK_EXPERIMENTS)}); "
                             "run it with --tier full")


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


def experiment_budgets(experiment: str, budgets: list[float], tier: str = "full") -> list[float]:
    ordered = sorted(budgets)
    if tier == "quick":
        return [ordered[len(ordered) // 2]] if ordered else []
    if experiment == "E14":
        return [ordered[len(ordered) // 2]] if ordered else []
    return e11.pick_budgets(budgets) if experiment == "E11" else list(budgets)


def settings_of(experiment: str, name: str, w: dict, budgets: list[float], params: dict, max_requests: int,
                window: int, tier: str = "full") -> dict:
    s = dict(experiment=experiment, workload=name, prep=w["prep"], stream=w["stream"], coord_bytes=w["coord_bytes"],
             budgets=experiment_budgets(experiment, budgets, tier), k=10, target=world_mod.TARGET,
             delta=world_mod.DELTA, audit=world_mod.AUDIT, seed=0, history_frac=0.3, params=params,
             max_requests=max_requests, tier=tier, code_hash=provenance.code_hash(experiment),
             prep_hash=provenance.prep_hash(Path(w["prep"])), libraries=provenance.library_versions())
    if tier == "quick":
        s["sample_requests"] = QUICK_SAMPLE_REQUESTS
    quick = tier == "quick"
    extra = {
        "E3": dict(bits=list(e3.BITS)),
        "E7": dict(degrees=list(e7.DEGREES), efs=list(e7.EFS), ef_construction=e7.EF_CONSTRUCTION,
                   fast_efs=list(e7.FAST_EFS), rerank_factor=e7.RERANK_FACTOR),
        "E8": dict(caps=list(e8.CAPS)),
        "E9": dict(window=window),
        "intruder_check": dict(max_queries=intruder_check.MAX_QUERIES),
        "E10": dict(bits=list(e10.BITS), max_queries=e10.MAX_QUERIES, nearest=e10.NEAREST),
        "E11": dict(grid=QUICK_E11_GRID if quick else dict(shares=list(e11.SHARES), precisions=list(e11.PRECISIONS),
                                                            deltas=list(e11.DELTAS)),
                    request_limit=QUICK_E11_REQUESTS if quick else 0, nearest=e10.NEAREST),
        "E12": dict(earlier_share=e12.EARLIER_SHARE, nearest=e10.NEAREST),
        "E14": dict(seeds=e14.SEEDS, shifts=list(e14.SHIFTS), p_values=list(e14.P_VALUES), margins=list(e14.MARGINS),
                    coarse_bits=list(e14.COARSE_BITS), precise_bits=list(e14.PRECISE_BITS), primary=e14.PRIMARY,
                    ess_target=e14.ESS_TARGET, stride_parts=e14.STRIDE_PARTS, nearest=e10.NEAREST),
        "E13": dict(grid=e13.QUICK_GRID if quick else dict(cs=list(e13.CS), degrees=list(e13.DEGREES),
                                                          efs=list(e13.EFS)), max_queries=e13.MAX_QUERIES),
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
    name: str                      # folder and store key (the quick tier adds -quick)
    world_name: str = ""
    tier: str = "full"
    window: int = DEFAULT_WINDOW

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


def index_keys(experiment: str, tier: str) -> list[str]:
    if experiment == "E7":
        return [f"degree{d}" for d in e7.DEGREES]
    grid = e13.QUICK_GRID if tier == "quick" else dict(degrees=e13.DEGREES)
    return ["core_m"] + [f"degree{d}" for d in grid["degrees"]]


def unit_keys(experiment: str, budget: float, tier: str = "full") -> list[str]:
    tag = budget_tag(budget)
    if experiment in INDEX_UNITS:
        return [f"{tag}/{key}" for key in index_keys(experiment, tier)]
    return [tag]


def e9_key(budget: float, method: str, mini: bool) -> str:
    return f"run_{budget:g}_{method}_mini-{'on' if mini else 'off'}"


def e9_units(budgets: list[float]) -> list[tuple[float, str, bool, bool]]:
    return [(b, method, lru, mini) for b in budgets for method, lru, mini in e9.variants()]


def missing_budgets(experiment: str, ctx: Context, budgets: list[float]) -> list[float]:
    return [b for b in experiment_budgets(experiment, budgets, ctx.tier)
            if any(common.read_unit(ctx.dirs[experiment], k, ctx.settings[experiment]) is None
                   for k in unit_keys(experiment, b, ctx.tier))]


def cache_signature(world: world_mod.World) -> str:
    """Identifies the built cache (entries, pool, model, m, lambda): equal for budgets that build the same cache."""
    h = hashlib.sha256()
    built = world.built
    h.update(np.ascontiguousarray(built.centers).tobytes())
    h.update(np.asarray(built.pool_ids, dtype=np.int64).tobytes())
    h.update(np.asarray([e["radius"] for e in world.entries()], dtype=np.float64).tobytes())
    h.update(f"{built.n_entries}|{built.m}|{built.lam}".encode())
    return h.hexdigest()[:16]


def rebudget(value, old: float, new: float):
    """Copy of a payload with every 'budget' field equal to old replaced by new."""
    if isinstance(value, dict):
        return {k: (new if k == "budget" and v == old else rebudget(v, old, new)) for k, v in value.items()}
    if isinstance(value, list):
        return [rebudget(v, old, new) for v in value]
    return value


def identical_earlier(ctx: Context, experiment: str, signature: str, budget: float):
    """Payload of a smaller budget whose cache is identical (same signature), else None."""
    for path in sorted((ctx.dirs[experiment] / common.UNITS_DIR).glob("b*.json")):
        try:
            body = json.loads(path.read_text())
            payload = body["payload"]
            old = float(body["unit"][1:])
        except (OSError, ValueError, KeyError):
            continue
        if (isinstance(payload, dict) and payload.get("signature") == signature and old < budget
                and common.read_unit(ctx.dirs[experiment], body["unit"], ctx.settings[experiment]) is not None):
            return dict(rebudget(payload, old, budget), reused_from_budget=old)
    return None


def payload_of(experiment: str, world: world_mod.World, ctx: Context) -> dict:
    extra: dict = {}
    if experiment in ("E4", "intruder_check"):
        module = e4 if experiment == "E4" else intruder_check
        rows, diagnostics = module.run(world)
        extra["diagnostics"] = diagnostics
    elif experiment == "E14":
        rows, extra["summary"] = e14.run(world)
    elif experiment == "E11":
        grid = QUICK_E11_GRID if ctx.tier == "quick" else {}
        rows = e11.run(world, request_limit=QUICK_E11_REQUESTS if ctx.tier == "quick" else 0, **grid)
    else:
        rows = dict(E1=e1.run, E3=e3.run, E8=e8.run, E10=e10.run, E12=e12.run)[experiment](world)
    for row in rows:
        row.setdefault("tier", ctx.tier)
    return dict(rows=rows, extra=extra, signature=cache_signature(world))


def world_budget(name: str, w: dict, budget: float, based: list[str], max_requests: int, params: dict,
                 world_workers: int, ctx: Context) -> None:
    """Build the cache of one budget and run the world-based experiments on it; every unit is written and pushed."""
    print(f"[{ctx.name}] building the cache at budget {budget:g}", flush=True)
    world = world_mod.build_world(w["prep"], w["stream"], budget, w["coord_bytes"], name=name,
                                  max_requests=max_requests, params=params)
    world.workers = world_workers
    if ctx.tier == "quick":
        world.sample_requests = QUICK_SAMPLE_REQUESTS
    signature = cache_signature(world)
    ctx.units(GOLDEN).cached(budget_tag(budget), lambda: golden.check(world, strict=True))
    n_requests = len(world.heldout_stream())
    for e in based:
        print(f"[{ctx.name}] {e} at budget {budget:g}", flush=True)
        units = ctx.units(e)
        if e in INDEX_UNITS:
            scoped = Scoped(units, budget_tag(budget))
            if e == "E7":
                e7.run(world, units=scoped)
            else:
                e13.run(world, units=scoped, grid=e13.QUICK_GRID if ctx.tier == "quick" else None)
            continue

        def compute(e=e):
            earlier = identical_earlier(ctx, e, signature, budget) if e in SPEND_NO_M else None
            if earlier is not None:
                print(f"[{ctx.name}] {e} at budget {budget:g}: same cache as budget {earlier['reused_from_budget']:g}, "
                      "reusing its result", flush=True)
                return earlier
            return payload_of(e, world, ctx)
        units.cached(budget_tag(budget), compute, queries=n_requests)
    print(f"[{ctx.name}] budget {budget:g} done", flush=True)


def e9_task(name, prep, stream, budget, coord_bytes, method, lru, mini, extra, window, ctx: Context) -> dict:
    print(f"[{ctx.name}] E9 at budget {budget:g}: {method}, mini cache {'on' if mini else 'off'}", flush=True)
    units = ctx.units("E9")

    def compute() -> dict:
        return e9.run_variant(name, prep, stream, budget, coord_bytes, method, lru, mini, extra, window)

    row = units.cached(e9_key(budget, method, mini), compute)
    print(f"[{ctx.name}] E9 partial: {e9_key(budget, method, mini)} hit_rate={row['hit_rate']:.4f} "
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
        return [read(e9_key(b, method, mini)) for b, method, _, mini in e9_units(budgets)], extra
    for budget in experiment_budgets(experiment, budgets, ctx.tier):
        if experiment in INDEX_UNITS:
            budget_rows = [row for key in unit_keys(experiment, budget, ctx.tier) for row in read(key)]
            for row in budget_rows:
                row.setdefault("tier", ctx.tier)
            module = {"E7": e7, "E13": e13}[experiment]
            if budget_rows:
                extra.setdefault("summaries", []).append(module.summary(ctx.world_name, budget, budget_rows))
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


def finish(experiment: str, ctx: Context, budgets: list[float], s, runtime: dict) -> None:
    rows, extra = collect(experiment, ctx, budgets)
    directory = ctx.dirs[experiment]
    common.write_result(directory, experiment, ctx.settings[experiment], dict(extra, rows=rows), rows, runtime)
    print(f"[{ctx.name}] {experiment}: wrote {common.result_path(directory, experiment)}", flush=True)
    if s is not None:
        n = s.push_tree(directory, f"micro/{experiment}/{ctx.name}")
        print(f"[{ctx.name}] {experiment}: pushed {n} files to micro/{experiment}/{ctx.name}", flush=True)


def total_units(experiment: str, budgets: list[float], tier: str = "full") -> int:
    if experiment == "E9":
        return len(e9_units(budgets))
    return sum(len(unit_keys(experiment, b, tier)) for b in experiment_budgets(experiment, budgets, tier))


def split_workers(workers: int, n_tasks: int, jobs: int | None, world_workers: int | None) -> tuple[int, int]:
    """Processes over tasks (budgets, E9 runs) and forked processes per task; --jobs and --world-workers override."""
    n = max(1, n_tasks)
    jobs_eff = jobs if jobs is not None else max(1, min(workers, n))
    per_task = world_workers if world_workers is not None else max(1, workers // jobs_eff)
    return jobs_eff, per_task


def run_workload(cfg: dict, name: str, w: dict, experiments: list[str], out_root: Path, store_url, upload: bool,
                 params: dict, max_requests: int, window: int, force: bool, workers: int = 1, probe: bool = True,
                 tier: str = "full", jobs: int | None = None, world_workers: int | None = None) -> None:
    budgets = budgets_for(cfg, name, w)
    folder = f"{name}-quick" if tier == "quick" else name
    names = list(experiments) + [GOLDEN]
    settings = {e: settings_of(e, name, w, budgets, params, max_requests, window, tier) for e in experiments}
    settings[GOLDEN] = dict(settings[experiments[0]], experiment=GOLDEN, code_hash=provenance.code_hash(GOLDEN))
    dirs = {e: common.out_dir(e, folder, out_root) for e in names}
    ctx = Context(dirs, settings, store_url, upload, folder, name, tier, window)
    pending = []
    for e in experiments:
        if not force and common.is_fresh(dirs[e], e, settings[e]):
            print(f"[{folder}] {e}: result with the same config hash exists, skipped")
        else:
            pending.append(e)
    if not pending:
        return
    based = [e for e in pending if e in WORLD_BASED]
    n_budgets = len(experiment_budgets("E1", budgets, tier))
    jobs_b, workers_b = split_workers(workers, n_budgets, jobs, world_workers)
    if "E7" in based and (jobs_b > 1 or workers_b > 1):
        raise SystemExit("E7 times single-thread searches on an idle machine: run it with --workers 1")
    preflight.run_checks(w["prep"], w["stream"], pending, max_requests, min(budgets), w["coord_bytes"], params, probe)
    if force:
        for e in pending + [GOLDEN]:
            shutil.rmtree(dirs[e] / common.UNITS_DIR, ignore_errors=True)
    s = store_mod.open_store(store_url) if upload else None
    runtime = provenance.runtime(workers, jobs_b, workers_b)
    with ExitStack() as stack:
        for e in pending:
            stack.enter_context(units_mod.Heartbeat(dirs[e], e, folder, total_units(e, budgets, tier),
                                                    push=ctx.push(e)))
        todo = {e: missing_budgets(e, ctx, budgets) for e in based}
        args = [(name, w, b, [e for e in based if b in todo[e]], max_requests, params, workers_b, ctx)
                for b in experiment_budgets("E1", budgets, tier) if any(b in todo[e] for e in based)]
        parallel_map(world_budget, args, min(jobs_b, max(len(args), 1)))
        for e in based:
            finish(e, ctx, budgets, s, runtime)
        if "E9" in pending:
            extra9 = e9.sim_extra(params)
            tasks = [(name, w["prep"], w["stream"], b, w["coord_bytes"], method, lru, mini, extra9, window, ctx)
                     for b, method, lru, mini in e9_units(budgets)]
            jobs_9, _ = split_workers(workers, len(tasks), jobs, None)
            parallel_map(e9_task, tasks, jobs_9)
            finish("E9", ctx, budgets, s, runtime)


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


def time_features(cfg: dict, name: str, w: dict, experiments: list[str], params: dict, tier: str, workers: int) -> None:
    """Time a probe cache and the core's feature pass on a few queries and project the E1-style feature pass.

    It times the cache build and the core's features only, not each experiment: E4, E10, E11, E13 and E9 cost more per
    query. Use it to see the build time and the per-query feature cost before a long run."""
    import time
    budgets = budgets_for(cfg, name, w)
    preflight.run_checks(w["prep"], w["stream"], experiments, 0, min(budgets), w["coord_bytes"], params, probe=False)
    t0 = time.perf_counter()
    world = world_mod.build_world(w["prep"], w["stream"], min(budgets), w["coord_bytes"], name=name,
                                  max_requests=preflight.PROBE_REQUESTS, params=params)
    build_s = time.perf_counter() - t0
    held = world.heldout()
    Q, truth = held["vec"][:200], held["truth"][:200]
    t0 = time.perf_counter()
    world.features(Q, truth)
    per_query = (time.perf_counter() - t0) / max(len(Q), 1)
    n_unique = len(np.unique(np.load(Path(w["prep"]) / "streams.npz")[w["stream"]][-int(0.7 * 100_000):]))
    n_unique = min(n_unique, QUICK_SAMPLE_REQUESTS) if tier == "quick" else n_unique
    n_budgets = len(experiment_budgets("E1", budgets, tier))
    print(f"[time] probe cache ({preflight.PROBE_REQUESTS} requests) built in {build_s:.0f} s; core features "
          f"{per_query * 1000:.1f} ms per query; about {n_unique} unique held-out queries per budget, {n_budgets} budget(s)")
    print(f"[time] core feature pass alone: about {per_query * n_unique / 60 / max(workers, 1):.0f} min per budget with "
          f"{workers} worker(s); the full-history cache build is longer than the probe's (it grows with the history) "
          "and is not parallel within a budget. Only the build and the core's features were timed, not the "
          f"experiments ({', '.join(experiments)}); this is an estimate.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--experiments", default=DEFAULT_EXPERIMENTS)
    ap.add_argument("--tier", default="full", choices=("full", "quick"))
    ap.add_argument("--config", type=Path, default=Path("experiments/matrix.json"))
    ap.add_argument("--out", type=Path, default=common.RESULTS_ROOT)
    ap.add_argument("--store", default=None)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--force", action="store_true", help="rerun experiments whose result is up to date")
    ap.add_argument("--max-requests", type=int, default=0, help="shorten the streams (tests)")
    ap.add_argument("--params", default="{}", help="JSON of recal.params.Params overrides (tests)")
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="E9 window of main-path requests")
    ap.add_argument("--workers", type=int, default=1,
                    help="worker processes for budgets and per-query work (E7 needs 1)")
    ap.add_argument("--jobs", type=int, default=None, help="override: processes over budgets and E9 runs")
    ap.add_argument("--world-workers", type=int, default=None, help="override: forked processes per budget")
    ap.add_argument("--no-probe", action="store_true", help="skip the preflight build of a small cache")
    ap.add_argument("--time-features", action="store_true",
                    help="time a probe cache build and the core's feature pass, print the projection, then stop")
    a = ap.parse_args(argv)
    cfg = json.loads(a.config.read_text())
    experiments = parse_experiments(a.experiments)
    check_tier(experiments, a.tier)
    if a.time_features:
        for name, w in workloads_for(cfg, a.dataset).items():
            time_features(cfg, name, w, experiments, json.loads(a.params), a.tier, a.workers)
        return 0
    if a.store is not None or not a.no_upload:
        store_mod.open_store(a.store)       # fails now when the store is not configured
    for name, w in workloads_for(cfg, a.dataset).items():
        run_workload(cfg, name, w, experiments, a.out, a.store, not a.no_upload, json.loads(a.params),
                     a.max_requests, a.window, a.force, a.workers, not a.no_probe, a.tier, a.jobs, a.world_workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
