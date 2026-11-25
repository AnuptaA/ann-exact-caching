"""The whole experiment matrix (Q1 to Q4) as jobs, from one configuration (the final design).

Each job line skips itself when its result exists and carries a marker run_jobs.py recognizes. The configuration
names the workloads (dataset, prep folder, stream, stored precision, budgets, DiskANN files); see
experiments/matrix.json. MedRAG's budgets come only from medrag_budgets.json (scripts/medrag_working_set.py);
without it no MedRAG cell is emitted.

Q1 384 runs, Q2 56, Q3 128 split points (hit-rate run plus a throughput arm) and 32 thread lines, Q4 93.
"""

import json
from collections import OrderedDict
from pathlib import Path

from recal import modes
from recal.jobspec import Job

# Absolute, so job lines run from any working directory (results stay relative to it).
REPO = Path(__file__).resolve().parents[1]
PY = f"{REPO}/.venv-core/bin/python"
ANN = f"{REPO}/.venv-ann/bin/python"
PYPATH = f"PYTHONPATH={REPO}"
ABLATIONS = ("--no-union", "--no-expansion", "--no-shells", "--lru", "--no-calibration", "--no-recalibration",
             "--no-mini-cache")
ROOT = "results"
STREAM_LENGTH = 100_000
REPLAY = f"{REPO}/simulations/approximate_hits/throughput_replay.py"
BEST_SPLIT = "recal.system_benefit.best_split"


# ---- configuration and workloads -------------------------------------------------------------------------------

def load_medrag_budgets(cfg: dict) -> dict | None:
    path = cfg.get("medrag_budgets_file")
    if path and Path(path).exists():
        return json.loads(Path(path).read_text())
    return None


def plan(cfg: dict, medrag: dict | None = None) -> tuple[OrderedDict, list[str]]:
    """workload -> {w, budgets, standin} for every workload that has its budgets, and the ones left out."""
    medrag = load_medrag_budgets(cfg) if medrag is None else medrag
    out, skipped = OrderedDict(), []
    for name, w in cfg["workloads"].items():
        if w["dataset"] == "medrag":
            if medrag is None:
                skipped.append(name)
                continue
            out[name] = {"w": w, "budgets": list(medrag["q1_budgets"]), "standin": float(medrag["standin_5pct"])}
        else:
            out[name] = {"w": w, "budgets": list(w["budgets"]), "standin": float(w["standin"])}
    return out, skipped


def selected(cfg: dict, wl: OrderedDict, question: str) -> OrderedDict:
    """The workloads of one question: its optional "workloads" list in the configuration, else all."""
    names = cfg[question].get("workloads")
    return wl if names is None else OrderedDict((n, e) for n, e in wl.items() if n in names)


def default_deviation(w: dict) -> str:
    """The paper's deviation factor (0.25 SPACEV, 0.075 otherwise), written explicitly in every QVCache cell."""
    from recal.qvcache_driver import default_deviation as paper_default
    return f"--deviation-factor {paper_default(w['coord_bytes']):g}"


def backend_flags(w: dict, backend: str) -> str:
    if backend == "exact":
        return "--backend exact"
    return f"--backend file --backend-file {w['diskann_file']} --backend-target 0.95"


def needs_of(w: dict, backend: str = "diskann", extra=()) -> tuple:
    ds = w["dataset"]
    names = [f"{ds}/prep"] + ([f"{ds}/diskann_answers"] if backend == "diskann" else []) + [f"{ds}/{e}" for e in extra]
    return tuple(names)


def cell(d: str, stem: str, method: str, w: dict, budget, flags: str, prep: str | None = None,
         stream: str | None = None, clone: str | None = None) -> str:
    """One job line. With clone=<job id> the command runs between the creation and the destruction of that job's
    pgvector clone (scripts/pg_clones.sh run; the command sees $PG_CLONES)."""
    b = budget if isinstance(budget, str) else f"{budget:g}"
    command = (f"{PYPATH} {PY} -u -m recal.cells --method {method} "
               f"--prep {prep or w['prep']} --stream {stream or w['stream']} --budget {b} "
               f"--coord-bytes {w['coord_bytes']:g} {flags} --out {d}")
    if clone is not None:
        command = f"bash scripts/pg_clones.sh run {clone} -- '{command}'"
    return f"[ -f {d}/{stem}.json ] || {{ mkdir -p {d} && {command} > {d}/log.txt 2>&1; }}  # recal.cells"


def hit_job(exp: str, name: str, w: dict, label: str, stem: str, method: str, budget, backend: str, flags: str,
            system: str, seed: int, **kw) -> Job:
    d = f"{ROOT}/{exp}/{name}/{label}"
    line = cell(d, stem, method, w, budget, f"{backend_flags(w, backend)} {flags} --seed {seed}", **kw)
    return Job(line, exp, name, system, "hit", d, needs_of(w, backend, kw.pop("extra_needs", ())))


# ---- Q1, Q2 ----------------------------------------------------------------------------------------------------

def q1_systems(cfg: dict, w: dict) -> list[tuple]:
    """(label, system, method, stem, backend, flags) of the system cells of one (workload, budget, seed): 16."""
    out = []
    for t in cfg["q1"]["targets"]:
        for mode, flag in (("deployable", ""), ("idealized", "--idealized")):
            out.append((f"recal_{mode}_t{t:g}", f"recal_{mode}", "recal", "recal", modes.recal_backend(mode),
                        f"--target {t:g} {flag} --strict-modes"))
    for backend in cfg["q1"]["backends"]:
        out.append((f"aker_{backend}", "aker", "aker", "aker", backend, ""))
        out.append((f"proximity-a_{backend}", "proximity-a", "proximity", "proximity_p0.1", backend,
                    "--prox-percentile 0.1 --tag _p0.1"))
        out.append((f"proximity-b_{backend}", "proximity-b", "proximity", "proximity_p1", backend,
                    "--prox-percentile 1 --tag _p1"))
        out.append((f"qvcache_{backend}", "qvcache", "qvcache", "qvcache", backend, default_deviation(w)))
    return out


def q1_jobs(cfg: dict, wl: OrderedDict) -> list[Job]:
    out = []
    for name, entry in selected(cfg, wl, "q1").items():
        w = entry["w"]
        for budget in entry["budgets"]:
            for seed in cfg["seeds"]:
                for label, system, method, stem, backend, flags in q1_systems(cfg, w):
                    out.append(hit_job("q1", name, w, f"{label}_b{budget:g}_s{seed}", stem, method, budget, backend,
                                       flags, system, seed))
    return out


def q2_jobs(cfg: dict, wl: OrderedDict) -> list[Job]:
    out = []
    for name, entry in selected(cfg, wl, "q2").items():
        w = entry["w"]
        for flag in ABLATIONS:
            for seed in cfg["seeds"]:
                label = flag.lstrip("-")
                out.append(hit_job("q2", name, w, f"{label}_s{seed}", "recal", "recal", entry["standin"],
                                   modes.recal_backend("deployable"),
                                   f"--target {cfg['q2']['target']:g} {flag} --strict-modes", f"recal_{label}", seed))
    return out


# ---- Q3 --------------------------------------------------------------------------------------------------------

def node_cache_nodes(w: dict, budget: float, node_share: int) -> int:
    """Nodes of DiskANN's cache at node_share percent of the budget, at stored precision."""
    from recal.accounting import budget_bytes
    from recal.system_benefit.nodecache import nodes_for_bytes
    total = budget_bytes(w["n_vectors"], w["dim"], budget, w["coord_bytes"])
    return nodes_for_bytes(total * node_share / 100.0, w["dim"], w["coord_bytes"], w["degree"])


Q3_SYSTEMS = {
    "recal": ("recal", "recal", "recal", "--target {t:g} --strict-modes"),
    "aker": ("aker", "aker", "aker", ""),
    "qvcache": ("qvcache", "qvcache", "qvcache", "{dev}"),
}


def q3_hit_jobs(cfg: dict, wl: OrderedDict) -> list[Job]:
    """The hit-rate run of every split point: the result cache at its share of the 5% budget (share 100 is Q1's run)."""
    q3, out = cfg["q3"], []
    for name, entry in selected(cfg, wl, "q3").items():
        w = entry["w"]
        for key in q3["systems"]:
            system, method, stem, flags = Q3_SYSTEMS[key]
            for ns in q3["node_shares"]:
                budget = entry["standin"] * (100 - ns) / 100.0
                f = flags.format(t=q3["target"], dev=default_deviation(w))
                out.append(hit_job("q3", name, w, f"{key}_n{ns}", stem, method, budget, "diskann", f, system, 0))
    return out


def arm_spec(name: str, path: str, nodes) -> str:
    return f"{name}={path},nodes={nodes}"


def q3_followups(cfg: dict, wl: OrderedDict, hits: list[Job], owner_out) -> list[Job]:
    """Split sweep replay, best split and thread lines per workload; they read the owners' result directories."""
    q3, out = cfg["q3"], []
    by_cell = {j.cell: j for j in hits}
    sat, reps = q3["saturation_clients"], q3["repetitions"]
    for name, entry in selected(cfg, wl, "q3").items():
        w, budget = entry["w"], entry["standin"]
        root = f"{ROOT}/q3/{name}"
        rw = w["replay_workload"]
        arms = [f"--arm nodeonly=none,nodes={node_cache_nodes(w, budget, 100)}"]
        arm_args = []
        for key in q3["systems"]:
            stem = Q3_SYSTEMS[key][2]
            for ns in q3["node_shares"]:
                job = by_cell[f"{ROOT}/q3/{name}/{key}_n{ns}"]
                path = f"{owner_out(job)}/{stem}.json"
                spec = arm_spec(f"{key}_n{ns}", path, node_cache_nodes(w, budget, ns))
                arms.append(f"--arm {spec}")
                arm_args.append(f"--arm {spec}")
        needs = (f"{w['dataset']}/diskann_index", f"{w['dataset']}/prep")
        run_dirs = tuple(owner_out(by_cell[f"{ROOT}/q3/{name}/{key}_n{ns}"]) + "/" for key in q3["systems"]
                         for ns in q3["node_shares"])
        sweep = (f"[ -f {root}/sweep.json ] || {{ mkdir -p {root} && {PYPATH} {ANN} -u {REPLAY} --workload {rw} "
                 f"{' '.join(arms)} --clients {sat} --saturation-clients {sat} --repetitions {reps} --io-complete --out {root}/sweep.json "
                 f"> {root}/log_sweep.txt 2>&1; }}  # throughput_replay.py")
        best = (f"[ -f {root}/best.json ] || {{ {PYPATH} {PY} -u -m {BEST_SPLIT} run --sweep {root}/sweep.json "
                f"--clients {sat} {' '.join(arm_args)} --out {root}/best.json > {root}/log_best.txt 2>&1; }}  "
                f"# {BEST_SPLIT}")
        get = lambda s: f"$({PYPATH} {PY} -m {BEST_SPLIT} get {root}/best.json {s}.spec)"  # noqa: E731
        clients = " ".join(map(str, q3["clients"]))
        thread_arms = " ".join([f"--arm nodeonly=none,nodes={node_cache_nodes(w, budget, 100)}"]
                               + [f"--arm {get(k)}" for k in q3["systems"]])
        threads = (f"[ -f {root}/threads.json ] || {{ {PYPATH} {ANN} -u {REPLAY} --workload {rw} {thread_arms} "
                   f"--clients {clients} --repetitions {reps} --io-complete --out {root}/threads.json "
                   f"> {root}/log_threads.txt 2>&1; }}  # throughput_replay.py")
        out += [Job(sweep, "q3", name, "split_sweep", "replay", f"{root}/sweep", needs,
                    (f"{root}/sweep.json", f"{root}/log_sweep.txt"), run_dirs),
                Job(best, "q3", name, "best_split", "best_split", f"{root}/best", (),
                    (f"{root}/best.json", f"{root}/log_best.txt"), (f"{root}/sweep.json",)),
                Job(threads, "q3", name, "thread_lines", "threads", f"{root}/threads", needs,
                    (f"{root}/threads.json", f"{root}/log_threads.txt"), run_dirs + (f"{root}/best.json",))]
    return out


# ---- Q4 --------------------------------------------------------------------------------------------------------

def deployable_flags(cfg: dict) -> str:
    return f"--target {cfg['q2']['target']:g} --strict-modes"


def q4_jobs(cfg: dict, wl: OrderedDict) -> list[Job]:
    q4, out = cfg["q4"], []
    flags = deployable_flags(cfg)
    dep = modes.recal_backend("deployable")
    for name in q4["grid_workloads"]:
        if name not in wl:
            continue
        w = wl[name]["w"]
        for b in q4["grid_budgets"]:
            out.append(hit_job("q4", name, w, f"budget_b{b:g}_s0", "recal", "recal", b, dep, flags, "recal_grid", 0))
        for h in q4["histories"]:
            frac = h / STREAM_LENGTH
            out.append(hit_job("q4", name, w, f"history_h{h}_s0", "recal", "recal", 0.05, dep,
                               f"{flags} --history {h} --history-frac {frac:g}", "recal_history", 0))
    for name in q4["shift_workloads"]:
        if name not in wl:
            continue
        w = wl[name]["w"]
        for kind in ("abrupt", "gradual", "correlated"):
            for on in (True, False):
                tag = "signals" if on else "periodic"
                for seed in cfg["shift_seeds"]:
                    prep = f"{q4['shift_prep_dir']}/{name}_{kind}_seed{seed}"
                    extra = "" if on else " --no-rebuild-signals"
                    d = f"{ROOT}/q4/{name}/shift_{kind}_{tag}_s{seed}"
                    line = cell(d, "recal", "recal", w, 0.05, f"{backend_flags(w, dep)} {flags}{extra} --seed {seed}",
                                prep=prep, stream=kind)
                    out.append(Job(line, "q4", name, f"recal_shift_{tag}", "hit", d,
                                   needs_of(w, dep, ("shift_preps",))))
    for name in q4["k_workloads"]:
        if name not in wl:
            continue
        w = wl[name]["w"]
        for k in q4["fixed_k"]:
            out.append(hit_job("q4", name, w, f"k{k}_s0", "recal", "recal", 0.05, dep, f"{flags} --k {k}",
                               "recal_fixed_k", 0))
        out.append(hit_job("q4", name, w, "mixed_s0", "recal", "recal", 0.05, dep, f"{flags} --k 10 --k-mixed",
                           "recal_mixed_k", 0))
    out += q4_update_jobs(cfg, wl)
    return out


UPDATE_SYSTEMS = {
    "recal": ("recal", "recal", ""),
    "aker": ("aker", "aker", ""),
    "proximity-a": ("proximity", "proximity_p0.1", "--prox-percentile 0.1 --tag _p0.1"),
    "proximity-b": ("proximity", "proximity_p1", "--prox-percentile 1 --tag _p1"),
}


def update_job_id(name: str, mode: str, rate: float, system: str, seed: int) -> str:
    return f"{name}_{mode}{round(100 * rate):d}_{system}_s{seed}"


def update_cells(cfg: dict) -> list[tuple]:
    """(system, mode, rate) of the update runs of one workload: ReCal and Aker on uniform and targeted at 1% and
    10%, Proximity A and B on targeted 10% only."""
    q4, out = cfg["q4"], []
    for system in ("recal", "aker"):
        out += [(system, mode, rate) for mode in q4["update_modes"] for rate in q4["update_rates"]]
    out += [(system, "targeted", max(q4["update_rates"])) for system in ("proximity-a", "proximity-b")]
    return out


def q4_update_jobs(cfg: dict, wl: OrderedDict) -> list[Job]:
    """Update runs: one independent job each, with its own prep folder and its own live pgvector clone
    (created before the job and destroyed after it by scripts/pg_clones.sh; DSN in $PG_CLONES/<job id>.dsn).
    Every system answers misses with live searches on that clone and applies the schedule's inserts and deletes
    in order; ground truth is the exact top-200 filtered by liveness."""
    q4, out = cfg["q4"], []
    for name in q4["update_workloads"]:
        if name not in wl:
            continue
        w, budget = wl[name]["w"], wl[name]["standin"]
        for system, mode, rate in update_cells(cfg):
            method, stem, flags = UPDATE_SYSTEMS[system]
            if method == "recal":
                flags = f"--target {cfg['q2']['target']:g}"
            job = update_job_id(name, mode, rate, system, 0)
            prep = f"{q4['update_prep_dir']}/{name}_{mode}{round(100 * rate):d}"
            d = f"{ROOT}/q4/{name}/updates/{job}"
            strict = " --strict-modes" if method == "recal" else ""
            line = cell(d, stem, method, w, budget,
                        f"--backend pgvector --pg-dsn @$PG_CLONES/{job}.dsn --updates {prep}/updates.npz "
                        f"{flags}{strict} --seed 0", prep=prep, clone=job)
            out.append(Job(line, "q4", name, f"update_{system}", "update", d,
                           needs_of(w, "exact", ("update_preps", "pgvector_db"))))
    return out


# ---- assembling ------------------------------------------------------------------------------------------------

EXPERIMENTS = ("q1", "q2", "q3", "q4")


def cell_jobs(cfg: dict, medrag: dict | None = None) -> tuple[OrderedDict, list[str]]:
    """Jobs of the cells that do not depend on other results, per experiment (Q3 hit-rate runs included)."""
    wl, skipped = plan(cfg, medrag)
    return OrderedDict([("q1", q1_jobs(cfg, wl)), ("q2", q2_jobs(cfg, wl)), ("q3", q3_hit_jobs(cfg, wl)),
                        ("q4", q4_jobs(cfg, wl))]), skipped


def all_lines(cfg: dict, only=None, medrag: dict | None = None) -> dict[str, list[str]]:
    """Every cell line before de-duplication (the follow-up Q3 jobs are added by runs())."""
    jobs, _ = cell_jobs(cfg, medrag)
    return {exp: [j.line for j in v] for exp, v in jobs.items() if only is None or exp in only}


def runs(cfg: dict, only=None, medrag: dict | None = None) -> dict:
    """Each distinct run once, with the mapping from every experiment cell to the run that produces it.

    jobs: experiment -> Jobs to execute (the first experiment, in q1..q4 order, that contains a run emits it);
    runs: hash -> {line, out, job fields, cells: [{experiment, cell, out}]}; cells: experiment -> [{cell, hash}];
    split_points: Q3 points {cell, workload, system, node_share, hit_hash, replay_hash, arm}; skipped: workloads left out.
    """
    from recal import matrix_hash
    wl, skipped = plan(cfg, medrag)
    cells, _ = cell_jobs(cfg, medrag)
    jobs, table, listed = OrderedDict((e, []) for e in EXPERIMENTS), {}, OrderedDict((e, []) for e in EXPERIMENTS)
    job_hashes = OrderedDict((e, []) for e in EXPERIMENTS)

    def add(job: Job) -> str:
        h, out = matrix_hash.cell_hash(job.line), matrix_hash.out_dir(job.line)
        entry = table.get(h)
        if entry is None:
            entry = table[h] = {"line": job.line, "out": out, "workload": job.workload, "system": job.system,
                                "kind": job.kind, "needs": list(job.needs), "outputs": list(job.produced),
                                "inputs": list(job.inputs), "cells": []}
            jobs[job.exp].append(job)
            job_hashes[job.exp].append(h)
        entry["cells"].append({"experiment": job.exp, "cell": out, "out": out})
        listed[job.exp].append({"cell": out, "hash": h})
        return h

    hit_hash = {}
    for exp, items in cells.items():
        for job in items:
            hit_hash[job.cell] = add(job)
    q3_hits = cells["q3"]
    follow = q3_followups(cfg, wl, q3_hits, lambda j: table[hit_hash[j.cell]]["out"])
    replay = {}
    for job in follow:
        replay[job.cell] = add(job)
    points = []
    for name, entry in wl.items():
        sweep = replay[f"{ROOT}/q3/{name}/sweep"]
        points.append({"cell": f"q3/{name}/nodeonly", "workload": name, "system": "nodeonly", "node_share": 100,
                       "hit_hash": None, "replay_hash": sweep, "arm": "nodeonly"})
        for key in cfg["q3"]["systems"]:
            for ns in cfg["q3"]["node_shares"]:
                points.append({"cell": f"q3/{name}/{key}_n{ns}", "workload": name, "system": key, "node_share": ns,
                               "hit_hash": hit_hash[f"{ROOT}/q3/{name}/{key}_n{ns}"], "replay_hash": sweep,
                               "arm": f"{key}_n{ns}"})
    keep = [e for e in EXPERIMENTS if only is None or e in only]
    return {"jobs": {e: jobs[e] for e in keep}, "job_hashes": {e: job_hashes[e] for e in keep}, "runs": table, "cells": {e: listed[e] for e in keep},
            "split_points": points, "skipped": skipped}


def sharing_table(cfg: dict, only=None, medrag: dict | None = None) -> dict:
    """Per experiment: cells, runs it executes, cells whose run is also used by another experiment, and cells that
    repeat a run of the same experiment; Q3 also splits its cells into split points and thread lines."""
    r = runs(cfg, only, medrag)
    out = {}
    for exp, cells in r["cells"].items():
        owners = {h: {c["experiment"] for c in r["runs"][h]["cells"]} for h in {c["hash"] for c in cells}}
        counts = {}
        for c in cells:
            counts[c["hash"]] = counts.get(c["hash"], 0) + 1
        out[exp] = {"cells": len(cells), "runs_emitted": len(r["jobs"][exp]),
                    "shared_with_other": sum(1 for c in cells if len(owners[c["hash"]]) > 1),
                    "repeated_within": sum(n - 1 for n in counts.values())}
    pts = r["split_points"]
    shared = sum(1 for p in pts if p["hit_hash"] and any(c["experiment"] != "q3" for c in r["runs"][p["hit_hash"]]["cells"]))
    out["q3_detail"] = {"split_points": len(pts), "node_only": sum(1 for p in pts if p["hit_hash"] is None),
                        "hit_rate_runs_shared_with_q1": shared,
                        "hit_rate_runs_new": len([p for p in pts if p["hit_hash"]]) - shared,
                        "thread_lines": sum(1 for j in r["jobs"].get("q3", []) if j.kind == "threads") * 4,
                        "replay_jobs": sum(1 for j in r["jobs"].get("q3", []) if j.kind == "replay")}
    out["total"] = {"cells": sum(v["cells"] for k, v in out.items() if k in EXPERIMENTS),
                    "runs_emitted": len(r["runs"]), "skipped": r["skipped"]}
    return out
