"""The whole experiment matrix (Q1 to Q4) as job lines, from one configuration.

Each line skips itself when its result exists and carries a marker run_jobs.py recognizes. The configuration
names the workloads (prep folder, stream, stored precision, budgets, DiskANN files); see
experiments/matrix.example.json. design_table() gives the cell count and the dimensions multiplied for every
experiment; tests compare it with the line counts and with the design.
"""

from recal import robustness
from recal.system_benefit import jobs as q3jobs
from recal.system_benefit.split import SPLITS

PY = ".venv-core/bin/python"
ANN = ".venv-ann/bin/python"
ABLATIONS = ("--no-union", "--no-expansion", "--no-shells", "--lru", "--no-calibration", "--no-recalibration",
             "--no-mini-cache")
BACKENDS = ("exact", "diskann")
ROOT = "results"


def backend_flags(w: dict, backend: str) -> str:
    if backend == "exact":
        return "--backend exact"
    return f"--backend file --backend-file {w['diskann_file']} --backend-target 0.95"


def cell(d: str, stem: str, method: str, w: dict, budget, flags: str) -> str:
    b = budget if isinstance(budget, str) else f"{budget:g}"
    return (f"[ -f {d}/{stem}.json ] || {{ mkdir -p {d} && PYTHONPATH=. {PY} -u -m recal.cells --method {method} "
            f"--prep {w['prep']} --stream {w['stream']} --budget {b} --coord-bytes {w['coord_bytes']:g} "
            f"{flags} --out {d} > {d}/log.txt 2>&1; }}  # recal.cells")


def q1_systems(cfg: dict) -> list[tuple]:
    """(label, method, stem, backend, flags) of every system cell of one (workload, budget, seed)."""
    q1, out = cfg["q1"], []
    for t in q1["targets"]:
        for mode, flag in (("deployable", ""), ("idealized", "--idealized")):
            out.append((f"recal_{mode}_t{t:g}", "recal", "recal", q1["recal_backend"], f"--target {t:g} {flag}"))
    for backend in q1["backends"]:
        out.append((f"aker_{backend}", "aker", "aker", backend, ""))
        out.append((f"proximity-a_{backend}", "proximity", "proximity_p0.1", backend,
                    "--prox-percentile 0.1 --tag _p0.1"))
        out.append((f"proximity-b_{backend}", "proximity", "proximity_p1", backend, "--prox-percentile 1 --tag _p1"))
        out.append((f"qvcache_{backend}", "qvcache", "qvcache", backend, ""))
    for d in q1["qvcache_sweep"]:
        out.append((f"qvcache-d{d:g}_exact", "qvcache", f"qvcache_d{d:g}", "exact",
                    f"--deviation-factor {d:g} --tag _d{d:g}"))
    return out


def q1_lines(cfg: dict) -> list[str]:
    out = []
    for name, w in cfg["workloads"].items():
        for budget in w["budgets"]:
            for seed in cfg["seeds"]:
                for label, method, stem, backend, flags in q1_systems(cfg):
                    d = f"{ROOT}/q1/{name}/{label}_b{budget:g}_s{seed}"
                    out.append(cell(d, stem, method, w, budget, f"{backend_flags(w, backend)} {flags} --seed {seed}"))
    return out


def q2_lines(cfg: dict) -> list[str]:
    q2, out = cfg["q2"], []
    for name, w in cfg["workloads"].items():
        for label, flag in [("default", "")] + [(f.lstrip("-"), f) for f in ABLATIONS]:
            for seed in cfg["seeds"]:
                d = f"{ROOT}/q2/{name}/{label}_s{seed}"
                out.append(cell(d, "recal", "recal", w, w["q2_budget"], f"{backend_flags(w, cfg['q1']['recal_backend'])} "
                                f"--target {q2['target']:g} {flag} --seed {seed}"))
    return out


def q3_nodes(w: dict, budget: float, share: float = 1.0) -> int:
    """Node cache nodes of `share` of the total budget at stored precision (DiskANN node: vector, links, header)."""
    from recal.accounting import budget_bytes
    from recal.system_benefit.nodecache import nodes_for_bytes
    return nodes_for_bytes(share * budget_bytes(w["n_vectors"], w["dim"], budget, w["coord_bytes"]), w["dim"],
                           w["coord_bytes"], w["degree"])


def q3_lines(cfg: dict) -> list[str]:
    q3, out = cfg["q3"], []
    for name, w in cfg["workloads"].items():
        root = f"{ROOT}/q3/{name}"
        nodes_dim = f"--n-vectors {w['n_vectors']} --dim {w['dim']} --degree {w['degree']}"
        curve, depth = f"{root}/node_cache_curve/curve.json", f"{root}/depth_latency/diskann.json"
        out += q3jobs.node_cache_curve_lines(w["index_dir"], w["queries"], f"{name}", w["curve_nodes"], w["node_bytes"])
        out += q3jobs.depth_table_lines(w["prep"], w["index_dir"], f"{name}", w["depths"])
        out.append(f"[ -f {root}/depth_latency/pgvector.json ] || {{ mkdir -p {root}/depth_latency && PYTHONPATH=. "
                   f"{ANN} -u scripts/depth_latency_pgvector.py --dsn \"$PG_DSN\" --queries {w['queries']} "
                   f"--ks {' '.join(map(str, q3['ks']))} --out {root}/depth_latency/pgvector.json "
                   f"> {root}/depth_latency/log_pg.txt 2>&1; }}")
        for budget in w["q3_budgets"]:
            b = f"{root}/b{budget:g}"
            out.append(f"[ -f {b}/chosen.json ] || {{ mkdir -p {b} && PYTHONPATH=. {PY} -u -m "
                       f"recal.system_benefit.choose run --prep {w['prep']} --stream {w['stream']} "
                       f"--total-fraction {budget:g} --curve {curve} --depth-table {depth} {nodes_dim} "
                       f"--coord-bytes {w['coord_bytes']:g} {backend_flags(w, 'diskann')} --target {q3['target']:g} "
                       f"--out {b}/chosen.json > {b}/log_choose.txt 2>&1; }}  # recal.system_benefit.choose")
            for s in SPLITS:
                if s == 0:
                    continue
                for seed in cfg["seeds"]:
                    out.append(cell(f"{b}/recal_s{q3jobs.pct(s)}_s{seed}", "recal", "recal", w, s * budget,
                                    f"{backend_flags(w, 'diskann')} --target {q3['target']:g} --seed {seed}"))
            share = f"$({PY} -m recal.system_benefit.choose get {b}/chosen.json recal_budget_fraction)"
            for seed in cfg["seeds"]:
                out.append(cell(f"{b}/recal_chosen_s{seed}", "recal", "recal", w, share,
                                f"{backend_flags(w, 'diskann')} --target {q3['target']:g} --seed {seed}"))
            for system in ("aker", "qvcache"):
                out.append(cell(f"{b}/{system}_chosen", system, system, w, share,
                                f"{backend_flags(w, 'diskann')} --seed 0"))
            nodes_chosen = f"$({PY} -m recal.system_benefit.choose get {b}/chosen.json node_cache_nodes)"
            arms = {"none": ("none", None), "nodecache": ("none", q3_nodes(w, budget)),
                    "aker": (f"{b}/aker_chosen/aker.json", nodes_chosen),
                    "qvcache": (f"{b}/qvcache_chosen/qvcache.json", nodes_chosen),
                    "recal": (f"{b}/recal_chosen_s0/recal.json", nodes_chosen)}
            for label, (run, nodes) in arms.items():
                spec = "none" if run == "none" else f"{label}={run}"
                node_flag = "" if nodes is None else f"--nodes {nodes} "
                clients = " ".join(map(str, q3jobs.CLIENTS))
                d = f"{b}/throughput_{label}"
                out.append(f"[ -f {d}/replay.json ] || {{ mkdir -p {d} && PYTHONPATH=. {ANN} -u {q3jobs.REPLAY} "
                           f"--workload {w['replay_workload']} --arm {spec} --clients {clients} {node_flag}"
                           f"--io-complete --repetitions {q3['repetitions']} --out {d}/replay.json "
                           f"> {d}/log.txt 2>&1; }}  # throughput_replay.py")
    return out


def q4_lines(cfg: dict) -> list[str]:
    q4, out = cfg["q4"], []
    wl = cfg["workloads"]
    for name in q4["grid_workloads"]:
        w = wl[name]
        out += robustness.grid_lines(w["prep"], w["stream"], name, seeds=cfg["seeds"],
                                     extra=f"--coord-bytes {w['coord_bytes']:g} {backend_flags(w, 'diskann')}")
    for name in q4["shift_workloads"]:
        w = wl[name]
        out += robustness.shift_lines(q4["shift_prep_dir"], name, seeds=q4["shift_seeds"],
                                      extra=f"--coord-bytes {w['coord_bytes']:g} {backend_flags(w, 'diskann')}")
    for name in q4["k_workloads"]:
        w = wl[name]
        out += robustness.k_lines(w["prep"], w["stream"], name, seeds=cfg["seeds"],
                                  extra=f"--coord-bytes {w['coord_bytes']:g} {backend_flags(w, 'diskann')}")
    for name in q4["burst_workloads"]:
        w = wl[name]
        out += robustness.burst_lines(w["prep"], w["stream"], name, budgets=w["q3_budgets"], seeds=cfg["seeds"],
                                      extra=f"--coord-bytes {w['coord_bytes']:g} {backend_flags(w, 'diskann')}")
    for name in q4["update_workloads"]:
        out += robustness.update_lines(wl[name], name, q4, cfg["seeds"])
    return out


EXPERIMENTS = {"q1": q1_lines, "q2": q2_lines, "q3": q3_lines, "q4": q4_lines}


def all_lines(cfg: dict, only=None) -> dict[str, list[str]]:
    return {name: fn(cfg) for name, fn in EXPERIMENTS.items() if only is None or name in only}


def design_table(cfg: dict) -> dict:
    """Per experiment: the number of cells and the dimensions multiplied, from the configuration alone."""
    wl, seeds = cfg["workloads"], len(cfg["seeds"])
    pairs = sum(len(w["budgets"]) for w in wl.values())
    systems = len(q1_systems(cfg))
    q3_pairs = sum(len(w["q3_budgets"]) for w in wl.values())
    n_split = len([s for s in SPLITS if s > 0])
    q4, q4w = cfg["q4"], cfg["q4"]
    grid_cells = len(robustness.BUDGETS) + len([h for h in robustness.HISTORIES if h != 30_000])
    n_updates = (len(q4["update_workloads"]) * len(q4["update_modes"]) * len(q4["update_rates"])
                 * len(q4["update_systems"]) * seeds)
    q3_per_pair = 1 + n_split * seeds + seeds + 2 + 5
    table = {
        "q1": {"cells": pairs * systems * seeds,
               "dimensions": f"{pairs} (workload, budget) x {systems} systems x {seeds} seeds"},
        "q2": {"cells": len(wl) * 8 * seeds, "dimensions": f"{len(wl)} workloads x 8 variants x {seeds} seeds"},
        "q3": {"cells": q3_pairs * q3_per_pair + len(wl) * 3,
               "dimensions": f"{q3_pairs} (workload, budget) x [1 choice + {n_split} splits x {seeds} seeds + "
                             f"chosen x {seeds} + Aker, QVCache + 5 throughput] + {len(wl)} workloads x 3 "
                             f"(curve, DiskANN and pgvector depth tables)"},
        "q4": {"cells": (len(q4["grid_workloads"]) * grid_cells * seeds
                         + len(q4["shift_workloads"]) * 3 * 2 * len(q4["shift_seeds"])
                         + len(q4["k_workloads"]) * 5 * seeds
                         + sum(len(wl[n]["q3_budgets"]) for n in q4["burst_workloads"]) * 2 * seeds + n_updates),
               "dimensions": f"grid {len(q4['grid_workloads'])} x {grid_cells} x {seeds}; shifts "
                             f"{len(q4['shift_workloads'])} x 3 x 2 x {len(q4['shift_seeds'])}; K "
                             f"{len(q4['k_workloads'])} x 5 x {seeds}; bursts "
                             f"{sum(len(wl[n]['q3_budgets']) for n in q4['burst_workloads'])} budgets x 2 x {seeds}; "
                             f"updates {len(q4['update_workloads'])} x {len(q4['update_modes'])} x "
                             f"{len(q4['update_rates'])} x {len(q4['update_systems'])} x {seeds}"},
    }
    table["total"] = {"cells": sum(v["cells"] for v in table.values()), "dimensions": ""}
    del q4w
    return table
