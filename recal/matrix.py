"""The whole experiment matrix (Q1 to Q4) as job lines, from one configuration.

Each line skips itself when its result exists and carries a marker run_jobs.py recognizes. The configuration
names the datasets with their prep folders, streams, stored precision and budgets; see
experiments/matrix.example.json.
"""

from recal import robustness
from recal.system_benefit import jobs as q3jobs
from recal.system_benefit.split import SPLITS

PY = ".venv-core/bin/python"
ABLATIONS = ("--no-union", "--no-expansion", "--no-shells", "--lru", "--no-calibration", "--no-recalibration",
             "--no-mini-cache")


def cell(d: str, stem: str, method: str, prep: str, stream: str, budget: float, flags: str) -> str:
    return (f"[ -f {d}/{stem}.json ] || {{ mkdir -p {d} && PYTHONPATH=. {PY} -u -m recal.cells --method {method} "
            f"--prep {prep} --stream {stream} --budget {budget:g} {flags} --out {d} > {d}/log.txt 2>&1; }}"
            "  # recal.cells")


def q1_lines(cfg: dict) -> list[str]:
    q1, out = cfg["q1"], []
    for name, ds in cfg["datasets"].items():
        base = f"--history-frac {cfg['history_frac']:g} --coord-bytes {ds['coord_bytes']:g}"
        for stream in ds["streams"]:
            for budget in ds["budgets"]:
                for seed in q1["seeds"]:
                    root = f"results/q1/{name}/{stream}"
                    for t in q1["targets"]:
                        for mode, flag in (("deployable", ""), ("idealized", "--idealized")):
                            d = f"{root}/recal_{mode}_t{t:g}_b{budget:g}_s{seed}"
                            out.append(cell(d, "recal", "recal", ds["prep"], stream, budget,
                                            f"{base} --target {t:g} {flag} --seed {seed}"))
                    for system, method, extra in (("aker", "aker", ""),
                                                  ("proximity-p0.1", "proximity", "--prox-percentile 0.1 --tag _p0.1"),
                                                  ("proximity-p1", "proximity", "--prox-percentile 1 --tag _p1")):
                        d = f"{root}/{system}_b{budget:g}_s{seed}"
                        stem = method + (extra.split("--tag ")[1] if "--tag" in extra else "")
                        out.append(cell(d, stem, method, ds["prep"], stream, budget, f"{base} {extra} --seed {seed}"))
    return out


def q2_lines(cfg: dict) -> list[str]:
    q2, out = cfg["q2"], []
    for c in q2["cells"]:
        ds = cfg["datasets"][c["dataset"]]
        base = (f"--history-frac {cfg['history_frac']:g} --coord-bytes {ds['coord_bytes']:g} "
                f"--target {q2['target']:g}")
        for label, flag in [("default", "")] + [(f.lstrip("-"), f) for f in ABLATIONS]:
            for seed in q2["seeds"]:
                d = f"results/q2/{c['dataset']}/{c['stream']}/{label}_s{seed}"
                out.append(cell(d, "recal", "recal", ds["prep"], c["stream"], q2["budget"],
                                f"{base} {flag} --seed {seed}"))
    return out


def q3_lines(cfg: dict) -> list[str]:
    q3, out = cfg["q3"], []
    for c in q3["cells"]:
        ds = cfg["datasets"][c["dataset"]]
        name = f"{c['dataset']}_{c['stream']}"
        out += q3jobs.split_cell_lines(ds["prep"], c["stream"], q3["total_fraction"], name, extra=f"--coord-bytes "
                                       f"{ds['coord_bytes']:g} --target {q3['target']:g}")
        root = f"simulations/approximate_hits/results/q3/{name}"
        arms = {"none": "none", "nodecache": "none", "aker": c["aker_run"]}
        nodes = {"nodecache": c["total_nodes"], "none": c["base_nodes"], "aker": c["base_nodes"]}
        for s in SPLITS:
            if 0 < s < 1:
                label = f"recal_s{q3jobs.pct(s)}"
                arms[label] = f"{root}/{label}/recal.json"
                nodes[label] = int(round((1 - s) * c["total_nodes"]))
        out += q3jobs.throughput_lines(c["workload"], name, arms, nodes)
        out += q3jobs.single_client_lines(c["workload"], name, {"none": "none", "aker": c["aker_run"]})
        out += q3jobs.depth_table_lines(ds["prep"], c["index_dir"], name, c["depths"])
        out += q3jobs.node_cache_curve_lines(c["index_dir"], c["queries"], name, c["curve_nodes"], c["node_bytes"])
    return out


def q4_lines(cfg: dict) -> list[str]:
    q4, out = cfg["q4"], []
    for c in q4["cells"]:
        ds = cfg["datasets"][c["dataset"]]
        name = f"{c['dataset']}_{c['stream']}"
        extra = f"--coord-bytes {ds['coord_bytes']:g} --target {q4['target']:g}"
        out += robustness.grid_lines(ds["prep"], c["stream"], name, seeds=q4["seeds"], extra=extra)
        out += robustness.k_lines(ds["prep"], c["stream"], name, seeds=q4["seeds"], extra=extra)
        for kind, stream in c.get("shifts", {}).items():
            out += robustness.shift_lines(ds["prep"], name, (kind,), seeds=q4["shift_seeds"], extra=extra)
        if c.get("bursts"):
            out += robustness.burst_lines(ds["prep"], c["stream"], name, seeds=q4["seeds"], extra=extra)
        if c.get("updates"):
            out += robustness.update_lines(ds["prep"], c["stream"], name, c["updates"], seeds=q4["seeds"], extra=extra)
    return out


EXPERIMENTS = {"q1": q1_lines, "q2": q2_lines, "q3": q3_lines, "q4": q4_lines}


def all_lines(cfg: dict, only=None) -> dict[str, list[str]]:
    return {name: fn(cfg) for name, fn in EXPERIMENTS.items() if only is None or name in only}
