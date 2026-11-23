#!/usr/bin/env python3
"""Numbers for the prunability report: runs per group of cells, sharing, relative cost, and the effect of each
candidate reduction. Reports only; the matrix is not changed.

  python -m scripts.prunability_stats --config experiments/matrix.example.json [--json out.json]

Cost weights are relative planning units, not measurements: one stored-answer replay of a baseline = 1; a ReCal
cell (construction, labeling, candidate searches) 3; real Aker 2; QVCache 2; the choice job 6; a node cache curve or
depth table 15; a throughput line (7 client counts x 3 repetitions on live DiskANN) 40; an update job (live pgvector
clone, per-event inserts and deletes) 60.
"""

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path

from recal import matrix

WEIGHTS = {"replay": 1, "recal": 3, "aker": 2, "qvcache": 2, "choose": 6, "measure": 15, "throughput": 40, "update": 60}

GROUPS = OrderedDict([
    ("Q1 ReCal deployable", ("q1", r"/recal_deployable_", "recal")),
    ("Q1 ReCal idealized", ("q1", r"/recal_idealized_", "recal")),
    ("Q1 Aker (both backends)", ("q1", r"/aker_", "aker")),
    ("Q1 Proximity A (both backends)", ("q1", r"/proximity-a_", "aker")),
    ("Q1 Proximity B (both backends)", ("q1", r"/proximity-b_", "aker")),
    ("Q1 QVCache default (both backends)", ("q1", r"/qvcache_(exact|diskann)_b", "qvcache")),
    ("Q1 QVCache deviation sweep", ("q1", r"/qvcache-d", "qvcache")),
    ("Q2 full ReCal bar", ("q2", r"/default_s", "recal")),
    ("Q2 ablation: no union", ("q2", r"/no-union_s", "recal")),
    ("Q2 ablation: no expansion", ("q2", r"/no-expansion_s", "recal")),
    ("Q2 ablation: no shells", ("q2", r"/no-shells_s", "recal")),
    ("Q2 ablation: LRU", ("q2", r"/lru_s", "recal")),
    ("Q2 ablation: no calibration", ("q2", r"/no-calibration_s", "recal")),
    ("Q2 ablation: no recalibration", ("q2", r"/no-recalibration_s", "recal")),
    ("Q2 ablation: no mini cache", ("q2", r"/no-mini-cache_s", "recal")),
    ("Q3 node cache curves and depth tables", ("q3", r"/(node_cache_curve|depth_latency)/", "measure")),
    ("Q3 split choice jobs", ("q3", r"/chosen\.json$", "choose")),
    ("Q3 split sweep (ReCal cells)", ("q3", r"/recal_s\d+_s\d", "recal")),
    ("Q3 chosen split (ReCal cells)", ("q3", r"/recal_chosen_s\d", "recal")),
    ("Q3 Aker and QVCache arm cells", ("q3", r"/(aker|qvcache)_chosen", "aker")),
    ("Q3 throughput: no cache", ("q3", r"/throughput_none", "throughput")),
    ("Q3 throughput: node cache only", ("q3", r"/throughput_nodecache", "throughput")),
    ("Q3 throughput: Aker", ("q3", r"/throughput_aker", "throughput")),
    ("Q3 throughput: QVCache", ("q3", r"/throughput_qvcache", "throughput")),
    ("Q3 throughput: ReCal", ("q3", r"/throughput_recal", "throughput")),
    ("Q4 budget grid", ("q4", r"/recal_b[0-9.]+_h30000_s", "recal")),
    ("Q4 history grid", ("q4", r"/recal_b[0-9.]+_h(10000|50000)_s", "recal")),
    ("Q4 shifts, signals on", ("q4", r"/shift_\w+_signals_s", "recal")),
    ("Q4 shifts, periodic only", ("q4", r"/shift_\w+_periodic_s", "recal")),
    ("Q4 fixed K", ("q4", r"/k\d+_s", "recal")),
    ("Q4 mixed K", ("q4", r"/mixed_s", "recal")),
    ("Q4 bursts with mini cache", ("q4", r"/burst_b[0-9.]+_mini_s", "recal")),
    ("Q4 bursts without mini cache", ("q4", r"/burst_b[0-9.]+_nomini_s", "recal")),
    ("Q4 updates: ReCal", ("q4", r"/updates/\S*_recal_s", "update")),
    ("Q4 updates: Aker", ("q4", r"/updates/\S*_aker_s", "update")),
    ("Q4 updates: Proximity A", ("q4", r"/updates/\S*_proximity-a_s", "update")),
    ("Q4 updates: Proximity B", ("q4", r"/updates/\S*_proximity-b_s", "update")),
])


def classify(exp: str, out: str, line: str):
    for name, (e, pattern, kind) in GROUPS.items():
        if e == exp and re.search(pattern, out):
            return name, kind
    raise ValueError(f"no group for {exp} {out}")


def weight_of(name: str, kind: str, out: str) -> float:
    if name.startswith("Q1") and "idealized" not in name and kind == "recal":
        return WEIGHTS["recal"]
    return WEIGHTS[kind]


def budget_of(out: str):
    found = re.search(r"(?:_b|/b)([0-9.]+?)(?:_|/)", out + "_")
    return float(found.group(1)) if found else None


def collect(cfg: dict) -> dict:
    result = matrix.runs(cfg)
    cells = []
    for exp, entries in result["cells"].items():
        lines = {c: None for c in ()}
        for entry in entries:
            run = result["runs"][entry["hash"]]
            name, kind = classify(exp, entry["cell"], run["line"])
            cells.append({"exp": exp, "out": entry["cell"], "hash": entry["hash"], "group": name, "kind": kind,
                          "line": run["line"], "weight": weight_of(name, kind, entry["cell"])})
    return {"cells": cells, "runs": result["runs"]}


def summarize(cells: list[dict]) -> dict:
    """Per group: cells, distinct runs in the group, runs it executes (first owner in matrix order), runs also used
    elsewhere, and the cost of the runs it executes."""
    owner, users = {}, {}
    for c in cells:
        owner.setdefault(c["hash"], c["group"])
        users.setdefault(c["hash"], set()).add(c["group"])
    rows = OrderedDict((g, {"cells": 0, "distinct": set(), "owned": 0, "shared_elsewhere": 0, "units": 0.0}) for g in GROUPS)
    for c in cells:
        row = rows[c["group"]]
        row["cells"] += 1
        row["distinct"].add(c["hash"])
    for g, row in rows.items():
        mine = row["distinct"]
        row["owned"] = sum(1 for h in mine if owner[h] == g)
        row["shared_elsewhere"] = sum(1 for h in mine if len(users[h]) > 1)
        weights = {c["hash"]: c["weight"] for c in cells if c["group"] == g}
        row["units"] = sum(weights[h] for h in mine if owner[h] == g)
        row["distinct"] = len(mine)
    return rows


def total_units(cells: list[dict]) -> tuple[int, float]:
    seen = {}
    for c in cells:
        seen.setdefault(c["hash"], c["weight"])
    return len(seen), float(sum(seen.values()))


def parts(out: str) -> dict:
    w = re.search(r"results/q\d/([^/]+)/", out)
    seed = re.search(r"_s(\d+)(/|$)", out)
    return {"workload": w.group(1) if w else None, "seed": int(seed.group(1)) if seed else None,
            "budget": budget_of(out)}


def rules(cfg: dict) -> OrderedDict:
    """Candidate reductions: name -> predicate(cell) that is True for cells to DROP."""
    wl = cfg["workloads"]
    q2b = {n: w["q2_budget"] for n, w in wl.items()}
    heavy = {"spacev-0.99", "tripclick"}
    skewed_pair = {"spacev-0.3", "spacev-0.6", "triviaqa-0.3", "triviaqa-0.6"}

    def sweep(c):
        p = parts(c["out"])
        return c["group"] == "Q1 QVCache deviation sweep" and (p["budget"] != q2b[p["workload"]] or p["seed"] != 0)

    def abl(c):
        return c["group"].startswith("Q2 ablation") and parts(c["out"])["workload"] in skewed_pair

    def split_seed(c):
        return c["group"] == "Q3 split sweep (ReCal cells)" and parts(c["out"])["seed"] == 1

    def throughput(c):
        if not (c["group"].startswith("Q3 throughput") or c["group"] == "Q3 Aker and QVCache arm cells"):
            return False
        p = parts(c["out"])
        return not (p["budget"] == 0.05 or p["budget"] == 0.001 or p["workload"] in heavy)

    def burst(c):
        p = parts(c["out"])
        return c["group"] == "Q4 bursts without mini cache" and p["budget"] != 0.05

    def updates(c):
        if not c["group"].startswith("Q4 updates"):
            return False
        job = c["out"].split("/updates/")[1]
        return c["group"] in ("Q4 updates: Proximity A", "Q4 updates: Proximity B") or re.search(r"_(uniform|targeted)5_", job)

    return OrderedDict([
        ("Q1 QVCache D sweep at the Q2 budget, seed 0 only", sweep),
        ("Q2 ablations on 4 of 8 workloads (drop Zipf 0.3 and 0.6)", abl),
        ("Q3 split sweep with one seed", split_seed),
        ("Q3 throughput at 5% (MedRAG 0.1%) for all, other budgets for SPACEV 0.99 and TripClick", throughput),
        ("Q4 no-mini bursts at the Q2 budget only", burst),
        ("Q4 updates: ReCal and Aker only, rates 1% and 10%", updates),
    ])


def effect(cells: list[dict], predicate) -> dict:
    kept = [c for c in cells if not predicate(c)]
    n0, u0 = total_units(cells)
    n1, u1 = total_units(kept)
    return {"dropped_cells": len(cells) - len(kept), "runs_before": n0, "runs_after": n1, "units_saved": u0 - u1}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    cfg = json.loads(a.config.read_text())
    data = collect(cfg)
    rows = summarize(data["cells"])
    n, u = total_units(data["cells"])
    print(f"all groups: {len(data['cells'])} cells, {n} distinct runs, {u:.0f} units")
    for g, r in rows.items():
        print(f"{g}: cells {r['cells']}, distinct {r['distinct']}, executed here {r['owned']}, shared elsewhere "
              f"{r['shared_elsewhere']}, units {r['units']:.0f}")
    effects = {name: effect(data["cells"], fn) for name, fn in rules(cfg).items()}
    for name, e in effects.items():
        print(f"{name}: drops {e['dropped_cells']} cells, runs {e['runs_before']} -> {e['runs_after']}, "
              f"saves {e['units_saved']:.0f} units")
    everything = effect(data["cells"], lambda c: any(fn(c) for fn in rules(cfg).values()))
    print(f"all reductions: runs {everything['runs_before']} -> {everything['runs_after']}, saves "
          f"{everything['units_saved']:.0f} units of {u:.0f}")
    if a.json:
        a.json.write_text(json.dumps({"groups": {g: {**r} for g, r in rows.items()}, "reductions": effects,
                                      "all": everything, "total_units": u}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
