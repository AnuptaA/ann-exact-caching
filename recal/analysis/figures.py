"""Figure inputs and tables from the matrix results: one function per table, rows as dicts.

Everything is derived from the result JSONs and the per-request records of the runs listed in matrix_runs.json.
"""

import csv
import json
import re
from pathlib import Path

import numpy as np

from recal import records
from recal.analysis import bursts, guarantee_table, hit_composition, hit_rate_at_recall, rates
from recal.analysis.rates import by_k
from recal.system_benefit.best_split import mean_qps, split_of

ARG_FIELDS = ("budget", "target", "seed", "idealized", "history", "history_frac", "k", "k_mixed", "no_mini_cache")
HIT_KINDS = ("hit", "update")


def result_files(out: Path) -> list[Path]:
    return sorted(p for p in out.glob("*.json") if "method" in json.loads(p.read_text()))


def load_run(out: Path):
    """(result, records columns) of a run directory, None when there is no result."""
    for p in result_files(out):
        res = json.loads(p.read_text())
        rec = res.get("records_file")
        cols = records.load(p.with_name(rec))[0] if rec and p.with_name(rec).exists() else None
        return res, cols
    return None


def is_failed(res: dict, cols) -> str | None:
    if "error" in res:
        return "error"
    if res.get("within_M") is False:
        return "over budget"
    if cols is None and res.get("method") in ("recal", "aker", "proximity", "qvcache"):
        return "no records"
    return None


def run_state(entry: dict, root: Path = Path(".")) -> str:
    """done, missing or failed for one run of matrix_runs.json."""
    outputs = entry["outputs"]
    if entry["kind"] in HIT_KINDS:
        run = load_run(root / outputs[0].rstrip("/"))
        if run is None:
            return "missing"
        return "failed" if is_failed(*run) else "done"
    return "missing" if any(p.endswith(".json") and not (root / p).exists() for p in outputs) else "done"


def check_runs(run_table: dict, root: Path = Path(".")) -> dict:
    """Every run of matrix_runs.json present and sound: lists of missing and failed run directories."""
    missing, failed = [], []
    for entry in run_table.values():
        state = run_state(entry, root)
        if state == "missing":
            missing.append(entry["out"])
        elif state == "failed":
            failed.append({"run": entry["out"], "reason": "no usable result"})
    return {"missing": missing, "failed": failed}


def cell_records(run_table: dict, cells: dict, root: Path = Path(".")):
    """Yield (experiment, cell path, run entry, result, columns) for every cell of the hit and update runs."""
    cache = {}
    for exp, items in cells.items():
        for item in items:
            entry = run_table[item["hash"]]
            if entry["kind"] not in HIT_KINDS:
                continue
            if item["hash"] not in cache:
                cache[item["hash"]] = load_run(root / entry["outputs"][0].rstrip("/"))
            run = cache[item["hash"]]
            if run is not None and run[1] is not None:
                yield exp, item["cell"], entry, run[0], run[1]


def base_row(exp: str, cell: str, entry: dict, res: dict) -> dict:
    args = res.get("args", {})
    row = {"experiment": exp, "cell": cell, "workload": entry["workload"], "system": entry["system"],
           "backend": res.get("backend"), "mode": res.get("mode"), "within_M": res.get("within_M")}
    row.update({k: args.get(k) for k in ARG_FIELDS})
    return row


def runs_table(run_table: dict, cells: dict, root: Path = Path(".")) -> list[dict]:
    """One row per experiment cell: its labels and the rates of its run."""
    return [dict(base_row(exp, cell, entry, res), **rates(cols))
            for exp, cell, entry, res, cols in cell_records(run_table, cells, root)]


def recal_cells(run_table: dict, cells: dict, root: Path = Path(".")):
    for item in cell_records(run_table, cells, root):
        if item[3].get("method") == "recal":
            yield item


def guarantee_rows(run_table: dict, cells: dict, root: Path = Path(".")) -> list[dict]:
    """Recall guarantee per ReCal run, then the table pooled over every ReCal run of every question."""
    rows, pooled, seen = [], [], set()
    for exp, cell, entry, res, cols in recal_cells(run_table, cells, root):
        target = res["args"]["target"]
        g = guarantee_table(cols, target)
        rows.append({"experiment": exp, "cell": cell, "workload": entry["workload"], "target": target,
                     "n_windows": g["n_windows"], "share_below_target": g["share_below_target"],
                     "mean_margin": g["mean_margin"], "min_margin": g["min_margin"]})
        key = (entry["out"], target)
        if key not in seen:                     # a run shared by several cells counts once
            seen.add(key)
            pooled += list(g["recall_of_hits"] - target)
    margin = np.asarray(pooled)
    rows.append({"experiment": "all", "cell": "pooled", "workload": "all", "target": None,
                 "n_windows": len(margin),
                 "share_below_target": float((margin < 0).mean()) if len(margin) else float("nan"),
                 "mean_margin": float(margin.mean()) if len(margin) else float("nan"),
                 "min_margin": float(margin.min()) if len(margin) else float("nan")})
    return rows


def composition_rows(run_table: dict, cells: dict, root: Path = Path(".")) -> list[dict]:
    return [dict(base_row(exp, cell, entry, res), **hit_composition(cols))
            for exp, cell, entry, res, cols in recal_cells(run_table, cells, root)]


def matched_rows(rows: list[dict]) -> list[dict]:
    """Q1: ReCal's hit rate at each baseline's system recall, from ReCal's T sweep on the baseline's backend
    (deployable on DiskANN, idealized on exact), per workload, budget and seed."""
    q1 = [r for r in rows if r["experiment"] == "q1"]
    curves: dict = {}
    for r in q1:
        if r["system"].startswith("recal_"):
            curves.setdefault((r["workload"], r["budget"], r["seed"], r["mode"]), []).append(
                (r["system_recall"], r["hit_rate"]))
    out = []
    for r in q1:
        if r["system"].startswith("recal_"):
            continue
        mode = "idealized" if r["backend"] == "exact" else "deployable"
        pts = curves.get((r["workload"], r["budget"], r["seed"], mode), [])
        out.append({"workload": r["workload"], "budget": r["budget"], "seed": r["seed"], "system": r["system"],
                    "backend": r["backend"], "baseline_recall": r["system_recall"], "baseline_hit_rate": r["hit_rate"],
                    "recal_hit_rate_at_recall": hit_rate_at_recall(pts, r["system_recall"])})
    return out


def q2_rows(rows: list[dict]) -> list[dict]:
    """Q2: every ablation against ReCal's default run (Q1, T 0.9, deployable) at the same workload and budget."""
    default = {(r["workload"], r["budget"], r["seed"]): r for r in rows
               if r["experiment"] == "q1" and r["system"] == "recal_deployable" and r["target"] == 0.9}
    out = []
    for r in rows:
        if r["experiment"] != "q2":
            continue
        d = default.get((r["workload"], r["budget"], r["seed"]))
        out.append(dict(r, default_hit_rate=d["hit_rate"] if d else None,
                        default_system_recall=d["system_recall"] if d else None))
    return out


def windows(cols: dict, size: int = 1000) -> list[dict]:
    """Rates in consecutive blocks of `size` scored requests (shift timelines)."""
    n = len(cols["path"])
    return [dict(start=s, **rates({k: v[s:s + size] for k, v in cols.items()}))
            for s in range(0, n - size + 1, size)]


def q4_tables(run_table: dict, cells: dict, root: Path = Path(".")) -> dict:
    """Q4: grid (budgets, histories), shift timelines, per-K rates, updates; bursts are in burst_rows."""
    grid, shifts, per_k, updates = [], [], [], []
    for exp, cell, entry, res, cols in cell_records(run_table, cells, root):
        if exp != "q4":
            continue
        row = dict(base_row(exp, cell, entry, res), **rates(cols))
        name = cell.rstrip("/").split("/")[-1]
        if "/updates/" in cell:
            updates.append(dict(row, job=name))
        elif name.startswith("shift_"):
            kind, signals = re.match(r"shift_(\w+?)_(signals|periodic)_s\d+", name).groups()
            shifts += [dict(w, workload=entry["workload"], shift=kind, signals=signals, seed=row["seed"], cell=cell)
                       for w in windows(cols)]
        elif name.startswith(("k", "mixed")):
            if name.startswith("mixed"):
                per_k += [dict(r, workload=entry["workload"], run="mixed", K=k) for k, r in by_k(cols).items()]
            else:
                per_k.append(dict(row, run="fixed", K=res["args"]["k"]))
        else:
            grid.append(row)
    return {"q4_grid": grid, "q4_shifts": shifts, "q4_per_k": per_k, "q4_updates": updates}


def burst_rows(run_table: dict, cells: dict, stream_keys, root: Path = Path(".")) -> list[dict]:
    """TripClick bursts from the Q1 run (mini cache on) and the Q2 no-mini run, same budget and seed."""
    with_mini, without = {}, {}
    for exp, cell, entry, res, cols in recal_cells(run_table, cells, root):
        if entry["workload"] != "tripclick":
            continue
        key = (res["args"]["budget"], res["args"]["seed"])
        if exp == "q1" and entry["system"] == "recal_deployable" and res["args"]["target"] == 0.9:
            with_mini[key] = cols
        if exp == "q2" and res["args"].get("no_mini_cache"):
            without[key] = cols
    out = []
    for key in sorted(set(with_mini) & set(without)):
        d = bursts.bursts_with_and_without_mini(with_mini[key], without[key], np.asarray(stream_keys))
        for variant in ("with_mini", "without_mini"):
            for part in ("burst", "other"):
                out.append(dict(d[variant][part], budget=key[0], seed=key[1], variant=variant, part=part,
                                burst_share=d["burst_share"]))
    return out


def q3_rows(run_table: dict, root: Path = Path(".")) -> dict:
    """Split points (throughput at saturation with system recall), best splits and thread lines per workload."""
    split_rows, best_rows, thread_rows = [], [], []
    for entry in run_table.values():
        if entry["kind"] not in ("replay", "best_split", "threads"):
            continue
        out = root / entry["outputs"][0]
        if not out.exists():
            continue
        data = json.loads(out.read_text())
        if entry["kind"] == "replay":
            sat = data[0].get("saturation_clients") if data else None
            qps = mean_qps(data, sat) if sat else {}
            meta = {r["arm"]: r for r in data if r["clients"] == sat}
            for arm, value in qps.items():
                system, share = ("nodeonly", 100) if arm == "nodeonly" else split_of(arm)
                split_rows.append({"workload": entry["workload"], "system": system, "node_share": share,
                                   "arm": arm, "clients": sat, "qps": value, "hit_rate": meta[arm]["hit_rate"],
                                   "system_recall": meta[arm].get("system_recall"), "nodes": meta[arm]["nodes"]})
        elif entry["kind"] == "best_split":
            best_rows += [dict(v, workload=entry["workload"], system=s) for s, v in data.items()]
        else:
            by = {}
            for r in data:
                by.setdefault((r["arm"], r["clients"]), []).append(r)
            for (arm, clients), rs in sorted(by.items()):
                thread_rows.append({"workload": entry["workload"], "arm": arm, "clients": clients,
                                    "qps": float(np.mean([r["qps"] for r in rs])),
                                    "qps_std": float(np.std([r["qps"] for r in rs])),
                                    "mean_ms": float(np.mean([r["mean_ms"] for r in rs])),
                                    "p99_ms": float(np.mean([r["p99_ms"] for r in rs])),
                                    "hit_rate": rs[0]["hit_rate"], "system_recall": rs[0].get("system_recall")})
    return {"q3_split": split_rows, "q3_best": best_rows, "q3_threads": thread_rows}


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r)) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
