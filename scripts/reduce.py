#!/usr/bin/env python3
"""Pull the results, check every expected run is there and build the figure inputs (CSV) from the records.

  python -m scripts.reduce --jobs jobs/ --config experiments/matrix.json [--store s3://bucket/prefix] \
      [--out figures_data] [--no-pull]

Reads matrix_runs.json (written by gen_matrix), lists the missing and failed runs, and writes figures_data/*.csv
from whatever is complete. Exit status 1 when a run is missing or failed.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recal import matrix, store as store_mod  # noqa: E402
from recal.analysis import figures  # noqa: E402
from scripts import run_shard  # noqa: E402


def pull_results(s, run_table: dict) -> int:
    n = 0
    for h, entry in run_table.items():
        exp = entry["cells"][0]["experiment"]
        job = {"exp": exp, "hash": h, "outputs": entry["outputs"], "kind": entry["kind"]}
        n += run_shard.pull_job(s, job)
    return n


def stream_keys(cfg: dict, name: str = "tripclick"):
    w = cfg["workloads"].get(name)
    if w is None:
        return None
    streams = Path(w["prep"]) / "streams.npz"
    return np.load(streams)[w["stream"]] if streams.exists() else None


def reduce(cfg: dict, run_table: dict, cells: dict, out: Path, root: Path = Path(".")) -> dict:
    report = figures.check_runs(run_table, root)
    rows = figures.runs_table(run_table, cells, root)
    tables = {"runs": rows, "guarantee": figures.guarantee_rows(run_table, cells, root),
              "composition": figures.composition_rows(run_table, cells, root),
              "q1_matched": figures.matched_rows(rows), "q2_ablations": figures.q2_rows(rows)}
    tables.update(figures.q4_tables(run_table, cells, root))
    tables.update(figures.q3_rows(run_table, root))
    keys = stream_keys(cfg)
    tables["q4_bursts"] = figures.burst_rows(run_table, cells, keys, root) if keys is not None else []
    for name, table in tables.items():
        figures.write_csv(out / f"{name}.csv", table)
    report["tables"] = {name: len(table) for name, table in tables.items()}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--jobs", type=Path, default=Path("jobs"))
    ap.add_argument("--store", default=None)
    ap.add_argument("--no-pull", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("figures_data"))
    a = ap.parse_args(argv)
    cfg = json.loads(a.config.read_text())
    medrag = matrix.load_medrag_budgets(cfg)
    run_table = json.loads((a.jobs / "matrix_runs.json").read_text())
    cells = matrix.runs(cfg, medrag=medrag)["cells"]
    if not a.no_pull:
        print(f"pulled {pull_results(store_mod.open_store(a.store), run_table)} run(s)")
    report = reduce(cfg, run_table, cells, a.out)
    for name, n in report["tables"].items():
        print(f"{name}: {n} rows")
    if report["missing"] or report["failed"]:
        print(f"{len(report['missing'])} missing, {len(report['failed'])} failed (see {a.out / 'report.json'})")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
