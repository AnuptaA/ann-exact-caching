#!/usr/bin/env python3
"""Job lines for the whole experiment matrix from one configuration file.

  python -m scripts.gen_matrix --config experiments/matrix.example.json --dry-run
  python -m scripts.gen_matrix --config experiments/matrix.json --out jobs/ [--only q1 q2] [--hashes]
  python simulations/approximate_hits/run_jobs.py jobs/q1.txt --workers 8

--dry-run prints the number of cells per experiment. --hashes also records the config hash every line stores
(one process per line; slow for thousands of lines).
"""

import argparse
import json
from pathlib import Path

from recal import matrix


def print_table(table: dict) -> None:
    print(f"{'':6}{'cells':>7}{'distinct runs':>15}{'shared runs':>13}")
    for name in matrix.EXPERIMENTS:
        row = table[name]
        print(f"{name:6}{row['cells']:7d}{row['runs_emitted']:15d}{row['shared_with_other'] + row['repeated_within']:13d}")
    d = table["q3_detail"]
    print(f"q3: {d['split_points']} split points ({d['node_only']} node-only) and {d['thread_lines']} thread lines; "
          f"hit-rate runs {d['hit_rate_runs_new']} new, {d['hit_rate_runs_shared_with_q1']} shared")
    t = table["total"]
    print(f"total: {t['cells']} cells, {t['runs_emitted']} distinct runs")
    if t["skipped"]:
        print(f"left out (no medrag_budgets.json): {', '.join(t['skipped'])}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--only", nargs="+", choices=sorted(matrix.EXPERIMENTS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--table", action="store_true", help="print cells, distinct runs and shared runs per experiment")
    ap.add_argument("--hashes", action="store_true")
    a = ap.parse_args(argv)
    cfg = json.loads(a.config.read_text())
    medrag = matrix.load_medrag_budgets(cfg)
    if a.table:
        print_table(matrix.sharing_table(cfg, medrag=medrag))
        return 0
    result = matrix.runs(cfg, a.only, medrag)
    counts = {name: len(v) for name, v in result["cells"].items()}
    if result["skipped"]:
        print(f"no medrag_budgets.json: left out {', '.join(result['skipped'])}")
    for name in result["cells"]:
        print(f"{name}: {len(result['cells'][name])} cells, {len(result['jobs'][name])} runs")
    if a.dry_run or a.out is None:
        return 0
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = {"config": str(a.config), "cells": counts,
                "runs_emitted": {k: len(v) for k, v in result["jobs"].items()}, "distinct_runs": len(result["runs"])}
    for name, v in result["jobs"].items():
        (a.out / f"{name}.txt").write_text("\n".join(j.line for j in v) + "\n")
    (a.out / "matrix_runs.json").write_text(json.dumps(result["runs"], indent=1))
    if a.hashes:
        from recal.measure import jobs_util
        manifest["hashes"] = {name: [jobs_util.jobline_hash(j.line) for j in v] for name, v in result["jobs"].items()}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
