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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--only", nargs="+", choices=sorted(matrix.EXPERIMENTS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--table", action="store_true", help="print the cells and dimensions of every experiment")
    ap.add_argument("--hashes", action="store_true")
    a = ap.parse_args(argv)
    cfg = json.loads(a.config.read_text())
    if a.table:
        for name, row in matrix.design_table(cfg).items():
            print(f"{name}: {row['cells']} cells = {row['dimensions']}")
        return 0
    lines = matrix.all_lines(cfg, a.only)
    counts = {name: len(v) for name, v in lines.items()}
    for name, n in counts.items():
        print(f"{name}: {n} cells")
    print(f"total: {sum(counts.values())} cells")
    if a.dry_run or a.out is None:
        return 0
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = {"config": str(a.config), "counts": counts}
    for name, v in lines.items():
        (a.out / f"{name}.txt").write_text("\n".join(v) + "\n")
    if a.hashes:
        from recal.measure import jobs_util
        manifest["hashes"] = {name: [jobs_util.jobline_hash(line) for line in v] for name, v in lines.items()}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
