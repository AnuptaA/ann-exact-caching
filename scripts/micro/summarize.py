#!/usr/bin/env python3
"""Pull micro/ from the store and write one table per experiment across workloads (CSV and Markdown).

  python -m scripts.micro.summarize [--store URL] [--out results/micro_summary] [--pulled DIR]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from recal import store as store_mod  # noqa: E402


def read_rows(directory: Path, experiment: str) -> list[dict]:
    """Rows of every workload's result file of one experiment, workload first."""
    rows = []
    for path in sorted(Path(directory, experiment).glob(f"*/{experiment.lower()}.json")):
        body = json.loads(path.read_text())
        rows += body["results"]["rows"]
    return rows


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def write_tables(rows: list[dict], csv_path: Path, md_path: Path) -> None:
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(" --- " for _ in columns) + "|"]
    lines += ["| " + " | ".join(fmt(r.get(c)) for c in columns) + " |" for r in rows]
    md_path.write_text("\n".join(lines) + "\n")


def summarize(pulled: Path, out: Path) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for experiment_dir in sorted(p for p in Path(pulled).iterdir() if p.is_dir()):
        rows = read_rows(pulled, experiment_dir.name)
        if rows:
            write_tables(rows, out / f"{experiment_dir.name}.csv", out / f"{experiment_dir.name}.md")
            written.append(experiment_dir.name)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", default=None)
    ap.add_argument("--pulled", type=Path, default=Path("results/micro_pulled"))
    ap.add_argument("--out", type=Path, default=Path("results/micro_summary"))
    ap.add_argument("--no-pull", action="store_true", help="summarize what is already in --pulled")
    a = ap.parse_args(argv)
    if not a.no_pull:
        n = store_mod.open_store(a.store).pull_prefix("micro", a.pulled)
        print(f"pulled {n} files from micro/")
    written = summarize(a.pulled, a.out)
    print("tables: " + (", ".join(written) if written else "none"))
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
