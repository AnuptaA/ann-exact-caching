"""Collect result files into one table: hit rate, equality/approximate split, recall."""

import csv
from pathlib import Path

from harness.runner import results

FIELDS = ["dataset", "stream", "system", "setting", "budget_percent", "stream_seed", "hit_rate",
          "equality_hit_rate", "approximate_hit_rate", "delivered_recall", "recall_on_hits",
          "recall_on_misses", "requests_measured", "ef_search", "peak_pool_vectors_sampled",
          "commit", "dirty"]


def rows(out_dir: Path) -> list[dict]:
    out = []
    for path in sorted(Path(out_dir).glob("*.npz")):
        header, _ = results.read_result(path)
        out.append({**header["cell"], **header["summary"], "ef_search": header["ef_search"],
                    "peak_pool_vectors_sampled": header.get("peak_pool_vectors_sampled"),
                    "commit": header["commit"], "dirty": header["dirty"]})
    return out


def write_csv(out_dir: Path, csv_path: Path) -> int:
    table = rows(out_dir)
    with Path(csv_path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(table)
    return len(table)


def format_table(table: list[dict]) -> str:
    head = f"{'dataset':8} {'stream':10} {'system':9} {'setting':7} {'b%':>3} {'hit':>6} {'recall':>7}"
    lines = [head]
    for r in table:
        lines.append(f"{r['dataset']:8} {r['stream']:10} {r['system']:9} {r['setting']:7} "
                     f"{r['budget_percent']:>3} {r['hit_rate']:6.3f} {r['delivered_recall']:7.3f}")
    return "\n".join(lines)
