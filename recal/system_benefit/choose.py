"""Choose the memory split from the history, and read the result back in job lines.

  python -m recal.system_benefit.choose run --prep <prep> --stream <name> --total-fraction 0.05 \\
      --curve node_cache_curve.json --depth-table depth_latency.json --dim 100 --degree 64 \\
      --coord-bytes 1 --out chosen.json [--k 10 ...recal.sim flags]
  python -m recal.system_benefit.choose get chosen.json recal_budget_fraction

The hit rate h(s) and ReCal's background work at each split come from construction runs on the first part of
the history; reads per search come from the node cache curve and the depth table.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from recal import accounting, sim
from recal.system_benefit import history_runs, nodecache, split


def reads_by_depth(curve: split.ReadsCurve, cache_bytes: float, depth_table, k: int):
    """Reads of one search of `depth` at a node cache size: the depth-k reads scaled by the latency ratio."""
    cache = min(max(cache_bytes, curve.bytes[0]), curve.bytes[-1])
    base = curve.at(cache)
    return lambda depth: base * (depth_table.ratio(depth, k) if depth_table is not None else 1.0)


def choose(runs: dict, curve: split.ReadsCurve, total_bytes: float, depth_table, k: int, dim: int,
           coord_bytes: float, degree: int) -> dict:
    background = {}
    for s, run in runs.items():
        cache_bytes = split.arm_budgets(total_bytes, s)["node_cache_bytes"]
        background[s] = split.background_reads_per_request(
            run["background"], max(run["n_scored"], 1), reads_by_depth(curve, cache_bytes, depth_table, k),
            run["rereads"])
    hits = {s: run["hit_rate"] for s, run in runs.items()}
    out = split.choose_split(hits, curve, total_bytes, background)
    best = out["best_split"]
    out["node_cache_nodes"] = nodecache.nodes_for_bytes((1 - best) * total_bytes, dim, coord_bytes, degree)
    out["total_bytes"] = total_bytes
    return out


def read_curve(path: Path) -> split.ReadsCurve:
    rows = json.loads(Path(path).read_text())["rows"]
    return split.ReadsCurve([r["cache_bytes"] for r in rows], [r["reads_per_search"] for r in rows])


def run_command(a, sim_args) -> int:
    from simulations.approximate_hits.timing_env import DepthLatency
    streams = np.load(Path(a.prep) / "streams.npz")
    n = len(streams[sim_args.stream])
    history_len = int(a.history_frac * n)
    db_rows, dim = a.n_vectors, a.dim
    total_bytes = accounting.budget_bytes(db_rows, dim, a.total_fraction, a.coord_bytes)
    runs = history_runs.split_runs(sim_args, a.total_fraction, history_len, split.SPLITS)
    table = DepthLatency.load(a.depth_table) if a.depth_table else None
    result = choose(runs, read_curve(a.curve), total_bytes, table, sim_args.k, dim, a.coord_bytes, a.degree)
    result["recal_budget_fraction"] = result["best_split"] * a.total_fraction
    result["history_len"] = history_len
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", parents=[sim.parser()], conflict_handler="resolve")
    r.add_argument("--total-fraction", type=float, required=True)
    r.add_argument("--curve", type=Path, required=True)
    r.add_argument("--depth-table", type=Path)
    r.add_argument("--n-vectors", type=int, required=True)
    r.add_argument("--dim", type=int, required=True)
    r.add_argument("--degree", type=int, default=64)
    g = sub.add_parser("get")
    g.add_argument("file", type=Path)
    g.add_argument("field")
    a = ap.parse_args(argv)
    if a.command == "get":
        print(json.loads(a.file.read_text())[a.field])
        return 0
    return run_command(a, a)


if __name__ == "__main__":
    raise SystemExit(main())
