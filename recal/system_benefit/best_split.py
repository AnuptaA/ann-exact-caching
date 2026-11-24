"""Best node-cache share of each result cache from the throughput sweep.

  python -m recal.system_benefit.best_split run --sweep sweep.json --clients 64 \
      --arm recal_n80=<run.json>,nodes=N --arm ... --out best.json
  python -m recal.system_benefit.best_split get best.json recal.spec

Arms are named <system>_n<node share>. The best split of a system is the node share with the highest mean
throughput over the repetitions at the saturation client count; its arm spec is what the thread-lines job replays.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def mean_qps(rows: list[dict], clients: int) -> dict:
    qps = defaultdict(list)
    for r in rows:
        if r["clients"] == clients:
            qps[r["arm"]].append(r["qps"])
    return {arm: sum(v) / len(v) for arm, v in qps.items()}


def split_of(arm: str) -> tuple[str, int]:
    system, share = arm.rsplit("_n", 1)
    return system, int(share)


def best_splits(rows: list[dict], clients: int, specs: dict) -> dict:
    """system -> {arm, node_share, qps, spec}; ties go to the larger node share (the cheaper result cache)."""
    qps = mean_qps(rows, clients)
    best = {}
    for arm, value in qps.items():
        if arm not in specs:
            continue
        system, share = split_of(arm)
        if system not in best or (value, share) > (best[system]["qps"], best[system]["node_share"]):
            best[system] = {"arm": arm, "node_share": share, "qps": value, "spec": specs[arm]}
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--sweep", type=Path, required=True)
    run.add_argument("--clients", type=int, required=True)
    run.add_argument("--arm", action="append", required=True)
    run.add_argument("--out", type=Path, required=True)
    get = sub.add_parser("get")
    get.add_argument("file", type=Path)
    get.add_argument("key", help="dotted key, for example recal.spec")
    a = ap.parse_args(argv)
    if a.cmd == "get":
        value = json.loads(a.file.read_text())
        for part in a.key.split("."):
            value = value[part]
        print(value)
        return 0
    specs = {spec.split("=", 1)[0]: spec for spec in a.arm}
    best = best_splits(json.loads(a.sweep.read_text()), a.clients, specs)
    tmp = a.out.with_name(f".{a.out.name}.tmp")
    tmp.write_text(json.dumps(best, indent=1))
    tmp.replace(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
