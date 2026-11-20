#!/usr/bin/env python3
"""Print the online_sim command line that reproduces a result JSON's run (its recorded args), with overrides
(doc 9as: timed recordings for the throughput replay from earlier untimed or sampled runs).

  python3 simulations/approximate_hits/rerun_cmd.py RESULT.json --set timing_every=1 --set out=DIR [--set k=v ...]
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--set", action="append", default=[], help="dest=value (value 'none' drops the option)")
    ap.add_argument("--python", default=".venv-core/bin/python")
    a = ap.parse_args()
    from simulations.approximate_hits.online_sim import make_parser
    args = dict(json.loads(Path(a.result).read_text())["args"])
    for kv in a.set:
        k, v = kv.split("=", 1)
        args[k] = None if v == "none" else v
    p = make_parser()
    out = []
    for act in p._actions:
        if not act.option_strings or act.dest == "help" or act.dest not in args:
            continue
        v = args[act.dest]
        flag = max(act.option_strings, key=len)
        if act.nargs == 0:                      # store_true / store_false
            if bool(v) != bool(act.default):
                out.append(flag)
            continue
        if v is None or v == "":
            continue
        if isinstance(v, (list, tuple)):
            if act.nargs == "*" and not v:
                out.append(flag)                 # an empty list (e.g. --detect with no tests)
                continue
            out.append(flag + " " + " ".join(str(x) for x in v))
        else:
            out.append(f"{flag} {v}")
    print(f"PYTHONPATH=. {a.python} -u simulations/approximate_hits/online_sim.py " + " ".join(out))


if __name__ == "__main__":
    main()
