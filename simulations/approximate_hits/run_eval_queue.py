#!/usr/bin/env python3
"""Run an evaluation script on run directories as their greedy results appear.

Waits for <run_dir>/results.json, then runs the given evaluation script on it
(skipping runs whose output file already exists), with limited concurrency.
"""

import sys
import time
import argparse
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--script", default="simulations/approximate_hits/eval_calibrated.py")
    ap.add_argument("--output", default="calibrated.json",
                    help="output file the script writes into each run dir")
    ap.add_argument("--extra", default="--targets 0.9 0.95 --workers 6")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--poll", type=int, default=60)
    args = ap.parse_args()

    pending = [Path(d) for d in args.run_dirs]
    running = {}
    while pending or running:
        for d, proc in list(running.items()):
            if proc.poll() is not None:
                print(f"done {d} (exit {proc.returncode})", flush=True)
                del running[d]
        for d in list(pending):
            if len(running) >= args.concurrency:
                break
            if (d / args.output).exists():
                pending.remove(d)
                continue
            if (d / "results.json").exists():
                log = open(d / (Path(args.output).stem + ".log"), "w")
                cmd = ["nice", "-n", "5", args.python, args.script, str(d)] + args.extra.split()
                print(f"start {d}", flush=True)
                running[d] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                              env={"OMP_NUM_THREADS": "4",
                                                   "PATH": "/usr/bin:/bin"})
                pending.remove(d)
        time.sleep(args.poll if (pending or running) else 0)


if __name__ == "__main__":
    main()
