#!/usr/bin/env python3
"""Job lists (one shell command per line) for the real Aker (eval_aker_real.py,
quality mode) over the coverage matrix (context doc 9): stable traces,
workload shifts (shift, multishift), data changes (churn / drift x gradual /
burst; random stream, as the simulator's data runs) and a simulated inexact
backend; budgets 1% and 10%; grid alpha_tighten {0.9 (authors' config), 0.25
(paper)} x reserve delta {0, 5, 10}. One run per cell (Aker is deterministic
for a given stream). Results: results/aker_real/<prep>/<setting>_b<B>/at<a>_d<d>.json.

  python aker_real_jobs.py --preps spacev-aker-0.6 ... [--only stable shift data backend]
"""

import argparse
import itertools
from pathlib import Path

RES = "simulations/approximate_hits/results"
PY = ".venv-ann/bin/python -u simulations/approximate_hits/aker_real/eval_aker_real.py"
GRID = list(itertools.product([0.9, 0.25], [0, 5, 10]))


def settings(prep, only):
    online = Path(RES) / "online" / prep
    streams = None
    if (online / "streams.npz").exists():
        import numpy as np
        streams = np.load(online / "streams.npz").files
    if "stable" in only:
        yield "trace", "--stream trace"
    if "shift" in only and streams is not None:
        if "phase_shift" in streams and not prep.startswith("spacev-real"):
            yield "shift", "--stream shift"
        if "multishift" in streams:
            yield "multishift", "--stream multishift"
    if "data" in only:
        for kind, sched in itertools.product(["churn", "drift"], ["gradual", "burst"]):
            if (online / f"data_{kind}.npz").exists():
                yield (f"data_{kind}_{sched}",
                       f"--stream random --data-change {kind} --schedule {sched}")
    if "backend" in only and (online / "topU1000.npy").exists():
        for r in (0.95, 0.9):
            yield f"bk_trace_r{r:g}", f"--stream trace --backend-recall {r}"
    bfile = Path(RES) / "backends" / prep / "diskann.npz"
    if "backend" in only and bfile.exists():
        for t in (0.95, 0.9):
            yield (f"bk_trace_diskann{t:g}",
                   f"--stream trace --backend-file {bfile} --backend-target {t:g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preps", nargs="+", required=True)
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.01, 0.1])
    ap.add_argument("--only", nargs="+", default=["stable", "shift", "data", "backend"])
    ap.add_argument("--only-diskann", action="store_true", help="only the DiskANN-answer runs")
    ap.add_argument("--tuned", action="store_true",
                    help="settings tuned to reach the recall target (doc 9j) on stable traces only: "
                         "Standard mode alpha_T 0.1 / 0.01 x alpha_L 1.0 / 1.1; Proximity mode with the "
                         "global threshold at 0.01-0.3 x the median squared NN distance; reserve 5")
    args = ap.parse_args()
    if args.tuned:
        prox_lib = ("LD_LIBRARY_PATH=${RECAL_THIRD_PARTY:-$HOME/third_party}/aker_prox/build/lib:"
                    "${RECAL_THIRD_PARTY:-$HOME/third_party}/boost-install/lib:"
                    "${RECAL_THIRD_PARTY:-$HOME/third_party}/faiss-install/lib64")
        for prep, b in itertools.product(args.preps, args.budgets):
            out = f"{RES}/aker_real/{prep}/trace_b{b:g}"
            confs = [(f"at{at:g}_al{al:g}_d5", f"--alpha-tighten {at} --alpha-loosen {al} --delta 5", "")
                     for at, al in itertools.product([0.1, 0.01], [1.0, 1.1])]
            confs += [(f"prox{f:g}_d5", f"--prox-frac {f} --delta 5", prox_lib + " ")
                      for f in (0.01, 0.03, 0.05, 0.1, 0.2, 0.3)]
            for tag, flags, env in confs:
                print(f"mkdir -p {out}/logs && [ -f {out}/{tag}.json ] || {env}{PY} --prep {RES}/online/{prep} "
                      f"--budget {b} {flags} --stream trace --out {out}/{tag}.json > {out}/logs/{tag}.txt 2>&1")
        return
    for prep in args.preps:
        for (name, flags), b, (at, d) in itertools.product(list(settings(prep, args.only)),
                                                           args.budgets, GRID):
            if args.only_diskann and "diskann" not in name:
                continue
            out = f"{RES}/aker_real/{prep}/{name}_b{b:g}"
            tag = f"at{at:g}_d{d}"
            print(f"mkdir -p {out}/logs && [ -f {out}/{tag}.json ] || {PY} "
                  f"--prep {RES}/online/{prep} --budget {b} --alpha-tighten {at} --delta {d} "
                  f"{flags} --out {out}/{tag}.json > {out}/logs/{tag}.txt 2>&1")


if __name__ == "__main__":
    main()
