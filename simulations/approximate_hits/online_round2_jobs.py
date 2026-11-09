#!/usr/bin/env python3
"""Write the job list (one shell command per line) for the round-2 online
experiments: detection x response factorial for our method (greedy cache),
LRU with and without the drift tests, oracle-calibration references, and the
Aker / QVCache tuning grids, on every workload.

  python online_round2_jobs.py [--multishift] > /tmp/round2_jobs.txt
  xargs -P 56 -I{} sh -c '{}' < /tmp/round2_jobs.txt
"""

import argparse
import itertools

RES = "simulations/approximate_hits/results/online"
SIM = ".venv/bin/python -u simulations/approximate_hits/online_sim.py"
WORKLOADS = [("esci1m", "shift", 0.1), ("esci1m", "random", 0.1),
             ("spacev-aker-0.6", "trace", 0.01), ("spacev-aker-0.6", "shift", 0.01),
             ("spacev-aker-0.3", "trace", 0.01), ("spacev-aker-0.99", "trace", 0.01),
             ("spacev-real", "trace", 0.01)] + \
    [(f"triviaqa-aker-{z}", "trace", 0.01) for z in ("0.3", "0.6", "0.99")]
# start mix -> mild shift -> full shift -> return (online_prepare --add-multishift)
MULTISHIFT = [("esci1m", "multishift", 0.1), ("spacev-aker-0.3", "multishift", 0.01),
              ("spacev-aker-0.6", "multishift", 0.01), ("spacev-aker-0.99", "multishift", 0.01)] + \
    [(f"triviaqa-aker-{z}", "multishift", 0.01) for z in ("0.3", "0.6", "0.99")]
DETECT = {"none": "", "score": "score", "retention": "retention", "hitrate": "hitrate",
          "all": "score retention hitrate"}


def ours():
    yield "static_default", "--method static"
    yield "greedy_none", "--method greedy --detect"
    for d, r in itertools.product(["score", "retention", "hitrate", "all"], ["recal", "rebuild"]):
        yield f"greedy_{d}_{r}", f"--method greedy --detect {DETECT[d]} --response {r}"
    yield "greedy_none_oraclecal", "--method greedy --detect --oracle-cal"
    yield "greedy_all_rebuild_oraclecal", f"--method greedy --detect {DETECT['all']} --oracle-cal"
    for K, d in itertools.product([10, 20, 50, 100], ["none", "all"]):
        yield f"lru_k{K}_{d}", f"--method lru --k-lru {K} --detect {DETECT[d]}"


def baselines():
    for ti, at, c in itertools.product([0.25, 0.5, 1.0], [0.25, 0.5, 0.9], [1.0, 2.0]):
        yield (f"aker_ti{ti}_at{at}_c{c}",
               f"--method aker --aker-tau-init {ti} --aker-alpha-t {at} --aker-cap {c}")
    for D, m, (dims, bk) in itertools.product([0.0, 0.025, 0.075, 0.15, 0.25, 0.5], [2, 4, 8],
                                              [(4, 4), (8, 4), (16, 8)]):
        yield (f"qvcache_D{D}_m{m}_r{dims}x{bk}",
               f"--method qvcache --qv-D {D} --qv-mini {m} --qv-dims {dims} --qv-buckets {bk}")


def sub(stream, budget_override):
    """Results folder name: r2_<stream> (default budgets) or r2_<stream>_b<B>."""
    return f"r2_{stream}" if budget_override is None else f"r2_{stream}_b{budget_override:g}"


def pick(workloads, budget_override, preps):
    return [(p, s, b if budget_override is None else budget_override)
            for p, s, b in workloads if preps is None or p in preps]


def best_baseline_jobs(seeds, budget_override=None, preps=None):
    """Best Aker / QVCache setting per workload on seed 0 (highest hit rate
    among settings whose mean served retention >= T, else the highest
    retention), re-run on fresh seeds so the reported numbers are not selected on
    the same run."""
    import json
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from online_summary import summarize
    grids = dict(baselines())
    out = []
    for prep, stream, budget in pick(WORKLOADS + MULTISHIFT, budget_override, preps):
        d0 = Path(f"{RES}/{prep}/{sub(stream, budget_override)}_s0")
        for fam in ("aker", "qvcache"):
            rows = []
            for name in grids:
                if name.startswith(fam) and (d0 / f"{name}.json").exists():
                    r = summarize(json.loads((d0 / f"{name}.json").read_text()), 5000)
                    rows.append((name, r))
            if not rows:
                continue
            ok = [x for x in rows if x[1]["ret"] is not None and x[1]["ret"] >= 0.9]
            # none meets T: the setting closest to the promise (highest retention)
            name = (max(ok, key=lambda x: x[1]["hit"]) if ok else
                    max(rows, key=lambda x: x[1]["ret"] or 0.0))[0]
            tag = name.split("_", 1)[1]
            for seed in seeds:
                o = f"{RES}/{prep}/{sub(stream, budget_override)}_s{seed}"
                out.append(f"mkdir -p {o}/logs && [ -f {o}/{name}.json ] || "
                           f"{SIM} --prep {RES}/{prep} --stream {stream} --budget {budget} "
                           f"--seed {seed} {grids[name]} --tag _{tag} --out {o} "
                           f"> {o}/logs/{name}.txt 2>&1")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--baseline-seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--multishift", action="store_true", help="the multishift workloads")
    ap.add_argument("--best-baselines", type=int, nargs="*", default=None,
                    help="only re-run the best seed-0 Aker / QVCache settings on these seeds")
    ap.add_argument("--budget", type=float, default=None,
                    help="override every workload's budget (results in r2_<stream>_b<B>_s<seed>)")
    ap.add_argument("--preps", nargs="*", default=None, help="only these preps")
    args = ap.parse_args()
    if args.best_baselines:
        for cmd in best_baseline_jobs(args.best_baselines, args.budget, args.preps):
            print(cmd)
        return
    jobs = []
    for prep, stream, budget in pick(MULTISHIFT if args.multishift else WORKLOADS,
                                     args.budget, args.preps):
        for seed in sorted(set(args.seeds) | set(args.baseline_seeds)):
            out = f"{RES}/{prep}/{sub(stream, args.budget)}_s{seed}"
            confs = (list(ours()) if seed in args.seeds else []) + \
                (list(baselines()) if seed in args.baseline_seeds else [])
            for name, flags in confs:
                tag = name.split("_", 1)[1] if "_" in name else ""
                method = name.split("_")[0]
                jobs.append((prep.startswith("esci"),
                             f"mkdir -p {out}/logs && [ -f {out}/{method}_{tag}.json ] || "
                             f"{SIM} --prep {RES}/{prep} --stream {stream} --budget {budget} "
                             f"--seed {seed} {flags} --tag _{tag} --out {out} "
                             f"> {out}/logs/{name}.txt 2>&1"))
    # ESCI runs are the longest: start them first
    for _, cmd in sorted(jobs, key=lambda j: not j[0]):
        print(cmd)


if __name__ == "__main__":
    main()
