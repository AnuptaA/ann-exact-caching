#!/usr/bin/env python3
"""Job lists (one shell command per line) for the data-change rounds
(context doc 5x, 5z): random query order, churn / drift x gradual / burst.

  --round 1  ESCI 1M, our configurations, seeds 0 and 1 (done)
  --round 2  ESCI: Aker / QVCache grids (seed 0) and LRU (seeds 0, 1);
             SPACEV Zipf 0.6 and real log: our main configurations and LRU
             (seeds 0, 1), Aker / QVCache grids (seed 0)

Our configurations (name: what it tests):
  static            upkeep on, nothing else adapts (no recalibration)
  slide_none        calibration on serve-time labels (sliding window), no test
  slide_score       + score test
  slide_ret         + retention test
  slide_both        + both
  slide_both_loose  both, 5% false-alarm chance per check
  slide_ret_loose   retention test only, 5% per check
  maint_*           maintained labels (kept current); round 1 only
  noupkeep_both     no cache upkeep on inserts (naive cache), sliding, both
  oracle            every query labeled for calibration (reference), no test
  known_reset       burst only: reset calibration at the burst, no test
Response to alarms: reset + shadow rebuild.
Baselines: Aker (tau init x alpha_T x cap x reserve 0 / 10, paper update
rules), QVCache (D x mini-indexes, PCA 16 dims x 8 buckets: best in every
earlier grid; deletes kept as published or removed), LRU (K x tests none /
score + retention; upkeep on).

  python online_data_jobs.py --round 2 > /tmp/data_jobs2.txt
"""

import argparse
import itertools

RES = "simulations/approximate_hits/results/online"
SIM = ".venv/bin/python -u simulations/approximate_hits/online_sim.py"
OURS = {
    "static": "--method static",
    "slide_none": "--method greedy --detect",
    "slide_score": "--method greedy --detect score",
    "slide_ret": "--method greedy --detect retention",
    "slide_both": "--method greedy --detect score retention",
    "slide_both_loose": "--method greedy --detect score retention --alarm-level 0.05",
    "slide_ret_loose": "--method greedy --detect retention --alarm-level 0.05",
    "maint_none": "--method greedy --detect --cal-labels maintained",
    "maint_ret": "--method greedy --detect retention --cal-labels maintained",
    "maint_both": "--method greedy --detect score retention --cal-labels maintained",
    "noupkeep_both": "--method greedy --detect score retention --no-maintain",
    "oracle": "--method greedy --detect --oracle-cal",
    "known_reset": "--method greedy --detect --reset-on-change 0.01",
}
OURS_SPACEV = ["static", "slide_none", "slide_score", "slide_both", "slide_ret_loose",
               "noupkeep_both", "oracle", "known_reset"]


def baselines():
    for ti, at, c, dlt in itertools.product([0.25, 0.5, 1.0], [0.25, 0.5, 0.9], [1.0, 2.0],
                                            [0, 10]):
        yield (f"aker_ti{ti}_at{at}_c{c}_d{dlt}",
               f"--method aker --aker-tau-init {ti} --aker-alpha-t {at} --aker-cap {c} "
               f"--aker-reserve {dlt}")
    for D, m, dl in itertools.product([0.0, 0.025, 0.075, 0.15, 0.25, 0.5], [2, 4, 8],
                                      ["keep", "remove"]):
        yield (f"qvcache_D{D}_m{m}_{dl}",
               f"--method qvcache --qv-D {D} --qv-mini {m} --qv-dims 16 --qv-buckets 8 "
               f"--qv-deletes {dl}")


def lru():
    for K, d in itertools.product([10, 20, 50, 100], ["none", "both"]):
        yield f"lru_k{K}_{d}", f"--method lru --k-lru {K} --detect" + (
            " score retention" if d == "both" else "")


def job(prep, budget, kind, sched, seed, name, flags):
    out = f"{RES}/{prep}/data_{kind}_{sched}_s{seed}"
    method = name.split("_")[0] if name.split("_")[0] in ("aker", "qvcache", "lru") \
        else flags.split()[1]
    tag = name[len(method) + 1:] if name.startswith(method + "_") else name
    return (f"mkdir -p {out}/logs && [ -f {out}/{method}_{tag}.json ] || "
            f"{SIM} --prep {RES}/{prep} --stream random --budget {budget} --data-change {kind} "
            f"--schedule {sched} --seed {seed} {flags} --response rebuild --tag _{tag} "
            f"--out {out} > {out}/logs/{name}.txt 2>&1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, choices=[1, 2], default=2)
    args = ap.parse_args()
    jobs = []
    work = itertools.product(("churn", "drift"), ("gradual", "burst"))
    for kind, sched in work:
        if args.round == 1:
            for seed, (name, flags) in itertools.product((0, 1), OURS.items()):
                if name == "known_reset" and sched != "burst":
                    continue
                jobs.append((0, job("esci1m", 0.1, kind, sched, seed, name, flags)))
            continue
        for prep, budget in (("esci1m", 0.1), ("spacev-aker-0.6", 0.01),
                             ("spacev-real", 0.01)):
            heavy = prep != "esci1m"
            if heavy:
                for seed, name in itertools.product((0, 1), OURS_SPACEV):
                    if name == "known_reset" and sched != "burst":
                        continue
                    jobs.append((1, job(prep, budget, kind, sched, seed, name, OURS[name])))
            for seed, (name, flags) in itertools.product((0, 1), lru()):
                jobs.append((1 if heavy else 0, job(prep, budget, kind, sched, seed, name, flags)))
            for name, flags in baselines():
                jobs.append((1 if heavy else 0, job(prep, budget, kind, sched, 0, name, flags)))
    # ESCI first (lighter); SPACEV after its data prep
    for _, cmd in sorted(jobs, key=lambda j: j[0]):
        print(cmd)


if __name__ == "__main__":
    main()
