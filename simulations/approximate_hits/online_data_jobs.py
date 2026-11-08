#!/usr/bin/env python3
"""Job list (one shell command per line) for the first data-change round
(context doc 5x): ESCI 1M, random query order, churn / drift x gradual /
burst, our method only, seeds 0 and 1.

Configurations (name: what it tests):
  static            upkeep on, nothing else adapts (no recalibration)
  slide_none        calibration on serve-time labels (sliding window), no test
  slide_score       + score test
  slide_ret         + retention test
  slide_both        + both
  slide_both_loose  both, retention / score at 5% false-alarm chance per check
  slide_ret_loose   retention test only, 5% per check
  maint_none        maintained labels (kept current), no test
  maint_ret         maintained labels + retention test
  maint_both        maintained labels + both tests
  noupkeep_both     no cache upkeep on inserts (naive cache), sliding, both
  oracle            every query labeled for calibration (reference), no test
  known_reset       burst only: reset calibration at the burst, no test
Response to alarms: reset + shadow rebuild (as concluded for workload shifts).

  python online_data_jobs.py > /tmp/data_jobs.txt
"""

import itertools

RES = "simulations/approximate_hits/results/online/esci1m"
SIM = ".venv/bin/python -u simulations/approximate_hits/online_sim.py"
CONFIGS = {
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


def main():
    jobs = []
    for kind, sched, seed in itertools.product(("churn", "drift"), ("gradual", "burst"), (0, 1)):
        out = f"{RES}/data_{kind}_{sched}_s{seed}"
        for name, flags in CONFIGS.items():
            if name == "known_reset" and sched != "burst":
                continue
            method = flags.split()[1]
            cmd = (f"mkdir -p {out}/logs && [ -f {out}/{method}_{name}.json ] || "
                   f"{SIM} --stream random --budget 0.1 --data-change {kind} --schedule {sched} "
                   f"--seed {seed} {flags} --response rebuild --tag _{name} --out {out} "
                   f"> {out}/logs/{name}.txt 2>&1")
            # maintained-label runs are the slowest: start them first
            jobs.append((not name.startswith("maint"), cmd))
    for _, cmd in sorted(jobs, key=lambda j: j[0]):
        print(cmd)


if __name__ == "__main__":
    main()
