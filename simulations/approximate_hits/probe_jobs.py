#!/usr/bin/env python3
"""
Probes for "why does LRU beat our greedy cache at a 10% budget?" (context doc 9h).
Stationary traces, budget 10%, seed 0. Job lines for mx_run_job.sh, results in
results/online/<prep>/probe_s0/<method><tag>.json.

  ours (greedy, no tests), warm-up 20k unless stated:
    P0   baseline                      (exists: r2_trace_b0.1_s0/greedy_none.json; re-run here for a clean pair)
    P1   --warm-split .5 .25 .125 .125   10k candidate queries, 5k scoring (default: 4k / 8k)
    P2   --max-ball-k 20                 balls of 10 or 20 neighbors only (default up to 500)
    P2b  --max-ball-k 50
    P3a  --warmup 40000                  everything scales: 8k candidates, 16k scoring
    P3b  --warmup 60000                  12k candidates, 24k scoring
    P4   P1 + --max-ball-k 20
  LRU (K = 20, our hit rule), the same warm-ups: L20 (20k), L40 (40k), L60 (60k).
All runs are compared on the common window of requests 60,000-100,000 (probe_summary.py).

  python probe_jobs.py > /tmp/mx/probe_jobs.txt
  python probe_jobs.py oracle > /tmp/mx/probe2_jobs.txt
"""

RES = "simulations/approximate_hits/results/online"
SIM = ".venv/bin/python -u simulations/approximate_hits/online_sim.py"
PREPS = ["spacev-aker-0.6", "triviaqa-dpr-0.6", "triviaqa-dpr-0.99"]
OURS = {"P0": "", "P1": "--warm-split 0.5 0.25 0.125 0.125", "P2": "--max-ball-k 20",
        "P2b": "--max-ball-k 50", "P3a": "--warmup 40000", "P3b": "--warmup 60000",
        "P4": "--warm-split 0.5 0.25 0.125 0.125 --max-ball-k 20"}
LRU = {"L20": "", "L40": "--warmup 40000", "L60": "--warmup 60000"}
# the extreme: greedy scores on ALL later requests (an oracle; see online_sim --oracle-score)
ORACLE = {"X1": "--oracle-score 20000 100000",     # scored on requests 20k-100k, measured on the same requests (in-sample, an upper bound)
          "X2": "--oracle-score 20000 60000"}      # scored on 20k-60k, measured on 60k-100k (out-of-sample control)


def main(only=None):
    for prep in PREPS:
        out = f"{RES}/{prep}/probe_s0"
        for tag, extra in (ORACLE if only == "oracle" else {} if only == "none" else OURS).items():
            print(f"mkdir -p {out}/logs && [ -f {out}/greedy_{tag}.json ] || {SIM} --prep {RES}/{prep} "
                  f"--stream trace --budget 0.1 --seed 0 --method greedy --detect {extra} --tag _{tag} "
                  f"--out {out} > {out}/logs/greedy_{tag}.txt 2>&1")
        for tag, extra in ({} if only in ("oracle", "none") else LRU).items():
            print(f"mkdir -p {out}/logs && [ -f {out}/lru_{tag}.json ] || {SIM} --prep {RES}/{prep} "
                  f"--stream trace --budget 0.1 --seed 0 --method lru --k-lru 20 --detect {extra} "
                  f"--tag _{tag} --out {out} > {out}/logs/lru_{tag}.txt 2>&1")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)      # "oracle": only the oracle probes
