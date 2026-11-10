#!/usr/bin/env python3
"""
Job lines for the timing plot (eval_timing_online.py; context doc 9k): every technique in front of
the same DiskANN (search list tuned to recall@10 0.95), SPACEV Zipf 0.3 / 0.6 / 0.99, cache budgets
1% and 10%. Aker and QVCache run their best setting meeting the target in the headline results
(plot_headline.collect: best hit rate among settings with mean retention of hits >= T - 0.01;
else the highest-retention setting). Ours and LRU start from the warm-up state of
online_sim --save-state with DiskANN's answers (results/timing_v2/<prep>/state_b<b>.npz); LRU runs
twice: cache maintenance on the request path (sync) and as background work.

  PYTHONPATH=. .venv/bin/python simulations/approximate_hits/timing_jobs.py > /tmp/mx/timing.txt
"""

import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from simulations.approximate_hits.plot_headline import collect  # noqa: E402

S = "simulations/approximate_hits"
R = f"{S}/results"
AKER_LIB = ("/home/jialind/third_party/aker/build/lib:/home/jialind/third_party/boost-install/lib:"
            "/home/jialind/third_party/faiss-install/lib64")
AKER_PROX_LIB = "/home/jialind/third_party/aker_prox/build/lib:" + AKER_LIB


def aker_args(name):
    """Real-Aker result name -> (eval_timing_online args, LD_LIBRARY_PATH)."""
    m = re.fullmatch(r"prox([0-9.]+)_d(\d+)", name)
    if m:
        return f"--prox-frac {m[1]} --aker-delta {m[2]}", AKER_PROX_LIB
    m = re.fullmatch(r"at([0-9.]+)(?:_al([0-9.]+))?_d(\d+)", name)
    if not m:
        raise SystemExit(f"unknown Aker setting {name}")
    al = f" --alpha-loosen {m[2]}" if m[2] else ""
    return f"--alpha-tighten {m[1]}{al} --aker-delta {m[3]}", AKER_LIB


def qv_args(name):
    m = re.fullmatch(r"qvcache_D([0-9.]+)_m(\d+)_r(\d+)x(\d+)", name)
    if not m:
        raise SystemExit(f"unknown QVCache setting {name}")
    return f"--qv-D {m[1]} --qv-mini {m[2]} --qv-dims {m[3]} --qv-buckets {m[4]}"


def main():
    for z in ("0.99", "0.6", "0.3"):
        prep = f"spacev-aker-{z}"
        for b in (0.01, 0.1):
            pick = collect(prep, "trace", b, True)
            out = f"{R}/timing_v2/{prep}"
            state = f"{out}/state_b{b:g}.npz"
            base = (f".venv-ann/bin/python -u {S}/eval_timing_online.py --prep {R}/online/{prep} "
                    f"--budget {b:g} --n 2000 --wait-quiet 30")
            jobs = [(f"ours_b{b:g}", f"--method ours --state {state}", ""),
                    (f"lru_sync_b{b:g}", f"--method lru --state {state}", ""),
                    (f"lru_bg_b{b:g}", f"--method lru --maintenance background --state {state} "
                                       f"--warmup 21000", "")]
            if pick.get("aker"):
                a, lib = aker_args(pick["aker"]["name"])
                jobs.append((f"aker_{pick['aker']['name']}_b{b:g}", f"--method aker {a}", lib))
            if pick.get("qvcache"):
                jobs.append((f"{pick['qvcache']['name']}_b{b:g}",
                             f"--method qvcache {qv_args(pick['qvcache']['name'])}", ""))
            for tag, extra, lib in jobs:
                env = f"LD_LIBRARY_PATH={lib} " if lib else ""
                js = f"{out}/{tag}.json"
                print(f"mkdir -p {out}/logs && [ -f {js} ] || {env}{base} {extra} --out {js} "
                      f"> {out}/logs/{tag}.txt 2>&1")


if __name__ == "__main__":
    main()
