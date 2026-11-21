#!/usr/bin/env python3
"""Job lines for the final-config sweep (doc 9at): continuous greedy, deep lists R = min(128, L) (balls up to K 100, or 50 when
R < 128), mini cache 2% of the budget, every cache size of the headline / timing plots.
  timed:    DiskANN backend, every request timed (cache path; misses composed from timing_v6 backend_s50_lat), --log-served
  headline: exact backend, untimed (the headline plot's setting)
"""
import json
import subprocess
import sys
from pathlib import Path

RES = "simulations/approximate_hits/results"
TEMPLATE = f"{RES}/ablation_dyn/tripclick-ip/b0.1/og_R128/ogreedy.json"
L = {"spacev": 100, "esci1m": 100, "tripclick-ip": 128, "medrag-ip": 128, "triviaqa": 1000}
CELLS = {w: [0.003, 0.01, 0.03, 0.1] for w in ("spacev-aker-0.3", "spacev-aker-0.6", "spacev-aker-0.99", "esci1m")}
CELLS.update({w: [0.003, 0.01, 0.03, 0.1] for w in ("triviaqa-ip-0.3", "triviaqa-ip-0.6", "triviaqa-ip-0.99")})
CELLS["tripclick-ip"] = [0.003, 0.01, 0.03, 0.1, 0.3]
CELLS["medrag-ip"] = [0.0001, 0.0003, 0.001, 0.003]
TIMED_SIZES = {"triviaqa": [0.01, 0.1]}           # timing_v6 has only these for TriviaQA


def cmd(src, sets):
    a = ["python3", "simulations/approximate_hits/rerun_cmd.py", src] + sum([["--set", s] for s in sets], [])
    return subprocess.run(a, capture_output=True, text=True, check=True).stdout.strip()


ONLY = None


def main():
    global ONLY
    which = sys.argv[1]
    ONLY = sys.argv[2] if len(sys.argv) > 2 else None                      # timed | headline
    for w, sizes in CELLS.items():
        if ONLY and not w.startswith(ONLY):
            continue
        fam = "spacev" if w.startswith("spacev") else "triviaqa" if w.startswith("triviaqa") else w
        R = min(500, L[fam]) if fam == "triviaqa" else min(128, L[fam])   # 11-21: TriviaQA L 1000 -> R 500 at no cost
        kmax = 500 if R >= 500 else 100 if R >= 128 else 50
        stream = "random" if w == "esci1m" else "trace"
        if which == "timed":
            sizes = [b for b in sizes if b in TIMED_SIZES.get(fam, sizes)]
        for b in sizes:
            out = f"{RES}/final/{w}/b{b:g}/{which}"
            sets = [f"prep={RES}/online/{w}", f"stream={stream}", f"budget={b}", f"R={R}", f"max_ball_k={kmax}",
                    "warmup=none", "mini_cache_share=0.02", f"out={out}"]
            tf = Path(f"{RES}/tuning_core/{w}/b{b:g}.json")
            if tf.exists():
                d = json.loads(tf.read_text())
                sets += [f"core_ef1={d['stage1']['ef']}", f"core_ef2={d['stage2']['ef']}"]
            if which == "timed":
                lat = f"{RES}/timing_v6/{w}/backend_s50_lat.npy"
                sets += [f"backend_file={RES}/backends/{w}/diskann.npz", "backend_target=0.95", f"timing_compose={lat}",
                         "timing_every=1", "log_served=True"]
            c = cmd(TEMPLATE, sets)
            print(f"mkdir -p {out} && [ -f {out}/DONE ] || {{ {c} > {out}/log.txt 2>&1 && touch {out}/DONE; }}")


if __name__ == "__main__":
    main()
