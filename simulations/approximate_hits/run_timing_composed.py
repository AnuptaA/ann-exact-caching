#!/usr/bin/env python3
"""
Runs the technique jobs of the final timing set as COMPOSED runs, many at once (context doc 9ad).

Each job line of `timing_jobs.py --set final` (technique runs; the DiskANN-only arms are run
separately, one at a time, on a quiet machine) gets `--compose-backend <timing_v4/<prep>/backend_s50_lat.npy>`
and `--allow-busy`, and runs pinned to its own block of cores (taskset), --workers at a time. Composed
runs charge each backend call that request's DiskANN-alone latency (measured with the 10 ms gap), so
they never read the disk and do not disturb each other: validated on SPACEV Zipf 0.99 and TriviaQA Zipf
0.6 (1%), composed = live within 3%, 12 at once = one at a time within 2%.

Order: drift runs first (they store the cell's shared warm-up product, which the continuous runs
load), then the rest, longest techniques first.

  PYTHONPATH=. .venv/bin/python simulations/approximate_hits/run_timing_composed.py --workers 12
"""

import re
import sys
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, ".")
from simulations.approximate_hits import timing_jobs as tj   # noqa: E402


def json_of(line):
    m = re.search(r"\[ -f ([^ ]*json) \]", line)
    return Path(m.group(1)) if m else None


def prep_of(line):
    m = re.search(r"timing_v[0-9]+/([^/ ]+)/", line)
    return m.group(1)


# v5 (doc 9af): one workload at a time, main cache sizes first, so complete workloads appear early; the
# shared warm-up products already exist (timing_v4/<prep>/warm_b*.pkl), drift still before continuous
WORKLOADS = ["spacev-aker-0.99", "esci1m", "triviaqa-ip-0.99", "spacev-aker-0.6", "triviaqa-ip-0.6", "spacev-real",
             "tripclick-ip", "spacev-aker-0.3", "triviaqa-ip-0.3", "medrag-ip"]
MAIN = {"medrag-ip": ("0.001", "0.0001")}


def order(line):
    n = json_of(line).name
    tech = 0 if n.startswith("oursdrift") else 1 if n.startswith("oursonline") else 2 if n.startswith("lru") else 3
    p = prep_of(line)
    b = re.search(r"_b([0-9.e-]+)\.json$", json_of(line).name).group(1)
    paper = n.startswith(("akerpaper", "qvpaper"))
    grp = 1 if paper else 0 if b in MAIN.get(p, ("0.01", "0.1")) else 2
    return (grp, WORKLOADS.index(p) if p in WORKLOADS else 99, tech, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--cores-per-job", type=int, default=4)
    ap.add_argument("--first-core", type=int, default=8)
    ap.add_argument("--set", default="final")
    ap.add_argument("--version", default="timing_v4", help="where the baseline latencies are (v5: timing_v5, doc 9af)")
    a = ap.parse_args()
    import io
    import contextlib
    buf = io.StringIO()
    sys.argv = ["timing_jobs.py", "--set", a.set]
    with contextlib.redirect_stdout(buf):
        tj.main()
    lines = [l for l in buf.getvalue().splitlines() if l.strip()]
    jobs = []
    for l in lines:
        j = json_of(l)
        if j is None or j.name.startswith("backend") or j.exists():
            continue
        if "_s20_" in j.name:               # sample-rate check cells: not part of the composed set
            continue
        lat = Path(tj.R) / a.version / prep_of(l) / "backend_s50_lat.npy"
        if not lat.exists():
            print(f"skip (no baseline latencies yet): {j}", flush=True)
            continue
        l = l.replace("--no-backend-arms", f"--no-backend-arms --compose-backend {lat}")
        l = l.replace("--wait-quiet 30", "--allow-busy")
        jobs.append(l)
    jobs.sort(key=order)
    print(f"{len(jobs)} composed jobs, {a.workers} at a time", flush=True)

    free = list(range(a.workers))
    import threading
    lock = threading.Lock()

    def run(l):
        with lock:
            w = free.pop(0)
        c0 = a.first_core + w * a.cores_per_job
        cmd = re.sub(r"(\.venv-ann/bin/python)", rf"taskset -c {c0}-{c0 + a.cores_per_job - 1} \1", l, count=1)
        try:
            subprocess.run(["sh", "-c", cmd], check=False)
        finally:
            with lock:
                free.append(w)
                free.sort()
        print(f"done {json_of(l).name} ({'ok' if json_of(l).exists() else 'MISSING'})", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(run, jobs))
    print("all done", flush=True)


if __name__ == "__main__":
    main()
