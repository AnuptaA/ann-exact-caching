#!/usr/bin/env python3
"""Runs a file of job lines (jobs_v6.py output) with --workers at a time, each pinned to its own block of
--cores-per-job cores (taskset; composed timing runs: CPU only, doc 9ad). Lines that only prepare (cp /
mkdir) run first. Order: --first budgets (main cache sizes) before the others.

  python3 simulations/approximate_hits/run_jobs.py /tmp/mx/v6_timing.txt --workers 8
"""
import re
import sys
import argparse
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lines")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cores-per-job", type=int, default=4)
    ap.add_argument("--first-core", type=int, default=8)
    ap.add_argument("--first", default="0.01,0.1,0.001,0.0001", help="budgets run first")
    a = ap.parse_args()
    lines = [l for l in open(a.lines).read().splitlines() if l.strip()]
    is_job = lambda l: "online_sim.py" in l or "hit_table.py" in l
    prep = [l for l in lines if not is_job(l)]
    jobs = [l for l in lines if is_job(l)]
    first = set(a.first.split(","))

    def key(l):
        mb = re.search(r"--budget ([0-9.e-]+)", l)
        b = mb.group(1) if mb else ""
        return (0 if b in first else 1)
    jobs.sort(key=key)                 # stable: registry order within each group
    for l in prep:
        subprocess.run(["sh", "-c", l], check=False)
    print(f"{len(jobs)} jobs, {a.workers} at a time", flush=True)
    free = list(range(a.workers))
    lock = threading.Lock()

    def run(l):
        with lock:
            w = free.pop(0)
        c0 = a.first_core + w * a.cores_per_job
        cmd = l.replace(".venv-core/bin/python", f"taskset -c {c0}-{c0 + a.cores_per_job - 1} .venv-core/bin/python", 1)
        try:
            subprocess.run(["sh", "-c", cmd], check=False)
        finally:
            with lock:
                free.append(w)
                free.sort()
        m = re.search(r"--out (\S+)", l) or re.search(r"(hit_tables/\S+\.npz)", l)
        print(f"done {m.group(1) if m else l[:80]}", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(run, jobs))
    print("all done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
