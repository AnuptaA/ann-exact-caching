#!/usr/bin/env python3
"""Runs a file of job lines (jobs_v6.py output) with --workers at a time, each pinned to its own block of
--cores-per-job cores (taskset; composed timing runs: CPU only, doc 9ad). Lines that only prepare (cp /
mkdir) run first, in order. Order: --first budgets (main cache sizes) before the others.
A job whose result JSON exists is skipped only if its config_hash (resolved settings plus code commit)
matches the job line;
stale or unreadable results are moved to <json>.stale and rerun. Exit status is 1 if any line failed.

  python3 simulations/approximate_hits/run_jobs.py /tmp/mx/v6_timing.txt --workers 8
"""
import os
import re
import sys
import shutil
import argparse
import subprocess
import threading
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from recal.measure import jobs_util
except ImportError:
    jobs_util = None

PYTHON = ".venv-core/bin/python"
JOB_MARKERS = ("online_sim.py", "hit_table", "recal.cells", "recal.sim")
Assignment = namedtuple("Assignment", "workers first_core pin notice")


def is_job(line):
    return any(m in line for m in JOB_MARKERS)


def assign_cores(n_cpu, workers, cores_per_job, first_core):
    """Workers and first core that fit the machine; pin is False when nothing fits."""
    if n_cpu < first_core + workers * cores_per_job:
        first_core = 0
    max_workers = n_cpu // cores_per_job if cores_per_job > 0 else 0
    if max_workers < 1:
        return Assignment(1, 0, False, f"{n_cpu} cores, {cores_per_job} per job: not pinning")
    if workers > max_workers:
        return Assignment(max_workers, first_core, True, f"{n_cpu} cores: workers {workers} -> {max_workers}")
    return Assignment(workers, first_core, True, "")


def pin_line(line, c0, c1):
    if "taskset" in line:
        return line
    return re.sub(rf"(\S*{re.escape(PYTHON)})", rf"taskset -c {c0}-{c1} \1", line)


def result_status(line, strict_hash, trust_existing):
    """(json path or None, 0 fresh / 1 missing / 2 stale) for a job line."""
    m = re.search(r"\[ -f (\S+\.json) \]", line)
    if not m:
        return None, 1
    path = m.group(1)
    if jobs_util is None:
        return path, (0 if Path(path).is_file() else 1)
    return path, jobs_util.check_result(path, line, strict_hash, trust_existing)


def job_name(line, path):
    if path:
        return path
    m = re.search(r"(hit_tables/\S+\.npz)", line) or re.search(r"--out (\S+)", line)
    return m.group(1) if m else line[:80]


def run_lines(lines, workers=8, cores_per_job=4, first_core=8, first="0.01,0.1,0.001,0.0001",
              trust_existing=False, strict_hash=False):
    """Run job lines: lines that are not jobs one after the other, jobs concurrently; returns the failed lines."""
    prep = [l for l in lines if not is_job(l)]
    jobs = [l for l in lines if is_job(l)]
    first = set(first.split(","))

    def key(l):
        mb = re.search(r"--budget ([0-9.e-]+)", l)
        b = mb.group(1) if mb else ""
        return (0 if b in first else 1)
    jobs.sort(key=key)                 # stable: registry order within each group
    failures = []
    for l in prep:
        rc = subprocess.run(["sh", "-c", l], check=False).returncode
        if rc != 0:
            print(f"FAILED prep (exit {rc}): {l[:100]}", flush=True)
            failures.append(l)
    asg = assign_cores(os.cpu_count() or 0, workers, cores_per_job, first_core)
    if asg.notice:
        print(f"notice: {asg.notice}", flush=True)
    pin = asg.pin and shutil.which("taskset") is not None
    print(f"{len(jobs)} jobs, {asg.workers} at a time", flush=True)
    free = list(range(asg.workers))
    lock = threading.Lock()

    def run(l):
        path, status = result_status(l, strict_hash, trust_existing)
        if status == 0:
            print(f"skip (fresh) {path}", flush=True)
            return
        if status == 2:
            print(f"stale result, rerun: {path}", flush=True)
            os.replace(path, path + ".stale")
        with lock:
            w = free.pop(0)
        c0 = first_core + w * cores_per_job
        cmd = pin_line(l, c0, c0 + cores_per_job - 1) if pin else l
        try:
            rc = subprocess.run(["sh", "-c", cmd], check=False).returncode
        finally:
            with lock:
                free.append(w)
                free.sort()
        if rc == 0 and path is not None and result_status(l, strict_hash, trust_existing)[1] != 0:
            rc = -1
        name = job_name(l, path)
        if rc != 0:
            print(f"FAILED (exit {rc}) {name}", flush=True)
            with lock:
                failures.append(l)
        else:
            print(f"done {name}", flush=True)

    with ThreadPoolExecutor(asg.workers) as ex:
        list(ex.map(run, jobs))
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lines")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cores-per-job", type=int, default=4)
    ap.add_argument("--first-core", type=int, default=8)
    ap.add_argument("--first", default="0.01,0.1,0.001,0.0001", help="budgets run first")
    ap.add_argument("--trust-existing", action="store_true", help="a result that exists counts as done")
    ap.add_argument("--strict-hash", action="store_true", help="no effect: a result without a scheme 2 config hash is always stale")
    a = ap.parse_args()
    lines = [l for l in open(a.lines).read().splitlines() if l.strip()]
    failures = run_lines(lines, a.workers, a.cores_per_job, a.first_core, a.first, a.trust_existing, a.strict_hash)
    if failures:
        print(f"{len(failures)} lines failed", flush=True)
        return 1
    print("all done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
