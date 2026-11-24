#!/usr/bin/env python3
"""Run one shard on this instance: pull its artifacts, run its jobs, push the results.

  python -m scripts.run_shard jobs/shards/spacev-0.6.json [--store s3://bucket/prefix] [--workers 8]

Jobs run in stage order. Hit-rate and update runs go through run_jobs (concurrent, skipped when a result with the
same config hash exists); throughput jobs are exclusive and run one after the other. A job whose result is already
here or in the store is not run again. Results are pushed to results/<question>/<run hash>/ after each stage.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recal import artifacts, store as store_mod  # noqa: E402
from simulations.approximate_hits import run_jobs  # noqa: E402

HIT_KINDS = ("hit", "update")


def marker(job: dict) -> Path:
    files = [p for p in job["outputs"] if not p.endswith("/")]
    return Path(files[0] + ".jobhash") if files else Path(job["outputs"][0].rstrip("/") + "/.jobhash")


def is_done(job: dict) -> bool:
    if job["kind"] in HIT_KINDS:
        return run_jobs.result_status(job["line"], False, False)[1] == 0
    files = [Path(p) for p in job["outputs"] if not p.endswith("/")]
    return all(f.exists() for f in files[:1]) and marker(job).exists() and marker(job).read_text() == job["hash"]


def outputs_exist(job: dict) -> bool:
    return all(Path(p).exists() for p in job["outputs"])


def push_job(s, job: dict) -> int:
    prefix = store_mod.result_prefix(job["exp"], job["hash"])
    n = 0
    for p in job["outputs"]:
        path = Path(p.rstrip("/"))
        if p.endswith("/") and path.is_dir():
            n += s.push_tree(path, prefix)
        elif path.is_file():
            n += s.push_file(f"{prefix}/{path.name}", path)
    if job["kind"] not in HIT_KINDS and marker(job).exists():
        n += s.push_file(f"{prefix}/.jobhash", marker(job))
    return n


def pull_job(s, job: dict) -> bool:
    """Fetch a finished job's results from the store; False when the store has none."""
    prefix = store_mod.result_prefix(job["exp"], job["hash"])
    keys = s.keys(prefix)
    if not keys:
        return False
    for p in job["outputs"]:
        path = Path(p.rstrip("/"))
        if p.endswith("/"):
            s.pull_prefix(prefix, path)
        else:
            if f"{prefix}/{path.name}" not in keys:
                return False
            s.pull_file(f"{prefix}/{path.name}", path)
    if job["kind"] not in HIT_KINDS and f"{prefix}/.jobhash" in keys:
        s.pull_file(f"{prefix}/.jobhash", marker(job))
    return True


def run_exclusive(job: dict) -> bool:
    rc = subprocess.run(["sh", "-c", job["line"]], check=False).returncode
    if rc == 0 and outputs_exist(job):
        marker(job).parent.mkdir(parents=True, exist_ok=True)
        marker(job).write_text(job["hash"])
        return True
    print(f"FAILED (exit {rc}) {job['outputs'][0]}", flush=True)
    return False


def run_shard(shard: dict, s=None, workers: int = 8, cores_per_job: int = 4, first_core: int = 8,
              root: Path = Path(".")) -> dict:
    summary = {"ran": 0, "skipped": 0, "failed": 0, "pushed": 0}
    if s is not None:
        artifacts.pull_artifacts(s, shard["artifacts"], root)
        for item in shard["pull_results"]:
            job = {"exp": item["question"], "hash": item["hash"], "outputs": [item["path"]], "kind": "x"}
            if not Path(item["path"].rstrip("/")).exists():
                pull_job(s, job)
    stages = sorted({j["stage"] for j in shard["jobs"]})
    for stage in stages:
        pending = []
        for job in (j for j in shard["jobs"] if j["stage"] == stage):
            if not is_done(job) and s is not None and pull_job(s, job) and is_done(job):
                summary["skipped"] += 1
            elif is_done(job):
                summary["skipped"] += 1
            else:
                pending.append(job)
        concurrent = [j for j in pending if not j["exclusive"]]
        if concurrent:
            failed = run_jobs.run_lines([j["line"] for j in concurrent], workers, cores_per_job, first_core)
            summary["failed"] += len(failed)
            summary["ran"] += len(concurrent) - len(failed)
        for job in (j for j in pending if j["exclusive"]):
            if run_exclusive(job):
                summary["ran"] += 1
            else:
                summary["failed"] += 1
        if s is not None:
            summary["pushed"] += sum(push_job(s, j) for j in shard["jobs"] if j["stage"] == stage
                                     and outputs_exist(j))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shard", type=Path)
    ap.add_argument("--store", default=None, help="s3://bucket/prefix or a directory (default: RECAL_STORE)")
    ap.add_argument("--no-store", action="store_true", help="run local files only")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cores-per-job", type=int, default=4)
    ap.add_argument("--first-core", type=int, default=8)
    a = ap.parse_args(argv)
    shard = json.loads(a.shard.read_text())
    s = None if a.no_store else store_mod.open_store(a.store)
    summary = run_shard(shard, s, a.workers, a.cores_per_job, a.first_core)
    print(json.dumps({"shard": shard["name"], **summary}))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
