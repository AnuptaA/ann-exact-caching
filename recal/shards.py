"""Shards of the experiment matrix: job files in dependency order for one instance each.

A shard is self-contained: it lists its jobs (stage order: hit-rate and update runs, split sweep, best split,
thread lines), the artifacts to pull, and the results of other shards its jobs read. Q3 jobs of a workload stay in
one shard. Throughput jobs are exclusive (they run alone, one after the other); the others may run concurrently.
"""

from collections import OrderedDict

from recal import matrix

BY = ("workload", "question", "system")


def keep(job, question=None, system=None, workload=None) -> bool:
    if question and job.exp not in question:
        return False
    if workload and job.workload not in workload:
        return False
    if system and not any(job.system == s or job.system.startswith(s) for s in system):
        return False
    return True


def shard_name(job, by: str) -> str:
    if by == "workload":
        return job.workload or "none"
    if by == "question":
        return job.exp
    return f"q3_{job.workload}" if job.exp == "q3" else job.system


def job_record(job, run_hash: str) -> dict:
    return {"line": job.line, "exp": job.exp, "workload": job.workload, "system": job.system, "kind": job.kind,
            "stage": job.stage, "exclusive": job.exclusive, "hash": run_hash, "needs": list(job.needs),
            "outputs": list(job.produced), "inputs": list(job.inputs)}


def build(cfg: dict, by: str = "workload", question=None, system=None, workload=None, medrag=None) -> OrderedDict:
    """shard name -> {jobs, artifacts, pull_results}."""
    if by not in BY:
        raise ValueError(f"--shard-by must be one of {BY}")
    runs = matrix.runs(cfg, medrag=medrag)
    producer = {}
    for exp, jobs in runs["jobs"].items():
        for job, h in zip(jobs, runs["job_hashes"][exp]):
            for path in job.produced:
                producer[path.rstrip("/")] = (exp, h)
    shards: OrderedDict = OrderedDict()
    for exp, jobs in runs["jobs"].items():
        for job, h in zip(jobs, runs["job_hashes"][exp]):
            if keep(job, question, system, workload):
                shards.setdefault(shard_name(job, by), []).append((job, h))
    out = OrderedDict()
    for name, items in shards.items():
        items = sorted(items, key=lambda x: x[0].stage)          # stable: matrix order inside a stage
        made = {p.rstrip("/") for job, _ in items for p in job.produced}
        pull, seen = [], set()
        for job, _ in items:
            for path in job.inputs:
                p = path.rstrip("/")
                if p not in made and p not in seen:
                    seen.add(p)
                    exp, h = producer[p]
                    pull.append({"path": path, "question": exp, "hash": h})
        out[name] = {"name": name, "jobs": [job_record(j, h) for j, h in items],
                     "artifacts": sorted({n for j, _ in items for n in j.needs}), "pull_results": pull}
    return out
