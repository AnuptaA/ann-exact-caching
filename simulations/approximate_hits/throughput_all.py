#!/usr/bin/env python3
"""Phase 2 of the throughput experiments (doc 9ap): for every cell in /tmp/mx/tp_all_cells.json, replay the recorded runs
(throughput/runs/<w>/<tag>_b<b>/*.json) through the real DiskANN backend (throughput_replay.py: none, ours LRU, ours
continuous greedy, Aker, QVCache) and measure DiskANN with the cache's memory as node cache (throughput_backend.py,
equal-memory vs base node cache). Sequential, on a quiet machine. Skips finished cells.
  PYTHONPATH=. .venv-ann/bin/python simulations/approximate_hits/throughput_all.py"""
import sys, json, glob, subprocess, re
from pathlib import Path
RES = Path("simulations/approximate_hits/results")
cells = json.loads(Path("/tmp/mx/tp_all_cells.json").read_text())
py = ".venv-ann/bin/python"
for c in cells:
    w, b = c["workload"], c["budget"]
    out = RES / "throughput" / f"replay_{w}_b{b:g}.json"
    arms = []
    for label, tag in (("ours_lru", "lru"), ("ours_cg", "oursonline"), ("aker", c.get("aker")), ("qvcache", c.get("qvcache"))):
        if not tag:
            continue
        js = glob.glob(str(RES / "throughput" / "runs" / w / f"{tag}_b{b:g}" / "*.json"))
        if js and Path(js[0]).parent.joinpath("DONE").exists():
            arms.append(f"{label}={js[0]}")
    if len(arms) < 2:
        print("skip", w, b, "recordings missing", flush=True)
        continue
    if not out.exists():
        cmd = [py, "simulations/approximate_hits/throughput_replay.py", "--workload", w, "--arm", "none"]
        for a in arms:
            cmd += ["--arm", a]
        cmd += ["--clients", "16", "--n", "12000", "--backend-procs", "8", "--out", str(out)]
        print(" ".join(cmd[:8]), "...", flush=True)
        subprocess.run(cmd, env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin"}, check=False)
    # equal-memory node cache vs the base node cache (batch_search, 16 threads)
    nd = json.loads((RES / "timing_v6" / w / "backend_s50.json").read_text())["per_budget"].get(f"{b:g}", {}).get("nodes_cached")
    if nd is None and w == "tripclick-ip":
        nd = json.loads((RES / "timing_v6" / w / "backend_s50.json").read_text())["per_budget"]["0.1"]["nodes_cached"] * b / 0.1
    for nodes, tagname in ((None, "base"), (int(nd) if nd else None, "equal")):
        if tagname == "equal" and not nd:
            continue
        o = RES / "throughput" / f"nodes_{w}_b{b:g}_{tagname}.json"
        if o.exists():
            continue
        cmd = [py, "simulations/approximate_hits/throughput_backend.py", "--workload", w, "--threads", "16", "--n", "3000", "--out", str(o)]
        if nodes is not None:
            cmd += ["--nodes", str(nodes)]
        subprocess.run(cmd, env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin"}, check=False)
print("done", flush=True)
