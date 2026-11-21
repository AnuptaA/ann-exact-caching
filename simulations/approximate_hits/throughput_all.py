#!/usr/bin/env python3
"""Phase 2 of the throughput experiments (doc 9ap): for every cell in $RECAL_QUEUE_DIR/tp_all_cells.json, replay the
recorded runs (throughput/runs/<w>/<tag>_b<b>/*.json) through the real DiskANN backend (throughput_replay.py: none, ours
LRU, ours continuous greedy, Aker, QVCache) and measure DiskANN with the cache's memory as node cache
(throughput_backend.py, equal-memory vs base node cache). Sequential, on a quiet machine. Skips cells whose output
loads as valid JSON; a failing subprocess stops the run. Outputs are written to a temp file and renamed.
  PYTHONPATH=. .venv-ann/bin/python simulations/approximate_hits/throughput_all.py"""
import os
import sys
import json
import glob
import subprocess
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

RES = Path("simulations/approximate_hits/results")
QUEUE_DIR = os.environ.get("RECAL_QUEUE_DIR", "/tmp/mx")
PY = ".venv-ann/bin/python"
ENV = {"PYTHONPATH": ".", "PATH": "/usr/bin:/bin"}


def valid_output(path):
    """True when the file exists and holds a non-empty JSON list or object."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return False
    return bool(data)


def find_recording(run_dir):
    """The single recorded run json of a finished run directory, or None if it is missing or not DONE."""
    js = sorted(glob.glob(str(Path(run_dir) / "*.json")))
    if len(js) > 1:
        raise RuntimeError(f"several recordings in {run_dir}: {js}")
    if not js or not (Path(run_dir) / "DONE").exists():
        return None
    return js[0]


def run_to_file(cmd, out):
    """Run cmd with --out set to a temp path, check the exit code and the output, then rename."""
    out = Path(out)
    tmp = out.with_name(out.name + ".part")
    tmp.unlink(missing_ok=True)
    subprocess.run(cmd + ["--out", str(tmp)], env=ENV, check=True)
    if not valid_output(tmp):
        raise RuntimeError(f"{cmd[1]} wrote no valid output to {tmp}")
    os.replace(tmp, out)


def main():
    cells = json.loads(Path(f"{QUEUE_DIR}/tp_all_cells.json").read_text())
    for c in cells:
        w, b = c["workload"], c["budget"]
        out = RES / "throughput" / f"replay_{w}_b{b:g}.json"
        arms = []
        for label, tag in (("ours_lru", "lru"), ("ours_cg", "oursonline"), ("aker", c.get("aker")), ("qvcache", c.get("qvcache"))):
            if not tag:
                continue
            rec = find_recording(RES / "throughput" / "runs" / w / f"{tag}_b{b:g}")
            if rec:
                arms.append(f"{label}={rec}")
        if len(arms) < 2:
            print("skip", w, b, "recordings missing", flush=True)
            continue
        warmup = json.loads(Path(arms[0].split("=", 1)[1]).read_text())["args"]["warmup"]
        if not valid_output(out):
            cmd = [PY, "simulations/approximate_hits/throughput_replay.py", "--workload", w, "--arm", "none"]
            for a in arms:
                cmd += ["--arm", a]
            cmd += ["--clients", "16", "--n", "12000", "--backend-procs", "8"]
            print(" ".join(cmd[:8]), "...", flush=True)
            run_to_file(cmd, out)
        # equal-memory node cache vs the base node cache (batch_search, 16 threads)
        per_budget = json.loads((RES / "timing_v6" / w / "backend_s50.json").read_text())["per_budget"]
        nd = per_budget.get(f"{b:g}", {}).get("nodes_cached")
        if nd is None and w == "tripclick-ip":
            nd = per_budget["0.1"]["nodes_cached"] * b / 0.1
        for nodes, tagname in ((None, "base"), (int(nd) if nd else None, "equal")):
            if tagname == "equal" and not nd:
                continue
            o = RES / "throughput" / f"nodes_{w}_b{b:g}_{tagname}.json"
            if valid_output(o):
                continue
            cmd = [PY, "simulations/approximate_hits/throughput_backend.py", "--workload", w, "--threads", "16", "--n", "3000",
                   "--warmup", str(warmup)]
            if nodes is not None:
                cmd += ["--nodes", str(nodes)]
            run_to_file(cmd, o)
    print("done", flush=True)


if __name__ == "__main__":
    main()
