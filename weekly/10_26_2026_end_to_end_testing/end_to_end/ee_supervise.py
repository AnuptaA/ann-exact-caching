#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Runs an ee_ script under caffeinate (when available) and resumes it if the process dies, up to --max-resumes times
(default 2). A clean abort by a safety rule (exit code 3) is never resumed. Child output goes to results/logs.

Usage:  python3 ee_supervise.py --script ee_run_d.py --tag D [--max-resumes N] -- <script arguments>"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
THESIS_ROOT = HERE.parents[2]
RESULTS = HERE.parent / "results"
EXIT_ABORT = 3

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--script", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-resumes", type=int, default=2)
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    run_args = [x for x in a.rest if x != "--"]
    (RESULTS / "logs").mkdir(parents=True, exist_ok=True)
    state_path = RESULTS / f"run_state_{a.tag}.json"
    stdout_log = RESULTS / "logs" / f"ee_stdout_{a.tag}.log"
    caff = ["caffeinate", "-i"] if shutil.which("caffeinate") else []
    cmd = caff + [sys.executable, "-B", str(HERE / a.script)] + run_args
    state = dict(started=datetime.now().isoformat(), command=" ".join(cmd), launches=[], resumes=0, final="running")
    state_path.write_text(json.dumps(state, indent=2))
    attempt = 0
    while True:
        t0 = time.time()
        with open(stdout_log, "a") as logf:
            logf.write(f"\n===== launch {attempt} at {datetime.now().isoformat()} =====\n")
            logf.flush()
            rc = subprocess.run(cmd, cwd=THESIS_ROOT, stdout=logf, stderr=subprocess.STDOUT).returncode
        state["launches"].append(dict(attempt=attempt, returncode=rc, seconds=round(time.time() - t0, 1),
                                      ended=datetime.now().isoformat()))
        if rc == 0:
            state["final"] = "completed"
            break
        if rc == EXIT_ABORT:
            state["final"] = "aborted by a safety rule (not resumed), see the ABORT file next to the results"
            break
        if state["resumes"] >= a.max_resumes:
            state["final"] = f"died (rc={rc}); gave up after {state['resumes']} resumes"
            break
        state["resumes"] += 1
        attempt += 1
        state_path.write_text(json.dumps(state, indent=2))
    state["ended"] = datetime.now().isoformat()
    state_path.write_text(json.dumps(state, indent=2))
    print(json.dumps(state, indent=2))
    return 0 if state["final"] == "completed" else (EXIT_ABORT if "aborted" in state["final"] else 1)

if __name__ == "__main__":
    sys.exit(main())
