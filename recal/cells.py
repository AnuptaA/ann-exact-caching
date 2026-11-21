#!/usr/bin/env python3
"""One experiment cell for any compared system, at an equal byte budget M.

M is a percentage of the dataset's stored bytes. ReCal charges its own structures (recal.sim). The advisor's
simulators count the budget in vectors; here the pool is chosen so that the system's bytes by structure
(recal.accounting) stay within M: a run is repeated with a rescaled pool when it used more than M.

  python -m recal.cells --method recal|prototype|aker|qvcache|proximity --prep <prep> --stream random \\
      --budget 0.05 --out <dir>

aker and qvcache are the simulators of simulations/approximate_hits/online_baselines.py at the papers' settings
(Aker: alpha_T 0.25, alpha_L 1.1, delta 0; QVCache: D 0.075, alpha 0.9, 4 mini-indexes). proximity and the real
Aker library need the Linux build (needs cluster check): `python -m recal.cells --method proximity` says so.
"""

import os
import json
import subprocess
import sys
import argparse
from pathlib import Path

import numpy as np

from recal import accounting, sim
from recal.measure.jobs_util import register_args, stamp_result

SIM = "simulations/approximate_hits/online_sim.py"
SIM_METHOD = {"prototype": "static", "aker": "aker", "qvcache": "qvcache"}


def counters(method: str, log: dict, d: int, args: dict) -> dict:
    cached, entries = log["cached"], log["entries"]
    if method == "aker":
        return accounting.aker_bytes(cached, entries, args.get("k", 10), args.get("aker_reserve", 0), d)
    if method == "qvcache":
        return accounting.qvcache_bytes(cached, d)
    shell = max(0.0, (log["memory"] - cached) * 32.0 / args.get("bits", 4))
    return accounting.prototype_bytes(cached, int(shell), entries, max(cached / max(entries, 1), 1.0), d,
                                      args.get("bits", 4))


def run_simulator(method: str, a, fraction: float, tag: str) -> Path:
    cmd = [sys.executable, SIM, "--prep", a.prep, "--stream", a.stream, "--method", SIM_METHOD[method],
           "--budget", f"{fraction:.6g}", "--k", str(a.k), "--out", a.out, "--tag", tag]
    if a.max_requests:
        cmd += ["--max-queries", str(a.max_requests)]
    subprocess.run(cmd, check=True, env={**os.environ, "PYTHONPATH": "."})
    return Path(a.out) / f"{SIM_METHOD[method]}{tag}.json"


def next_fraction(fraction: float, used: float, M: float, lo: float | None, hi: float | None) -> float:
    """Next pool fraction: scale by M / used, kept strictly between the largest feasible and the smallest
    infeasible fraction seen so far (bytes are not monotone in the pool)."""
    nxt = fraction * M / used * (0.99 if used > M else 0.99)
    if hi is not None and nxt >= hi:
        nxt = (lo + hi) / 2 if lo is not None else hi * 0.95
    if lo is not None and nxt <= lo:
        nxt = (lo + hi) / 2 if hi is not None else lo * 1.05
    return nxt


def run_baseline(method: str, a, attempts: int = 6) -> dict:
    db, *_ = sim.load_prep(Path(a.prep))
    d, n = db.shape[1], len(db)
    M = accounting.budget_bytes(n, d, a.budget)
    fraction = a.budget * 4 * d / (4 * d + 8 + 40)        # first guess: the per-vector charge plus a share of entries
    runs, best, lo, hi = [], None, None, None
    for attempt in range(attempts):
        path = run_simulator(method, a, fraction, f"{a.tag}_try{attempt}")
        res = json.loads(path.read_text())
        used = counters(method, res["log"][-1], d, res.get("args", {}))
        runs.append({"vector_fraction": fraction, "total_bytes": used["total"]})
        if used["total"] <= M:
            lo = fraction if lo is None else max(lo, fraction)
            if best is None or used["total"] > best[0]["total"]:
                best = (used, res, fraction)
            if used["total"] >= 0.95 * M:
                break
        else:
            hi = fraction if hi is None else min(hi, fraction)
        fraction = next_fraction(fraction, used["total"], M, lo, hi)
    used, res, fraction = best if best is not None else (used, res, fraction)
    res.update({"bytes": used, "M": M, "within_M": bool(used["total"] <= M), "byte_runs": runs,
                "vector_fraction": fraction, "method": method})
    for p in Path(a.out).glob(f"*{a.tag}_try*.json"):
        p.unlink()
    return res


def main() -> None:
    ap = sim.parser()
    ap.add_argument("--method", choices=["recal", "prototype", "aker", "qvcache", "proximity"], required=True)
    a = ap.parse_args()
    register_args(a)
    if a.method == "proximity":
        raise SystemExit("Proximity mode needs the real Aker library built on Linux: needs cluster check "
                         "(make q1-run-proximity-p0.1-b5, or aker_real with --prox-frac)")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    result = sim.run(a) if a.method == "recal" else run_baseline(a.method, a)
    stamp_result(result)
    tmp = out / f".{a.method}{a.tag}.json.tmp"
    tmp.write_text(json.dumps(result, default=str))
    tmp.replace(out / f"{a.method}{a.tag}.json")


if __name__ == "__main__":
    main()
